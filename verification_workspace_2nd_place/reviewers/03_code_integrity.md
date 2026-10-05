# Reviewer 3 — Code Integrity and Anti-Cheating (2nd place, NavAttack)

Working notes (appended as review proceeds; final handoff at the end).

Static, read-only. No competitor code executed or imported; no artifacts loaded; no images viewed; no web fetches.

## Working notes

### N1. Inference path read (inference_main.py, regfit.py, register3.py, portable3.py, tabular_adapter.py)
- Block loop `inference_main.py:334-477`, BLOCK=128. Per-scan preprocess (`:337-365`), batched registration (`:378-386`), per-row features (`:387-393`), TabICL on block (`:396`), linears (`:398-399`), CNNs (`:438-449`), stacker (`:451-462`), T (`:464-465`), blend (`:466`), governor (`:470-471`), non-finite->prior (`:474`), clip (`:475-477`).
- Registration: `register3.fit:137-167` loss = sum over rows of (1-ncc); `ncc:102-122` per-row flatten(1); `rodrigues:67-78` and `warp:87-99` per-row; Adam per-element. d(sum)/d(term_i)=1, so a NaN in one row cannot reach another row's gradient. `regfit._reject:62-65` per-row. Template frozen asset. -> row-independent.
- FeatPrep / linear / stack: frozen JSON constants (`portable3.py:51-57, 91-93, 143-145`). No test statistics.
- 3D CNN: `(t-mu)/sd` scalar per checkpoint (`inference_main.py:219`); GroupNorm net in eval (`:110`). 2D: TorchScript eval + frozen (`:83-84`; ts_export per overview F2).
- TabularAdapter.predict (`tabular_adapter.py:53-63`): NaN -> recipe median (`:60`), priors use frozen FeatPrep/coef (`:58`); not wrapped in try (a failure aborts the whole run rather than altering other rows).

### N2. TabICL bundled 2.2.0 (read via `unzip -p`, not extracted)
- `_sklearn/classifier.py` predict_proba: `X_encoder_.transform` (SimpleImputer fitted on train) -> `ensemble_generator_.transform(X, mode="test")` -> `_batch_forward_with_cache` -> `model_.forward_with_cache(X_test, cache)`.
- `_sklearn/preprocessing.py` EnsembleGenerator.transform mode="test" (`~1194-1202`) uses `unique_filter_.transform` + fitted `preprocessors_[norm].transform`; PreprocessingPipeline.transform (`~751-783`): fitted CustomStandardScaler (`(X-mean_)/scale_`), fitted normalizer (norm methods 'none','power' per freeze_tabular.py:21-23,84), fitted OutlierRemover bounds. No refit on test.
  - Caveat: `preprocessing.py ~771-778` — if `normalizer_.transform` raises ValueError for the batch, the WHOLE batch is clipped to train X_min_/X_max_. Clipping is a no-op for in-range rows; only out-of-range rows could change. Theoretical batch coupling, no information transfer; not expected to trigger (inputs finite, scaler clipped at +/-100).
- `_model/tabicl.py:734-797` forward_with_cache, repr mode: col_embedder (cache) -> row_interactor -> concat(cached train repr, test repr) -> `icl_predictor.forward_with_repr_cache(train_size=n_train)`.
- `_model/layers.py:432-437` MultiheadAttentionBlock: when train_size given, `k = v = q[..., :train_size, :]` -> every query (train or test) attends ONLY to train positions; train rows' updates depend only on train rows. No test->test attention in ICL. Encoder for tf_icl has no RoPE (`encoders.py` use_rope default False; learning.py:94-105 does not set it), so row position within block does not matter.
- Column embedder with cache: `layers.py InducedSelfAttentionBlock.induced_attention_with_cache` use_cache -> `multihead_attn2(src, cached_kv=...)`: test rows query only train-derived cached K/V. Only cross-row op is `skip_mask = (src == skip_value).all(dim=(-2,-1))` (column entirely NaN/-100 across the batch) — neutralised because tabular_adapter.py:60 imputes every non-finite value; reserved CLS columns are always -100 (constant, data-independent).
- Row interactor (`interaction.py`): attention across columns within a row (RoPE over columns), per row.
- ssmax uses key length (`attention.py:104-106`) = train size, constant.
- InferenceManager (`inference.py:1307`) splits leading batch dims only, never the row/sequence dim of ICL.
- Conclusion F3: no test-to-test information path in TabICL forward_with_cache, apart from the neutralised all-NaN column mask and the theoretical normalizer-ValueError clip.

### N3. Test-data use in training (focus 1)
- Test-uid grep: 524 uids from submission_format.csv (8-char) against all of `2nd Place/` (text and binary, `grep -wF`), `extracted/harness_zip/`, and the text of tabicl_lib.zip: 0 hits. Train uids (1,362): 0 hits in text files (no per-uid tables shipped).
- Training code enumerates scans from `labels.csv` (`src/folds.py:38`, `src/cache3.py:52`, `src/make_regfeat32.py:67`, `src/build3.py:171`); `src/eda_real.py:70` and `tools/hdr_fingerprint.py:7` glob the staged training niftis dir. `pipeline/00_stage_data.sh:4-39` stages only the training zip + train labels. No `submission_format`, `test_labels`, `/code_execution` or `submission.csv` read in src/ tabular/ tools/ pipeline/ except the inference entry copies (`src/main3.py`) and package/predict helpers.
- No pseudo-labeling: grep `pseudo` hits only "pseudo-site" (acquisition-group proxy over training data, `src/train_final3.py:45`, `src/roigeom.py:42,80`).
- Repro logs (`docs/repro/mode_full_2026-09-28/`): every data stage reports 1,362 scans (`cache3 1362 scans | failures: 0`; `regfeat32 preprocessed 1362 ... rejected 2`). Parity runs (`parity/scored_run.log`) contain only package versions and the static logger lines. No test path or count.
- "150-scan local verification set" (README:322, REPRODUCIBILITY.md) — origin not stated; competitors had no access to test scans (code-execution format), so it is presumably training scans. Open question, low.

### N4. Test logging (rule C, focus 3)
- `inference_main.py`: all logger calls are static string literals (`:230,233,235,240,242,255,305,307,362,383,447,481`); warnings fire once per run via `warned_*` flags. `main.py:10-17` prints package versions, and on failure only `type(exc).__name__` plus file:line:function (the exception message, e.g. the count f-string at `inference_main.py:275`, is never printed).
- TabICL: all `print` in `_model/inference.py` gated by `self.verbose` (default False; freeze uses verbose=False); `tabular_adapter.py:7` ignores warnings; classifier warns only about n_jobs vs cores (no data).
- Platform log for 322397 not available; competitor's local parity log shows only static lines. -> Rule C: pass.

### N5. Hard-coded values / JSON (focus 5)
- json.load (own python, data only) of inference_config.json, tabular_recipe.json, linear_lin*.json: no uid-like keys or string values (only model tags such as `e2dmipf2`), no per-row arrays keyed to test. Contents are feature names, FeatPrep constants (per-feature, train-fitted), coefficients, member specs, scalars (coef = 0.53121 x [1,0.4,0.7,0.3,0.3], intercept 0.18431, T 1.4, clip 0.005, prior 0.5484581497797357 = 747/1362, registration bounds 90 deg / 0.30, blend 0.35, TabICL T 1.2).
- No post-processing other than T, blend, governor, non-finite->prior, clip.

### N6. CSV vs code path (focus 7)
- Scored CSV sha256 293d7041…e37a. 115 rows = 0.995, 4 rows = 0.005 -> the clip `lo = max(0.005, 1e-6)` at `inference_main.py:475-477`. Next values inside the clip: 0.99476…, 0.00546…, so values approach the bounds continuously.
- 0 rows equal to (or within 1e-3 of) the prior 0.5484581497797357. Every governor rejection (`:471`), preprocessing failure (`:365`), non-finite fallback (`:474`) or missing uid (`:479`) would write exactly that value (it lies inside the clip). So none of these paths fired in the scored run. Training rejected 2/1,362, so 0/524 is plausible.
- Values are full float reprs (13-19 decimals, 407 distinct). No sign of rounding or hand editing. Extremes are spread over all 5 blocks of 128 (30/35/20/31/3; the last block has 12 rows).
- Block-level fallbacks (registration/CNN) cannot be detected from the CSV (G6).

### N7. Leaderboard-informed tuning (focus 4) and constant provenance (focus 6)
- `src/fixstack.py:2-5`: the fixed ratio is credited to "RESEARCH_NOTES §68: beats the fitted nnlogreg on 6/6 re-splits" (CV). "temperature T and clip from the board T-curve (§67/§72)". T=1.4 and clip=0.005 are argparse defaults (`:28-29`). Scale and intercept are fitted on training OOF only (`:52-54`). Reproduced: 0.53121 / 0.18431 (`docs/repro/.../08_stack.log`).
- Training-OOF evidence that T=1.4 was not CV-chosen: `08_stack.log` "[fixstack] cross-fit T=1: ll 0.2070 | T=1.4: ll 0.2167". T=1.4 is WORSE on training OOF. So the T value comes from public-LB feedback (plus the competitor's site-shift reasoning). The stale comment `inference_main.py:247-251` says the leave-site-out surface gave T=1.0.
- LB sequence (`docs/figures/make_figures.py:93-111`, typed constants from the competitor's notes; figures not viewed): v7 (T=1) pub 0.2611 -> v10a (only T=1.2 changed) 0.2507 -> v11 (T=1.4 + 34-ckpt CNN, fixed-ratio stack, governor) 0.2502. The private scores move the same way (0.2988 -> 0.2854 -> 0.2802), so the gain reflects a real train-to-test calibration shift and not public-slice overfit. The private values in make_figures.py are post-competition documentation and feed nothing in the pipeline (the script imports only argparse/json/re/sys/pathlib/matplotlib, per the overview).
- `src/train_final3.py:56-61`: `head_ml` is withheld from lin/lin4 because "the board refuted it in v9" (v9 pub 0.2664 vs v7 0.2611). This is one global feature-inclusion decision. Note that TabICL still uses `head_ml` as one of 4 acquisition descriptors (`tabular_recipe.json` features[64]).
- `src/cnn3.py:11-12` notes CV 0.3049 vs LB 0.3364 for v2. That diagnosis led to a fixed-schedule OOF protocol. Benign.
- TabICL blend 0.35 / T 1.2: hard-coded in `tabular/prepare_tabular.py:162` (and `_original/prepare_candidate.py:39`). Already present in `_original/bench_v16.py:47` as the v15 formula. Training-OOF blend grids exist (`_original/common_20260914.py:60-63` w in {.15,.3,.5}; `bench_v16.py:68` {.2,.4,.6}), but no code selects 0.35/1.2. The provenance is undocumented (dev notes not distributed). It may be OOF- or LB-informed and cannot be resolved statically.
- Member ratios 1:0.4:0.7:0.3:0.3: "chosen by hand" (`docs/stages/08-09_stack_and_package.md:23-24`). The 2D ratios can be overridden (`RATIO_B0/B3`). Provenance beyond the fixstack docstring is undocumented.
- Information capacity: 11 uploads in total, 9 scored (2 failed at startup). Each scored upload returns one public scalar. LB feedback touched about 3-5 global scalars (T, clip, head_ml inclusion, possibly the blend weight and T_tab, possibly the 2D ratios). No per-uid, per-site or per-row constants exist anywhere (N5), so there is no mechanism to encode test labels. -> This is ordinary LB-guided hyperparameter selection, not label probing.

---

## Subagent: Reviewer 3 — Code Integrity and Anti-Cheating

Status: pass with notes

No competitor code executed: yes (none executed or imported; no artifact loaded, unpickled or decompressed; tabicl_lib.zip read only via `unzip -l` / `unzip -p` to stdout; no images viewed; no web fetches; nothing written under `2nd Place/` or `references/`; test_labels.csv not opened)

Read-only inspection performed:
- `2nd Place/inference/`: main.py, inference_main.py (full), regfit.py, register3.py (transform/NCC/fit), portable3.py, tabular_adapter.py, nets3.py (encoder/HemiMIL), cnn2d.py (make_image/prep/predict), cnn3.py (to_input, header).
- `2nd Place/inference/tabicl_lib.zip` text members: `_sklearn/classifier.py` (predict_proba, _batch_forward_with_cache), `_sklearn/preprocessing.py` (all transform paths), `_model/tabicl.py` (forward_with_cache), `_model/learning.py`, `_model/encoders.py`, `_model/layers.py` (MultiheadAttentionBlock, ISAB), `_model/embedding.py` (cache path, feature_grouping), `_model/interaction.py`, `_model/attention.py`, `_model/ssmax.py`, `_model/inference.py` (batch splitting, prints).
- Training side: src/fixstack.py, src/train_final3.py:35-70, src/cnn3.py header, tabular/freeze_tabular.py, tabular/prepare_tabular.py:150-175, tabular/_original/{bench_v16,common_20260914,prepare_candidate}.py (blend refs), pipeline/08_stack.sh, docs/stages/08-09_stack_and_package.md, README.md:280-376, docs/figures/make_figures.py:85-145 and figures/README.md (text only), docs/repro/mode_full_2026-09-28/ logs (parity, stage_logs).
- Assets (JSON only, via json.load in my own script): extracted/harness_zip/assets/{inference_config,tabular_recipe,linear_lin,linear_lin4}.json.
- Scored CSV `references/submissions/comp_subs/2nd_place/2nd_place_submission.csv` (sha256 293d7041037dd6e38b55de53efbf5838df44e7de1f0a007257ca50fd5ef8e37a); submission_format.csv (uid list only); train_labels.csv (uid column only, for grep).
- Relied on Coordinator findings (not redone): harness code byte-identical to `2nd Place/inference/`; 60 asset hashes match SHA256SUMS.scored.txt; CSV format checks; recomputed log loss 0.273088 matches leaderboard.

Competition-specific checks applied:
- RF "Other" / GR D (independent processing; no pseudo-labeling or unsupervised learning on test; no batch norm/test stats/cross-case calibration/pooled TTA): applied to the full inference path and the TabICL internals.
- GR C (no test logging): applied to every logger/print in inference and in TabICL.
- RF "Other" (predictions in [0,1], no hand editing; code matches leaderboard): static CSV vs code-path consistency.
- Overview CS1, CS2, CS4, CS16; flags F1, F3, F4, F5, F6, F9, F14; gaps G4, G6.

Evidence:
- `inference/regfit.py:68-94`, `register3.py:67-78,87-122,137-167`: registration loss is a sum of per-row (1-NCC) terms; rodrigues, warp, ncc and Adam are all per-row/elementwise; rejection is per-row (`regfit.py:62-65`). Template is frozen. Row-independent.
- `inference/portable3.py:51-57,91-93,143-145`; `inference_main.py:219`: all scaling uses frozen training constants (FeatPrep JSON, scalar mu/sd per 3D checkpoint, ImageNet mean/std for 2D). No test statistics.
- `nets3.py:23-27,30-40,94-149`: 3D nets are GroupNorm + per-sample SE/pooling in eval mode. 2D nets are TorchScript eval and frozen (`inference_main.py:83-84`; F2 resolved by the overview). `cnn2d.py:226-237`: flip TTA is within-image. 3D tta=0 (config). F6 confirmed.
- TabICL (`tabicl_lib.zip`): predict_proba -> fitted encoders -> `EnsembleGenerator.transform(mode="test")` (fitted preprocessors only, no refit) -> `forward_with_cache` repr mode (`_model/tabicl.py:734-797`). In `layers.py:432-437`, with train_size set, `k = v = q[..., :train_size, :]`, so every query attends only to the cached training positions. The ICL encoder has no RoPE over rows. Column ISAB uses cached train-derived K/V (`layers.py` induced_attention_with_cache). The row interactor attends across columns within a row. ssmax uses the key length (train size). InferenceManager splits only leading batch dims. **No test-to-test attention.** Residual cross-row couplings: (i) the all-NaN-column skip mask (`layers.py` ISAB forward_with_cache), neutralised by median imputation at `tabular_adapter.py:60`; (ii) if `normalizer_.transform` raises ValueError, the whole batch is clipped to train min/max (`_sklearn/preprocessing.py` PreprocessingPipeline.transform). This is a no-op for in-range rows and only theoretical (norm methods 'none'/'power', finite inputs clipped to +/-100). F3 resolved.
- `inference_main.py:230-481`, `main.py:10-17`: all logs are static strings. Warnings fire at most once. Crash output is type name + file:line only. TabICL prints are gated on `verbose` (False). Parity log `docs/repro/.../parity/scored_run.log` shows only versions + static lines. GR C satisfied.
- Grep of 524 test uids (and 1,362 train uids) over `2nd Place/` (text + binary), the extracted harness and the tabicl zip text: 0 hits. JSON assets have no uid-keyed keys/values. Training code enumerates scans from labels.csv / the staged training niftis dir; repro logs show 1,362 scans in every data stage. No pseudo-labeling (the "pseudo" hits are the "pseudo-site" training proxy).
- `inference_main.py:231-232,479`: submission_format is used only for uid order; values are overwritten (F4).
- CSV: 115 x 0.995 and 4 x 0.005 = clip `max(0.005,1e-6)` (`inference_main.py:475-477`). **0 rows at the prior 0.5484581497797357** (none within 1e-3), so in the scored run no governor rejection, preprocessing failure, non-finite fallback or missing uid occurred. 407 distinct full-precision values. Extremes are spread over all 5 blocks. No hand-editing signature.
- `src/fixstack.py:2-5,28-29,52-54`; `docs/repro/.../stage_logs/08_stack.log`: the stack scale/intercept are fitted on training OOF only. The ratio is credited to CV re-splits. T=1.4 and clip=0.005 are "from the board T-curve". On training OOF, T=1.4 is worse than T=1 (0.2167 vs 0.2070), which confirms that T came from public-LB feedback.
- `src/train_final3.py:56-61`: `head_ml` is excluded from lin/lin4 because "the board refuted it in v9".
- `tabular/prepare_tabular.py:162`: blend_weight 0.35 and TabICL T 1.2 are hard-coded. Training-OOF blend grids exist in `_original/common_20260914.py:60-63` and `bench_v16.py:47,68`, but no selection code. Provenance is undocumented.
- `docs/figures/make_figures.py:93-111`: 9 scored + 2 failed submissions. Public and private scores are typed constants used for figures only. T-only change v7->v10a improved both public (0.2611->0.2507) and private (0.2988->0.2854).

Findings:
- Severity: low
  Summary: Public-leaderboard feedback chose a few global scalar hyperparameters: temperature T=1.4 (worse on training OOF), clip 0.005, the exclusion of `head_ml` from the linear members, and possibly the TabICL blend 0.35 / T_tab 1.2. No per-uid, per-row or per-site constant exists, and 9 scored submissions cannot encode test labels. This is ordinary LB-guided tuning, not label probing or leakage, and the private scores moved the same way as the public ones.
  Evidence: src/fixstack.py:4-5; docs/repro/mode_full_2026-09-28/stage_logs/08_stack.log ("T=1: ll 0.2070 | T=1.4: ll 0.2167"); src/train_final3.py:56-61; docs/figures/make_figures.py:93-111; tabular/prepare_tabular.py:162. Contradicts the wording "chosen by hand" in docs/stages/08-09_stack_and_package.md:23-24.
  Recommendation: Organizer to note it. No rule prohibits LB-guided hyperparameter choice (rule F's "do not overfit to the test set" concerns annotations). Optionally ask the competitor to document how 0.35/1.2 and the 2D ratios were chosen, and to correct the "chosen by hand" wording in the write-up.
- Severity: low
  Summary: Block-level exception fallbacks (F1) are a cross-case dependency only on failure paths. If batched registration raises, all ≤128 scans in the block use canonical-frame features (and are not governor-rejected). If a CNN member raises, that member and every later member are dropped for the whole block and the stacker is renormalised. So one scan's co-membership in a failing block can change another scan's prediction. No feature or label of one case enters another's computation, so this is not information leakage. A content-driven trigger is implausible: crops are fixed at 48³, NaNs propagate without raising, and per-scan read failures are caught individually (`:360-365`). The likely triggers are OOM or device errors. It is disclosed in README:328-332.
  Evidence: inference_main.py:378-386, 438-462; README.md:328-332.
  Recommendation: Accept as non-blocking. Optionally, if the platform log for submission 322397 still exists, organizers can check that it lacks the static strings "Registration failed for a block" and "Neural-network inference was unavailable". The CSV cannot show whether these paths fired (G6).
- Severity: low
  Summary: Floating-point batch coupling. Batched registration and CNN kernels, and TabICL's whole-block `predict_proba`, make results depend at the ~1e-5 level (max 1.3e-3 reported) on which scans share a block. TabICL has two theoretical batch-wide branches: the all-NaN-column mask, neutralised by imputation, and the normalizer-ValueError clip, a no-op for in-range rows. Neither carries information between cases.
  Evidence: README.md:317-326; tabicl `_model/layers.py` ISAB forward_with_cache skip_mask; `_sklearn/preprocessing.py` PreprocessingPipeline.transform try/except.
  Recommendation: None required. Informational.
- Severity: low (info)
  Summary: No test-data use in training, no pseudo-labeling, no hard-coded predictions or uids, and no test logging were found. The scored CSV matches the code path exactly: clip counts 115/4, zero prior-valued rows, no rounding.
  Evidence: N3–N6 above.
  Recommendation: None.

Open questions:
- What does the "150-scan local verification set" (README:322, REPRODUCIBILITY.md:30) contain? It is presumably a subset of the 1,362 training scans, since test scans were never distributed. Ask the competitor to confirm.
- How were the TabICL blend weight 0.35 / temperature 1.2 and the 2D stack ratios 0.3/0.3 chosen (OOF vs public LB)? The development notes (RESEARCH_NOTES §67/§72/§60b) are not distributed.
- Is the platform log of submission 322397 retained? It would confirm that no block-level fallback fired.
- The tabicl_lib.zip internals were read as shipped. That they are identical to the PyPI tabicl 2.2.0 wheel is not verified (no fetch). The independence conclusion rests on the shipped code itself, so it does not depend on that claim.
