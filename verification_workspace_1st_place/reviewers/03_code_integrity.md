## Subagent: Reviewer 3 — Code Integrity and Anti-Cheating (1st place, TheAvengers)

Status: needs organizer review (one medium item: leaderboard-informed temperature; otherwise pass with notes)

No competitor code executed: yes. No competitor modules imported, no artifacts loaded/unpickled, weights.zip not extracted, no images viewed. Only text reading (cat/sed/grep/diff) and grep of uid strings.

Path conventions: TREE = `1st Place/`; HZ = `verification_workspace_1st_place/extracted/harness_zip/`; SZ = `verification_workspace_1st_place/extracted/src_zip/src/`.

## 1. Scored inference code path (independently verified)

| Step | Evidence | Notes |
|---|---|---|
| Entry, config | HZ/main.py L20-22 (DATA_DIR fallback to `data/`), L24-26, L31 `MODE_INFERENCE="FAMILY_STACKING"`, L37-42 `FAMILY_MODELS=["Mashup_Internal"]`, L43 `TEMP=1.094` | Only one family active |
| Unzip, imports | HZ/main.py L63-69, L74, L79-147 | No import-time artifact loads/network in executed modules (see §3) |
| Arbiter | HZ/main.py L233 -> SZ/inference_feats.py L413-465 `LogitFamilyStackingInference` (default clip_bounds (0.01,0.99), L414) | |
| Model load | HZ/main.py L286-289 -> SZ/inference_2d_avg.py L861-887: loads `weights/mashup_ensemble_exp_{0..19}_fold_{0..4}.pth`, `build_model("siamese_internal")`, `model.eval()` (L881) | Silently skips missing files (L873-874) |
| Per-case loop | HZ/main.py L432-441: iterates `submission_format["uid"]`, one NIfTI at a time; `diagnostiquer_patient` L395-408 | No accumulation across uids |
| Preprocessing | SZ/preprocessing.py `prepare_all_inputs` L662-741 -> `preprocess_patient_numpy` L190-219 and `extract_siamese_patches` L302-470 | All statistics (p95, p98, background median, min/max) from the single volume |
| Model input | SZ/inference_2d_avg.py L889-921: channel 0 of L/R/Global, flip, fade, center-crop, fixed ImageNet mean/std constants | Batch size 1 |
| Forward | SZ/inference_2d_avg.py L942-957; SZ/builder.py L895-934 `MashupInternalEnsembleCNN` returns `(mean_logit, ...)`; `outputs[0]`=mean of 3 branch logits | BN layers in eval mode (running stats from training) |
| Ensemble | `_apply_recipe` L923-940: plain `np.mean` over 100 sigmoid probs (old T=0.884/clip 0.002 recipe commented out L925-937) | |
| Calibration | SZ/inference_feats.py L426-458: clip 1e-7, logit, mean over 1 family (identity), `expit(z/1.094)`, `clip(0.01,0.99)`; `post_process` identity L460-465 | Final p = clip(sigmoid(logit(mean100 p_i)/1.094), 0.01, 0.99) |
| Output | HZ/main.py L441-444 | 0.5 fallback if NIfTI missing (L435-439) or if patch keys missing (inference_2d_avg.py L943-944) |

Conclusion: Reviewer 1's path map is confirmed.

Tree vs harness for executed code: `diff` of TREE/src vs SZ shows `preprocessing.py`, `calibrators.py`, `striatum_*.py`, `inference_tta_wrappers.py` identical; `inference_2d_avg.py` hunks all lie in lines 17-210 (ResNet2D/ViT2D classes, L23-195), not in `SpatialMedialFadeInference` (L339) or `SiameseMashupInternalEnsemble` (L861+); `builder.py` hunk L1003-1012 (mednext suffix parsing) only; `inference_feats.py` hunk L28 (TabularInference pkl name) only. Tree `main.py` vs harness: only DATA_DIR (L20) and comments. So the executed path is byte-identical in the delivered tree apart from DATA_DIR.

## 2. Weights and training -> artifact correspondence

- `notes/weights_zip_listing.txt`: exactly 100 `.pth`, names `mashup_ensemble_exp_{0..19}_fold_{0..4}` (100 distinct), no `.pkl`/json/other. Sizes: 50 x 7,878,799 B (exp 0-9) and 50 x 7,878,907 B (exp 10-19). The 108-byte step coincides exactly with the one-character-longer filename stem, which is what `torch.save` of a state_dict embeds in its archive record names; content size ~7.88 MB is consistent with ~1.96 M float32 params + BN buffers. Consistent with a single training script writing `torch.save(model.state_dict(), .../mashup_ensemble_exp_{exp}_fold_{fold}.pth)` (TREE/4_siamese_training/train_siamese.py L306, L382). Not loaded; content not verified.
- Training architecture (train_siamese.py L171-215) matches inference builder (SZ/builder.py L886-934) by reading.

## 3. Training code path and test-data checks

Training data sources (grep of `read_csv|glob|cfg.paths.*|nib.load` over `1_initial_training`, `2_stacking`, `3_soft_labeling`, `4_siamese_training`, `5_pipeline`, `6_runtime`, `scripts`):
- Real data: `cfg.paths.niftis` + `cfg.paths.train_labels` (= `train_labels_with_folds.csv`, produced by `1_initial_training/prepare_folds.py` L12-24 from `train_labels_JNDlMjr.csv`, StratifiedKFold(5, shuffle, seed 42) on training uids). Every Block-1 trainer reads only these (train_3d_*.py L28-31, train_medicalnet.py L48-49/L241, train_svm.py L48-49, prepare_2d_dataset.py L22-27; 2D trainers read `folds_2d_csv` derived from it).
- Synthetic: `cfg.paths.synthetic_niftis` = `<root>/external/PET_GENERATED` (0_configs/default.yaml L14); soft-labelled in `3_soft_labeling/soft_labeling.py` L24, L217 (glob of that dir only) -> `synthetic_soft_labels.csv` (L254).
- Final Siamese (train_siamese.py): real patches from `folds_siamese.csv` built by `prepare_2d_dataset_siamese.py` L85-89 (`--mode real`: train niftis + train labels) and synthetic patches (L80-84: synthetic niftis + soft labels, `INVERT_Y=True`). Splits: real uids sorted, `np.roll(exp*len//20)`, `array_split` into 5 (L220-239); all synthetic rows appended to every training fold with `fold=-1` (L246-262) - synthetic never in validation. Early stopping / checkpoint selection on the real validation fold log loss (L355-387, PATIENCE=60). Soft targets used only for synthetic rows (`SOFT_LABEL_COL='label'`). `USE_RELABEL=False` (L98-99): an optional training-label relabel map (`is_pathologic_fixed_soft`, L241-244) is disabled and no relabel CSV is delivered. `SYNTH_LABELS=None` (L65).
- `stacking.py` L376-412 merges teacher OOFs on training uids only; `PredefinedSplit` on training folds.
- No training/preprocessing script references `submission_format`, `/code_execution`, test niftis, or test labels. The only `submission_format` references are in the inference entrypoints (main.py) and the local `validate_pipeline.py` phase 7 (format check, L711-727: counts rows/columns and checks NIfTI existence; no predictions or labels used). `/code_execution` appears only in the DrivenData runtime `justfile` L222-241 (docker mounts).
- `pseudo`: only the prefix `"siamois_final_pseudo"` of an older, inactive `SimpleSiameseEnsemble` family (HZ/main.py L260-261; TREE main.py L254; HZ/constants.py L9 commented; HZ/generate_tta_oof.py L164). No training code for it in tree; no such weights in weights.zip. Name is consistent with "pseudo"/soft labels on synthetic PET. No code path pseudo-labels test data. `manual`/`hand`: only `torch.manual_seed` and validate_pipeline's `p_manual` (manual re-computation of a mean).
- Participants never had test NIfTIs (code-execution competition); the team's data roots (`/data2/iceberg/dat_parkinson/niftis`, `<root>/niftis`) are training scans. Hard-coded uid in HZ/compare_inputs.py L15 (`05xt54fn`) is a training uid.

Test-UID grep (re-confirmed): 524 uids from `.../private/submission_format.csv` (uid column only; test_labels.csv not opened) grepped (text and binary-inclusive, `grep -a`) against `1st Place/` and `extracted/` (harness root, src.zip text members): **0 hits**.

Harness CSV uid coverage vs `.../public/train_labels.csv` (1,362 uids):
- `tta_siamese_predictions.csv`: 16,344 rows = 1,362 distinct uids x 12 TTA names; 1,362/1,362 in train, 0 in test. Produced by HZ/run_siamese_tta.py (L14-20, L206) from an older `mashup_attention` model's OOF; not the scored model.
- `validation_report.csv`: 500 rows, 100 distinct uids, family `3br_mean_logaux`; 100/100 in train, 0 in test. Produced by validate_pipeline.py L782 / validate_inference.py L353 (paired OOF vs re-inference check).
- Both contain training labels (release/data-retention item, already flagged by Coordinator/R1).

Within-training-CV note (info, not a rule issue): the synthetic soft labels come from Block-1 teachers fit on all real training folds, and the same soft-labelled synthetic set is added to every Siamese fold, so real-validation-fold OOF numbers (README L228-232: OOF log loss ~0.200) are optimistic relative to unseen data (private LB 0.2532). This is consistent with the team needing a post-hoc "offset" (see §5).

## 4. Per-observation independence (CS1) and test logging (CS4)

- Loop is strictly per uid (HZ/main.py L432-441); `prepare_all_inputs` takes one path; no list/batch of test cases is ever built. Batch size 1 for all 100 forwards (`unsqueeze(0)`, inference_2d_avg.py L913-915).
- BatchNorm: `model.eval()` at load (inference_2d_avg.py L881); forward under `torch.no_grad()` (L949). Running statistics come from training; no test-time BN adaptation. Dropout inactive in eval.
- Normalisation: per-volume statistics only (preprocessing.py L196-217: own p95/min/max; L338-339, L363, L386-396: own p98, own background median), plus fixed ImageNet mean/std constants (inference_2d_avg.py L907-908). MONAI `ScaleIntensityd` is per-image (L168).
- No scalers/reducers fitted at test time in the executed path. Only `.fit`/`fit_transform` calls in src are in `calibrators.py` (training-time calibrator classes, never called at inference) and `StackingInference.post_process` t-SNE branch (inference_feats.py L268-301) which is not on the executed path (arbiter is `LogitFamilyStackingInference`; no `.pkl` in weights.zip). openTSNE vendored but not used.
- Global/module state: `_DEFAULT_TRANSFORMS` (preprocessing.py L171) is a stateless MONAI Compose; `striatum_preproc._PREPROC` singleton (L122-147) is used only by TwoPhase code, whose calls are commented out in `prepare_all_inputs` (L675-685). No caches keyed by case; `SiameseMashupInternalEnsemble` holds only models and fixed transforms.
- Fallbacks: missing NIfTI -> 0.5 (main.py L435-439); missing patch keys -> 0.5 (inference_2d_avg.py L943-944); `extract_siamese_patches` falls back to mirrored/default centres when <2 peaks (L341-375) - all per case. Scored CSV has no 0.5 rows. No per-case try/except in main (an exception would abort the run).
- Test logging (rule C): executed path prints only at init (model list, T/clip, loaded count: main.py L247, inference_feats.py L422-423, inference_2d_avg.py L868, L886) and start/end messages. Per-case: (a) `logger.warning` with uid only if a test NIfTI is missing (main.py L438) - did not fire per CSV; (b) `extract_features` (computed for every case though unused by the scored model) logs the **names** of NaN/Inf radiomics features (preprocessing.py L579-585), no uid or values. Low severity; it reveals at most that some test case had a degenerate feature. `tqdm` disabled (`TQDM=False`, L27).
- Network: no `requests/urllib/socket/hf_hub/from_pretrained/torch.hub` in src (only a URL inside a MedNeXt comment). `inference_2d_rnn.py` L36 default `pretrained=True` is overridden with False at construction (R1) and the class is not instantiated in the scored config.
- Privacy (CS6): only `img.header.get_zooms()` (voxel spacing) is read from NIfTI headers (preprocessing.py L193; train_medicalnet.py L128).

Conclusion CS1: **pass** - each test case is processed independently in the scored path.

## 5. Hard-coded predictions / uid dictionaries / leaderboard-informed constants

- No hard-coded predictions, uid->value dictionaries, or test-uid lists anywhere (0 test-uid hits, §3). The only `uid`-keyed maps are training-time (`id_to_fold`, `relabel_map` (disabled), `synth_map`) in train_siamese.py L232, L243, L252.
- Grep `leaderboard|LB|public score|1.245|0.8835|0.884|0.9784|1.094|clip` over tree + harness (excluding vendored libs and runtime image files): the only leaderboard references are in `validate_pipeline.py` (identical text in TREE `competition-sfmn-parkinsons-runtime/submission_src/validate_pipeline.py` and HZ copy):
  - L58-59 `EXPECTED_T = 1.094`, `EXPECTED_CLIP = (0.01, 0.99)`;
  - L123-125 prints implied T for M=100 = 1.094/1.245 = 0.8787 and for M=20 = 1.094/1.245/0.99515 = 0.8830, "(facteur 1.245 = correction de décalage mesurée sur le leaderboard)" = "factor 1.245 = offset correction measured on the leaderboard";
  - L698-702 "la T soumise contient le facteur 1.245 de décalage leaderboard. Elle est donc DÉLIBÉRÉMENT sous-optimale sur l'OOF" = "the submitted T contains the 1.245 leaderboard offset factor; it is therefore DELIBERATELY sub-optimal on the OOF".
- Earlier frozen recipe (HZ/validate_inference.py L53-60, "recopié depuis la sortie 'RECETTE FIGEE' du notebook"): `EXPECTED_T = 0.8835259617309265`, `EXPECTED_CLIP = 0.002`, `EXPECTED_PREVALENCE = 0.54846` (= training prevalence, 747/1362, verified), `EXPECTED_LL_OOF = 0.1983`. Same recipe is the commented-out `_apply_recipe` in SZ/inference_2d_avg.py L925-936 (T 0.884, clip 0.002). Arithmetic: 0.88353 x 1.245 x 0.99515 = 1.0947 ~= 1.094. So the deployed T = (OOF-fitted T) x (leaderboard-measured factor 1.245), with a small M=20->100 ensemble-size correction (0.99515). The clip was also widened from 0.002 to 0.01 with no OOF justification in delivered code (provenance undetermined; plausibly also leaderboard-informed, but no text says so). The `LogitFamilyStackingInference` default T=0.9784 (inference_feats.py L414) is another, overridden value.
- The notebook that derived T, the factor 1.245 and the clip is **not delivered**.

Nature and magnitude:
- Two global scalars (T, clip) chosen with public-leaderboard feedback. No per-case information, no test uids, no labels; the same transform is applied to every case, so it does not break per-observation independence (rule D) in the sense of using other cases' data at inference. It is, however, a test-set-informed post-processing choice: README omits it; rule F says annotations must "not overfit to the test set", and the generic checklist flags "leaderboard-specific post-processing" / tuning or threshold selection on test data as an escalation trigger. Per Coordinator's `prediction_csv_check.txt`, the official score 0.253233 is computed on all 524 test rows, i.e. the public subset (262 rows) used for this feedback is **half of the final scoring set**.
- Direction: T moved from 0.88 (sharpening) to 1.094 (softening); i.e. LB feedback indicated the model was over-confident on test (consistent with OOF 0.198-0.200 vs test 0.253).
- Static, label-free magnitude on the scored CSV (inverting the deployed transform; exact because sharper T keeps 0.99-clipped rows clipped): replacing T=1.094 by the OOF-only T (0.8787 for M=100) changes the 415 unclipped predictions by mean |dp| = 0.026 (median 0.025, max 0.049) and pushes 94 more rows to a clip bound; T=1.0 would change them by mean 0.012. The log-loss effect cannot be computed without test labels (not opened by this reviewer). Expected order of magnitude: about 0.01-0.03 log loss, comparable to the 1st-2nd gap (0.2532 vs 0.2731 = 0.0199). **Recommendation:** Coordinator/organizer (who hold labels) compute the counterfactual score with p' = clip(sigmoid(logit(p)*1.094/T'), 0.01, 0.99) for T' = 0.8787 (OOF-only) and T' = 1.0, on the private-only 262 rows and on all 524, to see whether the ranking depends on the LB-tuned scalar. This is exact for all rows with the 0.01/0.99 clip (sharper T keeps the 109 clipped rows clipped). For the original frozen recipe (clip 0.002/0.998) the 109 rows at 0.99 are only bounded (their raw mean is >= 0.99348, so p' lies in [0.99673, 0.998]); bounding them both ways changes per-row loss by at most ~0.0013 for a positive (larger only if a 0.99 row is a true negative).

## 6. Static code <-> leaderboard match

- Scored CSV (`1st_place_submission.csv`, sha256 82878000...7ece): 524 rows, uid order == submission_format (written by `submission_format.loc[...]` and `to_csv`, main.py L441-444, which preserves order and columns). Values in [0.012561, 0.99]; 109 rows exactly `0.99` (the only short-decimal value); all other values are distinct full-precision floats. Inverting the deployed transform, the largest unclipped raw ensemble mean is 0.993435, just below the clip threshold sigmoid(logit(0.99)x1.094) = 0.993485; the smallest raw mean is 0.00837 (> 0.0065 lower-clip threshold, so lower clip not hit). This is exactly the output form of clip(sigmoid(logit(mean)/1.094), 0.01, 0.99). No 0.5 fallback rows. No sign of hand editing.
- Harness vs tree differences on the executed path: none apart from tree `main.py` L20 `DATA_DIR = Path()` (tree would look for `./submission_format.csv`, `./niftis` and exit without output in the container - reproducibility bug, low). All src diffs are in inactive classes (§1). Harness extras (8 scripts, 2 CSVs, `__pycache__`) are not imported by main.py.
- Reproducing the harness from the tree (`6_runtime/pack_production.py` L55-117): copies TREE/src -> src.zip (L64-66, L90) and every `*.pth` rglob'd under `4_siamese_training/results` (L71-76) -> weights.zip (L91), meta-model if present (L80-84); then `just pack-submission` zips all of `submission_src/*`. Gaps: (a) TREE/src lacks `src/libs/` (vendored openTSNE, ptwt, pykwalify, pywt, radiomics, ruamel.yaml, xgboost), but executed modules import `ptwt` (builder.py L10), `xgboost` (inference_feats.py L16), `radiomics` (preprocessing.py L27) at module level; the tree instead adds them to `runtime/pyproject.toml` L58-64 (a modified runtime image), so a tree re-pack only works if those packages are in the official image; (b) tree adds `src/models/`; (c) the scored weights.zip has no meta_model .pkl. So the scored zip was built from a different working copy, but the executed code is the same. Low.
- Weights correspondence: 100 checkpoints, names and count exactly match `train_siamese.py` output (20 exps x 5 folds, L76-77, L306); size pattern consistent with `torch.save(state_dict)` (§2). `scripts/download_weights.py` copies HF `02-siamese/*.pth` into the same location (alt. source). Content not verifiable statically.

## 7. Harness helper scripts (not executed by scoring)

| Script | Data touched | Test data / leaderboard? |
|---|---|---|
| validate_pipeline.py | team OOF (`/data/HOME/sansel/.../master_oof_predictions.csv`), training niftis; writes validation_report.csv | Phase 7 reads `submission_format.csv` only for format/NIfTI-existence counts (L711-727). Documents the LB factor (§5). No test predictions/labels. |
| validate_inference.py | same OOF; folds CSVs (L74-77); writes validation_report.csv | No. Holds pre-LB frozen recipe (T 0.8835, clip 0.002). |
| test_inference.py | OOF CSV, training niftis | No |
| generate_tta_oof.py | training `folds_siamese.csv` + `/data2/.../niftis` by fold (L39-40, L193-203); saves debug PNG grids of 10 training patients (L76-156) | No. (Images of training scans may exist on team disk; not in zip.) |
| run_siamese_tta.py | training patches + older `mashup_attention` OOF (L14-20) -> tta_siamese_predictions.csv | No |
| check_orientation.py, check_tta_augmentations.py | training folds CSV + niftis; save debug PNGs | No |
| compare_inputs.py | one training uid (`05xt54fn`), training mask bundle, spine-profile CSV | No |
| diagnose3d.py | MiniCNN3D OOF (training) | No |
| copy_weights.py | team home dirs -> weights | No |

None references test uids, submission scores or leaderboard values (other than validate_pipeline's text). Several fall back to `Path("data")` (would be the test mount if run in-container) but main.py never imports them.

## 8. Findings

- Severity: medium (needs organizer review)
  Summary: Deployed temperature T=1.094 is the OOF-fitted T (0.8835) multiplied by a "factor 1.245 = offset correction measured on the leaderboard"; clip also widened 0.002 -> 0.01 without OOF justification; deriving notebook not delivered; README silent.
  Evidence: HZ/validate_pipeline.py L58-59, L121-125, L696-702 (same in TREE submission_src/validate_pipeline.py); HZ/validate_inference.py L53-60; SZ/inference_2d_avg.py L925-936; HZ/main.py L43; SZ/inference_feats.py L414, L453-456.
  Recommendation: Organizer to rule whether a single global calibration scalar tuned on public-LB feedback is acceptable under rule F ("do not overfit to the test set") and the generic "no tuning/threshold selection on test data" check. Note the public subset is half of the final scoring set. Compute the counterfactual score with the OOF-only T (formula in §5) to see how much of the score and ranking depends on it. Ask the team to deliver the notebook and document the step. Static estimate: predictions shift by mean 0.026 (max 0.049) in probability; plausible log-loss effect about 0.01-0.03, the same order as the 1st-2nd gap (0.0199).
- Severity: low
  Summary: Tree `main.py` `DATA_DIR = Path()` would not find `data/submission_format.csv` in the container, and tree `src/` lacks the vendored `libs/` that the scored zip relies on for `ptwt`/`xgboost`/`radiomics` (tree instead modifies runtime pyproject). The executed inference code is otherwise byte-identical.
  Evidence: TREE submission_src/main.py L20 vs HZ/main.py L20-22; HZ/src.zip listing `src/libs/`; SZ/builder.py L10, inference_feats.py L16, preprocessing.py L27; TREE runtime/pyproject.toml L58-64.
  Recommendation: Team to fix DATA_DIR and vendor or document the extra deps for release.
- Severity: low
  Summary: Per-case log line can name NaN/Inf radiomics features (no uid, no values) for test cases; a uid is logged only if a test NIfTI is missing. Minimal info about the test set (rule C).
  Evidence: SZ/preprocessing.py L579-585; HZ/main.py L438.
  Recommendation: Note only; suggest removing for release.
- Severity: low / info
  Summary: Harness bundles training-uid OOF CSVs with labels and diagnostic scripts with team paths (already flagged to Release Readiness). Neither touches test data.
  Evidence: HZ/tta_siamese_predictions.csv (1,362 train uids), HZ/validation_report.csv (100 train uids); 0 test uids.
- Severity: info
  Summary: Siamese OOF is optimistic (soft labels from teachers trained on all real folds feed every fold), which helps explain the OOF (0.198-0.200) vs test (0.253) gap behind the LB "offset". Not a rules issue.
  Evidence: train_siamese.py L246-262; soft_labeling.py L151-217.

No finding of: test data in training, pseudo-labelling of test, test-label use, hard-coded predictions, cross-case pooling at inference, transductive t-SNE, network calls, or code/CSV mismatch.

## 9. Competition-specific checks applied

| Check | Result |
|---|---|
| CS1 independence | Pass (§4) |
| CS2 harness layout / runtime | Pass: main.py at root, writes submission.csv, reads `data/niftis` + `data/submission_format.csv` via fallback, offline, all assets bundled. Runtime not statically measurable; validate_pipeline extrapolates timing (L733-752). Low notes on tree repack (§6). |
| CS3 predictions format | Pass (§6) |
| CS4 test logging | Pass with low note (§4) |
| CS5 network / third-party uploads | Pass for inference. Training code downloads public HF checkpoints only (train_2d_resnet.py L128, train_2d_vit.py L104); no uploads found. |
| CS6 privacy | Pass: only voxel spacing read from headers |
| Rule F / leaderboard tuning | Needs organizer review (§5) |

## Open questions
- Was the factor 1.245 (and the clip 0.01) derived from one or several public-LB submissions? How many probes? (Platform submission history may show the T sweep; DrivenData can check.)
- Can the team deliver the notebook that produced "RECETTE FIGEE" and the 1.245 factor?

## Process confirmation
No competitor code executed, imported, or installed; no artifact loaded/unpickled; weights.zip not extracted (listing only); no images opened or rendered; test_labels.csv not opened; no network access used. The only computations were Python/grep over plain-text CSVs (scored CSV, harness CSVs, train_labels.csv, submission_format uid column) and arithmetic on constants.
