# Solution Overview — 2nd place, team NavAttack (DaT Parkinson's Challenge, comp 311)

Author: Reviewer 1 (Solution Explanation). Static, read-only review. No competitor code executed, no images viewed, no artifacts loaded/unpickled, no web fetches.

Source tree: `2nd Place/` (abbreviated `SP/` below). Shipped harness copy: `verification_workspace_2nd_place/extracted/harness_zip/` (abbreviated `HZ/`). The 13 shared modules in `SP/src/` and `SP/inference/` (cnn2d, cnn3, cnn4, features4/5/6/13, nets3, portable3, preprocessing3, regfit, register3, roigeom) were confirmed byte-identical with `cmp`. Line numbers cited for them hold for both copies.

---

## 1. Competition-Specific Checks

Each check comes from `references/verification_comp_specific_rules.md` (RF) or `notes/governing_rules.md` (GR A–F) and has an owner.

| # | Check | Source | Owner | Pointers from this overview |
|---|---|---|---|---|
| CS1 | Test samples are processed independently at inference. Look for batch-level normalization, test-set statistics, cross-case calibration, clustering, TTA that pools cases, pseudo-labeling, or unsupervised learning on test data. | RF "Other", GR D | Code Integrity | See §7 Flags F1–F6. Scans are processed in blocks of 128 (`inference_main.py:49,334`). Registration is batched (`regfit.py:68-94`). CNN batches are 32. TabICL `predict` runs on the block (`tabular_adapter.py:53-63`). Fallbacks apply to the whole block (`inference_main.py:381-386,439-449`). |
| CS2 | No logging of test-set information. | GR C | Code Integrity | Logger calls in `inference_main.py` are static strings. `main.py:10-17` prints package versions and the exception type plus file:line. The f-string in the `RuntimeError` at `inference_main.py:275` reports counts of missing feature names, not test data. |
| CS3 | Package layout: `main.py` at zip root, writes `submission.csv`, reads `/code_execution/data/niftis/` and `submission_format.csv`, runs offline within 3 h on an A100, bundles all assets. Compare the harness zip with the tree and the README. | RF "Other" | Code Integrity | Coordinator finding 1–2: zip is identical to `SP/inference/` plus the 60 assets. `inference_main.py:45-48` sets the paths. The TabICL adapter forces `HF_HUB_OFFLINE` (`tabular_adapter.py:5-6`). |
| CS4 | Predictions are probabilities in [0,1]. Check format, UID coverage and signs of hand editing. | RF "Other" | Code Integrity | Clip at `inference_main.py:475-477` with clip 0.005 from config. The 0.995/0.005 values are explained by the clip. Coordinator finding 3. |
| CS5 | Every external dataset and pretrained model is disclosed and cross-checked against what the code loads. | RF "Other" | Data Source | External models: TabICLv2 checkpoint (HF `jingang/TabICL`, BSD-3) and timm EfficientNet-B0/B3 ImageNet weights (Apache-2.0). MedicalNet appears only in unused archs (`nets3.py:225-308`). No external datasets found so far. |
| CS6 | Competitor annotations: training scans only, delivered, not tuned to the test set. | RF "Other", GR F | Data Source | No manual annotations found. The "acquisition groups" (`fam2`, `scale`) are derived automatically from NIfTI headers (`tabular/derive_groups.py`, `tools/hdr_fingerprint.py`). The Data Source reviewer should confirm whether these count as annotations. |
| CS7 | Pretrained-model license and pretraining data: no NC; pretraining data is also subject to the rules. | RF "Pretrained models" | Data Source / Eligibility | ImageNet-1k pretraining: THIRD_PARTY_NOTICES.md:61 itself states that ImageNet terms are non-commercial. **Flag F10.** MedicalNet is not used in the shipped model. |
| CS8 | License: solution under MIT; dependencies OSS and commercial-use OK. | RF "Release" | Release Readiness | `SP/LICENSE` is MIT. TabICL is BSD-3 with an Apache-2.0 subpackage. The dependency table is in THIRD_PARTY_NOTICES.md §4. CatBoost is imported only by `tabular/_original/bench_v16.py`. |
| CS9 | Privacy: no reading or derivation of patient-identifying header fields; no linking scans to individuals. | RF "Privacy" | Data Source / Code Integrity | Inference uses only the affine and zooms (`preprocessing3.py:76-87`). `tools/hdr_fingerprint.py` does a header survey (training only). **Flag F7.** |
| CS10 | No third-party APIs or network: no network at inference; no upload of comp data anywhere in training. | RF "Third-party APIs" | Code Integrity / Data Source | Network calls: `tools/fetch_backbones.py` (HF download of public weights), `cnn2d.build` with `pretrained=True` (timm/HF), and `scripts/install_weights.py --url` (Dropbox download of the competitor's own bundle). **Flag F8.** |
| CS11 | AI tools that retain uploaded data (Codex/ChatGPT) are forbidden on comp data. Check disclosures. | GR A | Data Source | See §6 F12 (comments refer to an "external review member E", `cnn2d.py:72`). |
| CS12 | External data licence must be permissive (CC0/CC-BY-like). | GR B | Data Source / Eligibility | No external data found. Only the pretrained weights above. |
| CS13 | Data used solely for the competition; retention or hosting of derived data. Public Dropbox weights bundles and label-bearing TabICL contexts. | GR E, RF "Release" | Release Readiness | Coordinator finding 4. README:131,153 link public Dropbox bundles. README:369-375 and models/README.md:25-31 say the frozen contexts let training labels be recovered. The organizer ruling is that weights may be held only privately with organizer-only access. |
| CS14 | Submission caps / multiple accounts. | RF "Submission limits" | Eligibility | Checked by DrivenData, no issues (organizer). |
| CS15 | Winner identity / sponsor / panel eligibility. | RF "Eligibility" | Eligibility | Confirmed by DrivenData (organizer). Static review cannot add to this. |
| CS16 | Leaderboard feedback used for tuning (overfit to the test set). Related to the GR F "do not overfit to the test set" wording. | GR F (spirit), RF CI | Code Integrity / organizer | `src/fixstack.py:4` says T and the clip were chosen "from the board T-curve". **Flag F9.** |
| CS17 | Reviewers must not view images. Figures are inferred from code. | RF "Handling" | All | `docs/figures/*.png/.svg` were not viewed. See §5 make_figures. |

## 2. Dataset Policy

**Forbidden**
- Any external data that is not freely and publicly available, including private clinical or institutional archives.
- Any data or model under an NC licence (no NC, CC NC, CC BY-NC).
- Data the competitor lacks rights to.
- Nothing in the competition data is in any public dataset (organizer). No dataset is named as forbidden.
- PPMI / other DaT-SPECT cohorts (grep aliases: `ppmi`, `loni`, `ida.loni.usc.edu`, `datscan`, `ioflupane`) are OK only if the DUA says the data is open source.

**Allowed**
- Competition training NIfTIs plus `train_labels.csv` (1,362 scans).
- External data and pretrained models that are freely and publicly available under licences allowing commercial use, with full rights, and disclosed to organizers.
- No datasets were specifically approved by the organizers.

**What this solution uses (static)**
- Competition training data only, for all training (`src/folds.py:38`, `src/cache3.py:52`). Labels are read from `DATA/labels.csv`.
- Pretrained: timm `efficientnet_b0.ra_in1k` and `efficientnet_b3.ra2_in1k` (ImageNet-1k pretraining, Apache-2.0 weights), and the TabICLv2 checkpoint (`tabicl-classifier-v2-20260212.ckpt`, BSD-3, pretrained on synthetic priors per TabICL papers — not verified here).
- Not used in the shipped model: MedicalNet (Tencent Med3D, 23 medical datasets). It is referenced in `nets3.py:225-308` archs `med*` and loaded from `<project>/external/`, which is absent. Shipped arch is `mil`.

**Unresolved questions**
- Q1: Does ImageNet-1k pretraining of the Apache-2.0 timm weights trigger the rule that pretraining data is subject to the same rules? ImageNet terms are non-commercial (THIRD_PARTY_NOTICES.md:61). Organizer ruling needed. This is a common pattern, but the rules file says pretraining data is subject to the same rules.
- Q2: The TabICLv2 pretraining data (synthetic priors). Data Source should confirm from the model card/paper; no fetch was done here.
- Q3: Do automatically derived acquisition-group labels (from header fingerprints) count as "annotations" under rule F? They are training-only and used for balanced subsetting and leave-family-out priors.

---

## 3. Pipeline summary (ordered)

### Training (pipeline/train_all.sh, MODE=full or MODE=fast)
`train_all.sh:38-40` and every stage script unset all `DAT_*` research switches except `DAT_DATA_DIR`. Stage 02 sets `DAT_MASK_V2=1` explicitly. Stage 04 sets mask/frame flags per cache.

| Stage | Script | What it does | Main code |
|---|---|---|---|
| 00 | `pipeline/00_stage_data.sh` | Unzips the training NIfTIs to `data/raw/niftis/`, copies the labels to `data/raw/labels.csv`, checks the sha256 of the competition download (info only), and prints counts. | shell |
| 01 | `pipeline/01_folds.sh` | StratifiedKFold(5, shuffle, seed 42) on labels → `outputs/folds.csv` | `src/folds.py:20-41` |
| 02 | `pipeline/02_features.sh` | Stages the frozen `template_v6.npy` (sha check), preprocesses each training scan, rigidly registers it to the template, extracts features13 (+ narrow v4 table), then applies filters (drop `hole_ml`, `head_ml`) | `src/make_regfeat32.py`, `src/filter_feature_tables.py` |
| 03 | `pipeline/03_linear.sh` | 5-fold OOF of `lin` (235 feats) / `lin4` (94 feats) L1 logistic, and full-data export to `linear_lin*.json`. Also fits LightGBM/MLP side models that are not shipped. | `src/linear_oof.py`, `src/train_final3.py` |
| 04 | `pipeline/04_caches.sh` | 48³ crop caches: `crops_v3_b.npz` (DAT_MASK_V2=0, for 3D) and `crops_v3_f2.npz` (mask+frame guards, for 2D) | `src/cache3.py` |
| 05 | `pipeline/05_cnn3d.sh` | 10 HemiMIL 3D runs (6 five-fold + 4 full-data), 20 epochs → 34 `.pt`, then `avg_member.py` → `oof_mil48y.csv` | `src/cnn3.py`, `src/nets3.py`, `src/avg_member.py` |
| 06 | `pipeline/06_cnn2d.sh` | EfficientNet-B0 ×2 seeds + B3, 5-fold + full, 30 epochs, on 3-MIP images; TorchScript export | `src/cnn2d.py`, `tools/ts_export.py`, `tools/fetch_backbones.py`, `src/avg_2d_seeds.py` |
| 07 | `pipeline/07_tabular.sh` | Acquisition groups from header survey + EDA geometry, 180 scale-space shape + 60 mixture features, TabICL recipe (feature selection, 4 priors, log prep), frozen contexts | `tabular/*.py`, `tools/hdr_fingerprint.py`, `src/eda_real.py` |
| 08 | `pipeline/08_stack.sh` | Fixed-ratio stacker (1:0.4:0.7:0.3:0.3) + 1-D logistic calibration on OOF logits, T=1.4, clip 0.005 → `ensemble_v14c.json`. `build3.py` writes the v14c `inference_config.json` and assets. | `src/fixstack.py`, `src/build3.py` |
| 09 | `pipeline/09_package.sh` | Copies `inference/` verbatim + assets, extends `feature_order` 235→479, adds `"tabicl": true`, runs static gates, writes the zip | `src/package_submission.py`, `src/check_package.py` |
| fast | `train_all.sh:57-81` | Copies the released CNN checkpoints and replays their OOF (`tools/replay_cnn_oof.py`) instead of retraining | |

### Inference (inference/main.py → inference_main.main)
1. `main.py:6-13` puts `tabicl_lib.zip` on sys.path, prints 7 package versions, and calls `inference_main.main()`. On an exception it prints only the type name and file:line (`main.py:14-18`).
2. Reads `submission_format.csv` for the uid list (`inference_main.py:231-232,254`). The `is_pathologic` column is cast to float and later overwritten (`:479`). Only uids/order are used.
3. Loads the config, linears and CNNs (`_load`, `:53-113`). 3D: `torch.load(weights_only=False)` of own checkpoints. 2D: `torch.jit.load` of TorchScript (`:83`). Device is CUDA if available, else CPU.
4. Loads `TabularAdapter` (frozen TabICL contexts via `pickle.load` of `lzma` files, `tabular_adapter.py:23-34`).
5. Feature-completeness probe on synthetic `np.ones` volumes (`:267-276`).
6. Loads the registration template (`:286-309`).
7. For each block of ≤128 uids (`:334`):
   a. Per scan: `pp.preprocess_full` (`:342-346`) loads the NIfTI, builds a head mask, takes the reference level, localizes the striatum, builds a canonical frame, and samples 48³ crops (`aligned`, `raw`, `aligned_f2`). A failure gives that scan the prior (`:360-365`).
   b. Block-batched rigid registration of the `aligned` crops to the template (`:374-386`, `regfit.register`).
   c. Per scan: features13 on the registered crop, plus scale-space shape and mixture features on the `aligned_f2` crop (`:387-393`) → 479-column row.
   d. TabICL probability for the block (`:396`).
   e. Linear members `lin`, `lin4` (`:398-399`).
   f. CNN members: `mil48y` (3D, 34 ckpts, tta=0 → no flip), `e2dmipf2` (B0 ×12 TorchScript), `e2db3f2` (B3 ×6) with flip-TTA averaging per image (`:438-449`, `cnn2d.predict`).
   g. Stacker: logits × coef + intercept (`:451-462`), renormalised if members are missing.
   h. Temperature T=1.4 on the stack logit (`:464-465`).
   i. Blend with TabICL: z = 0.65·logit(stack) + 0.35·logit(tab)/1.2 (`tabular_adapter.py:65-72`).
   j. Governor: a scan whose registration was rejected gets prior 0.5485 (`:470-471`). Non-finite values get the prior (`:474`).
   k. Clip to [0.005, 0.995] (`:475-477`).
8. Writes `submission.csv` in submission_format order (`:479-480`).

---

## 4. Plain-language component explanations (inference side, verified in code)

### 4.1 Preprocessing (`preprocessing3.py`)
- **Load** (`:76-87`): `nib.as_closest_canonical` to RAS, `get_zooms()[:3]` for voxel size, NaN→0. The only header items used are the affine orientation and the voxel spacing. No other header fields are read.
- **Coarse grid** (`:90-93`): resample to about 3 mm for masking and localization.
- **Head mask** (`:171-235`): Otsu threshold on log-intensity, closing, fill holes, largest component. With `DAT_MASK_V2` (default "1") a "disc guard" re-thresholds masks over 7.5 L (`:117-158,208-221`). It also measures `hole_ml` (filled voids).
- **Reference level** (`:241-284`): the mask is eroded by 10 mm (backing off to 6 mm or 3 mm if under 300 mL is left). The top 8% of voxels is removed and the reference is the mode of the 96-bin smoothed log histogram. Dividing by this reference gives SBR units.
- **Striatum localization** (`:395-498`): Gaussian smoothing at 4 mm and a search sphere of 45 mm around the head centroid. Peak 1 is the argmax with parabolic sub-voxel refinement. Peak 2 lies on the other side of the midline 22–56 mm away and must exceed 0.45× peak 1. If no second peak is found, peak 1 is mirrored. Alternative modes (`DAT_LOC_V2/V3/V4/COM`, `DAT_PAIR_V2`) are env-gated research paths and off by default.
- **Canonical frame** (`:507-552`): the L→R axis comes from the peak pair and the I→S axis from the head-mask PCA closest to +z. The "frame guard" (`frame_v2`) falls back to native +z when that axis is more than about 26° off. It is used only for the `aligned_f2` crop.
- **Crops** (`:699-796`): `aligned` (96 mm cube, 48³ at 2 mm, canonical frame, /ref), `raw` (no rotation), and `aligned_f2` (guarded frame) when `want_f2`. `meta` holds head_ml, inner_ml, hole_ml, two_blob, native_spacing, ref.
- Everything is per scan. No statistics come from other scans.

### 4.2 Registration and template_v6.npy (`register3.py`, `regfit.py`)
- 6-DOF rigid registration (rotation vector + shift). `warp` uses `affine_grid`/`grid_sample` (`register3.py:87-99`). The loss is 1−NCC per sample, summed (`:137-167`). Adam runs 120 steps at half resolution, then 80 at full.
- `regfit.register` (`regfit.py:111-117`) fits each crop against the frozen template, in chunks of 256. A fit is rejected if rotation exceeds 90°, NCC is below 0.30, or values are non-finite. A rejected fit keeps the canonical crop and triggers the prior governor.
- Independence claim: the loss is a sum of per-sample terms and Adam is elementwise, so the batch is pure vectorisation (`regfit.py:72-75`, `register3.py:141-143`). Statically this is correct because `ncc` is per-row (`register3.py:110-115`). The only coupling is floating-point/kernel order, which README:317-326 quantifies.
- **template_v6.npy**: built by `register3.main()` (`:196-259`). It reads `outputs/crops_v3_b.npz` (training cache, `aligned` key), uses log1p, starts from the plain mean, then runs 3 passes of register-to-mean and re-average (`build_template`, `:170-192`). Labels (`y`) are loaded (`:216`) but used only when passed through to the output npz (`:255`), never in building the template. The template is a 48³ float32 array in log1p(SBR) space and is frozen. REPRODUCIBILITY.md §4 claims a byte-identical rebuild. **Gap:** `crops_v3_b.npz` is built by `cache3.py` from `labels.csv` uids only, so the template is training-only, assuming the cache had no test scans (it reads `DATA/labels.csv`, `cache3.py:52-53`).

### 4.3 Hand-crafted features
- **features4** (`features4.py:73-182`): ellipsoid ROIs (striatum, caudate, putamen, anterior and posterior putamen, both sides), placed from the clipped half-separation (`roigeom`). Outputs are means and p90s, min/max/mean/asym, specific binding (SBR−1), the occipital-like background box, putamen/caudate and posterior/anterior ratios, global percentiles, volumes above thresholds, blob shape (eigen elongation, components), gradient sharpness, L/R symmetry, `sep_mm`, and passthrough QC/acquisition fields (head_ml, inner_ml, hole_ml, two_blob, native_spacing, ref).
- **features5**: adds a 3×4×2 grid per striatum, ordered worse side first (`g00..g23_*`).
- **features6**: adds per-hemisphere isocontour shape stats (`h_*`), worse side first. The `DAT_ISO_ANCHOR` default "max" is used.
- **features13**: adds radial shells around each hemisphere's binding centroid (`s_r0..r5_*`).
- **roigeom**: clips the half-sep to [9, 24] mm. ROI scaling alpha is 0 (no-op).
- **scale_features** (inference-only module; `tabular/_original/scale_features_research.py` is its source): Gaussian scale-space Hessian curvature, contrast, and threshold-support volume/mass/axes at σ 1, 2, 3 per side, reported as min/max over sides (180 features, `shape__*`). Pure per-crop.
- **mixture_features**: 2-component Gaussian mixture EM over voxel coordinates weighted by uptake, per side, at two levels → separation, balance, widths, skew, and so on (60 features, `mixture__*`). Pure per-crop.

### 4.4 Linear models (`portable3.py`)
`FeatPrep` does median impute → log1p → percentile clip (0.5, 99.5) → standardize. All constants are frozen from training and stored in JSON (`portable3.py:17-75`). `predict_linear` = sigmoid(prep(X)·coef + b). `lin`: 235 features, C 0.2, 48 non-zero (per PROVENANCE). `lin4`: 94 features, C 1.0, 40 non-zero.

### 4.5 3D CNN (`nets3.py`, `cnn3.py`)
- Member `mil48y` = `HemiMIL` (`nets3.py:94-149`): an SE-residual GroupNorm encoder (stem 24, widths 32/64/128). The 48³ crop is split at the midline and the left half is mirrored. A shared encoder scores each half, and the two are combined by noisy-OR. **GroupNorm only, no BatchNorm**, so normalisation is per sample.
- Input: log1p(SBR), standardized by a scalar mu/sd stored per checkpoint (training-fold statistics, `cnn3.py:270-271`, used at `inference_main.py:111,219`).
- The 34 checkpoints are averaged in probability space. The config sets `tta: 0`, so there is no flip TTA (`_tta_views`, `inference_main.py:122-148`).
- Run families (names): `mil48a/b` (seed bases 3000/7000), `m31000/m33000/m35000/m37000` (5-fold each), `mf21000..mf27000` (full-data). The tags differ but per the config and PROVENANCE all share arch `mil`, 48³ input and the `aligned` key from `crops_v3_b.npz`.
- Training (`cnn3.py:267-355`): fixed schedule, AdamW, cosine with warmup, EMA weights, mixup, label smoothing, and augmentation (flip, affine, blur, gain jitter, sqrt noise). OOF is the final EMA model on its held-out fold, with no early stopping. `--aux`, `--offset` and `--init` are research options; their use in shipped runs is to be confirmed from `05_cnn3d.sh`.
- `cnn4.py` (per-hemisphere crop trainer) is imported at inference only for `_blur` (TTA level 2) and `to_input_hemi`. Neither is exercised by the shipped config (no hemi members, tta 0).

### 4.6 2D CNNs (`cnn2d.py`)
- Image: three MIPs (axial, coronal, sagittal) of log1p(SBR)/log1p(12) from the `aligned_f2` crop (`cnn2d.py:55-87`). The image is upsampled to 192 px (B0) or 224 px (B3) and normalised with fixed ImageNet mean/std (`:168-176`).
- Backbones: timm `efficientnet_b0.ra_in1k` / `efficientnet_b3.ra2_in1k`, `pretrained=True` at training (`:203-215`, `:246`), with the head zeroed.
- Inference: TorchScript files. `cnn2d.predict` calls `net.eval()` and averages sigmoid over the image and its L-R flip (`:226-237`). EfficientNet contains BatchNorm. `tools/ts_export.py:22,26` traces after `net.eval()` and applies `torch.jit.freeze(traced.eval())`, so BN uses frozen running statistics (Flag F2, resolved).
- Training: AdamW, cosine schedule, EMA with warmup, mixup p=1.0, per-sample augmentation, 30 epochs, fixed schedule. The final EMA model is saved. The "kill-bar readout" at `:368-379` reads an absent `champion_v7_CV0.2139/oof_feat3.csv` (stale, no effect).
- `e2dmipf2` member = 12 checkpoints (2 seeds × 5 folds + full). `e2db3f2` = 6.

### 4.7 TabICL tabular block (`tabular_adapter.py`, `tabular_recipe.json`)
- Inputs: 68 named features (64 L1-selected plus 4 acquisition descriptors `head_ml`, `inner_ml`, `ref`, `native_spacing`; `HZ/assets/tabular_recipe.json` features[64:68]) plus 4 "prior" scores. Each prior is a linear model over 235, 94, 415 or 295 features, recomputed per row (`tabular_adapter.py:56-59`). The total is 72 columns. NaNs are imputed with the recipe median.
- Two frozen TabICLClassifier states: `raw` (all 1,362 training rows as context) and `log` (a 1,148-row acquisition-balanced subset, seed 16000, with the first 64 columns log-prepped). Each runs `n_estimators` 8. The two probabilities are averaged (`:61`).
- Frozen contexts (`tabular_raw.pkl.xz`, `tabular_log.pkl.xz`, about 230 MB each): pickled `TabICLClassifier.__getstate__` dicts, restored with `__setstate__` (`:28-33`). Per the competitor (README:369-375), they hold KV-cache row representations (8 × 512-d per context row, with the label embedding added, so labels are recoverable) and sklearn preprocessor statistics, with "raw records removed" (CANDIDATE.json). **Not verified: we did not unpickle them.** Code for what is stripped is in `tabular/freeze_tabular.py` (§5).
- `training_policy` in the recipe: "context_priors: Four linear models trained with each context case acquisition family excluded; query_priors: same four refitted on all training cases; feature_selection: weighted L1 C=.2; log: acquisition-balanced subset". So context-row prior scores are computed leave-acquisition-family-out, and query rows use full-data priors.
- Blend: `blend_weight` 0.35 and `temperature` 1.2 (`:65-72`).
- Independence: test rows act as queries against a frozen, train-only context. Whether TabICL lets query rows in the same `predict_proba` call interact (for example column-embedding set attention over all rows, or ensemble preprocessing refit on test) depends on tabicl internals inside `tabicl_lib.zip`. **Flag F3** for Code Integrity.

### 4.8 Stacking, calibration, clipping
- `inference_config.json`: members `[lin, lin4, mil48y, e2dmipf2, e2db3f2]`, coef = 0.53121 × [1, 0.4, 0.7, 0.3, 0.3], intercept 0.18431, calibration {T 1.4, clip 0.005}, governor reject_to_prior true, prior 0.54846 (the training base rate 747/1362), registration {template_v6, steps 120/80, max_deg 90, min_ncc 0.3}, `tabicl: true` (unread).
- `src/fixstack.py`: computes the fixed-ratio logit sum on OOF predictions, fits a 1-D logistic regression (C=1e6) on all OOF rows to get scale and intercept, and reports cross-fit log loss. T and the clip are CLI defaults (1.4, 0.005). The docstring says these came "from the board T-curve" (`fixstack.py:4-5`), meaning leaderboard probing. **Flag F9.**
- Stale comment: `inference_main.py:247-251` says T=1.0, but the config uses 1.4. The competitor discloses this in README:366 and PROVENANCE §4.

### 4.9 Nondeterminism
- Block batching (128) and CNN batching (32) change floating-point summation order. README:321-326 reports shuffle effects of at most 1.3e-3 and a mean of 9e-6.
- Adam registration through `grid_sample` can flip scans near the NCC 0.30 threshold between devices.
- CUDA vs CPU was never measured locally (REPRODUCIBILITY.md:54-66).
- TabICL ensemble order depends on `PYTHONHASHSEED` at freeze time (frozen in the asset).
- `argsort` ties in feature selection are platform-dependent (training only).
- Training on MPS mixes numpy and torch RNG. The claimed tensor-identical retrain holds on MPS only.

(Sections 5–7 follow.)

---

## 5. Important files (what each does)

### 5.1 `src/` (training; shared modules listed in §4 are byte-identical to `inference/`)
| File | Role | Notes |
|---|---|---|
| `folds.py` | 5-fold StratifiedKFold (seed 42) on `DATA/labels.csv` → `outputs/folds.csv` (`:20-41`) | Training labels only |
| `cache3.py` | Single pass over the **training** uids from `labels.csv` (`:52-56`). `preprocess_full` per scan → `crops_v3{tag}.npz` (aligned/raw float16 crops, refs, sep, uids, y), `features_v4/5/6{tag}.csv`, `qc_v3{tag}.csv` | `y` is stored in the npz but used only by trainers |
| `make_regfeat32.py` | Stage 02. float32 preprocessing of training scans, batched registration to the frozen template (`:96`), features13 → rich/narrow tables | Training only |
| `filter_feature_tables.py` | Restricts the tables to the champion-v7 column list (drops `hole_ml`), then drops `head_ml` | Recovered ad-hoc steps |
| `linear_oof.py` | 5-fold OOF of `lin` (C 0.2) / `lin4` (C 1.0) → `oof_feat3_m2fnh.csv` | Excludes `SCANNER` cols and `head_ml` |
| `train_final3.py` | `fit_lin` = FeatPrep + L1 liblinear (`:89-97`). `--export` fits on all training rows and writes `linear_lin*.json`, plus LightGBM/MLP side models that are **not shipped** (config `gbms: {}`, `mlps: {}`). `SCANNER = {native_spacing, inner_ml, ref, head_ml}` are withheld from lin/lin4 (`:44-61`) | Comments cite leaderboard feedback ("the board refuted it in v9", `:57-60`). Flag F9 |
| `cnn3.py` / `nets3.py` / `cnn4.py` | 3D trainer / architectures / hemi-crop trainer (see §4.5) | `cnn4` is not used by the shipped members |
| `cnn2d.py` | 2D trainer, timm `pretrained=True` (see §4.6) | Network at training time (HF) |
| `avg_member.py` | Probability mean of several runs' OOF → one member OOF, clipped | `train_all.sh:79` fast mode averages **4** runs (`mil48a mil48b m31000 m33000`) for `oof_mil48y.csv`, while the shipped member averages 34 ckpts from 10 runs. See Gap G3 |
| `avg_2d_seeds.py` | Mean of the two B0 seeds' OOF → `oof_e2dmipf2x.csv` | |
| `fixstack.py` | Stage 08 stacker (§4.8) | T/clip "from the board T-curve" |
| `build3.py` | Writes the v14c `submission_src/` and `inference_config.json`: members, coef, calibration, governor, CNN specs, `prior` = mean of training labels (`:171`), registration bounds from `regfit` | Copies `main3.py` as the v14c `main.py` (not used in v17) |
| `main3.py` | v14c inference entry point. Differs from `inference/inference_main.py` only by the TabICL/shape/mixture additions and `need_f2` (diff checked) | Stale relative to v17 |
| `package_submission.py` | Stage 09: copies `inference/` verbatim, adds the assets, `tabicl: true`, and extends `feature_order` | |
| `check_package.py` | Static gates: testzip, `main.py` at root, no `.npz`, no `logger.x(f"...")`, shallow `exec` of `scale_features.py`/`mixture_features.py`, compile | Uses `exec` on package code (verification tool; not run by us) |
| `eda_real.py` | Training-data geometry/intensity survey → `eda_real_geometry.csv` (with labels merged) and summary. Reads only shape, zooms, affine axcodes and dtype from the header (`:27-56`) | Input to `derive_groups` |

### 5.2 `tabular/` (stage 07, TabICL block; training only)
| File | Role | Notes |
|---|---|---|
| `derive_groups.py` | Builds the acquisition groups `fam2` (thin39/full39/s246/s23/s24/small, from header `pixdim3`, `nz` and `native_spacing`) × `scale` (low/high from log10 p99). Runs `src/eda_real.py` and `tools/hdr_fingerprint.py` if their outputs are missing (`:57-73,76-95`) | Training scans only. These are automatic site/scanner proxies (pseudo-site). Used only to weight and partition training rows, never as a model input at test time. Test-time acquisition descriptors are the 4 numeric features `head_ml, inner_ml, ref, native_spacing` computed per scan |
| `extract_shape_features.py` | Batch `scale_features.extract_one` over `crops_v3_f2.npz['aligned']` (imports the shipped `inference/scale_features.py`) → `scale_f2.csv` | Same code at train and inference |
| `extract_mixture_features.py` | Batch `mixture_features.extract` → `mixture_shape.csv` | Same code |
| `prepare_tabular.py` | Builds `tabular_recipe.json`, `context.npz` and `feature_order_v17.json`. (1) Candidates: 235 rich + 180 shape + 4 acquisition descriptors. (2) Sample weights shrink each acquisition group's prevalence toward the global prior (`:76-83`). (3) 4 L1 prior models. Context-row anchors are cross-fitted **leave-one-fam2-out**; the query priors are refitted on all rows and stored in the recipe (`:123-141`). (4) Weighted L1 selector, top 64 by abs(coef) (`:142`). (5) Context = 64 + 4 + 4 anchors, median-imputed. (6) `log` subset = class-proportional draw per group, `default_rng(16000)` (`:152-158`). (7) `log_prep` = FeatPrep on the 64 columns. Blend weight 0.35, T 1.2 are **hard-coded** (`:162`) | All training rows. How 0.35/1.2 were chosen is not shown in code. See Gap G4 |
| `freeze_tabular.py` | Fits `TabICLClassifier(n_estimators=8, kv_cache='repr', ...)` on the context (raw: all rows; log: balanced rows). Zeroes the 4 reserved CLS column-cache slots and sets `_save_model_weights=False, _save_kv_cache=True, _save_training_data=False`. **Asserts `ensemble_generator_.X_`/`y_` and every `preprocessors_.X_transformed_` are None** (`:91-93`). Pickles `__getstate__()` into lzma (`:94-96`), then checks the reload (<1e-6) | Static evidence that raw feature rows/labels are not stored as arrays. Labels remain recoverable from the label-embedded KV cache (competitor's own statement). The sklearn preprocessor statistics (medians, scales, bounds) are stored |
| `verify_features.py`, `verify_prepare.py`, `verify_freeze.py` | Aggregate-only comparisons against archived originals | Not run by us |
| `_original/` (6 files) | Reference copies of the original research scripts, **not imported** by any pipeline stage. `prepare_candidate.py` / `freeze_candidate.py` are the originals of `prepare_tabular.py` / `freeze_tabular.py`: same numerical logic, plus in-place staging/patching of `inference_main.py` and `tabular_adapter.py` (`prepare_candidate.py:47-58`) and platform-version overlay dirs (`freeze_candidate.py:6`). `mixture_shape.py` holds the source of `inference/mixture_features.py`; the slice `AX=`..`def main` matches apart from a trailing blank line (diff checked). `scale_features_research.py` = `inference/scale_features.py` plus a CLI block. `common_20260914.py` loads data and computes per-group ("centres") metrics on training OOF. `bench_v16.py` is a CatBoost/ExtraTrees/HistGB benchmark (not shipped; imports CatBoost) | **No material difference** from the cleaned versions in feature, prior, selection or freeze logic. The original `prepare_candidate.py:57` status text read "HELD ... not approved for public submission" (stale) |

**TabICL independence (static, partial):** with a KV cache, `predict_proba` transforms test rows with the encoders fitted on training data (`ensemble_generator_.transform(X, mode="test")`) and runs `forward_with_cache` against the cached training representations (`tabicl/_sklearn/classifier.py` in `tabicl_lib.zip`, around lines 623-740). The column embedder documents that "inducing points will only attend to training data (positions < train_size)" (`tabicl/_model/embedding.py:341-375`). The `predict_proba` docstring notes that columns entirely NaN across the batch are masked. That would be a cross-row effect, but `tabular_adapter.py:60` median-imputes every NaN before the call, so it cannot trigger. Code Integrity should confirm `forward_with_cache` / `_compute_embeddings_with_cache(use_cache=True)` lets no test row attend to another (Flag F3).

### 5.3 `tools/`
| File | Role | Notes |
|---|---|---|
| `hdr_fingerprint.py` | Header-only survey of the **training** NIfTIs → `outputs/hdr_fingerprint.csv`. Reads the free-text fields `descrip`, `db_name`, `aux_file`, `intent_name`, plus qform/sform codes, cal/scl values, qoffsets, pixdim, quaternions, affine and dim (`:11-28`). Prints value counts of every field with ≤40 unique values (`:33-39`) | **Privacy check (F7).** The shipped log `docs/repro/.../stage_logs/hdr_fingerprint.log` shows all four free-text fields empty (`''` for 1,362/1,362) and nothing identifying. Only `pixdim1`, `pixdim3` and `aff_offdiag_max` are consumed downstream (`derive_groups.py:78`), and only `pixdim3` reaches the `fam2` rule. Not used at inference |
| `ts_export.py` | Loads each raw 2D state dict, calls `net.eval()`, `torch.jit.trace` + `torch.jit.freeze(traced.eval())` (`:21-29`). The trace sample is 64 training crops from `crops_v3_f2.npz` | Freezing in eval mode folds BatchNorm running stats into the graph, so the 2D members cannot use batch statistics at inference (resolves F2 statically) |
| `fetch_backbones.py` | Downloads the timm B0/B3 ImageNet weights via `timm.create_model(pretrained=True)` and checks the pinned sha256 (`:34-81`) | Network, training only |
| `scrub_torchscript_debug.py` | Rewrites `.ts.pt` zips without `*.debug_pkl` and checks the outputs are equal | Explains the 18 hash differences between `SHA256SUMS.public` and `SHA256SUMS.scored` |
| `replay_cnn_oof.py` | MODE=fast. Each released fold checkpoint predicts its held-out training fold → `oof_<tag>.csv` | Training data only |
| `verify_cnn.py` | Aggregate-only fidelity checks (caches, spot retrains, TS export) | Not run |

### 5.4 `scripts/`
| File | Role | Notes |
|---|---|---|
| `install_weights.py` | Extracts the 60 assets from a zip and verifies them against `models/SHA256SUMS.*`. `--url` downloads with `urllib.request.urlretrieve` (`:106-116`) | Network, user-initiated, competitor's own Dropbox bundle. Not part of the harness |
| `predict.sh` | Runs `inference/main.py` with `DATA_DIR`, `HF_HUB_OFFLINE=1`, in a temp dir (`:33-38`) | |
| `predict_folder.py` | Builds a competition-style layout (symlinks plus a synthetic `submission_format.csv`) for any folder, then calls `predict.sh` | |

### 5.5 `pipeline/`
See the §3 table. Points of detail:
- `04_caches.sh` builds `_b` with `DAT_MASK_V2=0` (no disc guard) and `_f2` with `DAT_MASK_V2=1 DAT_FRAME_V2=1`. At inference the 3D member is served disc-guarded crops (the mask-guard default is on), a train/serve mismatch on 29/1,362 training crops that the competitor discloses (REPRODUCIBILITY.md:218-221).
- `05_cnn3d.sh`: every run uses `--arch mil --size 48 --key aligned --crops crops_v3_b.npz --lr 1e-3 --epochs 20`. Seeds are given per family. The member OOF averages 4 runs; the shipped member averages 10 runs / 34 ckpts (disclosed in the header comment).
- `06_cnn2d.sh`: `--crops crops_v3_f2.npz --key aligned --views mip3 --epochs 30 --lr 1e-4 --aug 1.0`. B0 at 192 px with seed bases 9000/9100; B3 at 224 px with seed base 9000. Full-data seeds 9099/9199. Calls `tools/fetch_backbones.py` first (network).
- `07_tabular.sh`: derive_groups → shape/mixture extraction → prepare_tabular → freeze_tabular with `PYTHONHASHSEED=1`.
- `08_stack.sh`: `fixstack.py --ratio 1.0 0.4 0.7 0.3 0.3` (T and clip are defaults), then `build3.py --use-full --tta 0 --template template_v6.npy --key2d ...=aligned_f2`.
- `09_package.sh` → `src/package_submission.py`.
- `smoke_cnn.sh`: 1-epoch runs under `smoke_` tags.

### 5.6 `configs/`
`configs/feature_columns/features_v13_reg7f.header.csv` and `features_v4_reg7f.header.csv` hold one header line each. They give the column order and column list of the earlier "champion v7" tables that `filter_feature_tables.py` restricts to. They carry no data.

### 5.7 `models/`, `docs/`, root files
- `models/SHA256SUMS.scored.txt` (60 lines, matches the harness zip per Coordinator), `SHA256SUMS.public.txt` (18 TorchScript lines differ), `models/README.md` (bundles, label-recoverability statement).
- `docs/PROVENANCE.md` maps each archive entry to the script that produces it, with verification classes. Notes stale text (CANDIDATE.json status, the T comment) and leftovers (`tabicl: true` unread; the cnn2d readout).
- `docs/REPRODUCIBILITY.md` gives reproduction levels, nondeterminism, untested cases (no CUDA/CPU training, no Linux run except the platform), and the block-batching effect sizes.
- `docs/stages/*.md` gives per-stage commands and verification. `08-09_stack_and_package.md:23-24` says ratios, T=1.4 and the rejection bounds were "chosen by hand during development".
- `docs/repro/mode_full_2026-09-28/` holds logs of the MODE=full rerun (paths placeholdered). Grep found: no fallback warnings in the parity runs, the "readout skipped" FileNotFoundError lines (the stale cnn2d readout), and HF Hub "unauthenticated requests" warnings during 2D training (training-time network contact with the HF Hub for public backbone metadata/weights). No test UIDs (grep against all 524 `submission_format` uids: 0 hits in the tree or the extracted harness). No training UIDs either (0 hits for 1,362 uids).
- `docs/figures/make_figures.py` (code read only; images **not viewed**). Every number is a typed constant: `SCORES` (`:93-111`) = public/private log loss of 9 platform submissions from the "development notes"; `FAMILIES` (`:127-140`) = training-OOF aggregates per inferred acquisition centre; plus `linear_lin.json` coefficients. It imports only argparse/json/re/sys/pathlib/matplotlib and reads no scan or test file. `drift.png` = private − public per submission (`fig_drift`, `:685-710`). `score_progression.png` = the public/private dots per submission (`:309-357`). Neither uses test labels or per-scan data, but both use **leaderboard (including private) scores**. The private scores could only be known after the competition closed (presumed post-hoc). Organizer to inspect the images if needed. `family_loss.png` shows OOF metrics per inferred site (training only).
- `README.md`, `THIRD_PARTY_NOTICES.md`, `LICENSE` (MIT, Navid Shabani Barzegar), `requirements-*.txt`.

---

## 6. Claims vs. what the code does

| Claim (source) | Code | Assessment |
|---|---|---|
| Each scan is processed on its own; registration is a sum of per-scan terms; normalisation is frozen (README:317-326, regfit.py:37-39) | `register3.ncc` is per-row; Adam is elementwise; FeatPrep/CNN mu/sd/TabICL contexts are frozen; 3D nets use GroupNorm; 2D nets are frozen in eval mode | **Consistent.** The residual coupling is floating-point batch order and block-level fallbacks (disclosed) |
| "No test-scan value, identifier, count or path reaches the log" (README:357-359) | All `logger.*` calls are static (`inference_main.py:230-481`). The fallback warnings fire once per run (`:361-364,382-385,446-449`). `main.py` prints versions and exception type/location | Consistent. The `RuntimeError` message at `:275-276` includes counts of missing **feature names** (not test data), and it is raised only on a code/asset mismatch |
| Inference makes no network calls (README:286-287) | `tabular_adapter.py:5-6` sets `HF_HUB_OFFLINE`. TorchScript avoids timm. `allow_auto_download=False` at freeze. No urllib/requests in `inference/` | Consistent (static). tabicl internals were not fully audited (Code Integrity) |
| T = 1.4; the comment saying T = 1.0 is stale (README:366-367, PROVENANCE §4) | `inference_main.py:247-252` vs config `T: 1.4` | Consistent, disclosed |
| Ratios and T chosen "by hand" (stages/08-09:23-24) | `fixstack.py:4-5` says T and clip came "from the board T-curve". `train_final3.py:57-60` says "the board refuted it in v9". The submission history (figures README:22-32) shows the T=1.2 → T=1.4 versions submitted in sequence | **Inconsistent wording.** The code comments indicate public-leaderboard feedback guided T, the clip and the `head_ml` exclusion (F9) |
| Contexts hold no per-scan feature rows; labels recoverable (README:369-375) | `freeze_tabular.py:91-93` asserts training X/y and transformed X are not saved; KV cache only | Consistent with code. Not verified on the artifact (not unpickled) |
| Template is a label-free average of training crops (README:252-256) | `register3.build_template` uses no `y` | Consistent. Built from `crops_v3_b.npz` = training uids |
| mil48y averages 34 checkpoints | config `cnns[0].sources` has 34 entries, `tta: 0` | Consistent. The stacker weight was fit on a 4-run OOF (disclosed) |
| Training ImageNet weights are Apache-2.0; "never downloaded ImageNet images" (THIRD_PARTY:61) | `cnn2d.build(pretrained=True)` | Consistent; raises Q1 |
| `inference/` is byte-identical to the scored archive | Coordinator verified | Consistent |
| `tabular/_original/` is not imported | No import found; `prepare_tabular.py`/`freeze_tabular.py` are ports with the same logic | Consistent |

**Stale or unused code paths (no effect on the scored predictions):**
- `gbms`/`mlps` and `offset_member` (booster) paths.
- `hemi` input paths, `crops2` (foveated), `aligned_reg1`.
- TTA level 2 (`cnn4._blur`).
- MedicalNet archs (`nets3.py:225-308`).
- `DAT_*` env research switches in `preprocessing3`, `features6`, `roigeom`.
- `cnn2d.py:368-379` readout of the absent `champion_v7` OOF.
- `tabicl: true` (unread).
- `src/main3.py` (v14c entry).
- `CANDIDATE.json` status text.
- The `tabular_adapter` non-frozen path (`:35-46`, would need `tabular_context.npz`, not shipped).

---

## 7. Flags for downstream reviewers

| ID | Severity (prelim.) | Owner | Flag | Evidence |
|---|---|---|---|---|
| F1 | low | Code Integrity | Block processing (128) with block-level fallbacks. If registration raises, or a CNN raises, the whole block degrades together, so one scan's failure changes its block-mates' predictions. No such warning appeared in the competitor's local runs; the platform log for 322397 was not kept | `inference_main.py:334,378-386,438-462`; README:328-332; REPRODUCIBILITY.md:54-66 |
| F2 | resolved (info) | Code Integrity | BatchNorm in EfficientNet: frozen in eval mode at export, so no batch statistics | `tools/ts_export.py:22,26`; `cnn2d.py:228` |
| F3 | low–medium (unverified) | Code Integrity | TabICL `predict_proba` is called on a block of up to 128 test rows. Statically, the KV-cache path transforms test rows with train-fitted encoders and the column-embedder inducing points attend only to training positions. Confirm in `tabicl_lib.zip` (`tabicl/_model/tabicl.py` forward_with_cache, `embedding.py` `_compute_embeddings_with_cache`, `interaction.py`, `learning.py`) that no test row attends to another test row. The all-NaN column masking cannot trigger because of median imputation | `tabular_adapter.py:53-63`; `classifier.py` ~623-740; `embedding.py:341-375` |
| F4 | info | Code Integrity | `submission_format.csv` is read only for uids/order; the values are overwritten | `inference_main.py:231-232,479` |
| F5 | info | Code Integrity | Governor: rejected registration (NCC < 0.30 or rotation > 90°) → prior 0.5485 (training base rate, `build3.py:171`). Non-finite → prior. Clip [0.005, 0.995]. Matches the 115/4 boundary counts in the scored CSV | `inference_main.py:470-477` |
| F6 | info | Code Integrity | No test-time TTA across cases. 2D flip TTA is within-image; 3D tta=0 | config; `cnn2d.py:226-237` |
| F7 | low | Data Source (privacy) | `tools/hdr_fingerprint.py` reads free-text header fields (`descrip`, `db_name`, `aux_file`, `intent_name`) and fingerprints acquisition/site. The fields are empty in all training scans (shipped log). The derived site-proxy groups (`fam2`/`scale`) are used for training-set weighting, leave-family-out priors and the balanced context. At test time per-scan acquisition descriptors (`head_ml`, `inner_ml`, `ref`, `native_spacing`) are TabICL inputs, computed per scan. The organizer ruling says no extra inference restriction. No linking of scans to individuals found | `tools/hdr_fingerprint.py:11-39`; `derive_groups.py:76-95`; recipe features[64:68] |
| F8 | low | Data Source / Code Integrity | Network use is training/setup only: HF Hub for public timm weights (`fetch_backbones.py`, `cnn2d.build(pretrained=True)`; the HF warnings in the repro logs confirm contact), Dropbox via `install_weights.py --url`. No upload of competition data found anywhere | grep results §5.7 |
| F9 | medium (organizer judgment) | Code Integrity / organizer | Public-leaderboard feedback was used to choose the temperature (1.2 → 1.4), the clip and feature inclusion (`head_ml`). This is not test-label leakage per se, and it is common practice, but it conflicts with the "chosen by hand" doc wording and touches the GR F "do not overfit to the test set" spirit | `fixstack.py:4-5`; `train_final3.py:57-60`; `cnn3.py:11-12`; figures README:22-32 |
| F10 | medium (policy) | Data Source / Eligibility | ImageNet-1k pretraining of the 2D backbones. The competitor notes the ImageNet terms are non-commercial (THIRD_PARTY:61). The rules file says the pretraining data of pretrained models is subject to the same rules. Organizer ruling needed (Q1) | THIRD_PARTY_NOTICES.md:48-61 |
| F11 | high (release) | Release Readiness | Public Dropbox links to comp-trained weights. The frozen TabICL contexts encode recoverable training labels plus per-feature statistics. The organizer ruling allows only private, organizer-only hosting | README:131,142,153,369-375; models/README.md:25-31; THIRD_PARTY:175 |
| F12 | info | Data Source (AI tools, GR A) | No disclosure of AI coding tools; no mentions of ChatGPT/Codex/Claude/Copilot found. One comment cites an "external review member E" (`cnn2d.py:72`); provenance unclear. Ask the competitor if needed | grep |
| F13 | info | Data Source | No external datasets. No PPMI/LONI/DaTscan cohort references. MedicalNet code is present but unused (no `external/` dir; shipped arch `mil`) | grep; `nets3.py:299-323` |
| F14 | low | Code Integrity | `check_package.py:92-94` `exec`s package modules (a verification tool, not inference). We did not run it | |

---

## 8. Assumptions and gaps

- G1: No competitor code was run and no artifact was loaded. The contents of the `.pt`/`.ts.pt`/`.pkl.xz`/`.ckpt`/`template_v6.npy` files are taken from code and competitor docs. In particular, whether the pickled TabICL states truly lack training rows is inferred from the `freeze_tabular.py` assertions only.
- G2: The tabicl 2.2.0 internals (vendored zip) were only partly read (F3). The competitor claims the zip is unmodified versus the PyPI wheel RECORD. Not verified here (it would need the wheel; no web fetch done).
- G3: The stacker weight of `mil48y` was fit on a 4-run OOF, while the shipped member averages 10 runs. The weights of the full-data models and of `e2dmipf2`'s `full0` are not reflected in any OOF. This is disclosed and a modelling choice, not a rule issue.
- G4: The provenance of the hard-coded values `blend_weight=0.35`, TabICL `temperature=1.2`, the member ratios and the `n_estimators=8` is not in code (development notes are not distributed). Possibly LB-informed (F9).
- G5: `template_v6.npy` was built on 2026-07-25 from a cache that no longer exists. The competitor claims a byte-identical rebuild from the training-only cache. We trust that the cache contained only the 1,362 training uids (cache3 reads `labels.csv`).
- G6: Whether every CNN member actually ran (vs the silent block-level fallback) in the scored platform run cannot be checked. There is no platform log, and the CSV alone does not reveal it. Coordinator recomputed the private score 0.2731 from the CSV, which matches.
- G7: Images in `docs/figures/` were not viewed (organizer decision). Their content is inferred from `make_figures.py` constants.
- G8: The organizer ruling is pending on ImageNet pretraining (Q1), on LB-informed tuning (F9) and on whether automatic acquisition grouping counts as an "annotation" (Q3).
