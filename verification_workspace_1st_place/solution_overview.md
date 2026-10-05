# Solution Overview — 1st place (team TheAvengers: `dvtle`, `SamuelAnsel`)

Author: Reviewer 1 (Solution Explanation). Static, read-only review. No competitor code executed, imported, or loaded; no artifacts unpickled/torch-loaded; `weights.zip` not extracted; no images viewed.

Path conventions: `TREE/` = `1st Place/`; `HZ/` = `verification_workspace_1st_place/extracted/harness_zip/` (root members of the scored harness zip); `SZ/` = `verification_workspace_1st_place/extracted/src_zip/src/` (text members of the harness's `src.zip`). Line numbers refer to those copies.

---

## 1. Competition-Specific Checks

Sources: `references/verification_comp_specific_rules.md` (RF) and `notes/governing_rules.md` rules A–F (GR). Each check has one owner.

| ID | Check | Source | Owner |
|---|---|---|---|
| CS1 | Each test sample processed independently: no batch-level normalization, test-set statistics, cross-case calibration, clustering, pooled TTA, transductive embedding (e.g. t-SNE fit on test), pseudo-labelling or unsupervised learning on test data. | RF "Other checks" / GR-D | Code Integrity |
| CS2 | Harness zip has `main.py` at root, writes `submission.csv`, reads `/code_execution/data/niftis/` + `submission_format.csv`, offline, ≤3 h on 1×A100, bundles all assets. Compare harness zip vs source tree vs README. | RF | Code Integrity |
| CS3 | Predictions are probabilities in [0,1]; scored CSV format, UID coverage vs `submission_format.csv`, no hand-editing signs (log-loss metric). | RF | Code Integrity |
| CS4 | Submission does not print/log information about the test dataset. | GR-C | Code Integrity |
| CS5 | No network calls in inference; no uploads of competition data to hosted services/third-party APIs in training, preprocessing or annotation tooling. | RF "Third-party APIs" | Code Integrity |
| CS6 | No code reading/deriving patient-identifying NIfTI header fields or linking scans across UIDs to individuals. | RF "Privacy" | Code Integrity |
| CS7 | Every external dataset, annotation set and pretrained model disclosed to organizers; cross-check README/docs vs what code loads. | RF | Data Source |
| CS8 | External data must be freely and publicly available to all participants, under a permissive license not prohibiting commercial use (CC0/CC-BY); no NC licenses; competitor holds all rights; DUA-gated data OK only if DUA says data is open. | RF / GR-B | Data Source |
| CS9 | Pretraining data of every pretrained model subject to the same rules (no forbidden or NC-licensed pretraining data). Cross-winner pending ruling on ImageNet-family pretraining applies. | RF | Data Source |
| CS10 | No public/semi-public copy of competition data used (organizer: none exists); any match ⇒ leakage. | RF | Data Source |
| CS11 | Competitor annotations only on training scans, delivered with the solution, not tuned on test/leaderboard; no manual annotations of test data; solution runs automatically on test data as provided. | RF / GR-F | Data Source |
| CS12 | AI tools: no use of tools like Codex/ChatGPT that store/retain uploaded competition data; local model use OK. Note any disclosure/evidence. | GR-A | Data Source |
| CS13 | Winner not sponsor employee/annotation panel/contributing center — **confirmed by DrivenData**; age/territory (Russia) eligibility — organizer records. | RF | Eligibility |
| CS14 | Submission cap / multiple accounts / private sharing — **checked by DrivenData, no issues**. | RF | Eligibility |
| CS15 | All software required is open source permitting commercial use; reproducible with free OSS only (incl. vendored libs). | RF | Eligibility / Release Readiness |
| CS16 | Solution licensed under MIT; README license text must not conflict. | RF | Release Readiness |
| CS17 | Winners deliver code + docs, Winning Model Documentation, signed documents. | RF | Release Readiness |
| CS18 | Release = code only at announcement; weights trained on competition data may only be held privately (organizer-only access) before dataset publication. Check weight-download hosts and any bundled label-bearing files. | RF | Release Readiness |
| CS19 | Data used solely for the purpose/duration of the competition: check post-competition retention / bundling of competition-derived data (e.g. OOF CSVs with training uids+labels in harness zip, public weight hosts). | GR-E | Release Readiness |
| CS20 | Reviewers: no image viewing; record path/size/hash only. | RF "Verification handling" | All (process) |

## 2. Dataset Policy

**Forbidden**
- Any external data not freely and publicly available to all participants (incl. private clinical/institutional archives the competitor accesses via employment/clinical practice).
- Any data or pretrained model under an NC license (NC, CC NC, CC BY-NC), and any pretrained model whose pretraining data is forbidden/NC.
- Data the competitor does not hold all rights to use without third-party approval.
- Any public copy of competition data (organizer: none exists).
- Manual annotations of test data; pseudo-labelling / unsupervised learning on test data.

**Allowed**
- Competition training scans `data/niftis/{uid}.nii.gz` + `train_labels.csv`.
- External data/models that are freely public **and** permissively licensed for commercial use (CC0/CC-BY-like), disclosed to organizers.
- DUA-gated data only if the DUA says the data is open (PPMI aliases `ppmi`, `loni`, `ida.loni.usc.edu`).
- Competitor annotations of training data (included with solution). No organizer-approved additional datasets.

**External sources used by this solution (from README + code; status for Data Source reviewer)**
| Source | Where used | Enters final scored model? |
|---|---|---|
| Synthetic FP-CIT DaT PET (Won et al.; Zenodo 15377683; GitHub Krying/PD_SSL_ZOO → OneDrive "Synthetic FP-CIT-PET Samples") | Block 1 soft-labelled, then training data for Block 2 Siamese | **Yes** (as training data for the 100 Siamese models) |
| MedicalNet pretrained 3D ResNets (Tencent) | Block 1 teachers only | Indirectly only (via soft labels) |
| HF `edadaltocg/resnet50_cifar100` | Block 1 ResNet2D teacher | Indirectly only |
| HF `edadaltocg/vit_base_patch16_224_in21k_ft_cifar100` (ImageNet-21k → CIFAR-100) | Block 1 ViT2D teacher | Indirectly only |
| MedNeXt (MIC-DKFZ) / 3D-ResNets-PyTorch / MONAI DenseNet | architecture code, Block 1 | Indirectly only |

**Unresolved dataset-policy questions**
- Q1: License of the Zenodo 15377683 synthetic PET record / OneDrive copy (CC-BY? CC-BY-NC?) and whether OneDrive download is "freely and publicly available".
- Q2: Training data of the generative model behind the synthetic PET (e.g. PPMI or a private Korean hospital cohort). If the generator was trained on non-open data, are synthetic samples derived from it allowed?
- Q3: License + pretraining data of MedicalNet weights (MedicalNet repo license; "3DSeg-8" pretraining datasets' licenses) and of the two `edadaltocg` HF checkpoints (CIFAR-100 / ImageNet-21k / ImageNet-1k behind them). Falls under the pending cross-winner ImageNet ruling.
- Q4: Does knowledge distilled from Block-1 teachers into soft labels count as "using" those pretrained models in the final solution? (Factually: yes, the final model's training targets for synthetic data depend on them.)

## 3. Ordered Pipeline Summary

Orchestrator: `TREE/5_pipeline/run_pipeline.py` (phases L11-18; config `TREE/0_configs/default.yaml`).

**Block 1 — teachers + synthetic soft-labelling (real competition training data only)**
1. `phase1_prep`: `1_initial_training/prepare_folds.py` — reads `train_labels_JNDlMjr.csv`, StratifiedKFold(5, seed 42) → `train_labels_with_folds.csv` (L12-24). `prepare_2d_dataset.py` — 2D slices from training niftis.
2. `phase1_train`: 12 teachers from `stacking.active_models` (`default.yaml` L106-118) via `MODEL_TO_CMD` (`run_pipeline.py` L37-50): ResNet2D, ViT2D, MedicalNet 18/34/50/200, MedicalNet-ResNeXt 50/101, MedNeXt-B, DenseNet3D-121/201, DenseNet121+attention; plus pyradiomics-feature SVM (`train_svm.py`). All read `cfg.paths.train_labels` / `cfg.paths.niftis` (training only). Each writes 5-fold OOF CSVs.
3. `phase2`: `2_stacking/stacking.py` — merges teacher OOFs on training uids (L376-408), per-model calibration, Meta-A ElasticNet LR + Meta-B Caruana bagged ensemble selection, blend alpha, final calibrator; nested CV via `PredefinedSplit` on training folds (L412). Saves `meta_model_stacking.pkl` with keys at L657-669 — **no `post_processing` key**, so t-SNE/clipping branches of `StackingInference` stay off.
4. `phase3`: `3_soft_labeling/soft_labeling.py` — globs **only** `cfg.paths.synthetic_niftis` = `<root>/external/PET_GENERATED/*.nii.gz` (L24, L217); runs teachers + `StackingInference.predict` (no `post_process`) per synthetic file (L151-206); writes `synthetic_soft_labels.csv` with continuous `is_pathologic` (L203, L254). Synthetic volumes are **not** y-flipped here (README L216-219 "upside-down" bug kept on purpose).

**Block 2 — final Siamese**
5. `phase4_prep`: `4_siamese_training/prepare_2d_dataset_siamese.py` + `prepare_2d_dataset_striatum.py`, `--mode real` (training labels, `INVERT_Y=False`) and `--mode synthetic` (soft-label CSV, `INVERT_Y=True`) (L80-89). Both call the same `src.preprocessing.prepare_all_inputs` used at inference; save per-uid `.pt` patches (L/R 12-channel stacks; global 48-px patch).
6. `phase4_train`: `4_siamese_training/train_siamese.py` — 20 exps × 5 folds = 100 subprocess workers (L76-77, L417-476). Real training uids sorted, `np.roll` by `exp*len//20`, split into 5 chunks (L220-239); **all** soft-labelled synthetic samples appended to every fold's training set (L246-262); BCEWithLogits on soft targets, aux-head loss weight 0.3 (L298, L326-328); AdamW, cosine warm restarts, 150 epochs, patience 60; checkpoint on best real-validation-fold log loss (L379-387) → `mashup_ensemble_exp_{exp}_fold_{fold}.pth` (L306). `USE_DWT=False`, `USE_TTA=False`, `USE_SYNCHRO_FLIP=False`, `USE_RELABEL=False` (L94-99).

**Packing — `6_runtime/pack_production.py`**
7. Deletes old artifacts; copies `TREE/src/` → `submission_src/src` (L64-66); copies **every** `*.pth` under `cfg.paths.phase4_out` (rglob) → `submission_src/weights/` (L71-76); copies `meta_model_stacking.pkl` if present (L80-84); `uvx rpzip` → `src.zip`, `weights.zip` (L90-91); then `just pack-submission` which zips **everything in `submission_src/*`** (`runtime/justfile` L179, L188) → this is why the harness contains the 8 helper/diagnostic scripts, two CSVs and `__pycache__` that sat in the team's `submission_src/`. Then `just check-submission`, re-extracts, `just test-submission` (Docker, smoke test).
   - Differences vs scored harness: the scored `src.zip` also contains `src/libs/` (vendored openTSNE, ptwt, pykwalify, pywt, radiomics, ruamel.yaml, xgboost) that `TREE/src/` does not have, and lacks `TREE/src/models/`; the scored `weights.zip` has no `meta_model_stacking.pkl` (only the 100 `.pth`). So the scored zip was produced from a different working copy than the delivered tree.

**Inference — scored harness (`HZ/main.py`)**
8. `HZ/main.py` L63-69 unzips `src.zip` and `weights.zip` next to main.py; L74 prepends `./src/libs` to `sys.path`; L79-147 imports all inference modules (no import-time artifact loads or network calls found).
9. Config: `MODE_INFERENCE="FAMILY_STACKING"` (L31), `FAMILY_MODELS=["Mashup_Internal"]` (L37-42; DWT/MiniCNN3D/XGBoost commented), `TEMP=1.094` (L43). `DATA_DIR=/data2/iceberg/...` does not exist in the container → falls back to `Path("data")` (L20-22) → `/code_execution/data/niftis`, `data/submission_format.csv`.
10. `init_pipeline` (L228-389): `LogitFamilyStackingInference(model_names=["Mashup_Internal"], temperature=1.094)` (L233; class `SZ/inference_feats.py` L413-465, default clip (0.01, 0.99)). Loop over `["Mashup_Internal"]` → `SiameseMashupInternalEnsemble(WEIGHTS_DIR, n_exp=20, n_folds=5, use_dwt=False)` (L286-289) which loads `weights/mashup_ensemble_exp_{0..19}_fold_{0..4}.pth` into `build_model("siamese_internal")` (`SZ/inference_2d_avg.py` L861-884; `SZ/builder.py` L886-937, L1062-1065). `constants.prefix_map` entry `"TwoPhase_Mean": "twophase"` is **inert**: `prefix_map` is only read for RNN/corax/3D branches, and `TwoPhase_Mean` is not in `FAMILY_MODELS`.
11. Per test uid, in `submission_format` order (L432-441): `prepare_all_inputs(nifti)` (`SZ/preprocessing.py` L662-741) — single-volume only; `predict_for_model` routes `SiameseMashupInternalEnsemble` → `inputs["siamese_mashup_data"]` (L166-167, L216-222).
12. `SiameseMashupInternalEnsemble.predict` (`SZ/inference_2d_avg.py` L942-957): one forward per model; `outputs[0]` = mean of 3 branch logits; `sigmoid` → 100 probabilities → `_apply_recipe` = plain `np.mean` (L923-938; older temperature/clip recipe commented out).
13. `LogitFamilyStackingInference.predict` (`SZ/inference_feats.py` L426-458): clip p to [1e-7, 1-1e-7], logit, mean over 1 family (= identity), `expit(z/1.094)`, `clip(0.01, 0.99)`. `post_process` returns unchanged (L460-465). Effective formula: **p = clip(σ(logit(mean₁₀₀ pᵢ)/1.094), 0.01, 0.99)**.
14. `submission.csv` written (L443-444). If a nifti is missing → 0.5 (L435-439); none in scored CSV.

**Can the tree reproduce the harness?** For the executed path, yes in code: `SiameseMashupInternalEnsemble` + `MashupInternalEnsembleCNN` + `preprocessing.py` + `inference_feats.LogitFamilyStackingInference` are byte-identical between `TREE/src/` and `SZ/` (diffs only in `builder.py` mednext parsing, `inference_feats.TabularInference` pkl suffix, `inference_2d_avg` ResNet2D/ViT2D classes — none executed). Tree `main.py` has identical MODE/FAMILY/TEMP. Two caveats: (a) tree `main.py` L20 `DATA_DIR = Path()` always exists, so it would look for `./submission_format.csv` / `./niftis` and, in the container, exit without writing `submission.csv` (README L170 claims `data/niftis`); (b) tree `src/` lacks `libs/`, so imports of `xgboost`/`ptwt`/`radiomics` rely on the runtime image (tree `runtime/pyproject.toml` L58-64 lists them). The 100 weights would come from `train_siamese.py` (filenames and architecture match) or from the HF repo via `scripts/download_weights.py`.

## 4. Components

**Preprocessing (per case, `SZ/preprocessing.py`)**
- `preprocess_patient_numpy` (L190-219): nibabel load, resample to 2 mm (`ndimage.zoom`), 128³ crop centred on centre-of-mass of voxels > own 95th percentile, per-volume min-max scaling. All statistics are from the single volume.
- `extract_siamese_patches` (L302-470): Gaussian centre weighting, own 98th-percentile mask, `peak_local_max` to find 2 striatal peaks, rotate volume (≤±30°) to align them, SBR volume using per-volume background median; 32³ cubes around L and R striata and 48³ around midpoint; right side mirrored (`canonicalize_right`). Patch = [axial MIP, coronal MIP, 5 axial, 5 coronal] (12 ch).
- Siamese input (`inference_2d_avg.py` L893-921): channel 0 (axial MIP) only, expanded to 3 ch; global 48×48 + its flip on dim 1; L/R with medial linear fade (cols 22-31); L/R 24-px centre crop; ImageNet mean/std normalisation constants (fixed numbers, no ImageNet weights).
- Also computed per case but unused by the scored model: MONAI path + 2D slices, pyradiomics features (`extract_features` L472-589), etc.

**Architecture (final)**: `MashupInternalEnsembleCNN` (`SZ/builder.py` L895-934 ≡ `TREE/4_siamese_training/train_siamese.py` L171-215). Three independent 4-conv-layer encoders (32-64-128-256, BN, ReLU, GAP) for global/fade/tight views; each view pair fused as [a, b, |a−b|, a·b] → Linear(1024→256) → LayerNorm/SiLU/Dropout/Linear(256→1); output = mean of 3 logits. ≈1.96 M parameters (computed from layer shapes) ≈ README's "1.9M". **Randomly initialised — no pretrained weights.**

**Training**: see §3 step 6. Soft labels are continuous targets in BCE. Seeds `42+exp*10+fold`, cudnn deterministic (L42-50, L279).

**Soft-labelling**: §3 step 4. Labelled set = synthetic PET files only. Teachers + meta-model are trained on the 1,362 real training scans only.

**Pretrained weights — where they enter**
- `edadaltocg/resnet50_cifar100` via `hf_hub_download` in `TREE/1_initial_training/train_2d_resnet.py` L128-136 (strict load, CIFAR stem) — Block-1 ResNet2D only.
- `edadaltocg/vit_base_patch16_224_in21k_ft_cifar100` via `hf_hub_download` in `train_2d_vit.py` L96-113 (timm `vit_base_patch16_224`, `pretrained=False`, then manual load) — Block-1 ViT2D only.
- MedicalNet: loader exists (`TREE/src/models/medicalnet/model.py` L86-114) but only if `--pretrain_path` is given; `setting.py` L46-47 default `None`, and `run_pipeline.py` L43-49 never passes it → as delivered, MedicalNet teachers train **from scratch**. Whether the original run used Tencent weights is not determinable (harness `constants.py` L2 comment about "-ft" MedicalNet names hints at fine-tuned variants in earlier experiments).
- timm/torchvision: all inference classes use `weights=None` / `pretrained=False` (e.g. `SZ/inference_2d_rnn.py` L171-176 passes False). MONAI DenseNet121 built from scratch (`train_3d_densenet121.py` L186).
- **None enter the final scored Siamese model**; they influence it only through the synthetic soft labels.

**Ensembling**: 100 checkpoints (20 fold-shuffles × 5 folds), probability mean, then temperature 1.094 in logit space. Block-1 stacking is not used at inference.

**Post-processing**: T=1.094 then clip [0.01, 0.99] (`inference_feats.py` L414, L453-456). Scored CSV: 109 rows exactly 0.99, min 0.01256 (lower clip never hit) — consistent. Harness `validate_pipeline.py` L121-125 and L698-700 state that `EXPECTED_T = 1.094` **"contient le facteur 1.245 de décalage leaderboard"** ("factor 1.245 = offset correction measured on the leaderboard") and is "deliberately sub-optimal on the OOF". The temperature was therefore tuned partly on public-leaderboard feedback (a single global scalar). The notebook that derived T and the clip is not delivered.

**TTA**: none in the executed path (one forward per model). The G/G_flip pair is an architectural input, not TTA. TTA wrappers (`inference_tta_wrappers.py`) are imported but unused; `USE_TTA=False` in training.

**t-SNE**: only in `StackingInference.post_process` (`SZ/inference_feats.py` L86-99, L268-301). It would only `transform` a pickled reducer (or, absent a scaler, `StandardScaler().fit_transform` on the single sample). Inactive: (1) the scored mode uses `LogitFamilyStackingInference`, not `StackingInference`; (2) `stacking.py` never writes a `post_processing` key (L657-669); (3) no weights.zip `.pkl`. openTSNE is vendored but never imported outside `src/libs/`. Not fit on test cases.

**Nondeterminism**: inference is deterministic up to GPU float/cuDNN noise (eval mode, no dropout, deterministic preprocessing). Training: per-worker seeds and cudnn-deterministic, but README L221-232 reports run-to-run OOF variance (std 0.0119); multi-GPU concurrent workers; soft labels differ between 12-model and 3-model reproductions.

## 5. Important Files

### 5a. Source tree (`1st Place/`)
| File | Role | In scored path? |
|---|---|---|
| `README.md` | Solution writeup, setup, external data, license text (L247-250 "confidential", not MIT) | doc |
| `0_configs/default.yaml` | Paths (real, synthetic `external/PET_GENERATED`), teacher registry, `stacking.active_models` | training |
| `1_initial_training/prepare_folds.py` | Stratified 5-fold split of training labels | training |
| `1_initial_training/prepare_2d_dataset.py` | 2D slice dataset for 2D teachers | training |
| `1_initial_training/train_2d_resnet.py` | ResNet50 teacher, HF `edadaltocg/resnet50_cifar100` (L128-136) | training (Block 1) |
| `1_initial_training/train_2d_vit.py` | ViT-B/16 teacher, HF `edadaltocg/vit_base_patch16_224_in21k_ft_cifar100` (L104-113) | training (Block 1) |
| `1_initial_training/train_3d_densenet121/201.py`, `train_3d_attention.py` | MONAI DenseNet teachers, from scratch | training (Block 1) |
| `1_initial_training/train_medicalnet.py` + `models/` | MedicalNet ResNet/ResNeXt + MedNeXt teachers; pretrained only if `--pretrain_path` | training (Block 1) |
| `1_initial_training/train_svm.py` | pyradiomics features + SVC | training (Block 1) |
| `2_stacking/stacking.py`, `calibrators.py` | Meta-learner on teacher OOFs → `meta_model_stacking.pkl` | training (Block 1) |
| `3_soft_labeling/soft_labeling.py` | Applies teachers+meta to synthetic PET → `synthetic_soft_labels.csv` | training (Block 1→2) |
| `4_siamese_training/prepare_2d_dataset_siamese.py`, `prepare_2d_dataset_striatum.py` | Patch tensors for real (train labels) + synthetic (soft labels, y-inverted) | training (Block 2) |
| `4_siamese_training/train_siamese.py` | Trains the 100 `mashup_ensemble_exp_*_fold_*.pth` | **produces scored weights** |
| `5_pipeline/run_pipeline.py` | Phase orchestrator, `--start-at`, `--no-synth` | training |
| `6_runtime/pack_production.py` | Builds src.zip/weights.zip, packs and Docker-tests submission | packaging |
| `scripts/download_weights.py` | `huggingface_hub.snapshot_download("dinhviettoanle/dat-parkinson-models")` (L53-56); dispatches `01-soft_labeling/` (teacher folders, `SVM.pkl`, `tabular_oof.csv`, `synthetic_soft_labels.csv`) and `02-siamese/` (*.pth) | optional |
| `src/` | Inference library; tree copy of harness `src.zip` (minus `libs/`, plus `models/`) | yes (same code) |
| `competition-sfmn-parkinsons-runtime/submission_src/main.py`, `constants.py` | Tree inference entrypoint (same config as harness, `DATA_DIR=Path()`) | ≈ harness |
| `competition-sfmn-parkinsons-runtime/submission_src/test_inference.py`, `validate_pipeline.py` | Local validation vs OOF (differ from harness copies) | no |
| `competition-sfmn-parkinsons-runtime/runtime/*`, `justfile`, `Makefile` | DrivenData runtime (pyproject adds opentsne, ptwt, pykwalify, PyWavelets, pyradiomics, ruamel.yaml, xgboost L58-64) | env |
| `img/pipeline.png` (56,377 B, sha256 2483c5b6…4669), `img/data_processing.png` (146,608 B, sha256 e17b32f8…06e9) | README figures. **Not viewed.** README L176-179 says data_processing.png illustrates the striatum views → may depict real scan crops; organizer to inspect | doc |

### 5b. Harness zip root (`HZ/`)
| File | Role | Executed by scoring? |
|---|---|---|
| `main.py` | Entrypoint; FAMILY_STACKING / Mashup_Internal / T=1.094 | **yes** |
| `constants.py` | `prefix_map` (only `TwoPhase_Mean` active, inert for this config) | imported, inert |
| `src.zip` | Inference code + vendored `src/libs/` | **yes** |
| `weights.zip` | 100 × `weights/mashup_ensemble_exp_{0..19}_fold_{0..4}.pth` | **yes** |
| `validate_pipeline.py` | 9-phase local validation vs OOF; documents T=1.094 incl. **leaderboard factor 1.245** (L121-125, L698-700) | no |
| `validate_inference.py` | Checks inference vs `master_oof_predictions.csv` (older recipe T=0.8835) | no |
| `test_inference.py` | OOF reproduction smoke test | no |
| `generate_tta_oof.py`, `run_siamese_tta.py`, `check_tta_augmentations.py` | TTA experiments on training OOF (team paths `/data2/iceberg/...`, `/data/HOME/sansel/...`) | no |
| `check_orientation.py`, `compare_inputs.py`, `diagnose3d.py` | Debug/diagnostics on training scans / MiniCNN3D family | no |
| `copy_weights.py` | Copies weights from team home dirs (`/data/HOME/dvtle`, `/data/HOME/sansel`) | no |
| `tta_siamese_predictions.csv` (16,344 rows), `validation_report.csv` (500 rows) | Training-uid OOF outputs incl. training labels | no (data-retention item) |
| `__pycache__/` | stale bytecode | no |

### 5c. `src.zip` → `src/` (`SZ/`)
| File | Role | Executed? |
|---|---|---|
| `preprocessing.py` | `prepare_all_inputs`: per-case resample/crop/scale, siamese patches, pyradiomics features, 2D slices | **yes** |
| `inference_2d_avg.py` | `SiameseMashupInternalEnsemble` (L861-957) **active**; ResNet2D/ViT2D (differ from tree), other Siamese, TwoPhase classes inactive | **yes (one class)** |
| `builder.py` | `build_model("siamese_internal")` → `MashupInternalEnsembleCNN` (L886-937) | **yes** |
| `inference_feats.py` | `LogitFamilyStackingInference` (L413-465) **active**; `StackingInference` (t-SNE), `TabularInference`, `XGBoostEnsembleInference`, `SoloModelInference` inactive | **yes (one class)** |
| `calibrators.py` | Calibrator classes (star-imported) | imported only |
| `inference_2d_rnn/channel/corax/sagcorax.py`, `inference_3d.py`, `inference_tta_wrappers.py` | Other model families | imported only |
| `striatum_preproc.py`, `striatum_features.py`, `striatum_geometry.py`, `twophase_preproc.py` | TwoPhase scalar/mask features (needs `scaler_production.joblib` etc., absent) | imported only |
| `medicalnet/`, `mednext/` | 3D architectures | imported only |
| `libs/` | Vendored openTSNE, ptwt, pykwalify, pywt, radiomics, ruamel.yaml, xgboost 3.3.0 (Apache-2.0) | ptwt/radiomics/xgboost imported |

## 6. Claims (README) vs Code

| README claim | What code does | Verdict |
|---|---|---|
| Final model = Siamese on striatum crops, 3 light 4-layer CNN encoders, 1.9M params (L199-202) | `MashupInternalEnsembleCNN`, 3 encoders × 4 conv layers, ≈1.96M params | Consistent |
| Predictions = average of 100 models (20×5) (L205-207) | Mean of 100 sigmoid outputs (`inference_2d_avg.py` L923-957) | Consistent, but **README omits** T=1.094 logit scaling + clip [0.01,0.99] and that T includes a leaderboard-measured factor 1.245 |
| Synthetic DaT PET (Won et al., Zenodo 15377683; download from GitHub Krying/PD_SSL_ZOO → OneDrive) soft-labelled by Block-1 ensemble (L16-17, L71-81, L181-184) | `soft_labeling.py` labels only `external/PET_GENERATED`; used in all 100 Siamese trainings | Consistent |
| Block-1 = 12 NN + SVM; stacking LR + bag ensembling (L186-196) | `default.yaml` L106-118, `stacking.py` Meta-A/Meta-B | Consistent |
| Pretrained: MedicalNet, ResNet50 CIFAR-100, ViT in21k→CIFAR-100 (L238-245); Block 2 "no additional resources" | HF downloads confirmed for ResNet/ViT; MedicalNet pretrained loading not wired in pipeline; Siamese from scratch | Partly: MedicalNet weight use unverifiable from code |
| `python scripts/download_weights` places weights (L85-96) | Downloads HF repo `dinhviettoanle/dat-parkinson-models` (weights trained on competition data + `tabular_oof.csv` + soft labels) | Host visibility unknown → Release item |
| Pack step creates submission and smoke test (L153-160) | Yes, but zips whole `submission_src/` incl. diagnostic scripts and OOF CSVs | Consistent; explains harness extras |
| Inference default reads `data/niftis` (L170) | Tree `main.py` L20 `DATA_DIR=Path()` → `./niftis`; harness version falls back to `data/` | Tree claim wrong; harness OK |
| "Remaining errors": RandomHorizontalFlip flips top-down; synthetic upside-down at soft-labelling (L212-219) | `soft_labeling.py` uses `invert_y` default False; Phase 4 uses `INVERT_Y=True` for synthetic | Consistent with disclosure |
| License: materials confidential, not to be shared (L247-250) | No LICENSE file in tree root; `competition-sfmn-parkinsons-runtime/LICENSE` is DrivenData's runtime | **Conflicts with MIT requirement** |
| (not mentioned) | `tsne`, XGBoost, TwoPhase, MiniCNN3D, DWT families, many alt Siamese variants in code | Dead/inactive code |

## 7. Flags for Downstream Reviewers

**Code Integrity**
- CI-1 (info, likely OK): Executed path is per-case only (single NIfTI → patches → 100 CNNs → mean → T → clip). No batch/test-set statistics, no TTA, no pseudo-labelling, t-SNE inactive. Please confirm independently (CS1).
- CI-2 (medium, organizer awareness): `TEMP=1.094` includes a "factor 1.245 = offset correction measured on the leaderboard" (`HZ/validate_pipeline.py` L121-125, L698-700). A single global scalar tuned on public-LB feedback; not per-case information, but bears on rule F "do not overfit to the test set" and on documentation completeness. Clip bounds (0.01, 0.99) also not justified in delivered code.
- CI-3 (low): Harness ≠ tree in 4 root files and 3 src files, and harness has 8 extra scripts; none of the diffs touch the executed path. Tree `main.py` would not find data in the container (DATA_DIR bug). Tree `src/` lacks vendored `libs/`.
- CI-4 (low, CS4): per-case logging: `preprocessing.py` L585 warns NaN/Inf feature **names** (no uid/values) for the unused pyradiomics branch; `main.py` L438 logs uid only if image missing; `SiameseMashupInternalEnsemble` silently skips missing checkpoints (L874-875) — if 0 loaded, `np.mean([])`=NaN (not observed: scored CSV has no NaN).
- CI-5 (low): `prepare_all_inputs` computes many unused inputs per case (pyradiomics, MONAI 128³, 2D slices) — runtime only; within limits presumably (validate_pipeline extrapolates timing).

**Data Source**
- DS-1: Synthetic PET is the only external data in the final model; need license (Zenodo 15377683 / OneDrive) and the generator's training-data provenance (Q1-Q2).
- DS-2: Block-1 pretrained weights (two `edadaltocg` HF checkpoints; CIFAR-100, ImageNet-21k, and ResNet/ViT lineage) — license + pretraining data; pending cross-winner ImageNet ruling. They reach the final model only via soft labels.
- DS-3: MedicalNet: README lists it; code never passes `--pretrain_path`. Ask team whether Tencent MedicalNet weights (and their 3DSeg-8 pretraining datasets/licenses) were used in the original soft-labelling run.
- DS-4: "siamois_final_pseudo" = an earlier `SimpleSiameseEnsemble` variant (prefix only in `HZ/main.py` L260-261, commented in `constants.py` L9, TTA in `generate_tta_oof.py` L164, `copy_weights.py` L53). No training code, no weights in zip, not active. Name likely refers to synthetic soft ("pseudo") labels; no evidence of test-data pseudo-labelling anywhere (soft-labelling only globs `PET_GENERATED`). Coordinator found 0 test uids in tree/harness.
- DS-5 (CS12): No mention of AI coding tools in README; code comments are French with emoji style; no evidence either way.
- DS-6 (CS11): No manual annotations of competition data found (striatum localisation is algorithmic).

**Release Readiness / Eligibility**
- RR-1 (high): README license section is "confidential / do not share" — conflicts with required MIT; no LICENSE file for the solution.
- RR-2 (medium): `scripts/download_weights.py` points to HF `dinhviettoanle/dat-parkinson-models` containing weights trained on competition data plus `tabular_oof.csv` and `synthetic_soft_labels.csv`; must be private/organizer-only until dataset release (CS18). Visibility not checked (fetching competitor hosts is disallowed).
- RR-3 (medium, CS19): Harness zip ships `tta_siamese_predictions.csv` and `validation_report.csv` with training uids + labels; also diagnostic scripts with team-internal absolute paths. Should be removed from any release.
- RR-4 (low, CS15): Vendored `src/libs/` incl. 190 MB `libxgboost.so`; licenses of vendored libs (openTSNE BSD-3?, ptwt EUPL-1.2?, pykwalify MIT, PyWavelets MIT, pyradiomics BSD-3, ruamel.yaml MIT, xgboost Apache-2.0) to be confirmed; `runtime/pyproject.toml` modified to add them.
- RR-5 (low): `img/data_processing.png` may show real scan crops — not viewed; organizer to inspect before release.

## 8. Gaps and Assumptions

- Could not verify that the 100 `.pth` in `weights.zip` were produced by `train_siamese.py` (not loaded; filenames and architecture keys are consistent by code reading only).
- The notebook that froze the recipe (T, clip, "RECETTE FIGÉE", factor 1.245) is not delivered; T/clip derivation is not reproducible from the tree.
- `meta_model_stacking.pkl`, teacher weights, `synthetic_soft_labels.csv` and synthetic data are not in the submission; Block 1 is reproducible only by re-running (~3.5 days) or by the HF download.
- Number of synthetic PET samples used is not stated in code or README.
- Assumed the container cwd is `/code_execution` with data at `./data` (runtime `entrypoint.sh` L22-36 cd's there; `test-submission` mounts data at `/code_execution/data`).
- Assumed `prepare_all_inputs` side branches (pyradiomics etc.) have no cross-case state; grep found no module globals mutated, `build_twophase_data`/`build_tabular_data` calls are commented out (L675-685).
- Did not deep-read Block-1 teacher training scripts beyond data sources and pretrained-weight entry points; all read `cfg.paths.train_labels` / `cfg.paths.niftis`.
