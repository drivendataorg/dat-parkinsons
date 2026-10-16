# Provenance of the scored archive

This document maps every file of the scored archive to the code in this repository that
produces it, and records how each file was checked. `docs/REPRODUCIBILITY.md` covers
how far each stage reproduces on other machines.

| item | value |
|---|---|
| submission | v17, DrivenData submission id 322397 (team NavAttack) |
| archive | `submission.zip`, 1,090,109,497 bytes, 82 entries (1,138,933,628 bytes uncompressed) |
| archive sha256 | `88b5611e25170d88c2fbf21b62443dbc67b722c519b5a0391a60aab62bbec7fd` |
| compression | 3 entries stored (`tabicl_lib.zip` and the two `.pkl.xz` files), 79 deflated |
| scores | public log loss 0.2409, private log loss 0.2731, private AUROC 0.9524 |

The sha256 of the archive itself cannot be regenerated, because every zip entry carries a
timestamp. All checks below are per entry: name, size, CRC32 and, where stated, sha256.

## Verification classes

Each entry in the tables below carries one or two of these classes.

| class | meaning |
|---|---|
| byte-identical regeneration | A pipeline stage or script of this repository rebuilt the file and its sha256 equals the scored entry. |
| prediction-identical regeneration | A pipeline stage rebuilt the file. Its bytes differ for a documented reason, and loading it gives outputs identical to the scored file (max absolute difference 0.0). |
| verbatim copy | The repository ships the file, or installs it from the released weights, and the packaging step copies it unchanged. Its sha256 equals the scored entry. |
| frozen | Shipped as released. No script in this repository regenerates it. |

The 60 files under `assets/` are listed with their full sha256 in
`models/SHA256SUMS.scored.txt`. The other 22 entries are the contents of `inference/`,
which is byte-identical to the archive (22 of 22 sha256 equal). `pipeline/09_package.sh`
ignores `inference/assets/.gitkeep`, otherwise the rebuilt archive would have 83 entries.

## 1. Archive entries

### 1.1 Code, licence and metadata (22 entries)

Stage 09 (`pipeline/09_package.sh`, `src/package_submission.py`) copies all 22 from
`inference/`. The origin column names the step of the original project that wrote each
file. Paths such as `experiments/v17/...` refer to the original development tree, which
is not part of this repository.

| entry | bytes | sha256 (first 16) | what it is | origin | verification |
|---|---:|---|---|---|---|
| `CANDIDATE.json` | 3,731 | `3b17690df1b6dd60` | Candidate record: validation aggregates, persistence checks, native-scan checks. Contains stale status text (section 4). | Written by `experiments/v17/prepare_candidate.py`, `freeze_candidate.py` and `package_candidate.py` on 2026-09-14 | verbatim copy |
| `TABICL_MODEL_CARD.md` | 1,170 | `af2a149343cb949f` | Model card of the TabICLv2 checkpoint | README of the Hugging Face repository `jingang/TabICL`, copied by `experiments/20260914/build_stage.py` | verbatim copy (third party) |
| `cnn2d.py` | 17,864 | `c08fd0438f3cd601` | 2D EfficientNet reader: three-MIP images, TorchScript loading, training loop | Project source. Same file as `src/cnn2d.py`, and the same file that trained the 2D members | verbatim copy |
| `cnn3.py` | 26,252 | `8c5a15448c0b392b` | 3D CNN input transform used at inference, and the 3D trainer | Project source, `src/cnn3.py` | verbatim copy |
| `cnn4.py` | 16,881 | `092e4dc58d0a7ed3` | Per-hemisphere crop trainer and helpers. Imported at inference, not exercised by the shipped members | Project source, `src/cnn4.py` | verbatim copy |
| `features13.py` | 4,127 | `ffa6608fc37a2f20` | Radial specific-binding shells on top of `features6` (the lin and lin4 feature set) | Project source, `src/features13.py` | verbatim copy |
| `features4.py` | 8,230 | `869a77cf76279314` | Clinical ROI features in the canonical striatal frame | Project source, `src/features4.py` | verbatim copy |
| `features5.py` | 2,983 | `e091b2f42352ff9c` | `features4` plus a spatial specific-binding profile | Project source, `src/features5.py` | verbatim copy |
| `features6.py` | 5,806 | `a814e3992e09decb` | `features5` plus per-hemisphere shape descriptors | Project source, `src/features6.py` | verbatim copy |
| `inference_main.py` | 23,753 | `a52edb26132becf5` | Inference pipeline: preprocessing, registration, features, members, stacker, TabICL blend, governor, clip | The v14c entry point (`src/main3.py`) patched by `experiments/20260914/build_stage.py`, `package_diagnostic.py` and `experiments/v17/prepare_candidate.py`. Re-applying those patches reproduces the file byte for byte. | verbatim copy |
| `main.py` | 816 | `7e0332fbd8c41b11` | Wrapper: puts `tabicl_lib.zip` on `sys.path`, prints 7 package versions, prints only exception type and file:line on failure | Literal text in `experiments/20260914/package_diagnostic.py` | verbatim copy |
| `mixture_features.py` | 3,040 | `6ebc165475171e24` | 60 spatial uptake-mixture features | Slice of `experiments/v17/mixture_shape.py` (kept as `tabular/_original/mixture_shape.py`) cut by `prepare_candidate.py`. Re-running the slice reproduces the file byte for byte. | verbatim copy |
| `nets3.py` | 15,863 | `78d06a04b162bf2e` | 3D architectures, including HemiMIL (`arch mil`) | Project source, `src/nets3.py` | verbatim copy |
| `portable3.py` | 5,044 | `52d4bcb04eb76e9e` | `FeatPrep` and the JSON format of the linear models | Project source, `src/portable3.py` | verbatim copy |
| `preprocessing3.py` | 39,996 | `287ac71d61e3c63c` | NIfTI loading, head mask with disc guard, reference level, striatal frame, 48³ SBR crops | Project source, `src/preprocessing3.py` | verbatim copy |
| `regfit.py` | 5,345 | `d13ea78dc47ce698` | Rigid registration to the template and the rejection rule (90 degrees, NCC 0.30) | Project source, `src/regfit.py` | verbatim copy |
| `register3.py` | 11,938 | `93b51e53e38b441f` | Registration primitives and the template builder | Project source, `src/register3.py` | verbatim copy |
| `roigeom.py` | 6,248 | `e5f344f95cd1d286` | Striatal ROI placement geometry | Project source, `src/roigeom.py` | verbatim copy |
| `scale_features.py` | 3,349 | `6e6bd8290aad09bd` | 180 scale-space shape features | `experiments/20260914/scale_features.py` (kept as `tabular/_original/scale_features_research.py`) with its command-line block removed. The removal reproduces the file byte for byte. | verbatim copy |
| `tabicl_lib.zip` | 286,222 | `6a57bb28590e713f` | TabICL 2.2.0 source, imported with zipimport | Zip of the `.py` files of the tabicl 2.2.0 wheel, written by `package_diagnostic.py`. All 91 files match the sha256 values in the wheel's RECORD. | verbatim copy (third party) |
| `tabicl_licenses/LICENSE` | 12,996 | `c959a1db8bc4b2f4` | TabICL licence (BSD-3-Clause, with an Apache-2.0 section for one subpackage) | Licence file of the tabicl 2.2.0 wheel | verbatim copy (third party) |
| `tabular_adapter.py` | 3,940 | `518d344ee1dd7d30` | Loads the two frozen contexts, recomputes the 4 prior scores, predicts and blends | `experiments/20260914/tabular_adapter.py` plus the 7-line prior patch of `prepare_candidate.py`. Re-applying the patch reproduces the file byte for byte. | verbatim copy |

The 13 project modules `cnn2d.py`, `cnn3.py`, `cnn4.py`, `features4.py`, `features5.py`,
`features6.py`, `features13.py`, `nets3.py`, `portable3.py`, `preprocessing3.py`,
`regfit.py`, `register3.py` and `roigeom.py` are also byte-identical to the files of the
same name in `src/` (section 2.1).

### 1.2 Single assets (8 entries)

| entry | bytes | sha256 (first 16) | what it is | stage and script | verification |
|---|---:|---|---|---|---|
| `assets/inference_config.json` | 33,685 | `6c1a8bf6e3092ef9` | Members, stacker coefficients, calibration (T 1.4, clip 0.005), governor, CNN specs, registration settings, prior 0.5485, 479-name `feature_order` | 08: `src/fixstack.py` then `src/build3.py` write the v14c config (sha256 `dcf4e4747c5ed030`). 09: `src/package_submission.py` adds `"tabicl": true` and extends `feature_order` from 235 to 479 names | byte-identical regeneration from the raw data (the complete `MODE=full` run of `docs/REPRODUCIBILITY.md` section 5; also stages 08 and 09 alone, with the archived out-of-fold tables as input) |
| `assets/linear_lin.json` | 28,463 | `63d3799a19be6d1a` | L1 logistic regression on 235 features, C 0.2, 48 non-zero coefficients | 03: `pipeline/03_linear.sh` runs `src/train_final3.py --export` | byte-identical regeneration from the raw data |
| `assets/linear_lin4.json` | 11,699 | `e2d55d2f38d9e9af` | L1 logistic regression on 94 features, C 1.0, 40 non-zero coefficients | 03: same command | byte-identical regeneration from the raw data |
| `assets/tabicl-classifier-v2-20260212.ckpt` | 110,368,038 | `bdc7dbd5e4ff21f8` | Public TabICLv2 checkpoint, Hugging Face `jingang/TabICL` at revision `4dcd344ece2c00be9e831fdd35bed57b5ad83e19` | Not trained here. 09 copies the file named by `TABICL_CKPT`, by default `inference/assets/tabicl-classifier-v2-20260212.ckpt`, where `scripts/install_weights.py` installs it | frozen (third party, sha256 equal to the Hugging Face LFS object id) |
| `assets/tabular_log.pkl.xz` | 226,742,860 | `af158889b24828d1` | Frozen TabICL context on log-prepared features, 1,148 acquisition-balanced rows | 07: `pipeline/07_tabular.sh` runs `tabular/freeze_tabular.py` | prediction-identical regeneration |
| `assets/tabular_raw.pkl.xz` | 229,985,568 | `167455ac379843f8` | Frozen TabICL context on raw features, all 1,362 rows | 07: same | prediction-identical regeneration |
| `assets/tabular_recipe.json` | 211,539 | `211f53509c4979f0` | 64 selected features, 4 acquisition descriptors, 4 linear prior models, `log_prep`, medians, blend weight 0.35, temperature 1.2 | 07: `tabular/prepare_tabular.py` | byte-identical regeneration from the raw data |
| `assets/template_v6.npy` | 442,496 | `4d3629d5a06aadd8` | 48³ float32 registration template in log1p(SBR) space, a label-free average of the training crops | Written on 2026-07-25 by `src/register3.py` with its default arguments. No pipeline stage rebuilds it: 02 stages the released copy into `outputs/` and checks its sha256, and 08 `src/build3.py` copies it into the assets | byte-identical regeneration (`src/register3.py` run outside the stages, twice, `docs/REPRODUCIBILITY.md` section 4) |

For the two `.pkl.xz` files, the rebuilt pickles differ in bytes because torch writes a
per-process serialisation id when it pickles a tensor. All 14 KV-cache arrays of each
context are bitwise equal to the scored ones, 29 of 36 arrays are bitwise equal, the other
7 (scikit-learn preprocessor statistics) differ by at most 1.8e-15, and predictions through
the shipped `TabularAdapter` on the 1,362 training rows differ by 0.0 for the raw model, the
log model and their blend.

### 1.3 CNN checkpoints (52 entries)

The package member `mil48y` averages the 34 3D checkpoints. The member `e2dmipf2`
averages the 12 checkpoints of the two B0 runs `e2dmipf2` and `e2dmipf22`, and the member
`e2db3f2` averages the 6 B3 checkpoints.

| group | entries | trained by (original run) | stage and script in this repository | verification |
|---|---|---|---|---|
| 3D-A | `mil48a_fold0`..`4`, `mil48b_fold0`..`4` (10) | `cnn3.py` of the v3 archive, `fleet3.sh` (seed base 3000) and `fleet5.sh` (seed base 7000), 2026-07-24, MPS | 05: `pipeline/05_cnn3d.sh` runs `src/cnn3.py`. 08: `src/build3.py` loads each checkpoint and saves it again into `submission_src/assets/`. 09 copies | `MODE=fast`: verbatim copy. The re-save by `build3.py` under torch 2.13.0 is byte-identical. `MODE=full` (complete retrain, 2026-09-28): prediction-identical regeneration. All 10 have 38 of 38 state tensors `torch.equal` to the shipped files, with equal normalisation constants. The file bytes differ because the current `cnn3.py` stores 15 dictionary entries where the 2026-07-24 version stored 9. |
| 3D-B | `m31000_fold0`..`4`, `m33000_fold0`..`4` (10) | An intermediate `cnn3.py` of 2026-07-25 that no longer exists, `run_seeds.sh`, MPS | as 3D-A | `MODE=fast`: verbatim copy, byte-identical after the re-save. `MODE=full` (complete retrain, 2026-09-28): byte-identical regeneration from the raw data, 38 of 38 state tensors equal. |
| 3D-C | `mf21000_full0`, `mf23000_full0`, `mf25000_full0`, `mf27000_full0` (4) | The same 2026-07-25 intermediate, `run_wave3.sh`, `--full`, seed = base + 99, MPS | as 3D-A | `MODE=fast`: verbatim copy, byte-identical after the re-save. `MODE=full` (complete retrain, 2026-09-28): byte-identical regeneration from the raw data, 38 of 38 state tensors equal. |
| 3D-D | `m35000_fold0`..`4`, `m37000_fold0`..`4` (10) | The `cnn3.py` preserved as `cnn3.py.bak_v11` in the original tree, `night_queue3.sh`, 2026-08-22, MPS | as 3D-A | `MODE=fast`: verbatim copy, byte-identical after the re-save. `MODE=full` (complete retrain, 2026-09-28): byte-identical regeneration from the raw data, 38 of 38 state tensors equal. |
| 2D-B0 | `e2dmipf2_{fold0..4,full0}.ts.pt`, `e2dmipf22_{fold0..4,full0}.ts.pt` (12) | The shipped `cnn2d.py`, EfficientNet-B0 at 192 px, seeds 9000 + k and 9100 + k (full-data 9099 and 9199), `tools/gpu_queue2.sh`, `f2seed2_chain.sh`, `b3_chain.sh`, 2026-09-08, MPS. Exported by `tools/ts_export.py` | 06: `pipeline/06_cnn2d.sh` runs `src/cnn2d.py`, then `tools/ts_export.py`. 08: `src/build3.py` reads the raw state dict for the spec and copies the `.ts.pt`. 09 copies | `MODE=fast`: copy of the installed file. It equals the scored entry when the weights come from the scored archive, and `models/SHA256SUMS.public.txt` when they come from the released bundle (section 3). Re-export of the 18 original raw checkpoints with `tools/ts_export.py`: max absolute logit difference 0.0 on 256 images for every file, bytes differ. `MODE=full` (complete retrain, 2026-09-28): prediction-identical regeneration. All 12 raw state dicts are byte-identical to the original raw checkpoints, and the re-exported `.ts.pt` files give logits identical to the released ones, while their bytes differ (source paths). |
| 2D-B3 | `e2db3f2_{fold0..4,full0}.ts.pt` (6) | The shipped `cnn2d.py`, EfficientNet-B3 at 224 px, seeds 9000 + k and 9099, `tools/post_b3_chain.sh` and `v14c_chain.sh`, 2026-09-08, MPS | as 2D-B0 | As 2D-B0 for `MODE=fast`, the re-export and `MODE=full` (all 6 raw state dicts byte-identical to the originals, 574 of 574 tensors). |

Per-file sizes and hashes of the 52 checkpoints. The last column gives the hash of the
same file in the released weights bundle where it differs from the scored file.

| entry | bytes | sha256 (first 16) | group | public-bundle sha256 (first 16) |
|---|---:|---|---|---|
| `assets/e2db3f2_fold0.ts.pt` | 42,701,870 | `5d1493b1595b5005` | 2D-B3 | `fa2723dd1b5e155d` |
| `assets/e2db3f2_fold1.ts.pt` | 42,701,872 | `66dbf1c3513d3be4` | 2D-B3 | `971aaf658efe2d42` |
| `assets/e2db3f2_fold2.ts.pt` | 42,701,872 | `2d56c8bac4aeea69` | 2D-B3 | `afbb0eb8a85b140b` |
| `assets/e2db3f2_fold3.ts.pt` | 42,701,872 | `3ad38db8070a41e8` | 2D-B3 | `3f859689b027e56f` |
| `assets/e2db3f2_fold4.ts.pt` | 42,701,872 | `cbfb80018930b6c5` | 2D-B3 | `5c65bd3c19f73bde` |
| `assets/e2db3f2_full0.ts.pt` | 42,701,872 | `fb4e2d8d81f16d71` | 2D-B3 | `36c649b3399127ed` |
| `assets/e2dmipf22_fold0.ts.pt` | 16,006,252 | `b166646441a60676` | 2D-B0 | `ffa2a42bfe44f94e` |
| `assets/e2dmipf22_fold1.ts.pt` | 16,006,252 | `4805e24e5d594afc` | 2D-B0 | `b7f4fbf5e75f9b2d` |
| `assets/e2dmipf22_fold2.ts.pt` | 16,006,254 | `e860193352a2c71d` | 2D-B0 | `d0e3196b990eba70` |
| `assets/e2dmipf22_fold3.ts.pt` | 16,006,254 | `e5d5cd1661095b95` | 2D-B0 | `3a1b262e80fe68fe` |
| `assets/e2dmipf22_fold4.ts.pt` | 16,006,254 | `6c472b0343808b06` | 2D-B0 | `893e609b187bc9d3` |
| `assets/e2dmipf22_full0.ts.pt` | 16,006,254 | `000250ebe823c39d` | 2D-B0 | `581e7f3c6992e4c4` |
| `assets/e2dmipf2_fold0.ts.pt` | 16,006,017 | `7a9ca56564c98b12` | 2D-B0 | `3a61775c0235790a` |
| `assets/e2dmipf2_fold1.ts.pt` | 16,006,017 | `beed9ee20f315f01` | 2D-B0 | `c813595d1b3a721b` |
| `assets/e2dmipf2_fold2.ts.pt` | 16,006,019 | `4a1b7eab30430c2c` | 2D-B0 | `1bde49552c78abd2` |
| `assets/e2dmipf2_fold3.ts.pt` | 16,006,019 | `7bd6f9e18ba5fb74` | 2D-B0 | `e6dde1471cafc860` |
| `assets/e2dmipf2_fold4.ts.pt` | 16,006,019 | `c3de91bef2831b1f` | 2D-B0 | `b95ed7102d5587b9` |
| `assets/e2dmipf2_full0.ts.pt` | 16,006,019 | `bb7046be718f0522` | 2D-B0 | `529f863230dd77e6` |
| `assets/m31000_fold0.pt` | 3,597,645 | `115b2c3352c0ddf4` | 3D-B | same |
| `assets/m31000_fold1.pt` | 3,597,645 | `8807f4fe3cf100fa` | 3D-B | same |
| `assets/m31000_fold2.pt` | 3,597,645 | `18e62e94d3bbdf6a` | 3D-B | same |
| `assets/m31000_fold3.pt` | 3,597,645 | `4862883557f5cc98` | 3D-B | same |
| `assets/m31000_fold4.pt` | 3,597,645 | `3c368ae2dba6f336` | 3D-B | same |
| `assets/m33000_fold0.pt` | 3,597,645 | `9c93fbbbcfe0f972` | 3D-B | same |
| `assets/m33000_fold1.pt` | 3,597,645 | `2b761363b1bb8399` | 3D-B | same |
| `assets/m33000_fold2.pt` | 3,597,645 | `20a05fa66038d15b` | 3D-B | same |
| `assets/m33000_fold3.pt` | 3,597,645 | `ac5737bb4bf17f42` | 3D-B | same |
| `assets/m33000_fold4.pt` | 3,597,645 | `3e5523ef628efd1e` | 3D-B | same |
| `assets/m35000_fold0.pt` | 3,597,645 | `72ea9039b078a2b3` | 3D-D | same |
| `assets/m35000_fold1.pt` | 3,597,645 | `b36a3674c94447a6` | 3D-D | same |
| `assets/m35000_fold2.pt` | 3,597,645 | `7686727ec34ea93d` | 3D-D | same |
| `assets/m35000_fold3.pt` | 3,597,645 | `3b68dc90a1529ea4` | 3D-D | same |
| `assets/m35000_fold4.pt` | 3,597,645 | `d6b712bdaa7eb21e` | 3D-D | same |
| `assets/m37000_fold0.pt` | 3,597,645 | `142e173fafa13c2d` | 3D-D | same |
| `assets/m37000_fold1.pt` | 3,597,645 | `2b82af5649d3269b` | 3D-D | same |
| `assets/m37000_fold2.pt` | 3,597,645 | `868e100cc579d1b2` | 3D-D | same |
| `assets/m37000_fold3.pt` | 3,597,645 | `b3acbfb365896ec6` | 3D-D | same |
| `assets/m37000_fold4.pt` | 3,597,645 | `b2145fe520b9427a` | 3D-D | same |
| `assets/mf21000_full0.pt` | 3,597,689 | `faee71df643009f3` | 3D-C | same |
| `assets/mf23000_full0.pt` | 3,597,689 | `6c091b7e3b0a79c4` | 3D-C | same |
| `assets/mf25000_full0.pt` | 3,597,689 | `8c2651e76c8e6a8d` | 3D-C | same |
| `assets/mf27000_full0.pt` | 3,597,689 | `e4f5480073ba89da` | 3D-C | same |
| `assets/mil48a_fold0.pt` | 3,597,517 | `f44069e1a0ac5d88` | 3D-A | same |
| `assets/mil48a_fold1.pt` | 3,597,517 | `f5b8334fb9348758` | 3D-A | same |
| `assets/mil48a_fold2.pt` | 3,597,517 | `5b15ebeb8e17be08` | 3D-A | same |
| `assets/mil48a_fold3.pt` | 3,597,517 | `360039207d1cd2a2` | 3D-A | same |
| `assets/mil48a_fold4.pt` | 3,597,517 | `78e4eedd528d4d84` | 3D-A | same |
| `assets/mil48b_fold0.pt` | 3,597,517 | `ec5953b1f301c3cb` | 3D-A | same |
| `assets/mil48b_fold1.pt` | 3,597,517 | `67df4e2cbb83f750` | 3D-A | same |
| `assets/mil48b_fold2.pt` | 3,597,517 | `d79fe44603101594` | 3D-A | same |
| `assets/mil48b_fold3.pt` | 3,597,517 | `abb9a66fd6a187c4` | 3D-A | same |
| `assets/mil48b_fold4.pt` | 3,597,517 | `f3159f71785df295` | 3D-A | same |

## 2. Training modules in this repository and the original code

### 2.1 Byte-identical to the scored inference copies

These 13 files in `src/` are byte-identical to the files of the same name in `inference/`
and in the scored archive, and to the files in the original project's `src/`:
`cnn2d.py`, `cnn3.py`, `cnn4.py`, `features13.py`, `features4.py`, `features5.py`,
`features6.py`, `nets3.py`, `portable3.py`, `preprocessing3.py`, `regfit.py`,
`register3.py`, `roigeom.py`. Their hashes are in the table of section 1.1.

### 2.2 Byte-identical original files that are not in the archive

| file | sha256 (first 16) | original role |
|---|---|---|
| `src/avg_member.py` | `0815e2683195062d` | Probability average of 3D runs into one member OOF file (4 runs for `oof_mil48y.csv`) |
| `src/fixstack.py` | `84d1d1073ea719e7` | Fixed-ratio logit stacker with one 1-D logistic calibration |
| `src/main3.py` | `47a5e8bdb9b40dc7` | The v14c inference entry point. `build3.py` writes it to `submission_src/main.py`, and stage 09 does not use it |
| `src/train_final3.py` | `43163dcfe2f2183c` | Fits and exports `lin` and `lin4`, plus LightGBM and MLP side models that v17 does not ship |
| `tools/hdr_fingerprint.py` | `6e2d8064157fb5f5` | Header-only survey of the NIfTI files, input to the acquisition groups. `tabular/derive_groups.py` passes it the `niftis/` folder of `DAT_DATA_DIR` (default `data/raw/`). Run on its own from the repository root, it needs that folder as argument (`python tools/hdr_fingerprint.py data/raw/niftis`), because its default is the original tree's `data-demo/niftis` |
| `tabular/_original/bench_v16.py` | `3b00ec381fa4ac2c` | Reference copy of `experiments/v16/bench.py`. Not imported |
| `tabular/_original/common_20260914.py` | `d1eb6c86e5e14e9a` | Reference copy of `experiments/20260914/common.py`. Not imported |
| `tabular/_original/freeze_candidate.py` | `6f6fcd249a311710` | Reference copy of `experiments/v17/freeze_candidate.py`. Not imported |
| `tabular/_original/mixture_shape.py` | `a593f229776052f1` | Reference copy of `experiments/v17/mixture_shape.py`. Not imported |
| `tabular/_original/prepare_candidate.py` | `658545fdd8c85d13` | Reference copy of `experiments/v17/prepare_candidate.py`. Not imported |
| `tabular/_original/scale_features_research.py` | `39642772f1c36145` | Reference copy of `experiments/20260914/scale_features.py`. Not imported |

### 2.3 Files with the data-path patch

The original code read the competition data from a hard-coded `data-demo/` folder in the
project root. Five files carry one patch that replaces that location:

```python
DATA = Path(os.environ.get("DAT_DATA_DIR", str(ROOT / "data" / "raw")))
```

`NIFTI` becomes `DATA / "niftis"` and the labels file becomes `DATA / "labels.csv"`
(`DATA / args.labels` in `make_regfeat32.py`). The patch adds `import os` where the module
did not import it, and a one-line comment. Nothing else changes. In `eda_real.py` the
original labels path was the downloaded `train_labels_<id>.csv` in the project root, a
byte-identical copy of `data-demo/labels.csv`.

| file | original sha256 | sha256 in this repository |
|---|---|---|
| `src/cache3.py` | `6a702d10999140756a0af57fa35c802ba53e96af264daf0c6559957b9a78de29` | `99d328ea82dc7186bef49e9276341f42ac8aa5aa0272513ee05c1b6f6399e025` |
| `src/folds.py` | `c4840d0d83df5e34402e9fa9304f2aeb749a6cbefef64287a448bdb7b26e40a7` | `7697a6d6559e298b2a4d8f2abc27847414c0217242cbb9bfde1c3ce7c84d9ea2` |
| `src/make_regfeat32.py` | `a5354d38f310a054835814d443eb239f685d0b9c4195da44a57d48b6978d7114` | `5d4f39a086c4f4668a64edc92056750044b2c3e2d7108388954a2c6f5dc4dd84` |
| `src/build3.py` | `4ac31f6ac126a9276dc10d3d48334251eb23531484f3b5041f9c4ecafd838d84` | `7b8412476a128eda80bc4bb02c3f8b256aca1872e0f150984c0643b2921e2305` |
| `src/eda_real.py` | `1d10c50c8963f5631e88aa46996b61bcb471e112bbd6264d77538e2f56032729` | `c232ebe25ad66feba72dfec41898269004409c2309e9f202d6218309ebbfdaee` |

### 2.4 `tools/ts_export.py`

| file | original sha256 | sha256 in this repository |
|---|---|---|
| `tools/ts_export.py` | `c26358f82b1b3636f195d9453a412f012157828d465eab5c9bdba1ecb0f81919` | `4a3fb7f53b1a0ff860019d6523fe0b8fe14dc1e9b7722ed800e456e79b17378c` |

Two changes, both documented in the file header. The sample batch used for tracing and for
the eager-versus-traced check is read from `outputs/crops_v3_f2.npz`, the cache the 2D
members were trained on. The original read `outputs/crops_v3_m2.npz`, a cache that this
pipeline does not build. The values in that batch change only the printed check numbers,
because EfficientNet has no data-dependent control flow. The second change makes the final
message name the file the script writes, `outputs/<TAG>_meta.json`. The re-export of the 18
original raw checkpoints with the patched script gives `*_meta.json` files byte-identical
to the originals and logits identical to the scored `.ts.pt` files.

### 2.5 New scripts

"Recovered" below means the code was recovered from the original development session logs,
because the original step was typed interactively and never saved as a file.

| script | original step it reproduces |
|---|---|
| `src/filter_feature_tables.py` | Two recovered pandas steps: 2026-09-02 (restrict the registered tables to the champion-v7 column list, which drops `hole_ml`) and 2026-09-03 (drop `head_ml`), giving the `*_m2f.csv` and `*_m2fnh.csv` tables |
| `src/linear_oof.py` | The recovered step of 2026-09-03 that wrote `outputs/oof_feat3_m2fnh.csv`, the 5-fold OOF of `lin` and `lin4` read by the stacker |
| `src/avg_2d_seeds.py` | The inline pandas step of `tools/f2seed2_chain.sh` that wrote `oof_e2dmipf2x.csv`, the unclipped mean of the two B0 seeds |
| `src/package_submission.py` | `experiments/v17/package_candidate.py`, plus the config extension of `build_stage.py` and `prepare_candidate.py` |
| `src/check_package.py` | Gate 6 of `tools/build_f2two.sh` (no f-string logger calls) and the static checks of `package_candidate.py` (testzip, no `.npz`, shallow `/code_execution` import, compile) |
| `tools/replay_cnn_oof.py` | Rewrite of `experiments/20260914/replay_cnn_oof.py` and `experiments/v16/replay_2d.py`. Regenerates each run's OOF file from the released checkpoints in the trainers' format, for `MODE=fast` |
| `tools/fetch_backbones.py` | The unrecorded pre-download of the two ImageNet backbones into the Hugging Face cache, now with pinned revisions and sha256 |
| `tools/verify_cnn.py` | No original. Aggregate-only checks of the rebuilt caches, the fold-0 retrains and the TorchScript re-export |
| `tabular/derive_groups.py` | `tools/site_diag.py` merges plus the unsaved 2026-09-03 step that added the `fam2` and `scale` group columns (rules recovered) |
| `tabular/extract_shape_features.py` | The command-line block of `experiments/20260914/scale_features.py`, which wrote `scale_f2.csv` |
| `tabular/extract_mixture_features.py` | `main()` of `experiments/v17/mixture_shape.py`, which wrote `mixture_shape.csv` |
| `tabular/prepare_tabular.py` | `experiments/v17/prepare_candidate.py` with the research-path lookups of `bench.py` and `common.py` turned into arguments |
| `tabular/freeze_tabular.py` | `experiments/v17/freeze_candidate.py` |
| `tabular/verify_features.py`, `verify_prepare.py`, `verify_freeze.py` | No original. Aggregate-only comparisons with the archived features, recipe, context and predictions |
| `pipeline/00_stage_data.sh` | The unrecorded unpacking of the training archive and labels into `data-demo/` |
| `pipeline/01_folds.sh` | `cd src && python folds.py` |
| `pipeline/02_features.sh` | The 2026-09-02 `make_regfeat32.py` command (log path and worker count changed), then `src/filter_feature_tables.py` |
| `pipeline/03_linear.sh` | `src/linear_oof.py`, then the `train_final3.py --export` line of `tools/build_clean.sh` and its 235/94 column assertion |
| `pipeline/04_caches.sh` | `cache3.py _b` of 2026-07-24 (reproduced with `DAT_MASK_V2=0`) and `cache3.py _f2` of `tools/frame_cpu_chain.sh` |
| `pipeline/05_cnn3d.sh` | The 10 3D runs of `fleet3.sh`, `fleet5.sh`, `run_wave3.sh`, `run_seeds.sh` and `night_queue3.sh`, then `avg_member.py` |
| `pipeline/06_cnn2d.sh` | The 6 2D runs of `gpu_queue2.sh`, `f2seed2_chain.sh`, `b3_chain.sh`, `post_b3_chain.sh` and `v14c_chain.sh`, the two-seed average and `ts_export.py` |
| `pipeline/07_tabular.sh` | The v15a to v17 tabular chain (groups, shape and mixture features, prepare, freeze) |
| `pipeline/08_stack.sh` | The stacker and build part of `tools/build_f2two.sh v14c e2dmipf2x e2dmipf2 e2dmipf2+e2dmipf22`, as called by `tools/v14c_chain.sh` on 2026-09-08 |
| `pipeline/09_package.sh` | `experiments/v17/package_candidate.py` |
| `pipeline/smoke_cnn.sh` | No original. Runs every CNN training command for one epoch under `smoke_` tags |
| `pipeline/train_all.sh` | No single original command. Runs stages 00 to 09, and in `MODE=fast` stages the released CNN checkpoints and replays their OOF files |
| `scripts/install_weights.py` | No original. Installs the 60 assets and checks each sha256 against one of the two lists in `models/` |
| `scripts/predict.sh` | The runtime entrypoint of the competition container: `DATA_DIR` set, `python main.py` run in a working directory, `submission.csv` collected |
| `scripts/predict_folder.py` | No original. Builds the competition layout for any folder of NIfTI files or a single file, then calls `predict.sh` |

## 3. TorchScript debug records in the released weights

Each of the 18 `.ts.pt` files is a zip archive. When the original machine exported them,
torch wrote one `*.debug_pkl` record into each file. That record holds source-range
information, including the absolute paths of the exporting machine's Python files. A
re-export from this repository embeds the reproducer's own paths in the same way.

The released weights bundle carries the 18 files without that record. Each archive was
rewritten with every other record copied byte for byte, in the original order, with the
original compression type and timestamp:

```python
with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w") as zo:
    for info in zi.infolist():
        if info.filename.endswith(".debug_pkl"):
            continue
        ni = zipfile.ZipInfo(info.filename, date_time=info.date_time)
        ni.compress_type = info.compress_type
        ni.external_attr = info.external_attr
        zo.writestr(ni, zi.read(info.filename))
```

| files | records before, after | bytes before | bytes after |
|---|---|---:|---:|
| 6 `e2dmipf2_*.ts.pt` | 171, 170 | 16,006,017 to 16,006,019 | 15,985,910 to 15,985,914 |
| 6 `e2dmipf22_*.ts.pt` | 171, 170 | 16,006,252 to 16,006,254 | 15,986,250 to 15,986,254 |
| 6 `e2db3f2_*.ts.pt` | 269, 268 | 42,701,870 to 42,701,872 | 42,671,181 to 42,671,185 |

Result: every original and scrubbed pair, loaded with `torch.jit.load` on the CPU and fed
the same fixed synthetic batch at the member's resolution (192 px for B0, 224 px for B3),
gives `torch.equal` outputs for all 18 files. A byte scan of every record of the 18 scrubbed
archives found no path of the exporting machine's user directory.

`models/SHA256SUMS.public.txt` lists the scrubbed hashes. It differs from
`models/SHA256SUMS.scored.txt` in exactly these 18 lines, and the other 42 hashes are
equal. `scripts/install_weights.py` accepts a file that matches either list, so the scored
archive and the released bundle both install. The 18 scrubbed files are listed in the last
column of the table in section 1.3.

## 4. Stale text inside the scored archive

The archive was uploaded unchanged, and any edit would change its sha256, so the files
below keep text that was out of date when the package was scored.

`CANDIDATE.json` was written on 2026-09-14 before the upload and still describes the
package as a local candidate:

```json
"name": "v17-local-candidate",
"status": "HELD LOCALLY: validation improved; public leader-beating performance is unproven"
```

The archive with this content is platform submission 322397.

`inference_main.py`, lines 247 to 251, is a comment from an earlier leave-site-out
calibration study (`shift_cal2.py`) that says the temperature is 1.0. Line 252 reads the
temperature from the configuration with 1.0 as the fallback, and the shipped
`assets/inference_config.json` sets `"calibration": {"T": 1.4, "clip": 0.005}`. The scored
package therefore applies T = 1.4 to the stacked logit.

Leftovers without effect on the scored predictions:

- `assets/inference_config.json` has `"tabicl": true`. The inference code never reads this
  entry and always loads the TabICL block.
- The training loop of `cnn2d.py` tries, after each fold, to read an out-of-fold file of an
  earlier champion model that is not in this repository. The attempt sits in a `try` block
  and prints `readout skipped`. Inference never reaches it.
- `cnn2d.py`, `cnn3.py`, `cnn4.py`, `register3.py` and `regfit.py` define `ROOT` or `OUT`
  path constants at module level. Only their training code uses them.
