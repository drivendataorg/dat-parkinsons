# Stage 07: the TabICL block

Script: `pipeline/07_tabular.sh`, which runs the scripts in `tabular/`. Shared conventions
are described in [README.md](README.md).

## Purpose

Stage 07 rebuilds every tabular asset of the scored v17 archive from the raw data and the
outputs of stages 01 to 04:

- the acquisition groups: 6 scanner-geometry families (`fam2`) crossed with a low or high
  intensity scale, 10 groups in all
- 180 scale-space shape columns (`shape__`) and 60 spatial Gaussian-mixture columns
  (`mixture__`)
- `tabular_recipe.json`: 64 selected columns, 4 acquisition descriptors, 4 linear prior
  scores, the log-transform parameters (`log_prep`) and the column medians
- the 72-column context matrix
- the 479-name `feature_order` that stage 09 writes into `inference_config.json`
- the two frozen TabICL contexts, `tabular_raw.pkl.xz` and `tabular_log.pkl.xz`

The original chain and the port of each step:

| original (original development tree) | port |
|---|---|
| `tools/site_diag.py` plus an interactive step on 2026-09-03 that added `fam2` and `scale` and was never saved | `tabular/derive_groups.py` (rules recovered from the original development session logs) |
| `experiments/20260914/scale_features.py` (entry point, default `--frame f2`) | `tabular/extract_shape_features.py` |
| `experiments/v17/mixture_shape.py` (`main()`) | `tabular/extract_mixture_features.py` |
| `experiments/v17/prepare_candidate.py`, with `inputs` and `weights` from `experiments/v16/bench.py` and `load_data` from `experiments/20260914/common.py` | `tabular/prepare_tabular.py` |
| `experiments/v17/freeze_candidate.py` | `tabular/freeze_tabular.py` |
| none (verification only) | `tabular/verify_features.py`, `tabular/verify_prepare.py`, `tabular/verify_freeze.py` |

The feature scripts import the per-scan functions from the shipped
`inference/scale_features.py` and `inference/mixture_features.py`, so training and
inference use the same code. Both files are byte-identical to the research code once the
command-line block is removed (`scale_features`) or the slice used by
`prepare_candidate.py` is taken (`mixture_features`).

## Commands

From the repository root, with the weights installed by `scripts/install_weights.py`:

```bash
bash pipeline/07_tabular.sh --ckpt inference/assets/tabicl-classifier-v2-20260212.ckpt \
  --ref-assets inference/assets
```

`train_all.sh` runs `07_tabular.sh --ckpt "$TABICL_CKPT"`, where `TABICL_CKPT` defaults to
`inference/assets/tabicl-classifier-v2-20260212.ckpt`.

| option | default | effect |
|---|---|---|
| `--inputs regenerated\|archived` | `regenerated` | `archived` reads the feature tables, `scale_f2.csv` and `mixture_shape.csv` of the original run from `--archive-dir`, skips step 2 and writes to `outputs/tabular/archived_inputs/` |
| `--archive-dir DIR` | none | a copy of the original development tree. Needed for `--inputs archived` and for the comparisons with archived intermediate files. |
| `--linear-assets DIR` | `submission_src/assets` | where `linear_lin.json` and `linear_lin4.json` (stage 03) are read |
| `--ckpt FILE` | `$TABICL_CKPT`, else `inference/assets/tabicl-classifier-v2-20260212.ckpt` | the public TabICLv2 checkpoint |
| `--ref-assets DIR` | none | the `assets/` of the scored archive (after `install_weights.py`, `inference/assets`). Turns on the recipe sha256 and feature-order checks. The `context.npz` check also needs `--archive-dir`. |
| `--skip-freeze` | off | stop before step 4 |

The script unsets every `DAT_*` variable except `DAT_DATA_DIR` and exports
`KMP_DUPLICATE_LIB_OK=TRUE`, `PYTORCH_ENABLE_MPS_FALLBACK=1`, `PYTHONDONTWRITEBYTECODE=1`
and `OMP_NUM_THREADS=3` (unless already set). It runs the freeze step with
`PYTHONHASHSEED=1` unless the variable is already set (see "Deviations and limits").

The steps in regenerated mode:

```bash
# 1/4 acquisition groups
python tabular/derive_groups.py --native-spacing-csv outputs/features_v13_m2f.csv \
    --out outputs/tabular/acquisition_groups.csv
#     runs, only when their outputs are missing:
#       python src/eda_real.py                           -> outputs/eda_real_geometry.csv, eda_real_summary.json
#       python tools/hdr_fingerprint.py data/raw/niftis  -> outputs/hdr_fingerprint.csv
#     (LOKY_MAX_CPU_COUNT=3 caps the joblib workers of eda_real.py, and --force-scan re-runs both)
# 2/4 shape and mixture features (need outputs/crops_v3_f2.npz from stage 04)
python tabular/extract_shape_features.py --frame f2 --out-dir outputs/tabular
python tabular/extract_mixture_features.py --out-dir outputs/tabular
# 3/4 recipe, context and feature order
python tabular/prepare_tabular.py --inputs-dir outputs --groups-csv outputs/tabular/acquisition_groups.csv \
    --shape-csv outputs/tabular/scale_f2.csv --mixture-csv outputs/tabular/mixture_shape.csv \
    --linear-assets submission_src/assets --out-dir outputs/tabular
# 4/4 frozen TabICL contexts
PYTHONHASHSEED=1 python tabular/freeze_tabular.py --recipe outputs/tabular/tabular_recipe.json --ckpt <ckpt>
```

With `--ref-assets`, the script runs `tabular/verify_prepare.py` after step 3. With
`--archive-dir`, it also passes `--verify-against ORIGINAL_TREE/outputs/site_diag_rows.csv`
to step 1 and compares the step 2 tables with the archived ones through
`tabular/verify_features.py`.

The prediction checks run separately. They print aggregates only:

```bash
mkdir -p outputs/tabular/verify
python tabular/verify_freeze.py --assets inference/assets --save outputs/tabular/verify/pred_shipped.npz
python tabular/verify_freeze.py --assets outputs/tabular \
    --ckpt inference/assets/tabicl-classifier-v2-20260212.ckpt --save outputs/tabular/verify/pred_regen.npz
python tabular/verify_freeze.py --compare outputs/tabular/verify/pred_regen.npz outputs/tabular/verify/pred_shipped.npz
python tabular/verify_freeze.py --compare-states outputs/tabular/tabular_raw.pkl.xz inference/assets/tabular_raw.pkl.xz
```

In prediction mode `verify_freeze.py` rebuilds the (1362, 479) training matrix in
`feature_order`. This matrix is the input of `inference/tabular_adapter.py`
(`TabularAdapter.predict`), which picks the 68 recipe columns and recomputes the 4 prior
scores from it. The script loads the shipped adapter on the CPU, predicts in chunks of 128
rows and checks the first chunk against `adapter.predict()`.

## Inputs and outputs

| input | produced by |
|---|---|
| `data/raw/niftis/*.nii.gz`, `data/raw/labels.csv` | competition data (read by `eda_real.py` and `hdr_fingerprint.py`) |
| `outputs/features_v13_m2fnh.csv`, `features_v4_m2fnh.csv`, `features_v13_m2f.csv`, `folds.csv` | stages 01 and 02 |
| `outputs/crops_v3_f2.npz` (keys `aligned`, `uids`) | stage 04 |
| `submission_src/assets/linear_lin.json`, `linear_lin4.json` (235 and 94 names) | stage 03 |
| `tabicl-classifier-v2-20260212.ckpt` (sha256 `bdc7dbd5…`) | Hugging Face `jingang/TabICL`, revision `4dcd344e`, installed into `inference/assets/` |

Outputs in `outputs/tabular/`:

- `acquisition_groups.csv` (uid, fam2, scale) and `acquisition_groups_info.json`
- `scale_f2.csv` (uid and 180 columns) and `mixture_shape.csv` (uid and 60 columns). As in
  the original, the column names carry no `shape__` or `mixture__` prefix.
- `tabular_recipe.json`, `context.npz` (X of 1362 × 72, y, the balanced subset of 1,148
  rows, the 72 medians) and `feature_order_v17.json` (a JSON list of 479 names)
- `tabular_raw.pkl.xz`, `tabular_log.pkl.xz` and `persistence_checks.json`
- `prepare_info.json`, `verify_*_report.json` and `verify/` (per-row prediction files,
  which stay local)

The stage also writes `outputs/eda_real_geometry.csv`, `eda_real_summary.json` and
`hdr_fingerprint.csv`. Logs in `outputs/logs/`: `eda_real.log`, `hdr_fingerprint.log`,
`tabular_scale_f2.log`, `tabular_mixture_shape.log` and `tabular_freeze_<mode>.log`.

## Runtimes

Measured on an Apple M3 Pro shared with other jobs (CPU only), so the times are upper
bounds.

| step | measured | original run |
|---|---:|---:|
| `eda_real.py`, `hdr_fingerprint.py` and groups (skipped when their outputs exist) | 92 s | |
| shape features | 68-71 s | 61 s |
| mixture features (3 processes) | 8-11 s | 7 s |
| prepare | 3-5 s | not logged |
| freeze, raw and log contexts | 197-329 s (raw alone 102-177 s) | 159 s (raw alone 80 s) |
| `07_tabular.sh`, regenerated mode, image scans already cached | 4 min 42 s to 4 min 47 s | |
| `verify_freeze.py` prediction mode (1,362 rows, both contexts) | 65-88 s per assets directory | |

## Verification

Two passes were run. The first used archived inputs (the original tree's `outputs/`,
`experiments/20260914/scale_f2.csv` and `experiments/v17/mixture_shape.csv`). The second
ran end to end on regenerated inputs: the feature tables and folds of stages 01 and 02
(byte-identical to the archive), `crops_v3_f2.npz` rebuilt by stage 04, and features,
groups and recipe regenerated by stage 07. Both passes gave the results below.

- **Image scans.** `eda_real_geometry.csv` (`2205c3b9…`), `eda_real_summary.json`
  (`5826740b…`) and `hdr_fingerprint.csv` (`eac21254…`) are sha256-identical to the
  archived files.
- **Groups.** `fam2` and `scale` agree with the archived `site_diag_rows.csv` on 1,362 of
  1,362 scans, and both cross-tabulations are diagonal. Group sizes:

  | fam2 | scans | | scale | scans |
  |---|---:|---|---|---:|
  | s246 | 528 | | high | 489 |
  | thin39 | 255 | | low | 873 |
  | s23 | 207 | | | |
  | small | 153 | | | |
  | s24 | 149 | | | |
  | full39 | 70 | | | |

  The 10 groups: s246/high 270, s246/low 258, thin39/low 239, s23/low 207, s24/low 142,
  small/high 126, full39/high 70, small/low 27, thin39/high 16, s24/high 7.

  `native_spacing` in `features_v13_m2f.csv` equals the `qc_v3_m2.csv` column that
  `site_diag.py` used on all 1,362 scans. `acquisition_groups.csv` has the same sha256
  (`1b704a02…`) whether it is built from the archived or the regenerated `native_spacing`.
- **Shape and mixture features.** `scale_f2.csv` (`d5b6c473…`) and `mixture_shape.csv`
  (`4531e1be…`) are sha256-identical to the archived files: same columns, same uid order,
  max |diff| 0.
- **Recipe.** `tabular_recipe.json` has sha256 `211f53509c49…`, identical to the shipped
  file. The four priors have 48, 36, 63 and 66 nonzero coefficients, and the column
  selector has 63.
- **Context.** `context.npz` has sha256 `8765edc84fc4…`, identical to the archived
  `experiments/v17/rebuild/context.npz`. X, y, the balanced subset and the medians are
  bitwise equal.
- **Feature order.** `feature_order_v17.json` equals the `feature_order` of the shipped
  `inference_config.json` (479 names). The list grew in stages: 235 names (linear members),
  261 (v15a selection), 267 (v16 selection), 479 (all candidate and mixture columns).
- **Frozen contexts.** The `.pkl.xz` files differ from the shipped ones in bytes, as
  expected (see "Deviations and limits"). The final run gave raw `7b4c5350…`
  (229,981,844 bytes) and log `3a4b1e78…` (226,759,916 bytes). The shipped files are raw
  `167455ac…` (229,985,568 bytes) and log `af158889…` (226,742,860 bytes). A leaf-by-leaf
  comparison of the states found:
  - all 14 KV-cache arrays per context (column-cache keys and values and `row_repr`, for
    both ensemble groups) bitwise equal
  - 29 of 36 arrays bitwise equal, and the other 7 (scikit-learn preprocessing statistics)
    within 1.8e-15
  - no scalar differences apart from the recorded numpy, scikit-learn and torch versions
- **Predictions through the shipped `TabularAdapter`** (1,362 training rows, 479-column
  input): max |Δp| = 0.0 between regenerated and shipped states, for the raw context, the
  log context and the blended tabular probability. In-sample log loss 0.113362 and AUROC
  0.993002 for both.
- **Platform versions.** The prediction check was repeated in an environment with the
  platform's versions (numpy 2.2.6, scikit-learn 1.8.0, torch 2.12.1, pandas 3.0.3) and
  scipy 1.17.1. The platform ran scipy 1.15.3, whose macOS arm64 wheels fail to load on
  recent macOS, so this check approximates the platform environment. The regenerated
  states still gave max |Δp| 0.0 against the shipped states and against the predictions of
  the training environment, so pickles written under scikit-learn 1.9.0 load and predict
  under scikit-learn 1.8.0.
- **Reload check.** Reloading each context and predicting the probe rows `A[::37]` gives a
  difference of 0.0 for both. `inference/` stays unchanged: its 22 files are byte-equal to
  the scored archive and no `__pycache__` appears.

## Deviations and limits

- **Group rules.** No saved script contains the `fam2` and `scale` rules. They were
  recovered from the original development session logs:
  - `fam2`: `thin39` if the header slice spacing `pixdim3` rounds to 3.9 and the scan has
    fewer than 60 slices, `full39` for the other 3.9 scans, `s246`, `s23` or `s24` when the
    native spacing rounded to 2 decimals is close to 2.46, 2.3 or 2.4, `small` for the
    rest
  - `scale`: `low` if log10(p99 + 1) < 2.25, else `high`

  `tools/site_diag.py` itself is not run, because it needs OOF files that only feed its
  diagnostics. `derive_groups.py` performs the same merges (`eda_real_geometry`,
  `hdr_fingerprint` `pixdim3`, `native_spacing`). It reads `native_spacing` from
  `features_v13_m2f.csv`, where the original read `qc_v3_m2.csv`. The values are
  identical.
- **Hash seed of the freeze.** tabicl 2.2.0 orders its ensemble groups with
  `list(set(norm_methods))`, so the order follows Python's string hash seed and decides
  the float32 summation order of the ensemble average. The original freeze set no seed and
  got `['none', 'power']`. An unseeded rerun got `['power', 'none']`, which moved the
  predictions by up to 6.9e-7 (mean 4e-8). `07_tabular.sh` therefore runs the freeze with
  `PYTHONHASHSEED=1`, which gives `['none', 'power']`. `freeze_tabular.py` records the order
  in `persistence_checks.json` and warns when it differs.
- **Bytes of the frozen contexts.** The `.pkl.xz` bytes differ from run to run because
  torch's tensor pickling embeds a per-process serialisation id. A tensor pickled in two
  processes gives different bytes. Their hashes therefore never match the shipped files,
  and the checks above compare contents and predictions.
- **Library versions.** The freeze ran under numpy 2.5.1, scikit-learn 1.9.0, torch 2.13.0
  and scipy 1.18.0. The original freeze used overlay directories with the platform's
  versions, which were later deleted. Loading and prediction under the platform versions
  were checked (see above). Freezing inside the platform environment was not tried.
- **scikit-learn 1.9 warnings.** scikit-learn 1.9 emits a `FutureWarning` that `penalty`
  is deprecated, and a warning "Inconsistent values: penalty=l1 with l1_ratio=0.0". The
  fits remain L1: the nonzero counts and the recipe are identical. scikit-learn 1.10 will
  remove `penalty`.
- **Selection tie-breaks.** The 64th selected column (index 63) has a zero L1
  coefficient: the selector has 63 nonzero and 352 zero coefficients. The v15a
  feature-order step likewise takes 5 zero-coefficient columns, since its selector has 59
  nonzero. These picks follow the tie order of numpy's default `argsort` (introsort).
  `kind='stable'` would pick different columns. They reproduce on arm64 with numpy 2.5.1 on
  the same Mac as the original. On x86, numpy's AVX2 or AVX-512 SIMD sort may break the
  ties differently, which would change `features[63]` (and with it `median[63]`,
  `log_prep[63]` and `X[:, 63]`) and part of `feature_order[235:261]`. `prepare_info.json`
  reports `selected_zero_coef_positions`, and `verify_prepare.py` detects such a change
  against the shipped recipe. The code keeps the original `argsort` on purpose.
- **Narrow table alignment.** `prepare_candidate.py` used the narrow table by row
  position. The port aligns it on uid, which gives the same numbers when the row orders
  match. `prepare_info.json` records `narrow_row_order_matches_rich: true`. Row order still
  matters in two places: the balanced subset draws positions with `default_rng(16000)`
  within each sorted group, and X follows the row order of `features_v13_m2fnh.csv`.
- **Omitted packaging steps.** The original stage copy, the patches to
  `inference_main.py` and `tabular_adapter.py`, and `CANDIDATE.json` are not reproduced,
  because `inference/` already ships the patched code. `feature_order_v17.json` carries
  the only change to `inference_config.json`, which stage 09 applies.
- **Base feature order.** The first 235 names of `feature_order` default to the
  `linear_lin.json` names. They equal the `feature_order` of the v14c archive.
  `prepare_tabular.py --base-feature-order` overrides them.
- **Checkpoint location.** Without `--ckpt` or `TABICL_CKPT`, both `07_tabular.sh` and
  `tabular/freeze_tabular.py` use `inference/assets/tabicl-classifier-v2-20260212.ckpt`,
  where `install_weights.py` installs it. The commands above pass `--ckpt` only to make the
  path explicit.
- **Archived mode.** `--inputs archived` and every `--archive-dir` comparison need a copy
  of the original development tree, which is not distributed.
