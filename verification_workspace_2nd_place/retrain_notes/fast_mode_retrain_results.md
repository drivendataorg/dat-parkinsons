# 2nd place (NavAttack): fast-mode retrain and inference parity

Date: 2026-10-07
Machine: Apple M3 (base, not Pro), 16 GB unified memory, macOS. Competitor used an Apple M3 Pro, 18 GB.
Workspace: `verification_workspace_2nd_place/retrain/` (clean copy of `dat-park-navattack-code-archive`).

## Summary

`MODE=fast bash pipeline/train_all.sh` ran end to end without errors in 2,268 s (38 min) and produced
a valid submission package. The package is not bit-identical to the scored archive. The extra
differences beyond the ones the competitor predicted all trace back to stage 02 (registered
features), which runs image registration on the MPS GPU. The competitor's docs only claim byte
identity for that step on their M3 Pro and warn of small drift on other devices. On 64 private test
scans, the rebuilt package's predictions differ from the scored package's by at most 0.024 (mean
0.004), which is smaller than the difference between the scored package run locally on CPU and the
platform's own scored CSV (max 0.031). Log loss on those 64 scans: 0.2318 rebuilt vs 0.2315 scored
package (local) vs 0.2306 platform CSV; AUROC 0.9696 for all three.

Not covered by this run: CNN training (stages 05 and 06). Fast mode copies the submitted CNN
checkpoints and only replays their out-of-fold predictions. The fold-0 CNN spot retrains are still
outstanding.

## Setup

- Venvs `.venv-train` (torch 2.13.0, scikit-learn 1.9.0, numpy 2.5.1) and `.venv-infer`
  (torch 2.12.1, scikit-learn 1.8.0, numpy 2.2.6) in `retrain/`, Python 3.12.11 via uv.
- Weights installed from the scored archive
  `references/submissions/comp_subs/2nd_place/2nd_place_submission.zip`
  (sha256 `88b5611e25170d88c2fbf21b62443dbc67b722c519b5a0391a60aab62bbec7fd`, matches the README).
  `install_weights.py --verify-only`: 60/60 assets verified; `.ts.pt` files match `SHA256SUMS.scored.txt`.
- Raw 2D checkpoints: `navattack-supplemental-materials/cnn_raw/` copied to `models/cnn_raw/` (18 files).
- Data staged by `pipeline/00_stage_data.sh` from `references/data/public/niftis.zip` and
  `train_labels.csv`. Both sha256 values match the competition-download hashes pinned in the script.
  1,362 scans, 1,362 label rows, 0 missing.

Command (from `retrain/`):

```bash
caffeinate -i env \
  CHECK_AGAINST=/Users/matt/dd/dat-parkinsons/references/submissions/comp_subs/2nd_place/2nd_place_submission.zip \
  PYTHON="$PWD/.venv-train/bin/python" MODE=fast bash pipeline/train_all.sh
```

Log: `retrain/outputs/logs/train_all.log`.

## Stage-by-stage comparison

Reference values come from `docs/REPRODUCIBILITY.md`, `docs/stages/README.md` and the competitor's
`MODE=full` logs in `docs/repro/mode_full_2026-09-28/`.

| stage | time here | result |
|---|---:|---|
| 00 data | 0 s | OK |
| 01 folds | 3 s | `folds.csv` byte-identical (`550e2c0ab72e840c`) |
| 02 registered features | 529 s | mean NCC 0.7889, 2 rejected, 0 failed (all equal to reference). Every feature-table hash differs (e.g. `features_v13_m2fnh.csv` `099ae789…` vs `b1729a46…`). `feature_order.json` and `feature_union.json` identical |
| 03 linear members | 19 s | `lin` OOF ll 0.2323 / AUC 0.9651 (ref 0.2322 / 0.9650), 49 of 235 nonzero (ref 48). `lin4` 0.2381 / 0.9633 (ref 0.2378 / 0.9634), 38 of 94 nonzero (ref 40) |
| 04 crop caches | 1,046 s | `crops_v3_b.npz` (`ba2d413e…`) and `crops_v3_f2.npz` (`0a1fd5cf…`) byte-identical to the competitor's published hashes |
| 05/06 fast: staging + OOF replay | 180 s | 70 checkpoint files staged. Replayed OOF metrics match reference (e.g. `mil48y` ll 0.2611, `e2dmipf2x` 0.2825, `e2db3f2` 0.2808) |
| 07 TabICL | 448 s | groups and shape/mixture features ran. Selector nonzero 65 (ref 63). 61 of 64 selected features shared with the scored recipe. Contexts frozen with `ensemble_order ['none','power']`, `PYTHONHASHSEED=1` |
| 08 stacker | 20 s | cross-fit T=1 ll 0.2069 / AUC 0.97229 (ref 0.2070 / 0.97228). coef `[0.5318, 0.2127, 0.3722, 0.1595, 0.1595]`, intercept 0.1857 (scored `[0.5312, 0.2125, 0.3718, 0.1594, 0.1594]`, 0.1843). Logging gate OK |
| 09 package | 22 s | static gates PASSED, 82 entries, same order and compression |

### Linear model detail (rebuilt vs scored assets)

| model | nonzero rebuilt / scored | support difference | coef correlation | max abs coef diff | intercept |
|---|---|---|---:|---:|---|
| `lin` | 49 / 48 | rebuilt adds `h_spec_rel` | 0.9999 | 0.016 | 1.1372 vs 1.1358 |
| `lin4` | 38 / 40 | rebuilt drops `p97.0`, `aput_R_p90` | 0.9971 | 0.649 | 1.5221 vs 1.5151 |

Preprocessing statistics (median, mean, std) differ by at most about 0.2% relative.

### TabICL recipe detail

Selected features only in scored: `g07_better`, `shape__s1.0_curve1_max`, `shape__s3.0_t0.2_elong_max`.
Only in rebuilt: `h_p90_better`, `pc_ratio_L`, `ppu_R_p90`. `log_prep`, `median` and `priors` differ
accordingly. The final `feature_order` (479 names) differs at 136 positions.

### Archive comparison (`--check-against` the scored zip)

76 of 82 entries equal in CRC32 and size. Mismatches:

| entry | expected by competitor's docs? |
|---|---|
| `assets/inference_config.json` | yes (stacker constants) |
| `assets/tabular_raw.pkl.xz` | yes (pickle bytes change every run) |
| `assets/tabular_log.pkl.xz` | yes |
| `assets/linear_lin.json` | no; follows from stage 02 drift |
| `assets/linear_lin4.json` | no; follows from stage 02 drift |
| `assets/tabular_recipe.json` | no; follows from stage 02 drift |

The competitor's own clean fast-mode run on the M3 Pro got 79 of 82 (only the first three).

### Cause of the drift

`src/make_regfeat32.py:91-93` selects MPS for the registration fit. `docs/REPRODUCIBILITY.md` §2.1
states byte identity is shown for MPS on the M3 Pro only and that other devices should expect
feature drift "and possibly a slightly different L1 support". Stage 04, which does not use that
registration path, reproduced byte-identically here, which is consistent with the GPU registration
being the source. The joblib "A worker stopped while some jobs were given to the executor" warning in
stage 02 also appears in the competitor's reference logs.

## Inference parity on 64 private test scans

- Subset: the first 64 uids of `references/data/private/submission_format.csv`, with symlinked NIfTIs
  (one inference block, since the code batches up to 128 scans).
- Both packages run with `.venv-infer` on CPU via `main.py` with `DATA_DIR` set, as the competition
  runtime does:
  - scored: `retrain/inference/` (code byte-identical to the scored archive, weights from the scored zip)
  - rebuilt: `retrain/outputs/v17/submission/`
- Wall time about 3 min 50 s per package. No fallback or error warnings in either log.

| comparison | max abs diff | mean abs diff | median abs diff | scans with diff > 0.01 |
|---|---:|---:|---:|---:|
| scored package (local CPU) vs platform CSV | 3.09e-02 | 1.10e-03 | 1.28e-04 | 1 |
| rebuilt vs scored package (local CPU) | 2.44e-02 | 4.11e-03 | 1.50e-03 | 8 |
| rebuilt vs platform CSV | 2.72e-02 | 4.54e-03 | 2.00e-03 | 10 |

| predictions | log loss | AUROC |
|---|---:|---:|
| platform CSV | 0.23060 | 0.96961 |
| scored package, local CPU | 0.23146 | 0.96961 |
| rebuilt package | 0.23181 | 0.96961 |

One scan crosses 0.5 (platform 0.510, scored local 0.509, rebuilt 0.492). The metric is log loss, so
this has no special significance.

The local-vs-platform gap for the unchanged scored package (max 0.031) is larger than the competitor's
documented order and batch effects (max 1.3e-3), probably because the platform ran on an A100 GPU
over the full test set while this ran on CPU over a 64-scan block. The competitor notes that CPU vs
CUDA differences were never measured.

## Conclusion so far

Stages 00 to 04 and 07 to 09 run as described and rebuild a working package from raw data. The
retrained non-CNN components are very close to the submitted ones but not identical on this
hardware, with a prediction-level effect smaller than the CPU-vs-platform noise of the submitted
package itself. This is consistent with the competitor's documented reproducibility caveats.

## Outstanding

- Done 2026-10-07, see `cnn_retrain_spot_check_results.md`: CNN spot retrains (fold 0 of `mil48a` and `e2dmipf2`), compared tensor by tensor against
  `inference/assets/mil48a_fold0.pt` and `models/cnn_raw/e2dmipf2_fold0.pt`. Reference fold-0 log loss
  0.2544 and 0.2773. Commands in `docs/REPRODUCIBILITY.md` §3.2.
- Optional: `pipeline/smoke_cnn.sh` (one epoch of every CNN training command).
