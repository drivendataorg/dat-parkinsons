# 3rd place (paulonium): retrain verification log

Goal: confirm the submitted training code, run as documented, produces a model equivalent to the
submitted inference package. Full retrain (50 folds) is ~12–25 h here, so the plan is: verify data and
environment, reproduce the submitted predictions with the submitted package, build the training cache,
then train one member × one fold (`fu_a1_dnet` fold 0) in full and compare it with the submitted trace
of the same (member, fold) on the same held-out scans.

Issues that make the submitted code error or that are documented incorrectly are tracked separately in
[`repro_issues.md`](repro_issues.md).

## Setup

**Machine:** 1× NVIDIA L40S 46 GB
(driver 595.91.07, CUDA 13.2), 8 vCPU, 61 GB RAM, 250 GB disk. Author used 3× V100S, 24 cores, 252 GB.

**Environment:** `~/sfmn3/.venv`, Python 3.12.15 via `uv`, `pip install -r "3rd Place/requirements.txt"`.
All pins installed exactly: torch 2.10.0+cu128, monai 1.6.0, numpy 1.26.4, scipy 1.17.1,
nibabel 5.4.2, pandas 2.3.3, scikit-learn 1.9.0. No pretrained weights are downloaded (MONAI
DenseNet121 from scratch).

**Paths:**

| var / item | path |
|---|---|
| `DAT_NIFTIS` (train scans) | `~/dat-parkinsons/niftis` |
| `DAT_LABELS` | `~/dat-parkinsons/train_labels.csv` |
| `DAT_WORK` | `~/sfmn3/work` |
| test set (`DATA_DIR` layout) | `~/sfmn3/data/test` |
| submitted zip, unzipped | `~/sfmn3/submitted` |
| logs | `retrain_notes/logs/` (moved from `~/sfmn3/logs`) |
| helper scripts | `retrain_notes/scripts/` (`compare_fold.py`, `oof_from_traces.py`, `run_notf32.py`; moved from `~/sfmn3/`) |
| test prediction CSVs | not committed (predictions on the private test set) |

All commands run from `3rd Place/` (code reads `meta/*.csv` by relative path).

## 2026-10-07 — data and artifact verification

Source: the organizer's copies of the competition data and the winners' submissions.

- **Submitted zip** `comp_subs/3rd_place/3rd_place_submission.zip`: sha256 `43789dc1…8395` matches
  `hashes/submitted_artifacts.sha256`; `unzip -t` clean; 59-entry listing (names + sizes) identical to
  `notes/zip_listing.txt`. Its `assets/ellipse.ts.pt` is byte-identical to the repo's
  `inference/assets/ellipse.ts.pt` (sha256 `96331128…3b44`).
- **Scored csv** `3rd_place_submission.csv`: sha256 `104184ba…022a` matches.
- **Train scans + labels** (`data/final/public/`): 1,362 `.nii.gz` + `train_labels.csv` all match the
  organizer SHA-1 manifest `data/final/local_data_sha.txt`. uid set == `meta/uids.csv`; 747 abnormal /
  615 normal (as README); labels equal `meta/labels.npy` in cache order and `meta/cv_round2.csv`.
  Every file fully decompresses with a valid 3D NIfTI header.
- **Test set** (`data/final/private/`): 524 scans + `submission_format.csv`, `test_labels.csv`,
  `pub_sub_idx.csv` all match the SHA-1 manifest. Test uids == submitted csv uids; no overlap with train.
- **Score check of the submitted csv:** log loss on all 524 = **0.273213** (= private LB; the private
  score is computed on the full test set). Public-flagged half = 0.231707, which is this package's
  public score per the README (0.2317); the intake's 0.231321 is the team's other submission
  (`fusion10`, slope 0.85).

## 2026-10-07 — environment gates

- `meta/labels.npy` == `train_labels.csv` in `meta/uids.csv` order: **True**.
- `tests/test_t2_augment.py` (augmentation determinism, no data needed): **PASS**, 40/40 seeds
  bit-identical, control alive.

## 2026-10-07 — submitted package re-run on the test set

`cd ~/sfmn3/submitted && DATA_DIR=~/sfmn3/data/test OUTPUT_PATH=~/sfmn3/repro_submitted_test.csv python main.py`

- Loaded 50 members, f32, ellipse anchor; 0 hedged scans; 524 rows. Wall 7 min 29 s, peak RSS 6.3 GB.
- vs the scored csv: max |Δp| 8.6e-4, mean |Δp| 4.1e-5, corr 0.99999998, 54/524 rows exactly equal.
- Log loss **0.273217** (scored: 0.273213), AUC 0.9489 (same).
- Conclusion: the submitted inference package reproduces its scored predictions on this hardware;
  residual differences are GPU floating-point level (L40S here vs the runtime's GPU).

## 2026-10-07 — training cache build

`PYTHONPATH="$PWD" DAT_NIFTIS=… DAT_LABELS=… DAT_WORK=~/sfmn3/work python build/build_canonv2_cache.py`
(the README command without `PYTHONPATH` fails — repro_issues #1).

- Wrote `~/sfmn3/work/boxcache/canonv2.f16.npy` (4.1 GB) and `canonv2mask_ell.u8.npy` (2.1 GB),
  1,362 rows. Exit 0, wall **28 min** (~1 scan/s; single-process CPU preprocessing — load average 1.0
  on 8 vCPU — first ~7 min shared the CPU with the inference re-run). Log `logs/build_cache.log`.
- Why it's required: training reads only these memmaps (`box_cache`, `mask3d_cache`), never the
  niftis; the cache is the batch form of `inference/main.py` preprocessing; it is not shipped (derived
  from scan data); training asserts cache rows == label rows, so it cannot be built on a subset.

## 2026-10-07 — T3 forward equivalence on the rebuilt cache

`DAT_WORK=~/sfmn3/work DAT_MODELS=~/sfmn3/submitted/assets python tests/test_t3_forward_equivalence.py`

- 4 submitted traces (`fu_a10_f0`, `fu_a2_f2`, `fu_a4_f4`, `fu_a7_f1`), 32 cached scans each, weights
  transplanted into the repo's `DatNet` (strict load): max |Δp| **0.000e+00**, corr 1.000000.
  Perturbed-weights control diverges. **T3 PASS.** Submitted traces are f32.
- Shows the submitted weights are products of this repo's model code, and the rebuilt cache feeds them
  correctly.

## 2026-10-07 — fold-0 training run

Exact flags from `recipes/train_fusion10.sh` for member 1 (logs to `logs/` instead of the
repo's `logs/`), started 22:14 UTC:

```
DAT_WORK=~/sfmn3/work python -m datscan --member fu_a1_dnet --out ~/sfmn3/work/runs_fusion10 \
  --folds 0 --workers 2 --gpu 0 --set ndt_anchor=striatal anchor_guard=true rot_deg=31.7 \
  lesion_p=0.25 lesion_lo=0.45 lesion_hi=0.90 box_cache=$W/boxcache/canonv2.f16.npy \
  mask3d_cache=$W/boxcache/canonv2mask_ell.u8.npy sagfuse=attnres sagfuse_aux=0.3 sagfuse_aug=true \
  fuse_lr_mult=0.3 splits=meta/cv_round2.csv seed=1
```

The printed `ARM:` dict is identical to the author's retrain log (`docs/repro/logs/fu_a1_dnet_f0.log`)
apart from cache paths. The original run additionally had `save_swa=True`, which only controls whether
the EMA weights are written to disk, not the best-epoch checkpoint that ships.

## Reference numbers for the fold-0 retrain

`fu_a1_dnet` fold 0 (seed 1, `meta/cv_round2.csv`), best-epoch val log loss as printed in the logs:

| run | best val ll | final-epoch AUC | source |
|---|---|---|---|
| original (produced the submitted trace) | 0.1645 @ ep 82 | 0.9852 | `docs/repro/logs_original/fu_a1_dnet_f0.log` |
| author's post-competition retrain | 0.1717 @ ep 106 | 0.9823 | `docs/repro/logs/fu_a1_dnet_f0.log` |

Original per-fold wall time 51–53 min, author's retrain 58–62 min (9 trainers sharing 3 V100S).

**Result:** exit 0, wall **31.7 min** (~16 s/epoch; GPU utilisation only 6–13% — bound by the 2
DataLoader workers the launcher specifies; left unchanged to stay like-for-like). Peak RSS 5.8 GB.
Log `logs/fu_a1_dnet_f0.log`; outputs in `~/sfmn3/work/runs_fusion10/`.

Best-so-far val log loss vs the two reference logs, same member/fold/seed/split:

| epoch | here | original (submitted) | author's retrain |
|---|---|---|---|
| 10 | 0.2817 | 0.3013 | 0.3033 |
| 20 | 0.2264 | 0.2373 | 0.2277 |
| 30 | 0.2028 | 0.1937 | 0.1983 |
| 40 | 0.1875 | 0.1937 | 0.1881 |
| 60 | 0.1809 | 0.1754 | 0.1775 |
| 70 | 0.1756 | — | 0.1736 |
| final best | **0.1694 @ 108** | 0.1645 @ 82 | 0.1717 @ 106 |

## 2026-10-07 — retrained fold 0 vs submitted `fu_a1_dnet_f0.ts.pt`

`scripts/compare_fold.py`: same 273 held-out training scans (cv_round2 fold 0); submitted trace
re-scored with main.py's flip-TTA; retrained = its dumped best-epoch OOF.

| | val log loss | AUC |
|---|---|---|
| submitted trace | 0.1646 | 0.9845 |
| retrained here | 0.1694 | 0.9830 |

- Agreement: logit corr **0.986**, prob corr 0.993, mean |Δp| 0.025, max |Δp| 0.29, 97.1% of scans on
  the same side of 0.5.
- The submitted trace's 0.1646 equals the original log's best 0.1645 → the submitted file is that run's
  best-epoch checkpoint.
- Gap 0.0048 ll is within the README's per-fold seed noise (0.005–0.009). GPU training is not bitwise
  reproducible (README; also the author's own retrain landed at 0.1717), so "very close", not identical,
  is the expected outcome.

## 2026-10-07 — export + trace verification of the retrained fold

- `build/export_members.py` **fails** on this GPU (TF32 → GPU gate 2.2e-3 > 1e-4; repro_issues #3).
  Re-run through `scripts/run_notf32.py` (TF32 off, script unchanged): `EXPORT fu_a1_dnet f0 cpu 0.00e+00
  gpu 9.54e-07`; then the expected `FileNotFoundError` for fold 1 (not trained). Output
  `~/sfmn3/work/export_fusion10/fu_a1_dnet_f0.ts.pt`.
- `build/verify_traces.py` on it (with `PYTHONPATH` incl. `inference/`, repro_issues #2):
  `TRACECHECK 1 fold-models on 16 held-out scans | worst CPU 0.00e+00 | worst GPU 1.91e-06 | PASS`
  (same with TF32 on or off). Its earlier run on the empty dir printed a vacuous PASS (repro_issues #4).

## 2026-10-07 — retrained fold inside the inference package, test set

Copy of the unzipped submission (`~/sfmn3/swapped`) with only `assets/fu_a1_dnet_f0.ts.pt` replaced by
the retrained export (sha256 prefix `31ecde7f…` → `9d441419…`); 50 members + ellipse.
`DATA_DIR=~/sfmn3/data/test python main.py` → exit 0, 50 modules loaded, 0 hedged, 524 rows, 7 min.

| package (scored on all 524 test scans) | log loss | AUC |
|---|---|---|
| scored csv (official) | 0.273213 | 0.9489 |
| submitted package re-run here | 0.273217 | 0.9489 |
| submitted package, 1 of 50 members retrained here | 0.272910 | 0.9491 |

Swapped vs re-run predictions: max |Δp| 0.0066, mean |Δp| 0.0005, corr 0.999997.

## Conclusion (fold-level check)

- The submitted training code, run as documented (plus the import-path workarounds), trains a member
  that matches the submitted one functionally: 0.1694 vs 0.1646 held-out ll on the same fold, logit
  corr 0.986, within documented seed noise.
- The export path produces a TorchScript trace that the submitted `main.py` loads and runs unchanged,
  and the package's test score is unchanged to the 4th decimal.
- Not run (time): the other 49 folds (~31 min each here, ~26 h sequential), calibration
  (`build/calibrate.py`, needs all 50 OOF dumps) and `build_roster.py` packaging. The author's own
  full retrain reports OOF ll 0.2065 vs 0.2067 for the package (`docs/repro/md5_weights_training_new_dat.txt`).

## 2026-10-08 — calibrate.py + build_roster.py, run on OOF rebuilt from the submitted traces

Full retrain ruled out (time). The repo ships no OOF dumps, so to exercise the last two pipeline steps
the 10 × 5 OOF files were **rebuilt from the 50 submitted fold traces**: each `<member>_f<k>.ts.pt` is
scored (flip-TTA, as in `main.py` / `compare_fold.py`, TF32 off) on its own held-out rows (the recipe's
`cv_round` split for that member, fold k) from the rebuilt training cache, and written in the format
`datscan/train.py` dumps (`<member>__best_oof_{idx,p}_fold<k>.npy`, int64 / float32). Row selection and
order were checked against the real `fu_a1_dnet` fold-0 dump from the retrain: identical.

Script `scripts/oof_from_traces.py`; output `~/sfmn3/work/oof_from_traces/`; log
`logs/oof_from_traces.log`. Wall 3.5 min, peak RSS 7.8 GB. Per-fold val ll 0.150–0.293;
`fu_a1_dnet` f0 = 0.1646, as in the fold-0 comparison.

What this does and does not show: it runs the two scripts end-to-end on real inputs of the right shape
and checks their outputs against the documented numbers. It does **not** retrain anything: the OOF are
the submitted models' own held-out predictions, which is what the author's training run would have
dumped (up to GPU float noise), not a new run's.

### calibrate.py

`python build/calibrate.py --recipe recipes/fusion10_s078.json --runs ~/sfmn3/work/oof_from_traces`
(README command, unchanged) → **exit 0**. Log `logs/calibrate.log`.

```
10 members | OOF AUC 0.9759 (slope-invariant) | logit sd 4.052 | opt_temp 1.035
 slope    intercept   OOF ll
  0.74 -0.152585033   0.2101
  0.78 -0.147954807   0.2067
  0.85 -0.136268607   0.2024
  0.92 -0.120933653   0.1998
  1.00 -0.100128135   0.1985
```

| quantity | here | documented | source |
|---|---|---|---|
| intercept b at slope 0.78 | −0.147955 | −0.147825 | `recipes/fusion10_s078.json` |
| OOF ll at 0.78 | 0.2067 | 0.2067 | `docs/repro/md5_weights_training_new_dat.txt` (package) |
| OOF ll / AUC at 0.85 | 0.2024 / 0.9759 | 0.2023 / 0.9759 | `recipes/fusion10_s078.json` `oof_log_loss_auc` |
| opt_temp | 1.035 | 1.036 | README / `build_roster.py` docstring |

The documented intercept is reproduced to 1.3e-4 (residual = GPU differences in the rebuilt OOF). The
**shipped** `calibration.json` has b = −0.16455, which this step does not produce from the shipped
models' own OOF (a 0.017 gap is ~100× the float residual). This supports finding F5 in
`verification_report.md`: the shipped intercept is not reproducible from the documented procedure, and
the value −0.16455 appears nowhere in the repo. Effect on the score: negligible (see the test-set run below).

### build_roster.py

`python build/build_roster.py --recipe recipes/fusion10_s078.json --assets ~/sfmn3/submitted/assets
--runs ~/sfmn3/work/oof_from_traces --slope 0.78 --out ~/sfmn3/roster/submission.zip` → **exit 0**.
Log `logs/build_roster.log`.

```
ROSTER submission.zip: 10 members, 50 fold-models
ROSTER calibration a=0.78 b=-0.147955 | OOF ll 0.2067 AUC 0.9759 | opt_temp 1.031
ROSTER wrote /home/ec2-user/sfmn3/roster/submission.zip (2.84 GB)
```

Rebuilt package vs the submitted zip (staging dir `~/sfmn3/roster/submission_stage` vs unzipped
submission, sha256 per file):

- Same 57-file list (excluding `__pycache__`).
- Byte-identical: all 50 `*.ts.pt` (hard-linked from the submitted assets, so expected), `ellipse.ts.pt`,
  `canon_const.json`, `canonize.py`, `datprep_box.py`, `datprep_iso.py`, `glandrm.py` (the last five
  copied from the repo's `inference/`).
- Differs: `assets/calibration.json` (b −0.147955 vs −0.164553, as above) and `main.py`. The `main.py`
  diff is exactly the one already recorded as F6 (`notes/main_py_harness_vs_tree.diff`: the repo's
  `inference/main.py` has an extra `ex_*` path); nothing else.

So `build_roster.py` packages the right members and the right inference code; the two differences from
the submitted zip are both known findings, not new ones.

### verify_pkg.sh on the rebuilt zip, and a full test-set run

`DAT_NIFTIS=… DAT_LABELS=… DAT_WORK=~/sfmn3/work bash build/verify_pkg.sh ~/sfmn3/roster/submission.zip`
(README step 5, unchanged) → **exit 0**. Log `logs/verify_pkg_roster.log`.

- Smoke, 32 training scans, fresh unzip: 50 modules f32, 0 hedged, AUC 1.0000, ll 0.1321, wall 100 s.
- Runtime, 64 scans: 119 s. Run twice: sha256 identical → `BIT-IDENTICAL PASS`.

The fresh unzip (`~/sfmn3/work/verify_submission`) was then run on all 524 test scans
(`DATA_DIR=~/sfmn3/data/test python main.py`, exit 0, 0 hedged; log `logs/infer_roster_test.log`):

| package (all 524 test scans) | log loss | AUC |
|---|---|---|
| scored csv (official) | 0.273213 | 0.9489 |
| submitted package re-run here | 0.273217 | 0.9489 |
| rebuilt by `build_roster.py` (b = −0.147955, repo `main.py`) | 0.272992 | 0.9489 |

Rebuilt vs submitted re-run: max |Δp| 0.0041, mean |Δp| 0.0013 (the intercept difference; the repo
`main.py` `ex_*` path doesn't fire). The documented intercept would have scored 0.0002 better than the
shipped one, so the F5 mismatch does not favour the submission.

## Conclusion (pipeline-level check, 2026-10-08)

Every documented step has now been run on this machine: cache build, training (1 of 50 folds),
export + trace verification, calibration, packaging, package verification, test-set inference.
All run to completion; the only failures are the logged repro issues (#1–#4, import paths and TF32),
each with a one-line workaround. Training reproduces a submitted fold within seed noise; calibration
reproduces the documented intercept and OOF numbers from the submitted models' own held-out
predictions; packaging rebuilds the submitted zip apart from two already-known differences (F5, F6).
Not run: the other 49 training folds (out of scope for time).
