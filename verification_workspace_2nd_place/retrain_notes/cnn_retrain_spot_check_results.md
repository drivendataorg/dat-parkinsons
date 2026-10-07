# 2nd place (NavAttack): CNN retrain spot check (stages 05 and 06)

Date: 2026-10-07
Machine: Apple M3 (base, not Pro), 16 GB unified memory, macOS, PyTorch MPS. Competitor used an Apple M3 Pro, 18 GB.
Workspace: `verification_workspace_2nd_place/retrain/`, after the fast-mode run in
`fast_mode_retrain_results.md` (which built the crop caches and replayed the CNN out-of-fold files).

## Summary

Full CNN retraining (52 checkpoints, about 5.7 h on the competitor's machine) was out of scope. Instead,
fold 0 of one 3D run and one 2D run was retrained with the competitor's exact arguments and seeds and
compared with the submitted checkpoints. Total training time was about 15 minutes.

- **3D HemiMIL (`mil48a` fold 0): exact reproduction.** All 38 weight tensors bitwise identical to the
  submitted `mil48a_fold0.pt`, normalisation constants equal, held-out predictions identical.
- **2D EfficientNet-B0 (`e2dmipf2` fold 0): reproduced within floating-point noise.** Same fold-0 log loss
  and AUROC as the original to four decimals. 49 of 360 tensors bitwise identical, the rest differ at
  about 1e-5 relative overall. Held-out predictions differ by at most 2.3e-5.

The CNN training code and recipe produce the submitted models. The 2D difference matches the pattern
seen in stage 02 of the fast-mode run: small GPU rounding differences between a base M3 and the
competitor's M3 Pro. The competitor reported 360/360 identical tensors on their own machine.

## Why a spot check is meaningful here

- Training uses 5 stratified folds (`outputs/folds.csv`, byte-identical to the competitor's). The 52 CNN
  checkpoints are 45 per-fold models plus 7 full-data models:

  | member | runs | fold models | full-data models | checkpoints |
  |---|---|---:|---:|---:|
  | 3D `mil48y` | `mil48a`, `mil48b`, `m31000`, `m33000`, `m35000`, `m37000` (5-fold) + `mf21000`-`mf27000` (full) | 30 | 4 | 34 |
  | 2D B0 `e2dmipf2` | `e2dmipf2`, `e2dmipf22` | 10 | 2 | 12 |
  | 2D B3 `e2db3f2` | `e2db3f2` | 5 | 1 | 6 |

- Each fold is an independent run seeded with `seed_base + k` (`torch.manual_seed`, `np.random.seed` at
  `src/cnn3.py:268` and `src/cnn2d.py:241`), so `--only-fold 0` repeats exactly the run that produced the
  shipped fold-0 checkpoint.
- The input caches `crops_v3_b.npz` and `crops_v3_f2.npz` were byte-identical to the competitor's
  published hashes (fast-mode run, stage 04).
- The two runs cover the 3D member (the largest CNN weight in the stacker, 34 of 52 checkpoints) and the
  B0 2D member. The B3 member was not retrained; it uses the same `cnn2d.py` code path with a larger
  backbone (about 13 min per fold on the competitor's machine).
- Epochs were not reduced, since a shortened run would not be comparable with any submitted checkpoint.

## Commands

From `retrain/`. Arguments and environment copied from `pipeline/05_cnn3d.sh` and `pipeline/06_cnn2d.sh`.
The `chk_` tags write new files and do not touch the staged submitted checkpoints.

```bash
PY="$PWD/.venv-train/bin/python"
export KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=2

# 3D: mil48a fold 0 (seed 3000)
(cd src && caffeinate -i "$PY" -u cnn3.py --tag chk_mil48a --arch mil --size 48 --key aligned \
   --crops crops_v3_b.npz --lr 1e-3 --epochs 20 --seed-base 3000 --only-fold 0) | tee outputs/logs/chk_mil48a.log

# ImageNet backbones (network, one time, sha256-pinned)
"$PY" tools/fetch_backbones.py

# 2D: e2dmipf2 fold 0 (seed 9000)
(cd src && HF_HUB_OFFLINE=1 caffeinate -i "$PY" -u cnn2d.py --tag chk_e2dmipf2 --crops crops_v3_f2.npz \
   --key aligned --views mip3 --res 192 --epochs 30 --lr 1e-4 --aug 1.0 --seed-base 9000 --only-fold 0) \
   | tee outputs/logs/chk_e2dmipf2.log

# comparisons
"$PY" tools/verify_cnn.py spot --ref-dir outputs
"$PY" -I ../retrain_notes/compare_cnn_spot.py
```

`--ref-dir outputs` points `verify_cnn.py` at the out-of-fold files replayed from the submitted
checkpoints during the fast-mode run (competitor-reported agreement with the training-time originals
about 1e-7 to 1e-5). `compare_cnn_spot.py` (in this folder) compares the retrained and submitted
checkpoints tensor by tensor; the 2D reference is the raw state dict `models/cnn_raw/e2dmipf2_fold0.pt`.

The 2D run prints `readout skipped: FileNotFoundError(...)` after training. This is a harmless optional
diagnostic (`src/cnn2d.py:369-378`, inside `try/except`) that looks for
`champion_v7_CV0.2139/oof_feat3.csv`, a folder from the competitor's development tree that is not part
of the release. The checkpoint is saved before it runs.

## Results

| | 3D `mil48a` fold 0 | 2D `e2dmipf2` fold 0 |
|---|---|---|
| wall time on the M3 | 423 s | 455 s |
| competitor's time (M3 Pro) | 367 s | 337 s |
| fold-0 log loss / AUROC | 0.2544 / 0.9576 | 0.2773 / 0.9516 |
| original fold 0 | 0.2544 / 0.9576 | 0.2773 / 0.9516 |
| comparison runs, fold 0 | other 3D seeds: 0.2496-0.2702 | `e2dmipf22` 0.2814 / 0.9538, `e2db3f2` 0.3137 / 0.9425 |
| weight tensors bitwise equal | 38 / 38 | 49 / 360 |
| max abs weight difference | 0 | 1.7e-02 (BatchNorm `running_var`, about 1e-5 relative) |
| relative L2 difference, all float weights | 0 | 9.6e-06 |
| normalisation constants mu / sd | equal | n/a |
| held-out predictions vs original | identical (max diff 0.0) | max diff 2.3e-05, mean 3.6e-06, logit corr 0.9999999999 |
| held-out scans with diff > 0.05 | 0 | 0 |

### 2D difference detail

The 311 non-identical 2D tensors are 131 weights, 82 biases, 49 BatchNorm running means and 49 BatchNorm
running variances. The largest absolute differences are in BatchNorm running variances (for example
`blocks.3.0.bn1.running_var`, max diff 1.7e-02, relative 1.3e-05). The largest relative differences
(up to 13%) are in tensors with very small values, such as `blocks.1.1.bn1.running_mean` (max value
5.2e-03, max diff 7.1e-04) and `blocks.1.0.bn3.bias` (max value 4.4e-03, max diff 5.1e-04). None of this
changes the fold-0 metrics at four decimals.

## Conclusion

Rerunning the competitor's CNN training recipe reproduces the submitted models: exactly for the 3D
member and within floating-point noise for the 2D B0 member. Combined with the fast-mode run (stages
00-04 and 07-09), every stage of the training pipeline has now been exercised from raw data, and the
results match the submitted package closely, with all deviations explained by documented
hardware-dependent GPU rounding.

Not checked: retraining the B3 member (`e2db3f2`) and the other 50 CNN checkpoints.

## TorchScript re-export check

Inference loads the 2D members as TorchScript (`*.ts.pt`), produced from the raw checkpoints by
`tools/ts_export.py`. Neither the fast-mode run (which staged the submitted `.ts.pt` files) nor the
spot retrain exercised this step. The 18 raw checkpoints from the supplemental `cnn_raw/` bundle were
copied to `outputs/chk_orig_<tag>_<part>.pt` and exported, then compared with the submitted `.ts.pt`
files in `inference/assets/` (installed from the scored zip):

```bash
for t in e2dmipf2 e2dmipf22 e2db3f2; do
  for p in fold0 fold1 fold2 fold3 fold4 full0; do cp models/cnn_raw/${t}_${p}.pt outputs/chk_orig_${t}_${p}.pt; done
  OMP_NUM_THREADS=3 "$PY" tools/ts_export.py chk_orig_$t
done
"$PY" tools/verify_cnn.py ts --assets inference/assets \
  --pairs chk_orig_e2dmipf2=e2dmipf2 chk_orig_e2dmipf22=e2dmipf22 chk_orig_e2db3f2=e2db3f2 \
  --out-name chk_verify_ts_reexport.json
```

| | e2dmipf2 (B0) | e2dmipf22 (B0) | e2db3f2 (B3) |
|---|---|---|---|
| export time (CPU) | 37 s | 36 s | 89 s |
| exporter's eager vs traced check, max logit diff | 9.1e-06 to 2.3e-05 | 1.4e-05 to 2.7e-05 | 1.7e-05 to 3.6e-05 |
| re-exported vs submitted `.ts.pt`, max abs logit diff on 256 images | 0.0 (6/6 files) | 0.0 (6/6 files) | 0.0 (6/6 files) |
| file sha256 equal | no | no | no |

All 18 re-exported TorchScript files give logits identical to the submitted ones. File hashes differ
because TorchScript archives record source paths, as documented by the competitor
(`docs/stages/04-06_cnn.md`). The comparison took 543 s; results in
`retrain/outputs/chk_verify_ts_reexport.json`.

This confirms the submitted 2D inference files, including all six B3 files, are faithful exports of
the raw checkpoints in the supplemental `cnn_raw/` bundle. It does not show that B3 training
reproduces those raw checkpoints.
