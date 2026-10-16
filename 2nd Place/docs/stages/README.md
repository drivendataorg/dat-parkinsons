# Stage reference

One document per group of training stages. Each gives the purpose of the stages, the exact
commands (the pipeline script and the original command it reproduces), inputs and outputs,
measured runtimes, verification results and known deviations.

| stages | document | scripts | main outputs |
|---|---|---|---|
| 00-03 | [00-03_features_and_linear.md](00-03_features_and_linear.md) | `00_stage_data.sh`, `01_folds.sh`, `02_features.sh`, `03_linear.sh` | `data/raw/`, `outputs/folds.csv`, registered feature tables, `oof_feat3_m2fnh.csv`, `linear_lin.json`, `linear_lin4.json` |
| 04-06 | [04-06_cnn.md](04-06_cnn.md) | `04_caches.sh`, `05_cnn3d.sh`, `06_cnn2d.sh`, `smoke_cnn.sh`, and in `MODE=fast` `tools/replay_cnn_oof.py` | two crop caches, 34 3D checkpoints, 18 2D checkpoints and their TorchScript exports, three OOF files for the stacker |
| 07 | [07_tabular.md](07_tabular.md) | `07_tabular.sh` | `outputs/tabular/`: acquisition groups, shape and mixture features, recipe, feature order, frozen TabICL contexts |
| 08-09 | [08-09_stack_and_package.md](08-09_stack_and_package.md) | `08_stack.sh`, `09_package.sh` | `submission_src/assets/`, `outputs/v17/submission.zip` and `MANIFEST.json` |

## Running the stages

`pipeline/train_all.sh` runs every stage in order (see the top-level README). To run the
stages one at a time, stage the data (`00_stage_data.sh` with the two downloads) and install
the weights (`scripts/install_weights.py`), then:

```bash
export PYTHON="$PWD/.venv-train/bin/python"   # absolute path, see "Interpreter" below
export TABICL_CKPT=inference/assets/tabicl-classifier-v2-20260212.ckpt
bash pipeline/00_stage_data.sh      # verifies the staged data
bash pipeline/01_folds.sh
bash pipeline/02_features.sh
bash pipeline/03_linear.sh
bash pipeline/04_caches.sh
bash pipeline/05_cnn3d.sh
bash pipeline/06_cnn2d.sh
bash pipeline/07_tabular.sh --ckpt "$TABICL_CKPT"
bash pipeline/08_stack.sh
bash pipeline/09_package.sh --overwrite
```

`07_tabular.sh` takes the TabICL checkpoint from `--ckpt`, else from `TABICL_CKPT`, and
`09_package.sh` takes it from `TABICL_CKPT`. Both fall back to
`inference/assets/tabicl-classifier-v2-20260212.ckpt`, where `install_weights.py` puts it.
`tabular/freeze_tabular.py` and `src/package_submission.py`, run directly, have the same
default. The export above only makes the path explicit, and `train_all.sh` sets the
variable itself.

## Conventions shared by the scripts

- **Paths.** Each script finds the repository root from its own location, so it can be
  started from any directory. Paths in these documents are relative to the root.
- **Interpreter.** `PYTHON` selects the Python interpreter, default `python`. Give it as
  an absolute path or as a command on `PATH`, because the scripts change directory before
  they call it. They also convert a relative path to an absolute one before the first `cd`.
- **`DAT_*` variables.** They switch research variants of the preprocessing. Scripts 01
  to 07 unset every `DAT_*` variable except `DAT_DATA_DIR` (the data location). Stages 02
  and 04 then set `DAT_MASK_V2` and `DAT_FRAME_V2` to the values of the original runs.
  `00_stage_data.sh` reads only `DAT_DATA_DIR`. 08 and 09 run no preprocessing and set
  none. `train_all.sh` unsets all of them before the first stage.
- **Other variables.** Scripts 01 to 09 export `KMP_DUPLICATE_LIB_OK=TRUE` and
  `PYTORCH_ENABLE_MPS_FALLBACK=1`, which the original macOS runs used and which have no
  effect elsewhere. Thread counts are listed per stage.
- **Devices.** Registration and CNN training pick the Apple GPU (MPS) if present, then
  CUDA, then the CPU.
- **Logs.** Each stage writes its logs to `outputs/logs/`. `train_all.sh` also copies its
  whole output to `outputs/logs/train_all.log`.
- **Data handling.** The scripts print aggregates only (counts, hashes, metrics). The
  cache logs of stage 04 are the exception: a `FAIL` line would name a scan's uid.
  `outputs/` and `submission_src/` hold files derived from the competition data, some of
  them with uids and labels, and `.gitignore` excludes both directories.
- **Data path patches.** `src/build3.py`, `src/cache3.py`, `src/eda_real.py`,
  `src/folds.py` and `src/make_regfeat32.py` read the data from `DAT_DATA_DIR` or
  `data/raw/`. The original files read `data-demo/`. Each patch is marked in the file.
  Give `DAT_DATA_DIR` as an absolute path, because most stages call these modules after a
  `cd src`.

## The original development tree

The documents name scripts and directories such as `tools/build_f2two.sh`,
`tools/v14c_chain.sh`, `fleet3.sh`, `experiments/v17/prepare_candidate.py`,
`champion_v7_CV0.2139/` and `data-demo/`. These belonged to the original development tree,
which is not distributed. They are named by their path relative to that tree's root so
that every pipeline command can be traced to the command it reproduces. An "archived
original" is a file from that tree's `outputs/` directory or from an archived submission.

A few steps of the original chain ran as one-off inline code and were never saved as
scripts. Their code was recovered from the original development session logs and now lives
in `src/filter_feature_tables.py`, `src/linear_oof.py` and the group rules of
`tabular/derive_groups.py`.

## Measurement environment

All runtimes in these documents were measured on an Apple M3 Pro (11 CPU cores, 18 GB
unified memory, MPS) shared with other jobs, so they are upper bounds. The exception is the
complete `MODE=full` run of 2026-09-28, which had the machine to itself. The training
environment was Python 3.12.11 with `requirements-train.txt` (torch 2.13.0, numpy 2.5.1,
scipy 1.18.0, scikit-learn 1.9.0, pandas 3.0.5). The "original run" columns come from the
logs of the original runs on the same laptop.

## Reproduction status by stage

| stage | result on the development machine |
|---|---|
| 00-03 | every regenerated file sha256-identical to the archived original |
| 04 | both caches match the originals in content (normalisation constants, replayed OOF, QC tables). The container bytes differ, and the originals had no recorded hash. Two rebuilds, on 25 and 27 September, gave byte-identical files. |
| 05-06 | complete retrain of all 16 runs on MPS: all 52 checkpoints tensor-identical to the shipped and original ones (42 byte-identical), every out-of-fold file identical. TorchScript re-export gives identical logits. |
| 07 | every compared file identical except the two frozen contexts, whose bytes change on every run and whose predictions are identical |
| 08 | on archived inputs, every output byte-identical except `candidate_v14c.zip`, which matches the archived v14c zip entry by entry (name, order, CRC32, size) |
| 09 | 82 of 82 archive entries with the same name, size, CRC32 and compression as the scored archive. The zip bytes differ through entry timestamps. |

End-to-end run of `MODE=fast bash pipeline/train_all.sh` in a clean copy of the repository:
From the staged raw data, with the weights installed from the scored archive and the raw 2D
checkpoints in `models/cnn_raw/`, the run took 32 min on the Apple M3 Pro (1,944 s, with other jobs
sharing the machine for part of it). Stage timings: 02 registered features 563 s, 03 linear
members 27 s, 04 crop caches 896 s, checkpoint staging and out-of-fold replay 120 s, 07 TabICL
block 294 s, 08 stacker 19 s, 09 packaging 23 s. The replayed CNN out-of-fold predictions
matched the training-time files to 1.6e-7 (3D member), 4.1e-6 (B0 pair) and 1.4e-5 (B3). The
stacker scale and intercept moved by 1.4e-9 and 8.8e-8 as a result. Against the scored
archive, 79 of 82 entries were identical in name, size and CRC32. The three that differ are
`assets/inference_config.json` (those last digits of the stacker constants) and the two frozen
TabICL contexts (new pickle bytes, identical predictions). The rebuilt package, run on the
150-scan local verification set in the platform-pinned environment, gave predictions within
1.0e-8 of the scored package (log loss equal to 8 decimals).

End-to-end run of `bash pipeline/train_all.sh` in `MODE=full` (the default), in a clean copy
of the released code archive with fresh environments, the released weights bundle and the
data staged from the competition download (2026-09-27 23:42 to 2026-09-28 05:52). The run
took 6 h 10 min (22,199 s): 02 registered features 494 s, 03 linear members 19 s, 04 crop
caches 791 s, 05 3D CNN 11,791 s, 06 2D CNNs with export 8,760 s, 07 TabICL block 301 s,
08 stacker 19 s, 09 packaging 23 s. Every stage reproduced its original outputs:

- all 52 CNN checkpoints tensor-identical;
- the stacker and `inference_config.json` byte-identical;
- 52 of 82 archive entries identical to the scored archive. The 30 that differ are the 18
  re-exported `.ts.pt` files, the two TabICL contexts and ten 3D checkpoints whose metadata
  keys changed.

The rebuilt package predicted the 150-scan verification set exactly as the scored package
(maximum absolute difference 0.0). Details are in `docs/REPRODUCIBILITY.md`, section 5, and
the logs in `docs/repro/mode_full_2026-09-28/`.

## Checks of the scored package

These checks ran the unmodified v17 inference code on a local set of 150 scans in the
competition layout, which the original build gates used and which is not distributed. The
reference is the set of predictions recorded for those scans on 2026-09-14, when the
package was built.

- In a fresh environment (Python 3.12.11, torch 2.13, numpy 2.5.1, scipy 1.18,
  scikit-learn 1.9) the predictions matched to max |diff| 4.0e-6.
- In the platform-pinned environment of `requirements-inference.txt` (torch 2.12.1,
  numpy 2.2.6, scipy 1.17.1, scikit-learn 1.8.0, pandas 3.0.3) they matched exactly
  (max |diff| 0.0).
- Removing the `*.debug_pkl` records from the 18 TorchScript files, as in the released
  weights bundle, leaves the outputs bit-identical.
- scipy 1.15.3, the version the platform ran, has macOS arm64 wheels that fail to load on
  recent macOS. `requirements-inference.txt` therefore pins scipy 1.15.3 on x86_64 only
  and 1.17.1 on other machines.
