# Complete `MODE=full` run, 27-28 September 2026

These files are the record of one run of `pipeline/train_all.sh` in `MODE=full`, from the
raw competition data to `outputs/v17/submission.zip`. The run is summarised in
`docs/REPRODUCIBILITY.md`, section 5.

## Setup

- **Code.** The released code archive `dat-park-navattack-code.zip` (sha256 `907c5486…`),
  unzipped into an empty folder. Every `.py` and `.sh` file of this repository is
  byte-identical to that archive. Later changes touched documentation only.
- **Environments.** Fresh `.venv-train` and `.venv-infer`, created with
  `uv venv --python 3.12.11` and `requirements-train.txt` or `requirements-inference.txt`
  (`run/freeze_train.txt`, `run/freeze_infer.txt`).
- **Weights.** `scripts/install_weights.py --from-zip navattack_v17_weights.zip`, with all 60
  hashes verified (`run/setup_install_weights.log`). `MODE=full` reads only the registration
  template and the public TabICL checkpoint from it.
- **Data.** `pipeline/00_stage_data.sh` on the original competition download. The zip and
  labels hashes matched (`run/setup_stage_data.log`).
- **Backbones.** `tools/fetch_backbones.py` filled an empty Hugging Face cache before the run,
  and both hashes matched (`run/setup_fetch_backbones.log`). Stage 06 checked them again.
- **Machine.** Apple M3 Pro, 18 GB, macOS 27.0, on AC power, otherwise idle, kept awake with
  `caffeinate` (`run/metadata_start.txt`).
- **Command.** `CHECK_AGAINST=<scored submission.zip> PYTHON=<copy>/.venv-train/bin/python
  bash pipeline/train_all.sh`.

A one-epoch rehearsal of the same command (`EPOCHS=1`) had passed in a separate copy first.

## Result

`train_all.sh` exited 0 after 22,199 s (6 h 10 min):

- all 52 CNN checkpoints came out tensor-identical to the scored ones;
- the stacker came out byte-identical;
- 52 of 82 archive entries are identical to the scored archive. The 30 that differ have
  documented causes.

The rebuilt package and the scored archive predicted the 150-scan local verification set
identically (maximum absolute difference 0.0). `run/retrain_report.md` lists every check.

## Files

| folder | content |
|---|---|
| `run/` | `retrain_report.md` and `.json` (all checks, aggregates only), `train_all_console.log` (the complete console output of the run), `driver.log`, `metadata_start.txt`, package freezes, setup logs, the template rebuild log and the logs of the TorchScript and TabICL checks |
| `stage_logs/` | the per-stage and per-run logs that the pipeline writes to `outputs/logs/` |
| `results/` | `cv_<tag>.json` of the nine five-fold runs, `ensemble_v14c.json`, the package `MANIFEST.json` (per-entry sha256 and the comparison with the scored archive) and the TabICL `persistence_checks.json` |
| `parity/` | the inference logs of the rebuilt and the scored package on the 150 verification scans |

A small export script, kept with the run and not part of this repository, copied these
files. The same harness launched the run and did the checks after `train_all.sh`: package
parity, template rebuild and report. The export replaced local paths with
placeholders (`<repo>` for the repository copy, `<run>` for its parent folder,
`<original-project>` for the original development tree, `~` for the home folder) and removed
the host name. It then checked every file for personal strings and for the 1,362 training
scan identifiers, and found none. The predictions themselves are not included, because they
are per-scan values.
