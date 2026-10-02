# Stages 08-09: stacker and submission package

Scripts: `pipeline/08_stack.sh`, `pipeline/09_package.sh`, and the static gates in
`src/check_package.py`. Shared conventions are described in [README.md](README.md).

## Purpose

- **`08_stack.sh`** runs the fixed-ratio stacker with its one-dimensional calibration
  (`src/fixstack.py`) and the image-ensemble package spec (`src/build3.py`). Together they
  produce the image and linear half of v17, first built for submission v14c:
  `submission_src/assets/inference_config.json` and 55 image, linear and template assets.
- **`09_package.sh`** assembles v17 as a directory and as `submission.zip`. It takes the
  code from `inference/` verbatim, adds the assets of stage 08, extends the configuration
  for the tabular block, and adds the tabular recipe, the two frozen TabICL contexts and
  the TabICLv2 checkpoint.
- **`src/check_package.py`** runs the data-free static gates of the original build. Stage
  09 calls it, and it also runs on its own.

`fixstack.py` fixes the logit ratio of the five members lin, lin4, mil48y, e2dmipf2 and
e2db3f2 at 1 : 0.4 : 0.7 : 0.3 : 0.3 and fits a single `LogisticRegression(C=1e6)` on the
blended logit of all OOF rows. The fit gives scale 0.53121 and intercept 0.18431, and each
member coefficient is the scale times its ratio (0.3718 for mil48y, 0.1594 for each 2D
member). The ratios, the temperature T = 1.4 and the registration rejection bounds were
chosen by hand during development. Only the scale and the intercept are fitted.
`build3.py` takes the prior from the label mean of `data/raw/labels.csv`, the rejection
bounds from `regfit.MAX_DEG` (90) and `regfit.MIN_NCC` (0.30), and the template from
`outputs/template_v6.npy`.

## Commands

From the repository root:

```bash
bash pipeline/08_stack.sh
TABICL_CKPT=inference/assets/tabicl-classifier-v2-20260212.ckpt bash pipeline/09_package.sh --overwrite
# optional: compare every entry with the scored archive
TABICL_CKPT=inference/assets/tabicl-classifier-v2-20260212.ckpt \
  bash pipeline/09_package.sh --overwrite --check-against path/to/submission.zip
# static gates alone, on a zip or a staged directory
python src/check_package.py outputs/v17/submission.zip
```

`train_all.sh` runs `08_stack.sh` and then `09_package.sh --overwrite`, adding
`--check-against "$CHECK_AGAINST"` when that variable is set. It exports `TABICL_CKPT`
itself. Logs are appended to `outputs/logs/08_stack.log` and `outputs/logs/09_package.log`.

### 08_stack.sh

The script reproduces the stacker and build part of `tools/build_f2two.sh v14c e2dmipf2x
e2dmipf2 e2dmipf2+e2dmipf22`, which `tools/v14c_chain.sh` called on 2026-09-08, with the
same variables and argument lists:

1. The stacker:

   ```bash
   python src/fixstack.py --feat oof_feat3_m2fnh.csv --cnn oof_mil48y.csv \
     --extra oof_e2dmipf2x.csv e2dmipf2x --extra oof_e2db3f2.csv e2db3f2 \
     --ratio 1.0 0.4 0.7 0.3 0.3 --out ensemble_v14c.json
   ```

   T = 1.4 and clip = 0.005 are argparse defaults of `fixstack.py`. The last two ratios
   can be overridden with `RATIO_B0` and `RATIO_B3` (default 0.3, the v17 values).
2. The inline member rename, copied verbatim from `build_f2two.sh`, sets
   `members = ['lin','lin4','mil48y','e2dmipf2','e2db3f2']` in
   `outputs/ensemble_v14c.json`. `fixstack.py` names the 2D column after its OOF file
   (`e2dmipf2x`), and inference expects the member name `e2dmipf2`.
3. The package spec:

   ```bash
   cd src && python -u build3.py --ensemble ensemble_v14c.json \
     --cnn-tags mil48y=mil48a+mil48b+m31000+m33000+m35000+m37000+mf21000+mf23000+mf25000+mf27000 \
                e2dmipf2=e2dmipf2+e2dmipf22 e2db3f2=e2db3f2 \
     --use-full --zip candidate_v14c.zip --tta 0 --template template_v6.npy \
     --key2d e2dmipf2=aligned_f2 e2db3f2=aligned_f2 \
     --crops2d e2dmipf2=crops_v3_f2.npz e2db3f2=crops_v3_f2.npz
   ```

4. GATE 6, copied verbatim: `grep -nE "logger\.[a-z]+\(f" submission_src/main.py` must
   print nothing.

The script exports `KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1
OMP_NUM_THREADS=3 HF_HUB_OFFLINE=1`, as the original did, and sets no `DAT_*` variable.
It adds three parts to the original:

- A precondition block that lists every input file, the 52 checkpoints that v17 ships and
  the 18 raw 2D state dicts. `build3.py` skips a missing checkpoint without an error, and
  with `--use-full` it would also ship any stray `<source>_full0.pt` it finds. The script
  refuses to build in either case.
- The linear-member check (235 and 94 features, no `head_ml` or `hole_ml`), copied
  verbatim from `build_f2two.sh`.
- A closing summary of sha256 values.

The original `build_f2two.sh` also re-ran `train_final3.py --export`. In this pipeline
that step belongs to stage 03.

### 09_package.sh

The script reproduces `experiments/v17/package_candidate.py`, the last step of the
original lineage (v14c archive, then `experiments/20260914/build_stage.py`, then
`experiments/v17/prepare_candidate.py` and `freeze_candidate.py`). It runs:

```bash
python -u src/package_submission.py --inference-dir inference --build-assets submission_src/assets \
  --tabular-dir outputs/tabular --feature-order outputs/tabular/feature_order_v17.json \
  --tabicl-ckpt "${TABICL_CKPT:-inference/assets/tabicl-classifier-v2-20260212.ckpt}" \
  --out-dir outputs/v17 [extra arguments]
```

Extra arguments pass through to `package_submission.py`:

| argument | effect |
|---|---|
| `--overwrite` | replace an existing `outputs/v17/submission/` and `submission.zip`. Without it the script refuses, as the original did. |
| `--check-against REF.zip` | compare the name, size, CRC32 and compression type of every entry with a reference archive |
| `--check-only` | build nothing, and run the static gates (and `--check-against`) on the existing zip |

How the package is put together:

- **Code.** Every file of `inference/` is copied verbatim: the inference code, `main.py`,
  `CANDIDATE.json`, `tabicl_lib.zip`, `tabicl_licenses/LICENSE` and
  `TABICL_MODEL_CARD.md`. Files under `inference/assets/` are ignored, which keeps
  `.gitkeep` and any installed weights out of the archive (with `.gitkeep` it would have
  83 entries).
- **Assets.** The image, linear and template files come from `submission_src/assets/`,
  selected by name from its `inference_config.json` with the rule `build3.py` uses for its
  own zip (55 files).
- **Configuration.** The script loads the configuration written by `build3.py`, sets
  `tabicl = true` and appends every name of `feature_order_v17.json` that is missing from
  `feature_order`, then writes it with `json.dumps(indent=2)`. This follows lines 42-46 of
  `build_stage.py` and lines 42-45 of `prepare_candidate.py`. The feature-order file may be
  a JSON list or `{"feature_order": [...]}`, holding either all 479 names or only the
  244-name extension. All three forms were tested and give the same 479-name order. The
  code never reads the `tabicl` key.
- **Tabular files.** `tabular_recipe.json` and the two states it names come from
  `outputs/tabular/`, and the checkpoint is stored under the name the recipe expects.
- **Zip rules** (as in `package_candidate.py`): `ZIP_STORED` for `.xz` and `.zip`,
  `ZIP_DEFLATED` level 6 otherwise. `__pycache__`, `.pyc`, `.tmp`, `submission.csv`,
  `.DS_Store` and `.gitkeep` are skipped, and no `.npz` may be shipped.

The static gates of `check_package.py` then run on the finished zip:

- GATE 6 of `build_f2two.sh` (no `logger.<level>(f"...")` call) on `inference_main.py` and
  `main.py`. From v15 on the logging calls live in `inference_main.py`, and `main.py` is a
  small wrapper.
- From `package_candidate.py`: `testzip()` passes, `main.py` sits at the archive root, no
  `.npz` is present, `scale_features.py` and `mixture_features.py` execute with
  `__file__ = /code_execution/<name>` (the platform's shallow working directory, where
  submission v15a2 failed at startup), and every `.py` compiles.

The gates read no competition data. The shallow-import check needs numpy and scipy.

`outputs/v17/MANIFEST.json` records `zip_bytes`, `sha256`, `entries`,
`shallow_container_import_passed`, `static_checks`, `check_against` and `files`, which
lists the name, size, CRC32 and sha256 of every entry.

## Inputs and outputs

| stage | reads | writes |
|---|---|---|
| 08 | `outputs/oof_feat3_m2fnh.csv`, `oof_mil48y.csv`, `oof_e2dmipf2x.csv`, `oof_e2db3f2.csv` | `outputs/ensemble_v14c.json` |
| 08 | `outputs/feature_order.json`, `outputs/feature_union.json`, `submission_src/assets/linear_lin{,4}.json` (stage 03) | `submission_src/assets/inference_config.json` and 55 assets |
| 08 | `outputs/template_v6.npy`, `outputs/{mil48a,mil48b,m31000,m33000,m35000,m37000}_fold{0-4}.pt`, `outputs/mf{21,23,25,27}000_full0.pt` | `submission_src/*.py` and `main.py` (v14c code, unused by 09) |
| 08 | `outputs/{e2dmipf2,e2dmipf22,e2db3f2}_{fold0-4,full0}.pt` and the matching `.ts.pt`, `data/raw/labels.csv` (for the prior) | `candidate_v14c.zip` at the repository root (530.4 MB, 70 entries) |
| 09 | `inference/` (all files outside `inference/assets/`), `submission_src/assets/` | `outputs/v17/submission/` |
| 09 | `outputs/tabular/tabular_recipe.json`, `tabular_raw.pkl.xz`, `tabular_log.pkl.xz`, `feature_order_v17.json` | `outputs/v17/submission.zip` |
| 09 | the TabICLv2 checkpoint (`TABICL_CKPT`) | `outputs/v17/MANIFEST.json` |

## Runtimes

Measured on an Apple M3 Pro shared with other jobs, on archived upstream inputs.

| step | measured |
|---|---:|
| `08_stack.sh` (`fixstack.py` takes a few seconds, `build3.py` re-serialises 52 checkpoints and writes a 530 MB zip) | 35 s |
| `09_package.sh`, three runs | 30-50 s |
| of which copying and zipping | 26-43 s |
| of which `testzip`, shallow import, compilation and hashing | about 5 s |
| `09_package.sh --check-only --check-against` | 3 s |

## Verification

**Setup.** These results come from a copy of the repository code (`src/`, `tools/`,
`inference/`, `configs/`, `pipeline/`) whose `outputs/` held the archived originals from the
original tree's `outputs/`. The 18 `.ts.pt` files and the tabular files came from the
scored v17 archive. The environment was Python 3.12.11 with scikit-learn 1.9.0, torch
2.13.0 and pandas 3.0.5. `data/raw/labels.csv` has the same sha256 as the original
`data-demo/labels.csv`.

**Copied and regenerated files.** Some inputs were taken from the scored archive, and a
match with that archive only shows they were copied faithfully: the 18 `.ts.pt` files, the
two `linear_*.json` files, the three tabular files and the TabICL checkpoint. These
outputs were rebuilt and still match the originals:

- `ensemble_v14c.json`, rebuilt from the archived OOF tables
- the v14c `inference_config.json`, rebuilt from `feature_union.json`, the label prior and
  the registration bounds
- the 34 3D `.pt` files, re-serialised by `torch.save` under torch 2.13
- the 70-entry `candidate_v14c.zip`
- the v17 configuration written by 09
- the layout of the v17 zip: entry set, entry order and compression rules

**Stage 08.**

- `outputs/ensemble_v14c.json` after the rename: sha256
  `6459a002a9ef4a90d4e40471c867930db11e98ee5c4ea2310466abd5ceb19c21`, byte-identical to the
  archived file. Scale 0.5312107357995878 and intercept 0.18431409136922594, |diff| = 0.0
  for both. Cross-fitted log loss 0.20695414933159145 at T = 1 and 0.21671393562606034 at
  T = 1.4 (|diff| 0.0), AUROC 0.97228.
- `submission_src/assets/inference_config.json`: sha256
  `dcf4e4747c5ed0305cc6411aa1f2d6011bb55a8b091c94eb72bbf9fa923e3707`, identical to the v14c
  configuration.
- All 55 assets (34 re-serialised 3D `.pt`, 18 `.ts.pt`, 2 linear JSON files, the
  template) are sha256-identical to the assets of the scored archive.
- `candidate_v14c.zip`: 70 of 70 entries have the same names, order, CRC32 and size as the
  archived v14c submission archive.
- GATE 6 passed. Linear check: 235 and 94 features, no `head_ml` or `hole_ml`.

**Stage 09**, against the scored archive (sha256 `88b5611e…`):

- 82 of 82 entries have the same name, size, CRC32 and compression type, in the same
  order. The compressed sizes match for all 82 entries, and so does the total size
  (1,090,109,497 bytes).
- The zip sha256 differs (for example `0c151e3d…`) because entry timestamps differ: the
  date and time match for 28 of 82 entries.
- `assets/inference_config.json` has sha256
  `6c1a8bf6e3092ef988f904fabba3452adc08f1c87d42780f1198910886fdb4b8`, equal to the shipped
  file.
- The TabICL checkpoint sha256 `bdc7dbd5e4ff21f8…` matches.

**Static gates.**

- The scored archive passes: 0 logging f-strings in `inference_main.py` and `main.py`,
  the shallow `/code_execution` import of `scale_features.py` and `mixture_features.py`
  works, 18 of 18 `.py` files compile, `testzip` passes, 0 `.npz` files.
- The rebuilt zip and the staged directory pass.
- The v14c archive fails as expected (it has no `inference_main.py` or
  `scale_features.py`). A synthetic package was caught on all three counts: a
  `logger.info(f"…")` line, an `IndexError` from `Path(__file__).parents[1]` (the failure
  of submission v15a2) and a syntax error.
- Without `--overwrite` the script refuses to replace an existing package. With it, the
  script rebuilds.

**End-to-end pipeline run.** `MODE=fast` in a clean copy of the repository:
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

**`MODE=full` in a clean copy of the released code archive** (2026-09-28, 6 h 10 min in
total, `docs/REPRODUCIBILITY.md` section 5):

- **Stage 08** took 19 s. It rebuilt `outputs/ensemble_v14c.json` and the v14c
  `inference_config.json` sha256-identical to the originals (scale 0.5312107357995878,
  intercept 0.18431409136922594, cross-fit log loss 0.20695 at T 1 and 0.21671 at T 1.4).
- **Stage 09** took 23 s and passed the static gates. Against the scored archive, 52 of 82
  entries were identical in name, size and CRC32, `assets/inference_config.json` included.
  The 30 that differ are:
  - the 18 re-exported `.ts.pt` files, whose logits equal the released ones;
  - the two TabICL contexts;
  - the 10 `mil48a` and `mil48b` checkpoints, with identical tensors and extra metadata
    entries (see below).
- **Predictions.** The rebuilt package, run on the 150-scan verification set in the
  platform-pinned environment, gave predictions identical to the scored package (maximum
  absolute difference 0.0).

In a full pipeline run, `--check-against` compares against files that stages 01 to 08
regenerate, and some entries cannot match the scored archive byte for byte:

- `tabular_raw.pkl.xz` and `tabular_log.pkl.xz` differ in every run (see
  [07_tabular.md](07_tabular.md)). In the stage 07 checks their predictions were identical
  to those of the shipped states.
- The 18 `.ts.pt` files differ when `inference/assets/` was installed from the released
  weights bundle, whose TorchScript files have their `*.debug_pkl` records removed, and
  when `MODE=full` re-exports them.
- In `MODE=full` the 10 `mil48a` and `mil48b` `.pt` files differ in bytes, because the
  current `cnn3.py` saves six metadata keys that their 2026-07-24 originals lack (see
  [04-06_cnn.md](04-06_cnn.md)). The other 24 3D files were byte-identical after the re-save
  on MPS. On another device the retrained weights, and with them all 34 files, differ.
- `assets/inference_config.json` matches only when the stacker's scale and intercept come
  out bit-identical, which requires bit-identical OOF inputs. The `MODE=full` run on MPS
  met that condition.

For these entries the test is a comparison of predictions on scans. The rebuilt package
runs with `cd outputs/v17/submission && DATA_DIR=/path/to/DATA_DIR python main.py`, the
released one with `bash scripts/predict.sh /path/to/DATA_DIR submission.csv`, and the two
CSV files can then be compared.

## Deviations and limits

- **Raw 2D checkpoints are required.** `build3.py` reads `outputs/<tag>_<part>.pt` (the
  state dict) of every 2D source to get the kind, backbone, resolution and slabs. It then
  ships `<tag>_<part>.ts.pt` and deletes the re-saved `.pt` from the assets. Stage 06 (or
  `MODE=fast`) must therefore leave both files in `outputs/`.
- **`candidate_v14c.zip`** lands at the repository root because the original `--zip`
  argument is kept verbatim. It is an intermediate that 09 does not read, and
  `.gitignore` excludes it.
- **`submission_src/main.py`** is `src/main3.py`, the v14c entry point. v17 ships
  `inference/main.py` and `inference/inference_main.py`, so 09 takes all code from
  `inference/` and only the assets from `submission_src/assets/`.
- **Gates that are not reproduced.** The data gates of `build_f2two.sh` (4: clean-room
  run, 5: numerical parity, 7: forced-reject toggle) and the v11 column-set gate need the
  local 150-scan verification folder and the v11 archive. Neither is distributed, so 08
  does not run them.
- **Stacker inputs.** The stacker is fitted on the four-run `oof_mil48y.csv`, while the
  shipped 3D member averages 34 checkpoints. The scored submission made this choice, and
  the pipeline keeps it.
- **Re-serialised 3D checkpoints.** `build3.py` loads and re-saves each 3D checkpoint with
  `torch.load` and `torch.save`. Under torch 2.13 the bytes were identical to the shipped
  files. Another torch version may change the bytes with equal tensors.
- **Exit codes.** `package_submission.py` exits 1 if a static gate fails or if
  `--check-against` finds a mismatch. It writes `submission.zip` and `MANIFEST.json` before
  that check. With `--check-against`, a regenerated package, whose frozen TabICL contexts
  always differ in bytes, therefore yields a usable zip and a non-zero exit, and
  `09_package.sh` stops under `set -e`. Without `--check-against`, a package that passes
  the static gates exits 0. The `check_against` field of `MANIFEST.json` lists the entries that changed.
  With `CHECK_AGAINST` set, `train_all.sh` catches this exit and prints a note. A TabICL
  checkpoint with the wrong sha256 only triggers a printed warning, and `--check-against`
  also reports it.
- **Checkpoint location.** `09_package.sh` without `TABICL_CKPT`, and
  `src/package_submission.py` without `--tabicl-ckpt`, use
  `inference/assets/tabicl-classifier-v2-20260212.ckpt`, where `install_weights.py`
  installs it. The commands above set `TABICL_CKPT` only to
  make the path explicit.
- **Shell changes from the original.** The original piped the output of each step
  through `grep -v -i warn` and `tail`. In 08 and 09 these filters were dropped because
  they can fail under `pipefail`, so the logs of these two stages are fuller. Logs are
  appended with `tee -a`. `PYTHON` defaults to `python`.
