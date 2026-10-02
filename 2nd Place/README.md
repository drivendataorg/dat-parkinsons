# Solution: DaT Parkinson's Challenge (2nd place)

Username: NavAttack

This repository reproduces submission **v17** (DrivenData submission id 322397,
private log loss 0.2731 and AUROC 0.9524, public log loss 0.2409), from the raw
competition data to the submission archive, and runs it on new scans.

## Summary

The model reads one 3D DaT-SPECT scan (NIfTI) and returns the probability that the
scan is abnormal. On the normal path each scan is processed on its own ("Run inference"
describes the batching and the error fallbacks).

![v17 inference pipeline](docs/figures/pipeline_diagram.png)

1. **Preprocessing (per scan).** A log-intensity Otsu threshold gives the head mask,
   with a guard that re-thresholds masks larger than 7.5 L (reconstruction discs).
   The reference level is the mode of log-intensity inside the mask eroded by 10 mm,
   without the top 8% of voxels. Two striatal peaks define a canonical frame, and a
   96 mm cube at 2 mm (48³ voxels) is sampled in specific-binding-ratio (SBR) units.
   For the feature models the crop is registered rigidly to a study template.
2. **Members.**
   - `lin`, `lin4`: L1 logistic regressions on 235 and 94 hand-crafted striatal
     features (ROI uptake, putamen/caudate ratios, asymmetry, shape).
   - `mil48y`: a 3D hemisphere multiple-instance CNN (shared encoder per hemisphere,
     noisy-OR over the two sides), trained from scratch, averaged over 34 checkpoints.
   - `e2dmipf2`, `e2db3f2`: ImageNet-pretrained EfficientNet-B0 (12 checkpoints) and
     B3 (6 checkpoints) on three maximum-intensity projections, shipped as TorchScript.
   - A TabICLv2 in-context tabular block on 64 selected features, 4 acquisition
     descriptors and 4 linear prior scores (`lin`, `lin4`, and `lin` extended with
     scale-space shape or spatial Gaussian-mixture features), with a raw and a
     log-feature context.
3. **Combination.** Fixed logit ratio 1 : 0.4 : 0.7 : 0.3 : 0.3 for the image and
   linear members, one 1-D calibration, temperature 1.4. The final logit is
   0.65 × stack logit + 0.35 × TabICL logit / 1.2. A scan whose registration fails
   the quality check receives the training prior 0.5485, and outputs are clipped to
   [0.005, 0.995].

The write-up for DrivenData (design choices, ablations, what did not work) was
submitted separately to DrivenData. `docs/figures/` holds its charts, and `docs/stages/`
and `docs/REPRODUCIBILITY.md` hold the technical detail.

## Repository layout

```
.
├── README.md                    this file
├── LICENSE                      MIT (the solution code and the fine-tuned weights)
├── THIRD_PARTY_NOTICES.md       licences of bundled/pretrained third-party components
├── .gitignore                   keeps data, outputs and weights out of version control
├── requirements-inference.txt   inference environment (pinned to the DrivenData runtime)
├── requirements-inference.lock.txt  exact freeze of the verified inference environment
├── requirements-train.txt       training environment (frozen development environment)
├── inference/                   the scored v17 inference code, byte-identical to submission 322397
│   ├── main.py                  entry point (reads $DATA_DIR, writes ./submission.csv)
│   ├── inference_main.py, *.py  preprocessing, features, model code
│   ├── CANDIDATE.json, TABICL_MODEL_CARD.md   scored-archive metadata (verbatim)
│   ├── tabicl_lib.zip           TabICL 2.2.0, vendored unmodified (BSD-3-Clause)
│   ├── tabicl_licenses/LICENSE
│   └── assets/                  model weights (installed by scripts/install_weights.py, ships empty)
├── scripts/
│   ├── install_weights.py       install + sha256-verify the 60 weight files
│   ├── predict.sh               inference on a competition-style data directory
│   └── predict_folder.py        inference on any folder of NIfTIs (or one file)
├── pipeline/                    training, one numbered script per stage
│   ├── train_all.sh             runs stages 00-09 (MODE=full or MODE=fast)
│   ├── 00_stage_data.sh ... 09_package.sh
│   └── smoke_cnn.sh             1-epoch exercise of every CNN training command
├── src/                         training modules (original code, see docs/PROVENANCE.md)
├── tabular/                     TabICL block: groups, shape/mixture features, recipe, freeze
│   └── _original/               reference copies of six original scripts (not imported)
├── tools/                       header survey, TorchScript export, backbone fetch, replays
├── configs/feature_columns/     column orders of the feature tables (header lines only)
├── models/                      SHA256SUMS of the weights, weights README
└── docs/                        figures, provenance, reproducibility, per-stage notes,
                                 logs of the complete MODE=full run (repro/)
```

`inference/assets/` ships empty (it holds only a `.gitkeep`). These locations are
created locally and are not part of the repository: `data/` (competition data),
`outputs/` (intermediate files and the rebuilt package), `submission_src/` and
`candidate_v14c.zip` (training stages 03 and 08), `models/cnn_raw/` (the optional raw 2D
checkpoints), the virtual environments and the installed files in `inference/assets/`.

## Setup

1. **Python 3.12.** The solution was developed with Python 3.12.11 on macOS (Apple
   Silicon). Inference was run on macOS (CPU) and, for the scored submission, on Linux
   in DrivenData's CUDA container. Training was run only on macOS with the Apple GPU
   (MPS). The code selects CUDA, MPS or CPU automatically, but training on CUDA or CPU
   and anything on Windows were not tested.

   Unpack the code archive. All commands in this README run from its folder:

   ```bash
   unzip dat-park-navattack-code.zip && cd dat-park-navattack
   ```

   With [uv](https://docs.astral.sh/uv/):

   ```bash
   # inference only
   uv venv --python 3.12.11 .venv-infer
   uv pip install --python .venv-infer/bin/python -r requirements-inference.txt
   source .venv-infer/bin/activate    # the commands below call this interpreter as `python`
   # training (this environment also runs inference)
   uv venv --python 3.12.11 .venv-train
   uv pip install --python .venv-train/bin/python -r requirements-train.txt
   ```

   With plain pip: `python3.12 -m venv .venv-infer && source .venv-infer/bin/activate && pip install -r requirements-inference.txt`.

   `requirements-inference.txt` pins the versions the DrivenData runtime printed during
   the competition (torch 2.12.1, numpy 2.2.6, scipy 1.15.3 on x86_64, scikit-learn
   1.8.0, pandas 3.0.3). The frozen TabICL contexts are pickled scikit-learn/numpy
   objects, so keep scikit-learn 1.8.x and numpy 2.2.x for inference. On an NVIDIA
   machine install the CUDA build of torch 2.12.1 first (see the comment in the file).
   `requirements-inference.lock.txt` is an exact freeze, every dependency included, of
   the platform-pinned environment in which the unmodified code reproduced the
   14 September predictions exactly (Python 3.12.11, macOS arm64). Install it in place of
   `requirements-inference.txt` to reproduce that environment on a Mac. On Linux x86_64
   use `requirements-inference.txt`, which pins scipy 1.15.3 as on the platform.

   Apart from Python, the scripts need bash. Training also needs `unzip`, `cmp` and
   `sha256sum` or `shasum` (stages 00 and 02), which standard Linux and macOS installs
   provide.

2. **Model weights** (1.14 GB, 60 files). Either bundle works:

   - the released weights bundle `navattack_v17_weights.zip` (link: https://www.dropbox.com/scl/fi/petgufcyq2zl5xmox4f2l/navattack_v17_weights.zip?rlkey=fdf3jzwxqe13eg4brggz0tda6&st=z0ja4s95&dl=1), or
   - the scored archive itself (`submission.zip` of submission 322397,
     sha256 `88b5611e25170d88c2fbf21b62443dbc67b722c519b5a0391a60aab62bbec7fd`).

   Download the bundle, then install it:

   ```bash
   python scripts/install_weights.py --from-zip path/to/navattack_v17_weights.zip
   # or: python scripts/install_weights.py --from-zip path/to/submission.zip
   # or, only if the link downloads the zip file directly (for example a Dropbox link
   # ending in ?dl=1, not a Google Drive share page):
   #     python scripts/install_weights.py --url https://www.dropbox.com/scl/fi/petgufcyq2zl5xmox4f2l/navattack_v17_weights.zip?rlkey=fdf3jzwxqe13eg4brggz0tda6&st=z0ja4s95&dl=1
   ```

   The script extracts the assets into `inference/assets/` and checks every sha256
   against `models/SHA256SUMS.public.txt` or `models/SHA256SUMS.scored.txt`. The two
   lists differ only for the 18 TorchScript files: the released copies have their
   `*.debug_pkl` records removed (they held the exporting machine's file paths) and
   produce bit-identical outputs.

   Only for `MODE=fast` training: download the optional bundle
   `navattack_v17_cnn_raw_checkpoints.zip` (456 MB, 18 raw 2D state dicts, link:
   https://www.dropbox.com/scl/fi/pmhki703xdrc0emo5a2ms/navattack_v17_cnn_raw_checkpoints.zip?rlkey=sbggsruaxbpyrasz9kxx22pll&st=ub1l57uf&dl=1). The zip already contains the `cnn_raw/` folder, so unzip it into
   `models/`:

   ```bash
   unzip navattack_v17_cnn_raw_checkpoints.zip -d models/    # creates models/cnn_raw/
   ```

   The files must end up as `models/cnn_raw/e2dmipf2_fold0.pt` and so on.

3. **Data** (training only). Download the training NIfTI archive and the training
   labels from the competition's data page, then:

   ```bash
   bash pipeline/00_stage_data.sh path/to/niftis_<id>.zip path/to/train_labels_<id>.csv
   ```

   Expected structure before training:

   ```
   data/raw/
   ├── labels.csv          uid,is_pathologic (1,362 rows)
   └── niftis/<uid>.nii.gz (1,362 scans)
   ```

   Inference needs no training data. `DAT_DATA_DIR=/absolute/path` points the training
   code at another data directory. Give it as an absolute path, because some stages
   read it after changing directory.

## Hardware

All training and development ran on one laptop:

- Apple M3 Pro, 11 CPU cores (5 performance, 6 efficiency), 14-core integrated GPU
  used through PyTorch MPS, 18 GB unified memory, macOS. No CUDA.

| step | time on the M3 Pro |
|---|---:|
| Training, `MODE=full` (all models), measured end to end in a clean copy of this repo on an idle machine | 6.2 h (22,199 s) |
| of which 3D CNN (10 runs, 34 checkpoints, MPS) | 3.28 h |
| of which 2D CNNs (18 checkpoints, MPS, with the TorchScript export) | 2.43 h |
| of which crop caches, registered features, TabICL block, stacking, packaging | 27 min |
| Training, `MODE=fast` (no CNN training), measured end to end in a clean copy of this repo | 32 min |
| Inference start-up, CPU (loading all models, once per run) | about 35 s |
| Inference, 150 scans, CPU | 497-530 s on an idle machine (3.3-3.5 s per scan), 13-17 min while other jobs shared it |
| Inference, 1,000 scans, CPU (estimate from the 150-scan runs) | about 1-1.9 h |

The scored submission ran in DrivenData's runtime (1× NVIDIA A100 80 GB, 24 vCPU,
220 GB RAM, 3 h limit) on the full test set. Memory use of inference is given under
"Run inference".

## Run training

`pipeline/train_all.sh` runs every stage in order from the staged raw data:

```bash
PYTHON="$PWD/.venv-train/bin/python" bash pipeline/train_all.sh              # MODE=full: retrain everything
PYTHON="$PWD/.venv-train/bin/python" MODE=fast bash pipeline/train_all.sh    # reuse the released CNN checkpoints
# optional: compare the produced archive entry by entry with the scored one
CHECK_AGAINST=/path/to/submission.zip PYTHON="$PWD/.venv-train/bin/python" MODE=fast bash pipeline/train_all.sh
```

Set `PYTHON` to an absolute path or to a command on `PATH`. The stage scripts change
directory before they call it (they also convert a relative path to an absolute one).

The result is `outputs/v17/submission.zip` (plus the unzipped `outputs/v17/submission/`
and `MANIFEST.json`), the same layout as the scored archive. To predict with the
retrained package, run it the way the competition runtime does:

```bash
cd outputs/v17/submission && DATA_DIR=/path/to/DATA_DIR python main.py   # writes ./submission.csv
```

(`scripts/install_weights.py` accepts only the released hashes. Re-frozen TabICL
contexts get new hashes even when their predictions are identical, because torch writes
a per-process id into pickled tensors, and retrained checkpoints differ anyway.)

Intermediate files go to `outputs/`, plus `submission_src/` (stages 03 and 08) and
`candidate_v14c.zip` (stage 08). In the clean `MODE=fast` run they took about 5.4 GB:
4.3 GB in `outputs/` (2.1 GB of it the package in `outputs/v17/`), 550 MB in
`submission_src/` and 530 MB for `candidate_v14c.zip`. The two crop caches are 446 MB each.

| stage | script | what it does | inputs → outputs |
|---|---|---|---|
| 00 | `00_stage_data.sh` | stage and check the raw data | download → `data/raw/` |
| 01 | `01_folds.sh` | stratified 5-fold split, seed 42 | labels → `outputs/folds.csv` |
| 02 | `02_features.sh` | preprocess, register to the frozen template, extract 240 + 98 features, filter to the v17 column sets | NIfTIs, `template_v6.npy` → `features_v13_m2fnh.csv`, `features_v4_m2fnh.csv` |
| 03 | `03_linear.sh` | out-of-fold and full-data L1 logistic regressions | feature tables → `oof_feat3_m2fnh.csv`, `linear_lin{,4}.json` |
| 04 | `04_caches.sh` | 48³ crop caches for the CNNs (3D without the disc guard, as trained, and 2D with the disc and frame guards) | NIfTIs → `crops_v3_b.npz`, `crops_v3_f2.npz` |
| 05 | `05_cnn3d.sh` | 10 HemiMIL runs (6 five-fold, 4 full-data), 20 epochs each | `crops_v3_b.npz` → 34 `.pt`, `oof_mil48y.csv` |
| 06 | `06_cnn2d.sh` | EfficientNet-B0 ×2 seeds and B3, five-fold + full data, 30 epochs, TorchScript export | `crops_v3_f2.npz` → 18 `.ts.pt`, `oof_e2dmipf2x.csv`, `oof_e2db3f2.csv` |
| 07 | `07_tabular.sh` | acquisition groups, 180 shape + 60 mixture features, TabICL recipe, frozen contexts | tables, `crops_v3_f2.npz` → `outputs/tabular/` |
| 08 | `08_stack.sh` | fixed-ratio stacker + calibration, image package spec | OOF files, checkpoints → `submission_src/assets/` |
| 09 | `09_package.sh` | assemble the archive with the verbatim `inference/` code, static gates | → `outputs/v17/submission.zip` |

Notes:

- **Network.** `MODE=full` downloads the ImageNet weights `timm/efficientnet_b0.ra_in1k`
  and `timm/efficientnet_b3.ra2_in1k` from the Hugging Face Hub once
  (`tools/fetch_backbones.py` pins their sha256). Nothing else needs a network.
- **Frozen inputs.** Training uses two released files that it does not retrain:
  the registration template `template_v6.npy` (a label-free average of the registered
  training crops, built in July 2026 by `src/register3.py`) and the public
  TabICLv2 checkpoint (`huggingface.co/jingang/TabICL`, file
  `tabicl-classifier-v2-20260212.ckpt`, revision `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`).
  The template can also be rebuilt: `src/register3.py` with its default arguments, run on
  the stage-04 cache `outputs/crops_v3_b.npz`, reproduced it byte for byte in 8-9 min on
  the M3 Pro, twice (command and checks in `docs/REPRODUCIBILITY.md`, section 4).
- **Determinism.** A complete `MODE=full` run reproduced the scored submission on the
  development machine (Apple M3 Pro, MPS). It ran in a clean copy of this repository, with
  fresh environments, the released weights bundle and the raw data staged from the
  competition download.
  - All 52 CNN checkpoints came out tensor-identical to the scored ones, 42 of them byte
    for byte.
  - Stages 01-03, the TabICL recipe, the stacker and `inference_config.json` came out
    byte-identical.
  - The rebuilt package predicted 150 local scans exactly as the scored one (maximum
    absolute difference 0.0).

  52 of the 82 archive entries are byte-identical to the scored archive. The other 30
  differ in bytes only: the 18 re-exported TorchScript files record the exporting
  machine's paths, the two re-frozen TabICL contexts get new pickle bytes, and ten 3D
  checkpoints carry six extra metadata entries. On CUDA or CPU expect CNN results within
  the seed spread (per-run out-of-fold log loss ±0.001-0.003). A clean `MODE=fast` run
  rebuilt the archive with 79 of 82 entries identical and predicted within 1.0e-8 of the
  scored package. See `docs/REPRODUCIBILITY.md`, section 5, and the run's logs in
  `docs/repro/`.
- **Smoke test.** `PYTHON="$PWD/.venv-train/bin/python" bash pipeline/smoke_cnn.sh` runs
  every CNN training command for one epoch under `smoke_` tags (about 11 min on the M3 Pro).
- `docs/stages/` has one document per stage group (00-03, 04-06, 07, 08-09) with the
  exact original commands, runtimes and verification results.

## Run inference

Inference needs only the inference environment and the installed weights. It makes no
network calls and uses CUDA if available, otherwise the CPU (never MPS).

**Competition layout** (a `submission_format.csv` with a `uid` column and `niftis/<uid>.nii.gz`):

```bash
bash scripts/predict.sh /path/to/DATA_DIR submission.csv
```

`predict.sh` uses the interpreter in `PYTHON` if that is set, otherwise
`.venv-infer/bin/python` if it exists, otherwise `python`.

**Any folder of NIfTI files, or a single new scan:**

```bash
python scripts/predict_folder.py /path/to/folder_or_scan.nii.gz predictions.csv
```

`predict_folder.py` runs `predict.sh` with the interpreter that runs it.

Both write a CSV with columns `uid,is_pathologic`, where `is_pathologic` is the
probability that the scan is abnormal. `predict.sh` calls `inference/main.py` exactly
as the competition runtime did (`DATA_DIR` environment variable, output
`./submission.csv` in a temporary working directory). No interim data files are written.

Peak resident memory on the M3 Pro CPU (`/usr/bin/time -l`) was 2.2 GB for one scan and
2.7-5.2 GB for runs of 20 to 150 scans. The upper end was measured on an idle machine with
memory to spare. For 150 scans macOS also reported peak memory footprints of 4.8-5.0 GB,
which count compressed pages. Scans are processed in blocks of up to 128, so allow about
8 GB.

On the normal path, each scan's prediction uses only that scan's voxels and the frozen
model files. The registration objective is a sum of per-scan terms, the rejection
threshold is a fixed per-scan NCC value, and all normalisation statistics are frozen
from training. The code registers scans and runs the CNNs in blocks of up to 128 scans,
so the order of floating-point sums inside those batched kernels depends on which scans
share a block. On the 150-scan local set, shuffling the input order changed predictions
by 9e-6 on average and by at most 1.3e-3 on one scan. Predicting five scans one at a
time changed them by at most 3.1e-4. In a separate check in a fresh environment, one
scan predicted alone differed by 1.5e-5 from the same scan in a 20-scan batch, and a
two-scan folder differed from that batch by at most 4.4e-5.

The error fallbacks act on a whole block. If registration raises an exception, every
scan in the block uses the canonical frame. If a CNN raises one, that member and every
later member are dropped for the whole block, and the stacker is renormalised over the
rest. Each fallback prints one static warning line. None of these warnings appeared in
the local runs. A scan that cannot be read receives the prior on its own.

**In the official runtime container.** `inference/` plus the installed
`inference/assets/` is the content of the submitted archive. Build it from the
repository root with

```bash
(cd inference && zip -r ../submission.zip . -x '__pycache__/*' 'assets/.gitkeep')
```

This gives an archive of 82 entries with the same names as the scored one (with the
released weights bundle, the 18 TorchScript files are the copies without debug
records). It runs unchanged in the competition runtime repository
(`drivendataorg/competition-sfmn-parkinsons-runtime`, `just test-submission`).
`pipeline/09_package.sh` is for use after `train_all.sh`, because it reads the training
outputs in `outputs/` and `submission_src/`.

Pitfalls:

- Files must be named `<uid>.nii.gz`. A scan that cannot be read, or whose registration
  fails the quality check, receives the prior 0.5485 by design.
- Leave every `DAT_*` environment variable unset. They switch research variants of the
  preprocessing.
- `inference/main.py` exits with code 1 if `psutil` or `huggingface-hub` is missing
  (it prints their versions, and TabICL imports both).
- Log lines are static strings (the competition forbade logging test data), apart from
  the installed package versions and, on a crash, the exception type name and the code
  locations of the traceback. No test-scan value, identifier, count or path reaches the log.

## Provenance of the scored archive

- `inference/` is byte-identical to the code in the scored archive (sha256
  `88b5611e…`). `inference/CANDIDATE.json` was written before the upload and still calls
  the package a held-local candidate. The archive with that hash is platform submission
  322397. A comment in `inference_main.py` mentions T = 1.0. The shipped configuration
  uses T = 1.4.
- `docs/PROVENANCE.md` maps every file of the archive to the script that produced it.
- The two TabICL context files (`tabular_raw.pkl.xz`, `tabular_log.pkl.xz`) cover all
  1,362 training scans (raw context) and an acquisition-balanced subset of 1,148 of them
  (log context). For every context scan they store eight 512-dimensional TabICL row
  representations with the scan's label embedding added, so the training labels can be
  recovered from them. They also store per-feature statistics of the 72 inputs: medians,
  means, scales, outlier bounds, minima and maxima. They contain no identifiers, images
  or per-scan feature rows.

## Licence

The solution code and the fine-tuned weights are released under the MIT licence
(`LICENSE`). Bundled third-party components keep their licences
(`THIRD_PARTY_NOTICES.md`): TabICL 2.2.0 and its checkpoint (BSD-3-Clause) and the timm
EfficientNet ImageNet weights (Apache-2.0) from which the 2D members were fine-tuned.
