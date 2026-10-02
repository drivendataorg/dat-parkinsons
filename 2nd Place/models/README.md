# Model weights

The trained weights are distributed separately from the code. `scripts/install_weights.py`
installs them into `inference/assets/` and checks every file against one of the two
checksum lists in the repository's `models/` folder. The weights bundle carries only
`SHA256SUMS.public.txt`.

| file | content |
|---|---|
| `SHA256SUMS.scored.txt` | sha256 of the 60 asset files exactly as they are in the scored archive (submission 322397) |
| `SHA256SUMS.public.txt` | the same list for the released weights bundle: identical except for the 18 TorchScript files, whose `*.debug_pkl` records were removed with `tools/scrub_torchscript_debug.py` (outputs are bit-identical) |

Bundles:

| bundle | size | content | needed for |
|---|---:|---|---|
| `navattack_v17_weights.zip` | 1.14 GB | the 60 files of `inference/assets/`: 34 3D-CNN checkpoints, 18 TorchScript 2D-CNN checkpoints, the two linear models, the registration template, the TabICLv2 checkpoint, the TabICL recipe and the two frozen TabICL contexts, `inference_config.json` | inference, and training in both modes (template, TabICL checkpoint) |
| `navattack_v17_cnn_raw_checkpoints.zip` | 456 MB | the 18 raw 2D state dicts (`e2dmipf2_*`, `e2dmipf22_*`, `e2db3f2_*` for folds 0-4 and the full-data model) from which the TorchScript files were exported | `MODE=fast` training only. The zip already contains the `cnn_raw/` folder. From the repository root run `unzip navattack_v17_cnn_raw_checkpoints.zip -d models/`, which creates `models/cnn_raw/`. |

The scored archive `submission.zip` itself can replace the first bundle.

The fine-tuned weights are released under the MIT licence (`LICENSE`). The pretrained
components they derive from keep their own licences (`THIRD_PARTY_NOTICES.md`).

The two frozen TabICL contexts (`tabular_raw.pkl.xz`, `tabular_log.pkl.xz`) cover all
1,362 training scans (raw context) and an acquisition-balanced subset of 1,148 of them
(log context). For every context scan they store eight 512-dimensional TabICL row
representations with the scan's label embedding added, so the training labels can be
recovered from them. They also store per-feature statistics of the 72 inputs: medians,
means, scales, outlier bounds, minima and maxima. They contain no identifiers, images or
per-scan feature rows.
