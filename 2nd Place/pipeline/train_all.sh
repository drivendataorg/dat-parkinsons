#!/usr/bin/env bash
# train_all.sh -- raw competition data -> the v17 submission package, in one command.
#
#   MODE=full (default)  trains every model from scratch: registered features and the
#                        two L1 models, all 34 3D-CNN and 18 2D-CNN checkpoints, the
#                        stacker, the TabICL block, then packages the submission.
#                        About 6.4 h on an Apple M3 Pro (MPS); see README "Hardware".
#   MODE=fast            identical, except the CNNs are NOT retrained: the released
#                        checkpoints are copied in (3D from inference/assets/, raw 2D
#                        state dicts from models/cnn_raw/) and their out-of-fold
#                        predictions are replayed (tools/replay_cnn_oof.py). About
#                        32 min on the same machine (measured). models/cnn_raw/ is
#                        created by: unzip navattack_v17_cnn_raw_checkpoints.zip -d models/
#
# Both modes need (see README "Setup"):
#   data/raw/niftis/<uid>.nii.gz and data/raw/labels.csv    (competition training data)
#   inference/assets/ populated by scripts/install_weights.py  (for the frozen
#       registration template template_v6.npy and the public TabICLv2 checkpoint;
#       MODE=fast also takes the 3D and TorchScript 2D checkpoints from there)
#   network once in MODE=full, to download the two ImageNet EfficientNet weights
#
# Output: outputs/v17/submission.zip (+ MANIFEST.json, submission/ directory).
# Optional: CHECK_AGAINST=/path/to/scored/submission.zip compares every archive entry
# (name, size, CRC32) with the scored v17 archive.
#
# Usage:  PYTHON=/path/to/python3.12 [MODE=fast] bash pipeline/train_all.sh
# A relative PYTHON (e.g. .venv-train/bin/python) is made absolute before the cd below,
# and every stage inherits that absolute path.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
case "$PYTHON" in */*) [ ! -d "$(dirname "$PYTHON")" ] || PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")";; esac
export PYTHON
cd "$ROOT"
MODE="${MODE:-full}"
export TABICL_CKPT="${TABICL_CKPT:-$ROOT/inference/assets/tabicl-classifier-v2-20260212.ckpt}"
# research switches of the preprocessing code must stay at their defaults
for v in $(env | sed -n 's/^\(DAT_[A-Z0-9_]*\)=.*/\1/p'); do
  [ "$v" = "DAT_DATA_DIR" ] || { echo "[train_all] unsetting $v"; unset "$v"; }
done
mkdir -p outputs/logs
LOG=outputs/logs/train_all.log
exec > >(tee -a "$LOG") 2>&1
T0=$(date +%s)
step () { echo; echo "================ [train_all:$MODE] $* ($(( $(date +%s) - T0 )) s elapsed) ================"; }

case "$MODE" in full|fast) ;; *) echo "MODE must be full or fast"; exit 2;; esac
[ -f inference/assets/template_v6.npy ] && [ -f "$TABICL_CKPT" ] || {
  echo "missing inference/assets/template_v6.npy or the TabICL checkpoint: run scripts/install_weights.py first"; exit 1; }

step "00 verify staged data";           bash pipeline/00_stage_data.sh
step "01 folds";                        bash pipeline/01_folds.sh
step "02 registered feature tables";    bash pipeline/02_features.sh
step "03 linear members lin / lin4";    bash pipeline/03_linear.sh
step "04 CNN input caches";             bash pipeline/04_caches.sh

if [ "$MODE" = full ]; then
  step "05 3D hemisphere-MIL CNN (34 checkpoints)"; bash pipeline/05_cnn3d.sh
  step "06 2D EfficientNet members (18 checkpoints)"; bash pipeline/06_cnn2d.sh
else
  step "05/06 fast mode: stage released CNN checkpoints"
  n=0
  stage () {   # stage SRC: copy into outputs/ unless a file of that name is already there
    if [ ! -f "$1" ]; then
      echo "missing $1 (see README, Setup step 2)"
      case "$1" in models/cnn_raw/*) echo "the raw 2D checkpoints come from: unzip navattack_v17_cnn_raw_checkpoints.zip -d models/";; esac
      exit 1
    fi
    [ -e "outputs/$(basename "$1")" ] || { cp "$1" outputs/; n=$((n+1)); }
  }
  for s in mil48a mil48b m31000 m33000 m35000 m37000; do for k in 0 1 2 3 4; do
    stage "inference/assets/${s}_fold${k}.pt"; done; done
  for s in mf21000 mf23000 mf25000 mf27000; do stage "inference/assets/${s}_full0.pt"; done
  for s in e2dmipf2 e2dmipf22 e2db3f2; do for k in fold0 fold1 fold2 fold3 fold4 full0; do
    stage "models/cnn_raw/${s}_${k}.pt"; stage "inference/assets/${s}_${k}.ts.pt"; done; done
  echo "staged $n checkpoint files into outputs/ (files already there were kept)"
  step "05/06 fast mode: replay out-of-fold predictions"
  OMP_NUM_THREADS=3 KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1 "$PYTHON" -u tools/replay_cnn_oof.py
  ( cd src && "$PYTHON" -u avg_member.py mil48y mil48a mil48b m31000 m33000 )
  "$PYTHON" -u src/avg_2d_seeds.py
fi

step "07 TabICL block";                 bash pipeline/07_tabular.sh --ckpt "$TABICL_CKPT"
step "08 stacker + image package spec"; bash pipeline/08_stack.sh
step "09 package submission.zip"
if [ -n "${CHECK_AGAINST:-}" ]; then
  bash pipeline/09_package.sh --overwrite --check-against "$CHECK_AGAINST" || echo "[train_all] some entries differ from the reference archive (expected for the re-frozen TabICL contexts and any retrained or scrubbed checkpoint; see docs/stages/08-09_stack_and_package.md)"
else
  bash pipeline/09_package.sh --overwrite
fi
step "done: outputs/v17/submission.zip"
