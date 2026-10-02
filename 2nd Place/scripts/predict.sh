#!/usr/bin/env bash
# Run v17 inference on a data directory laid out like the competition test set:
#   <DATA_DIR>/submission_format.csv   (columns uid,is_pathologic; values are ignored)
#   <DATA_DIR>/niftis/<uid>.nii.gz
# Writes the predictions to OUT_CSV (default: submission.csv in the current directory).
#
# Usage (from anywhere):  bash scripts/predict.sh /path/to/DATA_DIR [OUT_CSV]
# Needs the inference environment (requirements-inference.txt) and weights installed
# with scripts/install_weights.py. Runs on CUDA if available, otherwise CPU; no network.
# Interpreter: $PYTHON if set, else .venv-infer/bin/python in the repository if it
# exists, else `python` on PATH. A relative PYTHON (e.g. .venv-infer/bin/python) is
# made absolute before the script changes directory.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$(cd "${1:?usage: predict.sh DATA_DIR [OUT_CSV]}" && pwd)"
OUT_CSV="${2:-submission.csv}"
case "$OUT_CSV" in /*) ;; *) OUT_CSV="$(pwd)/$OUT_CSV" ;; esac
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$ROOT/.venv-infer/bin/python" ]; then PYTHON="$ROOT/.venv-infer/bin/python"; else PYTHON=python; fi
fi
case "$PYTHON" in */*) [ ! -d "$(dirname "$PYTHON")" ] || PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")";; esac

[ -f "$DATA_DIR/submission_format.csv" ] || { echo "missing $DATA_DIR/submission_format.csv" >&2; exit 1; }
[ -d "$DATA_DIR/niftis" ] || { echo "missing $DATA_DIR/niftis/" >&2; exit 1; }
[ -d "$(dirname "$OUT_CSV")" ] || { echo "output directory $(dirname "$OUT_CSV") does not exist" >&2; exit 1; }
command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "python interpreter '$PYTHON' not found: activate .venv-infer or set PYTHON=/path/to/python (see README, Setup)" >&2; exit 1; }
"$PYTHON" -c "import torch, sklearn, nibabel" >/dev/null 2>&1 || {
  echo "'$PYTHON' cannot import torch, scikit-learn and nibabel: install requirements-inference.txt into it (see README, Setup)" >&2; exit 1; }
"$PYTHON" "$ROOT/scripts/install_weights.py" --verify-only >/dev/null || {
  echo "weights missing or corrupt: run scripts/install_weights.py first" >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
# The entry point writes ./submission.csv into its working directory.
( cd "$WORK" && DATA_DIR="$DATA_DIR" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON" "$ROOT/inference/main.py" )
mv "$WORK/submission.csv" "$OUT_CSV"
echo "wrote $OUT_CSV"
