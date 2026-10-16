#!/usr/bin/env bash
# 01_folds.sh -- shared 5-fold CV assignment (outputs/folds.csv).
#
# Reproduces the original `cd src && python folds.py` (src/folds.py __main__:
# StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on the labels.csv
# row order). Every OOF member (lin, lin4, CNNs, stacker) uses this file.
#
# Note: src/folds.py returns an existing outputs/folds.csv unchanged when its uid
# set and fold count already match, so a re-run never reshuffles folds. Delete
# outputs/folds.csv to force regeneration.
#
# Usage: bash pipeline/01_folds.sh        (PYTHON=/path/to/python to override)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
# a relative PYTHON (e.g. .venv-train/bin/python) is made absolute: the steps below cd first
case "$PYTHON" in */*) [ ! -d "$(dirname "$PYTHON")" ] || PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")";; esac

# Keep every research gate at its default: only DAT_DATA_DIR (data location) is honoured.
for v in $(env | sed -n 's/^\(DAT_[A-Z0-9_]*\)=.*/\1/p'); do
  [ "$v" = "DAT_DATA_DIR" ] || { echo "[01_folds] unsetting $v"; unset "$v"; }
done
export KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1

mkdir -p "$ROOT/outputs"
( cd "$ROOT/src" && "$PYTHON" folds.py )
