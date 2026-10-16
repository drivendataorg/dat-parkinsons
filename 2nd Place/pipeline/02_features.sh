#!/usr/bin/env bash
# 02_features.sh -- registered hand-crafted feature tables for lin / lin4.
#
# Reproduces the original 2026-09-02 "m2" build and the two ad-hoc filters:
#   1. stage the frozen registration template outputs/template_v6.npy
#      (sha256 4d3629d5...; src/register3.py rebuilds it byte for byte outside this
#      stage, see docs/REPRODUCIBILITY.md section 4) and
#      the narrow column header outputs/features_v4_reg7f.csv;
#   2. the original command, verbatim apart from the log path and worker count:
#        cd src && DAT_MASK_V2=1 KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1 \
#          OMP_NUM_THREADS=1 REFEAT_JOBS=4 python -u make_regfeat32.py \
#          --rich features_v13_m2.csv --narrow features_v4_m2.csv \
#          --narrow-like features_v4_reg7f.csv
#      -> outputs/features_v13_m2.csv, outputs/features_v4_m2.csv
#      (float32 preprocessing on CPU, rigid registration to the template on
#       MPS > CUDA > CPU, features13 extraction)
#   3. python src/filter_feature_tables.py
#      -> outputs/features_v{13,4}_m2f.csv, outputs/features_v{13,4}_m2fnh.csv
#
# REFEAT_JOBS (joblib workers) only changes the runtime: preprocessing and feature
# extraction are per-scan and independent. Default 3 here; the original used 4.
#
# Usage:
#   bash pipeline/02_features.sh [path/to/template_v6.npy]
#     default template source: inference/assets/template_v6.npy (shipped with v17)
#   PYTHON=/path/to/python REFEAT_JOBS=4 bash pipeline/02_features.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
# a relative PYTHON (e.g. .venv-train/bin/python) is made absolute: the steps below cd first
case "$PYTHON" in */*) [ ! -d "$(dirname "$PYTHON")" ] || PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")";; esac
JOBS="${REFEAT_JOBS:-3}"
TEMPLATE_SRC="${1:-$ROOT/inference/assets/template_v6.npy}"
TEMPLATE_SHA=4d3629d5a06aadd8d638e2333f37169de5a460524be3314944553b23fb2b0672

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

# Research gates stay at their defaults: only DAT_DATA_DIR is honoured; DAT_MASK_V2=1
# is set explicitly below because the original command set it.
for v in $(env | sed -n 's/^\(DAT_[A-Z0-9_]*\)=.*/\1/p'); do
  [ "$v" = "DAT_DATA_DIR" ] || { echo "[02_features] unsetting $v"; unset "$v"; }
done
export KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$ROOT/outputs/logs"

# 1. inputs that are not rebuilt here
if [ ! -f "$ROOT/outputs/template_v6.npy" ]; then
  [ -f "$TEMPLATE_SRC" ] || { echo "[02_features] template not found: $TEMPLATE_SRC" >&2; exit 1; }
  cp "$TEMPLATE_SRC" "$ROOT/outputs/template_v6.npy"
fi
[ "$(sha256 "$ROOT/outputs/template_v6.npy")" = "$TEMPLATE_SHA" ] \
  || { echo "[02_features] outputs/template_v6.npy sha256 mismatch" >&2; exit 1; }
cp "$ROOT/configs/feature_columns/features_v4_reg7f.header.csv" "$ROOT/outputs/features_v4_reg7f.csv"

# 2. registered float32 feature tables (original command)
t0=$SECONDS
( cd "$ROOT/src" && DAT_MASK_V2=1 OMP_NUM_THREADS=1 REFEAT_JOBS="$JOBS" \
    "$PYTHON" -u make_regfeat32.py --rich features_v13_m2.csv --narrow features_v4_m2.csv \
    --narrow-like features_v4_reg7f.csv ) 2>&1 | tee "$ROOT/outputs/logs/regfeat_m2.log"
echo "[02_features] make_regfeat32 wall time $((SECONDS - t0)) s" | tee -a "$ROOT/outputs/logs/regfeat_m2.log"

# 3. m2 -> m2f -> m2fnh (recovered ad-hoc filters)
"$PYTHON" "$ROOT/src/filter_feature_tables.py" --tag m2
echo "[02_features] done"
