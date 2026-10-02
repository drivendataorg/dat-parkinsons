#!/usr/bin/env bash
# 07_tabular.sh -- v17 TabICL block: acquisition groups, scale-space shape and
# spatial mixture features, recipe/context/feature order, frozen TabICL contexts.
#
# Reproduces, in order (original script -> port):
#   site_diag.py + unsaved fam2/scale step  -> tabular/derive_groups.py
#       (runs src/eda_real.py and tools/hdr_fingerprint.py when their outputs are missing)
#   experiments/20260914/scale_features.py  -> tabular/extract_shape_features.py
#   experiments/v17/mixture_shape.py        -> tabular/extract_mixture_features.py
#   experiments/v17/prepare_candidate.py    -> tabular/prepare_tabular.py
#   experiments/v17/freeze_candidate.py     -> tabular/freeze_tabular.py
#
# Inputs (regenerated mode, the default):
#   outputs/features_v13_m2fnh.csv, features_v4_m2fnh.csv, features_v13_m2f.csv, folds.csv
#   outputs/crops_v3_f2.npz (DAT_MASK_V2=1 DAT_FRAME_V2=1 cache3.py _f2)
#   linear feature lists linear_lin.json / linear_lin4.json (--linear-assets)
#   TabICL checkpoint tabicl-classifier-v2-20260212.ckpt (--ckpt, else $TABICL_CKPT, else
#   inference/assets/, where scripts/install_weights.py puts it; sha256 bdc7dbd5e4ff21f8...)
# Outputs: outputs/tabular/{acquisition_groups.csv, scale_f2.csv, mixture_shape.csv,
#   tabular_recipe.json, context.npz, feature_order_v17.json, tabular_raw.pkl.xz,
#   tabular_log.pkl.xz, persistence_checks.json, *_info.json}
#
# Archived mode (--inputs archived --archive-dir OLD_PROJECT_ROOT) reads the archived
# feature tables, scale_f2.csv and mixture_shape.csv of the original run instead and
# writes to outputs/tabular/archived_inputs/. With --archive-dir, each step is also
# compared against the archived originals (aggregates only); --ref-assets DIR (the
# assets/ dir of the scored v17 zip) enables the recipe sha256 / feature-order checks
# on its own (the context.npz check also needs --archive-dir).
#
# Usage: bash pipeline/07_tabular.sh [--inputs regenerated|archived] [--archive-dir DIR]
#          [--linear-assets DIR] [--ckpt FILE] [--ref-assets DIR] [--skip-freeze]
#        PYTHON=/path/to/python to override the interpreter.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
# a relative PYTHON (e.g. .venv-train/bin/python) is made absolute: the steps below cd first
case "$PYTHON" in */*) [ ! -d "$(dirname "$PYTHON")" ] || PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")";; esac
MODE=regenerated; ARCHIVE=""; LINEAR="$ROOT/submission_src/assets"; CKPT=""; REF_ASSETS=""; FREEZE=1
while [ $# -gt 0 ]; do
  case "$1" in
    --inputs) MODE="$2"; shift 2;;
    --archive-dir) ARCHIVE="$2"; shift 2;;
    --linear-assets) LINEAR="$2"; shift 2;;
    --ckpt) CKPT="$2"; shift 2;;
    --ref-assets) REF_ASSETS="$2"; shift 2;;
    --skip-freeze) FREEZE=0; shift;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
# Checkpoint: --ckpt, else $TABICL_CKPT, else inference/assets/, where scripts/install_weights.py
# puts it (same convention as 09_package.sh).
[ -n "$CKPT" ] || CKPT="${TABICL_CKPT:-$ROOT/inference/assets/tabicl-classifier-v2-20260212.ckpt}"

# Keep every research gate at its default: only DAT_DATA_DIR (data location) is honoured.
for v in $(env | sed -n 's/^\(DAT_[A-Z0-9_]*\)=.*/\1/p'); do
  [ "$v" = "DAT_DATA_DIR" ] || { echo "[07_tabular] unsetting $v"; unset "$v"; }
done
export KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
cd "$ROOT"
mkdir -p outputs/logs outputs/tabular

if [ "$MODE" = archived ]; then
  [ -n "$ARCHIVE" ] || { echo "--inputs archived needs --archive-dir" >&2; exit 2; }
  OUT=outputs/tabular/archived_inputs
  INPUTS="$ARCHIVE/outputs"
  SHAPE="$ARCHIVE/experiments/20260914/scale_f2.csv"
  MIX="$ARCHIVE/experiments/v17/mixture_shape.csv"
elif [ "$MODE" = regenerated ]; then
  OUT=outputs/tabular
  INPUTS=outputs
  SHAPE=outputs/tabular/scale_f2.csv
  MIX=outputs/tabular/mixture_shape.csv
else
  echo "--inputs must be regenerated or archived" >&2; exit 2
fi
mkdir -p "$OUT"

echo "[07_tabular] 1/4 acquisition groups ($MODE)"
VERIFY_GROUPS=()
[ -n "$ARCHIVE" ] && VERIFY_GROUPS=(--verify-against "$ARCHIVE/outputs/site_diag_rows.csv")
"$PYTHON" tabular/derive_groups.py --native-spacing-csv "$INPUTS/features_v13_m2f.csv" \
  --out "$OUT/acquisition_groups.csv" ${VERIFY_GROUPS[@]+"${VERIFY_GROUPS[@]}"}

if [ "$MODE" = regenerated ]; then
  echo "[07_tabular] 2/4 shape and mixture features from outputs/crops_v3_f2.npz"
  "$PYTHON" tabular/extract_shape_features.py --frame f2 --out-dir outputs/tabular | tee outputs/logs/tabular_scale_f2.log
  "$PYTHON" tabular/extract_mixture_features.py --out-dir outputs/tabular | tee outputs/logs/tabular_mixture_shape.log
  if [ -n "$ARCHIVE" ]; then
    "$PYTHON" tabular/verify_features.py \
      --pair outputs/tabular/scale_f2.csv "$ARCHIVE/experiments/20260914/scale_f2.csv" \
      --pair outputs/tabular/mixture_shape.csv "$ARCHIVE/experiments/v17/mixture_shape.csv" \
      --report outputs/tabular/verify_features_report.json
  fi
else
  echo "[07_tabular] 2/4 skipped (archived shape/mixture tables)"
fi

echo "[07_tabular] 3/4 recipe, context and feature order"
"$PYTHON" tabular/prepare_tabular.py --inputs-dir "$INPUTS" --groups-csv "$OUT/acquisition_groups.csv" \
  --shape-csv "$SHAPE" --mixture-csv "$MIX" --linear-assets "$LINEAR" --out-dir "$OUT"
if [ -n "$REF_ASSETS" ]; then
  # Recipe sha256 and feature order need only the scored archive; the context check needs the old tree.
  REF_CONTEXT=()
  [ -n "$ARCHIVE" ] && REF_CONTEXT=(--ref-context "$ARCHIVE/experiments/v17/rebuild/context.npz")
  "$PYTHON" tabular/verify_prepare.py --dir "$OUT" --ref-recipe "$REF_ASSETS/tabular_recipe.json" \
    --ref-config "$REF_ASSETS/inference_config.json" ${REF_CONTEXT[@]+"${REF_CONTEXT[@]}"} \
    --report "$OUT/verify_prepare_report.json"
fi

if [ "$FREEZE" = 1 ]; then
  echo "[07_tabular] 4/4 freeze TabICL contexts (CPU, ~3-8 min)"
  # PYTHONHASHSEED=1 fixes tabicl's set-based ensemble order to the shipped [none, power]
  # (the original run was unseeded); without it predictions differ at the 1e-6 level.
  PYTHONHASHSEED="${PYTHONHASHSEED:-1}" "$PYTHON" tabular/freeze_tabular.py --recipe "$OUT/tabular_recipe.json" --ckpt "$CKPT" \
    | tee "outputs/logs/tabular_freeze_$MODE.log"
else
  echo "[07_tabular] 4/4 freeze skipped"
fi
echo "[07_tabular] done -> $OUT"
