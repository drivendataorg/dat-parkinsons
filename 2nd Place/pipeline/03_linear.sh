#!/usr/bin/env bash
# 03_linear.sh -- the two L1-logistic stack members, lin and lin4.
#
#   (a) python src/linear_oof.py
#       -> outputs/oof_feat3_m2fnh.csv (5-fold OOF of lin/lin4; the stacker's --feat input).
#          Reproduces the ad-hoc heredoc of 2026-09-03 04:27 UTC.
#   (b) the original export, tools/build_clean.sh line 17 (v12b build; repeated
#       byte-identically for v13b, v13c and v14c):
#         cd src && python -u train_final3.py --rich-features features_v13_m2fnh.csv \
#                  --narrow-features features_v4_m2fnh.csv --export
#       with KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=3 HF_HUB_OFFLINE=1
#       -> submission_src/assets/linear_lin.json, linear_lin4.json (shipped in v17 as
#          assets/linear_lin*.json), plus side outputs v17 does not ship:
#          submission_src/assets/lgbm_gbm_s*.txt(.meta.json), mlp_mlp_s*.json,
#          outputs/oof_feat3.csv, outputs/feature_order.json, outputs/feature_union.json
#   (c) the column-set assertion that followed it in build_clean.sh (235/94, no head_ml/hole_ml).
#
# train_final3 also fits LightGBM (5 seeds x 700 rounds, num_threads=4 hard-coded in
# src/train_final3.py) and an MLP bag for every fold; the whole step takes about 1 min on an Apple M3 Pro.
#
# Usage: bash pipeline/03_linear.sh        (PYTHON=/path/to/python to override)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
# a relative PYTHON (e.g. .venv-train/bin/python) is made absolute: the steps below cd first
case "$PYTHON" in */*) [ ! -d "$(dirname "$PYTHON")" ] || PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")";; esac

# Research gates stay at their defaults (DAT_ADMIT_HEAD_ML in particular must be unset).
for v in $(env | sed -n 's/^\(DAT_[A-Z0-9_]*\)=.*/\1/p'); do
  [ "$v" = "DAT_DATA_DIR" ] || { echo "[03_linear] unsetting $v"; unset "$v"; }
done
export KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=3 HF_HUB_OFFLINE=1
mkdir -p "$ROOT/outputs/logs" "$ROOT/submission_src/assets"

# (a) cross-fitted OOF for the stacker
t0=$SECONDS
"$PYTHON" "$ROOT/src/linear_oof.py" 2>&1 | { grep -v -i warn || true; } | tee "$ROOT/outputs/logs/linear_oof.log"
echo "[03_linear] linear_oof wall time $((SECONDS - t0)) s" | tee -a "$ROOT/outputs/logs/linear_oof.log"

# (b) full-data fit and export (original command)
t0=$SECONDS
set +e
( cd "$ROOT/src" && "$PYTHON" -u train_final3.py --rich-features features_v13_m2fnh.csv \
    --narrow-features features_v4_m2fnh.csv --export ) 2>&1 | { grep -v -i warn || true; } \
    | tee "$ROOT/outputs/logs/train_final3_export.log"
rc=${PIPESTATUS[0]}
set -e
[ "$rc" = 0 ] || { echo "[03_linear] train_final3.py exited with status $rc" >&2; exit 1; }
echo "[03_linear] train_final3 wall time $((SECONDS - t0)) s" | tee -a "$ROOT/outputs/logs/train_final3_export.log"
grep -q "exported: L1 kept" "$ROOT/outputs/logs/train_final3_export.log" \
  || { echo "[03_linear] train_final3 export failed" >&2; exit 1; }

# (c) the build_clean.sh assertion
( cd "$ROOT" && "$PYTHON" - <<'PY'
import json
for f, n in [('submission_src/assets/linear_lin.json', 235), ('submission_src/assets/linear_lin4.json', 94)]:
    j = json.load(open(f))
    assert 'head_ml' not in j['features'] and 'hole_ml' not in j['features'], f
    assert len(j['features']) == n, (f, len(j['features']))
    print(f, len(j['features']), 'features, no head_ml/hole_ml,',
          sum(c != 0 for c in j['coef']), 'non-zero coefficients')
PY
)
echo "[03_linear] done"
