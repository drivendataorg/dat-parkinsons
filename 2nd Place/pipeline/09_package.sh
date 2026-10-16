#!/usr/bin/env bash
# 09_package.sh -- assemble the final v17 submission (directory + submission.zip).
#
# Reproduces experiments/v17/package_candidate.py on a stage built from:
#   inference/                     verbatim scored code, CANDIDATE.json, tabicl_lib.zip, licences
#   submission_src/assets/         55 image/linear/template assets + inference_config.json
#                                  (pipeline/08_stack.sh, build3.py)
#   outputs/tabular/               tabular_recipe.json, tabular_{raw,log}.pkl.xz and
#                                  feature_order_v17.json (tabular stage)
#   inference/assets/tabicl-classifier-v2-20260212.ckpt   public TabICLv2 checkpoint
# Writes outputs/v17/{submission/,submission.zip,MANIFEST.json}, then runs the
# static gates (src/check_package.py): log-compliance grep, shallow
# /code_execution import of scale_features.py and mixture_features.py, compile
# of every .py.
#
# Usage:  bash pipeline/09_package.sh [extra package_submission.py args]
#   e.g.  bash pipeline/09_package.sh --overwrite --check-against /path/to/scored/submission.zip
#         Against the scored v17 zip, expect every entry to match except tabular_raw.pkl.xz and
#         tabular_log.pkl.xz (re-frozen contexts: same predictions, new pickle bytes), the 18
#         .ts.pt if the public weights bundle was installed (debug records removed), and any
#         retrained checkpoint. The script then exits 1 after writing the zip; compare
#         predictions with scripts/predict.sh.
# Optional env: TABICL_CKPT=<path> overrides the checkpoint location.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
# a relative PYTHON (e.g. .venv-train/bin/python) is made absolute: the steps below cd first
case "$PYTHON" in */*) [ ! -d "$(dirname "$PYTHON")" ] || PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")";; esac
cd "$ROOT"
export KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=3
mkdir -p outputs/logs
exec > >(tee -a outputs/logs/09_package.log) 2>&1
echo "########## 09_package start $(date) ##########"
"$PYTHON" -u src/package_submission.py \
    --inference-dir inference \
    --build-assets submission_src/assets \
    --tabular-dir outputs/tabular \
    --feature-order outputs/tabular/feature_order_v17.json \
    --tabicl-ckpt "${TABICL_CKPT:-inference/assets/tabicl-classifier-v2-20260212.ckpt}" \
    --out-dir outputs/v17 "$@"
echo "09_package done $(date)"
