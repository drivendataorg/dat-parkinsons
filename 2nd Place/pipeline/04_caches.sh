#!/usr/bin/env bash
# 04_caches.sh -- the two CNN input caches (48^3 / 2 mm / 96 mm SBR crops, float16).
#
#  (a) outputs/crops_v3_b.npz  -- the 3D HemiMIL training cache (member mil48y).
#      Original: `cd src && python cache3.py _b`, run on Jul 24 BEFORE the disc
#      guard existed in preprocessing3.head_mask. The guard is now ON by default
#      (DAT_MASK_V2 defaults to "1"), so the unguarded Jul-24 path is selected
#      with DAT_MASK_V2=0. Every other DAT_* gate stays unset (defaults).
#  (b) outputs/crops_v3_f2.npz -- the 2D EfficientNet training cache (members
#      e2dmipf2 / e2dmipf22 / e2db3f2): disc guard ON + frame guard ON.
#      Original (tools/frame_cpu_chain.sh): `export OMP_NUM_THREADS=2
#      DAT_MASK_V2=1 DAT_FRAME_V2=1; cd src && python -u cache3.py _f2`.
#
# Side outputs of cache3.py (not used by the CNNs): features_v{4,5,6}_<tag>.csv
# and qc_v3_<tag>.csv in outputs/.
#
# Runtime: ~6-15 min per cache (joblib, 3 CPU workers). Peak RAM ~2 GB.
#
# Usage:  bash pipeline/04_caches.sh             # both caches
#         ONLY=b  bash pipeline/04_caches.sh     # only crops_v3_b
#         ONLY=f2 bash pipeline/04_caches.sh     # only crops_v3_f2
#         FORCE=1 ...                            # rebuild even if the .npz exists
#         PYTHON=/path/to/python ...             # interpreter (default: python)
# Data location: data/raw/{niftis,labels.csv} or DAT_DATA_DIR.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
# a relative PYTHON (e.g. .venv-train/bin/python) is made absolute: the steps below cd first
case "$PYTHON" in */*) [ ! -d "$(dirname "$PYTHON")" ] || PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")";; esac
ONLY="${ONLY:-b f2}"
FORCE="${FORCE:-0}"

# Research gates stay at their defaults: only DAT_DATA_DIR (data location) is kept.
for v in $(env | sed -n 's/^\(DAT_[A-Z0-9_]*\)=.*/\1/p'); do
  [ "$v" = "DAT_DATA_DIR" ] || { echo "[04_caches] unsetting $v"; unset "$v"; }
done
export KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$ROOT/outputs/logs"

build () {   # build <tag> <env assignments...>
  local tag="$1"; shift
  local out="$ROOT/outputs/crops_v3${tag}.npz" log="$ROOT/outputs/logs/cache3${tag}.log"
  if [ -f "$out" ] && [ "$FORCE" != "1" ]; then
    echo "[04_caches] $(basename "$out") exists -- skipping (FORCE=1 rebuilds)"; return 0
  fi
  echo "[04_caches] building $(basename "$out") with $* ($(date))"
  local t0=$SECONDS
  ( cd "$ROOT/src" && env "$@" "$PYTHON" -u cache3.py "$tag" ) > "$log" 2>&1
  grep -E '^\[cache3\]' "$log" || true
  echo "[04_caches] $(basename "$out") done in $((SECONDS - t0)) s (log: outputs/logs/$(basename "$log"))"
}

for which in $ONLY; do
  case "$which" in
    b)  build _b  DAT_MASK_V2=0 OMP_NUM_THREADS=2 ;;
    f2) build _f2 DAT_MASK_V2=1 DAT_FRAME_V2=1 OMP_NUM_THREADS=2 ;;
    *)  echo "[04_caches] unknown cache '$which' (use b or f2)"; exit 2 ;;
  esac
done
