#!/usr/bin/env bash
# 06_cnn2d.sh -- train the 2D EfficientNet members (18 checkpoints), write the
# two-seed OOF the stacker was fitted on, and export TorchScript.
#
# Runs (exact arguments of the original 2026-09-08 runs, verified against
# outputs/cv_<tag>.json and the *_full.log headers; cwd src/):
#   e2dmipf2        tools/gpu_queue2.sh:27   B0 5-fold, seed-base 9000
#   e2dmipf22       tools/f2seed2_chain.sh:6 B0 5-fold, seed-base 9100
#   e2dmipf2:full   tools/b3_chain.sh:8      B0 --full,  seed 9099
#   e2dmipf22:full  tools/b3_chain.sh:9      B0 --full,  seed 9199
#   e2db3f2         tools/post_b3_chain.sh:14 B3 5-fold, seed-base 9000
#   e2db3f2:full    tools/v14c_chain.sh:5    B3 --full,  seed 9099
# B0 = efficientnet_b0.ra_in1k at 192 px (cnn2d.py default backbone);
# B3 = efficientnet_b3.ra2_in1k at 224 px. Every run: --crops crops_v3_f2.npz
# --key aligned --views mip3 --epochs 30 --lr 1e-4 --aug 1.0 (other args default).
# Outputs: outputs/<tag>_fold{0-4}.pt + oof_<tag>.csv + cv_<tag>.json, or <tag>_full0.pt.
#
# Then:
#   python src/avg_2d_seeds.py      -> outputs/oof_e2dmipf2x.csv = 0.5*(e2dmipf2 + e2dmipf22)
#     (the package member named 'e2dmipf2' averages the 12 checkpoints of BOTH seeds;
#      its stacker weight was fitted on this two-seed OOF; e2db3f2's on oof_e2db3f2.csv)
#   python tools/ts_export.py <tag> -> outputs/<tag>_{fold0-4,full0}.ts.pt + <tag>_meta.json
#     (from the repo root, for e2dmipf2, e2dmipf22, e2db3f2; tools/v14b_chain.sh:6,
#      tools/v14c_chain.sh:7). The .ts.pt files are what inference loads.
#
# ImageNet weights: fetched from the Hugging Face Hub by timm on first use and
# checked against pinned sha256 values by tools/fetch_backbones.py (step 0).
# Network is needed once; afterwards `export HF_HUB_OFFLINE=1` (as the original
# chains did) is fine.
#
# Env of the original chains: KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1
# OMP_NUM_THREADS=2 (3 for the B3 full run) HF_HUB_OFFLINE=1; gpu_queue2.sh also
# exported DAT_MASK_V2=1, which cnn2d.py never reads (it does not import
# preprocessing3), so it is not set here.
#
# Prerequisites: outputs/folds.csv (01_folds.sh), outputs/crops_v3_f2.npz (04_caches.sh).
# Runtime (original, Apple M3 Pro MPS): B0 5-fold ~1500 s, B0 full ~373 s,
# B3 5-fold ~3900 s, B3 full ~968 s, export ~1 min per tag. Not bit-reproducible.
#
# Environment knobs:
#   EPOCHS=N        override --epochs (default: the original 30)
#   TAG_PREFIX=p_   tags become p_<tag> so a rerun never clobbers (default: none)
#   RUNS="..."      subset/order of runs (default: all six above, original order)
#   EXPORT=0        skip the TorchScript export
#   FORCE=1         allow overwriting existing <tag>_fold*.pt / <tag>_full0.pt
#   PYTHON=...      interpreter (default: python)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
# a relative PYTHON (e.g. .venv-train/bin/python) is made absolute: the steps below cd first
case "$PYTHON" in */*) [ ! -d "$(dirname "$PYTHON")" ] || PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")";; esac
EPOCHS="${EPOCHS:-30}"
P="${TAG_PREFIX:-}"
FORCE="${FORCE:-0}"
EXPORT="${EXPORT:-1}"
RUNS="${RUNS:-e2dmipf2 e2dmipf22 e2dmipf2:full e2dmipf22:full e2db3f2 e2db3f2:full}"

for v in $(env | sed -n 's/^\(DAT_[A-Z0-9_]*\)=.*/\1/p'); do
  [ "$v" = "DAT_DATA_DIR" ] || { echo "[06_cnn2d] unsetting $v"; unset "$v"; }
done
export KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$ROOT/outputs/logs"
for f in folds.csv crops_v3_f2.npz; do
  [ -f "$ROOT/outputs/$f" ] || { echo "[06_cnn2d] missing outputs/$f (run pipeline/01_folds.sh and 04_caches.sh)"; exit 1; }
done

# step 0: ImageNet backbones (download once, verify pinned sha256)
BLOG="$ROOT/outputs/logs/${P}fetch_backbones.log"
if ! ( cd "$ROOT" && OMP_NUM_THREADS=2 "$PYTHON" tools/fetch_backbones.py ) > "$BLOG" 2>&1; then
  grep '^\[backbones\]' "$BLOG" || tail -5 "$BLOG"
  echo "[06_cnn2d] backbone download/check failed (log: outputs/logs/$(basename "$BLOG"))"; exit 1
fi
grep '^\[backbones\]' "$BLOG" || true

COMMON=(--crops crops_v3_f2.npz --key aligned --views mip3 --epochs "$EPOCHS" --lr 1e-4 --aug 1.0)

guard () {
  local tag="$1"; shift
  local hit
  for g in "$@"; do
    for hit in "$ROOT"/outputs/$g; do
      if [ -e "$hit" ] && [ "$FORCE" != "1" ]; then
        echo "[06_cnn2d] refusing to overwrite outputs/$(basename "$hit") (tag $tag); set FORCE=1 or TAG_PREFIX=..."
        exit 3
      fi
    done
  done
}

train () {   # train <run>   (run = <name> or <name>:full)
  local run="$1" name="${1%%:full}" args omp=2 full=() sfx=""
  case "$name" in
    e2dmipf2)  args=(--res 192 --seed-base 9000) ;;
    e2dmipf22) args=(--res 192 --seed-base 9100) ;;
    e2db3f2)   args=(--res 224 --backbone efficientnet_b3.ra2_in1k --seed-base 9000) ;;
    *) echo "[06_cnn2d] unknown run '$run'"; exit 2 ;;
  esac
  if [ "$run" != "$name" ]; then
    full=(--full); sfx="_full"
    [ "$name" = e2db3f2 ] && omp=3          # tools/v14c_chain.sh exported OMP_NUM_THREADS=3
  fi
  local tag="${P}${name}" log="$ROOT/outputs/logs/${P}${name}${sfx}.log"
  if [ -n "$sfx" ]; then guard "$tag" "${tag}_full0.pt"; else guard "$tag" "${tag}_fold*.pt"; fi
  echo "[06_cnn2d] $tag: cnn2d.py --tag $tag ${COMMON[*]} ${args[*]} ${full[*]:-} ($(date))"
  local t0=$SECONDS
  ( cd "$ROOT/src" && OMP_NUM_THREADS="${OMP_NUM_THREADS:-$omp}" \
      "$PYTHON" -u cnn2d.py --tag "$tag" "${COMMON[@]}" "${args[@]}" ${full[@]+"${full[@]}"} ) > "$log" 2>&1
  grep -E '\] (fold [0-9]: ll|OOF|full-data)' "$log" || true
  echo "[06_cnn2d] $tag${sfx} done in $((SECONDS - t0)) s"
}

for run in $RUNS; do train "$run"; done

# two-seed OOF of the B0 reader (stacker input for package member 'e2dmipf2')
if [ -f "$ROOT/outputs/oof_${P}e2dmipf2.csv" ] && [ -f "$ROOT/outputs/oof_${P}e2dmipf22.csv" ]; then
  ( cd "$ROOT" && "$PYTHON" src/avg_2d_seeds.py "${P}e2dmipf2x" "${P}e2dmipf2" "${P}e2dmipf22" ) \
    2>&1 | tee "$ROOT/outputs/logs/${P}e2dmipf2x_avg.log"
else
  echo "[06_cnn2d] avg_2d_seeds skipped: needs oof_${P}e2dmipf2.csv and oof_${P}e2dmipf22.csv"
fi

# TorchScript export (needs all six checkpoints of a tag)
if [ "$EXPORT" = 1 ]; then
  for name in e2dmipf2 e2dmipf22 e2db3f2; do
    tag="${P}${name}" have=1
    for part in fold0 fold1 fold2 fold3 fold4 full0; do
      [ -f "$ROOT/outputs/${tag}_${part}.pt" ] || have=0
    done
    if [ "$have" = 1 ]; then
      echo "[06_cnn2d] ts_export $tag ($(date))"
      t0=$SECONDS
      ( cd "$ROOT" && OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" "$PYTHON" tools/ts_export.py "$tag" ) \
        > "$ROOT/outputs/logs/${tag}_ts_export.log" 2>&1
      { grep -v -i warn "$ROOT/outputs/logs/${tag}_ts_export.log" || true; } | tail -7
      echo "[06_cnn2d] ts_export $tag done in $((SECONDS - t0)) s"
    else
      echo "[06_cnn2d] ts_export $tag skipped: not all of ${tag}_{fold0-4,full0}.pt exist"
    fi
  done
fi
