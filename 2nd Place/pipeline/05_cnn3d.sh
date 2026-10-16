#!/usr/bin/env bash
# 05_cnn3d.sh -- train the 3D HemiMIL member 'mil48y' (34 checkpoints) and write
# the 4-run OOF the v17 stacker was fitted on.
#
# The 10 run families, with the exact arguments of the original runs (verified
# against outputs/cv_<tag>.json and the checkpoint metadata; see
# docs/stages/04-06_cnn.md for provenance):
#
#   family   seed-base  kind    original script (cwd src/ unless noted)
#   mil48a   3000       5-fold  fleet3.sh      (Jul 24; env PYTORCH_ENABLE_MPS_FALLBACK=1)
#   mil48b   7000       5-fold  fleet5.sh      (Jul 24; same env)
#   mf21000  21000      --full  run_wave3.sh   (Jul 25; + KMP_DUPLICATE_LIB_OK, OMP_NUM_THREADS=2)
#   mf23000  23000      --full  run_wave3.sh
#   mf25000  25000      --full  run_wave3.sh
#   mf27000  27000      --full  run_wave3.sh
#   m31000   31000      5-fold  run_seeds.sh   (Jul 25; same env as run_wave3)
#   m33000   33000      5-fold  run_seeds.sh
#   m35000   35000      5-fold  night_queue3.sh (Aug 22; ran `python -u src/cnn3.py` from the
#   m37000   37000      5-fold  night_queue3.sh  repo root -- cwd does not affect cnn3.py)
#
# Every run: --arch mil --size 48 --key aligned --crops crops_v3_b.npz --lr 1e-3
# --epochs 20 (all other cnn3.py defaults). Fold k uses seed seed_base+k; a --full
# model uses seed seed_base+99 and trains on all 1362 rows.
# Outputs per 5-fold run: outputs/<tag>_fold{0-4}.pt, oof_<tag>.csv, cv_<tag>.json.
# Outputs per --full run: outputs/<tag>_full0.pt.
#
# Then the member OOF: (cd src && python avg_member.py mil48y mil48a mil48b m31000 m33000)
# -> outputs/oof_mil48y.csv. This is the 4-run average (written Jul 25, before
# m35000/m37000 existed) and it is the file the v14c/v17 stacker was fitted on.
# The SHIPPED member averages more: all 6 five-fold runs (30 checkpoints) plus the
# 4 full-data models = 34 checkpoints (src/build3.py --cnn-tags
# mil48y=mil48a+mil48b+m31000+m33000+m35000+m37000+mf21000+mf23000+mf25000+mf27000
# --use-full). The full-data models cannot contribute to an OOF.
#
# Prerequisites: outputs/folds.csv (pipeline/01_folds.sh) and outputs/crops_v3_b.npz
# (pipeline/04_caches.sh).
#
# Runtime (original, Apple M3 Pro MPS): ~1690-1820 s per 5-fold run, ~420 s per
# full-data model; ~3.35 h for all ten. Training is not bit-reproducible across
# devices (MPS/CUDA/CPU); expect per-run OOF within the seed spread (~0.001-0.003).
#
# Environment knobs:
#   EPOCHS=N        override --epochs (default: the original 20)
#   TAG_PREFIX=p_   tags become p_<tag> (e.g. p_mil48a_fold0.pt) so a rerun never
#                   clobbers another set of checkpoints (default: none)
#   RUNS="..."      subset/order of families (default: all ten, original order)
#   FORCE=1         allow overwriting existing <tag>_fold*.pt / <tag>_full0.pt
#   PYTHON=...      interpreter (default: python)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
# a relative PYTHON (e.g. .venv-train/bin/python) is made absolute: the steps below cd first
case "$PYTHON" in */*) [ ! -d "$(dirname "$PYTHON")" ] || PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")";; esac
EPOCHS="${EPOCHS:-20}"
P="${TAG_PREFIX:-}"
FORCE="${FORCE:-0}"
RUNS="${RUNS:-mil48a mil48b mf21000 mf23000 mf25000 mf27000 m31000 m33000 m35000 m37000}"

# Research gates stay at their defaults: only DAT_DATA_DIR (data location) is kept.
for v in $(env | sed -n 's/^\(DAT_[A-Z0-9_]*\)=.*/\1/p'); do
  [ "$v" = "DAT_DATA_DIR" ] || { echo "[05_cnn3d] unsetting $v"; unset "$v"; }
done
export KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
mkdir -p "$ROOT/outputs/logs"
for f in folds.csv crops_v3_b.npz; do
  [ -f "$ROOT/outputs/$f" ] || { echo "[05_cnn3d] missing outputs/$f (run pipeline/01_folds.sh and 04_caches.sh)"; exit 1; }
done

COMMON=(--arch mil --size 48 --key aligned --crops crops_v3_b.npz --lr 1e-3 --epochs "$EPOCHS")

guard () {   # guard <tag> <glob...>: refuse to overwrite checkpoints unless FORCE=1
  local tag="$1"; shift
  local hit
  for g in "$@"; do
    for hit in "$ROOT"/outputs/$g; do
      if [ -e "$hit" ] && [ "$FORCE" != "1" ]; then
        echo "[05_cnn3d] refusing to overwrite outputs/$(basename "$hit") (tag $tag); set FORCE=1 or TAG_PREFIX=..."
        exit 3
      fi
    done
  done
}

train () {   # train <family>
  local fam="$1" seed full=()
  case "$fam" in
    mil48a) seed=3000 ;;  mil48b) seed=7000 ;;
    m31000|m33000|m35000|m37000) seed="${fam#m}" ;;
    mf21000|mf23000|mf25000|mf27000) seed="${fam#mf}"; full=(--full) ;;
    *) echo "[05_cnn3d] unknown family '$fam'"; exit 2 ;;
  esac
  local tag="${P}${fam}" log="$ROOT/outputs/logs/${P}${fam}.log"
  if [ ${#full[@]} -gt 0 ]; then guard "$tag" "${tag}_full0.pt"; else guard "$tag" "${tag}_fold*.pt"; fi
  echo "[05_cnn3d] $tag: cnn3.py --tag $tag ${COMMON[*]} --seed-base $seed ${full[*]:-} ($(date))"
  local t0=$SECONDS
  ( cd "$ROOT/src" && "$PYTHON" -u cnn3.py --tag "$tag" "${COMMON[@]}" --seed-base "$seed" ${full[@]+"${full[@]}"} ) > "$log" 2>&1
  grep -E '\] (fold [0-9]|OOF|full-data)' "$log" || true
  echo "[05_cnn3d] $tag done in $((SECONDS - t0)) s"
}

for fam in $RUNS; do train "$fam"; done

# Member OOF (4 runs; the stacker's input). Skipped unless all four OOFs exist.
need=(mil48a mil48b m31000 m33000)
if all=1; for f in "${need[@]}"; do [ -f "$ROOT/outputs/oof_${P}${f}.csv" ] || all=0; done; [ "$all" = 1 ]; then
  ( cd "$ROOT/src" && "$PYTHON" avg_member.py "${P}mil48y" "${P}mil48a" "${P}mil48b" "${P}m31000" "${P}m33000" ) \
    2>&1 | tee "$ROOT/outputs/logs/${P}mil48y_avg.log"
else
  echo "[05_cnn3d] avg_member skipped: needs oof_${P}{mil48a,mil48b,m31000,m33000}.csv"
fi
