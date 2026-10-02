#!/usr/bin/env bash
# smoke_cnn.sh -- fast end-to-end exercise of the CNN training commands.
#
# NOT a reproduction: every run is trained for ONE epoch. It proves that the
# real pipeline scripts (05_cnn3d.sh, 06_cnn2d.sh) and every step behind them
# (cnn3.py 5-fold and --full, avg_member.py, cnn2d.py B0 and B3 5-fold / --full,
# avg_2d_seeds.py, tools/ts_export.py) execute from this repository
# on the real caches. All tags get the prefix 'smoke_' (e.g.
# outputs/smoke_mil48a_fold0.pt), so shipped-tag files are never touched;
# FORCE=1 is set so a smoke rerun may overwrite earlier smoke_* files only.
#
# Default subset (override with SMOKE_3D / SMOKE_2D, same syntax as RUNS):
#   3D: mil48a (5-fold) + mf21000 (--full)
#   2D: e2dmipf2 + e2dmipf22 (5-fold), e2dmipf2:full, e2db3f2 (5-fold), e2db3f2:full
#   then avg_member.py smoke_mil48y smoke_mil48a, avg_2d_seeds.py on the two B0
#   seeds, ts_export.py for smoke_e2dmipf2 and smoke_e2db3f2 (the tags with all
#   six checkpoints). An empty SMOKE_3D / SMOKE_2D skips that half.
# Prerequisites: outputs/folds.csv, outputs/crops_v3_b.npz, outputs/crops_v3_f2.npz,
# network (or a populated HF cache) for the ImageNet backbones.
# Runtime: ~10-15 min on an Apple M3 Pro (MPS) shared with other jobs.
#
# Usage: bash pipeline/smoke_cnn.sh          (PYTHON=/path/to/python to override)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
# a relative PYTHON (e.g. .venv-train/bin/python) is made absolute: the steps below cd first
case "$PYTHON" in */*) [ ! -d "$(dirname "$PYTHON")" ] || PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")";; esac
export PYTHON EPOCHS=1 TAG_PREFIX=smoke_ FORCE=1
SMOKE_3D="${SMOKE_3D-mil48a mf21000}"
SMOKE_2D="${SMOKE_2D-e2dmipf2 e2dmipf22 e2dmipf2:full e2db3f2 e2db3f2:full}"
T0=$SECONDS

if [ -n "$SMOKE_3D" ]; then
  echo "=== [smoke] 3D: $SMOKE_3D ($(date))"
  t=$SECONDS
  RUNS="$SMOKE_3D" bash "$ROOT/pipeline/05_cnn3d.sh"
  if [ -f "$ROOT/outputs/oof_smoke_mil48a.csv" ]; then
    ( cd "$ROOT/src" && "$PYTHON" avg_member.py smoke_mil48y smoke_mil48a )
  fi
  echo "=== [smoke] 3D stage: $((SECONDS - t)) s"
fi

if [ -n "$SMOKE_2D" ]; then
  echo "=== [smoke] 2D: $SMOKE_2D ($(date))"
  t=$SECONDS
  RUNS="$SMOKE_2D" bash "$ROOT/pipeline/06_cnn2d.sh"
  echo "=== [smoke] 2D stage: $((SECONDS - t)) s"
fi

# expected artefacts of the default subset
missing=0
expect () { for f in "$@"; do [ -f "$ROOT/outputs/$f" ] || { echo "[smoke] MISSING outputs/$f"; missing=$((missing + 1)); }; done; }
case " $SMOKE_3D " in *" mil48a "*)
  expect smoke_mil48a_fold{0,1,2,3,4}.pt oof_smoke_mil48a.csv cv_smoke_mil48a.json oof_smoke_mil48y.csv ;; esac
case " $SMOKE_3D " in *" mf21000 "*) expect smoke_mf21000_full0.pt ;; esac
case " $SMOKE_2D " in *" e2dmipf2 "*)
  expect smoke_e2dmipf2_fold{0,1,2,3,4}.pt oof_smoke_e2dmipf2.csv cv_smoke_e2dmipf2.json ;; esac
case " $SMOKE_2D " in *" e2dmipf22 "*)
  expect smoke_e2dmipf22_fold{0,1,2,3,4}.pt oof_smoke_e2dmipf2x.csv ;; esac
case " $SMOKE_2D " in *" e2dmipf2:full "*)
  expect smoke_e2dmipf2_full0.pt smoke_e2dmipf2_{fold0,fold1,fold2,fold3,fold4,full0}.ts.pt smoke_e2dmipf2_meta.json ;; esac
case " $SMOKE_2D " in *" e2db3f2:full "*) expect smoke_e2db3f2_full0.pt ;; esac
case " $SMOKE_2D " in *" e2db3f2 "*)
  expect smoke_e2db3f2_fold{0,1,2,3,4}.pt oof_smoke_e2db3f2.csv cv_smoke_e2db3f2.json ;; esac
case " $SMOKE_2D " in *" e2db3f2 "*" e2db3f2:full "*|*" e2db3f2:full "*" e2db3f2 "*)
  expect smoke_e2db3f2_{fold0,fold1,fold2,fold3,fold4,full0}.ts.pt smoke_e2db3f2_meta.json ;; esac
echo "=== [smoke] total $((SECONDS - T0)) s; missing artefacts: $missing"
[ "$missing" = 0 ]
