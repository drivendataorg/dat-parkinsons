#!/usr/bin/env bash
# 08_stack.sh -- fixed-ratio stacker, 1-D calibration and image-ensemble package
# spec (the "v14c" image + linear half of the v17 submission).
#
# Reproduces the stacker and build steps of the original tools/build_f2two.sh,
# exactly as tools/v14c_chain.sh called it on 2026-09-08:
#     bash tools/build_f2two.sh v14c e2dmipf2x e2dmipf2 e2dmipf2+e2dmipf22
# Same variables, same argument lists, same inline member-rename snippet.
#
#   1. src/fixstack.py   fixed logit ratio (lin, lin4, mil48y, e2dmipf2, e2db3f2)
#                        = (1, 0.4, 0.7, 0.3, 0.3), one LogisticRegression(C=1e6)
#                        on the blended logit -> scale 0.53121, intercept 0.18431.
#                        T=1.4 and clip=0.005 are fixstack's argparse defaults.
#   2. inline rename     members -> ['lin','lin4','mil48y','e2dmipf2','e2db3f2']
#                        (fixstack names the 2D column after the OOF file, e2dmipf2x).
#   3. src/build3.py     writes submission_src/assets/inference_config.json, copies
#                        the 34 3D + 18 TorchScript 2D checkpoints and the template
#                        into submission_src/assets/ and packs candidate_v14c.zip
#                        (at the repo root, as in the original).
#
# NOT run here (upstream stages): `cd src && python train_final3.py ... --export`,
# which writes submission_src/assets/linear_lin{,4}.json and
# outputs/feature_{order,union}.json; the OOF tables; the CNN checkpoints.
# The original's data-verify gates 4/5/7 need a held-out 150-scan folder and are
# not part of this script; gate 6 (static logging grep) is.
#
# RATIO_B0 / RATIO_B3 (default 0.3 each, the v17 values) override the stack ratios of
# the two 2D members (e2dmipf2, e2db3f2). Leave both unset to reproduce v17.
#
# Usage (from anywhere):  bash pipeline/08_stack.sh        [PYTHON=/path/to/python]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
# a relative PYTHON (e.g. .venv-train/bin/python) is made absolute: the steps below cd first
case "$PYTHON" in */*) [ ! -d "$(dirname "$PYTHON")" ] || PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")";; esac
cd "$ROOT"
export KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=3 HF_HUB_OFFLINE=1
mkdir -p outputs/logs
exec > >(tee -a outputs/logs/08_stack.log) 2>&1

# Arguments of the original call: build_f2two.sh v14c e2dmipf2x e2dmipf2 e2dmipf2+e2dmipf22
VER=v14c; TAG2D=e2dmipf2x; MEMBER2D=e2dmipf2; SRC2D=e2dmipf2+e2dmipf22; ZIP="candidate_${VER}.zip"
MEMBERS="['lin','lin4','mil48y','${MEMBER2D}','e2db3f2']"
TAGS="mil48y=mil48a+mil48b+m31000+m33000+m35000+m37000+mf21000+mf23000+mf25000+mf27000 ${MEMBER2D}=${SRC2D} e2db3f2=e2db3f2"
T0=$(date +%s); echo "########## 08_stack start $(date) ##########"

echo "########## preconditions (outputs of stages 01-07) ##########"
missing=0
need() { for f in "$@"; do [ -f "$f" ] || { echo "MISSING $f"; missing=1; }; done; }
need outputs/oof_feat3_m2fnh.csv outputs/oof_mil48y.csv outputs/oof_${TAG2D}.csv outputs/oof_e2db3f2.csv \
     outputs/feature_order.json outputs/feature_union.json outputs/template_v6.npy \
     submission_src/assets/linear_lin.json submission_src/assets/linear_lin4.json
# build3.py silently skips checkpoints that do not exist, so list every one it must ship.
for s in mil48a mil48b m31000 m33000 m35000 m37000; do for k in 0 1 2 3 4; do need outputs/${s}_fold${k}.pt; done; done
for s in mf21000 mf23000 mf25000 mf27000; do need outputs/${s}_full0.pt; done
# 2D members: build3 reads the state-dict .pt for the architecture metadata, then ships the .ts.pt.
for s in e2dmipf2 e2dmipf22 e2db3f2; do for k in fold0 fold1 fold2 fold3 fold4 full0; do
  need outputs/${s}_${k}.pt outputs/${s}_${k}.ts.pt; done; done
# With --use-full build3 would also ship any <src>_full0.pt / <src>_fold*.pt it finds;
# the v17 member set has none of these, so refuse to build if they appear.
for s in mil48a mil48b m31000 m33000 m35000 m37000; do [ ! -f outputs/${s}_full0.pt ] || { echo "UNEXPECTED outputs/${s}_full0.pt"; missing=1; }; done
for s in mf21000 mf23000 mf25000 mf27000; do for k in 0 1 2 3 4; do [ ! -f outputs/${s}_fold${k}.pt ] || { echo "UNEXPECTED outputs/${s}_fold${k}.pt"; missing=1; }; done; done
[ "$missing" = 0 ] || { echo "08_stack: inputs missing or unexpected (see above); run stages 01-07 first"; exit 1; }
echo "all inputs present"

echo "########## $VER: members (must be 235/94 columns, no head_ml) ##########"
"$PYTHON" - <<'PY'
import json
for f in ['submission_src/assets/linear_lin.json','submission_src/assets/linear_lin4.json']:
    j = json.load(open(f)); assert 'head_ml' not in j['features'] and 'hole_ml' not in j['features'], f; print(f, len(j['features']), 'features, no head_ml/hole_ml')
PY

echo "########## stacker ##########"
[ -z "${RATIO_B0:-}${RATIO_B3:-}" ] || echo "WARNING: RATIO_B0/RATIO_B3 are set (B0 ${RATIO_B0:-0.3}, B3 ${RATIO_B3:-0.3}). v17 used 0.3 for both."
"$PYTHON" src/fixstack.py --feat oof_feat3_m2fnh.csv --cnn oof_mil48y.csv --extra oof_${TAG2D}.csv ${TAG2D} --extra oof_e2db3f2.csv e2db3f2 --ratio 1.0 0.4 0.7 ${RATIO_B0:-0.3} ${RATIO_B3:-0.3} --out ensemble_${VER}.json
"$PYTHON" - "$VER" "$MEMBERS" <<'PY'
import json, sys, ast
e = json.load(open(f'outputs/ensemble_{sys.argv[1]}.json')); e['members'] = ast.literal_eval(sys.argv[2]); json.dump(e, open(f'outputs/ensemble_{sys.argv[1]}.json', 'w'), indent=2); print('members', e['members'], 'coef', [round(c, 4) for c in e['params']['coef']])
PY

echo "########## build ##########"
( cd src && "$PYTHON" -u build3.py --ensemble ensemble_${VER}.json --cnn-tags $TAGS --use-full --zip "$ZIP" --tta 0 --template template_v6.npy --key2d ${MEMBER2D}=aligned_f2 e2db3f2=aligned_f2 --crops2d ${MEMBER2D}=crops_v3_f2.npz e2db3f2=crops_v3_f2.npz )

echo "########## GATE 6: logging (must print nothing) ##########"
grep -nE "logger\.[a-z]+\(f" submission_src/main.py && { echo FAIL; exit 1; } || echo OK

"$PYTHON" - "$ZIP" <<'PY'
import hashlib, json, sys, zipfile
cfg = json.load(open('submission_src/assets/inference_config.json'))
print('scale', json.load(open('outputs/ensemble_v14c.json'))['params']['scale'], 'intercept', cfg['stacker']['intercept'])
print('members', cfg['members'], 'cnn sources', [len(c['sources']) for c in cfg['cnns']], 'feature_order', len(cfg['feature_order']))
for f in ['outputs/ensemble_v14c.json', 'submission_src/assets/inference_config.json', sys.argv[1]]:
    print('sha256', hashlib.sha256(open(f, 'rb').read()).hexdigest(), f)
print('zip entries', len(zipfile.ZipFile(sys.argv[1]).namelist()))
PY
echo "BUILD_${VER}_DONE  $(( $(date +%s) - T0 )) s  $(date)"
