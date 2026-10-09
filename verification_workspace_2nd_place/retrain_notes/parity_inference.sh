#!/usr/bin/env bash
# Run the scored and the rebuilt 2nd-place packages on the first N private test scans and
# compare their predictions with each other, the platform CSV and the test labels.
# Usage: bash parity_inference.sh [N] [WORK_DIR]   (defaults: 64, a new temp dir)
set -euo pipefail
N="${1:-64}"
S="${2:-$(mktemp -d)}"
R=/Users/matt/dd/dat-parkinsons/references
W=/Users/matt/dd/dat-parkinsons/verification_workspace_2nd_place/retrain
PY="$W/.venv-infer/bin/python"

mkdir -p "$S/data/niftis" "$S/run_scored" "$S/run_rebuilt"
head -$((N + 1)) "$R/data/private/submission_format.csv" > "$S/data/submission_format.csv"
tail -n +2 "$S/data/submission_format.csv" | cut -d, -f1 | while read -r u; do
  ln -sf "$R/data/private/niftis/$u.nii.gz" "$S/data/niftis/"
done

for pkg in scored rebuilt; do
  case $pkg in scored) MAIN="$W/inference/main.py" ;; rebuilt) MAIN="$W/outputs/v17/submission/main.py" ;; esac
  ( cd "$S/run_$pkg" && time (DATA_DIR="$S/data" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 "$PY" "$MAIN" > log.txt 2>&1) )
done

"$PY" -I - "$S" "$R" <<'EOF'
import sys, pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
S, R = sys.argv[1], sys.argv[2]
a = pd.read_csv(S + "/run_scored/submission.csv").set_index("uid").is_pathologic
b = pd.read_csv(S + "/run_rebuilt/submission.csv").set_index("uid").is_pathologic.reindex(a.index)
p = pd.read_csv(R + "/submissions/comp_subs/2nd_place/2nd_place_submission.csv").set_index("uid").is_pathologic.reindex(a.index)
y = pd.read_csv(R + "/data/private/test_labels.csv").set_index("uid").is_pathologic.reindex(a.index)
for nm, (u, v) in {"scored-local vs platform CSV": (a, p), "rebuilt vs scored-local": (b, a), "rebuilt vs platform CSV": (b, p)}.items():
    d = (u - v).abs()
    print(f"{nm:30s} max {d.max():.2e} mean {d.mean():.2e} median {d.median():.2e} n>0.01 {(d > 0.01).sum()}")
for nm, v in [("platform", p), ("scored-local", a), ("rebuilt", b)]:
    print(f"{nm:13s} logloss {log_loss(y, v):.5f} auc {roc_auc_score(y, v):.5f}")
EOF
echo "work dir: $S"
