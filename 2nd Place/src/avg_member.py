"""avg_member.py — average several training runs of one architecture into a single
stacker member.

Separate members of the same architecture correlate ~0.99 and measurably hurt the
stack (RESEARCH_NOTES 5b); averaging them cuts seed variance without giving the
stacker another weight to fit. Probability-space average, matching what main.py's
`_cnn_prob` does across checkpoints.

    python avg_member.py mil48 mil48a mil48b
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

OUT = Path(__file__).resolve().parent.parent / "outputs"


def main(name, *srcs):
    base = None
    cols = []
    for s in srcs:
        d = pd.read_csv(OUT / f"oof_{s}.csv")
        cols.append(s)
        base = d if base is None else base.merge(d[["uid", s]], on="uid")
    p = base[cols].to_numpy(float).mean(1)
    for s in cols:
        v = base[s].to_numpy(float)
        print(f"  {s:<8} ll {log_loss(base.y, v):.4f} auc {roc_auc_score(base.y, v):.4f}")
    print(f"  {name:<8} ll {log_loss(base.y, p):.4f} auc {roc_auc_score(base.y, p):.4f}"
          f"   <- mean of {len(cols)} runs")
    out = base[["uid", "y", "fold"]].copy()
    out[name] = np.clip(p, 1e-6, 1 - 1e-6)
    out.to_csv(OUT / f"oof_{name}.csv", index=False)
    print(f"[avg] wrote oof_{name}.csv")


if __name__ == "__main__":
    main(*sys.argv[1:])
