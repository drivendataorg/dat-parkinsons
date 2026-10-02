"""avg_2d_seeds.py -- two-seed average of the 2D three-MIP reader (stacker input).

Reproduces the inline pandas step of the original tools/f2seed2_chain.sh (run
after the second seed e2dmipf22 finished, 2026-09-08), which wrote
outputs/oof_e2dmipf2x.csv:

    a = oof_e2dmipf2.csv; b = oof_e2dmipf22.csv
    d = a.merge(b[['uid','e2dmipf22']], on='uid')
    d['e2dmipf2x'] = 0.5 * (d.e2dmipf2 + d.e2dmipf22)
    d[['uid','y','fold','e2dmipf2x']].to_csv('outputs/oof_e2dmipf2x.csv', index=False)

The package member called 'e2dmipf2' averages the checkpoints of BOTH runs
(e2dmipf2 + e2dmipf22), and its stacker weight was fitted on this two-seed OOF,
not on oof_e2dmipf2.csv. Probability-space mean with NO clipping (unlike
avg_member.py, which clips to [1e-6, 1-1e-6]): the original step did not clip,
and byte-identical output needs the same arithmetic.

Usage (from anywhere; paths resolve against the repo root):
    python src/avg_2d_seeds.py                                  # e2dmipf2x = mean(e2dmipf2, e2dmipf22)
    python src/avg_2d_seeds.py smoke_e2dmipf2x smoke_e2dmipf2 smoke_e2dmipf22
    python src/avg_2d_seeds.py --in-dir <dir> --out-dir <dir>   # read/write elsewhere (verification)

Prints only aggregate log loss / AUROC.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="*", default=["e2dmipf2x", "e2dmipf2", "e2dmipf22"],
                    help="OUT_NAME RUN_A RUN_B (default: e2dmipf2x e2dmipf2 e2dmipf22)")
    ap.add_argument("--in-dir", default=str(OUT), help="directory holding oof_<run>.csv")
    ap.add_argument("--out-dir", default=str(OUT), help="directory for oof_<out>.csv")
    args = ap.parse_args()
    if len(args.names) != 3:
        raise SystemExit("expected exactly three names: OUT_NAME RUN_A RUN_B")
    name, ta, tb = args.names
    src, dst = Path(args.in_dir), Path(args.out_dir)

    a = pd.read_csv(src / f"oof_{ta}.csv")
    b = pd.read_csv(src / f"oof_{tb}.csv")
    d = a.merge(b[["uid", tb]], on="uid")
    if len(d) != len(a) or len(d) != len(b):
        raise SystemExit(f"row mismatch after merge: {len(a)} / {len(b)} -> {len(d)}")
    d[name] = 0.5 * (d[ta] + d[tb])
    d[["uid", "y", "fold", name]].to_csv(dst / f"oof_{name}.csv", index=False)

    for c in (ta, tb, name):
        v = d[c].to_numpy(float)
        print(f"  {c:<12} ll {log_loss(d.y, np.clip(v, 1e-6, 1 - 1e-6)):.4f} "
              f"auc {roc_auc_score(d.y, v):.4f}")
    print(f"[avg_2d_seeds] wrote oof_{name}.csv ({len(d)} rows)")


if __name__ == "__main__":
    main()
