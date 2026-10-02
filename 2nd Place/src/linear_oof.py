"""
linear_oof.py -- cross-fitted out-of-fold predictions of the two L1-logistic
members, lin and lin4, which the stacker (fixstack.py --feat) reads as
outputs/oof_feat3_m2fnh.csv.

This file did not exist in the original project. It reproduces the ad-hoc
heredoc of 2026-09-03 04:27 UTC (recovered from the original development session logs) that wrote
outputs/oof_feat3_m2fnh.csv, keeping only its "no_head_ml" variant, which is the
one that was written to disk. The logic and the order of operations are unchanged:

  * lin  = train_final3.fit_lin on outputs/features_v13_m2f.csv, C = 0.2
  * lin4 = train_final3.fit_lin on outputs/features_v4_m2f.csv,  C = 1.0
  * table merged (inner, left order) with outputs/folds.csv on uid
  * columns = every column not in train_final3.DROP | train_final3.SCANNER | {'head_ml'}
    (235 for lin, 94 for lin4); float64 with non-finite values set to NaN
  * one fit per fold k in sorted order: train on fold != k, predict fold == k
  * written as uid, y, fold, lin, lin4, gbm, mlp. y is the float label, predictions
    are NOT clipped (train_final3.py clips to [1e-6, 1-1e-6], this did not), and
    gbm / mlp are NaN because v17 does not use them.

It reads the m2f tables (which still contain head_ml) and drops head_ml itself,
exactly as the original did. Since head_ml is excluded explicitly, the result
does not depend on DAT_ADMIT_HEAD_ML.

Prints only the per-member column count, OOF log loss and AUROC.

Usage: python src/linear_oof.py            (from any directory)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))
import train_final3 as T  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="lin/lin4 5-fold OOF for the stacker")
    ap.add_argument("--rich", default="features_v13_m2f.csv")
    ap.add_argument("--narrow", default="features_v4_m2f.csv")
    ap.add_argument("--C-rich", type=float, default=0.2)
    ap.add_argument("--C-narrow", type=float, default=1.0)
    ap.add_argument("--folds", default="folds.csv")
    ap.add_argument("--out", default="oof_feat3_m2fnh.csv")
    args = ap.parse_args()

    folds = pd.read_csv(OUT / args.folds)
    base = None
    for name, fn, C in [("lin", args.rich, args.C_rich), ("lin4", args.narrow, args.C_narrow)]:
        d = pd.read_csv(OUT / fn).merge(folds[["uid", "fold"]], on="uid")
        y = d.is_pathologic.to_numpy(float)
        extra = {"head_ml"}
        cols = [c for c in d.columns if c not in T.DROP | T.SCANNER | extra]
        X = d[cols].to_numpy(np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        p = np.zeros(len(d))
        for f in sorted(d.fold.unique()):
            tr, va = np.flatnonzero(d.fold != f), np.flatnonzero(d.fold == f)
            p[va], _ = T.fit_lin(X[tr], y[tr], X[va], C)
        print(f"{name:<5} ncols={len(cols)} OOF ll {log_loss(y, p):.4f} auc {roc_auc_score(y, p):.4f}",
              flush=True)
        if name == "lin":
            base = d[["uid", "fold"]].copy()
            base["y"] = y
        base[name] = p
    base["gbm"] = np.nan
    base["mlp"] = np.nan
    base[["uid", "y", "fold", "lin", "lin4", "gbm", "mlp"]].to_csv(OUT / args.out, index=False)
    print(f"wrote outputs/{args.out}")


if __name__ == "__main__":
    main()
