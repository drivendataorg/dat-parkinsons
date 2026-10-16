"""
fixstack.py — v11/v12 stacker: FIXED logit ratio (lin, lin4, mil48y) = (1.0, 0.4,
0.9) followed by a 1-D logistic calibration (RESEARCH_NOTES §68: beats the
fitted nnlogreg on 6/6 re-splits), temperature T and clip from the board
T-curve (§67/§72), governor flag (§69a). Emits an ensemble json in build3's
schema: coef = scale * ratio, intercept = b0.

    python fixstack.py --feat oof_feat3_m2f.csv --cnn oof_mil48y.csv --out ensemble_v12a.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

OUT = Path(__file__).resolve().parent.parent / "outputs"
EPS = 1e-6
lg = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="oof_feat3.csv")
    ap.add_argument("--cnn", default="oof_mil48y.csv")
    ap.add_argument("--ratio", type=float, nargs="+", default=[1.0, 0.4, 0.9])
    ap.add_argument("--extra", nargs=2, action="append", default=[],
                    help="extra member: OOF csv and column (v13a: oof_e2dmip.csv e2dmip)")
    ap.add_argument("--T", type=float, default=1.4)
    ap.add_argument("--clip", type=float, default=0.005)
    ap.add_argument("--out", default="ensemble_v12a.json")
    args = ap.parse_args()
    f = pd.read_csv(OUT / args.feat)[["uid", "y", "fold", "lin", "lin4"]]
    m = pd.read_csv(OUT / args.cnn)[["uid", "mil48y"]]
    d = f.merge(m, on="uid")
    members = ["lin", "lin4", "mil48y"]
    for fn, col in args.extra:
        d = d.merge(pd.read_csv(OUT / fn)[["uid", col]], on="uid"); members.append(col)
    assert len(args.ratio) == len(members), (args.ratio, members)
    y = d["y"].to_numpy(float)
    Z = np.column_stack([lg(d[c]) for c in members]) @ np.asarray(args.ratio)
    # cross-fitted CV of the 1-D calibration (honest number for the record)
    p = np.zeros(len(y))
    for k in range(5):
        tr, va = d.fold != k, d.fold == k
        lr = LogisticRegression(C=1e6).fit(Z[tr].reshape(-1, 1), y[tr])
        p[va] = lr.predict_proba(Z[va].reshape(-1, 1))[:, 1]
    pc = np.clip(p, args.clip, 1 - args.clip)
    pT = 1 / (1 + np.exp(-lg(p) / args.T)); pT = np.clip(pT, args.clip, 1 - args.clip)
    print(f"[fixstack] cross-fit T=1: ll {log_loss(y, pc):.4f} auc {roc_auc_score(y, pc):.5f} | "
          f"T={args.T}: ll {log_loss(y, pT):.4f}")
    lr = LogisticRegression(C=1e6).fit(Z.reshape(-1, 1), y)
    scale, b0 = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    coef = (scale * np.asarray(args.ratio)).tolist()
    ens = dict(members=members, best="fixed_ratio",
               params=dict(kind="fixed", coef=coef, intercept=b0, ratio=args.ratio, scale=scale),
               calibration=dict(T=args.T, clip=args.clip),
               governor=dict(reject_to_prior=True),
               cv=dict(fixed_T1=dict(logloss=float(log_loss(y, pc)), auroc=float(roc_auc_score(y, pc))),
                       fixed_T=dict(T=args.T, logloss=float(log_loss(y, pT)))))
    (OUT / args.out).write_text(json.dumps(ens, indent=2))
    print(f"[fixstack] wrote {args.out}: coef {np.round(coef, 4).tolist()} intercept {b0:.4f}")


if __name__ == "__main__":
    main()
