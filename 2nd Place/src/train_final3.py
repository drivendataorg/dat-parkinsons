"""
train_final3.py — fit the feature models, emit honest OOF, export portable assets.

Feature tables (nested, all produced by one cache3.py pass):
  v4 = clinical ROI measurements
  v5 = v4 + the striatal grid profile
  v6 = v5 + per-hemisphere shape descriptors, ordered worse-side-first

Four members with genuinely different inductive biases:

  * `lin`  — L1 logistic regression on **v6** (clinical ROIs + striatal grid
             profile + per-hemisphere shape, ordered worse-side-first). The
             hemisphere block is the feature-space analogue of what makes the MIL
             CNN work, and it is worth ~0.011 on its own (0.2614 -> 0.2500,
             repeated CV).
  * `lin4` — the same model on the pure clinical table v4. Deliberately a
             narrower view, so its mistakes differ from `lin`'s.
  * `gbm`  — LightGBM on v6, which is where the tree model finds its interactions
             (it is clearly worse on v4 alone).
  * `mlp`  — a small ReLU network on v6, bagged over seeds; individually the
             weakest, but its errors decorrelate from the linear members.

Nothing is early-stopped or selected against the fold it scores, so `oof_*.csv`
is an unbiased input to the stacker. Every model exports to plain numpy / native
LightGBM text — no sklearn object crosses into the runtime container.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

import portable3 as P

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
ASSETS = ROOT / "submission_src" / "assets"
DROP = {"uid", "is_pathologic", "y", "fold"}
# Columns withheld from the feature matrix because they describe the ACQUISITION
# rather than the patient. `native_spacing` is the pseudo-site label itself and
# keying on it killed the PSF lever (§15); `ref` is the reference level, whose
# site eta^2 is 0.513 among normals and which measurably HURTS leave-site-out when
# exposed (0.2590 -> 0.2633); `inner_ml` helps but is dominated by `head_ml`.
#
# v8: `head_ml` is NO LONGER withheld. Measured 2026-07-26 (§50) — exposing it is
# worth -0.0038 CV and **-0.0107 leave-site-out** on `lin` (-0.0025/-0.0100 on
# `lin4`), -0.0034 at the stack, favourable on 5 of 5 independent re-splits, and it
# survives the adversarial test that matters: holding out ENTIRE field-of-view
# strata, so the model must predict scans whose head size range it has never seen,
# it still gains -0.0058 to -0.0063 across 4/5/6 strata. A site lookup would be
# useless there; a monotone physical effect extrapolates, and this one does.
# 2026-09-03 (RESEARCH_NOTES §74c): `head_ml` is withheld AGAIN. v8 exposed it, the
# board refuted it in v9 (prevalence-tracking, §60b), and because this set no longer
# listed it the v12a/v13a "restricted to v11's columns" refits silently re-admitted
# it. Re-admit only by explicit opt-in: DAT_ADMIT_HEAD_ML=1.
SCANNER = {"native_spacing", "inner_ml", "ref"} | (set() if __import__("os").environ.get("DAT_ADMIT_HEAD_ML") == "1" else {"head_ml"})

LGB_PARAMS = dict(objective="binary", learning_rate=0.02, num_leaves=8,
                  min_data_in_leaf=25, feature_fraction=0.4, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l1=0.5, lambda_l2=5.0, max_depth=4,
                  verbose=-1, num_threads=4, deterministic=True)
LGB_ROUNDS = 700
LGB_SEEDS = (0, 1, 2, 3, 4)
MLP_SEEDS = (0, 1, 2)
MLP_KW = dict(hidden_layer_sizes=(64,), alpha=3.0, max_iter=1200,
              learning_rate_init=3e-3, early_stopping=False)


def load(fn, use_scanner=False):
    d = pd.read_csv(OUT / fn)
    cols = [c for c in d.columns if c not in DROP and (use_scanner or c not in SCANNER)]
    X = d[cols].to_numpy(np.float64)
    X = np.where(np.isfinite(X), X, np.nan)
    return d["uid"].to_numpy(), X, d["is_pathologic"].to_numpy(np.float64), cols


def _impute(A, B):
    med = np.nanmedian(A, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return (np.where(np.isfinite(A), A, med),
            np.where(np.isfinite(B), B, med), med)


def fit_lin(Xtr, ytr, Xva, C, seed=0):
    # random_state is pinned: liblinear's coordinate descent shuffles features and
    # defaults to an unseeded RNG, which made the L1 support (and the reported CV)
    # jitter by ~0.0003 between otherwise identical runs.
    prep = P.FeatPrep().fit(Xtr)
    m = LogisticRegression(C=C, penalty="l1", solver="liblinear", max_iter=8000,
                           random_state=seed)
    m.fit(prep.transform(Xtr), ytr)
    return m.predict_proba(prep.transform(Xva))[:, 1], (prep, m)


def fit_gbm(Xtr, ytr, Xva):
    A, B, med = _impute(Xtr, Xva)
    ps = np.zeros(len(B))
    boosters = []
    for s in LGB_SEEDS:
        bst = lgb.train(dict(LGB_PARAMS, seed=s), lgb.Dataset(A, label=ytr),
                        num_boost_round=LGB_ROUNDS)
        ps += bst.predict(B)
        boosters.append(bst)
    return ps / len(LGB_SEEDS), (boosters, med)


def fit_mlp(Xtr, ytr, Xva):
    prep = P.FeatPrep().fit(Xtr)
    A, B = prep.transform(Xtr), prep.transform(Xva)
    ps = np.zeros(len(B))
    nets = []
    for s in MLP_SEEDS:
        m = MLPClassifier(random_state=s, **MLP_KW).fit(A, ytr)
        ps += m.predict_proba(B)[:, 1]
        nets.append(m)
    return ps / len(MLP_SEEDS), (prep, nets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rich-features", default="features_v6_c.csv")
    ap.add_argument("--narrow-features", default="features_v4_c.csv")
    ap.add_argument("--C-rich", type=float, default=0.2)
    ap.add_argument("--C-narrow", type=float, default=1.0)
    ap.add_argument("--folds", default="folds.csv")
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()

    fo = pd.read_csv(OUT / args.folds)
    uids, Xr, y, cols_r = load(args.rich_features)      # v6
    uid_n, Xn, y2, cols_n = load(args.narrow_features)  # v4
    assert (uids == uid_n).all() and (y == y2).all()
    fold = fo.set_index("uid").loc[uids, "fold"].to_numpy()

    oof = {k: np.zeros(len(y)) for k in ("lin", "lin4", "gbm", "mlp")}
    for k in sorted(np.unique(fold)):
        tr, va = fold != k, fold == k
        oof["lin"][va] = fit_lin(Xr[tr], y[tr], Xr[va], args.C_rich)[0]
        oof["lin4"][va] = fit_lin(Xn[tr], y[tr], Xn[va], args.C_narrow)[0]
        oof["gbm"][va] = fit_gbm(Xr[tr], y[tr], Xr[va])[0]
        oof["mlp"][va] = fit_mlp(Xr[tr], y[tr], Xr[va])[0]

    df = pd.DataFrame({"uid": uids, "y": y.astype(int), "fold": fold})
    for k, v in oof.items():
        v = np.clip(v, 1e-6, 1 - 1e-6)
        df[k] = v
        print(f"  {k:5s} ll {log_loss(y, v):.4f}  auc {roc_auc_score(y, v):.4f}")
    df.to_csv(OUT / "oof_feat3.csv", index=False)
    print("[final3] wrote oof_feat3.csv")

    if not args.export:
        return
    ASSETS.mkdir(parents=True, exist_ok=True)
    _, (prep_r, m_r) = fit_lin(Xr, y, Xr[:1], args.C_rich)
    P.dump_linear(ASSETS / "linear_lin.json", prep_r, m_r.coef_[0], m_r.intercept_[0], cols_r)
    _, (prep_n, m_n) = fit_lin(Xn, y, Xn[:1], args.C_narrow)
    P.dump_linear(ASSETS / "linear_lin4.json", prep_n, m_n.coef_[0], m_n.intercept_[0], cols_n)

    _, (boosters, med) = fit_gbm(Xr, y, Xr[:1])
    for s, bst in zip(LGB_SEEDS, boosters):
        P.dump_lgbm(ASSETS / f"lgbm_gbm_s{s}.txt", bst, med, cols_r)

    _, (prep_m, nets) = fit_mlp(Xr, y, Xr[:1])
    for s, net in zip(MLP_SEEDS, nets):
        P.dump_mlp(ASSETS / f"mlp_mlp_s{s}.json", prep_m, net, cols_r)

    (OUT / "feature_order.json").write_text(json.dumps(
        {"lin": cols_r, "lin4": cols_n, "gbm": cols_r, "mlp": cols_r}, indent=2))
    # main.py materialises the union once per scan; each model selects by name
    union = list(cols_r) + [c for c in cols_n if c not in set(cols_r)]
    (OUT / "feature_union.json").write_text(json.dumps(union, indent=2))
    print(f"[final3] exported: L1 kept {int((m_r.coef_[0] != 0).sum())}/{len(cols_r)} "
          f"rich features, {int((m_n.coef_[0] != 0).sum())}/{len(cols_n)} narrow")


if __name__ == "__main__":
    main()
