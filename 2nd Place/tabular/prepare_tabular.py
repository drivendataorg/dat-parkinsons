"""Prepare the v17 TabICL recipe, context matrix and 479-name feature order.

Faithful port of experiments/v17/prepare_candidate.py (the script that built
v17's assets/tabular_recipe.json and experiments/v17/rebuild/context.npz),
with its research-path lookups (experiments/v16/bench.py:inputs/weights and
experiments/20260914/common.py:load_data) replaced by explicit arguments.
The deployment-stage copy and the source patches of inference_main.py and
tabular_adapter.py are omitted: inference/ already holds the shipped code.

Numerical logic is unchanged:
  * candidates = 235 rich clinical columns (linear_lin.json order) + 180 scale-space
    shape columns ('shape__') + 4 acquisition descriptors (head_ml, inner_ml, ref,
    native_spacing); rows in features_v13_m2fnh.csv order
  * sample weights: per fam2/scale group, prevalence shrunk toward the global prior
    with smoothing 30 (bench.weights)
  * four L1 priors (liblinear, random_state 0, max_iter 5000) on FeatPrep(log) inputs:
    rich 235 C=.2, narrow 94 C=1, rich+shape 415 C=.2, rich+mixture 295 C=.2;
    context anchors cross-fitted leave-one-fam2-out, query priors refitted on all rows
  * selection: weighted L1 C=.2 on the 415 candidates, top 64 by |coef| (np.argsort)
  * context = 64 selected + 4 descriptors + 4 anchors, median-imputed
  * log subset: default_rng(16000), class-proportional draw per sorted group
  * log_prep = FeatPrep().fit on the 64 selected columns of all rows

The feature order reproduces the three appends that led to v17's 479 names:
v14c's 235 linear names -> build_stage.py (v15a) unweighted top-64 + descriptors
-> experiments/v16/prepare.py weighted selection -> prepare_candidate.py all
candidate + mixture columns.

Outputs (default outputs/tabular/): tabular_recipe.json, context.npz,
feature_order_v17.json, prepare_info.json (aggregates only).
"""
import os
os.environ['OMP_NUM_THREADS'] = '3'; os.environ['OPENBLAS_NUM_THREADS'] = '3'
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from portable3 import FeatPrep  # noqa: E402  (src/portable3.py == shipped inference/portable3.py)

META = ['head_ml', 'inner_ml', 'ref', 'native_spacing']


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return pd.read_csv(path, dtype={'uid': str})


def load_table(path, cols, folds, groups):
    """experiments/20260914/common.py:load_data with explicit paths."""
    d = read(path).set_index('uid')
    f = folds.loc[d.index]
    g = groups.loc[d.index]
    grp = (g.fam2 + '/' + g.scale).to_numpy()
    X = d[cols].to_numpy(float)
    X[~np.isfinite(X)] = np.nan
    y = d.is_pathologic.to_numpy(int)
    assert np.array_equal(y, f.y.to_numpy(int))
    assert len(d.index.unique()) == len(d)
    return X, y, f.fold.to_numpy(int), grp, d.index.to_numpy(), cols


def weights(y, g):
    """experiments/v16/bench.py:weights (verbatim logic)."""
    # Shrink each acquisition group's prevalence toward the training prevalence.
    prior = y.mean(); w = np.ones(len(y))
    for s in np.unique(g):
        sel = g == s; p = (y[sel].sum() + 30 * prior) / (sel.sum() + 30)
        w[sel] = np.where(y[sel], prior / p, (1 - prior) / (1 - p))
    return w / w.mean()


def main():
    ap = argparse.ArgumentParser(description='v17 tabular recipe/context/feature order')
    ap.add_argument('--inputs-dir', default=str(ROOT / 'outputs'),
                    help='dir with features_v13_m2fnh.csv, features_v4_m2fnh.csv, features_v13_m2f.csv, folds.csv')
    ap.add_argument('--groups-csv', default=str(ROOT / 'outputs/tabular/acquisition_groups.csv'))
    ap.add_argument('--shape-csv', default=str(ROOT / 'outputs/tabular/scale_f2.csv'))
    ap.add_argument('--mixture-csv', default=str(ROOT / 'outputs/tabular/mixture_shape.csv'))
    ap.add_argument('--linear-assets', default=str(ROOT / 'submission_src/assets'),
                    help='dir with linear_lin.json and linear_lin4.json (feature lists of the linear members)')
    ap.add_argument('--base-feature-order', default=None,
                    help='optional JSON list: the pre-TabICL inference feature_order (default: linear_lin.json features,'
                         ' which equals the v14c archive feature_order)')
    ap.add_argument('--out-dir', default=str(ROOT / 'outputs/tabular'))
    args = ap.parse_args()

    inp = Path(args.inputs_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    lin_assets = Path(args.linear_assets)
    cols = json.loads((lin_assets / 'linear_lin.json').read_text())['features']
    cols4 = json.loads((lin_assets / 'linear_lin4.json').read_text())['features']
    folds = read(inp / 'folds.csv').set_index('uid')
    groups = read(args.groups_csv).set_index('uid')

    # --- bench.inputs(): rich 235 + shape 180 + 4 descriptors ---------------------------
    X, y, fold, g, u, cols = load_table(inp / 'features_v13_m2fnh.csv', cols, folds, groups)
    shape = read(args.shape_csv).set_index('uid').loc[u]
    meta = read(inp / 'features_v13_m2f.csv').set_index('uid').loc[u, META]
    X = np.column_stack([X, shape, meta])
    cols = cols + ['shape__' + c for c in shape.columns] + list(meta.columns)
    # --- load_data('narrow'): the original used the v4 table positionally; align on uid ----
    narrow, _, _, _, u4, narrow_cols = load_table(inp / 'features_v4_m2fnh.csv', cols4, folds, groups)
    narrow_same_order = bool(np.array_equal(u4, u))
    if not narrow_same_order:
        narrow = pd.DataFrame(narrow, index=u4).loc[u].to_numpy()
    mixframe = read(args.mixture_csv).set_index('uid').loc[u]
    mix = mixframe.to_numpy(); mixcols = ['mixture__' + c for c in mixframe.columns]

    # --- prepare_candidate.py:19-36 --------------------------------------------------------
    family = np.array([s.split('/')[0] for s in g]); anchors = np.zeros((len(y), 4)); priors = []
    specs = [(X[:, :235], cols[:235], .2), (narrow, narrow_cols, 1.), (X[:, :-4], cols[:-4], .2),
             (np.column_stack([X[:, :235], mix]), cols[:235] + mixcols, .2)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')

        def fit(F, tr, C, sample_weight=True):
            prep = FeatPrep().fit(F[tr])
            model = LogisticRegression(C=C, penalty='l1', solver='liblinear', random_state=0, max_iter=5000).fit(
                prep.transform(F[tr]), y[tr], sample_weight=weights(y[tr], g[tr]) if sample_weight else None)
            return prep, model

        for j, (F, names, C) in enumerate(specs):
            for group in np.unique(family):
                tr = np.where(family != group)[0]; va = np.where(family == group)[0]; prep, model = fit(F, tr, C)
                anchors[va, j] = model.decision_function(prep.transform(F[va]))
            prep, model = fit(F, np.arange(len(y)), C)
            priors.append(dict(features=names, prep=prep.to_dict(), coef=model.coef_[0].tolist(),
                               intercept=float(model.intercept_[0])))
        prep, selector = fit(X[:, :-4], np.arange(len(y)), .2); keep = np.argsort(-abs(selector.coef_[0]))[:64]
        # build_stage.py (v15a): unweighted selection, used only for the feature_order lineage
        v15_prep = FeatPrep().fit(X[:, :-4])
        v15_sel = LogisticRegression(C=.2, penalty='l1', solver='liblinear', random_state=0, max_iter=5000).fit(
            v15_prep.transform(X[:, :-4]), y)
        v15_keep = np.argsort(-abs(v15_sel.coef_[0]))[:64]
    warning_kinds = sorted({f'{w.category.__name__}: {str(w.message)[:90]}' for w in caught})

    F = np.column_stack([X[:, keep], X[:, -4:], anchors]); median = np.nan_to_num(np.nanmedian(F, axis=0))
    F = np.where(np.isfinite(F), F, median)
    chosen = []; rng = np.random.default_rng(16000); prior = y.mean()
    for group in np.unique(g):
        ix = np.where(g == group)[0]; pos = ix[y[ix] == 1]; neg = ix[y[ix] == 0]
        n = min(len(pos) / prior, len(neg) / (1 - prior))
        if min(len(pos), len(neg)) == 0: chosen.extend(ix); continue
        chosen.extend(rng.choice(pos, max(1, int(n * prior)), replace=False))
        chosen.extend(rng.choice(neg, max(1, int(n * (1 - prior))), replace=False))
    np.savez_compressed(out / 'context.npz', X=F, y=y, balanced=np.sort(chosen), median=median)
    recipe = dict(features=[cols[i] for i in keep] + cols[-4:], n_selected=64, transforms=['raw', 'log'], n_estimators=8,
        checkpoint='tabicl-classifier-v2-20260212.ckpt', log_prep=FeatPrep().fit(F[:, :64]).to_dict(), median=median.tolist(),
        frozen_states={kind: f'tabular_{kind}.pkl.xz' for kind in ['raw', 'log']}, blend_weight=.35, temperature=1.2, priors=priors,
        training_policy=dict(context_priors='Four linear models trained with each context case acquisition family excluded',
                             query_priors='Same four models refitted on all training cases',
                             feature_selection='Weighted L1 C=.2; original clinical and curvature features',
                             raw='All training cases', log='Acquisition-balanced subset; seed16000'))
    (out / 'tabular_recipe.json').write_text(json.dumps(recipe, indent=2))

    # --- feature_order lineage (v14c 235 -> v15a 261 -> v16 -> v17 479) --------------------
    order = list(json.loads(Path(args.base_feature_order).read_text())) if args.base_feature_order else list(cols[:235])
    v15_selected = [cols[i] for i in v15_keep] + META           # build_stage.py:25,44-45
    v16_selected = recipe['features']                           # experiments/v16/prepare.py:17,32-33
    stages = {}
    for stage, names in [('v15a', v15_selected), ('v16', v16_selected), ('v17', cols + mixcols)]:
        for name in names:
            if name not in order: order.append(name)
        stages[stage] = len(order)
    (out / 'feature_order_v17.json').write_text(json.dumps(order, indent=2))

    sel_coef = abs(selector.coef_[0][keep])
    info = dict(rows=int(len(y)), candidates=len(cols), context_shape=list(F.shape), balanced_rows=int(len(chosen)),
                prior_models=len(priors), prior_nonzero=[int(np.count_nonzero(p['coef'])) for p in priors],
                prior_features=[len(p['features']) for p in priors],
                selector_nonzero=int(np.count_nonzero(selector.coef_[0])),
                selected_zero_coef_positions=[int(i) for i in np.where(sel_coef == 0)[0]],
                v15a_selector_nonzero=int(np.count_nonzero(v15_sel.coef_[0])),
                narrow_row_order_matches_rich=narrow_same_order,
                feature_order_lengths=stages, sklearn_warnings=warning_kinds,
                outputs={name: sha256(out / name) for name in ['tabular_recipe.json', 'context.npz', 'feature_order_v17.json']})
    (out / 'prepare_info.json').write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != 'outputs'}), flush=True)


if __name__ == '__main__':
    main()
