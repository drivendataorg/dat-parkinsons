"""Prediction-level check of frozen TabICL contexts through the shipped TabularAdapter.

The pkl.xz states cannot be byte-reproduced across library versions, so this compares
predictions instead. It rebuilds the (n, 479) training feature matrix in feature_order
(rich table + 'shape__' + descriptors + 'mixture__' columns, the adapter's input
contract), loads inference/tabular_adapter.py:TabularAdapter on an assets directory
(tabular_recipe.json + tabular_raw/log.pkl.xz + checkpoint) on CPU, and predicts all
rows in chunks (TabICL predictions are row-independent). Per-model ('raw', 'log') and
blended tabular probabilities are saved to an .npz; only aggregates are printed.

  predict:  --assets DIR [--ckpt FILE] --save FILE.npz
  compare:  --compare A.npz B.npz   (max/mean |dp|, correlation, log loss/AUROC of each)

If DIR lacks the checkpoint, a temporary directory of symlinks is used.
"""
import os
import sys
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True
import argparse
import importlib.util
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ['head_ml', 'inner_ml', 'ref', 'native_spacing']


def read(path):
    return pd.read_csv(path, dtype={'uid': str})


def feature_matrix(inputs_dir, shape_csv, mixture_csv, order):
    rich = read(Path(inputs_dir) / 'features_v13_m2fnh.csv').set_index('uid')
    u = rich.index
    meta = read(Path(inputs_dir) / 'features_v13_m2f.csv').set_index('uid').loc[u, META]
    shape = read(shape_csv).set_index('uid').loc[u]
    mix = read(mixture_csv).set_index('uid').loc[u]
    cols = []
    for name in order:
        if name.startswith('shape__'): cols.append(shape[name[len('shape__'):]])
        elif name.startswith('mixture__'): cols.append(mix[name[len('mixture__'):]])
        elif name in META: cols.append(meta[name])
        else: cols.append(rich[name])
    M = np.column_stack([c.to_numpy(float) for c in cols])
    return M, rich.is_pathologic.to_numpy(int)


def load_adapter(assets, order):
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'inference/tabicl_lib.zip')]
    spec = importlib.util.spec_from_file_location('tabular_adapter', ROOT / 'inference/tabular_adapter.py')
    mod = importlib.util.module_from_spec(spec); sys.modules['tabular_adapter'] = mod; spec.loader.exec_module(mod)
    return mod.TabularAdapter(str(assets), order, device='cpu')


def predict(args):
    import torch
    torch.set_num_threads(3)
    order = json.loads(Path(args.feature_order).read_text())
    M, y = feature_matrix(args.inputs_dir, args.shape_csv, args.mixture_csv, order)
    assets = Path(args.assets)
    recipe = json.loads((assets / 'tabular_recipe.json').read_text())
    tmp = None
    if not (assets / recipe['checkpoint']).exists():
        if not args.ckpt: raise SystemExit('assets dir lacks the checkpoint; pass --ckpt')
        tmp = tempfile.TemporaryDirectory(prefix='tabular_assets_')
        for name in ['tabular_recipe.json', *recipe['frozen_states'].values()]:
            os.symlink((assets / name).resolve(), Path(tmp.name) / name)
        os.symlink(Path(args.ckpt).resolve(), Path(tmp.name) / recipe['checkpoint'])
        assets = Path(tmp.name)
    t0 = time.time()
    ad = load_adapter(assets, order)
    print('adapter loaded', round(time.time() - t0), 's', flush=True)
    per = {kind: [] for kind, _ in ad.models}; blend = []
    for s in range(0, len(M), args.chunk):
        full = M[s:s + args.chunk]
        X = full[:, ad.order]
        if ad.priors:
            scores = np.column_stack([prep.transform(full[:, idx]) @ coef + intercept for idx, prep, coef, intercept in ad.priors])
            X = np.column_stack([X, scores])
        X = np.where(np.isfinite(X), X, ad.median)
        parts = {kind: model.predict_proba(ad._transform(X, kind))[:, 1] for kind, model in ad.models}
        for kind in parts: per[kind].append(parts[kind])
        blend.append(np.mean([parts[k] for k, _ in ad.models], axis=0))
        if s == 0:  # same as the adapter's own predict() on this chunk
            check = float(np.abs(ad.predict(full) - blend[-1]).max())
            print('adapter.predict consistency (first chunk) max |dp|', check, flush=True)
            assert check < 1e-7
        print('rows', min(s + args.chunk, len(M)), 'seconds', round(time.time() - t0), flush=True)
    out = {f'p_{k}': np.concatenate(v) for k, v in per.items()}; out['p'] = np.concatenate(blend); out['y'] = y
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.save, **out)
    print(json.dumps(dict(rows=len(y), seconds=round(time.time() - t0), **metrics(y, out['p']))), flush=True)
    if tmp: tmp.cleanup()


def metrics(y, p):
    from sklearn.metrics import log_loss, roc_auc_score
    q = np.clip(p, .005, .995)
    return dict(logloss=float(log_loss(y, q)), auroc=float(roc_auc_score(y, q)))


def compare(a_path, b_path, report):
    a = np.load(a_path); b = np.load(b_path); assert np.array_equal(a['y'], b['y'])
    rep = {}
    for key in [k for k in a.files if k.startswith('p')]:
        d = np.abs(a[key] - b[key])
        rep[key] = dict(max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()), p99_abs_diff=float(np.percentile(d, 99)),
                        rows_diff_gt_0_01=int((d > .01).sum()), corr=float(np.corrcoef(a[key], b[key])[0, 1]),
                        a=metrics(a['y'], a[key]), b=metrics(b['y'], b[key]))
    print(json.dumps(rep, indent=1), flush=True)
    if report: Path(report).write_text(json.dumps(rep, indent=2))


def walk(obj, path, out, seen):
    """Collect every tensor/ndarray/scalar leaf of a pickled TabICL state by attribute path."""
    import torch
    if id(obj) in seen: return
    if isinstance(obj, torch.Tensor): out[path] = obj.detach().cpu().to(torch.float64).numpy() if obj.is_floating_point() else obj.cpu().numpy(); return
    if isinstance(obj, np.ndarray): out[path] = obj; return
    if isinstance(obj, (int, float, bool, str, type(None), np.generic)): out[path] = obj; return
    seen.add(id(obj))
    if isinstance(obj, dict): items = obj.items()
    elif isinstance(obj, (list, tuple)): items = enumerate(obj)
    elif hasattr(obj, '__dict__'): items = vars(obj).items()
    else: return
    for k, v in items: walk(v, f'{path}/{k}', out, seen)


def compare_states(a_path, b_path, report):
    """Leaf-by-leaf comparison of two frozen states (aggregates only). Loads them one after the other."""
    import gc, lzma, pickle
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'inference/tabicl_lib.zip')]
    leaves = []
    for p in [a_path, b_path]:
        with lzma.open(p, 'rb') as f: state = pickle.load(f)
        out = {}; walk(state, '', out, set()); leaves.append(out); del state; gc.collect()
    A, B = leaves
    rep = dict(leaves=len(A), leaves_ref=len(B), only_in_one=sorted(set(A) ^ set(B))[:20], arrays=0, arrays_bitwise_equal=0,
               array_max_abs_diff=0.0, kv_cache_arrays=0, kv_cache_max_abs_diff=0.0, scalar_mismatches=[])
    for k in sorted(set(A) & set(B)):
        a, b = A[k], B[k]
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            rep['arrays'] += 1
            same = a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == 'f')
            rep['arrays_bitwise_equal'] += bool(same)
            if a.shape == b.shape and a.dtype.kind in 'fiub' and not same:
                d = float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                rep['array_max_abs_diff'] = max(rep['array_max_abs_diff'], d)
                if 'kv_cache' in k: rep['kv_cache_max_abs_diff'] = max(rep['kv_cache_max_abs_diff'], d)
            if 'kv_cache' in k: rep['kv_cache_arrays'] += 1
        elif not isinstance(a, np.ndarray) and not isinstance(b, np.ndarray):
            if a != b and not (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b)):
                rep['scalar_mismatches'].append(k)
        else:
            rep['scalar_mismatches'].append(k)
    print(json.dumps(rep, indent=1), flush=True)
    if report: Path(report).write_text(json.dumps(rep, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compare-states', nargs=2, metavar=('A', 'B'), help='compare two pkl.xz states leaf by leaf')
    ap.add_argument('--assets'); ap.add_argument('--ckpt'); ap.add_argument('--save')
    ap.add_argument('--inputs-dir', default=str(ROOT / 'outputs'))
    ap.add_argument('--shape-csv', default=str(ROOT / 'outputs/tabular/scale_f2.csv'))
    ap.add_argument('--mixture-csv', default=str(ROOT / 'outputs/tabular/mixture_shape.csv'))
    ap.add_argument('--feature-order', default=str(ROOT / 'outputs/tabular/feature_order_v17.json'))
    ap.add_argument('--chunk', type=int, default=128)
    ap.add_argument('--compare', nargs=2, metavar=('A', 'B')); ap.add_argument('--report')
    args = ap.parse_args()
    if args.compare: compare(*args.compare, args.report)
    elif args.compare_states: compare_states(*args.compare_states, args.report)
    else: predict(args)


if __name__ == '__main__':
    main()
