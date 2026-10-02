"""Compare prepare_tabular.py outputs with the archived v17 originals (aggregates only).

Checks, for the verification step of the v17 tabular rebuild:
  * tabular_recipe.json vs the shipped assets/tabular_recipe.json: sha256, then a
    structured diff (per top-level key: list/name mismatches, max |numeric diff|)
  * context.npz vs experiments/v17/rebuild/context.npz (optional; only in the original
    project tree): per array shape, exact-equal count, max |diff| (NaN-aware); never
    prints array contents
  * feature_order_v17.json vs feature_order in the shipped assets/inference_config.json
Prints and optionally writes a JSON report. No per-scan value is printed.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def flatten(obj, prefix=''):
    """Yield (path, leaf) pairs of a JSON structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f'{prefix}.{k}' if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f'{prefix}[{i}]')
    else:
        yield prefix, obj


def json_diff(a, b):
    fa = dict(flatten(a)); fb = dict(flatten(b))
    report = {}
    tops = list(dict.fromkeys(list(a.keys()) + list(b.keys())))
    for top in tops:
        ka = {k: v for k, v in fa.items() if k == top or k.startswith(top + '.') or k.startswith(top + '[')}
        kb = {k: v for k, v in fb.items() if k == top or k.startswith(top + '.') or k.startswith(top + '[')}
        missing = len(set(ka) ^ set(kb))
        num_diffs = []; other_mismatch = 0; exact = 0
        for k in set(ka) & set(kb):
            x, y = ka[k], kb[k]
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool):
                d = abs(float(x) - float(y)) if np.isfinite(x) and np.isfinite(y) else (0.0 if (x == y or (np.isnan(x) and np.isnan(y))) else np.inf)
                num_diffs.append(d); exact += d == 0
            else:
                other_mismatch += x != y; exact += x == y
        report[top] = dict(leaves=len(ka), leaves_ref=len(kb), structural_mismatch=missing, exact_equal=int(exact),
                           non_numeric_mismatch=int(other_mismatch),
                           max_abs_numeric_diff=float(max(num_diffs)) if num_diffs else 0.0)
    return report


def compare_arrays(a, b):
    out = {}
    for k in sorted(set(a.files) | set(b.files)):
        if k not in a.files or k not in b.files:
            out[k] = 'missing in one file'; continue
        x, y = a[k], b[k]
        r = dict(shape=list(x.shape), shape_ref=list(y.shape), dtype=str(x.dtype), dtype_ref=str(y.dtype))
        if x.shape == y.shape:
            xf = x.astype(np.float64); yf = y.astype(np.float64)
            same = (xf == yf) | (np.isnan(xf) & np.isnan(yf))
            r.update(exact_equal=int(same.sum()), total=int(same.size),
                     max_abs_diff=float(np.nanmax(np.abs(xf - yf))) if (~same).any() else 0.0,
                     bitwise_equal=bool(np.array_equal(x, y, equal_nan=True)))
            if xf.ndim == 2 and (~same).any():
                r['columns_with_diff'] = [int(c) for c in np.where((~same).any(axis=0))[0]]
        out[k] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=str(ROOT / 'outputs/tabular'), help='prepare_tabular.py output dir')
    ap.add_argument('--ref-recipe', required=True, help='shipped assets/tabular_recipe.json')
    ap.add_argument('--ref-context', default=None,
                    help='archived experiments/v17/rebuild/context.npz (optional; skipped when absent)')
    ap.add_argument('--ref-config', required=True, help='shipped assets/inference_config.json')
    ap.add_argument('--report', default=None, help='optional JSON report path')
    args = ap.parse_args()
    d = Path(args.dir)
    rep = {}
    ra, rb = d / 'tabular_recipe.json', Path(args.ref_recipe)
    rep['recipe_sha256_equal'] = sha256(ra) == sha256(rb)
    rep['recipe_sha256'] = sha256(ra)
    A = json.loads(ra.read_text()); B = json.loads(rb.read_text())
    rep['recipe_json_equal'] = A == B
    if not rep['recipe_json_equal']:
        rep['recipe_diff'] = json_diff(A, B)
        fa, fb = A['features'], B['features']
        rep['recipe_feature_positions_differing'] = [i for i in range(min(len(fa), len(fb))) if fa[i] != fb[i]]
        rep['recipe_feature_sets_equal'] = set(fa) == set(fb)
    if args.ref_context:
        with np.load(d / 'context.npz') as a, np.load(args.ref_context) as b:
            rep['context'] = compare_arrays(a, b)
    fo = json.loads((d / 'feature_order_v17.json').read_text())
    ref_fo = json.loads(Path(args.ref_config).read_text())['feature_order']
    rep['feature_order'] = dict(length=len(fo), length_ref=len(ref_fo), equal=fo == ref_fo,
                                first_mismatch=next((i for i in range(min(len(fo), len(ref_fo))) if fo[i] != ref_fo[i]), None),
                                same_set=set(fo) == set(ref_fo))
    print(json.dumps(rep, indent=1), flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(rep, indent=2))


if __name__ == '__main__':
    main()
