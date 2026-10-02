"""Compare a regenerated feature CSV with its archived original (aggregates only).

Used to verify outputs/tabular/scale_f2.csv against experiments/20260914/scale_f2.csv
and outputs/tabular/mixture_shape.csv against experiments/v17/mixture_shape.csv.
Reports sha256 equality, column-name/order equality, uid-set and uid-order equality,
exact-equal cell count, max |diff| and max |diff|/column-std after aligning rows by uid.
Never prints uids or values.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def compare(new, ref):
    a = pd.read_csv(new, dtype={'uid': str}); b = pd.read_csv(ref, dtype={'uid': str})
    r = dict(sha256_equal=sha256(new) == sha256(ref), shape=list(a.shape), shape_ref=list(b.shape),
             columns_equal=list(a.columns) == list(b.columns), column_sets_equal=set(a.columns) == set(b.columns),
             uid_sets_equal=set(a.uid) == set(b.uid), uid_order_equal=bool(len(a) == len(b) and (a.uid.to_numpy() == b.uid.to_numpy()).all()))
    if r['column_sets_equal'] and r['uid_sets_equal']:
        cols = [c for c in b.columns if c != 'uid']
        A = a.set_index('uid').loc[b.uid, cols].to_numpy(float); B = b.set_index('uid')[cols].to_numpy(float)
        diff = np.abs(A - B); sd = B.std(axis=0); sd = np.where(sd > 0, sd, 1.0)
        r.update(cells=int(diff.size), exact_equal_cells=int((diff == 0).sum()), max_abs_diff=float(diff.max()),
                 mean_abs_diff=float(diff.mean()), max_standardized_diff=float((diff / sd).max()),
                 columns_with_any_diff=int((diff > 0).any(axis=0).sum()),
                 rows_with_any_diff=int((diff > 0).any(axis=1).sum()),
                 rows_with_standardized_diff_gt_1e_3=int(((diff / sd) > 1e-3).any(axis=1).sum()))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', nargs=2, action='append', metavar=('NEW', 'REF'), required=True)
    ap.add_argument('--report', default=None)
    args = ap.parse_args()
    rep = {Path(n).name: compare(n, r) for n, r in args.pair}
    print(json.dumps(rep, indent=1), flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(rep, indent=2))


if __name__ == '__main__':
    main()
