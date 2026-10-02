"""Batch extraction of the 180 scale-space shape descriptors (v17 'shape__' columns).

Reproduces the `if __name__ == '__main__'` block of experiments/20260914/scale_features.py
(kept verbatim in tabular/_original/scale_features_research.py), run with its defaults
(--frame f2, no --intrinsic), which wrote experiments/20260914/scale_f2.csv.

The per-scan function is imported from the shipped inference/scale_features.py, which
is the research module with only the CLI block stripped, so training features and
inference features come from the same code. Input: outputs/crops_v3_<frame>.npz
(keys 'aligned', 'uids'; built by `DAT_MASK_V2=1 DAT_FRAME_V2=1 python cache3.py _f2`).
Output: outputs/tabular/scale_<frame>.csv (uid + 180 unprefixed columns; the
'shape__' prefix is added by prepare_tabular.py, as in the original bench.inputs()).
"""
import os
import sys
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_shipped(name):
    """Import inference/<name>.py without putting inference/ on sys.path."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / 'inference' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame', default='f2'); ap.add_argument('--intrinsic', action='store_true')
    ap.add_argument('--crops', default=None, help='default outputs/crops_v3_<frame>.npz')
    ap.add_argument('--out-dir', default=str(ROOT / 'outputs/tabular'))
    args = ap.parse_args()
    extract_one = load_shipped('scale_features').extract_one
    z = np.load(Path(args.crops) if args.crops else ROOT / 'outputs' / f'crops_v3_{args.frame}.npz')
    t0 = time.time(); rows = []
    for i, crop in enumerate(z['aligned']):
        rows.append(extract_one(crop, intrinsic=args.intrinsic))
        if i % 200 == 0: print('processed', i, 'elapsed', round(time.time() - t0), flush=True)
    d = pd.DataFrame(rows); d.insert(0, 'uid', z['uids'])
    assert np.isfinite(d.drop(columns='uid').to_numpy()).all()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / f'{"scale2" if args.intrinsic else "scale"}_{args.frame}.csv', index=False)
    print('complete', d.shape, 'seconds', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
