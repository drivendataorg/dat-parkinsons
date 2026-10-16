"""Batch extraction of the 60 spatial uptake-mixture descriptors (v17 'mixture__' columns).

Reproduces main() of experiments/v17/mixture_shape.py (kept verbatim in
tabular/_original/mixture_shape.py), which wrote experiments/v17/mixture_shape.csv and
mixture_shape_checks.json: OMP/OPENBLAS threads 1, ProcessPoolExecutor(max_workers=3),
pool.map(extract, crops, chunksize=4), finiteness assert, uid column first.

The per-scan function is imported from the shipped inference/mixture_features.py,
which prepare_candidate.py cut byte-for-byte out of mixture_shape.py (from 'AX=' to
'def main():'), so training features and inference features come from the same code.
Input: outputs/crops_v3_f2.npz ('aligned', 'uids'). Output: outputs/tabular/
mixture_shape.csv (uid + 60 unprefixed columns) and mixture_shape_checks.json.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import sys
sys.dont_write_bytecode = True
import argparse
import importlib.util
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_shipped(name):
    """Import inference/<name>.py without putting inference/ on sys.path.

    Runs at import time so spawned pool workers (which re-import this module)
    resolve the pickled function reference to the same shipped module.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / 'inference' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


extract = load_shipped('mixture_features').extract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crops', default=str(ROOT / 'outputs/crops_v3_f2.npz'))
    ap.add_argument('--out-dir', default=str(ROOT / 'outputs/tabular'))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    z = np.load(args.crops); crops = z['aligned']; u = z['uids']; t0 = time.time(); rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        for row in pool.map(extract, crops, chunksize=4):
            rows.append(row)
            if len(rows) % 200 == 0: print('Mixture shape cases processed', len(rows), 'seconds', round(time.time() - t0), flush=True)
    d = pd.DataFrame(rows); assert np.isfinite(d.to_numpy()).all(); d.insert(0, 'uid', u); d.to_csv(out / 'mixture_shape.csv', index=False)
    r = dict(rows=len(d), features=len(d.columns) - 1, seconds=round(time.time() - t0))
    (out / 'mixture_shape_checks.json').write_text(json.dumps(r, indent=2)); print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
