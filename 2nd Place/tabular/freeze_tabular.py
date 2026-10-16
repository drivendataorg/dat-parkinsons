"""Fit and freeze the two v17 TabICL contexts (tabular_raw.pkl.xz, tabular_log.pkl.xz).

Port of experiments/v17/freeze_candidate.py. Settings are unchanged:
TabICLClassifier(n_estimators=8, batch_size=1, device='cpu', use_amp=False,
use_fa3=False, kv_cache='repr', allow_auto_download=False, random_state=42,
n_jobs=3, inference_config safety_factor .08 / min_batch_size 1 for COL/ROW/ICL),
torch.set_num_threads(3); 'raw' = all rows of context.npz, 'log' = FeatPrep(log_prep)
on columns 0-63 plus raw columns 64-71 on the balanced rows. The 4 reserved CLS
column-cache slots are zeroed, only the KV cache is saved (no weights, no training
arrays), state['model_path'] is the bare checkpoint filename, and the state is
written with pickle protocol 5 inside lzma preset 6 via a .tmp file. The reload
check on probe rows A[::37] must match to < 1e-6.

Differences from the original: the TabICL 2.2.0 code is imported from the shipped
inference/tabicl_lib.zip and FeatPrep from src/portable3.py (byte-identical to the
shipped copy); the original prepended overlay dirs with the platform's library
versions (numpy 2.2.6, scikit-learn 1.8.0, torch 2.12.1, scipy 1.15.3), whereas
this runs under whatever the active environment provides. Paths are arguments;
CANDIDATE.json is not touched. Writes persistence_checks.json next to the states.

Determinism: tabicl orders its ensemble groups by list(set(norm_methods)), which
follows Python's string hash seed. The original run was unseeded and got
['none', 'power']; PYTHONHASHSEED=1 reproduces that order (07_tabular.sh sets it),
which makes predictions match the shipped states exactly. The pkl.xz bytes still
differ run to run because torch embeds a per-process serialization id when
pickling tensors.
"""
import os
import sys
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
sys.dont_write_bytecode = True
import argparse
import gc
import hashlib
import json
import lzma
import pickle
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'inference/tabicl_lib.zip')]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from portable3 import FeatPrep  # noqa: E402
from tabicl import TabICLClassifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recipe', default=str(ROOT / 'outputs/tabular/tabular_recipe.json'))
    ap.add_argument('--context', default=None, help='context.npz (default: next to the recipe)')
    ap.add_argument('--ckpt', default=os.environ.get('TABICL_CKPT'),
                    help='TabICL checkpoint file (default: $TABICL_CKPT, else inference/assets/<recipe checkpoint>)')
    ap.add_argument('--out-dir', default=None, help='default: the recipe directory')
    ap.add_argument('--kinds', default=None, help='comma list subset of the recipe transforms (default: all)')
    args = ap.parse_args()

    torch.set_num_threads(3)
    recipe_path = Path(args.recipe)
    recipe = json.loads(recipe_path.read_text())
    out = Path(args.out_dir) if args.out_dir else recipe_path.parent
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt) if args.ckpt else ROOT / 'inference/assets' / recipe['checkpoint']
    if not ckpt.exists():
        raise SystemExit(f'TabICL checkpoint not found: {ckpt} (pass --ckpt)')
    ctx = np.load(Path(args.context) if args.context else recipe_path.parent / 'context.npz')
    X = ctx['X']; y = ctx['y']; idx = ctx['balanced']
    prep = FeatPrep.from_dict(recipe['log_prep']); t0 = time.time()
    checks_path = out / 'persistence_checks.json'
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
    kinds = args.kinds.split(',') if args.kinds else recipe['transforms']
    for kind in kinds:
        A = X if kind == 'raw' else np.column_stack([prep.transform(X[:, :64]), X[:, 64:]])
        use = np.arange(len(y)) if kind == 'raw' else idx
        model = TabICLClassifier(n_estimators=8, batch_size=1, device='cpu', use_amp=False, use_fa3=False, kv_cache='repr',
            allow_auto_download=False, model_path=str(ckpt), random_state=42, n_jobs=3, verbose=False,
            inference_config={key: dict(safety_factor=.08, min_batch_size=1) for key in ['COL_CONFIG', 'ROW_CONFIG', 'ICL_CONFIG']})
        model.fit(A[use], y[use]); probe = A[::37]; before = model.predict_proba(probe)[:, 1]
        # tabicl 2.2.0 groups ensemble members with list(set(norm_methods)), so their order (and the
        # float32 summation order of the ensemble average) follows the string hash seed. The shipped
        # states have ['none', 'power'] (reproduced with PYTHONHASHSEED=1; see pipeline/07_tabular.sh).
        ensemble_order = list(model.model_kv_cache_.keys())
        if ensemble_order != ['none', 'power']:
            print(f'WARNING: ensemble order {ensemble_order} differs from the shipped states ["none", "power"]; '
                  'predictions will differ at the 1e-6 level. Set PYTHONHASHSEED=1 to match.', flush=True)
        reserved = model.model_.col_embedder.reserve_cls_tokens
        assert reserved == 4
        for cache in model.model_kv_cache_.values():
            for kv in cache.col_cache.kv.values(): kv.key[:, :reserved].zero_(); kv.value[:, :reserved].zero_()
        model._save_model_weights = False; model._save_kv_cache = True; model._save_training_data = False
        state = model.__getstate__(); state['model_path'] = recipe['checkpoint']; eg = state['ensemble_generator_']
        assert eg.X_ is None and eg.y_ is None and all(p.X_transformed_ is None for p in eg.preprocessors_.values())
        path = out / recipe['frozen_states'][kind]; tmp = path.with_suffix('.tmp')
        with lzma.open(tmp, 'wb', preset=6) as f: pickle.dump(state, f, protocol=5)
        tmp.replace(path)
        del state, model; gc.collect()
        with lzma.open(path, 'rb') as f: state = pickle.load(f)
        state['model_path'] = str(ckpt); state['device'] = 'cpu'
        restored = TabICLClassifier.__new__(TabICLClassifier); restored.__setstate__(state); after = restored.predict_proba(probe)[:, 1]
        error = float(abs(before - after).max()); assert error < 1e-6
        with path.open('rb') as f: digest = hashlib.file_digest(f, 'sha256').hexdigest()
        checks[kind] = dict(max_probability_difference=error, bytes=path.stat().st_size, sha256=digest, raw_records_removed=True,
                            rows=int(len(use)), columns=int(A.shape[1]), ensemble_order=ensemble_order,
                            pythonhashseed=os.environ.get('PYTHONHASHSEED'))
        print(kind, 'frozen', json.dumps(checks[kind]), 'seconds', round(time.time() - t0), flush=True)
        del restored, state; gc.collect()
        checks_path.write_text(json.dumps(checks, indent=2))
    env = dict(numpy=np.__version__, torch=torch.__version__)
    try:
        import sklearn, scipy
        env.update(sklearn=sklearn.__version__, scipy=scipy.__version__)
    except ImportError:
        pass
    checks['environment'] = env
    checks_path.write_text(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
