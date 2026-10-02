"""replay_cnn_oof.py -- regenerate the CNN out-of-fold predictions from released checkpoints.

Used by the fast reproduction mode (`MODE=fast bash pipeline/train_all.sh`), which
skips CNN training and reuses the released v17 checkpoints. The stacker
(src/fixstack.py) still needs each CNN member's out-of-fold (OOF) predictions.
Those files contain training uids and labels, so they are not distributed.
This script rebuilds them: every fold checkpoint predicts the fold it was held out
from, exactly as the trainers did at the end of each fold, and the result is
written in the same format as src/cnn3.py / src/cnn2d.py write it:

    outputs/oof_<tag>.csv   columns uid, y, fold, <tag>
      3D runs (mil48a mil48b m31000 m33000 m35000 m37000): clipped to [1e-6, 1-1e-6],
          no test-time augmentation (cnn3.py, tta=False; HemiMIL is flip-invariant)
      2D runs (e2dmipf2 e2dmipf22 e2db3f2): unclipped, flip-TTA (cnn2d.predict)

The member averages are then produced by the original scripts:
    python src/avg_member.py mil48y mil48a mil48b m31000 m33000
    python src/avg_2d_seeds.py

Fidelity (measured when this repo was assembled, on the rebuilt caches): replayed
3D OOF differ from the training-time OOF by at most 6e-7, 2D OOF by at most
1.4e-5 (float non-associativity across devices and batch sizes). The stacker
coefficients therefore match the shipped ones to about 1e-6, not bit for bit.

Inputs: outputs/crops_v3_b.npz, outputs/crops_v3_f2.npz, outputs/folds.csv,
checkpoints in --ckpt-dir (default outputs/): <tag>_fold{0..4}.pt (3D) and
<tag>_fold{0..4}.ts.pt (2D). Prints aggregates only.

Usage (repo root):  python tools/replay_cnn_oof.py [--ckpt-dir DIR] [--only 3d|2d] [--force]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))

RUNS3D = ["mil48a", "mil48b", "m31000", "m33000", "m35000", "m37000"]
RUNS2D = ["e2dmipf2", "e2dmipf22", "e2db3f2"]


def load(cache: str):
    z = np.load(OUT / cache, allow_pickle=True)
    uids = z["uids"].astype(str)
    y = z["y"].astype(np.float32)
    folds = pd.read_csv(OUT / "folds.csv").set_index("uid").loc[uids, "fold"].to_numpy()
    return z, uids, y, folds


def write(tag, uids, y, fold, p):
    path = OUT / f"oof_{tag}.csv"
    pd.DataFrame({"uid": uids, "y": y.astype(int), "fold": fold, tag: p}).to_csv(path, index=False)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    print(f"[replay] {tag:<9} OOF ll {log_loss(y, pc):.4f} auc {roc_auc_score(y, pc):.4f} -> {path.name}", flush=True)


def replay_3d(ckpt: Path, force: bool):
    import torch
    import cnn3
    from nets3 import build
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    z, uids, y, fold = load("crops_v3_b.npz")
    x = cnn3.to_input(z["aligned"].astype(np.float32), "log")   # as cnn3.main
    del z
    gc.collect()
    for tag in RUNS3D:
        if (OUT / f"oof_{tag}.csv").exists() and not force:
            print(f"[replay] {tag}: oof exists, kept (use --force to rebuild)")
            continue
        p = np.zeros(len(y))
        for k in range(5):
            va = np.where(fold == k)[0]
            ck = torch.load(ckpt / f"{tag}_fold{k}.pt", map_location="cpu", weights_only=False)
            net = build(ck["arch"], in_ch=ck["in_ch"], drop=ck.get("drop", 0.3))
            net.load_state_dict(ck["state"])
            net.eval()
            xb = torch.from_numpy((x[va] - ck["mu"]) / ck["sd"]).float()
            p[va] = cnn3.predict(net, xb, bs=16, tta=False)
            del net, xb
        write(tag, uids, y, fold, np.clip(p, 1e-6, 1 - 1e-6))


def replay_2d(ckpt: Path, force: bool):
    import torch
    import cnn2d
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    z, uids, y, fold = load("crops_v3_f2.npz")
    crops = z["aligned"].astype(np.float32)
    del z
    # image geometry: the export's <tag>_meta.json if present, else the packaged spec
    cfg_path = ROOT / "inference" / "assets" / "inference_config.json"
    specs = {s["tag"]: s for s in json.loads(cfg_path.read_text())["cnns"]} if cfg_path.exists() else {}
    for tag in RUNS2D:
        if (OUT / f"oof_{tag}.csv").exists() and not force:
            print(f"[replay] {tag}: oof exists, kept (use --force to rebuild)")
            continue
        meta_path = ckpt / f"{tag}_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else \
            specs.get("e2dmipf2" if tag == "e2dmipf22" else tag, {})
        if not meta:
            raise SystemExit(f"no geometry for {tag}: need {meta_path.name} or inference/assets/inference_config.json")
        res = int(meta["res"])
        slabs = tuple(map(tuple, meta.get("slabs", cnn2d.SLABS)))
        views = meta.get("views", "mip3")
        p = np.zeros(len(y))
        for k in range(5):
            ids = np.where(fold == k)[0]
            net = torch.jit.load(str(ckpt / f"{tag}_fold{k}.ts.pt"), map_location=dev).eval()
            X = torch.from_numpy(cnn2d.make_images(crops[ids], slabs=slabs, views=views))
            p[ids] = cnn2d.predict(net, X, res, dev, bs=24)
            del net, X
            gc.collect()
        write(tag, uids, y, fold, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt-dir", default=str(OUT))
    ap.add_argument("--only", choices=["3d", "2d"], default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ckpt = Path(a.ckpt_dir)
    if a.only in (None, "3d"):
        replay_3d(ckpt, a.force)
    if a.only in (None, "2d"):
        replay_2d(ckpt, a.force)


if __name__ == "__main__":
    main()
