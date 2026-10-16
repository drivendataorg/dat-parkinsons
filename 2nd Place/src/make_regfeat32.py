"""
make_regfeat32.py — the registered training tables, built at the SAME PRECISION
`main.py` uses at inference.

The defect this fixes. `cache3.py` extracts the canonical-frame features from the
full-precision crop and only *then* casts to float16 for storage (RESEARCH_NOTES
§8 — that provenance question was settled once already). But `make_regfeat.py`
loads `crops_v3_b.npz["aligned"]`, which is the float16 array, registers *that*,
and extracts from it. So every registered table since v6 has been float16-derived
while `main.py` registers and reads a float32 crop. The models are fitted on
slightly different numbers than they are served.

Through v6 this was tolerable and documented: gate 5 read 6.9e-4 mean on the
150-scan verify set and HANDOFF §0.00 attributes the residual to "the reference
(float16 crop cache), not the package". v7 makes it matter more. Unclamping
`half_sep` means ROI masks are keyed at 0.1 mm resolution for the 43% of scans
that previously all shared one mask, so a sub-0.1 mm difference in detected
separation can now select a *different mask* — a discrete jump, not a rounding
error. Gate 5 went 1.60e-02 max / 6.9e-4 mean to 3.34e-02 / 9.4e-4.

The fix is to remove the discrepancy at its source rather than to widen the gate's
tolerance: preprocess from the NIfTI at float32, exactly as `main.py` does, and
extract from that. Registration is then the only remaining backend-dependent step.

Cost: a full preprocessing pass (~0.63 s/scan single-threaded, parallelised here)
instead of loading a cache. Worth it once.

Usage:
    python -u make_regfeat32.py --rich features_v13_reg7f.csv \
                                --narrow features_v4_reg7f.csv
"""
from __future__ import annotations

import argparse
import importlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

import preprocessing3 as pp
import regfit

ROOT = Path(__file__).resolve().parent.parent
# Competition data (niftis/ + labels.csv). Patched for the prize repo; was ROOT/"data-demo".
DATA = Path(os.environ.get("DAT_DATA_DIR", str(ROOT / "data" / "raw")))
OUT = ROOT / "outputs"
NIFTI = DATA / "niftis"


def _one(uid: str):
    """Exactly main.py's per-scan preprocessing call for the 96 mm / 48^3 box."""
    try:
        d = pp.preprocess_full(NIFTI / f"{uid}.nii.gz", crop_mm=96.0, out_vox=48,
                               out_spacing=2.0)
        return uid, d["aligned"].astype(np.float32), float(d["sep"]), d["meta"], True
    except Exception:                                            # noqa: BLE001
        return uid, np.zeros((48, 48, 48), np.float32), 38.0, {}, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--template", default="template_v6.npy")
    ap.add_argument("--module", default="features13")
    ap.add_argument("--rich", default="features_v13_reg7f.csv")
    ap.add_argument("--narrow", default="features_v4_reg7f.csv")
    ap.add_argument("--narrow-like", default="features_v4_c.csv")
    ap.add_argument("--steps", type=int, nargs=2, default=[120, 80])
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("REFEAT_JOBS", 6)))
    ap.add_argument("--device", default=None, help="torch device for the registration fit (default: mps/cuda/cpu)")
    args = ap.parse_args()

    lab = pd.read_csv(DATA / args.labels)
    uids = lab["uid"].astype(str).tolist()
    y = lab["is_pathologic"].to_numpy(np.float32)

    t0 = time.time()
    res = Parallel(n_jobs=args.jobs, batch_size=8)(delayed(_one)(u) for u in uids)
    bad = [u for u, _, _, _, ok in res if not ok]
    al = np.stack([r[1] for r in res])
    sep = np.array([r[2] for r in res], np.float64)
    meta = [r[3] for r in res]
    print(f"[regfeat32] preprocessed {len(uids)} at float32 "
          f"({len(bad)} failed) in {time.time() - t0:.0f}s", flush=True)

    dev = (torch.device(args.device) if args.device else
           torch.device("mps") if torch.backends.mps.is_available()
           else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    tpl = regfit.load_template(args.template, dev)
    t1 = time.time()
    reg, rot, sh, q, rej = regfit.register(al, tpl, steps=tuple(args.steps))
    print(f"[regfeat32] registered | mean ncc {q.mean():.4f} | rejected {int(rej.sum())} "
          f"| {time.time() - t1:.0f}s", flush=True)

    mod = importlib.import_module(args.module)
    feats = Parallel(n_jobs=args.jobs, batch_size=16)(
        delayed(mod.extract)(reg[i].astype(np.float32), float(sep[i]), meta[i])
        for i in range(len(uids)))
    rich = pd.DataFrame(feats)
    rich.insert(0, "uid", uids)
    rich["is_pathologic"] = y
    rich.to_csv(OUT / args.rich, index=False)

    like = pd.read_csv(OUT / args.narrow_like, nrows=2)
    cols = [c for c in like.columns if c in rich.columns]
    missing = [c for c in like.columns if c not in rich.columns]
    if missing:
        raise SystemExit(f"narrow table would be missing {missing[:5]}")
    rich[cols].to_csv(OUT / args.narrow, index=False)
    print(f"[regfeat32] wrote {args.rich} ({rich.shape[1] - 2} feats) and "
          f"{args.narrow} ({len(cols) - 2} feats)", flush=True)


if __name__ == "__main__":
    main()
