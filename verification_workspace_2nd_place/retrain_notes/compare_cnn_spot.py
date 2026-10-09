"""Compare fold-0 spot-retrained CNN checkpoints with the submitted ones, tensor by tensor.

Usage (from verification_workspace_2nd_place/retrain/):
    .venv-train/bin/python -I ../retrain_notes/compare_cnn_spot.py

Pairs:
    outputs/chk_mil48a_fold0.pt   vs inference/assets/mil48a_fold0.pt    (3D HemiMIL, seed 3000)
    outputs/chk_e2dmipf2_fold0.pt vs models/cnn_raw/e2dmipf2_fold0.pt    (2D EfficientNet-B0, seed 9000)

Prints aggregates only: how many state tensors are exactly equal, the largest absolute
difference, and the relative L2 difference of all float weights combined.
"""
from pathlib import Path

import torch

PAIRS = [
    ("3D mil48a fold0", "outputs/chk_mil48a_fold0.pt", "inference/assets/mil48a_fold0.pt"),
    ("2D e2dmipf2 fold0", "outputs/chk_e2dmipf2_fold0.pt", "models/cnn_raw/e2dmipf2_fold0.pt"),
]

for name, new_p, ref_p in PAIRS:
    if not Path(new_p).exists():
        print(f"{name}: {new_p} not found, skipped")
        continue
    new = torch.load(new_p, map_location="cpu", weights_only=False)
    ref = torch.load(ref_p, map_location="cpu", weights_only=False)
    a, b = new["state"], ref["state"]
    if set(a) != set(b):
        print(f"{name}: state keys differ ({len(set(a) ^ set(b))} keys only on one side)")
    keys = sorted(set(a) & set(b))
    n_equal = sum(torch.equal(a[k], b[k]) for k in keys)
    fk = [k for k in keys if a[k].is_floating_point()]
    max_abs = max((a[k] - b[k]).abs().max().item() for k in fk)
    diff2 = sum(((a[k].double() - b[k].double()) ** 2).sum().item() for k in fk)
    norm2 = sum((b[k].double() ** 2).sum().item() for k in fk)
    print(f"{name}: {n_equal}/{len(keys)} tensors exactly equal | max |diff| {max_abs:.3e} "
          f"| relative L2 diff {(diff2 / norm2) ** 0.5:.3e}")
    for extra in ("mu", "sd"):
        if extra in ref:
            print(f"    {extra} equal: {new.get(extra) == ref[extra]}")
