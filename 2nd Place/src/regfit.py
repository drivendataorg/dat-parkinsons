"""
regfit.py — the registration applied at INFERENCE, and by the training cache, so
the two cannot drift.

`register3.py` estimates the residual pose for the whole cohort at once, which is
how the template gets built.  This module exposes the same estimate-and-warp for
one scan against a frozen template, and `augfeat.py`/`cache3.py` use exactly the
same two calls, so the feature table the models are fitted on and the feature
table computed inside the container come from one implementation.

WHY THE CROP IS WARPED RATHER THAN RE-SAMPLED FROM THE NATIVE VOLUME
-------------------------------------------------------------------
Both were built and measured.  Composing the correction into the canonical
rotation and re-sampling the NIfTI once is the theoretically cleaner operation —
one interpolation instead of two, and measurably sharper crops — and it is worth
0.0006 log loss, i.e. nothing, while costing a second full resampling of the
native volume per scan.  Warping the 48^3 crop is cheaper and marginally better,
so that is what ships.

WHAT IS *NOT* GUARDED, AND WHY
------------------------------
The obvious safety rule — reject a fit that rotates by more than ~30 degrees as a
failure — is wrong here, and measurably so:

    max rotation kept    15      25      40      60      90(=all)
    5-fold CV          0.2566  0.2550  0.2482  0.2452  0.2450
    leave-site-out     0.2761  0.2754  0.2637  0.2638  0.2638

Both surfaces improve monotonically as the guard is relaxed.  The large
corrections are real: the canonical frame takes its I-S axis from the principal
axis of the head mask, which is badly estimated when the field of view includes a
lot of neck, so a 40-60 degree correction is a fix, not a failure.  Only
catastrophic outcomes are rejected — a non-finite parameter, or a correlation so
low that the fit plainly did not converge — and rejection means falling back to
the canonical frame, which is exactly what v3 and v5 shipped.

PER-SAMPLE INDEPENDENCE.  The template is a frozen asset built from training
scans and shipped in the package.  Each test scan is fitted against it alone;
nothing about one test scan can reach another.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import register3 as R3

OUT = Path(__file__).resolve().parent.parent / "outputs"

MAX_DEG = 90.0        # see the table above: this is a sanity bound, not a filter
MIN_NCC = 0.30


def load_template(path="template_v6.npy", device=None) -> torch.Tensor:
    t = np.load(path if Path(path).exists() else OUT / path)
    dev = device or torch.device("cpu")
    return torch.from_numpy(t.astype(np.float32))[None, None].to(dev)


def _reject(rot, sh, q, max_deg, min_ncc):
    deg = np.rad2deg(np.linalg.norm(rot, axis=1))
    return ((deg > max_deg) | (q < min_ncc)
            | ~np.isfinite(rot).all(1) | ~np.isfinite(sh).all(1))


def fit_params(crops: np.ndarray, tpl: torch.Tensor, steps=(120, 80),
               weighted=False, chunk=256, max_deg=MAX_DEG, min_ncc=MIN_NCC):
    """Estimate (rot, shift) for a batch of canonical SBR crops.

    The objective is a sum of per-sample terms and Adam is elementwise, so a
    batch is pure vectorisation: fitting one scan alone gives the same answer as
    fitting it inside a batch of 256. That is what lets the training cache use
    the fast batched path and the container use batch size 1.
    """
    x = torch.from_numpy(np.log1p(np.clip(crops.astype(np.float32), 0, None)))[:, None]
    x = x.to(tpl.device)
    wmap = R3.spec_weight(tpl) if weighted else None
    rots, shs, qs = [], [], []
    for i in range(0, len(x), chunk):
        v = x[i:i + chunk]
        e = v.shape[0]
        rot, sh, _, q = R3.fit(
            v, tpl.expand(e, -1, -1, -1, -1), steps=steps,
            w=None if wmap is None else wmap.expand(e, -1, -1, -1, -1))
        rots.append(rot.cpu().numpy()); shs.append(sh.cpu().numpy()); qs.append(q.cpu().numpy())
    rot = np.concatenate(rots).astype(np.float32)
    sh = np.concatenate(shs).astype(np.float32)
    q = np.concatenate(qs)
    bad = _reject(rot, sh, q, max_deg, min_ncc)
    rot[bad] = 0.0
    sh[bad] = 0.0
    return rot, sh, q, bad


def apply_params(crops: np.ndarray, rot: np.ndarray, sh: np.ndarray,
                 chunk=256, device=None) -> np.ndarray:
    """Warp canonical SBR crops into the template frame."""
    dev = device or torch.device("cpu")
    out = []
    for i in range(0, len(crops), chunk):
        v = torch.from_numpy(np.ascontiguousarray(
            crops[i:i + chunk].astype(np.float32)))[:, None].to(dev)
        w = torch.from_numpy(rot[i:i + chunk]).to(dev)
        t = torch.from_numpy(sh[i:i + chunk]).to(dev)
        out.append(R3.warp(v, w, t).cpu().numpy()[:, 0])
    return np.concatenate(out)


def register(crops: np.ndarray, tpl: torch.Tensor, steps=(120, 80),
             weighted=False, chunk=256, max_deg=MAX_DEG, min_ncc=MIN_NCC):
    """fit + apply, the whole operation in one call. This is what both the
    training cache and `main.py` use, so there is one code path."""
    rot, sh, q, bad = fit_params(crops, tpl, steps=steps, weighted=weighted,
                                 chunk=chunk, max_deg=max_deg, min_ncc=min_ncc)
    return apply_params(crops, rot, sh, chunk=chunk, device=tpl.device), rot, sh, q, bad
