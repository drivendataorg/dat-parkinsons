"""
register3.py — replace the 2-axis canonical frame with RIGID REGISTRATION to a
study template.

WHY THIS, AND WHY NOW
---------------------
`preprocessing3.canonical_axes` pins only two things: the L->R axis, from the
striatal pair, and the I->S axis, from the principal axis of the head mask that
lies closest to +z.  The head-mask PCA is estimated on a mask whose extent
depends on the scanner's field of view, and nothing at all pins the remaining
in-plane rotation.  So every fixed-ellipsoid ROI and every fixed grid cell in
`features4/5/6` is read through a frame with a few degrees of subject-dependent
error.

The evidence that this is the binding constraint is already in the notes and was
read as something else.  v5's radial shells — the ONE feature idea in the whole
v4/v5 campaign that worked — are the only block in the table that is *frame
free*: they are centred on each hemisphere's own binding centroid, so frame error
cannot touch them.  Eighteen frame-free numbers matched the entire 217-column
frame-dependent table (0.2460 both).  That is what a frame-limited feature table
looks like.

So: estimate the missing pose per scan by registering to a template built from
the cohort, and read the SAME features in a frame that is actually common.

WHY RIGID AND NOT AFFINE
------------------------
A full affine has scale, and scale is not a nuisance here — it is signal.  An
abnormal striatum is *smaller* as well as fainter, so a 12-parameter fit is free
to shrink the template onto a degenerate blob and register the disease away.
Rotation and translation cannot absorb magnitude or extent, so a 6-DOF fit
removes the nuisance and provably cannot remove the signal.  `--scale` exists to
measure that claim rather than assert it.

The template is built from images only — no labels are read anywhere in this
file — and it is refined by alternating registration and averaging, three passes,
which is enough for a structure this stereotyped.

Per-sample independence: at inference each scan is registered to the frozen
template on its own.  Nothing crosses between test scans.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# transform
# --------------------------------------------------------------------------- #
def rodrigues(w: torch.Tensor) -> torch.Tensor:
    """(B,3) rotation vectors -> (B,3,3) rotation matrices, in ARRAY axis order
    (x = L-R, y = P-A, z = I-S)."""
    th = w.norm(dim=1, keepdim=True).clamp(min=1e-8)
    k = w / th
    K = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(K)
    th = th[:, :, None]
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


# grid_sample's grid orders its last dimension (W, H, D), i.e. the REVERSE of the
# array axes. Everything above is built in array order, so both the rotation and
# the shift get reversed exactly once, here.
_REV = [2, 1, 0]


def warp(vol: torch.Tensor, w: torch.Tensor, t: torch.Tensor,
         s: torch.Tensor | None = None) -> torch.Tensor:
    """Resample (B,1,N,N,N) under rotation `w`, shift `t` (normalised units) and
    optional log-scale `s`."""
    R = rodrigues(w)
    if s is not None:
        R = R * torch.exp(s)[:, None, None]
    R = R[:, _REV][:, :, _REV]
    theta = torch.zeros(vol.shape[0], 3, 4, device=vol.device, dtype=vol.dtype)
    theta[:, :, :3] = R
    theta[:, :, 3] = t[:, _REV]
    grid = F.affine_grid(theta, vol.shape, align_corners=False)
    return F.grid_sample(vol, grid, align_corners=False, padding_mode="border")


def ncc(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    """Per-sample normalised cross-correlation, flattened over the volume.

    `w` is an optional per-voxel weight. Unweighted NCC over the whole 96 mm box
    lets the head outline and the background compete with the striatum for six
    parameters; weighting by the template's own specific binding concentrates the
    fit on the structure whose pose we actually care about.
    """
    a = a.flatten(1)
    b = b.flatten(1)
    if w is None:
        a = a - a.mean(1, keepdim=True)
        b = b - b.mean(1, keepdim=True)
        return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)
    w = w.flatten(1)
    sw = w.sum(1, keepdim=True) + 1e-8
    a = a - (w * a).sum(1, keepdim=True) / sw
    b = b - (w * b).sum(1, keepdim=True) / sw
    num = (w * a * b).sum(1)
    den = torch.sqrt((w * a * a).sum(1) * (w * b * b).sum(1)) + 1e-8
    return num / den


def spec_weight(tpl: torch.Tensor, floor: float = 0.15) -> torch.Tensor:
    """Weight map from the template: specific binding, normalised, plus a floor
    so the surrounding tissue still constrains translation."""
    s = torch.clamp(torch.expm1(tpl), min=0.0) - 1.0
    s = torch.clamp(s, min=0.0)
    s = s / (s.amax() + 1e-8)
    return s + floor


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #
def fit(vols: torch.Tensor, tpl: torch.Tensor, steps=(120, 80), lr=0.05,
        use_scale=False, init=None, w=None):
    """Multi-resolution rigid fit of every volume in `vols` to `tpl`.

    Returns (w, t, s, ncc) with the parameters that map template coordinates to
    scan coordinates. Operates on the whole batch at once; the objective is
    per-sample so the batch is only a vectorisation, never a coupling.
    """
    B = vols.shape[0]
    dev = vols.device
    rot = torch.zeros(B, 3, device=dev, requires_grad=True)
    sh = torch.zeros(B, 3, device=dev, requires_grad=True)
    s = torch.zeros(B, device=dev, requires_grad=True)
    if init is not None:
        with torch.no_grad():
            rot.copy_(init[0]); sh.copy_(init[1]); s.copy_(init[2])
    params = [rot, sh] + ([s] if use_scale else [])
    for level, n_steps in enumerate(steps):
        f = 2 if level == 0 and len(steps) > 1 else 1
        v = F.avg_pool3d(vols, f) if f > 1 else vols
        tt = F.avg_pool3d(tpl, f) if f > 1 else tpl
        ww = None if w is None else (F.avg_pool3d(w, f) if f > 1 else w)
        opt = torch.optim.Adam(params, lr=lr / (level + 1))
        for _ in range(n_steps):
            opt.zero_grad()
            loss = (1.0 - ncc(warp(v, rot, sh, s if use_scale else None), tt, ww)).sum()
            loss.backward()
            opt.step()
    with torch.no_grad():
        q = ncc(warp(vols, rot, sh, s if use_scale else None), tpl, w)
    return rot.detach(), sh.detach(), s.detach(), q


def build_template(vols: torch.Tensor, passes=3, chunk=256, use_scale=False,
                   verbose=True, weighted=False, steps=(120, 80)):
    """Alternate register-to-mean and re-average. No labels are read."""
    tpl = vols.mean(0, keepdim=True)
    W = torch.zeros(vols.shape[0], 3, device=vols.device)
    T = torch.zeros_like(W)
    S = torch.zeros(vols.shape[0], device=vols.device)
    for p in range(passes):
        acc, qs = [], []
        wmap = spec_weight(tpl) if weighted else None
        for i in range(0, vols.shape[0], chunk):
            v = vols[i:i + chunk]
            w, t, s, q = fit(v, tpl.expand(v.shape[0], -1, -1, -1, -1),
                             use_scale=use_scale, steps=steps,
                             w=None if wmap is None else wmap.expand(v.shape[0], -1, -1, -1, -1),
                             init=(W[i:i + chunk], T[i:i + chunk], S[i:i + chunk]))
            W[i:i + chunk], T[i:i + chunk], S[i:i + chunk] = w, t, s
            acc.append(warp(v, w, t, s if use_scale else None))
            qs.append(q)
        tpl = torch.cat(acc).mean(0, keepdim=True)
        if verbose:
            print(f"  [tpl] pass {p}: mean ncc {torch.cat(qs).mean():.4f}", flush=True)
    return tpl, (W, T, S)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="crops_v3_b.npz")
    ap.add_argument("--key", default="aligned")
    ap.add_argument("--out", default="crops_reg.npz")
    ap.add_argument("--template", default="template_v6.npy")
    ap.add_argument("--scale", action="store_true", help="7-DOF instead of 6")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--subset", type=int, default=0,
                    help="build the template from this many scans only, to check "
                         "the result does not depend on the whole cohort")
    ap.add_argument("--weighted", action="store_true",
                    help="weight the NCC by the template's specific binding, so "
                         "the striatum rather than the head outline decides the pose")
    ap.add_argument("--steps", type=int, nargs=2, default=[120, 80])
    args = ap.parse_args()

    dev = device()
    z = np.load(OUT / args.crops, allow_pickle=True)
    sbr = z[args.key].astype(np.float32)
    uids, y = z["uids"].astype(str), z["y"]
    print(f"[reg] {sbr.shape} on {dev}", flush=True)

    # register on log1p: it compresses the striatal peak enough that the head
    # outline and the background also constrain the fit, instead of the striatum
    # alone deciding all six parameters.
    x = torch.from_numpy(np.log1p(np.clip(sbr, 0, None)))[:, None].to(dev)

    t0 = time.time()
    src = x if args.subset <= 0 else x[np.random.default_rng(0).choice(
        len(x), args.subset, replace=False)]
    tpl, _ = build_template(src, passes=args.passes, use_scale=args.scale,
                            weighted=args.weighted, steps=tuple(args.steps))
    print(f"[reg] template built ({time.time()-t0:.0f}s)", flush=True)

    wmap = spec_weight(tpl) if args.weighted else None
    W, T, S, Q = [], [], [], []
    for i in range(0, len(x), 256):
        v = x[i:i + 256]
        w, t, s, q = fit(v, tpl.expand(v.shape[0], -1, -1, -1, -1),
                         use_scale=args.scale, steps=tuple(args.steps),
                         w=None if wmap is None else wmap.expand(v.shape[0], -1, -1, -1, -1))
        W.append(w); T.append(t); S.append(s); Q.append(q)
    W, T, S, Q = torch.cat(W), torch.cat(T), torch.cat(S), torch.cat(Q)
    print(f"[reg] fit ncc mean {Q.mean():.4f} min {Q.min():.4f} "
          f"| rot deg mean {torch.rad2deg(W.norm(dim=1)).mean():.2f} "
          f"max {torch.rad2deg(W.norm(dim=1)).max():.2f} ({time.time()-t0:.0f}s)", flush=True)

    # warp the SBR volume itself (not the log), so downstream features see the
    # same quantity they see today
    raw = torch.from_numpy(sbr)[:, None].to(dev)
    outv = []
    for i in range(0, len(raw), 256):
        outv.append(warp(raw[i:i + 256], W[i:i + 256], T[i:i + 256],
                         S[i:i + 256] if args.scale else None).cpu().numpy()[:, 0])
    outv = np.concatenate(outv).astype(np.float16)

    np.save(OUT / args.template, tpl.cpu().numpy()[0, 0])
    np.savez_compressed(OUT / args.out, aligned=outv, raw=outv,
                        uids=np.array(uids), y=y, sep=z["sep"],
                        rot=W.cpu().numpy(), shift=T.cpu().numpy(),
                        logscale=S.cpu().numpy(), ncc=Q.cpu().numpy(),
                        geom=np.array([96.0, 48], np.float32))
    print(f"[reg] wrote {args.out} and {args.template} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
