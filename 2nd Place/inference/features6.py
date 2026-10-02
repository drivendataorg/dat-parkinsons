"""
features6.py — features5 plus per-hemisphere shape descriptors, ordered
"more affected side first".

The hemisphere-MIL CNN works because Parkinson's starts unilaterally: the
informative object is a single striatum, not the pair. The v4/v5 tables capture
that only through scalar min/max/asymmetry of ROI means. This module gives each
hemisphere its own shape description — how compact the surviving uptake is, how
sharp its boundary, how far it extends — and orders the two sides by uptake so
the descriptor is side-invariant.
"""
from __future__ import annotations

import os

import numpy as np
from scipy import ndimage

import features5
from features4 import VOX_MM, _X, _Y, _Z

_HALF = None

# Isocontour anchor for the per-hemisphere shape descriptors; see _hemi_stats.
# "max" reproduces every table built before 2026-07-26 exactly.
_ANCHOR = os.environ.get("DAT_ISO_ANCHOR", "max")


def _half_masks():
    global _HALF
    if _HALF is None:
        _HALF = (_X < 0, _X > 0)
    return _HALF


def _hemi_stats(c: np.ndarray, side: np.ndarray, anchor: str | None = None) -> dict:
    v = c[side]
    if v.size < 8:
        return {}
    peak = float(v.max())
    p99 = float(np.percentile(v, 99))
    p90 = float(np.percentile(v, 90))
    out = {"peak": peak, "mean": float(v.mean()), "p99": p99, "p90": p90}
    # THE ISOCONTOUR ANCHOR. Every shape descriptor below (hi_vol, elong, flat,
    # rms_mm, ap_extent, ap_centroid, ncc) is measured on this mask, so whatever
    # `thr` is anchored on is a quantity the whole per-hemisphere block is read
    # through — the same structural position as the reference level (v3), the
    # canonical frame (v6) and the ROI placement bound (v7).
    #
    # "max" is the historical behaviour: 50% of that scan's single brightest
    # voxel. `peak` sits in the sharpness family at site eta^2 0.44-0.50, the most
    # scanner-explained quantity in the table, and a single-voxel max is also the
    # noisiest statistic available. A sharper camera raises the peak, raises the
    # threshold, and shrinks the measured blob — so "shape" confounds biology with
    # resolution by construction.
    #
    # The alternatives keep the adaptive logic but anchor it on something stable
    # (a robust percentile), or drop adaptivity for an absolute SBR level, which
    # is meaningful because `c` is already reference-normalised (SBR = 1 is
    # non-specific binding). DEFAULT IS "max", i.e. an exact no-op.
    _a = anchor or _ANCHOR
    if _a == "max":      thr = max(0.5 * peak, 1.0)
    elif _a == "p99":    thr = max(0.5 * p99, 1.0)
    elif _a == "p90":    thr = max(0.5 * p90, 1.0)
    elif _a == "abs15":  thr = 1.5
    elif _a == "abs20":  thr = 2.0
    elif _a == "abs25":  thr = 2.5
    elif _a == "mixed":  thr = max(0.5 * peak, 2.0)
    else: raise ValueError(f"unknown isocontour anchor {_a!r}")
    m = (c >= thr) & side
    n = int(m.sum())
    vox_ml = (VOX_MM ** 3) / 1000.0
    out["hi_vol"] = n * vox_ml
    out["spec"] = float(np.clip(v - 1.0, 0, None).sum()) * vox_ml
    if n >= 20:
        pts = np.argwhere(m).astype(np.float32) * VOX_MM
        w = c[m].astype(np.float32)
        mu = (pts * w[:, None]).sum(0) / w.sum()
        d = pts - mu
        ev = np.clip(np.linalg.eigvalsh((d * w[:, None]).T @ d / w.sum()), 1e-6, None)
        out["elong"] = float(np.sqrt(ev[2] / ev[0]))
        out["flat"] = float(np.sqrt(ev[1] / ev[0]))
        out["rms_mm"] = float(np.sqrt(ev.sum()))
        # anterior-posterior extent of surviving uptake: shrinks posteriorly first
        ys = _Y[m]
        out["ap_extent"] = float(ys.max() - ys.min())
        out["ap_centroid"] = float((ys * c[m]).sum() / c[m].sum())
        lab, ncc = ndimage.label(m)
        out["ncc"] = float(ncc)
    else:
        out.update(elong=0.0, flat=0.0, rms_mm=0.0, ap_extent=0.0,
                   ap_centroid=0.0, ncc=0.0)
    return out


def extract(crop: np.ndarray, sep_mm: float = 26.0, extra: dict | None = None) -> dict:
    f = features5.extract(crop, sep_mm=sep_mm, extra=extra)
    c = np.asarray(crop, dtype=np.float32)
    left, right = _half_masks()
    base = "max" if _ANCHOR == "both" else _ANCHOR
    sl, sr = _hemi_stats(c, left, base), _hemi_stats(c, right, base)
    if not sl or not sr:
        return f
    if _ANCHOR == "both":
        # Decompose rather than replace. The peak-anchored descriptors carry real
        # discrimination (swapping them outright costs 0.010 of CV) but the anchor
        # itself is site eta^2 0.442, against 0.080 for the p90 anchor. Emitting
        # both lets L1 keep the discriminative block and add whatever the stable
        # anchor contributes, instead of trading one for the other.
        ql, qr = _hemi_stats(c, left, "p90"), _hemi_stats(c, right, "p90")
        SHAPE = ("hi_vol", "elong", "flat", "rms_mm", "ap_extent", "ap_centroid", "ncc")
        if ql and qr:
            qw, qb = (ql, qr) if sl.get("spec", 0) <= sr.get("spec", 0) else (qr, ql)
            for k in SHAPE:
                if k not in qw:
                    continue
                f[f"q_{k}_worse"] = float(qw[k])
                f[f"q_{k}_better"] = float(qb[k])
                den = abs(qw[k]) + abs(qb[k])
                f[f"q_{k}_rel"] = float((qb[k] - qw[k]) / den) if den > 1e-9 else 0.0
    # order by total specific uptake: worse side first -> side-invariant
    worse, better = (sl, sr) if sl.get("spec", 0) <= sr.get("spec", 0) else (sr, sl)
    for k in worse:
        f[f"h_{k}_worse"] = float(worse[k])
        f[f"h_{k}_better"] = float(better[k])
        denom = abs(worse[k]) + abs(better[k])
        f[f"h_{k}_ratio"] = float(worse[k] / better[k]) if abs(better[k]) > 1e-6 else 0.0
        f[f"h_{k}_rel"] = float((better[k] - worse[k]) / denom) if denom > 1e-9 else 0.0
    return f
