"""
features5.py — features4 plus a spatial specific-binding profile.

Rationale: nigrostriatal degeneration has a stereotyped spatial order — dorsal
posterior putamen first, then anterior putamen, caudate last — and it starts
asymmetrically. A handful of ellipsoid ROIs samples that gradient coarsely. This
module adds a coarse 3-D grid over each striatum, with the two hemispheres
ordered "more affected first" so the descriptor is side-invariant, which lets a
linear model read the gradient directly instead of inferring it.

Grid cells are means of a volume that has already been reference-normalised and
resampled into the canonical striatal frame, so cell k means the same anatomy in
every scan.
"""
from __future__ import annotations

import numpy as np

import features4
from features4 import VOX_MM, _X, _Y, _Z, extract as extract4
from roigeom import clip_half_sep, roi_key

# grid extent around each striatum centre (mm), and cell counts
GRID_X = (-11.0, 11.0, 3)      # medial <-> lateral
GRID_Y = (-16.0, 16.0, 4)      # posterior <-> anterior
GRID_Z = (-9.0, 9.0, 2)        # inferior <-> superior


def _edges(spec):
    lo, hi, n = spec
    return np.linspace(lo, hi, n + 1)


def _grid_masks(half_sep: float, side_sign: float):
    h = clip_half_sep(half_sep) * side_sign
    ex, ey, ez = _edges(GRID_X), _edges(GRID_Y), _edges(GRID_Z)
    masks = []
    for i in range(len(ex) - 1):
        for j in range(len(ey) - 1):
            for k in range(len(ez) - 1):
                m = ((_X - h) >= ex[i]) & ((_X - h) < ex[i + 1]) & \
                    (_Y >= ey[j]) & (_Y < ey[j + 1]) & \
                    (_Z >= ez[k]) & (_Z < ez[k + 1])
                masks.append(m)
    return masks


_GRID_CACHE: dict = {}


def grid_masks(half_sep: float):
    key = roi_key(half_sep)
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = (_grid_masks(key, -1.0), _grid_masks(key, +1.0))
    return _GRID_CACHE[key]


def extract(crop: np.ndarray, sep_mm: float = 26.0, extra: dict | None = None) -> dict:
    f = extract4(crop, sep_mm=sep_mm, extra=extra)
    c = np.asarray(crop, dtype=np.float32)
    ml, mr = grid_masks(sep_mm / 2.0)
    gl = np.array([c[m].mean() if m.any() else 0.0 for m in ml], dtype=np.float32)
    gr = np.array([c[m].mean() if m.any() else 0.0 for m in mr], dtype=np.float32)
    # order hemispheres by overall uptake: "worse" side first -> side-invariant
    if gl.sum() <= gr.sum():
        worse, better = gl, gr
    else:
        worse, better = gr, gl
    tot = float(max(worse.sum() + better.sum(), 1e-6))
    for i, (a, b) in enumerate(zip(worse, better)):
        f[f"g{i:02d}_worse"] = float(a)
        f[f"g{i:02d}_better"] = float(b)
        f[f"g{i:02d}_rel"] = float((a + b) / tot)     # scale-free shape of the profile
    f["grid_worse_sum"] = float(worse.sum())
    f["grid_better_sum"] = float(better.sum())
    f["grid_asym"] = float((better.sum() - worse.sum()) / max(better.sum() + worse.sum(), 1e-6))
    return f
