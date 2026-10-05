"""
features13.py — features6 plus RADIAL SPECIFIC-BINDING SHELLS.

WHY (RESEARCH_NOTES §29–§30)
---------------------------
`is_pathologic` in this competition means *abnormal DaT scan* — PD, MSA, PSP,
DLB — not Parkinson's disease. Measured on this cohort, relative tracer loss is
flat across sub-regions (posterior putamen 27.5%, putamen 24.9%, caudate 25.4%)
and the putamen:caudate ratio carries AUROC 0.511. So the rostro-caudal gradient
prior that v2, v3 and every v4 attempt built on is the wrong prior for this label:
what separates the classes is how much specific binding there is and how it is
distributed, not which sub-region failed.

This block measures exactly that, and nothing else. For each hemisphere: find the
centroid of specific binding (SBR − 1), then take the mean SBR in concentric
shells at 0–3, 3–6, 6–9, 9–12, 12–16, 16–20 mm. That is a radial profile of the
whole structure, agnostic to which part of it degenerated. The two hemispheres are
ordered worse-side-first, as everywhere else in this table.

`s_r2_worse` alone reaches **AUROC 0.840**, beating the previous best single
feature (posterior-putamen SBR, 0.827).

WHAT WAS DELIBERATELY LEFT OUT
------------------------------
The same experiment produced normalised profile shape, edge width, r80/r20 and
compactness descriptors. All of them measure *sharpness*, all of them are
therefore partly measuring the scanner's reconstruction resolution rather than the
patient, and adding them makes leave-site-out worse (0.2718 vs 0.2692) for no
random-split gain. That is the third time a sharpness-flavoured feature has failed
the site gate — see §15 (PSF keyed on voxel spacing) and §30. Only the magnitude
shells ship.
"""
from __future__ import annotations

import numpy as np

import features6
from features4 import _X, _Y, _Z

SHELLS = (0.0, 3.0, 6.0, 9.0, 12.0, 16.0, 20.0)
_HALF = None


def _halves():
    global _HALF
    if _HALF is None:
        _HALF = (_X < 0, _X > 0)
    return _HALF


def _shells(c: np.ndarray, side: np.ndarray) -> dict:
    """Mean SBR in concentric shells around this hemisphere's binding centroid.

    Always returns every key. A hemisphere with no specific binding at all has no
    binding centroid, so it falls back to the hemisphere's geometric centre — the
    shells are then simply measuring a dead hemisphere, which is exactly what the
    model should see. Returning a partial dict instead would let the names arrive
    missing, be median-imputed, and quietly turn a maximally abnormal scan into an
    average one. `main.py`'s feature-completeness guard caught precisely this.
    """
    spec = np.clip(c - 1.0, 0.0, None)
    w = spec * side
    tot = float(w.sum())
    if tot < 1e-6:
        w = side.astype(np.float32)
        tot = float(w.sum())
        if tot < 1e-6:
            return {f"r{i}": 0.0 for i in range(len(SHELLS) - 1)}
    cx = float((_X * w).sum() / tot)
    cy = float((_Y * w).sum() / tot)
    cz = float((_Z * w).sum() / tot)
    r = np.sqrt((_X - cx) ** 2 + (_Y - cy) ** 2 + (_Z - cz) ** 2)
    out = {}
    for i, (a, b) in enumerate(zip(SHELLS[:-1], SHELLS[1:])):
        m = side & (r >= a) & (r < b)
        out[f"r{i}"] = float(c[m].mean()) if m.sum() >= 8 else 0.0
    return out


def extract(crop: np.ndarray, sep_mm: float = 26.0, extra: dict | None = None) -> dict:
    f = features6.extract(crop, sep_mm=sep_mm, extra=extra)
    c = np.asarray(crop, dtype=np.float32)
    left, right = _halves()
    a, b = _shells(c, left), _shells(c, right)
    if not a or not b:                       # cannot happen now; belt and braces
        z = {f"r{i}": 0.0 for i in range(len(SHELLS) - 1)}
        a, b = a or z, b or z
    sl = float(np.clip(c[left] - 1.0, 0, None).sum())
    sr = float(np.clip(c[right] - 1.0, 0, None).sum())
    worse, better = (a, b) if sl <= sr else (b, a)
    for k in worse:
        f[f"s_{k}_worse"] = float(worse[k])
        f[f"s_{k}_better"] = float(better[k])
        den = abs(worse[k]) + abs(better[k])
        f[f"s_{k}_rel"] = float((better[k] - worse[k]) / den) if den > 1e-9 else 0.0
    return f
