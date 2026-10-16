"""
preprocessing3.py — physiologically-correct DaT-SPECT normalization + localization.

WHY THIS EXISTS (the v2 defect, measured 2026-07-24)
----------------------------------------------------
`preprocessing2.reference_level` builds its "brain" mask with
`thr = percentile(vol[vol>0], 12)`. On tightly-cropped scans that lands inside
brain tissue; on large-FOV scans (256^3, ~4% of voxels non-air) the non-zero
voxels are dominated by low-level scatter, so the mask swallows the scatter halo
and the reference collapses onto the SCATTER peak instead of the non-specific
TISSUE peak. Consequence: striatal SBR spans 3.5 .. 300 across the cohort (85x),
CV=1.28 *within the normal class alone*, and SBR — the actual clinical biomarker
for DaT-SPECT — carries AUROC 0.56, i.e. no usable signal. Every downstream SBR
feature was therefore measuring scanner FOV, not dopaminergic loss.

WHAT THIS DOES INSTEAD
----------------------
1. Head mask from an Otsu split of the LOG intensity histogram, which is cleanly
   bimodal (scatter vs. tissue) at every FOV.  Resolution-independent.
2. Reference = MODE of the log-intensity distribution of head tissue after the
   striatum is excluded. The mode tracks the dominant non-specific binding
   compartment (cortex/white matter, the occipital stand-in) and is insensitive
   to mask over-inclusion, unlike a median.
3. Striatum localization from the TWO striatal blobs (left + right) rather than a
   single brightest-blob centre of mass, anchored to the head centroid so distal
   hot spots cannot capture it.
4. Optional striatum-anchored rigid alignment: the L->R striatal axis plus the
   head's superior-inferior axis define a canonical frame, so head tilt/rotation
   is removed per-sample without any template or cross-sample statistics.

Everything is computed from a single scan's own voxels (per-sample independence
rule) using numpy/scipy/nibabel only, all pre-installed in the runtime.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import os
import numpy as np
import nibabel as nib
from scipy import ndimage

TARGET_SPACING = 2.0            # mm isotropic for the resampled crop
CROP_MM = 96.0                  # physical side of the striatal crop (mm)
OUT_VOX = int(round(CROP_MM / TARGET_SPACING))   # 48

# --- per-hemisphere sampling (v4) ------------------------------------------- #
# The 96 mm / 2 mm crop spends ~85% of its voxels on tissue that carries no
# dopaminergic signal, and the hemisphere-MIL model then throws away half of what
# is left. Sampling each striatum in its own tight box puts the whole voxel
# budget on the structure at 1.25 mm instead of 2 mm.
HEMI_SPACING = 1.25             # mm isotropic
HEMI_SHAPE = (32, 48, 32)       # voxels along (L->R, P->A, I->S) = 40 x 60 x 40 mm
HEMI_OFFSET_MM = 19.0           # |x| of each hemisphere centre in the canonical
                                # frame. Deliberately a CONSTANT, not sep/2: the
                                # detected peaks migrate medially with putaminal
                                # loss, so a sep-derived offset would make the
                                # crop geometry partly a function of the label.
LOCATE_MM = 3.0                 # coarse spacing for masking/localization
STRIATUM_SEARCH_MM = 45.0       # max distance from head centroid to a striatum
MIN_BLOB_SEP_MM = 22.0          # plausible separation of the two striatal maxima
MAX_BLOB_SEP_MM = 56.0          # (they sit ~25-50 mm apart in an adult head)
NOMINAL_SEP_MM = 38.0           # cohort median, used when only one peak is found
PAIR_SOFT_RATIO = 0.7           # v13 soft pairing (mode 2): in-plane candidate wins
                                # if >= this fraction of the unconstrained peak
PAIR_MAX_OFFPLANE_MM = 12.0     # v13 pairing gate: max A-P / S-I offset between
                                # the two striatal maxima (18 deg of head roll or
                                # yaw at a 39 mm separation; real pairs sit at
                                # 2-3 mm, RESEARCH_NOTES §73m)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_canonical(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """NIfTI -> float32 RAS array + voxel spacing (mm).
    axis0 = L->R, axis1 = P->A, axis2 = I->S."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asarray(img.dataobj, dtype=np.float32)
    data = np.squeeze(data)
    if data.ndim == 4:
        data = data[..., 0]
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float32)
    zooms = np.where(np.isfinite(zooms) & (zooms > 0), zooms, 1.0).astype(np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data, zooms


def _coarse(vol: np.ndarray, zooms: np.ndarray, spacing: float = LOCATE_MM):
    factors = np.clip((zooms / spacing).astype(np.float64), 1e-3, 1.0)
    small = ndimage.zoom(vol, factors, order=1)
    return small, (zooms / factors).astype(np.float32)


# --------------------------------------------------------------------------- #
# 1. Head mask via log-histogram Otsu
# --------------------------------------------------------------------------- #
def _otsu(values: np.ndarray, nbins: int = 256) -> float:
    """Otsu threshold on a 1-D sample."""
    lo, hi = float(values.min()), float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return lo
    hist, edges = np.histogram(values, bins=nbins, range=(lo, hi))
    hist = hist.astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    with np.errstate(invalid="ignore", divide="ignore"):
        m0 = np.cumsum(hist * centers) / np.maximum(w0, 1e-9)
        m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1e-9)
    between = w0 * w1 * (m0 - m1) ** 2
    between[~np.isfinite(between)] = -1.0
    return float(centers[int(np.argmax(between))])


MASK_MAX_ML = 7500.0            # a head mask above this is a reconstruction disc
                                # (largest legitimate head+neck FOV in the cohort
                                # is 7.15 L on the 2.3 mm site; discs are >= 8 L)


def _disc_guard(small: np.ndarray, m: np.ndarray, thr: float, vox_ml: float,
                passes: int = 3) -> np.ndarray:
    """Re-threshold inside an implausibly large mask (see head_mask). Iterates:
    a 20 L disc can need two splits (disc -> disc-with-ring -> head)."""
    for _ in range(passes):
        if m.sum() * vox_ml <= MASK_MAX_ML:
            return m
        m2, thr2 = _split_inside(small, m, thr, vox_ml)
        if m2 is None:
            return m
        m, thr = m2, thr2
    return m


def _split_inside(small, m, thr, vox_ml):
    vals = small[m]
    vals = vals[vals > 0]
    if vals.size < 64:
        return None, thr
    lv = np.log(vals + 1e-3)
    lo, hi = np.percentile(lv, [1.0, 99.5])
    lv = lv[(lv >= lo) & (lv <= hi)]
    thr2 = np.exp(_otsu(lv)) - 1e-3
    if not (thr2 > thr):
        return None, thr
    m2 = small > thr2
    m2 = ndimage.binary_closing(m2, iterations=1)
    m2 = ndimage.binary_fill_holes(m2)
    lab, n = ndimage.label(m2)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        m2 = lab == int(np.argmax(sizes))
    ml = m2.sum() * vox_ml
    if ml < 1200.0 or ml >= m.sum() * vox_ml:
        return None, thr                               # not a head either; keep
    return m2, thr2


HOLE_MULT = 1.2                 # threshold for the void measurement, as a
                                # multiple of the Otsu tissue floor. Otsu splits
                                # tissue from air, which is the wrong question for
                                # splitting CSF from tissue, so the void is read at
                                # its own level. §58b: 1.0/1.2/1.5 all pay and
                                # 0.7/0.85 do not, so this is the middle of the
                                # supported range, not a CV argmax — same argument
                                # as §39's 24 mm bound.


def head_mask(small: np.ndarray, return_holes: bool = False,
              hole_mult: float = HOLE_MULT, vox_ml: float | None = None):
    """Binary head/brain mask from a coarse volume.

    The log-intensity histogram of a SPECT volume is bimodal (scatter+air vs.
    tissue) regardless of FOV, so Otsu in log space finds the tissue floor
    robustly where a percentile of non-zero voxels does not.

    `return_holes` additionally returns the voxels that `binary_fill_holes`
    absorbs — sub-threshold regions ENCLOSED by tissue, i.e. the ventricles and
    large CSF spaces. Filling them is correct for a head mask and the volume of
    what was filled is a measurement in its own right (RESEARCH_NOTES §56): as
    one column it is worth -0.0017 CV / -0.0033 LSO on `lin`, it is nearly
    orthogonal to `head_ml` (|r| = 0.076), and it passes the §50 strata gate 9/9.
    Off by default so every existing caller is bit-identical.
    """
    pos = small[small > 0]
    if pos.size < 64:
        z = np.zeros_like(small, dtype=bool)
        return (z, z) if return_holes else z
    lv = np.log(pos + 1e-3)
    # trim the extreme tails so a few hot/cold voxels cannot drag the threshold
    lo, hi = np.percentile(lv, [1.0, 99.5])
    lv = lv[(lv >= lo) & (lv <= hi)]
    thr = np.exp(_otsu(lv)) - 1e-3
    m = small > max(thr, 0.0)
    m = ndimage.binary_closing(m, iterations=1)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        m = lab == int(np.argmax(sizes))
    degenerate = m.sum() < 32
    if degenerate:                                     # fall back
        m = small > np.percentile(pos, 60)
        m = ndimage.binary_fill_holes(m)
    if os.environ.get("DAT_MASK_V2", "1") == "1" and not degenerate:
        # v12: some reconstructions (the 3.9 mm site) carry a non-zero noise
        # floor across the WHOLE reconstruction disc, so the log-Otsu split lands
        # between air and disc rather than between scatter and tissue, and the
        # "head" becomes a 7-21 litre cylinder. A head cannot exceed ~7 L even
        # with neck in the field of view, so an implausible volume is the
        # trigger; the fix is a second log-Otsu WITHIN the disc, which recovers
        # the head (measured on all 84 masks over 6.5 L, RESEARCH_NOTES §73).
        # Triggered on plausibility, never on site, so legitimate large fields
        # of view (2.3 mm site, head+neck ~7 L) keep the shipped mask.
        # the coarse grid keeps the native spacing when that is coarser than
        # LOCATE_MM, so the caller passes the true voxel volume
        m = _disc_guard(small, m, thr,
                        LOCATE_MM ** 3 / 1000.0 if vox_ml is None else float(vox_ml))
    if not return_holes:
        return m
    if degenerate:
        return m, np.zeros_like(m)                     # no usable void measurement
    # the void is read at its own threshold, on the same coarse grid
    h = small > max(thr * hole_mult, 0.0)
    h = ndimage.binary_closing(h, iterations=1)
    hf = ndimage.binary_fill_holes(h)
    lab, n = ndimage.label(hf)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        hf = lab == int(np.argmax(sizes))
    return m, hf & ~h


# --------------------------------------------------------------------------- #
# 2. Reference (non-specific binding) level = mode of the tissue histogram
# --------------------------------------------------------------------------- #
ERODE_MM = 10.0                 # peel the scatter rim off the head mask


def inner_mask(mask: np.ndarray, sp: np.ndarray, erode_mm: float = ERODE_MM) -> np.ndarray:
    """Erode the head mask to drop the scatter rim, backing off if that would
    leave too little tissue (small heads / tight FOV)."""
    for mm in (erode_mm, erode_mm * 0.6, erode_mm * 0.3):
        er = np.maximum(np.round(mm / sp).astype(int), 1)
        st = np.ones((2 * er[0] + 1, 2 * er[1] + 1, 2 * er[2] + 1), dtype=bool)
        e = ndimage.binary_erosion(mask, structure=st)
        if e.sum() * float(np.prod(sp)) / 1000.0 >= 300.0:      # >= 300 mL left
            return e
    return mask


def reference_from_mask(small: np.ndarray, mask: np.ndarray) -> float:
    """Mode of log-intensity inside the head, striatum excluded.

    Mode (not median) because the head mask may over-include a scatter rim; a
    median shifts with that contamination while the mode stays pinned to the
    dominant non-specific tissue compartment.
    """
    vals = small[mask]
    vals = vals[vals > 0]
    if vals.size < 32:
        pos = small[small > 0]
        return float(np.median(pos)) if pos.size else 1.0
    # drop the specific-binding tail (striatum is ~1-3% of head volume)
    cut = np.percentile(vals, 92.0)
    body = vals[vals <= cut]
    if body.size < 32:
        body = vals
    lv = np.log(body + 1e-3)
    lo, hi = np.percentile(lv, [2.0, 98.0])
    if hi <= lo:
        return float(np.median(body))
    hist, edges = np.histogram(lv, bins=96, range=(lo, hi))
    # light smoothing so the mode is not a sampling artefact
    hist = ndimage.uniform_filter1d(hist.astype(np.float64), 5)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ref = float(np.exp(centers[int(np.argmax(hist))]) - 1e-3)
    if not np.isfinite(ref) or ref <= 0:
        ref = float(np.median(body))
    return max(ref, 1e-6)


# --------------------------------------------------------------------------- #
# 3. Two-blob striatal localization
# --------------------------------------------------------------------------- #
def _peak_com(vol: np.ndarray, idx: tuple, sp: np.ndarray,
              radius_mm: float = 8.0, frac: float = 0.6) -> np.ndarray:
    """Windowed intensity centre-of-mass around an integer argmax.

    Battery v2 (RESEARCH_NOTES §71) measured that the pipeline's single point
    of failure under acquisition change is this localisation: argmax+parabolic
    refinement jumps under reconstruction noise and migrates under blur, and
    every downstream stage (crop centre, canonical frame, sep-anchored ROIs,
    the CNN's input) inherits the error. A centre-of-mass over the voxels
    above `frac` of the local peak within `radius_mm` is blur-invariant for a
    symmetric kernel and averages noise over ~10^2 voxels. Enabled by
    DAT_LOC_COM=1 (read per call so A/B needs no re-import); the incumbent
    parabolic path stays the default until the retable/regate completes.
    """
    lo = [max(0, int(idx[a] - np.ceil(radius_mm / sp[a]))) for a in range(3)]
    hi = [min(vol.shape[a], int(idx[a] + np.ceil(radius_mm / sp[a])) + 1)
          for a in range(3)]
    sub = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    peak = float(vol[tuple(int(i) for i in idx)])
    w = np.clip(sub - frac * peak, 0.0, None)
    if w.sum() <= 0:
        return np.array(idx, dtype=np.float64)
    g = np.indices(sub.shape, dtype=np.float64)
    com = np.array([float((g[a] * w).sum() / w.sum()) for a in range(3)])
    return com + np.array(lo, dtype=np.float64)


def _subvoxel_peak(vol: np.ndarray, idx: tuple) -> np.ndarray:
    """Parabolic sub-voxel refinement around an integer argmax.

    The correction is clamped to +-0.5 voxel. Without the clamp a locally flat
    neighbourhood gives a near-zero denominator and the "refinement" throws the
    estimate arbitrarily far away — that is what produced a 4.6 m striatal
    separation and an all-zero crop on one training scan.
    """
    p = np.array(idx, dtype=np.float64)
    for a in range(3):
        i = int(idx[a])
        if 0 < i < vol.shape[a] - 1:
            s = [slice(int(j), int(j) + 1) for j in idx]
            s[a] = slice(i - 1, i + 2)
            f = vol[tuple(s)].ravel()
            denom = (f[0] - 2 * f[1] + f[2])
            scale = max(abs(f[1]), 1e-6)
            if abs(denom) > 1e-3 * scale:
                p[a] = i + float(np.clip(0.5 * (f[0] - f[2]) / denom, -0.5, 0.5))
    return p


def _blob_peak(sm: np.ndarray, region: np.ndarray, sp: np.ndarray,
               rel_thr: float = 0.65):
    """Selection-robust peak: rank candidate BLOBS by integrated mass, return
    the mass-weighted COM of the winner. A reconstruction-noise spike wins an
    argmax but has negligible integral next to a striatum; blur merges peaks
    but moves a blob's COM consistently. §71: the localiser's failures under
    acquisition change are wrong-basin argmax picks, which refinement cannot
    undo — selection must be robust. Env DAT_LOC_V2=1; incumbent default."""
    vals = np.where(region, sm, -np.inf)
    vmax = float(vals.max())
    if not np.isfinite(vmax) or vmax <= 0:
        return None
    m = region & (sm >= rel_thr * vmax)
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    best_mass, best = -1.0, None
    for i in range(1, n + 1):
        blob = lab == i
        mass = float((sm[blob] - rel_thr * vmax).sum())
        if mass > best_mass:
            best_mass, best = mass, blob
    w = np.where(best, sm - rel_thr * vmax, 0.0)
    g = np.indices(sm.shape, dtype=np.float64)
    tot = w.sum()
    return np.array([float((g[a] * w).sum() / tot) for a in range(3)])


def locate_striatum_consensus(small: np.ndarray, sp: np.ndarray,
                              mask: np.ndarray, agree_mm: float = 6.0):
    """Multi-smoothing consensus (§71 candidate b, env DAT_LOC_V3=1).

    The striatum persists across smoothing scales; a reconstruction-noise
    spike or a blur-created false basin does not. Run the full two-peak
    procedure at 4 mm and 7 mm: if the centres agree within `agree_mm`, keep
    the 4 mm answer (bit-identical to the incumbent on well-behaved scans —
    clean scans almost always agree, so the retable shift is confined to the
    scans that were unstable anyway); on disagreement take the 7 mm answer,
    whose heavier smoothing suppresses the spurious basin."""
    env = os.environ
    old = env.get("DAT_LOC_SMOOTH_MM")
    try:
        env["DAT_LOC_SMOOTH_MM"] = "4.0"
        c4, l4, r4 = locate_striatum(small, sp, mask)
        env["DAT_LOC_SMOOTH_MM"] = "7.0"
        c7, l7, r7 = locate_striatum(small, sp, mask)
    finally:
        if old is None:
            env.pop("DAT_LOC_SMOOTH_MM", None)
        else:
            env["DAT_LOC_SMOOTH_MM"] = old
    if float(np.linalg.norm((c4 - c7) * sp)) <= agree_mm:
        return c4, l4, r4
    return c7, l7, r7


def locate_striatum(small: np.ndarray, sp: np.ndarray, mask: np.ndarray):
    """Return (center_vox, left_peak_vox, right_peak_vox) in coarse-voxel coords.

    Finds the two striatal hot spots inside a search sphere around the head
    centroid, so a distal cortical/scalp hot spot cannot capture the estimate.
    Falls back gracefully to one blob, then to the head centroid.
    """
    if not np.any(mask):
        c = np.array(small.shape, dtype=np.float64) / 2.0
        return c, None, None
    centroid = np.array(ndimage.center_of_mass(mask.astype(np.float32)))
    _smooth_mm = float(os.environ.get("DAT_LOC_SMOOTH_MM", "4.0"))
    sm = ndimage.gaussian_filter(np.where(mask, small, 0.0),
                                 sigma=(_smooth_mm / sp))
    if os.environ.get("DAT_LOC_V4") == "1":
        # matched-filter detection (§71a): difference-of-Gaussians makes the
        # peak CONTRAST-based — a striatum-scale blob standing above its own
        # background — instead of intensity-based, which noise floors inflate.
        sm = sm - ndimage.gaussian_filter(np.where(mask, small, 0.0),
                                          sigma=(12.0 / sp))
        sm = np.clip(sm, 0.0, None)

    grid = np.indices(small.shape, dtype=np.float32)
    d2 = sum((((grid[a] - centroid[a]) * sp[a]) / STRIATUM_SEARCH_MM) ** 2 for a in range(3))
    search = mask & (d2 <= 1.0)
    if search.sum() < 16:
        search = mask
    cand = np.where(search, sm, -np.inf)

    p1_idx = np.unravel_index(int(np.argmax(cand)), cand.shape)
    _use_com = os.environ.get("DAT_LOC_COM") == "1"
    _use_v2 = os.environ.get("DAT_LOC_V2") == "1"
    p1 = None
    if _use_v2:
        p1 = _blob_peak(sm, np.isfinite(cand) & (cand > -np.inf) & search, sp)
    if p1 is None:
        p1 = _peak_com(sm, p1_idx, sp) if _use_com else _subvoxel_peak(sm, p1_idx)

    # Second peak: on the other side of the mid-sagittal plane, and at an
    # anatomically plausible distance from the first. The two striatal maxima sit
    # roughly 25-50 mm apart; anything outside that is a different structure or a
    # noise spike, and accepting it corrupts both the crop centre and the frame.
    dist_mm = np.sqrt(sum((((grid[a] - p1[a]) * sp[a]) ** 2) for a in range(3)))
    side = np.sign(centroid[0] - p1[0])
    if side == 0:
        side = 1.0
    other_side = ((grid[0] - centroid[0]) * side) > 0
    plausible = (dist_mm >= MIN_BLOB_SEP_MM) & (dist_mm <= MAX_BLOB_SEP_MM)
    _pair_mode = os.environ.get("DAT_PAIR_V2", "0")
    if _pair_mode == "1":
        # v13 (RESEARCH_NOTES §73m): the two striatal maxima are bilateral
        # copies, i.e. near-COPLANAR — on 1,355 two-peak scans the A-P and S-I
        # offsets have medians of 2.3 / 1.6 mm and 75th percentiles of 4.5 /
        # 3.0 mm. A candidate 15-30 mm off-plane is a different structure or a
        # noise blob, and accepting it shifts the crop centre and the frame.
        # The gate is a physical symmetry constraint, not a prior on the
        # separation, so `sep_mm` (a label-bearing measurement) is not biased.
        coplanar = ((np.abs(grid[1] - p1[1]) * sp[1]) <= PAIR_MAX_OFFPLANE_MM) & \
                   ((np.abs(grid[2] - p1[2]) * sp[2]) <= PAIR_MAX_OFFPLANE_MM)
        plausible = plausible & coplanar
    cand2 = np.where(search & plausible & other_side, sm, -np.inf)
    p2 = None
    if np.isfinite(cand2).any():
        p2_idx = np.unravel_index(int(np.argmax(cand2)), cand2.shape)
        if _pair_mode == "2":
            # SOFT coplanarity (§73m): a hard in-plane gate breaks yawed heads
            # whose real pair sits 20-30 mm off-plane in native coordinates.
            # Prefer the brightest IN-PLANE candidate only when it is at least
            # PAIR_SOFT_RATIO as bright as the unconstrained argmax; otherwise
            # the far candidate is a genuine striatum and is kept.
            coplanar = ((np.abs(grid[1] - p1[1]) * sp[1]) <= PAIR_MAX_OFFPLANE_MM) & \
                       ((np.abs(grid[2] - p1[2]) * sp[2]) <= PAIR_MAX_OFFPLANE_MM)
            cand2c = np.where(search & plausible & other_side & coplanar, sm, -np.inf)
            if np.isfinite(cand2c).any():
                p2c_idx = np.unravel_index(int(np.argmax(cand2c)), cand2c.shape)
                if sm[p2c_idx] >= PAIR_SOFT_RATIO * sm[p2_idx]:
                    p2_idx = p2c_idx
        # only trust it if it is a real hot spot, not background
        if sm[p2_idx] > 0.45 * sm[p1_idx]:
            cand2v = None
            if _use_v2:
                cand2v = _blob_peak(sm, search & plausible & other_side, sp)
            if cand2v is not None:
                cand = cand2v
            else:
                cand = _peak_com(sm, p2_idx, sp) if _use_com else \
                    _subvoxel_peak(sm, p2_idx)
            if MIN_BLOB_SEP_MM <= np.linalg.norm((cand - p1) * sp) <= MAX_BLOB_SEP_MM:
                p2 = cand

    if p2 is None:
        # mirror the found peak across the mid-sagittal plane as a stand-in
        p2 = p1.copy()
        p2[0] = 2 * centroid[0] - p1[0]
        if not (MIN_BLOB_SEP_MM <= np.linalg.norm((p2 - p1) * sp) <= MAX_BLOB_SEP_MM):
            # the mirror is implausible too -> trust the single peak, offset it
            # inward by half a nominal separation so the crop still straddles both
            p2 = p1.copy()
            p2[0] = p1[0] + side * (NOMINAL_SEP_MM / sp[0])
        center = 0.5 * (p1 + p2)
        return center, p1, None
    center = 0.5 * (p1 + p2)
    lo, hi = (p1, p2) if p1[0] < p2[0] else (p2, p1)
    return center, lo, hi


# --------------------------------------------------------------------------- #
# 4. Canonical frame + resampling
# --------------------------------------------------------------------------- #
FRAME_MIN_COS = float(os.environ.get("DAT_FRAME_MIN_COS", "0.9"))   # accept the PCA superior axis only within ~26 deg of native +z


def canonical_axes(mask: np.ndarray, sp: np.ndarray, left_mm, right_mm,
                   frame_v2: bool | None = None):
    """Rotation whose columns are the (L->R, P->A, I->S) unit vectors in mm space.

    The L->R axis comes from the striatal pair (a real anatomical landmark). The
    I->S axis is taken from the head's principal axis closest to +z and made
    orthogonal to L->R; A->P completes the right-handed frame.
    """
    ex = np.array([1.0, 0.0, 0.0])
    if left_mm is not None and right_mm is not None:
        v = np.asarray(right_mm) - np.asarray(left_mm)
        if np.linalg.norm(v) > 1e-6:
            ex = v / np.linalg.norm(v)
            if ex[0] < 0:
                ex = -ex
    idx = np.argwhere(mask)
    ez = np.array([0.0, 0.0, 1.0])
    if len(idx) >= 64:
        pts = idx * sp[None, :]
        pts = pts - pts.mean(0)
        # subsample for speed on big masks
        if len(pts) > 20000:
            pts = pts[:: len(pts) // 20000 + 1]
        _, _, vt = np.linalg.svd(pts, full_matrices=False)
        cand = vt[int(np.argmax(np.abs(vt[:, 2])))]
        # v14 frame guard (2026-09-08, RESEARCH_NOTES §76g): on masks that are not
        # head-shaped (a noise floor across the field of view makes the mask the
        # whole box; singular values ~1:0.96:0.96) the three principal axes are the
        # box diagonals and the one "closest to +z" is a coin toss at ~45 deg —
        # measured to flip under a 20 % gain change and to sit 42 deg from
        # superior at gain 1. A superior axis that far from the scanner's own
        # superior direction is not a head axis: fall back to native +z.
        _v2 = (os.environ.get("DAT_FRAME_V2", "0") == "1") if frame_v2 is None else bool(frame_v2)
        if _v2 and abs(cand[2]) < FRAME_MIN_COS:
            cand = np.array([0.0, 0.0, 1.0])
        if np.linalg.norm(cand) > 1e-6:
            ez = cand / np.linalg.norm(cand)
            if ez[2] < 0:
                ez = -ez
    ez = ez - np.dot(ez, ex) * ex
    if np.linalg.norm(ez) < 1e-6:
        ez = np.array([0.0, 0.0, 1.0]) - np.dot(np.array([0.0, 0.0, 1.0]), ex) * ex
    ez = ez / np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    ey = ey / (np.linalg.norm(ey) + 1e-9)
    return np.stack([ex, ey, ez], axis=1)          # (3,3), columns = frame axes


def resample_crop(vol: np.ndarray, zooms: np.ndarray, center_mm: np.ndarray,
                  R: np.ndarray | None = None, crop_mm: float = CROP_MM,
                  out_vox: int = OUT_VOX, out_spacing: float = TARGET_SPACING,
                  order: int = 1, cval: float = 0.0) -> np.ndarray:
    """Sample an out_vox^3 grid (out_spacing mm) centred at center_mm, optionally
    in the rotated canonical frame R. `cval` fills voxels outside the volume:
    0 (air) by default; v12 passes the reference level (RESEARCH_NOTES §73e),
    because on thin-slab acquisitions the field of view ends 25-45 mm below the
    striatum and what lies there is background tissue, not air."""
    half = (out_vox - 1) / 2.0
    axis = (np.arange(out_vox, dtype=np.float32) - half) * out_spacing
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    off = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=0)    # (3, M) mm
    if R is not None:
        off = R @ off
    mm = off + np.asarray(center_mm, dtype=np.float32)[:, None]
    coords = mm / np.asarray(zooms, dtype=np.float32)[:, None]
    out = ndimage.map_coordinates(vol, coords, order=order, mode="constant", cval=cval)
    return out.reshape(out_vox, out_vox, out_vox).astype(np.float32)


def resample_box(vol: np.ndarray, zooms: np.ndarray, center_mm: np.ndarray,
                 R: np.ndarray | None, shape=HEMI_SHAPE,
                 spacing: float = HEMI_SPACING, order: int = 1) -> np.ndarray:
    """Anisotropic-extent version of `resample_crop`: a `shape`-voxel box at
    `spacing` mm, centred at `center_mm` in the frame `R`."""
    axes = [(np.arange(n, dtype=np.float32) - (n - 1) / 2.0) * spacing for n in shape]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    off = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=0)
    if R is not None:
        off = R @ off
    mm = off + np.asarray(center_mm, dtype=np.float32)[:, None]
    coords = mm / np.asarray(zooms, dtype=np.float32)[:, None]
    out = ndimage.map_coordinates(vol, coords, order=order, mode="constant", cval=0.0)
    return out.reshape(shape).astype(np.float32)


def hemi_crops(vol: np.ndarray, zooms: np.ndarray, center_mm: np.ndarray,
               R: np.ndarray | None, shape=HEMI_SHAPE,
               spacing: float = HEMI_SPACING,
               offset_mm: float = HEMI_OFFSET_MM) -> np.ndarray:
    """Two tight per-hemisphere boxes, both delivered in the SAME handedness.

    index 0 — the subject's right striatum (+x of the canonical frame);
    index 1 — the left striatum, sampled through a mirrored frame so a shared
              encoder sees both sides as if they were right striata.

    Mirroring is done in the *sampling frame* rather than by flipping the array
    afterwards, so no extra interpolation is introduced.
    """
    Rm = np.eye(3) if R is None else np.asarray(R, dtype=np.float64)
    c = np.asarray(center_mm, dtype=np.float64)
    right = resample_box(vol, zooms, c + Rm @ np.array([offset_mm, 0.0, 0.0]),
                         Rm, shape, spacing)
    Rl = Rm.copy()
    Rl[:, 0] = -Rl[:, 0]            # mirror about the mid-sagittal plane
    left = resample_box(vol, zooms, c + Rm @ np.array([-offset_mm, 0.0, 0.0]),
                        Rl, shape, spacing)
    return np.stack([right, left], axis=0)


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def preprocess(path: str | Path, align: bool = True) -> dict:
    vol, zooms = load_canonical(path)
    if vol.ndim != 3 or vol.size == 0 or not np.any(vol > 0):
        return {"ok": False}
    small, sp = _coarse(vol, zooms)
    mask = head_mask(small, vox_ml=float(np.prod(sp)) / 1000.0)
    if not np.any(mask):
        return {"ok": False}

    inner = inner_mask(mask, sp)
    ref = reference_from_mask(small, inner)
    loc_dis_mm = float("nan")
    if os.environ.get("DAT_LOC_V3") == "1":
        c_vox, p_lo, p_hi = locate_striatum_consensus(small, sp, mask)
    elif os.environ.get("DAT_LOC_SENSOR") == "1":
        # sensor mode: incumbent 4 mm answer is KEPT (zero behaviour change);
        # the 7 mm re-detection only measures per-scan instability (§71b) for
        # the v12 governor. Costs one extra localisation (~0.1 s).
        c_vox, p_lo, p_hi = locate_striatum(small, sp, mask)
        _env = os.environ
        _old = _env.get("DAT_LOC_SMOOTH_MM")
        try:
            _env["DAT_LOC_SMOOTH_MM"] = "7.0"
            c7, _l7, _r7 = locate_striatum(small, sp, mask)
        finally:
            if _old is None:
                _env.pop("DAT_LOC_SMOOTH_MM", None)
            else:
                _env["DAT_LOC_SMOOTH_MM"] = _old
        loc_dis_mm = float(np.linalg.norm((c_vox - c7) * sp))
    else:
        c_vox, p_lo, p_hi = locate_striatum(small, sp, mask)
    center_mm = c_vox * sp

    left_mm = None if p_lo is None else p_lo * sp
    right_mm = None if p_hi is None else p_hi * sp
    R = canonical_axes(mask, sp, left_mm, right_mm) if align else None

    crop = resample_crop(vol, zooms, center_mm, R)

    # QC signals (cheap, useful as model features + for failure triage)
    head_ml = int(mask.sum()) * float(np.prod(sp)) / 1000.0
    sep_mm = float(np.linalg.norm(right_mm - left_mm)) if (p_lo is not None and p_hi is not None) else np.nan
    return {
        "ok": True, "zooms": zooms, "center_mm": center_mm, "crop": crop,
        "ref": float(ref), "R": R, "head_ml": head_ml,
        "inner_ml": int(inner.sum()) * float(np.prod(sp)) / 1000.0,
        "two_blob": bool(p_lo is not None and p_hi is not None),
        "blob_sep_mm": sep_mm, "sep_mm": float(sep_mm) if np.isfinite(sep_mm) else 26.0,
        "native_spacing": float(np.mean(zooms)), "native_vox": int(np.prod(vol.shape)),
    }


def geom_key(spacing: float, shape) -> str:
    """Canonical name for one hemisphere sampling geometry.

    Several members may want different boxes (the v4 stack ships a 1.9 mm and a
    1.25 mm family), so the crops have to be keyed by geometry rather than by
    position — silently feeding a model the wrong box would look like a mediocre
    member, not like a bug.
    """
    return f"{float(spacing):g}:" + "x".join(str(int(v)) for v in shape)


def compose_refinement(R: np.ndarray, center_mm: np.ndarray, rot: np.ndarray,
                       shift: np.ndarray, crop_mm: float = CROP_MM):
    """Fold a template-registration pose (register3 convention: rotation vector
    `rot` in ARRAY axes, `shift` in affine_grid units of the crop, i.e. ±1 =
    ±crop_mm/2) into the canonical frame, so the registered crop can be taken
    from the NATIVE volume in one interpolation (v13, RESEARCH_NOTES §73p).

    warp() makes out(g) = canon(Rr g + t) with canon(u) = native(c + R u), so
    out(g) = native((c + R t_mm) + (R Rr) g)."""
    from scipy.spatial.transform import Rotation
    Rr = Rotation.from_rotvec(np.asarray(rot, float)).as_matrix()
    t_mm = np.asarray(shift, float) * (crop_mm / 2.0)
    R = np.asarray(R, float)
    return R @ Rr, np.asarray(center_mm, float) + R @ t_mm


def preprocess_full(path: str | Path, want_hemi: bool = False,
                    hemi_spacing: float = HEMI_SPACING,
                    hemi_shape=HEMI_SHAPE,
                    hemi_geoms=None,
                    crop_mm: float = CROP_MM, out_vox: int = OUT_VOX,
                    out_spacing: float = TARGET_SPACING,
                    refine=None, want_f2: bool = False) -> dict:
    """Everything a model needs from one scan, in one pass.

    Used verbatim by BOTH the training cache (cache3.py) and the submission
    (main.py), so there is no way for the two to drift apart.

    `want_hemi` adds the two 1.25 mm per-hemisphere boxes. It is off by default so
    the v3 path costs exactly what it always did; `main.py` turns it on only when
    the shipped stack actually contains a hemisphere member.
    """
    vol, zooms = load_canonical(path)
    if vol.ndim != 3 or vol.size == 0 or not np.any(vol > 0):
        raise ValueError("empty volume")
    small, sp = _coarse(vol, zooms)
    mask, holes = head_mask(small, return_holes=True,
                            vox_ml=float(np.prod(sp)) / 1000.0)
    if not mask.any():
        raise ValueError("no head mask")
    inner = inner_mask(mask, sp)
    ref = max(reference_from_mask(small, inner), 1e-6)
    loc_dis_mm = float("nan")
    if os.environ.get("DAT_LOC_V3") == "1":
        c_vox, p_lo, p_hi = locate_striatum_consensus(small, sp, mask)
    elif os.environ.get("DAT_LOC_SENSOR") == "1":
        # sensor mode: incumbent 4 mm answer is KEPT (zero behaviour change);
        # the 7 mm re-detection only measures per-scan instability (§71b) for
        # the v12 governor. Costs one extra localisation (~0.1 s).
        c_vox, p_lo, p_hi = locate_striatum(small, sp, mask)
        _env = os.environ
        _old = _env.get("DAT_LOC_SMOOTH_MM")
        try:
            _env["DAT_LOC_SMOOTH_MM"] = "7.0"
            c7, _l7, _r7 = locate_striatum(small, sp, mask)
        finally:
            if _old is None:
                _env.pop("DAT_LOC_SMOOTH_MM", None)
            else:
                _env["DAT_LOC_SMOOTH_MM"] = _old
        loc_dis_mm = float(np.linalg.norm((c_vox - c7) * sp))
    else:
        c_vox, p_lo, p_hi = locate_striatum(small, sp, mask)
    center_mm = c_vox * sp
    lmm = None if p_lo is None else p_lo * sp
    rmm = None if p_hi is None else p_hi * sp
    R = canonical_axes(mask, sp, lmm, rmm)
    # v14 (§76g): the guarded-axis frame, served ONLY to members trained on it
    # (key `aligned_f2`); everything else keeps the shipped frame.
    R_f2 = canonical_axes(mask, sp, lmm, rmm, frame_v2=True) if want_f2 else None
    sep = float(np.linalg.norm(rmm - lmm)) if (lmm is not None and rmm is not None) else 26.0
    # v12 (§73e): out-of-volume voxels are filled with the reference level, i.e.
    # SBR 1.0, instead of 0. Env-gated until the gate passes; the training cache
    # and main.py share this code path so they cannot disagree.
    pad = float(ref) if os.environ.get("DAT_PAD_REF") == "1" else 0.0
    aligned = resample_crop(vol, zooms, center_mm, R, crop_mm, out_vox, out_spacing, cval=pad) / ref
    raw = resample_crop(vol, zooms, center_mm, None, crop_mm, out_vox, out_spacing, cval=pad) / ref
    meta = dict(head_ml=float(mask.sum() * np.prod(sp) / 1000.0),
                inner_ml=float(inner.sum() * np.prod(sp) / 1000.0),
                hole_ml=float(holes.sum() * np.prod(sp) / 1000.0),
                two_blob=float(p_lo is not None and p_hi is not None),
                native_spacing=float(np.mean(zooms)), ref=float(ref),
                loc_dis_mm=loc_dis_mm)
    out = dict(aligned=aligned, raw=raw, ref=float(ref), sep=sep, meta=meta)
    if R_f2 is not None:
        out["aligned_f2"] = resample_crop(vol, zooms, center_mm, R_f2, crop_mm, out_vox,
                                          out_spacing, cval=pad) / ref
    if refine is not None:
        # v13: `refine(aligned)` -> (rot, shift) from the template fit; the
        # registered crop is then resampled ONCE from the native volume with
        # the composed pose (no second interpolation of the 2 mm crop).
        rot, shift = refine(aligned)
        R2, c2 = compose_refinement(R, center_mm, rot, shift, crop_mm)
        out["aligned_reg1"] = resample_crop(vol, zooms, c2, R2, crop_mm, out_vox,
                                            out_spacing, cval=pad) / ref
        out["R_reg1"], out["center_reg1"] = R2, c2
    # `hemi_raw` shares only the localisation with `hemi` — it drops the canonical
    # rotation entirely — so a model trained on it is decorrelated from the rest
    # of the stack by construction rather than by a new seed.
    if hemi_geoms:
        # several boxes at once -> dicts keyed by geom_key, because the shipped
        # stack mixes a 1.9 mm and a 1.25 mm family and handing a model the wrong
        # box would look like a mediocre member rather than like a bug
        out["hemi"], out["hemi_raw"] = {}, {}
        for sp_, shape_ in hemi_geoms:
            k = geom_key(sp_, shape_)
            hs = tuple(int(v) for v in shape_)
            out["hemi"][k] = hemi_crops(vol, zooms, center_mm, R, hs, float(sp_)) / ref
            out["hemi_raw"][k] = hemi_crops(vol, zooms, center_mm, None, hs, float(sp_)) / ref
    elif want_hemi:
        hs = tuple(hemi_shape)
        out["hemi"] = hemi_crops(vol, zooms, center_mm, R, hs, hemi_spacing) / ref
        out["hemi_raw"] = hemi_crops(vol, zooms, center_mm, None, hs, hemi_spacing) / ref
    return out


if __name__ == "__main__":
    import sys, time
    for p in sys.argv[1:]:
        t0 = time.time()
        d = preprocess(p)
        if d["ok"]:
            sbr = d["crop"] / d["ref"]
            print(Path(p).stem, f"{time.time()-t0:.2f}s ref={d['ref']:.2f} "
                  f"sbrmax={sbr.max():.2f} two_blob={d['two_blob']} sep={d['blob_sep_mm']:.1f}mm "
                  f"head={d['head_ml']:.0f}mL")
        else:
            print(Path(p).stem, "FAILED")
