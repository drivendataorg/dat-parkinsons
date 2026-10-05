"""
features4.py — clinical DaT features computed in the canonical striatal frame.

Depends on preprocessing3: the input is a 48^3 / 2 mm SBR crop that has been
resampled in a frame where +x is the L->R striatal axis, +y is anterior and
+z is superior, centred on the midpoint of the two striata. That makes fixed
anatomical ROIs meaningful without any template registration, so the classic
clinical measurements (putamen and caudate specific binding, posterior-putamen
loss, left/right asymmetry) can be read off directly.

With the v2 pipeline these measurements were noise, because its reference level
tracked scanner FOV rather than non-specific binding (see preprocessing3 module
docstring). With the corrected reference, posterior-putamen SBR alone separates
the classes at AUROC ~0.84.

Pure numpy/scipy; identical code path at train and inference time.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from roigeom import clip_half_sep, roi_key, roi_scale

VOX_MM = 2.0
N = 48
_AX = (np.arange(N, dtype=np.float32) - (N - 1) / 2.0) * VOX_MM
_X, _Y, _Z = np.meshgrid(_AX, _AX, _AX, indexing="ij")


def _ellipsoid(cx, cy, cz, rx, ry, rz):
    return (((_X - cx) / rx) ** 2 + ((_Y - cy) / ry) ** 2 + ((_Z - cz) / rz) ** 2) <= 1.0


def _rois(half_sep: float) -> dict:
    """Anatomical ROIs, mirrored about x, placed relative to the detected
    striatal half-separation (adapts to head size)."""
    h = clip_half_sep(half_sep)
    # k scales the striatal ROI radii with the detected separation; k == 1.0
    # exactly when roigeom.ROI_SCALE_ALPHA == 0, which is the v7 geometry. The
    # background box below is deliberately NOT scaled — it is defined in absolute
    # crop coordinates and has nothing to do with striatal size.
    k = roi_scale(half_sep)
    out = {}
    for side, s in (("L", -1.0), ("R", +1.0)):
        out[f"stri_{side}"] = _ellipsoid(s * h, 0.0, 0.0, 9.0 * k, 11.0 * k, 7.0 * k)
        out[f"cau_{side}"] = _ellipsoid(s * (h - 3.0), 8.0, 2.0, 6.0 * k, 7.0 * k, 6.0 * k)
        out[f"put_{side}"] = _ellipsoid(s * (h + 3.0), -6.0, 0.0, 6.0 * k, 8.0 * k, 6.0 * k)
        out[f"aput_{side}"] = _ellipsoid(s * (h + 2.0), 0.0, 0.0, 5.0 * k, 6.0 * k, 5.0 * k)
        out[f"ppu_{side}"] = _ellipsoid(s * (h + 3.0), -12.0, 0.0, 5.0 * k, 6.0 * k, 5.0 * k)
    # occipital-ish background inside the crop, well behind the striata
    out["bg"] = (_Y < -34.0) & (np.abs(_X) < 26.0) & (np.abs(_Z) < 16.0)
    return out


_ROI_CACHE: dict[float, dict] = {}


def rois_for(half_sep: float) -> dict:
    key = roi_key(half_sep)
    if key not in _ROI_CACHE:
        _ROI_CACHE[key] = _rois(key)
    return _ROI_CACHE[key]


def _safe(a: np.ndarray, fn, default=0.0):
    if a.size == 0:
        return float(default)
    v = float(fn(a))
    return v if np.isfinite(v) else float(default)


def extract(crop: np.ndarray, sep_mm: float = 26.0, extra: dict | None = None) -> dict:
    """crop: 48^3 SBR volume in the canonical frame. Returns a flat feature dict."""
    c = np.asarray(crop, dtype=np.float32)
    f: dict[str, float] = {}
    R = rois_for(sep_mm / 2.0)

    # ---- ROI specific-binding measurements ---------------------------------- #
    for base in ("stri", "cau", "put", "aput", "ppu"):
        vals = {}
        for side in ("L", "R"):
            v = c[R[f"{base}_{side}"]]
            vals[side] = v
            f[f"{base}_{side}_mean"] = _safe(v, np.mean)
            f[f"{base}_{side}_p90"] = _safe(v, lambda a: np.percentile(a, 90))
        lo = min(f[f"{base}_L_mean"], f[f"{base}_R_mean"])
        hi = max(f[f"{base}_L_mean"], f[f"{base}_R_mean"])
        f[f"{base}_min"] = lo
        f[f"{base}_max"] = hi
        f[f"{base}_mean"] = 0.5 * (lo + hi)
        f[f"{base}_asym"] = (hi - lo) / max(0.5 * (hi + lo), 1e-6)
        # specific binding = SBR - 1 (non-specific background removed)
        f[f"{base}_spec_min"] = max(lo - 1.0, 0.0)
        f[f"{base}_spec_mean"] = max(0.5 * (lo + hi) - 1.0, 0.0)

    bg = c[R["bg"]]
    f["bg_mean"] = _safe(bg, np.mean, 1.0)
    f["bg_p90"] = _safe(bg, lambda a: np.percentile(a, 90), 1.0)

    # ---- clinical ratios ----------------------------------------------------- #
    eps = 1e-6
    for side in ("L", "R"):
        f[f"pc_ratio_{side}"] = f[f"put_{side}_mean"] / max(f[f"cau_{side}_mean"], eps)
        f[f"pa_ratio_{side}"] = f[f"ppu_{side}_mean"] / max(f[f"aput_{side}_mean"], eps)
    f["pc_ratio_min"] = min(f["pc_ratio_L"], f["pc_ratio_R"])
    f["pa_ratio_min"] = min(f["pa_ratio_L"], f["pa_ratio_R"])
    f["put_over_bg"] = f["put_min"] / max(f["bg_mean"], eps)
    f["stri_over_bg"] = f["stri_mean"] / max(f["bg_mean"], eps)
    # most-affected / least-affected side contrast (PD starts asymmetric)
    f["worst_ppu_over_best_cau"] = f["ppu_min"] / max(f["cau_max"], eps)

    # ---- global uptake distribution ------------------------------------------ #
    f["peak"] = float(c.max())
    for p in (99.9, 99.5, 99.0, 97.0, 90.0, 75.0, 50.0):
        f[f"p{p}"] = float(np.percentile(c, p))
    f["mean"] = float(c.mean())
    f["std"] = float(c.std())

    spec = np.clip(c - 1.0, 0.0, None)
    vox_ml = (VOX_MM ** 3) / 1000.0
    f["spec_total"] = float(spec.sum()) * vox_ml
    for thr in (1.5, 2.0, 3.0, 4.0, 5.0):
        f[f"vol_gt{thr}"] = float((c >= thr).sum()) * vox_ml
    peak = max(f["peak"], eps)
    for frac in (0.4, 0.5, 0.6, 0.75):
        f[f"volfrac_gt{frac}"] = float((c >= frac * peak).sum()) * vox_ml

    # ---- shape of the specific-uptake blob(s) --------------------------------- #
    hi_mask = c >= max(0.5 * peak, 1.0)
    nvox = int(hi_mask.sum())
    f["hi_vox"] = float(nvox) * vox_ml
    if nvox >= 20:
        pts = np.argwhere(hi_mask).astype(np.float32) * VOX_MM
        w = c[hi_mask].astype(np.float32)
        mu = (pts * w[:, None]).sum(0) / w.sum()
        d = pts - mu
        cov = (d * w[:, None]).T @ d / w.sum()
        ev = np.linalg.eigvalsh(cov)
        ev = np.clip(ev, 1e-6, None)
        f["blob_elong"] = float(np.sqrt(ev[2] / ev[0]))
        f["blob_flat"] = float(np.sqrt(ev[1] / ev[0]))
        f["blob_rms_mm"] = float(np.sqrt(ev.sum()))
        lab, ncc = ndimage.label(hi_mask)
        f["blob_n"] = float(ncc)
        if ncc > 0:
            sz = np.bincount(lab.ravel())[1:]
            f["blob_frac_largest"] = float(sz.max()) / float(nvox)
        else:
            f["blob_frac_largest"] = 0.0
    else:
        f["blob_elong"] = f["blob_flat"] = f["blob_rms_mm"] = 0.0
        f["blob_n"] = 0.0
        f["blob_frac_largest"] = 0.0

    # ---- boundary sharpness (partial-volume / resolution proxy) --------------- #
    gx, gy, gz = np.gradient(ndimage.gaussian_filter(c, 1.0), VOX_MM)
    gm = np.sqrt(gx * gx + gy * gy + gz * gz)
    band = hi_mask ^ ndimage.binary_erosion(hi_mask) if nvox >= 20 else np.zeros_like(hi_mask)
    f["grad_edge"] = _safe(gm[band], np.mean)
    f["grad_p99"] = float(np.percentile(gm, 99))
    f["grad_edge_norm"] = f["grad_edge"] / max(peak, eps)

    # ---- left/right symmetry of the whole crop -------------------------------- #
    flip = c[::-1]
    denom = float(np.abs(c).sum() + np.abs(flip).sum()) + eps
    f["lr_l1_asym"] = float(np.abs(c - flip).sum()) / denom
    top = c >= max(0.6 * peak, 1.0)
    f["lr_vol_asym"] = abs(float(top[:24].sum()) - float(top[24:].sum())) / max(float(top.sum()), 1.0)

    # ---- geometry / QC passed through ----------------------------------------- #
    f["sep_mm"] = float(sep_mm)
    if extra:
        # This allowlist is itself an exclusion rule, and §50 cost the project two
        # versions by leaving `head_ml` out of the model. `hole_ml` (§56) is the
        # void volume `head_mask` fills in and discards; it reaches a model only
        # if it is named here AND kept out of train_final3.SCANNER.
        for k in ("head_ml", "inner_ml", "hole_ml", "two_blob",
                  "native_spacing", "ref"):
            if k in extra and extra[k] is not None and np.isfinite(float(extra[k])):
                f[k] = float(extra[k])
    return f


FEATURE_ORDER: list[str] | None = None


def to_vector(f: dict, order: list[str]) -> np.ndarray:
    return np.array([f.get(k, np.nan) for k in order], dtype=np.float32)
