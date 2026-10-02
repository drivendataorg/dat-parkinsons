"""
roigeom.py — striatal ROI placement geometry, defined in ONE place.

Every anatomical ROI in `features4` and every grid cell in `features5` is placed
at an offset from `half_sep`, the detected half-separation of the two striata,
which is clipped to a plausible range before use. The training tables and the
packaged inference path must clip identically. HANDOFF §0.00 records what happens
otherwise: the first v6 build shipped `max_deg = 30` in `inference_config.json`
while the training tables used 90, and gate 5 returned the *same* number before
and after the fix, because 20 smoke scans cannot see a rule that changes 6% of
them. So the bounds live here and are imported, never re-typed.

THE UPPER BOUND. v6 and earlier used 20 mm. Measured 2026-07-26: detected
separation has median 39.0 mm and p99 53.3 mm, so **581 of 1362 training scans
(42.7%) sat exactly at that bound** and none reached the floor. For 43% of the
cohort the ROI geometry was therefore frozen while the anatomy kept moving
outward — a discontinuity in the *definition* of 166 of the 235 shipped features
(all 94 `features4` columns and all 72 `features5` columns), which is all of
`lin4` and most of `lin`, jointly 0.738 of the stacker's coefficient mass.
Anatomically the DaT peak sits ~25-28 mm from midline; at the bound the `put` ROI
(placed at h+3) lands at 23 mm, systematically medial for nearly half the cohort.

`sep_mm` is a BIOLOGY variable, not a scanner variable — site eta^2 = 0.030 among
normals against a table median of 0.156, and correlation with `head_ml` of 0.103.
It carries label AUROC 0.6625 (inverted): striatal extent shrinks with pathology.
That is what makes widening the bound legitimate rather than an instance of the
7-DOF failure mode (RESEARCH_NOTES §37), and it is the opposite of the reason the
change was originally proposed. See SWEEP_2026-07-26.md.

`DAT_HALF_SEP_MAX` exists so the bound can be swept without editing source; it is
read once at import so joblib workers inherit it. The shipped value is the
default below, and nothing in the submission package sets the variable.
"""
from __future__ import annotations

import os

import numpy as np

HALF_SEP_MIN = 9.0
# v7 = 24.0; v6 and earlier = 20.0. Measured end-to-end 2026-07-26, `lin` /
# `lin4` random-5-fold CV and leave-pseudo-site-out:
#     bound   lin CV    lin LSO   lin4 CV   lin4 LSO
#      20     0.2430    0.2620    0.2566    0.2798     <- v6
#      22     0.2351    0.2594    0.2518    0.2778
#      24     0.2340    0.2594    0.2490    0.2766     <- shipped
#      28     0.2346    0.2602    0.2494    0.2770
#     none    0.2346    0.2602    0.2494    0.2770
# 28 and unbounded are identical because max(sep)/2 = 27.99 mm on this cohort.
# The live question was "20 or not 20": every bound >= 22 beats 20 by ~0.008 of
# CV, and 22/24/28/none span 0.0008, well inside the 0.005 noise floor. So 24 is
# NOT a CV-tuned constant — it is the middle of the anatomically supported range
# (the DaT peak sits ~25-28 mm from midline, and `put` is placed at h+3, i.e.
# 27 mm). A finite bound is also kept deliberately: `preprocessing3` accepts a
# striatal pair out to 56 mm, and an unbounded rule would let a mislocalised scan
# place ROIs outside the 96 mm crop.
HALF_SEP_MAX = float(os.environ.get("DAT_HALF_SEP_MAX", "24.0"))


# --- ROI SIZE SCALING (v8 experiment; ALPHA = 0 is exactly v7) ---------------
# v7 made ROI *placement* follow the detected separation. The radii are still
# fixed constants chosen for an average head. If striatal size scales with
# striatal separation, a fixed radius under-samples a big striatum and
# over-samples a small one — the same defect v7 fixed, one axis over.
#
#     radius *= (clip(h) / HALF_SEP_REF) ** ROI_SCALE_ALPHA
#
# HALF_SEP_REF is the cohort median half-separation (sep median 39.01 mm), so
# alpha has no effect on a median-sized head at any value.
#
# MEASURED 2026-07-26 AND DEAD — ALPHA STAYS 0. Do not re-run this sweep.
#     alpha   lin CV   lin LSO   lin4 CV   lin4 LSO
#      0.00   0.2340   0.2594    0.2490    0.2766     <- v7, shipped
#      0.25   0.2340   0.2598    0.2484    0.2784
#      0.40   0.2338   0.2592    0.2501    0.2790
#      0.60   0.2345   0.2598    0.2523    0.2808
#      1.00   0.2359   0.2614    0.2582    0.2849
# Best case is -0.0002 on `lin`, far inside the 0.005 noise floor, while `lin4`
# degrades monotonically from alpha >= 0.4 on both surfaces. The 4-second
# pre-check called it: within pseudo-site among normals, partial R between
# `sep_mm` and posterior-putamen SBR is 0.148, under the 0.15 stop line. **That
# pre-check is now a validated screen for this family — run it before building
# anything.** Placement scales with separation; size does not.
#
# The code is kept, at ALPHA = 0 (an exact no-op), so the negative result lives
# where someone would otherwise re-implement it.
#
# The original warning, still the right instinct: `sep_mm` is
# label-correlated (AUROC 0.6625, inverted — striatal extent shrinks with
# pathology), so scaling ROI *size* by it hands the geometry a degree of freedom
# that can chase the pathology. That is RESEARCH_NOTES §37's 7-DOF failure mode,
# where a better fit scored much worse. Placement is safe because the ROI still
# has to land on real signal; size is not, because a smaller ROI on a degenerate
# striatum can manufacture contrast. **Reject any alpha that improves random-split
# CV while leave-site-out regresses**, however large the CV gain. That
# discriminator is what validated v7.
HALF_SEP_REF = 19.5
ROI_SCALE_ALPHA = float(os.environ.get("DAT_ROI_SCALE_ALPHA", "0.0"))


def clip_half_sep(half_sep: float) -> float:
    """Clip a detected half-separation (mm) to the plausible anatomical range."""
    return float(np.clip(float(half_sep), HALF_SEP_MIN, HALF_SEP_MAX))


def roi_scale(half_sep: float) -> float:
    """Multiplicative factor applied to ROI radii. Exactly 1.0 when ALPHA = 0."""
    if ROI_SCALE_ALPHA == 0.0:
        return 1.0
    return float((clip_half_sep(half_sep) / HALF_SEP_REF) ** ROI_SCALE_ALPHA)


def roi_key(half_sep: float) -> float:
    """Cache key for ROI/grid masks: the clipped value at 0.1 mm resolution."""
    return round(clip_half_sep(half_sep), 1)
