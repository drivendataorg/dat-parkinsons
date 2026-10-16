#!/usr/bin/env python3
"""Regenerate the write-up figures for the NavAttack v17 solution.

Every number drawn here is an aggregate: a platform score, a cohort-level
out-of-fold (OOF) statistic, a shipped constant, or a coefficient of the shipped
linear model. No per-scan value is read or stored. Each hard-coded number carries
a comment naming its source:

  development notes              the author's development notes (not distributed)
  inference/...                  the shipped v17 inference code in this repo
  assets: <file> <key>           a model file installed into inference/assets/
                                 by scripts/install_weights.py

Outputs (next to this script unless --out is given):
  score_progression.png   public and private log loss per scored submission
  family_loss.png         OOF performance per inferred acquisition centre
  lin_coefficients.png    15 largest standardised coefficients of `lin`
  pipeline_diagram.png    v17 inference pipeline (.svg as well)
  drift.png               public-to-private drift per scored submission

Usage, from the repository root:
  python docs/figures/make_figures.py --assets inference/assets
  python docs/figures/make_figures.py --lin PATH/linear_lin.json --lin4 PATH/linear_lin4.json
  python docs/figures/make_figures.py --print-tables      # markdown tables for the README

Without the two linear-model JSON files the script skips lin_coefficients.png
and draws the rest. Only the `coef` and `features` keys of those files are read.
Tested with Python 3.12 and matplotlib 3.11.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# --------------------------------------------------------------------------- #
# Design tokens: the light-mode reference palette of the dataviz method.
# Series pairs used below were run through its validator (light surface
# #fcfcfb): blue/orange and blue/red both pass every check.
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"      # categorical slot 1, diverging cool pole
ORANGE = "#eb6834"    # categorical slot 2
AQUA = "#1baf7a"      # categorical slot 3
VIOLET = "#4a3aa7"    # categorical slot 7
RED = "#e34948"       # diverging warm pole
DEEMPH = "#c3c2b7"    # de-emphasised marks
DPI = 200

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.monospace": ["Menlo", "DejaVu Sans Mono"],
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "svg.fonttype": "none",
})

# --------------------------------------------------------------------------- #
# DATA 1: platform scores of every scored submission, in submission-id order.
# Source: the author's development notes (not distributed), submission table.
# The change text condenses the development notes for each version.
# (version, submission id, public log loss, private log loss, drift, change)
# --------------------------------------------------------------------------- #
SCORES = [
    ("v2", 317177, 0.3364, 0.3612, 0.0248,
     "3D CNN ensemble, reference level tracked the field of view"),
    ("v3", 317209, 0.2599, 0.2992, 0.0393,
     "Reference-level fix, hemisphere-MIL 3D CNN"),
    ("v9", 317758, 0.2664, 0.3075, 0.0411,
     "Shells, registration, ROI bound, head and void volumes"),
    ("v7", 319837, 0.2611, 0.2988, 0.0377,
     "v9 without the head and void volumes"),
    ("v10a", 319839, 0.2507, 0.2854, 0.0347,
     "v7 with temperature T = 1.2"),
    ("v11", 319842, 0.2502, 0.2802, 0.0300,
     "T = 1.4, 34-checkpoint 3D CNN, fixed-ratio stack, governor"),
    ("v13c", 321081, 0.2462, 0.2738, 0.0276,
     "Head-mask disc guard, 2D EfficientNet-B0 MIP member"),
    ("v15a3", 322345, 0.2444, 0.2750, 0.0306,
     "Guarded-frame 2D (B0 pair, B3) and TabICLv2 block"),
    ("v17", 322397, 0.2409, 0.2731, 0.0322,
     "Balanced TabICL context, mixture shapes, 4 linear priors"),
]
# Development notes: v15a (322323) and v15a2 (322341) FAILED at startup, no score.
FAILED = [("v15a", 322323), ("v15a2", 322341)]

# --------------------------------------------------------------------------- #
# DATA 2: OOF performance of the v13a stack at temperature 1, per inferred
# acquisition centre. A centre is one geometry family at one raw-intensity scale
# (the fam2/scale groups of docs/stages/07_tabular.md). Six of the ten are drawn.
# Source: the author's development notes (not distributed), per-centre aggregates
#         of the OOF table of "the shipped v13a stack at T = 1".
#         Corrected on 30 Sep 2026: the thin-slab and 2.4 mm rows had used their
#         whole geometry families (255 and 149 scans), which include 16 and 7
#         high-count scans, so the two blue rows did not add up to the 36% legend.
# (short id, geometry line, intensity line, n, prevalence, OOF loss, OOF AUROC, hard)
# --------------------------------------------------------------------------- #
FAMILIES = [
    ("s246/high", "256³ @ 2.46 mm", "high counts, p99 ≈ 3400",
     270, 0.50, 0.190, 0.9735, False),
    ("s246/low", "256³ @ 2.46 mm", "quantised low counts, p99 ≈ 20",
     258, 0.69, 0.250, 0.9579, True),
    ("full39/high", "128³ @ 3.895 mm", "high counts, p99 ≈ 3900",
     70, 0.59, 0.106, 0.9958, False),
    ("thin39/low", "128×128×36-49 @ 3.9 mm slabs", "p99 ≈ 24, 90% oblique",
     239, 0.57, 0.245, 0.9614, True),
    ("s23/low", "128³ @ 2.3 mm", "max-normalised to 255",
     207, 0.50, 0.196, 0.9736, False),
    ("s24/low", "128×128×87 @ 2.4 mm", "3-21 grey levels",
     142, 0.45, 0.113, 0.9942, False),
]
COHORT_N = 1362            # development notes (1,362 scans, 747 abnormal)
COHORT_PREV = 747 / 1362   # development notes (prevalence 0.548)
HARD_SCAN_SHARE = 0.36     # development notes (s246/low 258 + thin39/low 239 = 497 scans, 36%)
HARD_LOSS_SHARE = 0.45     # development notes (123 of 276 nats)
OTHER_GROUPS_N = 4         # development notes (small/high, small/low, thin39/high, s24/high)


# --------------------------------------------------------------------------- #
# DATA 3: glosses for the linear-model feature names, from the feature code.
# --------------------------------------------------------------------------- #
BASE_GLOSS = {
    # inference/features4.py:157-162 (mean gradient on the >=50%-of-peak boundary / peak)
    "grad_edge_norm": "boundary sharpness of the uptake blob, relative to peak",
    "grad_edge": "boundary sharpness of the uptake blob",
    "grad_p99": "99th-percentile gradient magnitude of the crop",
    # inference/features4.py:103-107 (posterior / anterior putamen ROI means)
    "pa_ratio_min": "posterior:anterior putamen ratio, lower side",
    "pc_ratio_min": "putamen:caudate ratio, lower side",
    # inference/features4.py:131-145 (connected components of the >=50%-of-peak mask)
    "blob_n": "number of separate uptake blobs above 50% of peak",
    # inference/features4.py:96-98 (occipital box behind the striata)
    "bg_mean": "mean SBR of the occipital background box",
    "stri_over_bg": "striatal mean SBR over background",
    "spec_total": "total specific binding (SBR above 1) in the crop",
    "lr_vol_asym": "left/right volume asymmetry above 60% of peak",
    "sep_mm": "distance between the two striatal peaks",
}
# inference/features4.py:80-95: ROI prefixes and suffixes
ROI = {"stri": "striatum", "cau": "caudate", "put": "putamen",
       "aput": "anterior putamen", "ppu": "posterior putamen"}
# inference/features6.py:36-94: per-hemisphere statistics
HEMI = {"peak": "peak SBR", "mean": "mean SBR", "p99": "99th-percentile SBR",
        "p90": "90th-percentile SBR", "hi_vol": "volume above 50% of peak",
        "spec": "specific binding", "elong": "elongation of uptake",
        "flat": "flatness of uptake", "rms_mm": "spread of uptake (mm)",
        "ap_extent": "anterior-posterior extent of uptake",
        "ap_centroid": "anterior-posterior centroid of uptake",
        "ncc": "number of uptake blobs"}
# inference/features5.py:24-26,36-45: grid cell index = ix*8 + iy*2 + iz with
# 3 x-bins, 4 y-bins (posterior to anterior) and 2 z-bins (inferior to superior).
# The x-bin mirrors between hemispheres, so only y and z are glossed.
GRID_Y = ["most posterior", "posterior-mid", "anterior-mid", "most anterior"]
GRID_Z = ["inferior", "superior"]
# inference/features13.py:40: radial shell edges in mm
SHELLS = (0, 3, 6, 9, 12, 16, 20)


def gloss(name: str) -> str:
    """Short reader-facing description of a feature name, read off the feature code."""
    if name in BASE_GLOSS:
        return BASE_GLOSS[name]
    m = re.fullmatch(r"p(\d+(?:\.\d+)?)", name)
    if m:                                                   # features4.py:115-116
        return f"{float(m.group(1)):g}th-percentile SBR of the whole crop"
    m = re.fullmatch(r"volfrac_gt([\d.]+)", name)
    if m:                                                   # features4.py:125-127
        return f"volume at or above {float(m.group(1)) * 100:g}% of peak (mL)"
    m = re.fullmatch(r"vol_gt([\d.]+)", name)
    if m:                                                   # features4.py:123-124
        return f"volume with SBR of {float(m.group(1)):g} or more (mL)"
    m = re.fullmatch(r"g(\d\d)_(worse|better|rel)", name)
    if m:                                                   # features5.py:69-73
        i = int(m.group(1))
        where = f"{GRID_Y[(i % 8) // 2]}, {GRID_Z[i % 2]}"
        if m.group(2) == "rel":
            return f"grid-cell share of uptake, {where}"
        return f"grid-cell SBR, {where}, {m.group(2)} side"
    m = re.fullmatch(r"h_(.+)_(worse|better|rel|ratio)", name)
    if m and m.group(1) in HEMI:                            # features6.py:122-128
        what, kind = HEMI[m.group(1)], m.group(2)
        if kind == "rel":
            return f"{what}, better minus worse hemisphere (scaled)"
        if kind == "ratio":
            return f"{what}, worse over better hemisphere"
        return f"{what}, {kind} hemisphere"
    m = re.fullmatch(r"s_r(\d)_(worse|better|rel)", name)
    if m:                                                   # features13.py:88-95
        i = int(m.group(1))
        band = f"{SHELLS[i]}-{SHELLS[i + 1]} mm shell"
        if m.group(2) == "rel":
            return f"SBR in the {band}, better minus worse (scaled)"
        return f"SBR in the {band}, {m.group(2)} hemisphere"
    m = re.fullmatch(r"(stri|cau|put|aput|ppu)_(asym|min|max|mean|spec_min|spec_mean)", name)
    if m:                                                   # features4.py:86-95
        roi, kind = ROI[m.group(1)], m.group(2)
        return {"asym": f"{roi} left/right asymmetry",
                "min": f"{roi} mean SBR, lower side",
                "max": f"{roi} mean SBR, higher side",
                "mean": f"{roi} mean SBR, both sides",
                "spec_min": f"{roi} specific binding, lower side",
                "spec_mean": f"{roi} specific binding, both sides"}[kind]
    m = re.fullmatch(r"(stri|cau|put|aput|ppu)_(L|R)_(mean|p90)", name)
    if m:                                                   # features4.py:81-85
        stat = "mean" if m.group(3) == "mean" else "90th-percentile"
        return f"{ROI[m.group(1)]} {stat} SBR, {'left' if m.group(2) == 'L' else 'right'}"
    return ""


# --------------------------------------------------------------------------- #
# Shared drawing helpers
# --------------------------------------------------------------------------- #
def style_axes(ax, grid_axis: str = "x") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=0, labelsize=8.5, pad=4)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def header(fig, title: str, subtitle: str, x: float = 0.02, top: float = 0.965) -> None:
    fig.text(x, top, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(x, top - 0.055, subtitle, fontsize=9, color=INK2, va="top", linespacing=1.4)


def footer(fig, text: str, x: float = 0.02, y: float = 0.02) -> None:
    fig.text(x, y, text, fontsize=7.8, color=MUTED, va="bottom", linespacing=1.4)


def _px_to_data(ax, px: float) -> tuple[float, float]:
    """Convert a length in output pixels to data units along x and y."""
    fig = ax.figure
    pos = ax.get_position()
    w_px = pos.width * fig.get_figwidth() * DPI
    h_px = pos.height * fig.get_figheight() * DPI
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return px * abs(x1 - x0) / w_px, px * abs(y1 - y0) / h_px


def rounded_bar(ax, base: float, end: float, center: float, thick: float,
                color: str, horizontal: bool = True, radius_px: float = 8.0) -> None:
    """Bar that is square at the baseline and rounded at the data end.

    Call after the axis limits are final: the corner radius is set in pixels.
    """
    rx, ry = _px_to_data(ax, radius_px)
    if not horizontal:
        rx, ry = ry, rx                      # swap roles: length runs along y
    length = end - base
    sgn = 1.0 if length >= 0 else -1.0
    r_len = min(rx, abs(length))
    r_th = min(ry, thick / 2)
    lo, hi = center - thick / 2, center + thick / 2
    e0 = end - sgn * r_len
    pts = [(base, lo), (e0, lo), (end, lo), (end, lo + r_th), (end, hi - r_th),
           (end, hi), (e0, hi), (base, hi), (base, lo)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO,
             MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CLOSEPOLY]
    if not horizontal:
        pts = [(c, v) for (v, c) in pts]
    ax.add_patch(PathPatch(MPath(pts, codes), facecolor=color, edgecolor="none"))


def save(fig, out: Path, name: str, svg: bool = False) -> Path:
    path = out / name
    fig.savefig(path, dpi=DPI)
    if svg:
        fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Figure 1: score progression
# --------------------------------------------------------------------------- #
def fig_score_progression(out: Path) -> Path:
    n = len(SCORES)
    fig = plt.figure(figsize=(11, 5.6))
    ax = fig.add_axes([0.50, 0.13, 0.47, 0.66])
    style_axes(ax, "x")
    ys = list(range(n))
    for y, (ver, sid, pub, priv, _d, _c) in zip(ys, SCORES):
        ax.plot([pub, priv], [y, y], color=AXIS, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([pub], [y], s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter([priv], [y], s=70, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_xlim(0.225, 0.375)
    ax.set_xticks([0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Log loss (lower is better)", fontsize=9, color=INK2, labelpad=6)

    # direct labels on the prize entry only
    ver, sid, pub, priv, _d, _c = SCORES[-1]
    ax.annotate(f"{pub:.4f}", (pub, n - 1), xytext=(-9, 0), textcoords="offset points",
                ha="right", va="center", fontsize=8.5, color=INK)
    ax.annotate(f"{priv:.4f}", (priv, n - 1), xytext=(9, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8.5, color=INK)

    # row labels: version and one-line change, left of the plot
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    for y, (ver, sid, _p, _q, _d, change) in zip(ys, SCORES):
        prize = ver == "v17"
        fig.text(0.02, y, ver, transform=tr, ha="left", va="center", fontsize=9,
                 fontweight="bold" if prize else "bold", color=INK)
        fig.text(0.075, y, change + ("  (prize entry)" if prize else ""), transform=tr,
                 ha="left", va="center", fontsize=8.5, color=INK if prize else INK2)

    header(fig, "Public and private log loss of each scored submission",
           "Rows follow submission order, platform ids 317177 to 322397. "
           "The grey link spans the public-to-private drift.")
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=7.5,
                      markerfacecolor=BLUE, markeredgecolor=SURFACE, label="Public leaderboard"),
               Line2D([], [], marker="o", linestyle="none", markersize=7.5,
                      markerfacecolor=ORANGE, markeredgecolor=SURFACE, label="Private leaderboard")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.495, 0.855), ncol=2,
               frameon=False, fontsize=8.5, handletextpad=0.3, columnspacing=1.6)
    failed = ", ".join(f"{v} ({i})" for v, i in FAILED)
    footer(fig, f"Not shown: {failed} failed at startup and received no score. "
                "Source: DrivenData submission history for team NavAttack.")
    return save(fig, out, "score_progression.png")


# --------------------------------------------------------------------------- #
# Figure 2: per-family OOF performance, four small multiples on shared rows
# --------------------------------------------------------------------------- #
def fig_family_loss(out: Path) -> Path:
    rows = sorted(FAMILIES, key=lambda r: -r[5])            # hardest (highest loss) on top
    n = len(rows)
    fig = plt.figure(figsize=(12, 5.2))
    left, right, bottom, top, gap = 0.215, 0.985, 0.10, 0.70, 0.034
    widths = [1.0, 1.0, 1.0, 1.0]
    span = right - left - gap * (len(widths) - 1)
    axes, x = [], left
    for w in widths:
        wf = span * w / sum(widths)
        axes.append(fig.add_axes([x, bottom, wf, top - bottom]))
        x += wf + gap
    ys = list(range(n))
    colour = [BLUE if r[7] else DEEMPH for r in rows]
    thick = 0.46

    panels = [
        ("Scans", 3, (0, 300), [0, 100, 200, 300], "{:.0f}"),
        ("Prevalence (share abnormal)", 4, (0, 1.0), [0, 0.25, 0.5, 0.75, 1.0], "{:.2f}"),
        ("OOF log loss, lower is better", 5, (0, 0.30), [0, 0.1, 0.2, 0.3], "{:.3f}"),
    ]
    for pi, (ax, (title, col, lim, ticks, fmt)) in enumerate(zip(axes[:3], panels)):
        style_axes(ax, "x")
        ax.set_xlim(*lim)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontsize=9, color=INK, pad=8, fontweight="bold")
        for y, r, c in zip(ys, rows, colour):
            rounded_bar(ax, 0.0, r[col], y, thick, c, horizontal=True, radius_px=6)
            knock = dict(boxstyle="square,pad=0.15", facecolor=SURFACE, edgecolor="none") \
                if pi == 1 else None
            ax.annotate(fmt.format(r[col]), (r[col], y), xytext=(4, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=8, color=INK2, bbox=knock, zorder=4)
    # cohort prevalence reference line (solid hairline, labelled)
    axp = axes[1]
    axp.axvline(COHORT_PREV, color=MUTED, linewidth=0.9, zorder=1)
    axp.annotate(f"cohort {COHORT_PREV:.3f}", (COHORT_PREV, -0.5), xytext=(3, -2),
                 textcoords="offset points", ha="left", va="top", fontsize=7.5, color=MUTED)

    # AUROC: dots on a truncated axis (bars would have to start at zero)
    axa = axes[3]
    style_axes(axa, "x")
    axa.set_xlim(0.95, 1.0)
    axa.set_ylim(n - 0.5, -0.5)
    axa.set_xticks([0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
    axa.set_xticklabels(["0.95", "0.96", "0.97", "0.98", "0.99", "1"])
    axa.set_yticks([])
    axa.set_title("OOF AUROC, higher is better", loc="left", fontsize=9, color=INK,
                  pad=8, fontweight="bold")
    for y, r, c in zip(ys, rows, colour):
        axa.plot([0.95, r[6]], [y, y], color=GRID, linewidth=1, zorder=1)
        axa.scatter([r[6]], [y], s=64, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        right_side = r[6] < 0.985
        axa.annotate(f"{r[6]:.4f}", (r[6], y), xytext=(8 if right_side else -8, 0),
                     textcoords="offset points", ha="left" if right_side else "right",
                     va="center", fontsize=8, color=INK2)
    for ax in axes:
        ax.spines["left"].set_visible(False)

    # row labels: geometry on the first line, intensity scale below
    tr = blended_transform_factory(fig.transFigure, axes[0].transData)
    for y, r in zip(ys, rows):
        fig.text(0.02, y - 0.13, r[1], transform=tr, ha="left", va="center",
                 fontsize=8.8, color=INK)
        fig.text(0.02, y + 0.22, r[2], transform=tr, ha="left", va="center",
                 fontsize=8, color=INK2)

    total = sum(r[3] for r in rows)
    header(fig, "Out-of-fold performance by inferred acquisition centre",
           f"v13a stack at temperature 1, training cohort of {COHORT_N:,} scans. "
           f"Six inferred centres, {total:,} scans.\n"
           "Centres are inferred from image geometry and intensity scale. "
           "The data carry no centre label.")
    handles = [Line2D([], [], marker="s", linestyle="none", markersize=8, markerfacecolor=BLUE,
                      markeredgecolor=BLUE,
                      label=f"Two hardest centres: {HARD_SCAN_SHARE:.0%} of scans, "
                            f"{HARD_LOSS_SHARE:.0%} of OOF loss"),
               Line2D([], [], marker="s", linestyle="none", markersize=8, markerfacecolor=DEEMPH,
                      markeredgecolor=DEEMPH, label="Other large centres")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.815), ncol=2,
               frameon=False, fontsize=8.5, handletextpad=0.3, columnspacing=1.8)
    footer(fig, f"Not shown: {COHORT_N - total:,} scans in {OTHER_GROUPS_N} other groups. "
                "Prevalence reference line: cohort share of abnormal scans.")
    return save(fig, out, "family_loss.png")


# --------------------------------------------------------------------------- #
# Figure 3: coefficients of the shipped L1 model `lin`
# --------------------------------------------------------------------------- #
def load_linear(path: Path) -> tuple[list[str], list[float]]:
    d = json.loads(Path(path).read_text())
    return list(d["features"]), [float(c) for c in d["coef"]]


def top_coefficients(names, coefs, k=15):
    nz = [(abs(c), c, nm) for nm, c in zip(names, coefs) if c != 0.0]
    nz.sort(key=lambda t: -t[0])
    return [(nm, c) for _a, c, nm in nz[:k]], len(nz)


def fig_lin_coefficients(out: Path, lin_path: Path, lin4_path: Path | None) -> Path:
    names, coefs = load_linear(lin_path)
    top, n_nonzero = top_coefficients(names, coefs, 15)
    lin4_note = ""
    if lin4_path is not None and Path(lin4_path).exists():
        n4, c4 = load_linear(lin4_path)
        _t4, nz4 = top_coefficients(n4, c4, 15)
        lin4_note = f"The second linear member, lin4 ({len(n4)} features, {nz4} non-zero), is not shown. "

    k = len(top)
    fig = plt.figure(figsize=(11.5, 6.4))
    ax = fig.add_axes([0.505, 0.19, 0.47, 0.57])
    style_axes(ax, "x")
    lim = max(abs(c) for _n, c in top) * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_ylim(k - 0.5, -0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color=MUTED, linewidth=0.9, zorder=2)
    for y, (nm, c) in enumerate(top):
        rounded_bar(ax, 0.0, c, y, 0.56, RED if c > 0 else BLUE, horizontal=True, radius_px=6)
        ax.annotate(f"{c:+.2f}".replace("-", "\u2212"), (c, y), xytext=(4 if c > 0 else -4, 0),
                    textcoords="offset points", ha="left" if c > 0 else "right",
                    va="center", fontsize=8, color=INK2)
    ax.set_xlabel("Change in log-odds of an abnormal scan per 1 SD of the transformed feature",
                  fontsize=8.5, color=INK2, labelpad=6)

    tr = blended_transform_factory(fig.transFigure, ax.transData)
    for y, (nm, _c) in enumerate(top):
        fig.text(0.02, y, nm, transform=tr, ha="left", va="center", fontsize=8.5,
                 family="monospace", color=INK)
        fig.text(0.155, y, gloss(nm), transform=tr, ha="left", va="center",
                 fontsize=8.3, color=INK2)

    header(fig, "Largest coefficients of the shipped L1 logistic member lin",
           f"{k} of the {n_nonzero} non-zero coefficients, out of {len(names)} input features. "
           "Before the fit each feature is log1p-transformed,\nclipped at its 0.5th and 99.5th "
           "training percentiles and standardised, so bar lengths compare across features.")
    handles = [Line2D([], [], marker="s", linestyle="none", markersize=8, markerfacecolor=RED,
                      markeredgecolor=RED, label="Raises P(abnormal)"),
               Line2D([], [], marker="s", linestyle="none", markersize=8, markerfacecolor=BLUE,
                      markeredgecolor=BLUE, label="Lowers P(abnormal)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.53, 0.83), ncol=2,
               frameon=False, fontsize=8.5, handletextpad=0.3, columnspacing=1.8)
    footer(fig, "SBR: voxel value divided by the reference level. Worse hemisphere: the side with "
                "less total specific uptake. Correlated features share weight,\nso read the signs "
                "of related features together (for example h_p99_worse and h_p90_worse). "
                + lin4_note)
    return save(fig, out, "lin_coefficients.png")


# --------------------------------------------------------------------------- #
# Figure 4: pipeline diagram. Every constant is checked against the shipped code:
#   crop 96 mm, 48^3, 2 mm ........ inference/preprocessing3.py:44-46, inference_main.py:341-346
#   disc guard 7,500 mL ........... inference/preprocessing3.py:117, 208-217
#   reference = log-histogram mode in the head mask eroded by 10 mm
#                                  inference/preprocessing3.py:241-253, 256-284, 723-724
#   guarded frame, falls back to scanner z if the PCA superior axis tilts > ~26 deg
#                                  inference/preprocessing3.py:504, 537-541, 750-752
#   registration reject > 90 deg or NCC < 0.30 ... inference/regfit.py:52-53,
#                                  assets: inference_config.json "registration"
#   479 features = 239 + 180 + 60 . inference/inference_main.py:387-393
#   lin 235 / lin4 94 features .... assets: linear_lin.json, linear_lin4.json "features"
#   3D CNN, 34 checkpoints ........ assets: inference_config.json "cnns"[0].sources
#   B0 x 12 (192 px), B3 x 6 (224 px), flip averaging
#                                  assets: inference_config.json "cnns"[1..2]; inference/cnn2d.py:227-238
#   stack coef = 0.5312 x (1, 0.4, 0.7, 0.3, 0.3), intercept 0.1843
#                                  assets: inference_config.json "stacker"
#   T = 1.4, clip 0.005 ........... assets: inference_config.json "calibration";
#                                  inference/inference_main.py:464-477
#   TabICL 68 features (64 selected + 4 acquisition) + 4 priors, raw + log, T = 1.2,
#   weight 0.35 ................... assets: tabular_recipe.json; inference/tabular_adapter.py:53-72
#   prior 0.5485 .................. assets: inference_config.json "prior";
#                                  inference/inference_main.py:360-365, 468-474
# --------------------------------------------------------------------------- #
STACK_RATIO = (1, 0.4, 0.7, 0.3, 0.3)   # assets: inference_config.json stacker.coef / 0.5312
STACK_SCALE = 0.5312107357995878         # assets: inference_config.json stacker.coef[0]
STACK_INTERCEPT = 0.18431409136922594    # assets: inference_config.json stacker.intercept
T_STACK = 1.4                            # assets: inference_config.json calibration.T
T_TAB = 1.2                              # assets: tabular_recipe.json temperature
W_TAB = 0.35                             # assets: tabular_recipe.json blend_weight
PRIOR = 0.5484581497797357               # assets: inference_config.json prior (= 747/1362)
CLIP = 0.005                             # assets: inference_config.json calibration.clip


def _tint(hex_colour: str, amount: float) -> tuple[float, float, float]:
    h = hex_colour.lstrip("#")
    s = SURFACE.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    b = [int(s[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(bb + (cc - bb) * amount for cc, bb in zip(c, b))


def fig_pipeline(out: Path) -> Path:
    W, H = 166.0, 76.0
    fig = plt.figure(figsize=(16.6, 7.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def box(x, y, w, h, title, body="", hue=None, title_size=9.0, body_size=7.9):
        fill = _tint(hue, 0.10) if hue else _tint(INK, 0.035)
        edge = _tint(hue, 0.55) if hue else AXIS
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.1",
                                    facecolor=fill, edgecolor=edge, linewidth=1.0))
        cx = x + w / 2
        if body:
            nl = body.count("\n") + 1
            block = 1.7 + nl * 1.75
            y_title = y + h / 2 + block / 2 - 0.6
            ax.text(cx, y_title, title, ha="center", va="top", fontsize=title_size,
                    fontweight="bold", color=INK)
            ax.text(cx, y_title - 2.2, body, ha="center", va="top", fontsize=body_size,
                    color=INK2, linespacing=1.35)
        else:
            ax.text(cx, y + h / 2, title, ha="center", va="center", fontsize=title_size,
                    fontweight="bold", color=INK)

    def arrow(points):
        verts = [tuple(p) for p in points]
        codes = [MPath.MOVETO] + [MPath.LINETO] * (len(verts) - 1)
        ax.add_patch(FancyArrowPatch(path=MPath(verts, codes), arrowstyle="-|>",
                                     mutation_scale=9, color=INK2, linewidth=1.0,
                                     shrinkA=0, shrinkB=0.5))

    # title
    ax.text(1.2, H - 1.5, "v17 inference pipeline", fontsize=13, fontweight="bold",
            color=INK, va="top")
    ax.text(1.2, H - 4.6, "Each scan is processed on its own. All numbers are the shipped constants.",
            fontsize=9, color=INK2, va="top")

    # column 0 and 1: input and preprocessing
    box(1, 37, 11, 10, "NIfTI scan", "3D DaT-SPECT\nvolume")
    arrow([(12, 42), (15, 42)])
    ax.add_patch(FancyBboxPatch((15, 8), 29, 60, boxstyle="round,pad=0,rounding_size=1.1",
                                facecolor=_tint(INK, 0.035), edgecolor=AXIS, linewidth=1.0))
    ax.text(29.5, 66, "Preprocessing", ha="center", va="top", fontsize=9.5,
            fontweight="bold", color=INK)
    steps = [
        ("Head mask", "log-Otsu threshold with a disc guard:\na mask over 7,500 mL is re-thresholded"),
        ("Reference level", "mode of the log histogram inside the\nhead mask eroded by 10 mm. "
                            "Voxels are\ndivided by it (SBR units)"),
        ("Striatal localiser", "left and right striatal uptake peaks\nafter 4 mm smoothing"),
        ("Canonical frame", "axes from the inter-striatal line, plus a\nguarded frame "
                            "that falls back to scanner z\nwhen the head axis tilts over about 26°"),
        ("Crop", "96 mm cube resampled to 48³ voxels\nat 2 mm"),
    ]
    # spread the five steps evenly between the box title and the box floor
    y, floor = 60.5, 12.5
    heights = [2.1 + 1.75 * (b.count("\n") + 1) for _t, b in steps]
    step_gap = (y - floor - sum(heights)) / (len(steps) - 1)
    for i, ((t, b), h) in enumerate(zip(steps, heights), 1):
        ax.text(17, y, f"{i}", ha="left", va="top", fontsize=8.5, fontweight="bold", color=MUTED)
        ax.text(19.5, y, t, ha="left", va="top", fontsize=8.5, fontweight="bold", color=INK)
        ax.text(19.5, y - 2.1, b, ha="left", va="top", fontsize=7.8, color=INK2, linespacing=1.35)
        y -= h + step_gap

    # column 2: crops
    box(48, 57, 18, 10, "Registered crop", "canonical crop after a rigid fit\nto a template. "
        "Fit rejected if\nrotation > 90° or NCC < 0.30", body_size=7.4)
    box(48, 28, 18, 26, "Guarded-frame crop", "same 96 mm crop in the\nguarded frame")
    box(48, 12, 18, 10, "Canonical crop", "striatal frame, SBR units")
    arrow([(44, 62), (48, 62)])
    arrow([(44, 41), (48, 41)])
    arrow([(44, 17), (48, 17)])

    # column 3: feature extractors and CNNs
    box(70, 57, 24, 10, "Hand-crafted features (239)",
        "ROI SBR, clinical ratios, spatial grid,\nper-hemisphere shape, radial shells", hue=BLUE)
    box(70, 45, 24, 10, "Shape features (240)",
        "scale-space (180) and two-component\nGaussian mixture (60)", hue=VIOLET)
    box(70, 26, 24, 12, "2D CNNs on three MIPs",
        "EfficientNet-B0 × 12 (192 px) and\nB3 × 6 (224 px), TorchScript,\n"
        "left-right flip averaged", hue=ORANGE)
    box(70, 12, 24, 10, "3D hemisphere-MIL CNN", "34 checkpoints, averaged", hue=AQUA)
    arrow([(66, 62), (70, 62)])
    arrow([(66, 50), (70, 50)])
    arrow([(66, 32), (70, 32)])
    arrow([(66, 17), (70, 17)])

    # column 4: feature table and linear members
    box(98, 48, 18, 20, "Feature table", "479 columns\n(239 + 180 + 60)")
    box(98, 36, 18, 9, "lin and lin4", "L1 logistic regression on\n235 and 94 features", hue=BLUE)
    arrow([(94, 62), (98, 62)])
    arrow([(94, 50), (98, 50)])
    arrow([(107, 48), (107, 45)])

    # column 5: TabICL block and stack
    box(120, 48, 22, 20, "TabICLv2 block",
        "68 features (64 selected + 4\nacquisition) and 4 cross-fitted\nlinear priors. "
        "Raw and log\ncontexts averaged. T = 1.2", hue=VIOLET)
    ratio = " : ".join(f"{r:g}" for r in STACK_RATIO)
    box(120, 10, 22, 34, "Stack",
        f"fixed logit ratio\n{ratio}\n(lin, lin4, 3D, B0, B3)\n\n"
        f"1-D calibration: scale {STACK_SCALE:.3f},\nintercept {STACK_INTERCEPT:.3f}\n\n"
        f"temperature T = {T_STACK:g}")
    arrow([(116, 58), (120, 58)])
    arrow([(116, 40.5), (120, 40.5)])
    arrow([(94, 32), (120, 32)])
    arrow([(94, 17), (120, 17)])

    # column 6: blend, governor, clip, output
    box(146, 40, 19, 11, "Blend in logit space",
        f"{1 - W_TAB:.2f} × stack +\n{W_TAB:.2f} × TabICL")
    arrow([(142, 58), (155.5, 58), (155.5, 51)])
    arrow([(142, 43), (146, 43)])
    box(146, 22, 19, 14, "Governor and fallbacks",
        "rejected registration,\nunreadable scan or\nnon-finite value:\n"
        f"prior {PRIOR:.4f}")
    arrow([(155.5, 40), (155.5, 36)])
    box(146, 12, 19, 6.5, f"Clip to [{CLIP:g}, {1 - CLIP:g}]", title_size=8.5)
    arrow([(155.5, 22), (155.5, 18.5)])
    box(146, 2.5, 19, 6.5, "P(abnormal)", hue=RED, title_size=9.5)
    arrow([(155.5, 12), (155.5, 9)])

    return save(fig, out, "pipeline_diagram.png", svg=True)


# --------------------------------------------------------------------------- #
# Figure 5: public-to-private drift
# --------------------------------------------------------------------------- #
def fig_drift(out: Path) -> Path:
    fig = plt.figure(figsize=(9, 4.6))
    ax = fig.add_axes([0.085, 0.14, 0.895, 0.66])
    style_axes(ax, "y")
    n = len(SCORES)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 0.046)
    ax.set_yticks([0, 0.01, 0.02, 0.03, 0.04])
    ax.set_yticklabels(["0", "0.01", "0.02", "0.03", "0.04"])
    ax.set_xticks(range(n))
    ax.set_xticklabels([s[0] for s in SCORES], fontsize=8.8, color=INK)
    ax.spines["left"].set_visible(False)
    drifts = [round(s[3] - s[2], 4) for s in SCORES]
    lab = {0, drifts.index(max(drifts)), n - 1}             # v2 (lowest), maximum, v17
    for i, d in enumerate(drifts):
        rounded_bar(ax, 0.0, d, i, 0.3, BLUE, horizontal=False, radius_px=6)
        if i in lab:
            ax.annotate(f"+{d:.4f}", (i, d), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.3, color=INK)
    ax.set_ylabel("Private minus public log loss", fontsize=8.8, color=INK2, labelpad=6)
    header(fig, "Public-to-private drift of each scored submission",
           f"For every scored version the private log loss exceeded the public log loss, "
           f"by {min(drifts):.4f} to {max(drifts):.4f}.", top=0.955)
    footer(fig, "Bars follow submission order. Source: DrivenData submission history for team NavAttack.")
    return save(fig, out, "drift.png")


# --------------------------------------------------------------------------- #
def check_data() -> None:
    """The drift column of the source table must equal private minus public."""
    for ver, _sid, pub, priv, drift, _c in SCORES:
        if abs(round(priv - pub, 4) - drift) > 1e-9:
            raise SystemExit(f"drift mismatch for {ver}: {priv} - {pub} != {drift}")
    if abs(round(COHORT_PREV, 3) - 0.548) > 1e-9:
        raise SystemExit("cohort prevalence does not match 747/1362")
    hard_n = sum(r[3] for r in FAMILIES if r[7])
    if round(hard_n / COHORT_N, 2) != HARD_SCAN_SHARE:
        raise SystemExit("the hard rows do not add up to HARD_SCAN_SHARE of the cohort")
    if [round(STACK_SCALE * r, 6) for r in STACK_RATIO] != \
            [0.531211, 0.212484, 0.371848, 0.159363, 0.159363]:
        raise SystemExit("stack ratio does not reproduce inference_config.json stacker.coef")


def print_tables(lin_path: Path | None) -> None:
    print("| Version | Submission id | Public log loss | Private log loss | Drift | Change |")
    print("|---|---:|---:|---:|---:|---|")
    for ver, sid, pub, priv, drift, change in SCORES:
        print(f"| {ver} | {sid} | {pub:.4f} | {priv:.4f} | +{drift:.4f} | {change} |")
    print()
    print("| Inferred centre | Scans | Prevalence | OOF log loss | OOF AUROC |")
    print("|---|---:|---:|---:|---:|")
    for r in FAMILIES:
        print(f"| {r[1]}, {r[2]} | {r[3]} | {r[4]:.2f} | {r[5]:.3f} | {r[6]:.4f} |")
    if lin_path is not None and Path(lin_path).exists():
        names, coefs = load_linear(lin_path)
        top, _nz = top_coefficients(names, coefs, 15)
        print()
        print("| Rank | Feature | Meaning | Coefficient |")
        print("|---:|---|---|---:|")
        for i, (nm, c) in enumerate(top, 1):
            print(f"| {i} | `{nm}` | {gloss(nm)} | {c:+.3f} |")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--assets", type=Path, default=REPO / "inference" / "assets",
                    help="folder holding linear_lin.json and linear_lin4.json "
                         "(default: inference/assets of this repository)")
    ap.add_argument("--lin", type=Path, default=None, help="path to linear_lin.json")
    ap.add_argument("--lin4", type=Path, default=None, help="path to linear_lin4.json")
    ap.add_argument("--out", type=Path, default=HERE, help="output folder")
    ap.add_argument("--print-tables", action="store_true",
                    help="print the README data tables as markdown and exit")
    a = ap.parse_args(argv)
    lin = a.lin or (a.assets / "linear_lin.json")
    lin4 = a.lin4 or (a.assets / "linear_lin4.json")
    check_data()
    if a.print_tables:
        print_tables(lin if lin.exists() else None)
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    made = [fig_score_progression(a.out), fig_family_loss(a.out)]
    if lin.exists():
        made.append(fig_lin_coefficients(a.out, lin, lin4 if lin4.exists() else None))
    else:
        print(f"skipped lin_coefficients.png: {lin.name} not found "
              "(install the weights or pass --assets / --lin)", file=sys.stderr)
    made += [fig_pipeline(a.out), fig_drift(a.out)]
    for p in made:
        print(f"wrote {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
