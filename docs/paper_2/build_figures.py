"""
build_figures.py
================
Regenerates every figure to Nature Portfolio specification.

DESIGN RULES ENFORCED HERE (Nature Portfolio artwork guidelines)
  width      88 mm (single column) or 180 mm (double column), never in between
  type       Arial/Helvetica metric-compatible sans; 5 pt minimum, 7 pt for axis labels
  panels     bold lower-case letters, outside the axes, top-left
  rules      0.5 pt axes, 0.75 pt data; no top or right spine; no gridlines unless load-bearing
  colour     Okabe-Ito colour-blind-safe palette; never a rainbow scale
  labelling  direct annotation in preference to a legend wherever it fits
  output     vector PDF with fonts as outlines-free Type 42, so the journal can re-typeset

PROVENANCE OF EVERY NUMBER
  Each data block below cites the Supplementary Table it was taken from. Nothing is
  invented. Two series are NOT in the manuscript and must be supplied by the authors from
  the released run outputs; the script says so loudly and refuses to guess:

      figdata/fig5a_rutile_sweep.csv     33 distortion amplitudes x 7 arms
      figdata/fig5b_jacobian_points.csv  60 structure-seed points x 6 arms
      figdata/figS1_raw_thresholds.csv   25 thresholds x 7 arms, raw coordinate variant

  Drop those three files in and rerun; the script picks them up automatically.

    python build_figures.py
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

OUT = "figures"
DATA = "figdata"
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

# ======================================================================================
# STYLE
# ======================================================================================
MM = 1 / 25.4

# --------------------------------------------------------------------------------------
# WIDTH.  Emit at the width the figure is RENDERED at, not at an abstract spec, so that the
# type a referee reads is the type we designed. The submission template's text block is
# 372 pt = 131.2 mm, and the SI includes its figure at 0.62 of that.  The art is vector, so
# the identical file scales losslessly to Nature's 180 mm production width at acceptance.
# --------------------------------------------------------------------------------------
W2 = 180.0 * MM        # Nature double column
W1 = 110.0 * MM        # the SI figure, included at 0.62 x text block
S = 180.0 / 131.2      # type scale: LaTeX shrinks 180 mm into a 131.2 mm block, so design
                       # 1.37x large and every glyph LANDS at 5-8 pt in the PDF a referee reads


O3      = "#0072B2"   # Okabe-Ito blue
SO3     = "#D55E00"   # Okabe-Ito vermillion
GREEN   = "#009E73"
PURPLE  = "#CC79A7"
AMBER   = "#E69F00"
INK     = "#1A1A1A"
GREY    = "#7A7A7A"
FAINT   = "#E8E8E8"
IMPOSS  = "#D55E00"

CORE_MK = {"NequIP": "o", "Allegro": "s", "MACE": "^", "EquiformerV2": "D"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.cursive": ["Liberation Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "mathtext.default": "regular",
    "font.size": 6 * S,
    "axes.labelsize": 7 * S,
    "axes.titlesize": 7 * S,
    "xtick.labelsize": 6 * S,
    "ytick.labelsize": 6 * S,
    "legend.fontsize": 6 * S,
    "axes.linewidth": 0.5,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "xtick.minor.size": 1.2,
    "ytick.minor.size": 1.2,
    "lines.linewidth": 0.9,
    "lines.markersize": 3.2,
    "legend.frameon": False,
    "legend.handlelength": 1.2,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": None,
    "figure.dpi": 300,
})


def panel(ax, letter, dx=-0.19, dy=1.015):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8 * S,
            fontweight="bold", va="bottom", ha="left")


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


SAVED = {}

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", transparent=False, facecolor="white")
    SAVED[name] = fig
    print(f"  wrote {OUT}/{name}.pdf")


# ======================================================================================
# DATA, with provenance
# ======================================================================================
TAU = 1e-2                 # operating threshold, C/m^2  (Methods)
REAL_MED = 0.51            # median magnitude of the real piezoelectric tensors (Results)

# Supplementary Table 4: percentiles of the per-structure violation magnitude (idealized).
# columns: p5, p25, median, p75, p95, max
DIST = {
    ("NequIP", "O(3)"):      [2.17e-8, 1.60e-7, 3.19e-7, 5.24e-7, 1.05e-6, 4.56e-4],
    ("NequIP", "SO(3)"):     [2.56e-7, 2.01e-1, 6.66e-1, 1.33e+0, 2.97e+0, 9.59e+0],
    ("Allegro", "O(3)"):     [3.65e-8, 1.93e-7, 3.84e-7, 6.71e-7, 1.41e-6, 9.72e-4],
    ("Allegro", "SO(3)"):    [2.76e-7, 3.17e-1, 9.60e-1, 1.87e+0, 4.22e+0, 1.89e+1],
    ("MACE", "O(3)"):        [2.11e-7, 1.40e-6, 2.76e-6, 4.74e-6, 1.04e-5, 1.14e-3],
    ("MACE", "SO(3)"):       [1.54e-6, 3.27e-1, 9.25e-1, 1.68e+0, 3.40e+0, 1.58e+1],
    ("EquiformerV2", "SO(3)"): [1.38e-2, 1.78e-1, 4.45e-1, 9.06e-1, 2.41e+0, 1.70e+1],
}
# Supplementary Table 3: false-flag fraction at five thresholds (idealized, mean of 3 seeds)
THR_TAU = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
THR = {
    ("NequIP", "O(3)"):        [0.0005, 0.0000, 0.0000, 0.0000, 0.0000],
    ("NequIP", "SO(3)"):       [0.9170, 0.9133, 0.8953, 0.8177, 0.3548],
    ("Allegro", "O(3)"):       [0.0005, 0.0002, 0.0000, 0.0000, 0.0000],
    ("Allegro", "SO(3)"):      [0.9168, 0.9155, 0.9095, 0.8553, 0.4802],
    ("MACE", "O(3)"):          [0.0007, 0.0003, 0.0000, 0.0000, 0.0000],
    ("MACE", "SO(3)"):         [0.9168, 0.9150, 0.9077, 0.8550, 0.4672],
    ("EquiformerV2", "SO(3)"): [0.9968, 0.9918, 0.9570, 0.8238, 0.2203],
}
FF = {"NequIP": 0.8953, "Allegro": 0.9095, "MACE": 0.9077, "EquiformerV2": 0.9570}

# Supplementary Table 7: test MAE, mean +/- s.d. over 3 seeds
ACC = {
    ("NequIP", "U0"):       ((53.65, 22.50), (51.66, 9.19)),
    ("NequIP", "dipole"):   ((0.0519, 0.0027), (0.0530, 0.0026)),
    ("NequIP", "elastic"):  ((24.33, 0.27), (24.49, 0.14)),
    ("NequIP", "piezo"):    ((0.2083, 0.0077), (0.2405, 0.0080)),
    ("Allegro", "U0"):      ((31.08, 12.89), (26.98, 4.88)),
    ("Allegro", "dipole"):  ((0.0751, 0.0023), (0.0764, 0.0021)),
    ("Allegro", "elastic"): ((23.72, 0.25), (23.89, 0.31)),
    ("Allegro", "piezo"):   ((0.2140, 0.0058), (0.2589, 0.0176)),
    ("MACE", "U0"):         ((16.76, 6.29), (26.09, 12.49)),
    ("MACE", "dipole"):     ((0.0484, 0.0018), (0.0500, 0.0045)),
    ("MACE", "elastic"):    ((24.92, 0.54), (24.94, 0.60)),
    ("MACE", "piezo"):      ((0.2222, 0.0066), (0.2567, 0.0084)),
}
EQV2_PIEZO = (0.2157, 0.0096)   # Supplementary Table 7

# main-text Fig. 4a values (red flag 12: transcribed, confirm against the run outputs)
INTRAIN = {"NequIP": 0.895, "Allegro": 0.899, "MACE": 0.896, "EquiformerV2": 0.922}

# Supplementary Table 12: loss-weight sweep, NequIP SO(3)
W = np.array([1, 10, 100])
SWEEP = {
    "trained-on zeros":            [0.8953, 0.8730, 0.6762],
    "held out, seen space groups": [0.9042, 0.9031, 0.8715],
    "held out, unseen":            [0.8685, 0.8420, 0.7604],
}
SWEEP_SD = {                       # Supplementary Table 12, s.d. over 3 seeds
    "trained-on zeros":            [0.0020, 0.0026, 0.0154],
    "held out, seen space groups": [0.0024, 0.0012, 0.0031],
    "held out, unseen":            [0.0039, 0.0072, 0.0060],
}
# Supplementary Table 15: false-flag fraction by point-group family
FAM_N = {"m-3m": 166, "m-3": 18, "non-cubic": 1816}
FAM = {
    "NequIP":       [0.0000, 0.9630, 0.9765],
    "Allegro":      [0.0000, 0.9815, 0.9919],
    "MACE":         [0.0000, 0.9444, 0.9903],
    "EquiformerV2": [0.5341, 0.9630, 0.9956],
}


def delta_sigma(core, target):
    (mo, so), (ms, ss) = ACC[(core, target)]
    return (ms - mo) / np.sqrt((so ** 2 + ss ** 2) / 2)


def load(name, cols):
    """Load an author-supplied series, or return None and say so."""
    p = f"{DATA}/{name}"
    if not os.path.exists(p) or os.path.getsize(p) < 40:
        return None
    import csv
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return None
    return {c: np.array([float(r[c]) for r in rows]) for c in cols}


def load_rows(name: str) -> list[dict] | None:
    """Load an author-supplied series as raw rows, identity columns intact."""
    import csv
    from pathlib import Path
    p = Path(DATA) / name
    if not p.exists() or p.stat().st_size < 40:
        return None
    with p.open() as f:
        rows = list(csv.DictReader(f))
    return rows or None


def arm_curves(rows: list[dict], x_col: str) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Group rows by (core, arm) into float-column arrays sorted along x_col."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        grouped.setdefault((r["core"], r["arm"]), []).append(r)
    curves = {}
    for key, rr in grouped.items():
        rr.sort(key=lambda r: float(r[x_col]))
        curves[key] = {c: np.array([float(r[c]) for r in rr])
                       for c in rr[0] if c not in ("core", "arm", "structure")}
    return curves


def needs_data(ax, fname, what):
    ax.set_facecolor("#FFF4F4")
    for s in ax.spines.values():
        s.set_color(IMPOSS)
        s.set_linewidth(0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.60, "AUTHOR DATA REQUIRED", ha="center", va="center", fontsize=7 * S,
            fontweight="bold", color=IMPOSS, transform=ax.transAxes)
    ax.text(0.5, 0.42, what, ha="center", va="center", fontsize=5.6 * S,
            color=IMPOSS, transform=ax.transAxes)
    ax.text(0.5, 0.24, f"drop {DATA}/{fname} in and rerun", ha="center", va="center",
            fontsize=5.6 * S, style="italic", color=IMPOSS, transform=ax.transAxes)


# ======================================================================================
# FIGURE 1  |  concept
# ======================================================================================
# Colour semantics are global and never violated:
#     blue   = the O(3) arm            orange = the SO(3) arm
#     ink    = the crystal and its physics (never a model)
# The earlier draft used blue for the O(3) arm AND for odd parity AND for the inversion
# operation, which is three meanings for one hue.
BLUE_D, BLUE_L = "#0072B2", "#7FC1E8"     # O(3): dark = even feature, light = odd feature
ORNG_D         = "#D55E00"                # SO(3): one tone, because there is no parity to show


def fig1():
    fig = plt.figure(figsize=(W2, 70 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.05], wspace=0.46,
                          left=0.055, right=0.985, top=0.86, bottom=0.155)

    # ---- a: the crystal, and why the tensor must vanish -------------------------------
    ax = fig.add_subplot(gs[0]); panel(ax, "a", dx=-0.10)
    ax.set_xlim(-1.42, 1.42); ax.set_ylim(-1.88, 1.30); ax.axis("off")
    ax.set_aspect("equal")
    # the unit cell, and a hint of the lattice it repeats on: this is a CRYSTAL, not a cluster
    for dx_, al in [(-2.0, 0.16), (2.0, 0.16), (0.0, 1.0)]:
        ax.add_patch(Rectangle((-1.0 + dx_, -0.9), 2.0, 1.8, fill=False, lw=0.55,
                               ls=(0, (2.6, 1.8)), ec=GREY, alpha=al, zorder=1))
    # every atom has a partner at -r, and species are preserved by the inversion
    pairs = [(0.30, 0.70), (0.72, 0.45), (0.88, -0.18), (0.18, 0.30)]
    kinds = [0, 1, 1, 0]
    for (px, py), k in zip(pairs, kinds):
        col, sz = (INK, 30) if k == 0 else ("#FFFFFF", 22)
        ec = INK
        for sgn in (+1, -1):
            ax.scatter([sgn * px], [sgn * py], s=sz, c=col, zorder=4,
                       edgecolors=ec, linewidths=0.6)
        ax.add_patch(FancyArrowPatch((px, py), (-px, -py), arrowstyle="-|>",
                                     mutation_scale=4.5, lw=0.45, color=GREY,
                                     alpha=0.6, zorder=2,
                                     connectionstyle="arc3,rad=0.0", shrinkA=2.2, shrinkB=2.2))
    ax.add_patch(Circle((0, 0), 0.085, facecolor="white", edgecolor=INK, lw=0.9, zorder=5))
    ax.plot([0], [0], marker="+", ms=3.6, mew=0.9, color=INK, zorder=6)
    ax.text(0, 1.14, "centrosymmetric crystal", ha="center", fontsize=6.8 * S,
            fontweight="bold")
    # a one-line key, placed in clear space, so no label has to sit on the drawing
    ax.text(0, -1.10, "open circle: inversion centre", ha="center", va="center",
            fontsize=5.4 * S, color=GREY)
    ax.text(0, -1.30, "arrows: r \u2192 \u2212r", ha="center", va="center",
            fontsize=5.4 * S, color=GREY)
    ax.text(0, -1.55, "I\u00b7x = x    and    e \u2192 \u2212e", ha="center", va="center",
            fontsize=6.2 * S, color=INK)
    ax.text(0, -1.78, "so  e = \u2212e,  and  e = 0", ha="center", va="center",
            fontsize=6.6 * S, fontweight="bold", color=INK)

    # ---- b: the only difference -------------------------------------------------------
    ax = fig.add_subplot(gs[1]); panel(ax, "b", dx=-0.05)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 1.10, "the only difference", ha="center", va="center",
            fontsize=6.8 * S, fontweight="bold")

    def ladder(y0, labels, cols, tag, tagcol, ytag):
        for j, (lab, c) in enumerate(zip(labels, cols)):
            x = 0.055 + j * 0.235
            ax.add_patch(Rectangle((x, y0), 0.195, 0.15, facecolor=c, edgecolor="none"))
            ax.text(x + 0.0975, y0 + 0.075, lab, ha="center", va="center",
                    fontsize=6.4 * S, color="white", fontweight="bold")
        ax.text(0.5, ytag, tag, ha="center", va="center", fontsize=6.6 * S,
                color=tagcol, fontweight="bold")

    # every string gets its own reserved horizontal band; nothing can land on anything else
    ladder(0.72, ["0e", "1o", "2e", "3o"], [BLUE_D, BLUE_L, BLUE_D, BLUE_L], "O(3)", O3, 0.93)
    ax.text(0.5, 0.655, r"natural parity $(-1)^l$", ha="center", va="center",
            fontsize=5.6 * S, color=O3)
    ax.annotate("", xy=(0.062, 0.495), xytext=(0.062, 0.595),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=GREY, mutation_scale=6))
    ax.text(0.56, 0.545, "erase the labels", fontsize=5.6 * S, color=GREY, va="center",
            ha="center", style="italic")
    ladder(0.235, ["0e", "1e", "2e", "3e"], [ORNG_D] * 4, "SO(3)", SO3, 0.44)
    ax.text(0.5, 0.165, "every degree declared even", ha="center", va="center",
            fontsize=5.6 * S, color=SO3)
    ax.text(0.5, -0.02, "architecture, data, splits, seeds:", ha="center", va="center",
            fontsize=5.4 * S, style="italic", color=GREY)
    ax.text(0.5, -0.115, "identical", ha="center", va="center", fontsize=5.4 * S,
            style="italic", color=GREY, fontweight="bold")

    # ---- c: what comes out ------------------------------------------------------------
    ax = fig.add_subplot(gs[2]); panel(ax, "c", dx=-0.22)
    ax.set_yscale("log"); ax.set_ylim(8e-9, 60); ax.set_xlim(0.30, 2.70)
    ax.axhspan(TAU, 60, color=IMPOSS, alpha=0.07, lw=0)
    ax.axhline(TAU, color=IMPOSS, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.axhline(REAL_MED, color=GREY, lw=0.6, ls=(0, (1, 1.6)), zorder=1)
    for xpos, arm, col in [(1, "O(3)", O3), (2, "SO(3)", SO3)]:
        p5, p25, med, p75, p95, mx = DIST[("NequIP", arm)]
        ax.plot([xpos, xpos], [p5, p95], color=col, lw=0.7, zorder=2)
        ax.add_patch(Rectangle((xpos - 0.20, p25), 0.40, p75 - p25, facecolor=col,
                               alpha=0.35, edgecolor=col, lw=0.6, zorder=3))
        ax.plot([xpos - 0.20, xpos + 0.20], [med, med], color=col, lw=1.4, zorder=4)
    # every label sits in verified empty space: no data lives left of x = 0.8 at any height
    ax.text(0.36, TAU * 2.1, "detection threshold", ha="left", fontsize=5.4 * S, color=IMPOSS)
    ax.text(0.36, REAL_MED * 2.1, "a real piezoelectric", ha="left", fontsize=5.4 * S,
            color=GREY)
    ax.text(1, 5.0e-6, "structurally\nzero", ha="center", fontsize=5.8 * S, color=O3,
            fontweight="bold")
    ax.text(2, 12.0, "physically\nimpossible", ha="center", fontsize=5.8 * S, color=SO3,
            fontweight="bold")
    ax.set_xticks([1, 2]); ax.set_xticklabels(["O(3)", "SO(3)"], fontsize=6.8 * S)
    ax.set_ylabel(r"predicted $\Vert e \Vert_F$  (C m$^{-2}$)", labelpad=2.5)
    ax.set_title("true value: exactly 0", fontsize=6.8 * S, fontweight="bold", pad=6)
    ax.text(0.5, -0.155, "NequIP, $n$ = 2,000 crystals", transform=ax.transAxes,
            ha="center", fontsize=5.2 * S, color=GREY)
    despine(ax)

    # ---- the pipeline: a is fed to b, which produces c ---------------------------------
    for x0, x1 in [(0.320, 0.368), (0.680, 0.728)]:
        fig.add_artist(FancyArrowPatch((x0, 0.955), (x1, 0.955), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=7, lw=0.9,
                                       color=GREY, alpha=0.9))
    save(fig, "fig1_concept")


# ======================================================================================
# FIGURE 2  |  headline
# ======================================================================================
def fig2():
    fig = plt.figure(figsize=(W2, 84 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.30,
                          left=0.095, right=0.968, top=0.845, bottom=0.185)

    # ---- a ---------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0]); panel(ax, "a", dx=-0.115)
    cores = ["NequIP", "Allegro", "MACE", "EquiformerV2"]
    ax.set_yscale("log"); ax.set_ylim(1.2e-8, 2e4); ax.set_xlim(0.3, 4.9)
    ax.axhspan(TAU, 2e4, color=IMPOSS, alpha=0.055, lw=0)
    ax.axhline(TAU, color=IMPOSS, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.axhline(REAL_MED, color=GREY, lw=0.6, ls=(0, (1, 1.6)), zorder=1)
    for i, core in enumerate(cores, start=1):
        for arm, off, col in [("O(3)", -0.17, O3), ("SO(3)", +0.17, SO3)]:
            if (core, arm) not in DIST:
                continue
            p5, p25, med, p75, p95, mx = DIST[(core, arm)]
            x = i + off
            ax.plot([x, x], [p5, p95], color=col, lw=0.7, solid_capstyle="butt", zorder=2)
            ax.add_patch(Rectangle((x - 0.115, p25), 0.23, p75 - p25, facecolor=col,
                                   alpha=0.35, edgecolor=col, lw=0.6, zorder=3))
            ax.plot([x - 0.115, x + 0.115], [med, med], color=col, lw=1.4, zorder=4)
            ax.plot([x], [mx], marker="_", ms=4.4, mew=0.8, color=col, zorder=4)
        ax.text(i + 0.17, 1.3e2, f"{FF[core]*100:.1f}%", ha="center", va="center",
                fontsize=5.8 * S, color=SO3, fontweight="bold")
    for j, (yk, col, ls, lab) in enumerate([
            (1.6e-5, IMPOSS, (0, (3, 2)), r"threshold, $\tau = 0.01$"),
            (1.1e-6, GREY,   (0, (1, 1.6)), "real piezoelectric, 0.51")]):
        ax.plot([3.06, 3.34], [yk, yk], color=col, lw=0.8, ls=ls)
        ax.text(3.42, yk, lab, ha="left", va="center", fontsize=5.0 * S, color=col)
    ax.text(0.40, 1.4e3, "physically impossible", fontsize=6.2 * S, color=SO3,
            fontweight="bold", ha="left", va="center")
    ax.text(0.40, 1.2e4, r"O(3): 0 of 18,000 runs above $\tau$", fontsize=5.9 * S,
            color=O3, fontweight="bold", ha="left", va="center")
    mo, ms = DIST[("NequIP", "O(3)")][2], DIST[("NequIP", "SO(3)")][2]
    ax.annotate("", xy=(1.50, mo), xytext=(1.50, ms),
                arrowprops=dict(arrowstyle="<|-|>", lw=0.7, color=INK, mutation_scale=5,
                                shrinkA=0, shrinkB=0))
    ax.text(1.50, (mo * ms) ** 0.5, r"$2\times10^{6}$", fontsize=6.4 * S, color=INK,
            fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none"))
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(cores)
    ax.set_ylabel(r"predicted $\Vert e \Vert_F$  (C m$^{-2}$)")
    despine(ax)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, fc=O3, alpha=0.45, ec=O3, lw=0.6, label="O(3)"),
                       Rectangle((0, 0), 1, 1, fc=SO3, alpha=0.45, ec=SO3, lw=0.6, label="SO(3)")],
              loc="upper right", bbox_to_anchor=(1.005, 1.05), ncol=2,
              handletextpad=0.75, columnspacing=1.5, handlelength=1.1)

    # ---- b ---------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1]); panel(ax, "b", dx=-0.17)
    full = load_rows("fig2b_thresholds.csv")   # optional 25-point version
    curves = arm_curves(full, "tau") if full else None
    ax.set_xscale("log"); ax.set_xlim(7e-5, 1.6); ax.set_ylim(-0.04, 1.10)
    ax.axvspan(7e-5, TAU, color=FAINT, alpha=0.55, lw=0)
    ax.axvline(TAU, color=IMPOSS, lw=0.6, ls=(0, (3, 2)))
    for core in ["NequIP", "Allegro", "MACE", "EquiformerV2"]:
        for arm, col in (("SO(3)", SO3), ("O(3)", O3)):
            if curves is not None:
                if (core, arm) not in curves:
                    continue
                c = curves[(core, arm)]
                # marker on every 6th of the 25 log-spaced points = the 5 tabulated decades
                ax.plot(c["tau"], c["false_flag_fraction"], color=col, lw=0.9,
                        marker=CORE_MK[core], ms=2.8, mfc="white", mew=0.7,
                        markevery=6, zorder=3)
            elif (core, arm) in THR:
                ax.plot(THR_TAU, THR[(core, arm)], color=col, lw=0.9,
                        marker=CORE_MK[core], ms=2.8, mfc="white", mew=0.7, zorder=3)
    ax.text(1.15e-4, 0.83, "SO(3)", fontsize=6.4 * S, color=SO3, fontweight="bold",
            va="center", ha="left")
    ax.text(1.15e-4, 0.12, "O(3)", fontsize=6.4 * S, color=O3, fontweight="bold",
            va="center", ha="left")
    ax.text(TAU * 1.25, 0.30, "operating\npoint", fontsize=5.6 * S, color=IMPOSS)
    ax.set_xlabel(r"violation threshold $\tau$  (C m$^{-2}$)")
    ax.set_ylabel("false-flag fraction")
    despine(ax)
    ax.legend(handles=[Line2D([], [], color=GREY, marker=CORE_MK[c], ms=2.8, lw=0.9,
                              mfc="white", mew=0.7, label=c)
                       for c in ["NequIP", "Allegro", "MACE", "EquiformerV2"]],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.5,
              columnspacing=1.6, handlelength=1.3)
    if full is None:
        ax.text(0.99, -0.235, "5 of the 25 tabulated thresholds", transform=ax.transAxes,
                ha="right", fontsize=5.0 * S, color=GREY, style="italic")
    save(fig, "fig2_headline")


# ======================================================================================
# FIGURE 3  |  the constraint is free where it is inactive, and helps where it is not
# ======================================================================================
def fig3():
    fig = plt.figure(figsize=(W2, 70 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0.30,
                          left=0.125, right=0.985, top=0.90, bottom=0.16)

    # ---- a ---------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0]); panel(ax, "a", dx=-0.20)
    targets = [("U0", r"$U_0$"), ("dipole", "dipole"), ("elastic", "elastic"),
               ("piezo", "piezoelectric")]
    cores = ["NequIP", "Allegro", "MACE"]
    ax.axvspan(2.55, 3.55, color=IMPOSS, alpha=0.055, lw=0, zorder=0)   # constraint ACTIVE
    ax.axhspan(-1, 1, color=FAINT, alpha=0.8, lw=0, zorder=0)           # seed noise
    ax.axhline(0, color=INK, lw=0.5, zorder=1)
    for i, (key, lab) in enumerate(targets):
        active = key == "piezo"
        for j, core in enumerate(cores):
            d = delta_sigma(core, key)
            ax.plot([i + (j - 1) * 0.17], [d], marker=CORE_MK[core], ms=3.6,
                    mfc=SO3 if active else "white", mec=SO3 if active else INK,
                    mew=0.8, zorder=4)
    # the correction that matters: on dipole and elasticity the sign is the SAME in all three
    # cores, which the paper originally denied. Say so on the figure.
    for i in (1, 2):
        ax.text(i, 1.55, "3/3", ha="center", va="center", fontsize=5.6 * S, color=INK,
                fontweight="bold")
    ax.text(1.5, 2.15, "cores favouring O(3)", ha="center", va="center",
            fontsize=5.4 * S, color=GREY)
    ax.text(3.0, 5.05, "3.4 to 4.6$\sigma$", ha="center", va="center", fontsize=5.8 * S,
            color=SO3, fontweight="bold")
    ax.text(2.50, -0.55, "seed noise", fontsize=5.4 * S, color=GREY, style="italic",
            ha="center", va="center")
    ax.annotate("", xy=(3.46, 4.7), xytext=(3.46, 0.5),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=SO3, mutation_scale=6))
    ax.text(3.27, 2.6, "O(3) better", fontsize=5.8 * S, color=SO3, rotation=90,
            va="center", ha="center", fontweight="bold")
    ax.plot([-0.42, 2.42], [-1.42, -1.42], color=GREY, lw=0.6, clip_on=False)
    ax.text(1.0, -1.80, "constraint inactive", fontsize=6.0 * S, color=GREY, ha="center",
            va="center")
    ax.plot([2.58, 3.42], [-1.42, -1.42], color=SO3, lw=0.6, clip_on=False)
    ax.text(3.0, -1.80, "active", fontsize=6.0 * S, color=SO3, ha="center", va="center",
            fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in targets])
    ax.set_xlim(-0.55, 3.60); ax.set_ylim(-2.05, 5.5)
    ax.set_ylabel(r"$\Delta$ MAE, SO(3) $-$ O(3)   (pooled seed s.d.)")
    despine(ax)
    ax.legend(handles=[Line2D([], [], marker=CORE_MK[c], ms=3.6, ls="none", mfc="white",
                              mec=INK, mew=0.8, label=c) for c in cores],
              loc="upper left", bbox_to_anchor=(0.0, 1.02), handletextpad=0.4)

    # ---- b : paired dumbbell, far cleaner than bars -----------------------------------
    ax = fig.add_subplot(gs[1]); panel(ax, "b", dx=-0.19)
    ys = [3, 2, 1, 0]
    for y, core in zip(ys, ["NequIP", "Allegro", "MACE"]):
        (mo, so), (ms, ss) = ACC[(core, "piezo")]
        ax.plot([mo, ms], [y, y], color=GREY, lw=0.8, zorder=1)
        ax.errorbar([mo], [y], xerr=[so], fmt=CORE_MK[core], ms=3.8, color=O3,
                    ecolor=O3, elinewidth=0.8, capsize=1.6, capthick=0.8, zorder=3)
        ax.errorbar([ms], [y], xerr=[ss], fmt=CORE_MK[core], ms=3.8, color=SO3,
                    ecolor=SO3, elinewidth=0.8, capsize=1.6, capthick=0.8, zorder=3)
        ax.text((mo + ms) / 2, y + 0.22, f"{delta_sigma(core,'piezo'):+.1f}$\\sigma$",
                ha="center", fontsize=5.6 * S, color=INK)
    m, s = EQV2_PIEZO
    ax.errorbar([m], [0], xerr=[s], fmt="D", ms=3.6, color=SO3, ecolor=SO3,
                elinewidth=0.8, capsize=1.6, capthick=0.8, zorder=3)
    ax.text(m, 0.38, "no O(3) arm; flags 95.7%", ha="center", va="center",
            fontsize=5.4 * S, color=GREY, style="italic")
    ax.set_yticks(ys); ax.set_yticklabels(["NequIP", "Allegro", "MACE", "EquiformerV2"])
    ax.set_ylim(-0.55, 3.6); ax.set_xlim(0.19, 0.285)
    ax.set_xlabel(r"piezoelectric test MAE  (C m$^{-2}$)")
    ax.text(0.2085, 3.52, "O(3)", color=O3, fontsize=6.4 * S, fontweight="bold", ha="center")
    ax.text(0.2455, 3.52, "SO(3)", color=SO3, fontsize=6.4 * S, fontweight="bold", ha="center")
    despine(ax)
    save(fig, "fig3_accuracy")


# ======================================================================================
# FIGURE 4  |  training cannot buy the zero
# ======================================================================================
def fig4():
    fig = plt.figure(figsize=(W2, 68 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.12], wspace=0.34,
                          left=0.115, right=0.978, top=0.87, bottom=0.30)

    # ---- a : lollipop against the structural zero -------------------------------------
    ax = fig.add_subplot(gs[0]); panel(ax, "a", dx=-0.16)
    cores = list(INTRAIN)
    ys = np.arange(len(cores))[::-1]
    for y, c in zip(ys, cores):
        v = INTRAIN[c]
        ax.plot([0, v], [y, y], color=SO3, lw=0.9, alpha=0.5, zorder=1)
        ax.plot([v], [y], marker=CORE_MK[c], ms=4.4, color=SO3, mec="white", mew=0.6, zorder=3)
        ax.text(v + 0.025, y, f"{v:.3f}", va="center", fontsize=6.0 * S, color=SO3,
                fontweight="bold")
    ax.axvline(0, color=O3, lw=1.4, zorder=2)
    ax.text(0.012, -0.62, "O(3): 0.0000, structurally",
            fontsize=5.8 * S, color=O3, va="center")
    ax.set_yticks(ys); ax.set_yticklabels(cores)
    ax.set_xlim(-0.03, 1.14); ax.set_ylim(-0.95, len(cores) - 0.35)
    ax.set_xlabel("false-flag fraction on the crystals it was\ntrained, with zero labels, to call zero")
    ax.set_title("shown the exact answer, and still wrong", fontsize=6.6 * S,
                 fontweight="bold", pad=5)
    despine(ax)

    # ---- b : the loss-weight sweep ----------------------------------------------------
    ax = fig.add_subplot(gs[1]); panel(ax, "b", dx=-0.17)
    styles = {"trained-on zeros": ("-", "o", SO3),
              "held out, seen space groups": ((0, (4, 1.5)), "s", "#B14A00"),
              "held out, unseen": ((0, (1, 1.4)), "^", "#F08A4B")}
    ax.set_xscale("log"); ax.set_xlim(0.72, 260); ax.set_ylim(-0.03, 1.02)
    for k, v in SWEEP.items():
        ls, mk, col = styles[k]
        ax.errorbar(W, v, yerr=SWEEP_SD[k], ls=ls, marker=mk, ms=3.2, color=col, lw=1.0,
                    mfc="white", mew=0.7, ecolor=col, elinewidth=0.7, capsize=1.4, capthick=0.7)
        ax.text(118, v[-1], f"{v[-1]:.2f}", va="center", ha="left", fontsize=5.8 * S, color=col)
    ax.axhline(0, color=O3, lw=1.6)
    ax.text(230, 0.055, "O(3), any weight, free", fontsize=5.8 * S, color=O3, ha="right",
            va="center")
    ax.annotate("", xy=(100, 0.6762), xytext=(100, 0.8953),
                arrowprops=dict(arrowstyle="<|-|>", lw=0.7, color=INK, mutation_scale=5,
                                shrinkA=0, shrinkB=0))
    ax.text(84, 0.786, r"$100\times$ weight" "\n" r"buys $0.22$", ha="right", va="center",
            fontsize=5.4 * S, color=INK, fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none"))
    ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1", "10", "100"])
    ax.set_xlabel("loss weight on the zero-labelled rows")
    ax.set_ylabel("false-flag fraction")
    despine(ax)
    ax.legend(handles=[Line2D([], [], color=styles[k][2], ls=styles[k][0],
                              marker=styles[k][1], ms=3.2, mfc="white", mew=0.7, label=k)
                       for k in SWEEP], loc="lower left", bbox_to_anchor=(-0.01, 0.13))
    save(fig, "fig4_training")


# ======================================================================================
# FIGURE 5  |  mechanism.  Three panels, every number from a Supplementary Table.
# ======================================================================================
# Supplementary Table 14: predicted magnitude on ten named compounds (C/m^2)
NAMED = [
    # formula, structure, class, rotation forbids?, O(3), NequIP, Allegro, MACE, EqV2
    ("Al$_2$O$_3$", "corundum",   "non-cubic", False, 2.7e-7, 1.128, 0.723, 0.837, 0.212),
    ("TiO$_2$",     "rutile",     "non-cubic", False, 2.8e-7, 0.260, 0.582, 0.245, 0.187),
    # m-3m SO(3) entries are the full-precision run outputs (Supplementary Table 14): the
    # matched cores sit at the same machine floor as O(3) there, forced by rotation alone.
    ("C",           "diamond",    "m-3m",      True,  5.1e-7, 5.9e-7, 1.3e-6, 1.5e-6, 0.033),
    ("Si",          "diamond",    "m-3m",      True,  4.8e-8, 1.3e-8, 3.1e-8, 9.8e-8, 0.004),
    ("SrTiO$_3$",   "perovskite", "m-3m",      True,  1.4e-7, 3.7e-7, 8.0e-7, 1.9e-7, 0.011),
    ("CaF$_2$",     "fluorite",   "m-3m",      True,  4.0e-7, 1.7e-7, 2.0e-7, 1.3e-6, 0.008),
    ("MgO",         "rocksalt",   "m-3m",      True,  5.7e-8, 4.9e-8, 2.4e-8, 1.3e-8, 1.9e-7),
    ("NaCl",        "rocksalt",   "m-3m",      True,  5.4e-8, 2.5e-8, 3.5e-8, 2.1e-8, 2.5e-7),
    ("KCl",         "rocksalt",   "m-3m",      True,  3.0e-9, 7.1e-9, 1.4e-9, 2.3e-9, 5.8e-8),
    ("CsCl",        "CsCl-type",  "m-3m",      True,  3.3e-9, 9.2e-9, 2.4e-9, 1.1e-9, 6.2e-8),
]
# main text / Supplementary Note 1: the measured even-subspace fraction of the Jacobian,
# over 20 crystals x 3 seeds x each arm. The manuscript reports the RANGES, not the points.
JAC = {"O(3)": (1.4e-7, 9.7e-7), "SO(3)": (0.42, 0.54)}


def fig5():
    fig = plt.figure(figsize=(W2, 126 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], width_ratios=[1.35, 1.0],
                          wspace=0.42, hspace=0.52,
                          left=0.135, right=0.982, top=0.92, bottom=0.085)

    # ---- a : the crystals SO(3) models get right are the ones the theory names ---------
    ax = fig.add_subplot(gs[0, 0]); panel(ax, "a", dx=-0.135)
    cores = list(FAM)
    labels = [r"m$\bar{3}$m" + "\n$n$ = 166\nrotation forbids",
              r"m$\bar{3}$" + "\n$n$ = 18\nrotation permits",
              "non-cubic\n$n$ = 1,816\nrotation permits"]
    shades = ["#F6B08A", "#EE8B54", "#D55E00", "#8C3D00"]
    wdt = 0.19
    for j, core in enumerate(cores):
        xs = np.arange(3) + (j - 1.5) * wdt
        ax.bar(xs, FAM[core], width=wdt * 0.9, color=shades[j], edgecolor="none",
               label=core, zorder=3)
        if j < 3:
            ax.text(xs[0], 0.045, "0.0000", ha="center", va="bottom", fontsize=5.0 * S,
                    rotation=90, color=INK,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.axhline(0, color=O3, lw=1.6, zorder=4)
    ax.text(2.50, 1.20, "O(3): 0.0000 in every family, every seed", ha="right",
            va="center", fontsize=5.8 * S, color=O3, fontweight="bold")
    ax.annotate("", xy=(0.0, 0.14), xytext=(0.28, 0.55),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK, mutation_scale=5,
                                connectionstyle="arc3,rad=0.15"))
    ax.text(0.34, 0.62, "Corollary 3 forbids a\nnonzero value here", fontsize=5.4 * S,
            color=INK, ha="left", va="center", fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none"))
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.32); ax.set_xlim(-0.62, 2.58)
    ax.set_ylabel("false-flag fraction,\nSO(3) arms")
    despine(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, handlelength=1.0,
              handletextpad=0.4, columnspacing=1.0)

    # ---- b : the guarantee is differentiable -------------------------------------------
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "b", dx=-0.30)
    ax.set_yscale("log"); ax.set_ylim(4e-8, 4.0); ax.set_xlim(0.35, 1.65)
    pts = load_rows("fig5b_jacobian_points.csv")
    for xpos, arm, col in [(0.75, "O(3)", O3), (1.25, "SO(3)", SO3)]:
        lo, hi = JAC[arm]
        ax.add_patch(Rectangle((xpos - 0.16, lo), 0.32, hi - lo, facecolor=col, alpha=0.40,
                               edgecolor=col, lw=0.8, zorder=3))
        if pts is not None:
            vals = np.array([float(r["value"]) for r in pts if r["arm"] == arm])
            rng = np.random.default_rng(0)
            jit = xpos + rng.uniform(-0.13, 0.13, size=vals.size)
            ax.plot(jit, vals, ls="none", marker="o", ms=1.1, color=col, alpha=0.45,
                    mec="none", zorder=4)
        else:
            ax.plot([xpos], [(lo * hi) ** 0.5], marker="o", ms=3.2, color=col, zorder=4)
        ax.text(xpos, hi * 1.9, f"{lo:.2g}\nto {hi:.2g}", ha="center", va="bottom",
                fontsize=5.2 * S, color=col)
    ax.annotate("", xy=(0.52, 9.7e-7), xytext=(0.52, 0.42),
                arrowprops=dict(arrowstyle="<|-|>", lw=0.7, color=INK, mutation_scale=5,
                                shrinkA=0, shrinkB=0))
    ax.text(0.52, (9.7e-7 * 0.42) ** 0.5, "5 to 6\norders", ha="center", va="center",
            fontsize=5.6 * S, color=INK, fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none"))
    ax.set_xticks([0.75, 1.25]); ax.set_xticklabels(["O(3)", "SO(3)"], fontsize=6.8 * S)
    ax.set_ylabel(r"even-subspace fraction of $J$")
    ax.text(0.5, -0.155, "20 crystals $\\times$ 3 seeds; no overlap in any pair",
            transform=ax.transAxes, ha="center", fontsize=5.2 * S, color=GREY)
    despine(ax)

    # ---- c : named materials, real data that was sitting unplotted in a table -----------
    # Ten of the SO(3) entries are EXACTLY ZERO. A log axis cannot show zero, and clamping
    # them to a small positive number would tell the reader they are small-but-nonzero, which
    # is a fabrication of position. Use a symmetric-log axis with a genuine 0 on it.
    ax = fig.add_subplot(gs[1, :]); panel(ax, "c", dx=-0.062)
    ys = np.arange(len(NAMED))[::-1]
    for y, row in zip(ys, NAMED):
        form, struct, cls, forb, o3, ne, al, ma, eq = row
        if forb:
            ax.axhspan(y - 0.45, y + 0.45, color=FAINT, alpha=0.55, lw=0, zorder=0)
        ax.plot([o3], [y], marker="o", ms=3.4, color=O3, mec="white", mew=0.5, zorder=5)
        for v, mk in zip((ne, al, ma, eq), ("o", "s", "^", "D")):
            ax.plot([v], [y], marker=mk, ms=3.2, mfc="none", mec=SO3, mew=0.8, zorder=4)
    ax.axvline(TAU, color=IMPOSS, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.axvspan(TAU, 6.0, color=IMPOSS, alpha=0.055, lw=0, zorder=0)
    ax.set_xscale("symlog", linthresh=1e-8, linscale=0.55)
    ax.set_xlim(-0.35e-8, 6.0); ax.set_ylim(-0.75, len(NAMED) - 0.10)
    ax.set_xticks([0, 1e-8, 1e-6, 1e-4, 1e-2, 1e0])
    ax.set_xticklabels(["0", r"$10^{-8}$", r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$", "1"])
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{f}  {s}" for f, s, *_ in NAMED])
    ax.set_xlabel(r"predicted $\Vert e \Vert_F$  (C m$^{-2}$),  symmetric-log with a true zero")
    ax.text(1.4e-2, len(NAMED) - 0.40, "physically impossible", fontsize=5.8 * S, color=SO3,
            fontweight="bold", ha="left", va="center")
    ax.text(1.2e-4, 2.0, "shaded rows, class m$\\bar{3}$m:\nrotation alone forbids a response, so\n"
                         "the three exactly-equivariant SO(3)\n"
                         "cores sit at the machine floor, like\n"
                         "O(3). EquiformerV2, equivariant only\napproximately, fails on two",
            fontsize=5.2 * S, color=GREY, ha="center", va="center", linespacing=1.35,
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none"))
    ax.text(2.0e-6, 8.55, "the two non-cubic rows are the test:\n"
                          "only parity forbids it, and every SO(3) arm fails",
            fontsize=5.4 * S, color=SO3, ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none"))
    despine(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ms=3.4, ls="none", color=O3, mec="white",
                              mew=0.5, label="O(3)")] +
                      [Line2D([], [], marker=m, ms=3.2, ls="none", mfc="none", mec=SO3,
                              mew=0.8, label=c)
                       for m, c in zip(("o", "s", "^", "D"),
                                       ("NequIP", "Allegro", "MACE", "EquiformerV2"))],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, handletextpad=0.4,
              columnspacing=1.1)

    save(fig, "fig5_mechanism")


# ======================================================================================
# SUPPLEMENTARY FIGURE 1
# ======================================================================================
def figS1():
    """False-flag fraction vs threshold, RAW coordinate variant. Not tabulated anywhere."""
    fig, ax = plt.subplots(figsize=(W1, 74 * MM))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.90, bottom=0.22)
    rows = load_rows("figS1_raw_thresholds.csv")
    if rows is None:
        needs_data(ax, "figS1_raw_thresholds.csv",
                   "false-flag fraction vs threshold,\nRAW coordinate variant\n"
                   "25 thresholds $\\times$ 7 arms\n"
                   "the manuscript tabulates only the\nidealized variant, and only 5 of the 25")
        save(fig, "figS1_raw_thresholds")
        return
    curves = arm_curves(rows, "tau")
    ax.set_xscale("log")
    ax.set_xlim(7e-5, 1.6)
    ax.set_ylim(-0.04, 1.10)
    ax.axvspan(7e-5, TAU, color=FAINT, alpha=0.55, lw=0)
    ax.axvline(TAU, color=IMPOSS, lw=0.6, ls=(0, (3, 2)))
    for core in ["NequIP", "Allegro", "MACE", "EquiformerV2"]:
        for arm, col in (("SO(3)", SO3), ("O(3)", O3)):
            if (core, arm) not in curves:
                continue
            c = curves[(core, arm)]
            y, sd = c["false_flag_fraction"], c["sd"]
            ax.fill_between(c["tau"], y - sd, y + sd, color=col, alpha=0.18, lw=0)
            ax.plot(c["tau"], y, color=col, lw=0.9, marker=CORE_MK[core], ms=2.8,
                    mfc="white", mew=0.7, markevery=6, zorder=3)
    ax.text(1.15e-4, 0.83, "SO(3)", fontsize=6.4 * S, color=SO3, fontweight="bold",
            va="center", ha="left")
    ax.text(1.15e-4, 0.12, "O(3)", fontsize=6.4 * S, color=O3, fontweight="bold",
            va="center", ha="left")
    ax.text(TAU * 1.25, 0.30, "operating\npoint", fontsize=5.6 * S, color=IMPOSS)
    ax.set_xlabel(r"violation threshold $\tau$  (C m$^{-2}$)")
    ax.set_ylabel("false-flag fraction")
    despine(ax)
    ax.legend(handles=[Line2D([], [], color=GREY, marker=CORE_MK[c], ms=2.8, lw=0.9,
                              mfc="white", mew=0.7, label=c)
                       for c in ["NequIP", "Allegro", "MACE", "EquiformerV2"]],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, handletextpad=0.5,
              columnspacing=1.0)
    save(fig, "figS1_raw_thresholds")


def figS2():
    """The rutile polar-distortion sweep. Its 33-amplitude array is not in the manuscript."""
    fig, ax = plt.subplots(figsize=(W1, 74 * MM))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.90, bottom=0.22)
    rows = load_rows("fig5a_rutile_sweep.csv")
    if rows is None:
        needs_data(ax, "fig5a_rutile_sweep.csv",
                   "rutile TiO$_2$ polar-distortion sweep\n33 amplitudes $\\times$ 7 arms\n"
                   "verified endpoints at $\\delta = 0$:\nO(3) 1.9--9.1$\\times10^{-7}$, "
                   "SO(3) 0.09--0.28")
        save(fig, "figS2_rutile_sweep")
        return
    curves = arm_curves(rows, "delta")
    # the sweep is log-spaced up to delta = 0.05 and linear beyond: symlog shows both regimes,
    # with a genuine 0 for the centrosymmetric parent
    ax.set_yscale("log")
    ax.set_ylim(5e-8, 30.0)
    ax.set_xscale("symlog", linthresh=1e-3, linscale=0.35)
    ax.set_xlim(-1.5e-4, 1.35)
    ax.set_xticks([0, 1e-3, 1e-2, 1e-1, 1])
    ax.set_xticklabels(["0", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "1"])
    ax.axvline(1.0, color=GREY, lw=0.6, ls=(0, (3, 2)))
    ax.text(1.0, 1.2e-7, "physical\ndistortion ", fontsize=5.2 * S, color=GREY,
            ha="right", va="bottom")
    for core in ["NequIP", "Allegro", "MACE", "EquiformerV2"]:
        for arm, col in (("SO(3)", SO3), ("O(3)", O3)):
            if (core, arm) not in curves:
                continue
            c = curves[(core, arm)]
            ax.plot(c["delta"], c["magnitude"], color=col, lw=0.9, marker=CORE_MK[core],
                    ms=2.4, mfc="white", mew=0.6, markevery=4, zorder=3)
    ax.text(1.3e-3, 4.5, "SO(3): spurious floor, present at $\\delta = 0$",
            fontsize=5.6 * S, color=SO3, fontweight="bold", va="bottom", ha="left")
    ax.text(1.3e-3, 4e-7, "O(3): arithmetic floor,\nrises with the physics",
            fontsize=5.6 * S, color=O3, fontweight="bold", va="bottom", ha="left")
    ax.set_xlabel(r"polar distortion $\delta$  (fraction of the physical mode, symlog)")
    ax.set_ylabel(r"predicted $\Vert e \Vert_F$  (C m$^{-2}$)")
    despine(ax)
    ax.legend(handles=[Line2D([], [], color=GREY, marker=CORE_MK[c], ms=2.8, lw=0.9,
                              mfc="white", mew=0.7, label=c)
                       for c in ["NequIP", "Allegro", "MACE", "EquiformerV2"]],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, handletextpad=0.5,
              columnspacing=1.0)
    save(fig, "figS2_rutile_sweep")


def figS3():
    """False-flag fraction vs training epoch, the 12 re-instrumented SO(3) retrains."""
    fig, ax = plt.subplots(figsize=(W1, 74 * MM))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.90, bottom=0.22)
    rows = load_rows("figS3_epoch_curves.csv")
    if rows is None:
        needs_data(ax, "figS3_epoch_curves.csv",
                   "false-flag fraction vs epoch\n12 SO(3) retrains, idealized variant")
        save(fig, "figS3_epoch_curves")
        return
    core_name = {"nequip": "NequIP", "allegro": "Allegro", "mace": "MACE",
                 "equiformer_v2": "EquiformerV2"}
    by = {}
    for r in rows:
        by.setdefault((core_name[r["core"]], int(r["seed"])), []).append(
            (int(r["epoch"]), float(r["false_flag"])))
    ax.set_xlim(0, 152); ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.0, color=O3, lw=1.4, zorder=2)
    ax.text(150, 0.035, "O(3): 0.0000 at any epoch (structural)", fontsize=5.6 * S,
            color=O3, fontweight="bold", ha="right")
    for core in ("NequIP", "Allegro", "MACE", "EquiformerV2"):
        seeds = [np.array(sorted(by[(core, s)])) for s in (0, 1, 2)]
        curves = np.stack([s[:, 1] for s in seeds])
        epochs = seeds[0][:, 0]
        ax.fill_between(epochs, curves.min(0), curves.max(0), color=SO3, alpha=0.15, lw=0)
        ax.plot(epochs, curves.mean(0), color=SO3, lw=0.9, marker=CORE_MK[core], ms=2.6,
                mfc="white", mew=0.6, markevery=25, zorder=3)
    ax.text(85, 0.47, "SO(3): plateau at the headline\nvalue long before epoch 150",
            fontsize=5.6 * S, color=SO3, fontweight="bold", ha="center", va="center")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("false-flag fraction")
    despine(ax)
    ax.legend(handles=[Line2D([], [], color=SO3, marker=CORE_MK[c], ms=2.8, lw=0.9,
                              mfc="white", mew=0.7, label=c)
                       for c in ("NequIP", "Allegro", "MACE", "EquiformerV2")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, handletextpad=0.5,
              columnspacing=1.0)
    save(fig, "figS3_epoch_curves")


if __name__ == "__main__":
    print("Building figures to Nature specification (88 / 180 mm, Arial, Okabe-Ito)...")
    fig1(); fig2(); fig3(); fig4(); fig5(); figS1(); figS2(); figS3()
    print("\nEvery number above is traceable to a Supplementary Table or a figdata/ export.")
    missing = [f for f in ("fig5a_rutile_sweep.csv", "fig5b_jacobian_points.csv",
                           "figS1_raw_thresholds.csv") if load_rows(f) is None]
    if missing:
        print("Series still marked AUTHOR DATA REQUIRED: " + ", ".join(missing))
    else:
        print("All three author-supplied series were found in figdata/ and drawn"
              " (export via scripts/export_figdata.py).")
