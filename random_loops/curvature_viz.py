"""
curvature_viz.py

Forman-Ricci curvature: knotted vs unknotted proteins across chain lengths.

Usage
-----
    from curvature_viz import (
        plot_trend_line_mean, plot_trend_line_median,
        plot_trend_violin_mean, plot_trend_violin_median,
        run_all,
    )
    plot_trend_line_mean(Df)
    plot_trend_line_mean(Df, out_path="fig.pdf")
    run_all(Df, out_dir="plots/")

DataFrame columns
-----------------
    "Curvature"  float  – Forman-Ricci curvature per protein
    "Type"       str    – "knotted" and "unknotted"
    "Length"     int    – chain length (100, 150, ..., 500)
"""

from __future__ import annotations
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# ─────────────────────────────────────────────────────────────────────────────
# Aesthetics
# ─────────────────────────────────────────────────────────────────────────────

mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.alpha":        0.18,
    "grid.linewidth":    0.5,
    "axes.linewidth":    0.8,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

COLORS = {"knotted": "#E74C3C", "unknotted": "#2980B9"}
TYPES  = ["knotted", "unknotted"]
DPI    = 200



def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), format="pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved in {path}")

def _show_or_save(fig, out_path):
    _save(fig, out_path) if out_path else plt.show()

def _lengths(df, lengths, length_col):
    return sorted(df[length_col].unique()) if lengths is None else lengths

def _vals(df, L, t, curvature_col, type_col, length_col): # L in {100,150,...500}, t in ["knotted", "unknotted"],curvature_col=curvature value 
    return (df[(df[length_col] == L) & (df[type_col] == t)]
            [curvature_col].dropna().values)

def _boot_ci(vals, stat_fn, n_boot, ci, rng):
    boots = np.array([stat_fn(rng.choice(vals, len(vals), replace=True))
                      for _ in range(n_boot)])
    a = (100 - ci) / 2
    return float(np.percentile(boots, a)), float(np.percentile(boots, 100 - a))

def _compute_stats(df, lengths, curvature_col, type_col, length_col,
                   stat_fn, n_boot, ci):
    rng = np.random.default_rng(42)
    out = {}
    for t in TYPES:
        xs, centers, lo, hi, sd, all_vals = [], [], [], [], [], []
        for L in lengths:
            v = _vals(df, L, t, curvature_col, type_col, length_col)
            xs.append(L)
            all_vals.append(v)
            if len(v) < 4:
                for lst in (centers, lo, hi, sd):
                    lst.append(np.nan)
                continue
            centers.append(float(stat_fn(v)))
            sd.append(float(np.std(v, ddof=1)))
            l, h = _boot_ci(v, stat_fn, n_boot, ci, rng)
            lo.append(l); hi.append(h)
        out[t] = dict(
            xs=np.array(xs),
            centers=np.array(centers),
            lo=np.array(lo), hi=np.array(hi),
            sd=np.array(sd),
            all_vals=all_vals,
        )
    return out


# 1. plot_trend_line_mean


def plot_trend_line_mean(
    df: pd.DataFrame,
    lengths=None,
    curvature_col: str = "Curvature",
    type_col: str      = "Type",
    length_col: str    = "Length",
    n_boot: int        = 1000,
    ci: float          = 95.0,
    out_path=None,
) -> None:
    """
    Line plot - MEAN curvature vs chain length.
    Legend: Knotted / Unknotted only.
    """
    lengths = _lengths(df, lengths, length_col)
    S = _compute_stats(df, lengths, curvature_col, type_col,
                       length_col, np.mean, n_boot, ci)

    fig, ax = plt.subplots(figsize=(8, 5))
    dodge = {"knotted": -3.0, "unknotted": +3.0}

    for t in TYPES:
        col = COLORS[t]; s = S[t]; dg = dodge[t]
        # CI shade
        ax.fill_between(s["xs"], s["lo"], s["hi"], alpha=0.18, color=col)
        # ±1 SD error bars
        ax.errorbar(s["xs"] + dg, s["centers"],
                    yerr=s["sd"],
                    fmt="none", ecolor=col, elinewidth=1.4,
                    capsize=5, capthick=1.6, alpha=0.85, zorder=3)
        # trend line
        ax.plot(s["xs"], s["centers"], "-o", color=col, lw=2.2, ms=6,
                zorder=4, label=t.capitalize())

    ax.set_xlabel("Chain Length (residues)", fontsize=11)
    ax.set_ylabel("Mean Curvature", fontsize=11)
    ax.set_title("Mean Curvature vs Chain Length", fontsize=13, fontweight="bold")
    ax.set_xticks(lengths)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.legend(fontsize=10, framealpha=0.85)
    fig.tight_layout()
    _show_or_save(fig, out_path)



# 2. plot_trend_line_median


def plot_trend_line_median(
    df: pd.DataFrame,
    lengths=None,
    curvature_col: str = "Curvature",
    type_col: str      = "Type",
    length_col: str    = "Length",
    n_boot: int        = 1000,
    ci: float          = 95.0,
    out_path=None,
) -> None:
    """
    Line plot - MEDIAN curvature vs chain length.
    Legend: Knotted / Unknotted only.
    """
    lengths = _lengths(df, lengths, length_col)
    S = _compute_stats(df, lengths, curvature_col, type_col,
                       length_col, np.median, n_boot, ci)

    fig, ax = plt.subplots(figsize=(8, 5))
    dodge = {"knotted": -3.0, "unknotted": +3.0}

    for t in TYPES:
        col = COLORS[t]; s = S[t]; dg = dodge[t]
        ax.fill_between(s["xs"], s["lo"], s["hi"], alpha=0.18, color=col)
        ax.errorbar(s["xs"] + dg, s["centers"],
                    yerr=s["sd"],
                    fmt="none", ecolor=col, elinewidth=1.4,
                    capsize=5, capthick=1.6, alpha=0.85, zorder=3)
        ax.plot(s["xs"], s["centers"], "-o", color=col, lw=2.2, ms=6,
                zorder=4, label=t.capitalize())

    ax.set_xlabel("Chain Length (residues)", fontsize=11)
    ax.set_ylabel("Median Curvature", fontsize=11)
    ax.set_title("Median Curvature vs Chain Length", fontsize=13, fontweight="bold")
    ax.set_xticks(lengths)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.legend(fontsize=10, framealpha=0.85)
    fig.tight_layout()
    _show_or_save(fig, out_path)



# 3. plot_trend_violin_mean


def plot_trend_violin_mean(
    df: pd.DataFrame,
    lengths=None,
    curvature_col: str  = "Curvature",
    type_col: str       = "Type",
    length_col: str     = "Length",
    n_boot: int         = 1000,
    ci: float           = 95.0,
    violin_width: float = 16.0,
    bw_method: str      = "scott",
    out_path=None,
) -> None:
    """
    Violin spine - MEAN curvature vs chain length.
      - Connecting line: mean trend
    Legend: Knotted / Unknotted only.
    """
    lengths = _lengths(df, lengths, length_col)
    S = _compute_stats(df, lengths, curvature_col, type_col,
                       length_col, np.mean, n_boot, ci)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sign  = {"knotted": -1, "unknotted": +1}
    dodge = {"knotted": -violin_width * 0.4, "unknotted": +violin_width * 0.4}

    for t in TYPES:
        col = COLORS[t]; s = S[t]
        spine_pts = []

        for i, (L, vals) in enumerate(zip(lengths, s["all_vals"])):
            if len(vals) < 4:
                continue
            xc = float(L) + dodge[t]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kde = gaussian_kde(vals, bw_method=bw_method)
            yr = np.linspace(vals.min(), vals.max(), 300)
            d  = kde(yr); d = d / d.max() * violin_width * 0.85
            ax.fill_betweenx(yr, xc, xc + sign[t] * d, alpha=0.22, color=col)
            ax.plot(xc + sign[t] * d, yr, lw=0.7, color=col, alpha=0.5)

            mu = s["centers"][i]
            if not np.isfinite(mu):
                continue
            ax.plot([xc, xc], [s["lo"][i], s["hi"][i]],
                    color=col, lw=2.0, solid_capstyle="round", zorder=4)
            ax.scatter(xc, mu, color=col, s=55, zorder=6,
                       edgecolors="white", linewidths=0.9)
            spine_pts.append((xc, mu))

        if spine_pts:
            px, py = zip(*spine_pts)
            ax.plot(px, py, "-", color=col, lw=2.0, zorder=3,
                    label=t.capitalize())

    ax.set_xlabel("Chain Length (residues)", fontsize=11)
    ax.set_ylabel("Mean Curvature", fontsize=11)
    ax.set_title("Mean Curvature vs Chain Length", fontsize=13, fontweight="bold")
    ax.set_xticks(lengths)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.legend(fontsize=10, framealpha=0.85)
    fig.tight_layout()
    _show_or_save(fig, out_path)



# 4. plot_trend_violin_median


def plot_trend_violin_median(
    df: pd.DataFrame,
    lengths=None,
    curvature_col: str  = "Curvature",
    type_col: str       = "Type",
    length_col: str     = "Length",
    n_boot: int         = 1000,
    ci: float           = 95.0,
    violin_width: float = 16.0,
    bw_method: str      = "scott",
    out_path=None,
) -> None:
    """
    Violin spine - MEDIAN curvature vs chain length.
      - Connecting line: median trend
    Legend: Knotted / Unknotted only.
    """
    lengths = _lengths(df, lengths, length_col)
    S = _compute_stats(df, lengths, curvature_col, type_col,
                       length_col, np.median, n_boot, ci)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sign  = {"knotted": -1, "unknotted": +1}
    dodge = {"knotted": -violin_width * 0.4, "unknotted": +violin_width * 0.4}

    for t in TYPES:
        col = COLORS[t]; s = S[t]
        spine_pts = []

        for i, (L, vals) in enumerate(zip(lengths, s["all_vals"])):
            if len(vals) < 4:
                continue
            xc = float(L) + dodge[t]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kde = gaussian_kde(vals, bw_method=bw_method)
            yr = np.linspace(vals.min(), vals.max(), 300)
            d  = kde(yr); d = d / d.max() * violin_width * 0.85
            ax.fill_betweenx(yr, xc, xc + sign[t] * d, alpha=0.22, color=col)
            ax.plot(xc + sign[t] * d, yr, lw=0.7, color=col, alpha=0.5)

            med = s["centers"][i]
            if not np.isfinite(med):
                continue
            ax.plot([xc, xc], [s["lo"][i], s["hi"][i]],
                    color=col, lw=2.0, solid_capstyle="round", zorder=4)
            ax.scatter(xc, med, color=col, s=55, marker="D", zorder=6,
                       edgecolors="white", linewidths=0.9)
            spine_pts.append((xc, med))

        if spine_pts:
            px, py = zip(*spine_pts)
            ax.plot(px, py, "-", color=col, lw=2.0, zorder=3,
                    label=t.capitalize())

    ax.set_xlabel("Chain Length (residues)", fontsize=11)
    ax.set_ylabel("Median Curvature", fontsize=11)
    ax.set_title("Median Curvature vs Chain Length", fontsize=13, fontweight="bold")
    ax.set_xticks(lengths)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.legend(fontsize=10, framealpha=0.85)
    fig.tight_layout()
    _show_or_save(fig, out_path)




# ─────────────────────────────────────────────────────────────────────────────
# 5. plot_kde_by_length
# ─────────────────────────────────────────────────────────────────────────────

def plot_kde_by_length(
    df: pd.DataFrame,
    lengths=None,
    curvature_col: str = "Curvature",
    type_col: str      = "Type",
    length_col: str    = "Length",
    bw_method: str     = "scott",
    ncols: int         = 3,
    out_path=None,
) -> None:
    """
    KDE density curves - one subplot per chain length.

    Layout: 3 rows x ncols columns (default 3).
    Each subplot shows the KDE of knotted (red) and unknotted (blue)
    curvature values, with a light fill under each curve.
    Vertical dashed lines mark the group medians.
    """
    lengths = _lengths(df, lengths, length_col)
    n       = len(lengths)
    nrows   = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.8, nrows * 3.2),
        sharey=False,
    )
    axes_flat = np.array(axes).flatten()

    for idx, L in enumerate(lengths):
        ax = axes_flat[idx]

        for t in TYPES:
            col  = COLORS[t]
            vals = _vals(df, L, t, curvature_col, type_col, length_col)
            if len(vals) < 3:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kde = gaussian_kde(vals, bw_method=bw_method)
            xs = np.linspace(vals.min(), vals.max(), 400)
            ys = kde(xs)
            ax.plot(xs, ys, color=col, lw=2.0, label=t.capitalize())
            ax.fill_between(xs, ys, alpha=0.10, color=col)
            # median dashed line
            med = float(np.median(vals))
            ax.axvline(med, color=col, lw=1.1, ls="--", alpha=0.70)

        ax.set_title(f"L = {L}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Curvature", fontsize=8)
        ax.set_ylabel("Density",   fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.15, lw=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # hide unused axes
    for idx in range(len(lengths), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # shared legend in the first axis
    axes_flat[0].legend(fontsize=8, framealpha=0.85)

    fig.suptitle("KDE - Curvature Distribution per Chain Length",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _show_or_save(fig, out_path) 


# ─────────────────────────────────────────────────────────────────────────────
# 6. plot_violin_by_length
# ─────────────────────────────────────────────────────────────────────────────

def plot_violin_by_length(
    df: pd.DataFrame,
    lengths=None,
    curvature_col: str = "Curvature",
    type_col: str      = "Type",
    length_col: str    = "Length",
    ncols: int         = 3,
    out_path=None,
) -> None:
    """
    Violin + box - one subplot per chain length.

    Layout: 2 rows x ncols columns (default 4).
    Each subplot shows a violin + inner box for knotted (red) and
    unknotted (blue). The black horizontal bar is the median.
    """
    lengths = _lengths(df, lengths, length_col)
    n       = len(lengths)
    nrows   = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.2, nrows * 3.5),
        sharey=False,
    )
    axes_flat = np.array(axes).flatten()

    for idx, L in enumerate(lengths):
        ax = axes_flat[idx]

        groups = [
            _vals(df, L, t, curvature_col, type_col, length_col)
            for t in TYPES
        ]
        colors = [COLORS[t] for t in TYPES]

        # --- violin ---
        vp = ax.violinplot(
            [g for g in groups if len(g) >= 3],
            positions=[i for i, g in enumerate(groups) if len(g) >= 3],
            widths=0.55, showmeans=False, showmedians=False,
        )
        valid_bodies = [i for i, g in enumerate(groups) if len(g) >= 3]
        for body_idx, grp_idx in enumerate(valid_bodies):
            vp["bodies"][body_idx].set_facecolor(colors[grp_idx])
            vp["bodies"][body_idx].set_alpha(0.25)
        for key in ("cmins", "cmaxes", "cbars"):
            if key in vp:
                vp[key].set_visible(False)

        # --- box ---
        valid_data  = [g for g in groups if len(g) >= 3]
        valid_pos   = [i for i, g in enumerate(groups) if len(g) >= 3]
        valid_colors = [colors[i] for i in valid_pos]

        if valid_data:
            bp = ax.boxplot(
                valid_data,
                positions=valid_pos,
                widths=0.28,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker=".", markersize=2.5, alpha=0.30,
                                markerfacecolor="grey", markeredgewidth=0),
                medianprops=dict(color="black", lw=1.8),
            )
            for bi, col in enumerate(valid_colors):
                bp["boxes"][bi].set_facecolor(col)
                bp["boxes"][bi].set_alpha(0.75)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([t.capitalize() for t in TYPES], fontsize=7.5)
        ax.set_title(f"L = {L}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Curvature", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.15, lw=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # hide unused axes
    for idx in range(len(lengths), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # shared colour legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COLORS[t], alpha=0.75, label=t.capitalize())
        for t in TYPES
    ]
    fig.legend(handles=legend_handles, fontsize=9, framealpha=0.85,
               loc="lower right", ncol=2)

    fig.suptitle("Violin + Box - Curvature per Chain Length",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _show_or_save(fig, out_path)


# run_all


def run_all(df, lengths=None, out_dir=None):
    """Run all four plots. Pass out_dir='plots' to save as PDFs."""
    def _p(name):
        return (Path(out_dir) / f"{name}.pdf") if out_dir else None
    print("1/4  trend line   - mean ...");   plot_trend_line_mean(df,      lengths=lengths, out_path=_p("trend_line_mean"))
    print("2/4  trend line   - median ..."); plot_trend_line_median(df,    lengths=lengths, out_path=_p("trend_line_median"))
    print("3/4  trend violin - mean ...");   plot_trend_violin_mean(df,    lengths=lengths, out_path=_p("trend_violin_mean"))
    print("4/4  trend violin - median ..."); plot_trend_violin_median(df,  lengths=lengths, out_path=_p("trend_violin_median"))
    print("5/6  KDE by length ...");            plot_kde_by_length(df,          lengths=lengths, out_path=_p("kde_by_length"))
    print("6/6  violin by length ...");         plot_violin_by_length(df,       lengths=lengths, out_path=_p("violin_by_length"))
    print("Done.")



# CLI demo   python curvature_viz.py --out_dir plots/

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="plots_clean")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    lengths = [100, 150, 200, 250, 300, 350, 400, 450, 500]
    rows = []
    for L in lengths:
        n = 500
        k = rng.normal(loc=-0.30 - L * 0.0005, scale=0.30 + L * 0.0003, size=n)
        u = rng.normal(loc=-0.10 - L * 0.0002, scale=0.40 + L * 0.0002, size=n)
        for v in k: rows.append({"Curvature": v, "Type": "knotted",   "Length": L})
        for v in u: rows.append({"Curvature": v, "Type": "unknotted", "Length": L})

    Df = pd.DataFrame(rows)
    run_all(Df, out_dir=args.out_dir)
