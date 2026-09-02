"""
curvature_viz.py

Forman-Ricci curvature and H1 median persistence: knotted vs unknotted proteins
across chain lengths.

Usage
-----
    from curvature_viz import (
        plot_trend_line_mean, plot_trend_line_median,
        plot_trend_violin_mean, plot_trend_violin_median,
        plot_trend_boxplot_mean,                   # boxplot replacements for
        plot_trend_boxplot_median,                 # the error-bar trend figures
        plot_trend_boxplot_median_persistence,     # (mean + median, clearly titled)
        plot_ks_by_length,                         # Figure 6: KS D vs chain length
        plot_ks_permutation_by_length,             # Figure 6 (ties-safe permutation p)
        plot_trend_line_median_persistence,
        plot_trend_violin_median_persistence,
        run_all,
    )
    plot_trend_line_mean(Df)
    plot_trend_line_mean(Df, out_path="fig.pdf")
    run_all(Df, out_dir="plots/")

DataFrame columns
-----------------
    "Curvature"   float  – Forman-Ricci curvature per protein
    "Persistence" float  – H1 median persistence per protein
    "Type"        str    – "knotted" and "unknotted"
    "Length"      int    – chain length (100, 150, ..., 500)
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

# Aesthetics

mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset":   "cm",
    "axes.formatter.use_mathtext": True,
    "font.size":          11,
    "axes.titlesize":     11,
    "axes.labelsize":     11,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.alpha":         0.18,
    "grid.linewidth":     0.5,
    "axes.linewidth":     0.8,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

FS       = 11   # axis labels, tick labels
FS_TITLE = 11   # plot titles
FS_LEG   = 10   # legends

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

def _vals(df, L, t, curvature_col, type_col, length_col):
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
    curvature_col: str = "Curvature_mean",
    type_col: str      = "Type",
    length_col: str    = "Length",
    n_boot: int        = 1000,
    ci: float          = 95.0,
    out_path=None,
) -> None:
    """Line plot - per-loop MEAN curvature vs chain length (mean across loops)."""
    lengths = _lengths(df, lengths, length_col)
    S = _compute_stats(df, lengths, curvature_col, type_col,
                       length_col, np.mean, n_boot, ci)

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

    ax.set_xlabel("Chain Length (residues)", fontsize=FS)
    ax.set_ylabel("Mean Curvature", fontsize=FS)
    ax.set_title("Mean Curvature vs Chain Length", fontsize=FS_TITLE, fontweight="bold")
    ax.set_xticks(lengths)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.legend(fontsize=FS_LEG, framealpha=0.85)
    fig.tight_layout()
    _show_or_save(fig, out_path)


# 2. plot_trend_line_median


def plot_trend_line_median(
    df: pd.DataFrame,
    lengths=None,
    curvature_col: str = "Curvature_median",
    type_col: str      = "Type",
    length_col: str    = "Length",
    n_boot: int        = 1000,
    ci: float          = 95.0,
    out_path=None,
) -> None:
    """Line plot - per-loop MEDIAN curvature vs chain length (median across loops;
    this is the paper's Figure 5 quantity: 'median of the per-loop medians')."""
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

    ax.set_xlabel("Chain Length (residues)", fontsize=FS)
    ax.set_ylabel("Median Curvature", fontsize=FS)
    ax.set_title("Median Curvature vs Chain Length", fontsize=FS_TITLE, fontweight="bold")
    ax.set_xticks(lengths)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.legend(fontsize=FS_LEG, framealpha=0.85)
    fig.tight_layout()
    _show_or_save(fig, out_path)


# 2b. plot_trend_boxplot  (grouped boxplot vs chain length)


def _trend_boxplot(
    df, lengths, value_col, type_col, length_col,
    ylabel, title, out_path,
    box_width=16.0, dodge=11.0, connect_medians=True,
):
    """
    Shared engine for the grouped-boxplot trend figures.

    For every chain length, two boxplots (knotted / unknotted) are drawn side by
    side, so the full distribution of the per-loop values at each length is
    visible: median, inter-quartile range, whiskers and outliers. This is the
    reviewer-requested replacement for the line-plus-error-bar figure, whose
    error bars (±1 SD) were ambiguous. A boxplot is the standard way to compare
    the two populations without hiding the shape of the data.
    """
    sign = {"knotted": -1.0, "unknotted": +1.0}

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    from matplotlib.patches import Patch

    for t in TYPES:
        col = COLORS[t]
        positions, data, med_pts = [], [], []
        for L in lengths:
            v = _vals(df, L, t, value_col, type_col, length_col)
            if len(v) < 4:
                continue
            pos = float(L) + sign[t] * dodge
            positions.append(pos)
            data.append(v)
            med_pts.append((pos, float(np.median(v))))
        if not data:
            continue

        bp = ax.boxplot(
            data, positions=positions, widths=box_width,
            patch_artist=True, showfliers=True, manage_ticks=False,
            flierprops=dict(marker=".", markersize=2.5, alpha=0.30,
                            markerfacecolor="grey", markeredgewidth=0),
            medianprops=dict(color="black", lw=1.6),
            whiskerprops=dict(color=col, lw=1.0),
            capprops=dict(color=col, lw=1.0),
        )
        for box in bp["boxes"]:
            box.set_facecolor(col)
            box.set_alpha(0.55)
            box.set_edgecolor(col)

        # faint trend line through the medians so the length dependence stays readable
        if connect_medians and len(med_pts) > 1:
            px, py = zip(*med_pts)
            ax.plot(px, py, "-", color=col, lw=1.4, alpha=0.55, zorder=1)

    ax.set_xlabel("Chain Length (residues)", fontsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
    ax.set_xticks(list(lengths))
    ax.set_xticklabels([str(L) for L in lengths])
    ax.set_xlim(min(lengths) - 50, max(lengths) + 50)
    ax.grid(axis="y", alpha=0.15, lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [Patch(facecolor=COLORS[t], alpha=0.55, edgecolor=COLORS[t],
                     label=t.capitalize()) for t in TYPES]
    ax.legend(handles=handles, fontsize=FS_LEG, framealpha=0.85)
    fig.tight_layout()
    _show_or_save(fig, out_path)


def plot_trend_boxplot_mean(
    df: pd.DataFrame,
    lengths=None,
    curvature_col: str = "Curvature_mean",
    type_col: str      = "Type",
    length_col: str    = "Length",
    out_path=None,
    **kwargs,
) -> None:
    """Grouped boxplot - per-loop MEAN curvature vs chain length. Boxplot
    replacement for the mean error-bar figure; knotted vs unknotted per L."""
    lengths = _lengths(df, lengths, length_col)
    _trend_boxplot(df, lengths, curvature_col, type_col, length_col,
                   ylabel="Mean Curvature",
                   title="Mean Curvature vs Chain Length",
                   out_path=out_path, **kwargs)


def plot_trend_boxplot_median(
    df: pd.DataFrame,
    lengths=None,
    curvature_col: str = "Curvature_median",
    type_col: str      = "Type",
    length_col: str    = "Length",
    out_path=None,
    **kwargs,
) -> None:
    """Grouped boxplot - per-loop MEDIAN curvature vs chain length. Boxplot
    replacement for the median error-bar figure (paper's Figure 5 quantity);
    knotted vs unknotted side by side per L."""
    lengths = _lengths(df, lengths, length_col)
    _trend_boxplot(df, lengths, curvature_col, type_col, length_col,
                   ylabel="Median Curvature",
                   title="Median Curvature vs Chain Length",
                   out_path=out_path, **kwargs)


def plot_trend_boxplot_median_persistence(
    df: pd.DataFrame,
    lengths=None,
    persistence_col: str = "Persistence",
    type_col: str        = "Type",
    length_col: str      = "Length",
    out_path=None,
    **kwargs,
) -> None:
    """Grouped boxplot - H1 persistence vs chain length (boxplot replacement for
    the persistence error-bar trend figure). Knotted vs unknotted per L."""
    lengths = _lengths(df, lengths, length_col)
    _trend_boxplot(df, lengths, persistence_col, type_col, length_col,
                   ylabel="H1 Persistence",
                   title="H1 Persistence vs Chain Length",
                   out_path=out_path, **kwargs)


# 2c. plot_ks_by_length  (Figure 6: KS statistic vs chain length)


def _ks_stars(p):
    """Significance stars matching the Figure 6 caption
    (ns p>=0.05, * p<0.05, ** p<0.01, *** p<0.001)."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def plot_ks_by_length(
    df: pd.DataFrame,
    lengths=None,
    curvature_col: str = "Curvature_median",
    type_col: str      = "Type",
    length_col: str    = "Length",
    alpha: float       = 0.05,
    out_path=None,
) -> None:
    """
    Figure 6: two-sample Kolmogorov-Smirnov D-statistic between the knotted and
    unknotted per-loop curvature distributions at each chain length.

    p-values are Benjamini-Hochberg FDR-corrected across all chain lengths
    simultaneously (as stated in the paper). Bars are green where the corrected
    p-value < alpha, grey otherwise, with ns / * / ** / *** annotations.

    Uses the per-loop MEDIAN curvature by default so it is consistent with the
    median-based Figure 5.
    """
    from scipy.stats import ks_2samp
    try:
        from statsmodels.stats.multitest import multipletests
        _have_sm = True
    except Exception:
        multipletests = None
        _have_sm = False

    lengths = _lengths(df, lengths, length_col)

    D_stats, p_raw, ns_pairs = [], [], []
    for L in lengths:
        a = _vals(df, L, "knotted",   curvature_col, type_col, length_col)
        b = _vals(df, L, "unknotted", curvature_col, type_col, length_col)
        if len(a) < 3 or len(b) < 3:
            D_stats.append(np.nan); p_raw.append(np.nan)
            ns_pairs.append((len(a), len(b)))
            continue
        d, p = ks_2samp(a, b, alternative="two-sided")
        D_stats.append(float(d)); p_raw.append(float(p))
        ns_pairs.append((len(a), len(b)))

    D_stats = np.array(D_stats)
    p_raw   = np.array(p_raw)

    # BH-FDR across all chain lengths simultaneously
    p_fdr = np.full_like(p_raw, np.nan)
    mask  = ~np.isnan(p_raw)
    if mask.any():
        if _have_sm:
            _, p_corr, _, _ = multipletests(p_raw[mask], alpha=alpha, method="fdr_bh")
        else:
            # manual Benjamini-Hochberg fallback
            pv = p_raw[mask]
            order = np.argsort(pv)
            m = len(pv)
            corr = np.empty(m)
            prev = 1.0
            for rank in range(m - 1, -1, -1):
                idx = order[rank]
                val = pv[idx] * m / (rank + 1)
                prev = min(prev, val)
                corr[idx] = prev
            p_corr = corr
        p_fdr[mask] = p_corr

    green, grey = "#27AE60", "#95A5A6"
    x = np.arange(len(lengths))

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [green if (not np.isnan(p) and p < alpha) else grey for p in p_fdr]
    bars = ax.bar(x, np.nan_to_num(D_stats), color=colors, edgecolor="white",
                  linewidth=0.6, width=0.72, zorder=3)

    for xi, d, p in zip(x, D_stats, p_fdr):
        if np.isnan(d):
            continue
        ax.text(xi, d + 0.006, _ks_stars(p) if not np.isnan(p) else "",
                ha="center", va="bottom", fontsize=FS_LEG)

    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in lengths])
    ax.set_xlabel("Chain Length", fontsize=FS)
    ax.set_ylabel("KS Statistic", fontsize=FS)
    ax.set_title("Kolmogorov-Smirnov Statistic (Knotted vs Unknotted)",
                 fontsize=FS_TITLE, fontweight="bold")
    ax.grid(axis="y", alpha=0.15, lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    # mathtext for the comparison operators: Computer Modern's *text* font renders
    # "<" as "¡" and has no "≥" glyph, so render them in math mode instead.
    handles = [Patch(facecolor=green, label=rf"$p < {alpha:g}$"),
               Patch(facecolor=grey,  label=rf"$p \geq {alpha:g}$")]
    ax.legend(handles=handles, fontsize=FS_LEG, framealpha=0.85, loc="upper left")
    fig.tight_layout()
    _show_or_save(fig, out_path)


def plot_ks_permutation_by_length(
    df: pd.DataFrame,
    lengths=None,
    curvature_col: str = "Curvature_median",
    type_col: str      = "Type",
    length_col: str    = "Length",
    n_perm: int        = 5000,
    alpha: float       = 0.05,
    seed: int          = 0,
    out_path=None,
) -> None:
    """
    Figure 6 (permutation variant, ties-safe).

    Bar height is still the two-sample Kolmogorov-Smirnov D-statistic between the
    knotted and unknotted per-loop curvature distributions at each chain length,
    so it is directly comparable to the standard KS panel. The p-value, however,
    is obtained by a LABEL-PERMUTATION test (shuffle the knotted/unknotted labels
    `n_perm` times and count how often the permuted D >= observed D), then
    BH-FDR-corrected across chain lengths.

    This is the appropriate significance test when the statistic is the per-loop
    MEDIAN curvature: the median lands on a small set of integer/half-integer
    values (heavy ties), which violates the continuous-distribution assumption of
    the asymptotic KS p-value. The permutation null makes no such assumption and
    is exact up to Monte-Carlo error, so it stays valid under ties.
    """
    from scipy.stats import ks_2samp
    try:
        from statsmodels.stats.multitest import multipletests
        _have_sm = True
    except Exception:
        multipletests = None
        _have_sm = False

    lengths = _lengths(df, lengths, length_col)
    rng = np.random.default_rng(seed)

    D_stats, p_perm = [], []
    for L in lengths:
        a = _vals(df, L, "knotted",   curvature_col, type_col, length_col)
        b = _vals(df, L, "unknotted", curvature_col, type_col, length_col)
        if len(a) < 3 or len(b) < 3:
            D_stats.append(np.nan); p_perm.append(np.nan)
            continue
        d_obs = float(ks_2samp(a, b, alternative="two-sided").statistic)
        pool  = np.concatenate([a, b])
        na    = len(a)
        count = 0
        for _ in range(n_perm):
            rng.shuffle(pool)
            d = ks_2samp(pool[:na], pool[na:], alternative="two-sided").statistic
            if d >= d_obs:
                count += 1
        # add-one correction keeps the permutation p-value valid (never exactly 0)
        p = (count + 1) / (n_perm + 1)
        D_stats.append(d_obs); p_perm.append(p)

    D_stats = np.array(D_stats)
    p_perm  = np.array(p_perm)

    p_fdr = np.full_like(p_perm, np.nan)
    mask  = ~np.isnan(p_perm)
    if mask.any():
        if _have_sm:
            _, p_corr, _, _ = multipletests(p_perm[mask], alpha=alpha, method="fdr_bh")
        else:
            pv = p_perm[mask]; order = np.argsort(pv); m = len(pv)
            corr = np.empty(m); prev = 1.0
            for rank in range(m - 1, -1, -1):
                idx = order[rank]
                prev = min(prev, pv[idx] * m / (rank + 1))
                corr[idx] = prev
            p_corr = corr
        p_fdr[mask] = p_corr

    green, grey = "#27AE60", "#95A5A6"
    x = np.arange(len(lengths))
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [green if (not np.isnan(p) and p < alpha) else grey for p in p_fdr]
    ax.bar(x, np.nan_to_num(D_stats), color=colors, edgecolor="white",
           linewidth=0.6, width=0.72, zorder=3)
    for xi, d, p in zip(x, D_stats, p_fdr):
        if np.isnan(d):
            continue
        ax.text(float(xi), d + 0.006, _ks_stars(p) if not np.isnan(p) else "",
                ha="center", va="bottom", fontsize=FS_LEG)

    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in lengths])
    ax.set_xlabel("Chain Length", fontsize=FS)
    ax.set_ylabel("KS Statistic", fontsize=FS)
    ax.set_title("Kolmogorov-Smirnov Statistic (Knotted vs Unknotted)\n"
                 f"permutation test, {n_perm} permutations",
                 fontsize=FS_TITLE, fontweight="bold")
    ax.grid(axis="y", alpha=0.15, lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=green, label=rf"$p < {alpha:g}$"),
               Patch(facecolor=grey,  label=rf"$p \geq {alpha:g}$")]
    ax.legend(handles=handles, fontsize=FS_LEG, framealpha=0.85, loc="upper left")
    fig.tight_layout()
    _show_or_save(fig, out_path)


# 3. plot_trend_violin_mean


def plot_trend_violin_mean(
    df: pd.DataFrame,
    lengths=None,
    curvature_col: str  = "Curvature_mean",
    type_col: str       = "Type",
    length_col: str     = "Length",
    n_boot: int         = 1000,
    ci: float           = 95.0,
    violin_width: float = 16.0,
    bw_method: str      = "scott",
    out_path=None,
) -> None:
    """Violin spine - per-loop MEAN curvature vs chain length."""
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

    ax.set_xlabel("Chain Length (residues)", fontsize=FS)
    ax.set_ylabel("Mean Curvature", fontsize=FS)
    ax.set_title("Mean Curvature vs Chain Length", fontsize=FS_TITLE, fontweight="bold")
    ax.set_xticks(lengths)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.legend(fontsize=FS_LEG, framealpha=0.85)
    fig.tight_layout()
    _show_or_save(fig, out_path)


# 4. plot_trend_violin_median


def plot_trend_violin_median(
    df: pd.DataFrame,
    lengths=None,
    curvature_col: str  = "Curvature_median",
    type_col: str       = "Type",
    length_col: str     = "Length",
    n_boot: int         = 1000,
    ci: float           = 95.0,
    violin_width: float = 16.0,
    bw_method: str      = "scott",
    out_path=None,
) -> None:
    """Violin spine - per-loop MEDIAN curvature vs chain length."""
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

    ax.set_xlabel("Chain Length (residues)", fontsize=FS)
    ax.set_ylabel("Median Curvature", fontsize=FS)
    ax.set_title("Median Curvature vs Chain Length", fontsize=FS_TITLE, fontweight="bold")
    ax.set_xticks(lengths)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.legend(fontsize=FS_LEG, framealpha=0.85)
    fig.tight_layout()
    _show_or_save(fig, out_path)


# 5. plot_kde_by_length

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
    """KDE density curves - one subplot per chain length."""
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
            med = float(np.median(vals))
            ax.axvline(med, color=col, lw=1.1, ls="--", alpha=0.70)

        ax.set_title(f"L = {L}", fontsize=FS_TITLE, fontweight="bold")
        ax.set_xlabel("Curvature", fontsize=FS)
        ax.set_ylabel("Density",   fontsize=FS)
        ax.grid(axis="y", alpha=0.15, lw=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(len(lengths), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    axes_flat[0].legend(fontsize=FS_LEG, framealpha=0.85)

    fig.suptitle("KDE - Curvature Distribution per Chain Length",
                 fontsize=FS_TITLE, fontweight="bold", y=1.01)
    fig.tight_layout()
    _show_or_save(fig, out_path)


# 6. plot_violin_by_length

def plot_violin_by_length(
    df: pd.DataFrame,
    lengths=None,
    curvature_col: str = "Curvature",
    type_col: str      = "Type",
    length_col: str    = "Length",
    ncols: int         = 3,
    out_path=None,
) -> None:
    """Violin + box - one subplot per chain length."""
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

        valid_data   = [g for g in groups if len(g) >= 3]
        valid_pos    = [i for i, g in enumerate(groups) if len(g) >= 3]
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
        ax.set_xticklabels([t.capitalize() for t in TYPES], fontsize=FS)
        ax.set_title(f"L = {L}", fontsize=FS_TITLE, fontweight="bold")
        ax.set_ylabel("Curvature", fontsize=FS)
        ax.grid(axis="y", alpha=0.15, lw=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(len(lengths), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COLORS[t], alpha=0.75, label=t.capitalize())
        for t in TYPES
    ]
    fig.legend(handles=legend_handles, fontsize=FS_LEG, framealpha=0.85,
               loc="lower right", ncol=2)

    fig.suptitle("Violin + Box - Curvature per Chain Length",
                 fontsize=FS_TITLE, fontweight="bold", y=1.01)
    fig.tight_layout()
    _show_or_save(fig, out_path)


# 7. plot_trend_line_median_persistence

def plot_trend_line_median_persistence(
    df: pd.DataFrame,
    lengths=None,
    persistence_col: str = "Persistence",
    type_col: str        = "Type",
    length_col: str      = "Length",
    n_boot: int          = 1000,
    ci: float            = 95.0,
    out_path=None,
) -> None:
    """Line plot - MEDIAN H1 persistence vs chain length."""
    lengths = _lengths(df, lengths, length_col)
    S = _compute_stats(df, lengths, persistence_col, type_col,
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

    ax.set_xlabel("Chain Length (residues)", fontsize=FS)
    ax.set_ylabel("Median H1 Persistence", fontsize=FS)
    ax.set_title("Median H1 Persistence vs Chain Length", fontsize=FS_TITLE, fontweight="bold")
    ax.set_xticks(lengths)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.legend(fontsize=FS_LEG, framealpha=0.85)
    fig.tight_layout()
    _show_or_save(fig, out_path)


# 8. plot_trend_violin_median_persistence

def plot_trend_violin_median_persistence(
    df: pd.DataFrame,
    lengths=None,
    persistence_col: str = "Persistence",
    type_col: str        = "Type",
    length_col: str      = "Length",
    n_boot: int          = 1000,
    ci: float            = 95.0,
    violin_width: float  = 16.0,
    bw_method: str       = "scott",
    out_path=None,
) -> None:
    """Violin spine - MEDIAN H1 persistence vs chain length."""
    lengths = _lengths(df, lengths, length_col)
    S = _compute_stats(df, lengths, persistence_col, type_col,
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

    ax.set_xlabel("Chain Length (residues)", fontsize=FS)
    ax.set_ylabel("Median H1 Persistence", fontsize=FS)
    ax.set_title("Median H1 Persistence vs Chain Length", fontsize=FS_TITLE, fontweight="bold")
    ax.set_xticks(lengths)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.legend(fontsize=FS_LEG, framealpha=0.85)
    fig.tight_layout()
    _show_or_save(fig, out_path)


# Data loading helpers (mirror notebook logic + add persistence)

import json as _json

def _load_ph(outputs_dir, ptype, L, j):
    path = Path(outputs_dir) / f"{ptype}_jodelle_{L}_{j}_ph.json"
    with open(path) as f:
        return _json.load(f)

def _adjacency(ph, L):
    M = np.zeros((L, len(ph["barcode"][0])))
    for i, rep in enumerate(ph["representatives"]):
        for j in rep:
            M[j - 1, i] = 1
    return M

def _hyperedge_curvatures(M):
    """Per-hyperedge Forman-Ricci curvature F(e) = 2|e| - D for one loop."""
    n_edges = M.shape[1]
    curv = np.empty(n_edges)
    for e in range(n_edges):
        verts = np.where(M[:, e] == 1)[0]
        D = sum(np.sum(M[k, :]) for k in verts)
        curv[e] = 2 * len(verts) - D
    return curv

def _mean_curvature(M):
    """Per-loop MEAN hyperedge curvature."""
    c = _hyperedge_curvatures(M)
    return float(np.mean(c)) if len(c) else float("nan")

def _median_curvature(M):
    """Per-loop MEDIAN hyperedge curvature.

    This matches the protein pipeline, whose per-protein descriptor is the
    median hyperedge curvature (Curv_median in extract_features.py), and the
    Figure 5 caption ("median of the per-loop medians"). Both statistics are
    kept so the mean- and median-based figures can be produced side by side.
    """
    c = _hyperedge_curvatures(M)
    return float(np.median(c)) if len(c) else float("nan")

def _median_persistence(ph):
    births = np.array(ph["barcode"][0])
    deaths = np.array(ph["barcode"][1])
    return float(np.median(deaths - births))

def load_real_data(outputs_dir="outputs",
                   lengths=(100, 150, 200, 250, 300, 350, 400, 450, 500),
                   n_files=10):
    """Load knotted/unknotted JSON data and return a DataFrame with columns:

        Curvature_mean    per-loop MEAN hyperedge curvature
        Curvature_median  per-loop MEDIAN hyperedge curvature (matches proteins)
        Curvature         alias of Curvature_mean (backward compatibility)
        Persistence       per-loop median H1 persistence
        Type, Length

    Both the mean and the median are stored so the mean- and median-based
    figures can be plotted side by side.
    """
    rows = []
    for L in lengths:
        print(f"  L={L} ...", end=" ", flush=True)
        for ptype, label in (("knots", "knotted"), ("unknots", "unknotted")):
            all_ph = []
            for j in range(n_files):
                all_ph.extend(_load_ph(outputs_dir, ptype, L, j))
            for ph in all_ph:
                M    = _adjacency(ph, L)
                cmean = _mean_curvature(M)
                cmed  = _median_curvature(M)
                pers = _median_persistence(ph)
                rows.append({"Curvature_mean": cmean, "Curvature_median": cmed,
                              "Curvature": cmean,   # alias (legacy)
                              "Persistence": pers,
                              "Type": label, "Length": L})
        print("done")
    return pd.DataFrame(rows)


# run_all


def run_all(df, lengths=None, out_dir=None):
    """Run all plots. Pass out_dir='plots' to save as PDFs."""
    def _p(name):
        return (Path(out_dir) / f"{name}.pdf") if out_dir else None
    print("1/11  trend line    - mean curvature ...");      plot_trend_line_mean(df,                      lengths=lengths, out_path=_p("trend_line_mean"))
    print("2/11  trend line    - median curvature ...");    plot_trend_line_median(df,                    lengths=lengths, out_path=_p("trend_line_median"))
    print("3/11  trend violin  - mean curvature ...");      plot_trend_violin_mean(df,                    lengths=lengths, out_path=_p("trend_violin_mean"))
    print("4/11  trend violin  - median curvature ...");    plot_trend_violin_median(df,                  lengths=lengths, out_path=_p("trend_violin_median"))
    print("5/11  KDE by length ...");                       plot_kde_by_length(df,                        lengths=lengths, out_path=_p("kde_by_length"))
    print("6/11  violin by length ...");                    plot_violin_by_length(df,                     lengths=lengths, out_path=_p("violin_by_length"))
    print("7/11  trend line    - median persistence ...");  plot_trend_line_median_persistence(df,         lengths=lengths, out_path=_p("trend_line_median_persistence"))
    print("8/11  trend violin  - median persistence ...");  plot_trend_violin_median_persistence(df,       lengths=lengths, out_path=_p("trend_violin_median_persistence"))
    print("9/11  trend boxplot - mean curvature ...");      plot_trend_boxplot_mean(df,                   lengths=lengths, out_path=_p("trend_boxplot_mean"))
    print("10/11 trend boxplot - median curvature ...");    plot_trend_boxplot_median(df,                 lengths=lengths, out_path=_p("trend_boxplot_median"))
    print("11/13 trend boxplot - median persistence ...");  plot_trend_boxplot_median_persistence(df,      lengths=lengths, out_path=_p("trend_boxplot_median_persistence"))
    print("12/13 KS statistic vs length (Fig 6, median) ..."); plot_ks_by_length(df,                        lengths=lengths, out_path=_p("ks_statistic"))
    print("13/13 KS statistic - permutation (ties-safe) ..."); plot_ks_permutation_by_length(df,           lengths=lengths, out_path=_p("ks_statistic_permutation"))
    print("Done.")


# python curvature_viz.py [--out_dir my_plots] [--outputs_dir outputs]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",     default="my_plots")
    parser.add_argument("--outputs_dir", default="outputs")
    args = parser.parse_args()

    print("Loading data from", args.outputs_dir)
    Df = load_real_data(outputs_dir=args.outputs_dir)
    print(f"DataFrame: {len(Df)} rows\n{Df.head()}\n")
    run_all(Df, out_dir=args.out_dir)
