#!/usr/bin/env python3
"""
Script 1: Protein vs Homolog Analysis
--------------------------------------
For every original class auto-detects its Unknotted_Homologs and
KnotProt_Homologs groups (PDB_Homologs are excluded). Produces:
  1. Original vs Unknotted_Homologs  (primary comparison)
  2. Original vs Unknotted_Homologs vs KnotProt_Homologs  (three-way)

PDB_Homologs folders are intentionally ignored.

Output layout:
  results/comparison/<OriginalClass>/
    <Original>_vs_Unknotted_Homologs/
        feature_distributions/    violin per feature, auto y-axis
        kde_plots/                smoothed density (kept)
        hist_plots/               un-smoothed histograms (shared bins) — added
        correlation/              Curv_mean|median vs H1 features only
        curv_vs_pers_scatter.pdf

    <Original>_vs_KnotProt_Homologs/
        (same structure)

    <Original>_vs_Unknotted_vs_KnotProt/
        feature_distributions/
        kde_plots/                smoothed density (kept)
        hist_plots/               un-smoothed histograms (shared bins) — added
        curv_vs_pers_scatter.pdf

No statistics. No overview folder. No PDB_Homologs.

Usage:
  python analysis_proteins_vs_homologs.py
  python analysis_proteins_vs_homologs.py --class AOTCases
"""

from __future__ import annotations

import sys
import argparse
import warnings
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).parent))
from config import config

import matplotlib as mpl
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

# Single font-size constants  change here, takes effect everywhere
FS       = 11   # axis labels, tick labels
FS_TITLE = 11   # plot titles
FS_LEG   = 10   # legends

# CONFIG & CONSTANTS
FEATURES_CSV = None
COMPARE_ROOT = None
DPI          = 200


def _resolve_paths():
    global FEATURES_CSV, COMPARE_ROOT
    FEATURES_CSV = config.VIZ_FEATURES_DIR / "protein_features.csv"
    COMPARE_ROOT = config.VIZ_COMPARE_DIR


FEATURE_COLS = [
    "H1_count", "H1_mean", "H1_median", "H1_max", "H1_sum", "H1_std",
    "Curv_mean", "Curv_median", "Curv_max",
    "HG_n_nodes", "HG_n_hyperedges",
    "HG_avg_edge_size", "HG_max_edge_size",
    "HG_connectivity", "HG_density",
]

FEATURE_DISPLAY = {
    "H1_count":        "H1 Bar Count",
    "H1_mean":         "Mean Persistence",
    "H1_median":       "Median Persistence",
    "H1_max":          "Max Persistence",
    "H1_sum":          "Sum Persistence",
    "H1_std":          "Std Persistence",
    "Curv_mean":       "Mean Curvature",
    "Curv_median":     "Median Curvature",
    "Curv_max":        "Max Curvature",
    "HG_n_nodes":      "HG Nodes",
    "HG_n_hyperedges": "HG Hyperedges",
    "HG_avg_edge_size":"HG Avg Edge Size",
    "HG_max_edge_size":"HG Max Edge Size",
    "HG_connectivity": "HG Connectivity",
    "HG_density":      "HG Density",
}

_ROLE_COLORS = {
    "unknotted": "#2980B9",
    "knotprot":  "#27AE60",
    "original":  "#C0392B",
}


def _role_key(class_name: str, orig: str) -> str:
    suffix = class_name[len(orig):].lstrip("_").lower()
    for key in _ROLE_COLORS:
        if key in suffix:
            return key
    return "original"

def group_color(class_name: str, orig: str) -> str:
    return _ROLE_COLORS.get(_role_key(class_name, orig), "#C0392B")

_DISPLAY_NAMES = {
    "Kplus31":  "K+3(1)",
    "Splus31":  "S+3(1)",
    "K41":      "K4(1)",
    "S41":      "S4(1)",
    "Kminus31": "K-3(1)",
    "Sminus31": "S-3(1)",
    "Kminus52": "K-5(2)",
}

def _pretty(name: str) -> str:
    """Return the display name for a class, falling back to the original."""
    return _DISPLAY_NAMES.get(name, name)

def short_label(class_name: str, orig: str) -> str:
    if class_name == orig:
        return _pretty(orig)
    suffix = class_name[len(orig):].lstrip("_")
    if "Unknotted_Homologs" in suffix:
        return f"Unknotted homologs {_pretty(orig)}"
    if "KnotProt_Homologs" in suffix:
        return f"KnotProt homologs {_pretty(orig)}"
    return suffix if suffix else _pretty(class_name)

def pair_dir_name(g1: str, g2: str, orig: str) -> str:
    return f"{short_label(g1,orig)}_vs_{short_label(g2,orig)}"

def savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# DATA LOADING & CLASS DETECTION

def load_data() -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        print(f"Features CSV not found: {FEATURES_CSV}")
        print("Run extract_features.py first.")
        sys.exit(1)
    return pd.read_csv(FEATURES_CSV)


def detect_families(df: pd.DataFrame) -> dict:
    """
    Auto-detect homolog families.
    PDB_Homologs are excluded  only Unknotted_Homologs and
    KnotProt_Homologs are used as comparator groups.
    """
    all_cls = sorted(df["Protein_Class"].unique())
    originals = [c for c in all_cls
                 if not any(c.startswith(o+"_") for o in all_cls if o != c)]
    families = {}
    for orig in originals:
        homologs = sorted(
            c for c in all_cls
            if c.startswith(orig+"_") and "PDB_Homologs" not in c
        )
        if homologs:
            families[orig] = homologs
    return families


def avail_features(df_sub: pd.DataFrame) -> list:
    return [f for f in FEATURE_COLS
            if f in df_sub.columns
            and df_sub[f].notna().sum() > 3
            and df_sub[f].nunique() > 1]


# PLOT PRIMITIVES

def _violin_box_ax(ax, data_groups, colors, labels):
    all_vals = np.concatenate([d for d in data_groups if len(d)])
    if not len(all_vals):
        return 1.0
    q1, q3 = np.percentile(all_vals, [5, 95])
    pad    = (q3 - q1) * 0.5 if q3 > q1 else max(abs(q3)*0.5, 0.5)
    y_min  = q1 - pad
    y_max  = q3 + pad * 1.2

    n = len(data_groups)

    parts = ax.violinplot(data_groups, positions=range(n),
                          widths=0.6, showmeans=False, showmedians=False)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(colors[i]); b.set_alpha(0.22)
    for k in ["cmins","cmaxes","cbars"]:
        if k in parts: parts[k].set_visible(False)

    bp = ax.boxplot(data_groups, positions=range(n),
                    widths=0.30, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=3, alpha=0.35,
                                    markerfacecolor="gray", markeredgewidth=0),
                    medianprops=dict(color="black", lw=2.0))
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i]); box.set_alpha(0.8)

    # Auto y-axis  no stat brackets
    ax.set_ylim(y_min - pad*0.15, y_max)
    ax.set_xticks(range(n))
    ax.set_xticklabels([textwrap.fill(l, 16) for l in labels], rotation=0, ha="center", fontsize=FS)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return y_max


# OVERVIEW PLOTS

# PAIRWISE DEEP-DIVE PLOTS

def plot_pairwise_violin_per_feature(df_pair, feats, g1, g2, orig, out_dir):
    """Violin+box per feature, two groups, auto y-axis, no stat brackets."""
    colors = [group_color(g1, orig), group_color(g2, orig)]
    labels = [short_label(g1, orig), short_label(g2, orig)]
    out_dir.mkdir(parents=True, exist_ok=True)
    for feat in feats:
        d1 = df_pair[df_pair["Protein_Class"]==g1][feat].dropna().values
        d2 = df_pair[df_pair["Protein_Class"]==g2][feat].dropna().values
        if not (len(d1) and len(d2)): continue
        fig, ax = plt.subplots(figsize=(5, 5.5))
        # auto y-axis via _violin_box_ax with no ylim_override
        _violin_box_ax(ax, [d1, d2], colors, labels)
        ax.set_ylabel(FEATURE_DISPLAY.get(feat, feat), fontsize=FS)
        ax.set_title(f"{FEATURE_DISPLAY.get(feat,feat)}\n"
                     f"{short_label(g1,orig)} vs {short_label(g2,orig)}",
                     fontsize=FS_TITLE, fontweight="bold")
        fig.tight_layout()
        savefig(fig, out_dir / f"{feat}.pdf")


def plot_curv_vs_pers_scatter(df_pair, g1, g2, orig, out_path):
    if "Curv_mean" not in df_pair.columns or "H1_mean" not in df_pair.columns: return
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for g in [g1, g2]:
        sub = df_pair[df_pair["Protein_Class"]==g][["Curv_mean","H1_mean"]].dropna()
        if sub.empty: continue
        ax.scatter(sub["H1_mean"], sub["Curv_mean"],
                   c=group_color(g, orig), s=35, alpha=0.68,
                   label=short_label(g, orig),
                   edgecolors="white", linewidths=0.3)
        ax.scatter(sub["H1_mean"].mean(), sub["Curv_mean"].mean(),
                   c=group_color(g, orig), s=130, marker="D", zorder=6,
                   edgecolors="white", linewidths=1.4)
    ax.axhline(0, color="#AAAAAA", lw=0.9, ls="--", alpha=0.55)
    ax.axvline(0, color="#AAAAAA", lw=0.9, ls="--", alpha=0.55)
    ax.set_xlabel("Mean H1 Persistence  (death − birth)", fontsize=FS)
    ax.set_ylabel("Mean Forman-Ricci Curvature", fontsize=FS)
    ax.set_title("Persistence vs Curvature  (◆ = group mean)",
                 fontsize=FS_TITLE, fontweight="bold")
    ax.legend(fontsize=FS_LEG, framealpha=0.92, edgecolor="#DDDDDD", fancybox=False)
    ax.grid(axis="both", alpha=0.13, lw=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, out_path)


def _shared_bins(value_arrays, max_bins=30):
    """
    Compute a common set of histogram bin edges across several groups so the
    two/three populations are binned identically and can be compared fairly.
    Uses the Freedman-Diaconis rule on the pooled data, clamped to a sensible
    number of bins.
    """
    pooled = np.concatenate([v for v in value_arrays if len(v)])
    if len(pooled) < 2:
        return np.linspace(pooled.min() - 0.5, pooled.max() + 0.5, 3) if len(pooled) else np.array([0, 1])
    lo, hi = float(pooled.min()), float(pooled.max())
    if hi <= lo:
        return np.linspace(lo - 0.5, lo + 0.5, 3)
    q1, q3 = np.percentile(pooled, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        width = 2 * iqr / (len(pooled) ** (1.0 / 3.0))
        n_bins = int(np.clip(np.ceil((hi - lo) / width), 5, max_bins)) if width > 0 else 15
    else:
        n_bins = 15
    return np.linspace(lo, hi, n_bins + 1)


def plot_hist_all_features(df_pair, feats, g1, g2, orig, out_dir):
    """Overlaid density histogram per feature, shared bins, two groups."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for feat in feats:
        v1 = df_pair[df_pair["Protein_Class"]==g1][feat].dropna().values
        v2 = df_pair[df_pair["Protein_Class"]==g2][feat].dropna().values
        if len(v1) < 2 and len(v2) < 2: continue
        bins = _shared_bins([v1, v2])
        fig, ax = plt.subplots(figsize=(6, 4))
        for g, vals in [(g1, v1), (g2, v2)]:
            if len(vals) < 1: continue
            col = group_color(g, orig)
            ax.hist(vals, bins=bins, density=True, alpha=0.45, color=col,
                    edgecolor=col, linewidth=0.8,
                    label=f"{short_label(g,orig)} (n={len(vals)})")
        ax.set_xlabel(FEATURE_DISPLAY.get(feat, feat), fontsize=FS)
        ax.set_ylabel("Density", fontsize=FS)
        ax.set_title(f"{FEATURE_DISPLAY.get(feat,feat)}\n"
                     f"{short_label(g1,orig)} vs {short_label(g2,orig)}",
                     fontsize=FS_TITLE, fontweight="bold")
        ax.legend(fontsize=FS_LEG, framealpha=0.9)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        savefig(fig, out_dir / f"{feat}.pdf")


def plot_kde_all_features(df_pair, feats, g1, g2, orig, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = [group_color(g1, orig), group_color(g2, orig)]
    for feat in feats:
        fig, ax = plt.subplots(figsize=(6, 4))
        for g, col in zip([g1, g2], colors):
            vals = df_pair[df_pair["Protein_Class"]==g][feat].dropna().values
            if len(vals) < 4: continue
            try:
                kde = gaussian_kde(vals, bw_method="scott")
                xs  = np.linspace(vals.min(), vals.max(), 300)
                ax.plot(xs, kde(xs), color=col, lw=2,
                        label=f"{short_label(g,orig)} (n={len(vals)})")
                ax.fill_between(xs, kde(xs), alpha=0.08, color=col)
            except Exception:
                pass
        ax.set_xlabel(FEATURE_DISPLAY.get(feat, feat), fontsize=FS)
        ax.set_ylabel("Density", fontsize=FS)
        ax.set_title(f"{FEATURE_DISPLAY.get(feat,feat)}\n"
                     f"{short_label(g1,orig)} vs {short_label(g2,orig)}",
                     fontsize=FS_TITLE, fontweight="bold")
        ax.legend(fontsize=FS_LEG, framealpha=0.9)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        savefig(fig, out_dir / f"{feat}.pdf")


def plot_pairwise_ecdf(df_pair, feats, g1, g2, orig, out_dir):
    """ECDF per feature, two groups. No binning, plots every point as a step."""
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = [group_color(g1, orig), group_color(g2, orig)]
    for feat in feats:
        fig, ax = plt.subplots(figsize=(6, 4))
        drawn = False
        for g, col in zip([g1, g2], colors):
            vals = df_pair[df_pair["Protein_Class"]==g][feat].dropna().values
            if len(vals) < 1: continue
            xs = np.sort(vals)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            # start the curve flat at 0 before the first point
            ax.step(np.concatenate([xs[:1], xs]),
                    np.concatenate([[0.0], ys]),
                    where="post", color=col, lw=2,
                    label=f"{short_label(g,orig)} (n={len(vals)})")
            # rug of the raw values along the bottom
            ax.plot(xs, np.full_like(xs, -0.02, dtype=float), "|",
                    color=col, ms=6, mew=0.8, alpha=0.6, clip_on=False)
            drawn = True
        if not drawn:
            plt.close(fig); continue
        ax.axhline(0.5, color="#AAAAAA", lw=0.8, ls=":", alpha=0.7)
        ax.set_ylim(-0.04, 1.02)
        ax.set_xlabel(FEATURE_DISPLAY.get(feat, feat), fontsize=FS)
        ax.set_ylabel("Cumulative fraction", fontsize=FS)
        ax.set_title(f"{FEATURE_DISPLAY.get(feat,feat)}\n"
                     f"{short_label(g1,orig)} vs {short_label(g2,orig)}",
                     fontsize=FS_TITLE, fontweight="bold")
        ax.legend(fontsize=FS_LEG, framealpha=0.9, loc="lower right")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        savefig(fig, out_dir / f"{feat}.pdf")


def plot_correlation_curv_vs_h1(df_pair, feats, g1, g2, orig, out_dir):
    """
    Correlation scatter: Curv_mean and Curv_median vs H1 features only.
    One plot per (curv_anchor, h1_feature) combination.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = [group_color(g1, orig), group_color(g2, orig)]
    H1_FEATS = [f for f in feats if f.startswith("H1_") and f in df_pair.columns]
    for anchor, anchor_label in [("Curv_mean",   "Mean Curvature"),
                                  ("Curv_median", "Median Curvature")]:
        if anchor not in df_pair.columns: continue
        for h1 in H1_FEATS:
            fig, ax = plt.subplots(figsize=(6, 5))
            for g, col in zip([g1, g2], colors):
                sub = df_pair[df_pair["Protein_Class"]==g][[h1, anchor]].dropna()
                if sub.empty: continue
                ax.scatter(sub[h1], sub[anchor], c=col, s=28, alpha=0.6,
                           label=short_label(g, orig),
                           edgecolors="white", linewidths=0.3)
            ax.set_xlabel(FEATURE_DISPLAY.get(h1, h1), fontsize=FS)
            ax.set_ylabel(anchor_label, fontsize=FS)
            ax.legend(fontsize=FS_LEG, framealpha=0.9)
            ax.grid(alpha=0.15)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            fig.tight_layout()
            savefig(fig, out_dir / f"{anchor}_vs_{h1}.pdf")


def plot_threeway_violin_per_feature(df_three, three_classes, orig, feats, out_dir):
    """Violin+box for three groups, auto y-axis, no stat brackets."""
    colors = [group_color(g, orig) for g in three_classes]
    labels = [short_label(g, orig) for g in three_classes]
    out_dir.mkdir(parents=True, exist_ok=True)
    for feat in feats:
        dg = [df_three[df_three["Protein_Class"]==g][feat].dropna().values
              for g in three_classes]
        if not any(len(d) for d in dg): continue
        fig, ax = plt.subplots(figsize=(6, 5.5))
        _violin_box_ax(ax, dg, colors, labels)
        ax.set_ylabel(FEATURE_DISPLAY.get(feat, feat), fontsize=FS)
        ax.set_title(f"{FEATURE_DISPLAY.get(feat,feat)}\n"
                     f"{short_label(three_classes[0],orig)} vs Unknotted vs KnotProt",
                     fontsize=FS_TITLE, fontweight="bold")
        fig.tight_layout()
        savefig(fig, out_dir / f"{feat}.pdf")


def plot_kde_threeway(df_three, three_classes, orig, feats, out_dir):
    """KDE for three groups."""
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = [group_color(g, orig) for g in three_classes]
    for feat in feats:
        fig, ax = plt.subplots(figsize=(6, 4))
        for g, col in zip(three_classes, colors):
            vals = df_three[df_three["Protein_Class"]==g][feat].dropna().values
            if len(vals) < 4: continue
            try:
                kde = gaussian_kde(vals, bw_method="scott")
                xs  = np.linspace(vals.min(), vals.max(), 300)
                ax.plot(xs, kde(xs), color=col, lw=2.2,
                        label=f"{short_label(g,orig)} (n={len(vals)})")
                ax.fill_between(xs, kde(xs), alpha=0.08, color=col)
            except Exception:
                pass
        ax.set_xlabel(FEATURE_DISPLAY.get(feat, feat), fontsize=FS)
        ax.set_ylabel("Density", fontsize=FS)
        ax.set_title(f"{FEATURE_DISPLAY.get(feat,feat)}\nOriginal vs Unknotted vs KnotProt",
                     fontsize=FS_TITLE, fontweight="bold")
        ax.legend(fontsize=FS_LEG, framealpha=0.9)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        savefig(fig, out_dir / f"{feat}.pdf")


def plot_hist_threeway(df_three, three_classes, orig, feats, out_dir):
    """Overlaid density histogram per feature, shared bins, three groups."""
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = [group_color(g, orig) for g in three_classes]
    for feat in feats:
        arrays = [df_three[df_three["Protein_Class"]==g][feat].dropna().values
                  for g in three_classes]
        if not any(len(a) >= 2 for a in arrays): continue
        bins = _shared_bins(arrays)
        fig, ax = plt.subplots(figsize=(6, 4))
        for g, col, vals in zip(three_classes, colors, arrays):
            if len(vals) < 1: continue
            ax.hist(vals, bins=bins, density=True, alpha=0.40, color=col,
                    edgecolor=col, linewidth=0.8,
                    label=f"{short_label(g,orig)} (n={len(vals)})")
        ax.set_xlabel(FEATURE_DISPLAY.get(feat, feat), fontsize=FS)
        ax.set_ylabel("Density", fontsize=FS)
        ax.set_title(f"{FEATURE_DISPLAY.get(feat,feat)}\nOriginal vs Unknotted vs KnotProt",
                     fontsize=FS_TITLE, fontweight="bold")
        ax.legend(fontsize=FS_LEG, framealpha=0.9)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        savefig(fig, out_dir / f"{feat}.pdf")


def plot_curv_vs_pers_scatter_multi(df_three, three_classes, orig, out_path):
    """Curvature vs persistence scatter for three groups."""
    if "Curv_mean" not in df_three.columns or "H1_mean" not in df_three.columns: return
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for g in three_classes:
        sub = df_three[df_three["Protein_Class"]==g][["Curv_mean","H1_mean"]].dropna()
        if sub.empty: continue
        col = group_color(g, orig)
        ax.scatter(sub["H1_mean"], sub["Curv_mean"],
                   c=col, s=35, alpha=0.68,
                   label=short_label(g, orig),
                   edgecolors="white", linewidths=0.3)
        ax.scatter(sub["H1_mean"].mean(), sub["Curv_mean"].mean(),
                   c=col, s=130, marker="D", zorder=6,
                   edgecolors="white", linewidths=1.4)
    ax.axhline(0, color="#AAAAAA", lw=0.9, ls="--", alpha=0.55)
    ax.axvline(0, color="#AAAAAA", lw=0.9, ls="--", alpha=0.55)
    ax.set_xlabel("Mean H1 Persistence", fontsize=FS)
    ax.set_ylabel("Mean Forman-Ricci Curvature", fontsize=FS)
    ax.set_title("Persistence vs Curvature  (◆ = group mean)\nOriginal vs Unknotted vs KnotProt",
                 fontsize=FS_TITLE, fontweight="bold")
    ax.legend(fontsize=FS_LEG, framealpha=0.92, edgecolor="#DDDDDD", fancybox=False)
    ax.grid(axis="both", alpha=0.13, lw=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, out_path)

# FAMILY ORCHESTRATOR

def process_family(orig: str, homologs: list, df: pd.DataFrame):
    """
    Produce three comparison folders per family:
      1. <orig>_vs_Unknotted_Homologs/
      2. <orig>_vs_KnotProt_Homologs/
      3. <orig>_vs_Unknotted_vs_KnotProt/    three-group

    No statistics are run here. No overview folder.
    Plots:
      feature_distributions/   violin per feature (auto y-axis)
      kde_plots/               KDE per feature
      correlation/             Curv_mean|median vs H1 features only
      curv_vs_pers_scatter.pdf
    """
    out = COMPARE_ROOT / orig
    df_fam = df[df["Protein_Class"].isin([orig] + homologs)].copy()

    # Identify unknotted and knotprot homologs
    unk_cls = next((h for h in homologs if "Unknotted_Homologs" in h), None)
    kp_cls  = next((h for h in homologs if "KnotProt_Homologs"  in h), None)

    # Pairwise: orig vs unknotted
    if unk_cls:
        _plot_pair(orig, unk_cls, orig, df_fam,
                   out / pair_dir_name(orig, unk_cls, orig))

    # Pairwise: orig vs knotprot
    if kp_cls:
        _plot_pair(orig, kp_cls, orig, df_fam,
                   out / pair_dir_name(orig, kp_cls, orig))

    # Three-group: orig vs unknotted vs knotprot
    if unk_cls and kp_cls:
        three_dir = out / f"{_pretty(orig)}_vs_Unknotted_vs_KnotProt"
        three_classes = [orig, unk_cls, kp_cls]
        df_three = df_fam[df_fam["Protein_Class"].isin(three_classes)].copy()
        feats_3  = avail_features(df_three)
        plot_threeway_violin_per_feature(df_three, three_classes, orig, feats_3,
                                         three_dir / "feature_distributions")
        plot_kde_threeway(df_three, three_classes, orig, feats_3,
                          three_dir / "kde_plots")
        plot_hist_threeway(df_three, three_classes, orig, feats_3,
                           three_dir / "hist_plots")
        plot_curv_vs_pers_scatter_multi(df_three, three_classes, orig,
                                        three_dir / "curv_vs_pers_scatter.pdf")


def _plot_pair(g1: str, g2: str, orig: str, df_fam: pd.DataFrame,
               pair_dir: Path):
    """All plots for one pairwise comparison (no stats)."""
    df_pair = df_fam[df_fam["Protein_Class"].isin([g1, g2])].copy()
    feats_p = avail_features(df_pair)
    plot_pairwise_violin_per_feature(df_pair, feats_p, g1, g2, orig,
                                     pair_dir / "feature_distributions")
    plot_kde_all_features(df_pair, feats_p, g1, g2, orig, pair_dir / "kde_plots")
    plot_hist_all_features(df_pair, feats_p, g1, g2, orig, pair_dir / "hist_plots")
    plot_pairwise_ecdf(df_pair, feats_p, g1, g2, orig, pair_dir / "ecdf_plots")
    plot_correlation_curv_vs_h1(df_pair, feats_p, g1, g2, orig,
                                 pair_dir / "correlation")
    plot_curv_vs_pers_scatter(df_pair, g1, g2, orig,
                               pair_dir / "curv_vs_pers_scatter.pdf")





# MAIN

def main():
    parser = argparse.ArgumentParser(
        description="Script 1  Protein vs homolog analysis (plots only, no statistics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--class", dest="cls", default=None,
                        help="Analyse one family only, e.g. AOTCases")
    args = parser.parse_args()
    _resolve_paths()

    print("=" * 65)
    print("PROTEIN vs HOMOLOG ANALYSIS  (Unknotted + KnotProt | no PDB)")
    print("Comparators: Unknotted_Homologs + KnotProt_Homologs (PDB excluded)")
    print("=" * 65)
    print(f"Start   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input   : {FEATURES_CSV}")
    print(f"Output  : {COMPARE_ROOT}\n")

    df = load_data()

    families = detect_families(df)
    if not families:
        print("No homolog families detected.")
        print("Ensure homolog names start with original class name + '_'.")
        sys.exit(1)

    if args.cls:
        if args.cls not in families:
            print(f"Class '{args.cls}' not found. Available: {list(families.keys())}")
            sys.exit(1)
        families = {args.cls: families[args.cls]}

    for orig, homologs in families.items():
        print(f"\n{'─'*65}")
        print(f"CLASS: {orig}")
        print(f"  Homologs: {', '.join(short_label(h, orig) for h in homologs)}")
        process_family(orig, homologs, df)

    print(f"\n{'='*65}")
    print(f"Done.  Results in: {COMPARE_ROOT}")
    print(f"End   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()