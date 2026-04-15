#!/usr/bin/env python3
"""
statistical_tests.py
--------------------
Compares each original protein class against its homolog groups using
KS and Levene tests on Curv_median. Two comparison modes:

  Mode 1 — Original vs Unknotted Homologs  (2-group)
  Mode 2 — Original vs Unknotted Homologs vs KnotProt Homologs  (3-group)

FIXED PAIRS (no PDB homologs):
  K41     vs  K41_Unknotted_Homologs   [+  K41_KnotProt_Homologs]
  S41     vs  S41_Unknotted_Homologs   [+  S41_KnotProt_Homologs]
  Kplus31 vs  Kplus31_Unknotted_Homologs [+ Kplus31_KnotProt_Homologs]
  Splus31 vs  Splus31_Unknotted_Homologs [+ Splus31_KnotProt_Homologs]

Tests:
  Kolmogorov-Smirnov  (distributional difference, FDR-BH corrected)
  Levene              (variance equality, FDR-BH corrected)
  Chirality           (median direction, assigned only when KS significant)

Usage:
  python statistical_tests.py                   # 2-group mode
  python statistical_tests.py --three-way       # 3-group mode
  python statistical_tests.py --stats-only
"""

from __future__ import annotations

import sys
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy.stats import ks_2samp, levene, kruskal
from statsmodels.stats.multitest import multipletests
from itertools import combinations

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import config

import matplotlib as mpl
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

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
FEATURES_CSV = None
OUT_ROOT     = None
DPI          = 200
STATS_ONLY   = False
THREE_WAY    = False   # set by --three-way flag

def _resolve_paths():
    global FEATURES_CSV, OUT_ROOT
    FEATURES_CSV = config.VIZ_FEATURES_DIR / "protein_features.csv"
    suffix   = "three_way" if THREE_WAY else "two_group"
    OUT_ROOT = config.VIZ_COMPARE_DIR / f"curvature_original_vs_homologs_{suffix}"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_ROOT}")

FOCUS_FEATURES = ["Curv_median"]
FEATURE_DISPLAY = {"Curv_median": "Median Curvature"}

# Fixed pairs — Unknotted Homologs comparator for all four classes
FIXED_PAIRS = {
    "K41":     {"unknotted": "K41_Unknotted_Homologs",
                "knotprot":  "K41_KnotProt_Homologs"},
    "S41":     {"unknotted": "S41_Unknotted_Homologs",
                "knotprot":  "S41_KnotProt_Homologs"},
    "Kplus31": {"unknotted": "Kplus31_Unknotted_Homologs",
                "knotprot":  "Kplus31_KnotProt_Homologs"},
    "Splus31": {"unknotted": "Splus31_Unknotted_Homologs",
                "knotprot":  "Splus31_KnotProt_Homologs"},
}

GROUP_COLORS = {
    "original":  "#C0392B",
    "unknotted": "#2980B9",
    "knotprot":  "#27AE60",
}

def stars(p: float) -> str:
    if p < 0.0001: return "****"
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "ns"

def savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        print(f"Features CSV not found: {FEATURES_CSV}")
        sys.exit(1)
    return pd.read_csv(FEATURES_CSV)


def detect_pairs(df: pd.DataFrame) -> dict:
    """
    Restrict FIXED_PAIRS to classes actually present in the data.
    Returns {orig: {"unknotted": cls, "knotprot": cls_or_None}}
    """
    all_cls = set(df["Protein_Class"].unique())
    valid = {}
    for orig, groups in FIXED_PAIRS.items():
        unk = groups["unknotted"]
        kp  = groups["knotprot"]
        if orig not in all_cls:
            print(f"  [SKIP] {orig}: original class not in data")
            continue
        if unk not in all_cls:
            print(f"  [SKIP] {orig}: unknotted homologs ({unk}) not in data")
            continue
        kp_present = kp if kp in all_cls else None
        if THREE_WAY and kp_present is None:
            print(f"  [WARN] {orig}: KnotProt homologs ({kp}) not in data — "
                  f"will run 2-group only for this class")
        valid[orig] = {"unknotted": unk, "knotprot": kp_present}
    return valid


# ─────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────

def _ks_levene(a: np.ndarray, b: np.ndarray) -> dict:
    """Run KS + Levene on two arrays. Return raw p-values and stats."""
    out = {}
    if len(a) < 3 or len(b) < 3:
        return {"KS_stat": np.nan, "KS_p": np.nan,
                "Lev_stat": np.nan, "Lev_p": np.nan,
                "median_diff": np.nan, "variance_ratio": np.nan}
    try:
        ks_s, ks_p = ks_2samp(a, b, alternative="two-sided")
        out["KS_stat"] = float(ks_s)
        out["KS_p"]    = float(ks_p)
    except Exception:
        out["KS_stat"] = np.nan
        out["KS_p"]    = np.nan
    try:
        lv_s, lv_p = levene(a, b, center="median")
        out["Lev_stat"] = float(lv_s)
        out["Lev_p"]    = float(lv_p)
    except Exception:
        out["Lev_stat"] = np.nan
        out["Lev_p"]    = np.nan
    out["median_diff"]    = float(np.median(a) - np.median(b))
    out["variance_ratio"] = (float(np.var(a) / np.var(b))
                             if np.var(b) > 0 else np.nan)
    return out


def run_two_group(df: pd.DataFrame, pairs: dict) -> pd.DataFrame:
    """
    For each class: Original vs Unknotted_Homologs.
    Returns a long DataFrame with one row per (class, feature).
    """
    rows = []
    for orig, groups in pairs.items():
        unk = groups["unknotted"]
        for feat in FOCUS_FEATURES:
            if feat not in df.columns:
                continue
            a = df[df["Protein_Class"] == orig][feat].dropna().values
            b = df[df["Protein_Class"] == unk][feat].dropna().values
            stats = _ks_levene(a, b)
            rows.append({
                "original_class":  orig,
                "homolog_class":   unk,
                "comparison":      "Original_vs_Unknotted",
                "feature":         feat,
                "display":         FEATURE_DISPLAY.get(feat, feat),
                "n_original":      len(a),
                "n_homolog":       len(b),
                "median_original": float(np.median(a)) if len(a) else np.nan,
                "median_homolog":  float(np.median(b)) if len(b) else np.nan,
                "std_original":    float(np.std(a, ddof=1)) if len(a) > 1 else np.nan,
                "std_homolog":     float(np.std(b, ddof=1)) if len(b) > 1 else np.nan,
                **stats,
            })
    return pd.DataFrame(rows)


def run_three_group(df: pd.DataFrame, pairs: dict) -> pd.DataFrame:
    """
    For each class with all three groups present:
      Original vs Unknotted  +  Original vs KnotProt  +  Kruskal-Wallis omnibus
    No Unknotted vs KnotProt comparison.
    Returns one row per (class, comparison, feature).
    """
    rows = []
    for orig, groups in pairs.items():
        unk = groups["unknotted"]
        kp  = groups["knotprot"]
        for feat in FOCUS_FEATURES:
            if feat not in df.columns:
                continue
            a = df[df["Protein_Class"] == orig][feat].dropna().values
            b = df[df["Protein_Class"] == unk][feat].dropna().values

            # ── always do Original vs Unknotted ──────────────────────
            s = _ks_levene(a, b)
            row_base = {
                "original_class": orig,
                "feature":        feat,
                "display":        FEATURE_DISPLAY.get(feat, feat),
                "n_original":     len(a),
                "median_original":float(np.median(a)) if len(a) else np.nan,
                "std_original":   float(np.std(a, ddof=1)) if len(a) > 1 else np.nan,
            }
            rows.append({
                **row_base,
                "comparison":    "Original_vs_Unknotted",
                "homolog_class": unk,
                "n_homolog":     len(b),
                "median_homolog":float(np.median(b)) if len(b) else np.nan,
                "std_homolog":   float(np.std(b, ddof=1)) if len(b) > 1 else np.nan,
                **s,
            })

            # ── KnotProt comparisons (if available) ──────────────────
            if kp is not None:
                c = df[df["Protein_Class"] == kp][feat].dropna().values

                # Original vs KnotProt
                s2 = _ks_levene(a, c)
                rows.append({
                    **row_base,
                    "comparison":    "Original_vs_KnotProt",
                    "homolog_class": kp,
                    "n_homolog":     len(c),
                    "median_homolog":float(np.median(c)) if len(c) else np.nan,
                    "std_homolog":   float(np.std(c, ddof=1)) if len(c) > 1 else np.nan,
                    **s2,
                })

                # Kruskal-Wallis omnibus (3-group)
                groups_data = [g for g in [a, b, c] if len(g) >= 3]
                if len(groups_data) == 3:
                    try:
                        kw_s, kw_p = kruskal(*groups_data)
                    except Exception:
                        kw_s, kw_p = np.nan, np.nan
                else:
                    kw_s, kw_p = np.nan, np.nan
                rows.append({
                    "original_class": orig,
                    "feature":        feat,
                    "display":        FEATURE_DISPLAY.get(feat, feat),
                    "comparison":    "KruskalWallis_omnibus",
                    "homolog_class": f"{unk} + {kp}",
                    "n_original":    len(a),
                    "n_homolog":     len(b) + len(c),
                    "median_original": float(np.median(a)) if len(a) else np.nan,
                    "median_homolog":  np.nan,
                    "std_original":    float(np.std(a, ddof=1)) if len(a) > 1 else np.nan,
                    "std_homolog":     np.nan,
                    "KS_stat":        np.nan,
                    "KS_p":           np.nan,
                    "Lev_stat":       np.nan,
                    "Lev_p":          np.nan,
                    "median_diff":    np.nan,
                    "variance_ratio": np.nan,
                    "KW_stat":        float(kw_s),
                    "KW_p":           float(kw_p),
                })

    df_out = pd.DataFrame(rows)
    if "KW_stat" not in df_out.columns:
        df_out["KW_stat"] = np.nan
        df_out["KW_p"]    = np.nan
    return df_out


def apply_fdr(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply FDR-BH separately for KS and Levene across all rows.
    Chirality assigned after FDR on KS.
    """
    for raw_col, fdr_col, sig_col, star_col in [
        ("KS_p",  "KS_p_fdr",  "KS_significant",  "KS_stars"),
        ("Lev_p", "Lev_p_fdr", "Lev_significant", "Lev_stars"),
        ("KW_p",  "KW_p_fdr",  "KW_significant",  "KW_stars"),
    ]:
        if raw_col not in results_df.columns:
            results_df[fdr_col] = np.nan
            results_df[sig_col] = False
            results_df[star_col] = ""
            continue
        mask = ~results_df[raw_col].isna()
        p_vals = results_df.loc[mask, raw_col].values
        if len(p_vals) > 0:
            rej, p_fdr, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")
            results_df.loc[mask, fdr_col]  = p_fdr
            results_df.loc[mask, sig_col]  = rej
            results_df.loc[mask, star_col] = [stars(p) for p in p_fdr]
        else:
            results_df[fdr_col]  = np.nan
            results_df[sig_col]  = False
            results_df[star_col] = ""

    # Chirality from KS significance + median_diff
    if "KS_significant" in results_df.columns and "median_diff" in results_df.columns:
        def _chir(row):
            if not row.get("KS_significant", False):
                return "not_significant"
            diff = row.get("median_diff", np.nan)
            if pd.isna(diff):
                return np.nan
            orig = row.get("original_class", "")
            hom  = row.get("homolog_class",  "")
            if diff > 0.05:
                return f"{orig}_more_negative"
            elif diff < -0.05:
                return f"{hom}_more_negative"
            return "no_dominant_direction"
        results_df["chirality"] = results_df.apply(_chir, axis=1)

    return results_df


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────

def plot_violin(df: pd.DataFrame, pairs: dict, feat: str, out_dir: Path):
    """One violin+box per class, showing all available groups."""
    for orig, groups in pairs.items():
        classes = [orig, groups["unknotted"]]
        labels  = ["Original", "Unknotted"]
        colors  = [GROUP_COLORS["original"], GROUP_COLORS["unknotted"]]
        if THREE_WAY and groups["knotprot"]:
            classes.append(groups["knotprot"])
            labels.append("KnotProt")
            colors.append(GROUP_COLORS["knotprot"])

        plot_rows = []
        for cls, lbl in zip(classes, labels):
            sub = df[df["Protein_Class"] == cls][feat].dropna()
            for v in sub:
                plot_rows.append({"Group": lbl, feat: v})
        if not plot_rows:
            continue
        pdf = pd.DataFrame(plot_rows)
        n_groups = len(labels)
        fig, ax = plt.subplots(figsize=(2.5 * n_groups + 1, 6))
        sns.violinplot(data=pdf, x="Group", y=feat, ax=ax,
                       palette=dict(zip(labels, colors)),
                       order=labels, inner="box", cut=0)
        ax.set_title(f"{FEATURE_DISPLAY.get(feat, feat)}\n{orig}", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(FEATURE_DISPLAY.get(feat, feat), fontsize=10)
        fig.tight_layout()
        savefig(fig, out_dir / f"{orig}_{feat}_violin.pdf")


def plot_effect_sizes(results_df: pd.DataFrame, out_path: Path):
    """KS D-statistic and median difference bar charts."""
    df = results_df[results_df["comparison"] != "KruskalWallis_omnibus"].copy()
    if df.empty:
        return

    comparisons = df["comparison"].unique()
    n = len(comparisons)
    fig, axes = plt.subplots(1, n * 2, figsize=(7 * n, max(5, len(df) // n * 0.6 + 2)))
    if n * 2 == 2:
        axes = list(axes)

    col = 0
    for cmp in comparisons:
        sub = df[df["comparison"] == cmp]
        labels = sub["original_class"] + "\nvs\n" + sub["homolog_class"].str.split("_").str[-2]

        ax1 = axes[col]
        sig = sub.get("KS_significant", pd.Series([False]*len(sub))).values
        bar_colors = ["#E74C3C" if s else "#BDC3C7" for s in sig]
        ax1.barh(labels, sub["KS_stat"], color=bar_colors, edgecolor="black", height=0.6)
        ax1.axvline(0.3, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax1.set_xlabel("KS D-statistic", fontsize=10)
        ax1.set_title(f"{cmp}\nKS D (red=significant)", fontsize=9, fontweight="bold")
        ax1.set_xlim(0, 1)
        ax1.grid(axis="x", alpha=0.2)

        ax2 = axes[col + 1]
        md_colors = ["#E74C3C" if x < 0 else "#2980B9"
                     for x in sub["median_diff"].fillna(0)]
        ax2.barh(labels, sub["median_diff"], color=md_colors, edgecolor="black", height=0.6)
        ax2.axvline(0, color="black", lw=1)
        ax2.set_xlabel("Median diff (Original − Homolog)", fontsize=10)
        ax2.set_title(f"{cmp}\nDirection of shift", fontsize=9, fontweight="bold")
        ax2.grid(axis="x", alpha=0.2)
        col += 2

    fig.suptitle("Effect Sizes — Curv_median", fontsize=12, fontweight="bold")
    fig.tight_layout()
    savefig(fig, out_path)


def plot_variance(results_df: pd.DataFrame, out_path: Path):
    """Log2 variance ratio coloured by Levene significance."""
    df = results_df[
        results_df["comparison"].isin(["Original_vs_Unknotted", "Original_vs_KnotProt"])
        & results_df["variance_ratio"].notna()
    ].copy()
    if df.empty:
        return
    df["log2_vr"] = np.log2(df["variance_ratio"])
    labels = df["original_class"] + "\nvs\n" + df["homolog_class"].str.split("_").str[-2]
    colors = ["#E74C3C" if s else "#BDC3C7"
              for s in df.get("Lev_significant", pd.Series([False]*len(df)))]
    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.4)))
    ax.barh(labels, df["log2_vr"], color=colors, edgecolor="black", height=0.7)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("log₂(Variance Ratio)\n(Original / Homolog)", fontsize=11)
    ax.set_title("Variance Comparison (Red = Levene significant, FDR-corrected)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    savefig(fig, out_path)


def plot_boxplots(df: pd.DataFrame, pairs: dict, feat: str, out_dir: Path):
    """Boxplot for each class pair."""
    for orig, groups in pairs.items():
        classes = [orig, groups["unknotted"]]
        labels  = ["Original", "Unknotted"]
        colors  = [GROUP_COLORS["original"], GROUP_COLORS["unknotted"]]
        if THREE_WAY and groups["knotprot"]:
            classes.append(groups["knotprot"])
            labels.append("KnotProt")
            colors.append(GROUP_COLORS["knotprot"])

        rows = []
        for cls, lbl in zip(classes, labels):
            for v in df[df["Protein_Class"] == cls][feat].dropna():
                rows.append({"Group": lbl, feat: v})
        if not rows:
            continue
        pdf = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(2.5 * len(labels) + 1, 6))
        sns.boxplot(data=pdf, x="Group", y=feat, ax=ax,
                    palette=dict(zip(labels, colors)), order=labels)
        sns.stripplot(data=pdf, x="Group", y=feat, ax=ax,
                      color="black", alpha=0.3, size=3, order=labels)
        ax.set_title(f"{FEATURE_DISPLAY.get(feat, feat)}\n{orig}", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(FEATURE_DISPLAY.get(feat, feat), fontsize=10)
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        savefig(fig, out_dir / f"{orig}_{feat}_boxplot.pdf")


# ─────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────

def write_summary(results_df: pd.DataFrame, pairs: dict, out_path: Path):
    mode = "3-group (Original / Unknotted / KnotProt)" if THREE_WAY else "2-group (Original / Unknotted)"
    lines = [
        "=" * 80,
        f"CURVATURE ANALYSIS — {mode}",
        "=" * 80,
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Feature   : Median Curvature (Curv_median)",
        f"Tests     : KS (distribution) | Levene (variance)"
        + (" | Kruskal-Wallis (omnibus)" if THREE_WAY else ""),
        f"FDR       : Benjamini-Hochberg, applied separately per test",
        "",
        "PAIRS:",
    ]
    for orig, g in pairs.items():
        kp_str = f" | KnotProt: {g['knotprot']}" if g.get("knotprot") else ""
        lines.append(f"  {orig:<12} → Unknotted: {g['unknotted']}{kp_str}")
    lines.append("")

    for feat in FOCUS_FEATURES:
        feat_df = results_df[results_df["feature"] == feat]
        if feat_df.empty:
            continue
        lines += ["─" * 80, f"{FEATURE_DISPLAY[feat]}", "─" * 80, ""]

        for cmp in feat_df["comparison"].unique():
            sub = feat_df[feat_df["comparison"] == cmp]
            lines.append(f"  {cmp}:")
            header = (f"    {'Class':<20} {'n_orig':>6} {'n_hom':>6}  "
                      f"{'KS_D':>6} {'KS_p_fdr':>10} {'Lev_p_fdr':>10}  Chirality")
            lines.append(header)
            lines.append("    " + "-" * 80)
            for _, row in sub.iterrows():
                ks_d   = f"{row.get('KS_stat', np.nan):.3f}" if not pd.isna(row.get('KS_stat', np.nan)) else "n/a"
                ks_p   = f"{row.get('KS_p_fdr', np.nan):.2e}{row.get('KS_stars','')}" if not pd.isna(row.get('KS_p_fdr', np.nan)) else "n/a"
                lev_p  = f"{row.get('Lev_p_fdr', np.nan):.2e}{row.get('Lev_stars','')}" if not pd.isna(row.get('Lev_p_fdr', np.nan)) else "n/a"
                chir   = str(row.get("chirality", ""))[:28]
                lines.append(
                    f"    {row['original_class']:<20} "
                    f"{int(row.get('n_original',0)):>6} {int(row.get('n_homolog',0)):>6}  "
                    f"{ks_d:>6} {ks_p:>10} {lev_p:>10}  {chir}"
                )
            lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Summary: {out_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="KS + Levene tests — Original vs Unknotted [vs KnotProt] Homologs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--three-way", dest="three_way", action="store_true",
                        default=False,
                        help="Include KnotProt Homologs as a third group")
    parser.add_argument("--stats-only", dest="stats_only", action="store_true",
                        default=False, help="CSVs and report only, no plots")
    args = parser.parse_args()

    global STATS_ONLY, THREE_WAY
    STATS_ONLY = args.stats_only
    THREE_WAY  = args.three_way
    _resolve_paths()

    mode = "3-WAY (Original / Unknotted / KnotProt)" if THREE_WAY else "2-GROUP (Original / Unknotted)"
    print("=" * 65)
    print(f"STATISTICAL TESTS — {mode}")
    print(f"Feature : Curv_median | Tests: KS + Levene")
    print("=" * 65)
    print(f"Start   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input   : {FEATURES_CSV}")
    print(f"Output  : {OUT_ROOT}\n")

    df    = load_data()
    pairs = detect_pairs(df)
    if not pairs:
        print("ERROR: No valid pairs found in data.")
        sys.exit(1)

    print(f"Valid pairs: {len(pairs)}")
    for orig, g in pairs.items():
        kp_str = f" | KnotProt: {g['knotprot']}" if g.get("knotprot") else " | KnotProt: not available"
        print(f"  {orig} → Unknotted: {g['unknotted']}{kp_str}")
    print()

    # ── Run statistics ────────────────────────────────────────
    if THREE_WAY:
        results_df = run_three_group(df, pairs)
    else:
        results_df = run_two_group(df, pairs)

    if results_df.empty:
        print("ERROR: No results generated.")
        sys.exit(1)

    results_df = apply_fdr(results_df)

    # ── Save CSVs ─────────────────────────────────────────────
    results_df.to_csv(OUT_ROOT / "comparison_results.csv", index=False)
    print(f"Main results: {OUT_ROOT / 'comparison_results.csv'}")

    per_dir = OUT_ROOT / "by_class"
    per_dir.mkdir(parents=True, exist_ok=True)
    for orig in results_df["original_class"].unique():
        sub = results_df[results_df["original_class"] == orig]
        sub.to_csv(per_dir / f"{orig}_results.csv", index=False)
    print(f"Per-class results: {per_dir}")

    write_summary(results_df, pairs, OUT_ROOT / "summary_report.txt")

    # ── Plots ─────────────────────────────────────────────────
    if not STATS_ONLY:
        plots_dir = OUT_ROOT / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        for feat in FOCUS_FEATURES:
            if feat in df.columns:
                plot_violin(df, pairs, feat, plots_dir / "violins")
                plot_boxplots(df, pairs, feat, plots_dir / "boxplots")

        plot_effect_sizes(results_df, plots_dir / "effect_sizes.pdf")
        plot_variance(results_df, plots_dir / "variance_comparison.pdf")
        print(f"Plots: {plots_dir}")

    print(f"\n{'='*65}")
    print(f"Done. Results in: {OUT_ROOT}")
    print(f"End   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
