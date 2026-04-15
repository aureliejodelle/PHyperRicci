#!/usr/bin/env python3
"""
Script 7: Visualize Persistent Homology Results
------------------------------------------------
Reads saved JSON outputs from step 4 and produces publication-quality
PDF figures for each protein:

  1. Persistence diagram     (H0 + H1 scatter on birth/death axes)
  2. Barcode diagram         (H0 + H1 horizontal bars)
  3. Most persistent cycle   (3D Cα backbone + cycle edges overlay)

Inputs:
  - processed_data/Persistent_homology/<class>/barcodes/<protein>.json
  - processed_data/Persistent_homology/<class>/PH_1/<protein>.json
  - raw_data/coordinates_data/<class>/<protein>.csv

Outputs (under processed_data/Persistent_homology/<class>/):
  - persistent_diagram/<protein>.pdf
  - barcode_plots/<protein>.pdf
  - most_persistent_cycle/<protein>.pdf

Usage examples
--------------
  # All classes, all plot types (original behaviour):
  python visualize_ph.py

  # Only barcodes for all classes:
  python visualize_ph.py --plots barcodes

  # Only persistence diagram + most-persistent cycle:
  python visualize_ph.py --plots persistent_diagram most_persistent_cycle

  # All plots, single class:
  python visualize_ph.py --classes AOTCases

  # Specific plot(s) + specific class(es):
  python visualize_ph.py --plots barcodes --classes AOTCases OTCases

  # All classes whose name starts with 'k':
  python visualize_ph.py --group k

  # All classes whose name starts with 's':
  python visualize_ph.py --group s

Available --plots values:
  persistent_diagram   barcode   most_persistent_cycle
"""

import sys
import json
import argparse
import threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for all threads
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import config, add_dataset_arg, apply_dataset_from_args

# ============================================================
# CONFIG
# ============================================================
NUM_WORKERS  = 8     # threads — matplotlib Agg is thread-safe
DPI          = 150   # resolution for rasterized elements inside PDF

# All recognised plot keys and their output subfolder names
PLOT_KEY_MAP = {
    "persistent_diagram":    "persistent_diagram",
    "barcodes":              "barcode_plots",
    "most_persistent_cycle": "most_persistent_cycle",
}
ALL_PLOTS = list(PLOT_KEY_MAP.keys())

# Colour palette
C_H0    = "#4878CF"   # steel blue  — H0
C_H1    = "#D65F5F"   # crimson     — H1
C_DIAG  = "#888888"   # grey        — diagonal
C_BACK  = "#AECDE8"   # light blue  — backbone atoms
C_CYCLE = "#D65F5F"   # crimson     — cycle edges
C_NODE  = "#B22222"   # dark red    — cycle nodes


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Persistent Homology results (Script 7)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--plots", nargs="+",
        choices=ALL_PLOTS,
        default=ALL_PLOTS,
        metavar="PLOT",
        help=(
            "Which plot type(s) to generate. Choices: "
            + ", ".join(ALL_PLOTS)
            + "  (default: all)"
        ),
    )
    parser.add_argument(
        "--classes", nargs="+",
        default=None,
        metavar="CLASS",
        help="Protein class name(s) to process. Default: all detected classes.",
    )
    parser.add_argument(
        "--group", choices=["k", "s"],
        default=None,
        metavar="LETTER",
        help=(
            "Process only classes whose name starts with this letter "
            "(k or s). Overridden by --classes if both are given."
        ),
    )
    add_dataset_arg(parser)
    return parser.parse_args()


# ============================================================
# DATA LOADING
# ============================================================

def load_barcodes(barcode_json: Path) -> Tuple[List, List]:
    """Return (bar0, bar1) — lists of [birth, death_or_None]."""
    with open(barcode_json) as f:
        data = json.load(f)
    return data.get("dim_0", []), data.get("dim_1", [])


def load_ph1(ph1_json: Path) -> Dict:
    with open(ph1_json) as f:
        return json.load(f)


def load_coords(csv_path: Path) -> np.ndarray:
    """Return (N, 3) float array of Cα coordinates."""
    return pd.read_csv(csv_path, header=0).values.astype(float)


# ============================================================
# PLOT 1: PERSISTENCE DIAGRAM
# ============================================================

def plot_persistence_diagram(
    bar0: List, bar1: List, protein_id: str, out_path: Path
):
    """Scatter plot of (birth, death) points for H0 and H1."""

    def parse(bars):
        births, deaths = [], []
        for b in bars:
            if b[0] is not None:
                births.append(float(b[0]))
                deaths.append(float(b[1]) if b[1] is not None else np.inf)
        return np.array(births), np.array(deaths)

    b0, d0 = parse(bar0)
    b1, d1 = parse(bar1)

    all_finite = np.concatenate([
        d0[np.isfinite(d0)], d1[np.isfinite(d1)],
        b0, b1
    ])
    lo = float(all_finite.min()) if len(all_finite) else 0.0
    hi = float(all_finite.max()) * 1.05 if len(all_finite) else 1.0
    cap = hi * 1.1   # where to draw infinite bars

    d0 = np.where(np.isfinite(d0), d0, cap)
    d1 = np.where(np.isfinite(d1), d1, cap)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([lo, cap], [lo, cap], color=C_DIAG, lw=1, ls="--", label="diagonal", zorder=1)

    if len(b0):
        ax.scatter(b0, d0, c=C_H0, s=40, label="H₀", zorder=3,
                   edgecolors="white", linewidths=0.4)
    if len(b1):
        ax.scatter(b1, d1, c=C_H1, s=55, marker="D", label="H₁", zorder=4,
                   edgecolors="white", linewidths=0.4)

    # Mark infinite bars with a dashed line at cap
    inf_mask = d1 >= cap - 1e-9
    if inf_mask.any():
        ax.axhline(cap, color=C_H1, lw=0.8, ls=":", alpha=0.6, label="∞ death")

    ax.set_xlim(lo, cap)
    ax.set_ylim(lo, cap * 1.02)
    ax.set_xlabel("Birth", fontsize=12)
    ax.set_ylabel("Death", fontsize=12)
    ax.set_title(f"Persistence Diagram — {protein_id}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=DPI)
    plt.close(fig)


# ============================================================
# PLOT 2: BARCODE DIAGRAM
# ============================================================

def plot_barcodes(
    bar0: List, bar1: List, protein_id: str, out_path: Path
):
    """
    Compact horizontal barcode plot — fixed page size (9x7 inches) regardless
    of bar count.
    """

    def parse(bars):
        result = []
        for b in bars:
            if b[0] is not None:
                result.append((float(b[0]), b[1]))
        return result

    b0 = parse(bar0)
    b1 = parse(bar1)

    all_finite_deaths = [float(d) for _, d in b0 + b1 if d is not None]
    cap = max(all_finite_deaths) * 1.1 if all_finite_deaths else 1.0

    def get_death(d):
        return float(d) if d is not None else cap

    n0, n1 = len(b0), len(b1)

    fig, (ax0, ax1) = plt.subplots(
        2, 1,
        figsize=(9, 7),
        gridspec_kw={"height_ratios": [max(n0, 1), max(n1, 1)]},
    )
    fig.suptitle(f"Barcodes — {protein_id}", fontsize=12, fontweight="bold", y=0.98)

    lw0 = max(0.4, min(2.5, 40.0 / max(n0, 1)))
    lw1 = max(0.4, min(2.5, 40.0 / max(n1, 1)))

    for i, (birth, death) in enumerate(b0):
        y = (i + 0.5) / max(n0, 1)
        ax0.plot([birth, get_death(death)], [y, y],
                 lw=lw0, color=C_H0, alpha=0.8, solid_capstyle="butt")
    ax0.set_title(f"H0  ({n0} components)", fontsize=10, pad=3)
    ax0.set_xlabel("Filtration value", fontsize=9)
    ax0.set_yticks([])
    ax0.set_ylim(0, 1)
    ax0.grid(axis="x", alpha=0.2, lw=0.5)
    ax0.tick_params(axis="x", labelsize=8)

    if n1:
        persistences = [get_death(d) - b for b, d in b1]
        max_p = max(persistences) if persistences else 1.0
        cmap  = cm.get_cmap("Reds")
        for i, ((birth, death), pers) in enumerate(zip(b1, persistences)):
            y     = (i + 0.5) / max(n1, 1)
            alpha = 0.4 + 0.6 * (pers / max(max_p, 1e-10))
            color = cmap(0.35 + 0.55 * (pers / max(max_p, 1e-10)))
            ax1.plot([birth, get_death(death)], [y, y],
                     lw=lw1, color=color, alpha=alpha, solid_capstyle="butt")
    ax1.set_title(f"H1  ({n1} loops)", fontsize=10, pad=3)
    ax1.set_xlabel("Filtration value", fontsize=9)
    ax1.set_yticks([])
    ax1.set_ylim(0, 1)
    ax1.grid(axis="x", alpha=0.2, lw=0.5)
    ax1.tick_params(axis="x", labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, format="pdf", dpi=DPI)
    plt.close(fig)


# ============================================================
# PLOT 3: MOST PERSISTENT CYCLE (3D)
# ============================================================

def plot_most_persistent_cycle(
    ph1_data: Dict, coords: np.ndarray, protein_id: str, out_path: Path
):
    """3D scatter of Cα atoms with most persistent H1 cycle overlaid."""
    repre = ph1_data.get("representatives", [])
    bar1  = ph1_data.get("dim_1_barcode", [])

    if not repre or not bar1:
        return False

    def persistence(b):
        birth = float(b[0]) if b[0] is not None else 0.0
        death = float(b[1]) if b[1] is not None else birth + 1e9
        return death - birth

    best_idx   = int(np.argmax([persistence(b) for b in bar1]))
    best_edges = repre[best_idx]

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=C_BACK, s=12, alpha=0.45, linewidths=0,
        label="Cα backbone", depthshade=True,
    )

    plotted_label = False
    for edge in best_edges:
        if len(edge) >= 2:
            u, v = int(edge[0]) - 1, int(edge[1]) - 1
            if 0 <= u < len(coords) and 0 <= v < len(coords):
                lbl = "cycle edges" if not plotted_label else None
                ax.plot(
                    [coords[u, 0], coords[v, 0]],
                    [coords[u, 1], coords[v, 1]],
                    [coords[u, 2], coords[v, 2]],
                    color=C_CYCLE, lw=2.5, alpha=0.9, label=lbl,
                )
                plotted_label = True

    cycle_nodes = sorted({int(n) - 1 for edge in best_edges for n in edge
                          if 0 <= int(n) - 1 < len(coords)})
    if cycle_nodes:
        cn = coords[cycle_nodes]
        ax.scatter(cn[:, 0], cn[:, 1], cn[:, 2],
                   c=C_NODE, s=45, zorder=5, label="cycle nodes",
                   edgecolors="darkred", linewidths=0.5)

    pers = persistence(bar1[best_idx])
    ax.set_title(
        f"Most Persistent H₁ Cycle — {protein_id}\n"
        f"persistence = {pers:.3f}  |  {len(best_edges)} edges",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("X (Å)", fontsize=9)
    ax.set_ylabel("Y (Å)", fontsize=9)
    ax.set_zlabel("Z (Å)", fontsize=9)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
    ax.view_init(elev=25, azim=35)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=DPI)
    plt.close(fig)
    return True


# ============================================================
# PROCESS ONE PROTEIN
# ============================================================

def visualize_protein(
    protein_id: str,
    class_name: str,
    dirs:        Dict[str, Path],
    plots:       List[str],
) -> Dict:
    barcode_json = config.PH_DIR / class_name / "barcodes" / f"{protein_id}.json"
    ph1_json     = config.PH_DIR / class_name / "PH_1"     / f"{protein_id}.json"
    coord_csv    = config.COORDINATES_DIR / class_name      / f"{protein_id}.csv"

    try:
        if not barcode_json.exists():
            return {"protein_id": protein_id, "status": "missing_barcode"}

        bar0, bar1 = load_barcodes(barcode_json)

        if "persistent_diagram" in plots:
            plot_persistence_diagram(
                bar0, bar1, protein_id,
                dirs["persistent_diagram"] / f"{protein_id}.pdf"
            )

        if "barcodes" in plots:
            plot_barcodes(
                bar0, bar1, protein_id,
                dirs["barcodes"] / f"{protein_id}.pdf"
            )

        if "most_persistent_cycle" in plots:
            if ph1_json.exists() and coord_csv.exists():
                ph1_data = load_ph1(ph1_json)
                coords   = load_coords(coord_csv)
                plot_most_persistent_cycle(
                    ph1_data, coords, protein_id,
                    dirs["most_persistent_cycle"] / f"{protein_id}.pdf"
                )

        return {"protein_id": protein_id, "status": "ok"}

    except Exception as e:
        return {"protein_id": protein_id, "status": "failed", "error": str(e)}


# ============================================================
# PROCESS ONE CLASS
# ============================================================

def visualize_class(class_name: str, plots: List[str]) -> Dict:
    barcode_dir = config.PH_DIR / class_name / "barcodes"

    if not barcode_dir.exists():
        print(f"  ⚠  No barcodes dir for {class_name}")
        return {"total": 0, "ok": 0, "failed": 0}

    proteins = sorted(p.stem for p in barcode_dir.glob("*.json"))
    if not proteins:
        print(f"  ⚠  No barcode files in {class_name}")
        return {"total": 0, "ok": 0, "failed": 0}

    # Create only the output subdirectories that are needed
    base = config.VIZ_PH_DIR / class_name
    subdir_map = {
        "persistent_diagram":    "persistent_diagram",
        "barcodes":              "barcode_plots",
        "most_persistent_cycle": "most_persistent_cycle",
    }
    dirs = {}
    for key in plots:
        d = base / subdir_map[key]
        d.mkdir(parents=True, exist_ok=True)
        dirs[key] = d

    stats = {"total": len(proteins), "ok": 0, "failed": 0}
    lock  = threading.Lock()

    desc = f"  {class_name[:38]:<38}"
    with tqdm(total=len(proteins), desc=desc, unit="prot",
              bar_format="{l_bar}{bar:25}{r_bar}") as pbar:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = {
                ex.submit(visualize_protein, pid, class_name, dirs, plots): pid
                for pid in proteins
            }
            for future in as_completed(futures):
                result = future.result()
                with lock:
                    key = "ok" if result["status"] == "ok" else "failed"
                    stats[key] += 1
                pbar.update(1)
                pbar.set_postfix(ok=stats["ok"], fail=stats["failed"], refresh=False)

    tqdm.write(
        f"  ✅ {class_name}: "
        f"ok={stats['ok']}  failed={stats['failed']}  total={stats['total']}"
    )
    return stats


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    apply_dataset_from_args(args)

    # ── resolve which plots to run ──────────────────────────
    plots = args.plots  # already validated by argparse

    # ── discover available classes ──────────────────────────
    if not config.PH_DIR.exists():
        print(f"❌ PH directory not found: {config.PH_DIR}")
        print("   Run step 4 first.")
        sys.exit(1)

    all_classes = sorted([
        d.name for d in config.PH_DIR.iterdir()
        if d.is_dir() and (d / "barcodes").exists()
    ])

    if not all_classes:
        print(f"❌ No processed classes found in {config.PH_DIR}")
        sys.exit(1)

    # ── filter classes ──────────────────────────────────────
    if args.classes:
        # explicit list takes priority
        missing = [c for c in args.classes if c not in all_classes]
        if missing:
            print(f"⚠  The following requested classes were not found and will be skipped: {missing}")
        classes = [c for c in args.classes if c in all_classes]
    elif args.group:
        classes = [c for c in all_classes if c.lower().startswith(args.group.lower())]
        if not classes:
            print(f"❌ No classes found starting with '{args.group}'")
            sys.exit(1)
    else:
        classes = all_classes

    if not classes:
        print("❌ No classes to process after filtering.")
        sys.exit(1)

    # ── banner ──────────────────────────────────────────────
    print("=" * 60)
    print("SCRIPT 7: PH VISUALIZATIONS")
    print("=" * 60)
    print(f"Start   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset : {config.dataset}")
    print(f"Input   : {config.PH_DIR}")
    print(f"Output  : {config.VIZ_PH_DIR}")
    print(f"Coords  : {config.COORDINATES_DIR}")
    print(f"Plots   : {', '.join(plots)}")
    print(f"Classes : {', '.join(classes)}")
    print()

    overall = {"total": 0, "ok": 0, "failed": 0}

    for class_name in classes:
        print(f"\n{'─' * 60}")
        print(f"CLASS: {class_name}")
        print("─" * 60)
        stats = visualize_class(class_name, plots)
        for k in overall:
            overall[k] += stats[k]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Classes  : {len(classes)}")
    print(f"  Plots    : {', '.join(plots)}")
    print(f"  Total    : {overall['total']}")
    print(f"  Success  : {overall['ok']}")
    print(f"  Failed   : {overall['failed']}")
    print(f"\n✅ Done. End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
