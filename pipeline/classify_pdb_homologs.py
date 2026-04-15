
"""
classify_pdb_homologs.py
------------------------
For each original protein class, iterates over the proteins in the
corresponding *_PDB_Homologs folder, uses Topoly to determine whether
each protein is knotted or unknotted, then copies its coordinate file
and all processed-data files into the appropriate destination:

  Knotted  -- class_name_KnotProt_Homologs
  Unknotted -- class_name_Unknotted_Homologs

Directories handled
-----------------------
  raw_data:
    phd/Database/raw_data/coordinates_data/<class>/
    phd/Database/raw_data/coordinates_data/<class>_PDB_Homologs/   ← source

  processed_data:
    hypergraphs/<class>/hyperedge_map/<pid>.json
    Persistent_homology/<class>/barcodes/<pid>.json
    Persistent_homology/<class>/PH_1/<pid>.json
    Persistent_homology/<class>/representatives/<pid>.json   (if present)
    ricci_curvature/<class>/<pid>.json

Usage
--------
  python classify_pdb_homologs.py                        # all classes
  python classify_pdb_homologs.py --dry-run              # preview only
  python classify_pdb_homologs.py --classes K41 S41      # specific classes
  python classify_pdb_homologs.py --base /path/to/phd/Database
"""

from __future__ import annotations

import sys
import json
import shutil
import argparse
import traceback
from pathlib import Path
from datetime import datetime

# --- Topoly import -----------------------------------------------------------------------------
try:
    import topoly          # noqa: F401  - just check it is installed
    TOPOLY_OK = True
except ImportError:
    TOPOLY_OK = False
    print("WARNING: topoly not installed. Run:  pip install topoly")

# --- Project layout ------------------------------------------------------

# Knot types treated as unknotted per supervisor definition
UNKNOTTED_TYPES = {"0_1", "2_1", "2_1s"}
_HERE        = Path(__file__).resolve().parent
_DEFAULT_BASE = _HERE / "../../Database"

# --- Class folders that have PDB_Homologs to classify --------------------
# Keys   = folder names on disk (same in raw_data and processed_data)
# Values = human-readable label used in log messages
ALL_CLASSES = {
    "K41":     "K4(1)",
    "S41":     "S4(1)",
    "Kplus31": "K+3(1)",
    "Splus31": "S+3(1)",
}

# --- Processed-data subtrees and their sub-folders ---------------------------
# Each value is a list of sub-folder names inside processed_data/<tree>/<class>/
# An empty list means files sit directly in the class folder.
PROC_SUBTREES: dict[str, list[str]] = {
    "hypergraphs":        ["hyperedge_map"],
    "Persistent_homology": ["barcodes", "PH_1", "representatives"],
    "ricci_curvature":    [],           # files sit directly in <class>/
}


# ----------------------------------------------------------------------------
# Knot detection
# ------------------------------------------------------------------------------

def is_knotted(pdb_path: Path) -> str | None:
    """
    Run Topoly on a .pdb, .xyz, or .csv coordinate file and decide if the protein is knotted.

    Returns True  if knotted,
            False if unknotted,
            None  if Topoly raises an exception (protein is logged as error).

    --- How Topoly returns results --------------------------------------
    The invariant functions (alexander, jones, etc.) have two return modes
    depending on the `closure` parameter:

      Deterministic closure (CLOSED=0, MASS_CENTER=1, DIRECTION=5):
       -- returns a single topology string, e.g. "3_1" or "0_1"

      Stochastic closure (TWO_POINTS=2 [default], ONE_POINT=3, RAYS=4):
        -- returns a dict of {topology_string: probability}, e.g.
          {"3_1": 0.85, "0_1": 0.15}
          The probabilities reflect how often each topology appeared
          across 'tries' (default 200) random closures of the open chain.

    We use the Alexander polynomial (fastest and most reliable for
    proteins) with the default stochastic TWO_POINTS closure and
    tries=200. This gives a probability distribution. We pick the
    topology with the highest probability (the mode) and classify
    based on that.

    --- Unknotted types (supervisor definition) -------------------------
        0_1   — trivial knot
        2_1   — Hopf link (can appear with probabilistic closure)
        2_1s  — slipknot variant sometimes returned by Topoly

    Any other type (3_1, 4_1, 3_1s, 4_1s, etc.) is KNOTTED.

    Note: 'hide_trivial=True' (Topoly default) suppresses 0_1 from the
    dict when other topologies are present. We pass hide_trivial=False
    so we always see the full distribution, making the dominant type
    unambiguous.
    """
    if not TOPOLY_OK:
        raise RuntimeError("topoly is not installed")

    # Knot types that count as unknotted (supervisor-defined)
    UNKNOTTED_TYPES = {"0_1", "2_1", "2_1s"}

    try:
        from topoly import alexander

        # --- Read coordinates into a Python list of lists ------------------
        # Topoly accepts coordinates directly as [[x,y,z], ...] which
        # avoids any file I/O issues (no temp files, no header problems).
        with open(pdb_path) as _f:
            raw_lines = _f.readlines()

        # Auto-detect delimiter: use comma if present, otherwise whitespace
        sample = next((l for l in raw_lines if l.strip()), "")
        delimiter = "," if "," in sample else None   # None -- split() on whitespace

        coords = []
        for line in raw_lines:
            parts = line.strip().split(delimiter)
            # Strip quotes and whitespace from each token (common in CSVs)
            parts = [p.strip().strip('"').strip("'") for p in parts]
            if len(parts) < 3:
                continue
            try:
                coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                # Skip header rows like "x,y,z" or comment lines
                continue

        if not coords:
            print(f" No valid coordinate rows in {pdb_path.name}")
            return None

        # --- Run Alexander polynomial ---------------------
        # Stochastic TWO_POINTS closure, tries=200.
        # hide_trivial=False so 0_1 always appears in the probability dict.
        result = alexander(
            coords,
            closure=2,          # Closure.TWO_POINTS (stochastic, recommended for proteins)
            tries=200,
            hide_trivial=False, # keep 0_1 visible in the probability dict
            translate=True,     # return topology name strings, not raw polynomial
        )

        # --- Stochastic closure -- dict {topology: probability} --------------------
        if isinstance(result, dict):
            if not result:
                print(f"        ⚠  Empty result for {pdb_path.name}")
                return None
            # Pick the topology with the highest probability
            dominant = max(result, key=result.get)
            prob      = result[dominant]
            print(f"topology: {result}  -- dominant: {dominant} ({prob:.1%})")
            return dominant

        # --- Deterministic closure -- single string ---------------------------
        if isinstance(result, str):
            print(f"topology: {result}")
            return result

        print(f"Unexpected Topoly return type {type(result)} for {pdb_path.name}")
        return None

    except Exception as exc:
        print(f"Topoly error for {pdb_path.name}: {exc}")
        return None


# ------------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------------

def copy_file(src: Path, dst_dir: Path, dry_run: bool) -> bool:
    """Copy src to dst_dir/src.name. Returns True if file existed."""
    if not src.exists():
        return False
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
    return True


def copy_proc_files(pid: str, src_class: str, dst_class: str,
                    proc_root: Path, dry_run: bool) -> dict[str, int]:
    """
    Copy all processed-data files for protein 'pid' from
    proc_root/<tree>/src_class/[sub]/ -- proc_root/<tree>/dst_class/[sub]/
    Returns a counter dict {tree: files_copied}.
    """
    counts: dict[str, int] = {}
    for tree, subs in PROC_SUBTREES.items():
        tree_src = proc_root / tree / src_class
        tree_dst = proc_root / tree / dst_class
        n = 0
        if not subs:
            # files directly in class folder
            src = tree_src / f"{pid}.json"
            if copy_file(src, tree_dst, dry_run):
                n += 1
        else:
            for sub in subs:
                src = tree_src / sub / f"{pid}.json"
                if copy_file(src, tree_dst / sub, dry_run):
                    n += 1
        counts[tree] = n
    return counts


# --------------------------------------------------------------------------------
# Per-class processing
# ---------------------------------------------------------------------------------------------

def process_class(class_folder: str, base: Path, dry_run: bool) -> dict:
    """
    Classify all proteins in <class_folder>_PDB_Homologs and route them
    to <class_folder>_KnotProt_Homologs or <class_folder>_Unknotted_Homologs.
    Returns a summary dict.
    """
    coord_root = base / "raw_data"  / "coordinates_data"
    proc_root  = base / "processed_data"

    src_dir     = coord_root / f"{class_folder}_PDB_Homologs"
    knotted_dir = coord_root / f"{class_folder}_KnotProt_Homologs"
    unknot_dir  = coord_root / f"{class_folder}_Unknotted_Homologs"

    summary = {
        "class":        class_folder,
        "knotted":      [],
        "unknotted":    [],
        "errors":       [],
        "skipped":      [],
        "topology_map": {},   # {pid: {"topology": ..., "class": ...}}
    }

    if not src_dir.exists():
        print(f" Source folder not found, skipping: {src_dir}")
        summary["errors"].append(f"Source folder missing: {src_dir}")
        return summary

    # Collect all coordinate files (.pdb, .xyz, .csv)
    coord_files = sorted(
        f for f in src_dir.iterdir()
        if f.suffix.lower() in (".pdb", ".xyz", ".csv") and f.is_file()
    )

    if not coord_files:
        print(f"  ⚠  No .pdb/.xyz/.csv files found in {src_dir}")
        return summary

    print(f"\n  CLASS: {class_folder}  ({len(coord_files)} PDB_Homologs proteins)")
    print(f"    src  -- {src_dir}")
    print(f"    knotted  -- {knotted_dir}")
    print(f"    unknot   -- {unknot_dir}")

    for pdb_path in coord_files:
        pid = pdb_path.stem    # protein ID without extension

        # --- Run Topoly -----------------------------------------------------
        topology = is_knotted(pdb_path)

        if topology is None:
            print(f"    [ERROR]   {pid}")
            summary["errors"].append(pid)
            continue

        is_unknotted = topology in UNKNOTTED_TYPES
        label = "unknotted" if is_unknotted else f"KNOTTED ({topology})"
        print(f"[{label}] {pid}")

        # --- Determine destination -----------------------------------------------------------
        dst_class  = (f"{class_folder}_Unknotted_Homologs"
                      if is_unknotted else
                      f"{class_folder}_KnotProt_Homologs")
        dst_coord  = unknot_dir if is_unknotted else knotted_dir

        # --- Copy coordinate file -----------------------------------------------------------
        copy_file(pdb_path, dst_coord, dry_run)

        # --- Copy processed-data files ------------------------------------------
        proc_counts = copy_proc_files(
            pid        = pid,
            src_class  = f"{class_folder}_PDB_Homologs",
            dst_class  = dst_class,
            proc_root  = proc_root,
            dry_run    = dry_run,
        )
        total_proc = sum(proc_counts.values())
        if total_proc == 0:
            print(f"(no processed-data files found for {pid})")
        else:
            detail = "  ".join(
                f"{t}:{n}" for t, n in proc_counts.items() if n > 0
            )
            action = "would copy" if dry_run else "copied"
            print(f"processed: {action} {total_proc} files  [{detail}]")

        # --- Track -----------------------------------------------------------------------
        # topology_map: flat {pid: {"topology": ..., "class": ...}} for JSON output
        summary["topology_map"][pid] = {
            "topology": topology,
            "class":    class_folder,
        }
        if is_unknotted:
            summary["unknotted"].append(pid)
        else:
            summary["knotted"].append(pid)

    return summary


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Classify proteins in *_PDB_Homologs folders as knotted/unknotted "
            "using Topoly and copy them into the appropriate destination folders."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base", type=Path, default=_DEFAULT_BASE,
        help="Path to phd/Database/ (default: sibling of this script)"
    )
    parser.add_argument(
        "--classes", nargs="+", default=list(ALL_CLASSES.keys()),
        metavar="CLASS",
        help=(
            f"Class folder names to process. "
            f"Default: all ({', '.join(ALL_CLASSES)})"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be done without copying any files"
    )
    args = parser.parse_args()

    if not TOPOLY_OK:
        print("ERROR: topoly is required. Install with:  pip install topoly")
        sys.exit(1)

    unknown_classes = set(args.classes) - set(ALL_CLASSES)
    if unknown_classes:
        print(f"ERROR: Unknown class(es): {', '.join(sorted(unknown_classes))}")
        print(f"Known classes: {', '.join(ALL_CLASSES)}")
        sys.exit(1)

    print("=" * 70)
    print("CLASSIFY PDB HOMOLOGS → KnotProt_Homologs / Unknotted_Homologs")
    print("=" * 70)
    print(f"Start    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base     : {args.base}")
    print(f"Classes  : {', '.join(args.classes)}")
    print(f"Dry-run  : {args.dry_run}")

    # --- Validate base path -------------------------------------------------------
    coord_root = args.base / "raw_data" / "coordinates_data"
    proc_root  = args.base / "processed_data"
    if not coord_root.exists():
        print(f"\nERROR: coordinates_data folder not found:\n  {coord_root}")
        sys.exit(1)

    # --- Process each class ------------------------------------------------------------
    all_summaries = []
    for cls in args.classes:
        try:
            summary = process_class(cls, args.base, args.dry_run)
            all_summaries.append(summary)
        except Exception:
            print(f"\n  UNEXPECTED ERROR processing {cls}:")
            traceback.print_exc()

    # --- Print summary ---------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    total_k = total_u = total_e = 0
    for s in all_summaries:
        k = len(s["knotted"])
        u = len(s["unknotted"])
        e = len(s["errors"])
        total_k += k; total_u += u; total_e += e
        print(
            f"  {s['class']:<14}"
            f"  knotted: {k:>4}   unknotted: {u:>4}   errors: {e:>3}"
        )
    print(f"  {'TOTAL':<14}  knotted: {total_k:>4}   unknotted: {total_u:>4}"
          f"   errors: {total_e:>3}")

    if args.dry_run:
        print("\n  ℹ  Dry-run - nothing was written. "
              "Re-run without --dry-run to apply.")

    print(f"\nDone. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Save JSON outputs ------------------------------------------------------
    if not args.dry_run:
        # 1. Flat topology map:  {pid: {"topology": "...", "class": "..."}}
        flat_map: dict = {}
        for s in all_summaries:
            flat_map.update(s.get("topology_map", {}))

        topology_path = args.base / "topology_classification.json"
        with open(topology_path, "w") as fh:
            json.dump(flat_map, fh, indent=2)
        print(f"Topology map -- {topology_path}")

        # 2. Detailed run log with full summary per class
        log_path = args.base / "classify_pdb_homologs_log.json"
        with open(log_path, "w") as fh:
            json.dump(
                {
                    "run_at":  datetime.now().isoformat(),
                    "classes": args.classes,
                    "results": [
                        {k: v for k, v in s.items() if k != "topology_map"}
                        for s in all_summaries
                    ],
                },
                fh, indent=2,
            )
        print(f"Run log -- {log_path}")


if __name__ == "__main__":
    main()
