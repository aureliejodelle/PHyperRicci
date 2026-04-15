#!/usr/bin/env python3
"""
run_visualizations.py  --  Visualization Pipeline
==================================================
Orchestrates the four visualization scripts in the correct order,
with dependency checking, skip-if-done logic, and full status reporting.

Steps
-----
  v1  visualize_ph.py
        Input : data/processed_data/Persistent_homology/  (step 4 output)
        Output: results/persistent_homology/
        Plots : persistence diagram, barcode, most-persistent cycle per protein
        Flags : --plots  --classes  --group k|s

  v2  extract_features.py
        Input : PH + hypergraph + Ricci JSON outputs (pipeline steps 4-6)
        Output: results/features/protein_features.csv
        Task  : 19 numerical features per protein

  v3  statistical_tests.py
        Input : protein_features.csv  (v2 output)
        Output: results/Features_Analysis/curvature_original_vs_homologs_*/
        Task  : KS + Levene tests, original vs homolog groups on Curv_median
        Flags : --three-way  --stats-only

  v4  analysis_proteins_vs_homologs.py
        Input : protein_features.csv  (v2 output)
        Output: results/Features_Analysis/<OriginalClass>/
        Task  : per-class comparison plots (violins, KDE, correlation)
        Flags : --class CLASS  --stats-only

Dependency chain
----------------
  step 4 (Julia PH)   ->  v1
  steps 4 + 5 + 6     ->  v2
  v2                  ->  v3, v4

Usage
-----
  python run_visualizations.py                        # run all steps
  python run_visualizations.py --step v1              # single step
  python run_visualizations.py --step v1 --force      # re-run even if done
  python run_visualizations.py --status               # show pipeline status

  # v1 options
  python run_visualizations.py --step v1 --plots barcodes
  python run_visualizations.py --step v1 --classes K41 Kplus31
  python run_visualizations.py --step v1 --group k

  # v3 options
  python run_visualizations.py --step v3 --three-way
  python run_visualizations.py --step v3 --stats-only

  # v4 options
  python run_visualizations.py --step v4 --class AOTCases
  python run_visualizations.py --step v4 --stats-only
"""

import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import config


# ================================================================
# STEP REGISTRY
# ================================================================
STEPS = {

    "v1": {
        "script":      "visualize_ph.py",
        "description": "Persistence diagrams, barcodes & cycle plots",
        "depends_on":  [],
        "input_check": {
            "type": "dir_json",
            "path": lambda: config.PH_DIR,
            "hint": "Run pipeline step 4 (Julia PH) first.",
        },
        "done_check": {
            "type": "dir_pdf",
            "path": lambda: config.VIZ_PH_DIR,
        },
        "output_dir":  lambda: config.VIZ_PH_DIR,
        "estimated":   "5-20 min  (one plot set per protein)",
        "accepts":     {"plots": True, "classes": True, "group": True},
    },

    "v2": {
        "script":      "extract_features.py",
        "description": "Extract 19 per-protein features -> protein_features.csv",
        "depends_on":  [],
        "input_check": {
            "type": "dir_json",
            "path": lambda: config.PH_DIR,
            "hint": "Run pipeline steps 4 5 6 first.",
        },
        "done_check": {
            "type": "file",
            "path": lambda: config.VIZ_FEATURES_DIR / "protein_features.csv",
        },
        "output_dir":  lambda: config.VIZ_FEATURES_DIR,
        "estimated":   "1-5 min",
        "accepts":     {},
    },

    "v3": {
        "script":      "statistical_tests.py",
        "description": "KS + Levene tests: original vs homolog groups on Curv_median",
        "depends_on":  ["v2"],
        "input_check": {
            "type": "file",
            "path": lambda: config.VIZ_FEATURES_DIR / "protein_features.csv",
            "hint": "Run step v2 first.",
        },
        "done_check": {
            "type": "dir_pdf",
            "path": lambda: config.VIZ_COMPARE_DIR / "curvature_original_vs_homologs_two_group",
        },
        "output_dir":  lambda: config.VIZ_COMPARE_DIR,
        "estimated":   "2-5 min",
        "accepts":     {"three_way": True, "stats_only": True},
    },

    "v4": {
        "script":      "analysis_proteins_vs_homologs.py",
        "description": "Per-class comparison plots: original vs homolog groups",
        "depends_on":  ["v2"],
        "input_check": {
            "type": "file",
            "path": lambda: config.VIZ_FEATURES_DIR / "protein_features.csv",
            "hint": "Run step v2 first.",
        },
        "done_check": {
            "type": "dir_pdf",
            "path": lambda: config.VIZ_COMPARE_DIR,
        },
        "output_dir":  lambda: config.VIZ_COMPARE_DIR,
        "estimated":   "5-15 min  (one sub-folder per class pair)",
        "accepts":     {"classes": True, "stats_only": True},
    },
}

RUN_ORDER = ["v1", "v2", "v3", "v4"]


# ================================================================
# HELPERS
# ================================================================

def _is_done(sid):
    spec = STEPS[sid]["done_check"]
    path = spec["path"]()
    if spec["type"] == "file":
        return path.exists() and path.stat().st_size > 0
    pat = "**/*.pdf" if spec["type"] == "dir_pdf" else "**/*.json"
    return path.exists() and any(path.rglob(pat))


def _input_ready(sid):
    spec = STEPS[sid]["input_check"]
    path = spec["path"]()
    hint = spec.get("hint", "")
    if spec["type"] == "file":
        if not path.exists():
            return False, "Missing: {}  ({})".format(path, hint)
        if path.stat().st_size == 0:
            return False, "Empty file: {}".format(path)
        return True, ""
    if not path.exists():
        return False, "Directory missing: {}  ({})".format(path, hint)
    if not any(path.rglob("*.json")):
        return False, "No JSON files in: {}  ({})".format(path, hint)
    return True, ""


def _missing_deps(sid):
    return [d for d in STEPS[sid]["depends_on"] if not _is_done(d)]


def _count_outputs(sid):
    d = STEPS[sid]["output_dir"]()
    if not d.exists():
        return {"pdfs": 0, "csvs": 0}
    return {
        "pdfs": len(list(d.rglob("*.pdf"))),
        "csvs": len(list(d.rglob("*.csv"))),
    }


def _build_extra_args(sid, args):
    """Translate user CLI args into flags forwarded to the child script."""
    accepts = STEPS[sid].get("accepts", {})
    extra = []

    # --plots (v1 only)
    if "plots" in accepts and args.plots:
        extra += ["--plots"] + args.plots

    # --classes (v1: --classes; v4: --class single value)
    if "classes" in accepts and args.classes:
        if sid == "v4":
            extra += ["--class", args.classes[0]]
            if len(args.classes) > 1:
                print("  Note: v4 (analysis_proteins_vs_homologs.py) takes one class "
                      "at a time. Using: {}".format(args.classes[0]))
        else:
            extra += ["--classes"] + args.classes

    # --group (v1 only)
    if "group" in accepts and args.group:
        extra += ["--group", args.group]

    # --three-way (v3 only)
    if "three_way" in accepts and args.three_way:
        extra += ["--three-way"]

    # --stats-only (v3 and v4)
    if "stats_only" in accepts and args.stats_only:
        extra += ["--stats-only"]

    return extra or None


# ================================================================
# EXECUTION
# ================================================================

def _run_step(sid, force=False, extra_args=None):
    spec = STEPS[sid]
    print()
    print("=" * 65)
    print("STEP {}:  {}".format(sid, spec["description"]))
    print("=" * 65)

    ready, reason = _input_ready(sid)
    if not ready:
        print("\nBlocked -- {}".format(reason))
        return False

    if not force and _is_done(sid):
        c = _count_outputs(sid)
        print("\nStep {} already done ({} PDFs, {} CSVs).".format(
            sid, c["pdfs"], c["csvs"]))
        print("   Output : {}".format(spec["output_dir"]()))
        print("   Use --force to re-run.")
        return True

    print("Estimated : {}".format(spec["estimated"]))
    print("Script    : {}".format(spec["script"]))
    print("Output    : {}".format(spec["output_dir"]()))
    if extra_args:
        print("Extra args: {}".format(" ".join(extra_args)))
    print()

    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = config.LOGS_DIR / "viz_{}_{}.log".format(sid, ts)
    cmd = [sys.executable, str(Path(__file__).parent / spec["script"])]
    if extra_args:
        cmd += extra_args

    try:
        with open(log, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                cwd=str(Path(__file__).parent),
            )
            for line in proc.stdout:
                print(line, end="", flush=True)
                lf.write(line)
            proc.wait()

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

        c = _count_outputs(sid)
        print("\nStep {} done -- {} PDFs, {} CSVs.".format(sid, c["pdfs"], c["csvs"]))
        print("Log: {}".format(log))
        return True

    except subprocess.CalledProcessError as e:
        print("\nStep {} failed (exit {}).".format(sid, e.returncode))
        print("Log: {}".format(log))
        return False
    except Exception as e:
        print("\nUnexpected error in {}: {}".format(sid, e))
        return False


def _ensure_deps(sid, force):
    missing = _missing_deps(sid)
    if not missing:
        return True
    print("\nStep {} requires: {}".format(sid, ", ".join(missing)))
    for dep in missing:
        print("\nAuto-running dependency: {}".format(dep))
        if not _run_step(dep, force):
            print("\nDependency {} failed -- cannot proceed.".format(dep))
            return False
    return True


def run_one(sid, force=False, extra_args=None):
    if sid not in STEPS:
        print("Unknown step '{}'. Valid: {}".format(sid, list(STEPS.keys())))
        return False
    if not _ensure_deps(sid, force):
        return False
    return _run_step(sid, force, extra_args)


def run_all(force=False):
    print()
    print("=" * 65)
    print("VISUALIZATION PIPELINE -- ALL STEPS")
    print("=" * 65)
    print("Start  : {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print(config.summary())
    print()
    _print_plan()

    for sid in RUN_ORDER:
        if not _run_step(sid, force):
            print("\nStopping after failed step {}.".format(sid))
            _print_footer(ok=False)
            return False

    _print_footer(ok=True)
    return True


# ================================================================
# STATUS / DISPLAY
# ================================================================

def _print_plan():
    print("  {:<6}  {:<44}  Status".format("Step", "Script"))
    print("  " + "-" * 60)
    for sid in RUN_ORDER:
        done  = _is_done(sid)
        ready, _ = _input_ready(sid)
        if done:
            status = "done"
        elif not ready:
            status = "blocked"
        else:
            status = "pending"
        print("  {:<6}  {:<44}  {}".format(sid, STEPS[sid]["script"], status))
    print()


def _print_footer(ok):
    print()
    print("=" * 65)
    print("ALL STEPS COMPLETED" if ok else "PIPELINE STOPPED DUE TO ERROR")
    print("End  : {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("=" * 65)


def print_status():
    print()
    print("=" * 65)
    print("VISUALIZATION PIPELINE STATUS")
    print("=" * 65)
    print()
    print(config.summary())
    print("""
Dependency tree:

  pipeline step 4  (Julia PH)
       |
       +-->  v1  visualize_ph.py
                 results/persistent_homology/<class>/
                 Flags: --plots  --classes  --group k|s

  pipeline steps 4 + 5 + 6
       |
       +-->  v2  extract_features.py
                 results/features/protein_features.csv
                      |
                      +-->  v3  statistical_tests.py
                      |        results/Features_Analysis/curvature_original_vs_homologs_*/
                      |        Flags: --three-way  --stats-only
                      |
                      +-->  v4  analysis_proteins_vs_homologs.py
                               results/Features_Analysis/<OriginalClass>/
                               Flags: --class CLASS  --stats-only
""")

    for sid in RUN_ORDER:
        spec  = STEPS[sid]
        done  = _is_done(sid)
        ready, reason = _input_ready(sid)
        miss  = _missing_deps(sid)
        accepts = spec.get("accepts", {})

        print("-" * 65)
        print("Step {}  {}".format(sid, spec["script"]))
        print("  {}".format(spec["description"]))
        print("  Status       : {}".format("Completed" if done else "Not done"))
        print("  Input ready  : {}".format("Yes" if ready else "NO -- " + reason))
        print("  Dependencies : {}".format(
            "OK" if not miss else "Missing: " + ", ".join(miss)))
        if accepts:
            flags = []
            if "plots" in accepts:
                flags.append("--plots PLOT [PLOT ...]")
            if "classes" in accepts:
                label = "--class CLASS" if sid == "v4" else "--classes CLASS [CLASS ...]"
                flags.append(label)
            if "group" in accepts:
                flags.append("--group k|s")
            if "three_way" in accepts:
                flags.append("--three-way")
            if "stats_only" in accepts:
                flags.append("--stats-only")
            print("  Accepts      : {}".format("  ".join(flags)))
        if done:
            c = _count_outputs(sid)
            print("  Output dir   : {}".format(spec["output_dir"]()))
            print("  Files found  : {} PDFs, {} CSVs".format(c["pdfs"], c["csvs"]))
        else:
            print("  Expected out : {}".format(spec["output_dir"]()))
        print("  Est. runtime : {}".format(spec["estimated"]))

    print()
    print("-" * 65)
    print("Common commands:")
    print("  python run_visualizations.py                          # All steps")
    print("  python run_visualizations.py --step v1               # PH plots")
    print("  python run_visualizations.py --step v1 --plots barcodes")
    print("  python run_visualizations.py --step v1 --group k")
    print("  python run_visualizations.py --step v1 --classes K41 Kplus31")
    print("  python run_visualizations.py --step v2               # Feature extraction")
    print("  python run_visualizations.py --step v3               # KS + Levene tests")
    print("  python run_visualizations.py --step v3 --three-way")
    print("  python run_visualizations.py --step v4               # Comparison plots (all classes)")
    print("  python run_visualizations.py --step v4 --class AOTCases")
    print("  python run_visualizations.py --force                 # Re-run all")
    print("  python run_visualizations.py --status                # This screen")
    print("=" * 65)


# ================================================================
# ENTRY POINT
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualization pipeline -- protein knot analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--step", choices=list(STEPS.keys()),
        help="Run a single step (dependencies auto-satisfied).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if outputs already exist.",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show pipeline status and exit.",
    )

    # ── flags forwarded to child scripts ──────────────────────
    parser.add_argument(
        "--plots", nargs="+",
        metavar="PLOT",
        default=None,
        help="(v1) Plot types: persistent_diagram  barcodes  most_persistent_cycle",
    )
    parser.add_argument(
        "--classes", nargs="+",
        metavar="CLASS",
        default=None,
        help="(v1) Class name(s) to process. (v4) Uses only the first value.",
    )
    parser.add_argument(
        "--class", dest="classes", nargs=1,
        metavar="CLASS",
        help="(v4) Single class to process (alias for --classes with one value).",
    )
    parser.add_argument(
        "--group", choices=["k", "s"],
        metavar="LETTER",
        default=None,
        help="(v1 only) Process classes starting with 'k' or 's'.",
    )
    parser.add_argument(
        "--three-way", dest="three_way", action="store_true",
        help="(v3 only) Include 3-group comparison (original vs unknotted vs knotprot).",
    )
    parser.add_argument(
        "--stats-only", dest="stats_only", action="store_true",
        help="(v3, v4) Print statistics without generating plots.",
    )

    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.step:
        extra = _build_extra_args(args.step, args)
        ok = run_one(args.step, args.force, extra)
    else:
        if any([args.plots, args.classes, args.group, args.three_way, args.stats_only]):
            print("Note: step-specific flags (--plots, --classes, --group, "
                  "--three-way, --stats-only) are only forwarded when using --step.\n"
                  "Running full pipeline without filters.")
        ok = run_all(args.force)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
