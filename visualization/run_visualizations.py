#!/usr/bin/env python3
"""
run_visualizations.py  Visualization Pipeline
==============================================
Orchestrates all visualization scripts in the correct order,
with dependency checking, skip-if-done logic, and full status reporting.

Steps
-----
  v1  visualize_ph.py
        Input : processed_data/Persistent_homology/  (step 4 output)
        Output: results/persistent_homology/
        Plots : persistence diagram, barcode, most-persistent cycle per protein

  v2  extract_features.py
        Input : PH + HG + RC JSON outputs (pipeline steps 4-6)
        Output: results/features/protein_features.csv
        Task  : 19 numerical features per protein, parallelised

  v2b extract_topological_features.py  [NEW — run after v2]
        Input : barcodes JSON + Cα coordinate CSVs
        Output: results/features/protein_features_extended.csv
        Task  : writhe, persistence entropy, persistent image PCA
                (better descriptors for knotting than Ricci curvature)

  v3  statistical_analysis.py
        Input : protein_features.csv
        Output: results/comparison/<OriginalClass>/

  v4  compare_classes.py
        Input : protein_features.csv
        Output: results/comparison/cross_class/

  v5  advanced_statistics.py
  v6  wasserstein_analysis.py
  v7  paired_statistical_analysis.py
  v8  cycle_curvature_analysis.py
  v9  size_control_analysis.py
  v10 random_control.py
  v11 paper_table.py
  v12 composite_main_figure.py

Dependency chain
----------------
  step 4 (Julia PH)       ->  v1
  steps 4 + 5 + 6         ->  v2
  v2                      ->  v3, v4, v5, v6
  v7                      ->  v9, v11
  v8                      ->  v10, v12

Usage
-----
  # Run everything:
  python run_visualizations.py

  # Run a single step:
  python run_visualizations.py --step v1
  python run_visualizations.py --step v4

  # Re-run even if outputs exist:
  python run_visualizations.py --force
  python run_visualizations.py --step v1 --force

  # Show pipeline status:
  python run_visualizations.py --status

  ── v1 (visualize_ph.py) specific options ──────────────────────────
  # Only barcode plots, all classes:
  python run_visualizations.py --step v1 --plots barcodes

  # Only persistence diagram + cycle, two classes:
  python run_visualizations.py --step v1 --plots persistent_diagram most_persistent_cycle --classes AOTCases OTCases

  # All plots for classes starting with 'k':
  python run_visualizations.py --step v1 --group k

  ── v3 (statistical_analysis.py) specific options ──────────────────
  # Single class only:
  python run_visualizations.py --step v3 --classes AOTCases

  ── v4 (compare_classes.py) specific options ───────────────────────
  # Only KDE and violin plots:
  python run_visualizations.py --step v4 --plots kde violin_boxplot

  # Group-split (k-classes vs s-classes):
  python run_visualizations.py --step v4 --group-split

  # Specific classes + specific plots + group-split:
  python run_visualizations.py --step v4 --classes AOTCases OTCases --plots kde scatter --group-split

  ── Notes ───────────────────────────────────────────────────────────
  --plots   applies to v1 and v4 (different valid values per step — see below)
  --classes applies to v1, v3, and v4
  --group   applies to v1 only  (k or s)
  --group-split  applies to v4 only

  v1 --plots choices : persistent_diagram  barcodes  most_persistent_cycle
  v4 --plots choices : kde  violin_boxplot  scatter  volcano  overview
"""

import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import config


# ================================================================
# PLOT CHOICES (for validation)
# ================================================================
V1_PLOTS = {"persistent_diagram", "barcodes", "most_persistent_cycle"}
V4_PLOTS = {"kde", "violin_boxplot", "scatter", "volcano", "overview"}


# ================================================================
# HELPERS FOR STEP REGISTRY
# ================================================================

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
        "estimated":   "5-20 min  (parallel, one plot set per protein)",
        "accepts":     {"plots": V1_PLOTS, "classes": True, "group": True},
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
        "estimated":   "1-5 min   (parallel, 8 workers)",
        "accepts":     {},
    },

    "v2b": {
        "script":      "extract_topological_features.py",
        "description": "Writhe + persistence entropy + persistent image PCA features",
        "depends_on":  ["v2"],
        "input_check": {
            "type": "file",
            "path": lambda: config.VIZ_FEATURES_DIR / "protein_features.csv",
            "hint": "Run step v2 first.",
        },
        "done_check": {
            "type": "file",
            "path": lambda: config.VIZ_FEATURES_DIR / "protein_features_extended.csv",
        },
        "output_dir":  lambda: config.VIZ_FEATURES_DIR,
        "estimated":   "10-30 min  (writhe is O(N²) per protein)",
        "accepts":     {"classes": True},
    },

    "v3": {
        "script":      "statistical_analysis.py",
        "description": "Per-class stats: violins, volcano, heatmap, radar, KDE ...",
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
        "estimated":   "5-15 min  (one sub-folder per original class)",
        "accepts":     {"classes": True},   # passes as --class (legacy single value)
    },

    "v4": {
        "script":      "compare_classes.py",
        "description": "Cross-class comparison: PCA, heatmap, radar, pairwise",
        "depends_on":  ["v2"],
        "input_check": {
            "type": "file",
            "path": lambda: config.VIZ_FEATURES_DIR / "protein_features.csv",
            "hint": "Run step v2 first.",
        },
        "done_check": {
            "type": "dir_pdf",
            "path": lambda: config.VIZ_COMPARE_DIR / "cross_class",
        },
        "output_dir":  lambda: config.VIZ_COMPARE_DIR / "cross_class",
        "estimated":   "2-10 min",
        "accepts":     {"plots": V4_PLOTS, "classes": True, "group_split": True, "include_homologs": True},
    },

    "v5": {
        "script":      "advanced_statistics.py",
        "description": "KS curvature (+ median) + persistent entropy",
        "depends_on":  ["v2"],
        "input_check": {
            "type": "file",
            "path": lambda: config.VIZ_FEATURES_DIR / "protein_features.csv",
            "hint": "Run step v2 first.",
        },
        "done_check": {
            "type": "dir_pdf",
            "path": lambda: config.VIZ_COMPARE_DIR / "advanced" / "ks_curvature",
        },
        "output_dir":  lambda: config.VIZ_COMPARE_DIR / "advanced",
        "estimated":   "2-10 min",
        "accepts":     {"classes": True},
    },

    "v6": {
        "script":      "wasserstein_analysis.py",
        "description": "Wasserstein persistence diagram distances (slow — O(n²))",
        "depends_on":  ["v2"],
        "input_check": {
            "type": "file",
            "path": lambda: config.VIZ_FEATURES_DIR / "protein_features.csv",
            "hint": "Run step v2 first.",
        },
        "done_check": {
            "type": "dir_pdf",
            "path": lambda: config.VIZ_COMPARE_DIR / "advanced" / "wasserstein",
        },
        "output_dir":  lambda: config.VIZ_COMPARE_DIR / "advanced" / "wasserstein",
        "estimated":   "30 min – several hours  (O(n²) Wasserstein, requires persim)",
        "accepts":     {},
    },

    "v7": {
        "script":      "paired_statistical_analysis.py",
        "description": "Paired Wilcoxon + sign test: curvature & persistence knotted vs unknotted",
        "depends_on":  ["v2"],
        "input_check": {
            "type": "file",
            "path": lambda: config.VIZ_FEATURES_DIR / "protein_features_no_outliers.csv",
            "hint": "Run step v2 first.",
        },
        "done_check": {
            "type": "dir_pdf",
            "path": lambda: config.RESULTS_DIR / "statistical_analysis",
        },
        "output_dir":  lambda: config.RESULTS_DIR / "statistical_analysis",
        "estimated":   "5-15 min  (317 pairs × 2 features)",
        "accepts":     {},
    },

    "v8": {
        "script":      "cycle_curvature_analysis.py",
        "description": "Cycle-level analysis: core/non-core curvature & persistence, knot-core overlap",
        "depends_on":  ["v2"],
        "input_check": {
            "type": "dir_json",
            "path": lambda: config.RC_DIR,
            "hint": "Run pipeline steps 5-6 first.",
        },
        "done_check": {
            "type": "dir_pdf",
            "path": lambda: config.RESULTS_DIR / "cycle_analysis",
        },
        "output_dir":  lambda: config.RESULTS_DIR / "cycle_analysis",
        "estimated":   "10-30 min  (one plot set per pair)",
        "accepts":     {},
    },

    "v9": {
        "script":      "size_control_analysis.py",
        "description": "Size-control: tests whether effects survive regressing out protein length",
        "depends_on":  ["v7"],
        "input_check": {
            "type": "file",
            "path": lambda: config.VIZ_FEATURES_DIR / "protein_features_no_outliers.csv",
            "hint": "Run step v2 first.",
        },
        "done_check": {
            "type": "dir_pdf",
            "path": lambda: config.RESULTS_DIR / "size_control",
        },
        "output_dir":  lambda: config.RESULTS_DIR / "size_control",
        "estimated":   "2-5 min",
        "accepts":     {},
    },

    "v10": {
        "script":      "random_control.py",
        "description": "Random-region negative control for knot-core localisation (permutation test)",
        "depends_on":  ["v8"],
        "input_check": {
            "type": "dir_json",
            "path": lambda: config.RC_DIR,
            "hint": "Run pipeline steps 5-6 and v8 first.",
        },
        "done_check": {
            "type": "dir_pdf",
            "path": lambda: config.RESULTS_DIR / "random_control",
        },
        "output_dir":  lambda: config.RESULTS_DIR / "random_control",
        "estimated":   "20-60 min  (499 permutations × 317 proteins)",
        "accepts":     {},
    },

    "v11": {
        "script":      "paper_table.py",
        "description": "Generate main results table: CSV + LaTeX + dataset overview figure",
        "depends_on":  ["v7", "v9"],
        "input_check": {
            "type": "file",
            "path": lambda: config.VIZ_FEATURES_DIR / "protein_features_no_outliers.csv",
            "hint": "Run step v2 first.",
        },
        "done_check": {
            "type": "file",
            "path": lambda: config.RESULTS_DIR / "paper_tables" / "main_results_table.csv",
        },
        "output_dir":  lambda: config.RESULTS_DIR / "paper_tables",
        "estimated":   "< 1 min",
        "accepts":     {},
    },

    "v12": {
        "script":      "composite_main_figure.py",
        "description": "Composite main figure: 4-panel layout per class for manuscript",
        "depends_on":  ["v7", "v8"],
        "input_check": {
            "type": "file",
            "path": lambda: config.VIZ_FEATURES_DIR / "protein_features_no_outliers.csv",
            "hint": "Run step v2 first.",
        },
        "done_check": {
            "type": "dir_pdf",
            "path": lambda: config.RESULTS_DIR / "main_figure",
        },
        "output_dir":  lambda: config.RESULTS_DIR / "main_figure",
        "estimated":   "2-5 min",
        "accepts":     {},
    },
}

RUN_ORDER = ["v1", "v2", "v2b", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12"]


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
    """
    Translate the user-facing CLI args into the extra arguments that will
    be forwarded to the child script, based on what each step accepts.
    """
    accepts = STEPS[sid].get("accepts", {})
    extra = []

    # --plots  (v1 and v4 accept this, but with different valid values)
    if "plots" in accepts and args.plots:
        valid   = accepts["plots"]
        chosen  = [p for p in args.plots if p in valid]
        invalid = [p for p in args.plots if p not in valid]
        if invalid:
            print("  ⚠  Ignoring --plots values not valid for {}: {}".format(
                sid, ", ".join(invalid)))
            print("     Valid choices for {}: {}".format(sid, ", ".join(sorted(valid))))
        if chosen:
            extra += ["--plots"] + chosen

    # --classes  (v1 accepts --classes; v3 uses --class legacy single value)
    if "classes" in accepts and args.classes:
        if sid == "v3":
            # v3 / statistical_analysis.py still uses the legacy --class flag
            # (single value); pass the first class or all if the script supports it
            extra += ["--class", args.classes[0]]
            if len(args.classes) > 1:
                print("  ⚠  Step v3 only accepts one class at a time via --class. "
                      "Using first: {}".format(args.classes[0]))
        else:
            extra += ["--classes"] + args.classes

    # --group  (v1 only)
    if "group" in accepts and args.group:
        extra += ["--group", args.group]

    # --group-split  (v4 only)
    if "group_split" in accepts and args.group_split:
        extra += ["--group-split"]

    # --include-homologs  (v4 only)
    if "include_homologs" in accepts and args.include_homologs:
        extra += ["--include-homologs"]

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
    cmd = [sys.executable,
           str(Path(__file__).parent / spec["script"])]
    # Forward dataset flag if not default
    if config.dataset != "default":
        cmd += ["--dataset", config.dataset]
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
    print("  {:<6}  {:<34}  Status".format("Step", "Script"))
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
        print("  {:<6}  {:<34}  {}".format(sid, STEPS[sid]["script"], status))
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
       |         results/persistent_homology/<class>/
       |           persistent_diagram/, barcode_plots/, most_persistent_cycle/
       |           Flags: --plots  --classes  --group k|s
       |
  pipeline steps 4+5+6
       |
       +-->  v2  extract_features.py
                 results/features/protein_features.csv
                      |
                      +-->  v2b extract_topological_features.py
                      |        Writhe · Pers. Entropy · Persistent Image PCA
                      |        results/features/protein_features_extended.csv
                      |        Flags: --classes
                      |
                      +-->  v3  statistical_analysis.py
                      |        results/comparison/<OriginalClass>/
                      |        Flags: --classes (single class)
                      |
                      +-->  v4  compare_classes.py
                      |        results/comparison/cross_class/
                      |        Flags: --plots  --classes  --group-split
                      |
                      +-->  v5  advanced_statistics.py        (fast: 2-10 min)
                      |        Flags: --classes
                      |
                      +-->  v6  wasserstein_analysis.py        (slow: O(n²))
                               results/comparison/advanced/wasserstein/
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
                flags.append("--plots ({})".format(", ".join(sorted(accepts["plots"]))))
            if "classes" in accepts:
                flags.append("--classes CLASS [CLASS ...]")
            if "group" in accepts:
                flags.append("--group k|s")
            if "group_split" in accepts:
                flags.append("--group-split")
            print("  Accepts      : {}".format("  ".join(flags)))
        if done:
            c = _count_outputs(sid)
            print("  Output dir   : {}".format(spec["output_dir"]()))
            print("  Files found  : {} PDFs, {} CSVs".format(c["pdfs"], c["csvs"]))
        else:
            print("  Expected out : {}".format(spec["output_dir"]()))
        print("  Est. runtime : {}".format(spec["estimated"]))

    # Feature CSV breakdown
    csv = config.VIZ_FEATURES_DIR / "protein_features.csv"
    print()
    print("-" * 65)
    print("Shared artefact: protein_features.csv")
    if csv.exists():
        try:
            import pandas as pd
            df  = pd.read_csv(csv)
            all_cls   = sorted(df["Protein_Class"].unique())
            originals = [c for c in all_cls
                         if not any(c.startswith(o + "_")
                                    for o in all_cls if o != c)]
            print("  Found: {}".format(csv))
            print("  {} proteins, {} classes ({} original, {} homolog groups)".format(
                len(df), len(all_cls), len(originals), len(all_cls) - len(originals)))
            k_cls = [c for c in originals if c.lower().startswith("k")]
            s_cls = [c for c in originals if c.lower().startswith("s")]
            if k_cls:
                print("  k-group ({} classes): {}".format(len(k_cls), ", ".join(k_cls)))
            if s_cls:
                print("  s-group ({} classes): {}".format(len(s_cls), ", ".join(s_cls)))
            for cls in all_cls:
                n    = (df["Protein_Class"] == cls).sum()
                role = "original" if cls in originals else "homolog "
                print("     [{}]  {:<50}  n={}".format(role, cls, n))
        except Exception as e:
            print("  Exists but could not parse: {}".format(e))
    else:
        print("  Not yet created -- run step v2 first.")

    # Results folder tree
    out_root = config.VIZ_COMPARE_DIR
    if out_root.exists():
        subdirs = sorted(d for d in out_root.iterdir() if d.is_dir())
        if subdirs:
            print()
            print("-" * 65)
            print("Output tree: {}".format(out_root))
            for d in subdirs:
                np_ = len(list(d.rglob("*.pdf")))
                nc  = len(list(d.rglob("*.csv")))
                print("  {}/   ({} PDFs, {} CSVs)".format(d.name, np_, nc))
                children = sorted(s for s in d.iterdir() if s.is_dir())
                for sub in children[:8]:
                    nsp = len(list(sub.rglob("*.pdf")))
                    print("    {}/   ({} PDFs)".format(sub.name, nsp))
                if len(children) > 8:
                    print("    ... and {} more".format(len(children) - 8))

    print()
    print("-" * 65)
    print("Common commands:")
    print("  python run_visualizations.py                              # All steps")
    print()
    print("  python run_visualizations.py --step v0 --classes MyClass  # adapt new PH format (run once per new dataset)")
    print("  python run_visualizations.py --step v1                   # PH plots only")
    print("  python run_visualizations.py --step v1 --plots barcodes  # Barcodes only")
    print("  python run_visualizations.py --step v1 --group k         # k-classes only")
    print("  python run_visualizations.py --step v1 --classes AOTCases OTCases")
    print("  python run_visualizations.py --step v2                   # Feature extraction")
    print("  python run_visualizations.py --step v2b                  # Writhe + PI features (run after v2)")
    print("  python run_visualizations.py --step v3 --classes AOTCases")
    print("  python run_visualizations.py --step v4                   # Cross-class (all)")
    print("  python run_visualizations.py --step v4 --plots kde violin_boxplot")
    print("  python run_visualizations.py --step v4 --group-split     # k vs s groups")
    print("  python run_visualizations.py --step v4 --plots kde --group-split")
    print("  python run_visualizations.py --step v4 --include-homologs")
    print("  python run_visualizations.py --step v4 --include-homologs --plots kde overview")
    print("  python run_visualizations.py --step v5                   # KS + entropy")
    print("  python run_visualizations.py --step v6                   # Wasserstein (slow)")
    print("  python run_visualizations.py --force                     # Re-run all")
    print("  python run_visualizations.py --status                    # This screen")
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
        help=(
            "Plot type(s) to generate (forwarded to v1 or v4). "
            "v1 choices: persistent_diagram barcodes most_persistent_cycle. "
            "v4 choices: kde violin_boxplot scatter volcano overview."
        ),
    )
    parser.add_argument(
        "--classes", nargs="+",
        metavar="CLASS",
        default=None,
        help=(
            "Class name(s) to process (forwarded to v1, v3, v4). "
            "v3 only uses the first value."
        ),
    )
    parser.add_argument(
        "--group", choices=["k", "s"],
        metavar="LETTER",
        default=None,
        help="(v1 only) Process classes starting with 'k' or 's'.",
    )
    parser.add_argument(
        "--group-split", dest="group_split", action="store_true",
        help=(
            "(v4 only) Also produce separate overview plots for "
            "k-classes and s-classes."
        ),
    )
    parser.add_argument(
        "--include-homologs", dest="include_homologs", action="store_true",
        help=(
            "(v4 only) Include homolog groups (e.g. AOTCases_KnotProt_Homologs) "
            "in the cross-class comparison alongside original classes."
        ),
    )

    args = parser.parse_args()

    if args.status:
        print_status()
        return

    # Warn if step-specific flags are given without the right --step
    if args.group and args.step and args.step != "v1":
        print("⚠  --group is only forwarded to v1. "
              "It will be ignored for step {}.".format(args.step))
    if args.group_split and args.step and args.step != "v4":
        print("⚠  --group-split is only forwarded to v4. "
              "It will be ignored for step {}.".format(args.step))
    if args.include_homologs and args.step and args.step != "v4":
        print("⚠  --include-homologs is only forwarded to v4. "
              "It will be ignored for step {}.".format(args.step))

    if args.step:
        extra = _build_extra_args(args.step, args)
        ok = run_one(args.step, args.force, extra)
    else:
        # Full pipeline: per-step extra args are built inside run_all_with_args
        if any([args.plots, args.classes, args.group, args.group_split, args.include_homologs]):
            print("ℹ  --plots / --classes / --group / --group-split / --include-homologs "
                  "flags are only applied when running a specific --step.\n"
                  "   Running full pipeline without filtering.")
        ok = run_all(args.force)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
