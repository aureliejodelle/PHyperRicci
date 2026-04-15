"""
run_pipeline.py — Modular Protein Knot Analysis Pipeline
----------------------------------------------------------------------------------
Steps
-----
  1a   Fetch protein IDs from KnotProt / RCSB         (Python, ~2-3 min)
  1b   Fetch similar chains via Selenium               (Python, ~20-30 min)
  2    Extract knot core ranges and sequences          (Python, ~5-10 min)
  3    Download structures, extract Cα coordinates     (Python, ~30-60 min)
  3b   Classify PDB homologs via Topoly (optional)     (Python, variable)
  4    Compute persistent homology                     (Julia, variable)
  5    Build hypergraphs from PH representatives       (Python, ~2-5 min)
  6    Compute Forman-Ricci curvature                  (Python, ~5-10 min)

Usage
-----
  python run_pipeline.py                          # Full pipeline (1a -- 6)
  python run_pipeline.py --step 4                # One step (auto-resolves deps)
  python run_pipeline.py --steps 5 6             # Multiple specific steps
  python run_pipeline.py --step 3b               # Classify PDB homologs
  python run_pipeline.py --step 1a --classes "K+3(1)" "K4(1)"
  python run_pipeline.py --status                # Show pipeline status
  python run_pipeline.py --force                 # Re-run everything
  python run_pipeline.py --no-parallel           # Disable Phase 2 parallelism
  python run_pipeline.py --step 3b --dry-run     # Preview step 3b
"""

from __future__ import annotations

import sys
import subprocess
import shutil
import argparse
import os
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(str(Path(__file__).parent))
from config import config


# -------------------------------------------------------------
# HELPER — readable by child scripts via env var
# -------------------------------------------------------------------

def get_selected_classes() -> Optional[List[str]]:
    """
    Read the PIPELINE_CLASSES env var set by this script.
    Returns a list of class names, or None (= all classes).

    Child scripts import this so all class-filtering goes through
    a single mechanism — the env var is the authoritative filter.

        from run_pipeline import get_selected_classes
        classes = get_selected_classes() or list(config.KNOT_TYPES.keys())
    """
    raw = os.environ.get("PIPELINE_CLASSES", "").strip()
    return [c.strip() for c in raw.split(",") if c.strip()] if raw else None


# -------------------------------------------------------------------
# JULIA DETECTION
# ------------------------------------------------------------------

def find_julia() -> Optional[str]:
    julia = shutil.which("julia")
    if julia:
        return julia
    candidates = [
        Path.home() / ".juliaup" / "bin" / "julia",
        Path.home() / "AppData" / "Local" / "Programs" / "Julia" / "bin" / "julia.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


JULIA_BIN = find_julia()


# ------------------------------------------------------------------
# STEP REGISTRY
# --------------------------------------------------------------------

STEPS: dict = {
    "1a": {
        "name":             "fetch_protein_ids.py",
        "description":      "Fetch protein IDs from KnotProt and RCSB",
        "lang":             "python",
        "optional":         False,
        "dependencies":     [],
        "required_outputs": [config.PROTEIN_LISTS_DIR / "all_protein_ids.csv"],
        "output_glob":      "*.csv",
        "estimated_time":   "2-3 min",
    },
    "1b": {
        "name":             "fetch_protein_ids_similar_chain.py",
        "description":      "Fetch similar chains (Selenium / Chrome)",
        "lang":             "python",
        "optional":         False,
        "dependencies":     ["1a"],
        "required_outputs": [config.SIMILAR_CHAINS_DIR / "similar_chains_simple.csv"],
        "output_glob":      "*.csv",
        "estimated_time":   "20-30 min",
    },
    "2": {
        "name":             "extract_knot_data.py",
        "description":      "Extract knot core ranges and sequences",
        "lang":             "python",
        "optional":         False,
        "dependencies":     ["1a"],
        "required_outputs": [config.KNOT_DATA_DIR / "knot_data_full.csv"],
        "output_glob":      "*.csv",
        "estimated_time":   "5-10 min",
    },
    "3": {
        "name":             "download_structures.py",
        "description":      "Download structures and extract Cα coordinates",
        "lang":             "python",
        "optional":         False,
        "dependencies":     ["1a", "1b"],
        "required_outputs": [config.COORDINATES_DIR],
        "output_glob":      "**/*.csv",
        "estimated_time":   "30-60 min",
    },
    "3b": {
        "name":             "classify_pdb_homologs.py",
        "description":      "Classify PDB homologs as knotted / unknotted (Topoly)",
        "lang":             "python",
        "optional":         True,
        "dependencies":     ["3", "4", "5", "6"],   # needs all processed data
        "required_outputs": [config.BASE_DIR / "topology_classification.json"],
        "output_glob":      "*.json",
        "estimated_time":   "variable (Topoly per protein)",
    },
    "4": {
        "name":             "persistent_homology.jl",
        "description":      "Compute persistent homology (Julia, multi-threaded)",
        "lang":             "julia",
        "optional":         False,
        "dependencies":     ["3"],
        "required_outputs": [config.PH_DIR],
        "output_glob":      "**/*.json",
        "estimated_time":   "variable (fully parallel)",
    },
    "5": {
        "name":             "compute_hypergraphs.py",
        "description":      "Build hypergraphs from PH representatives",
        "lang":             "python",
        "optional":         False,
        "dependencies":     ["4"],
        "required_outputs": [config.HG_DIR],
        "output_glob":      "**/*.json",
        "estimated_time":   "2-5 min",
    },
    "6": {
        "name":             "compute_curvature.py",
        "description":      "Compute Forman-Ricci curvature (raw + variants)",
        "lang":             "python",
        "optional":         False,
        "dependencies":     ["5"],
        "required_outputs": [
            config.RC_DIR,
            config.RC_NORM_DIR,
            config.RC_RATIO_DIR,
            config.RC_RESID_DIR,
        ],
        "output_glob":      "**/*.json",
        "estimated_time":   "5-10 min",
    },
}

# Ordered list for full-pipeline execution (3b excluded — run on demand only)
PIPELINE_ORDER = ["1a", "1b", "2", "3", "4", "5", "6"]

# Steps where --classes filtering applies
CLASSES_AWARE = {"1a", "1b", "2", "3", "4", "5", "6"}


# ------------------------------------------------------------------
# PIPELINE RUNNER
# -------------------------------------------------------------------------

class PipelineRunner:

    def __init__(self, dry_run: bool = False):
        self.scripts_dir = Path(__file__).parent
        self.logs_dir    = config.LOGS_DIR
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run     = dry_run

    # ----------------------------------------------------------
    # STATUS HELPERS
    # ----------------------------------------------------------

    def _count_outputs(self, step: str) -> int:
        """Count files matching each step's output_glob across all required_outputs."""
        total = 0
        glob  = STEPS[step].get("output_glob", "*")
        for output in STEPS[step]["required_outputs"]:
            if output.is_dir():
                total += sum(1 for _ in output.glob(glob))
            elif output.is_file():
                total += 1
        return total

    def check_step_done(self, step: str) -> bool:
        """Return True if all required outputs exist and are non-empty."""
        if step not in STEPS:
            return False
        glob = STEPS[step].get("output_glob", "*")
        for output in STEPS[step]["required_outputs"]:
            if output.is_dir():
                if not any(output.glob(glob)):
                    return False
            elif not output.exists():
                return False
        return True

    def check_dependencies(self, step: str) -> Tuple[bool, List[str]]:
        missing = [
            dep for dep in STEPS.get(step, {}).get("dependencies", [])
            if not self.check_step_done(dep)
        ]
        return len(missing) == 0, missing

    def ensure_dependencies(self, step: str, force: bool = False) -> bool:
        """Auto-run any missing dependencies before running `step`."""
        _, missing = self.check_dependencies(step)
        if not missing:
            return True
        print(f"\n  Step {step} requires: {', '.join(missing)}")
        for dep in missing:
            # Don't auto-run optional steps
            if STEPS.get(dep, {}).get("optional"):
                print(f"  Dependency {dep} is optional — skipping auto-run.")
                continue
            print(f"\n  Auto-running Step {dep} first...")
            if not self._execute_step(dep, force):
                print(f"\n  [FAIL] Could not run Step {dep}. Cannot proceed.")
                return False
            print(f"  [OK] Step {dep} completed.")
        return True

    # ----------------------------------------------------------
    # EXECUTION
    # ----------------------------------------------------------

    def run_step(
        self,
        step: str,
        force: bool = False,
        classes: Optional[List[str]] = None,
        extra_args: Optional[List[str]] = None,
    ) -> bool:
        """Run a single step, auto-resolving its dependencies first."""
        if step not in STEPS:
            print(f"\n  [ERROR] Unknown step: '{step}'")
            print(f"  Available: {', '.join(STEPS)}")
            return False

        if classes and step not in CLASSES_AWARE:
            print(f"  [WARN] --classes has no effect for step {step}")
            classes = None

        info = STEPS[step]
        print("\n" + "=" * 68)
        print(f"  STEP {step}: {info['description']}")
        if info.get("optional"):
            print("  (optional step)")
        if classes:
            print(f"  Classes : {', '.join(classes)}")
        if self.dry_run and step == "3b":
            print("  DRY-RUN: no files will be written.")
        print("=" * 68)

        # Skip check: bypass when --classes is set (partial re-run intended)
        if not force and self.check_step_done(step) and not classes:
            n = self._count_outputs(step)
            print(f"\n  Already completed ({n} output files). Use --force to re-run.")
            return True

        if not self.ensure_dependencies(step, force):
            return False

        return self._execute_step(step, force, classes=classes, extra_args=extra_args)

    def run_steps(
        self,
        steps: List[str],
        force: bool = False,
        classes: Optional[List[str]] = None,
    ) -> bool:
        """Run a list of steps sequentially."""
        for step in steps:
            if not self.run_step(step, force=force, classes=classes):
                return False
        return True

    def _build_command(self, step: str) -> Optional[List[str]]:
        info        = STEPS[step]
        lang        = info["lang"]
        script_path = self.scripts_dir / info["name"]

        if not script_path.exists():
            print(f"\n  [ERROR] Script not found: {script_path}")
            return None

        if lang == "python":
            return [sys.executable, str(script_path)]

        if lang == "julia":
            if not JULIA_BIN:
                print("\n  [ERROR] Julia not found. Install from https://julialang.org/downloads/")
                print(f"  Or run manually:  julia --threads auto {script_path}")
                return None
            n_threads = str(os.cpu_count() or 4)
            print(f"  Julia   : {JULIA_BIN}")
            print(f"  Threads : {n_threads}")
            return [JULIA_BIN, f"--threads={n_threads}", str(script_path)]

        print(f"\n  [ERROR] Unknown language '{lang}' for step {step}")
        return None

    def _execute_step(
        self,
        step: str,
        force: bool,
        classes: Optional[List[str]] = None,
        extra_args: Optional[List[str]] = None,
    ) -> bool:
        info = STEPS[step]
        cmd  = self._build_command(step)
        if cmd is None:
            return False

        # Append extra CLI arguments (e.g. --dry-run for step 3b,
        # or --classes for classify_pdb_homologs which parses its own args).
        if extra_args:
            cmd.extend(extra_args)

        print(f"\n  Starting: {info['description']}")
        print(f"  Time est: {info['estimated_time']}\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file  = self.logs_dir / f"step{step}_{timestamp}.log"

        # Environment: pass paths and class filter to child scripts
        env = os.environ.copy()
        env["PH_INPUT_ROOT"]  = str(config.COORDINATES_DIR.resolve())
        env["PH_OUTPUT_ROOT"] = str(config.PH_DIR.resolve())

        if classes:
            env["PIPELINE_CLASSES"] = ",".join(classes)
        else:
            env.pop("PIPELINE_CLASSES", None)   # never leak from a parent shell

        try:
            with open(log_file, "w") as log:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                    cwd=str(self.scripts_dir),
                )
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                proc.wait()

            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)

            print(f"\n  [OK] Step {step} completed successfully.")
            print(f"  Log: {log_file}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"\n  [FAIL] Step {step} exited with code {e.returncode}.")
            print(f"  Log: {log_file}")
            return False

        except Exception as e:
            print(f"\n  [ERROR] Unexpected error in step {step}: {e}")
            return False

    # ----------------------------------------------------------
    # FULL PIPELINE
    # ----------------------------------------------------------

    def run_all(self, force: bool = False, parallel: bool = True) -> bool:
        """
        Run the full pipeline (steps 1a → 6) in the correct order.
        Steps 1b and 2 are run in parallel (both depend only on 1a).
        Step 3b is never included in the full run — invoke explicitly.
        """
        start = datetime.now()
        print("\n" + "=" * 68)
        print("  PROTEIN KNOT ANALYSIS PIPELINE")
        print("=" * 68)
        print(f"  Start: {start.strftime('%Y-%m-%d %H:%M:%S')}")
        if not parallel:
            print("  Parallelism: disabled (--no-parallel)")

        # --- Phase 1: Protein IDs -------------------------------------------------
        self._phase_header("1", "Fetch Protein IDs")
        if self.check_step_done("1a") and not force:
            print("  Step 1a already done, skipping.")
        elif not self._execute_step("1a", force):
            return False

        # --- Phase 2: Similar Chains + Knot Data (parallel) ------------------------
        self._phase_header("2", "Similar Chains & Knot Core Data")
        steps_to_run = [s for s in ("1b", "2") if force or not self.check_step_done(s)]
        for s in ("1b", "2"):
            if s not in steps_to_run:
                n = self._count_outputs(s)
                print(f"  Step {s} already done ({n} files), skipping.")

        if steps_to_run:
            if parallel and len(steps_to_run) > 1:
                print(f"  Running {steps_to_run} in parallel ...\n")
                ok = self._run_parallel(steps_to_run, force)
            else:
                ok = all(self._execute_step(s, force) for s in steps_to_run)
            if not ok:
                return False

        # --- Phase 3: Download Structures --------------------------------------------
        self._phase_header("3", "Download Structures")
        if self.check_step_done("3") and not force:
            n = self._count_outputs("3")
            print(f"  Step 3 already done ({n} files), skipping.")
        elif not self._execute_step("3", force):
            return False

        # --- Phase 4: Persistent Homology (Julia) ------------------------------------
        self._phase_header("4", "Persistent Homology (Julia)")
        if self.check_step_done("4") and not force:
            n = self._count_outputs("4")
            print(f"  Step 4 already done ({n} files), skipping.")
        elif not self._execute_step("4", force):
            return False

        # --- Phase 5: Hypergraphs ----------------------------------------
        self._phase_header("5", "Hypergraph Computation")
        if self.check_step_done("5") and not force:
            n = self._count_outputs("5")
            print(f"  Step 5 already done ({n} files), skipping.")
        elif not self._execute_step("5", force):
            return False

        # --- Phase 6: Forman-Ricci Curvature ----------------------------------
        self._phase_header("6", "Forman-Ricci Curvature")
        if self.check_step_done("6") and not force:
            n = self._count_outputs("6")
            print(f"  Step 6 already done ({n} files), skipping.")
        elif not self._execute_step("6", force):
            return False

        elapsed = (datetime.now() - start).seconds
        print("\n" + "=" * 68)
        print("  PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 68)
        print(f"  Total time : {elapsed // 60}m {elapsed % 60}s")
        print(f"  End        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n  To classify PDB homologs (optional):")
        print("    python run_pipeline.py --step 3b")
        print("    python run_pipeline.py --step 3b --dry-run  # preview first")
        return True

    def _phase_header(self, num: str, label: str):
        print(f"\n{'─' * 68}")
        print(f"  Phase {num}: {label}")
        print(f"{'─' * 68}")

    def _run_parallel(self, steps: List[str], force: bool) -> bool:
        """Run multiple independent steps in parallel using threads."""
        results: dict[str, bool] = {}
        lock = threading.Lock()

        def run_one(step: str):
            ok = self._execute_step(step, force)
            with lock:
                results[step] = ok

        threads = [threading.Thread(target=run_one, args=(s,)) for s in steps]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        failed = [s for s, ok in results.items() if not ok]
        if failed:
            print(f"\n  [FAIL] Steps failed: {', '.join(failed)}")
        return not failed

    # ----------------------------------------------------------
    # STATUS DISPLAY
    # ----------------------------------------------------------

    def print_status(self):
        print("\n" + "=" * 68)
        print("  PIPELINE STATUS")
        print("=" * 68)


        # Julia check
        if JULIA_BIN:
            print(f"  Julia : {JULIA_BIN}")
        else:
            print("  [WARN] Julia not found. Step 4 will fail.")
            print("         Install from https://julialang.org/downloads/")
        print()

        # Per-step status
        lang_icon = {"python": "Py", "julia": "Jl"}
        for step, info in STEPS.items():
            done          = self.check_step_done(step)
            deps_ok, miss = self.check_dependencies(step)
            tag           = lang_icon.get(info["lang"], "??")
            optional      = " [optional]" if info.get("optional") else ""
            status_str    = "DONE" if done else "pending"

            print(f"  Step {step:3s} [{tag}]  {info['description']}{optional}")
            print(f"           status : {status_str}")

            if done:
                n = self._count_outputs(step)
                print(f"           outputs: {n} file(s)")
            if not deps_ok:
                print(f"           missing deps: {', '.join(miss)}")
            print()

        # Available classes
        print("  Available knot/slip classes:")
        for cls in config.KNOT_TYPES:
            pages = config.KNOT_TYPES[cls]["pages"]
            print(f"    {cls}  ({pages} page(s))")

        print(f"\n{'─' * 68}")
        print("  Common commands")
        print("─" * 68)
        cmds = [
            ("Full pipeline",                 "python run_pipeline.py"),
            ("Full pipeline (no parallel)",   "python run_pipeline.py --no-parallel"),
            ("Single step",                   "python run_pipeline.py --step 4"),
            ("Multiple steps",                "python run_pipeline.py --steps 5 6"),
            ("One class only",                "python run_pipeline.py --step 1a --classes \"K+3(1)\""),
            ("Multiple classes",              "python run_pipeline.py --step 3 --classes \"K+3(1)\" \"K4(1)\""),
            ("Classify PDB homologs",         "python run_pipeline.py --step 3b"),
            ("Classify PDB homologs dry-run", "python run_pipeline.py --step 3b --dry-run"),
            ("Force re-run",                  "python run_pipeline.py --force"),
            ("Status check",                  "python run_pipeline.py --status"),
        ]
        for label, cmd in cmds:
            print(f"  {label:<30}  {cmd}")
        print()


# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

def main():
    valid_classes = list(config.KNOT_TYPES.keys())
    valid_steps   = list(STEPS.keys())

    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Protein knot analysis pipeline (steps 1a → 6, plus optional step 3b).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mutually exclusive: --step / --steps / default (full pipeline)
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument(
        "--step",
        metavar="STEP",
        choices=valid_steps,
        help=f"Run a single step. Choices: {', '.join(valid_steps)}",
    )
    step_group.add_argument(
        "--steps",
        metavar="STEP",
        nargs="+",
        choices=valid_steps,
        help="Run a list of specific steps sequentially (e.g. --steps 5 6).",
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        metavar="CLASS",
        help=(
            "Restrict to specific knot/slip classes. "
            f"Available: {', '.join(valid_classes)}"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run step(s) even if outputs already exist.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show pipeline status and exit.",
    )
    parser.add_argument(
        "--no-parallel",
        dest="parallel",
        action="store_false",
        default=True,
        help="Disable parallel execution of steps 1b and 2 in the full pipeline.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="For step 3b only: preview actions without copying files.",
    )

    args   = parser.parse_args()
    runner = PipelineRunner(dry_run=args.dry_run)

    # --- Status ----------------------------------------------------------------
    if args.status:
        runner.print_status()
        sys.exit(0)

    # -- Validate --classes ----------------------------------------------------------
    classes = None
    if args.classes:
        unknown = [c for c in args.classes if c not in valid_classes]
        if unknown:
            print(f"\n[ERROR] Unknown class(es): {', '.join(unknown)}")
            print(f"  Available: {', '.join(valid_classes)}")
            sys.exit(1)
        classes = args.classes

    # -- --steps: multiple named steps ------------------------------------------------
    if args.steps:
        if classes:
            print(f"  Classes: {', '.join(classes)}")
        success = runner.run_steps(args.steps, force=args.force, classes=classes)
        sys.exit(0 if success else 1)

    # -- --step: single step ---------------------------------------------------------
    if args.step:
        if classes:
            print(f"  Classes: {', '.join(classes)}")

        # Step 3b: pass --dry-run and --classes directly as CLI args to the
        # classify_pdb_homologs.py script, which has its own argparse.
        extra: List[str] = []
        if args.step == "3b":
            if args.dry_run:
                extra.append("--dry-run")
            if classes:
                extra.extend(["--classes"] + [
                    config.sanitize_class_name(c) for c in classes
                ])

        success = runner.run_step(args.step, force=args.force,
                                  classes=classes, extra_args=extra or None)
        sys.exit(0 if success else 1)

    # --- Full pipeline --------------------------------------------------------------------------------
    if classes:
        print("[WARN] --classes only works with --step / --steps. Ignored for full run.")
    if args.dry_run:
        print("[WARN] --dry-run only applies to step 3b and is ignored here.")

    success = runner.run_all(force=args.force, parallel=args.parallel)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
