#!/usr/bin/env python3
"""
Script 3: Download Structures and Extract CA Coordinates
--------------------------------------------------------
Reads similar_chains_simple.csv and downloads ALL proteins:
- Original proteins (from the rows themselves)
- All homologous proteins grouped by source class × similarity type

Input:  config.get_similar_chains_path()
Output: config.COORDINATES_DIR/
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Set

import pandas as pd
import requests
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent))
from config import config


def get_selected_classes():
    """Read PIPELINE_CLASSES env var set by run_pipeline.py.
    Returns a list of class names, or None (= all classes)."""
    raw = os.environ.get("PIPELINE_CLASSES", "").strip()
    return [c.strip() for c in raw.split(",") if c.strip()] if raw else None

# CONFIG

NUM_WORKERS   = 10    # Parallel threads — safe for RCSB
REQUEST_DELAY = 0.05  # Seconds between requests per thread
TIMEOUT       = 30    # HTTP timeout



# HELPERS
def parse_bracket_list(value) -> List[str]:
    """Parse '[item1, item2]' → ['item1', 'item2']."""
    if not value or pd.isna(value):
        return []
    s = str(value).strip()
    if s in ("[]", ""):
        return []
    return [
        item.strip().strip("\"'")
        for item in s.strip("[]").split(",")
        if item.strip() and len(item.strip()) > 3
    ]


def simplify_chain(chain) -> str:
    """AAA → A. Returns '' if chain is empty/NaN."""
    if not chain or (isinstance(chain, float) and pd.isna(chain)):
        return ""
    s = str(chain).strip()
    if len(s) == 3 and len(set(s)) == 1:
        return s[0]
    return s


def make_filename(protein_id: str, chain: str) -> str:
    ch = simplify_chain(chain)
    return f"{protein_id.upper()}_{ch}.csv" if ch else f"{protein_id.upper()}.csv"



# CA EXTRACTION - PDB format

def extract_ca_from_pdb(pdb_text: str, chain: str) -> List[Tuple[float, float, float]]:
    atoms = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 55:
            continue
        if line[12:16].strip() != "CA":
            continue
        if chain and line[21].strip() != chain:
            continue
        try:
            atoms.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            continue
    return atoms



# CA EXTRACTION - mmCIF format

def extract_ca_from_cif(cif_text: str, chain: str) -> List[Tuple[float, float, float]]:
    """
    Parse _atom_site loop from mmCIF to extract CA coordinates.
    Uses auth_asym_id for chain matching (same as PDB chain letters).
    """
    atoms = []
    lines = cif_text.splitlines()
    in_atom_loop = False
    headers = []
    col = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "loop_":
            j = i + 1
            peek = []
            while j < len(lines) and lines[j].strip().startswith("_"):
                peek.append(lines[j].strip())
                j += 1
            if any("_atom_site." in h for h in peek):
                in_atom_loop = True
                headers = peek
                col = {h.split(".")[-1]: idx for idx, h in enumerate(headers)}
                i = j
                continue
            else:
                in_atom_loop = False

        if in_atom_loop:
            if line.startswith("_") or line in ("loop_", "") or line.startswith("#"):
                in_atom_loop = False
                i += 1
                continue

            parts = line.split()
            if len(parts) < len(headers):
                i += 1
                continue

            try:
                if parts[col.get("label_atom_id", 0)] != "CA":
                    i += 1
                    continue
                if chain:
                    chain_col = col.get("auth_asym_id", col.get("label_asym_id"))
                    if chain_col is not None and parts[chain_col] != chain:
                        i += 1
                        continue
                atoms.append((
                    float(parts[col["Cartn_x"]]),
                    float(parts[col["Cartn_y"]]),
                    float(parts[col["Cartn_z"]]),
                ))
            except (KeyError, ValueError, IndexError):
                pass

        i += 1

    return atoms



# DOWNLOADER

class ProteinDownloader:

    def __init__(self):
        self._local = threading.local()
        self._lock  = threading.Lock()
        self.stats  = {
            "total": 0, "existing": 0, "downloaded": 0,
            "failed": 0, "no_ca": 0, "total_atoms": 0,
            "by_category": {}
        }

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "s"):
            s = requests.Session()
            s.headers["User-Agent"] = "Mozilla/5.0 (PhD Research)"
            self._local.s = s
        return self._local.s

    def _bump(self, key: str, n: int = 1):
        with self._lock:
            self.stats[key] += n

    # ------------------------------------------------------------------
    def download_one(self, protein_id: str, chain: str, output_file: Path) -> Tuple[str, int]:
        """
        Download protein → try PDB → fallback to mmCIF → save CA coords as CSV.
        Returns (status, num_atoms)  where status ∈ {existing, downloaded, failed, no_ca}
        """
        if output_file.exists():
            try:
                n = len(pd.read_csv(output_file))
                self._bump("existing")
                self._bump("total_atoms", n)
                return "existing", n
            except Exception:
                output_file.unlink(missing_ok=True)

        pid     = protein_id.upper()
        chain_s = simplify_chain(chain)
        sess    = self._session()
        ca      = []
        got_response = False

        # ---1. Try PDB format -------------------------------------------------------------
        try:
            r = sess.get(f"https://files.rcsb.org/download/{pid.lower()}.pdb", timeout=TIMEOUT)
            if r.status_code == 200 and r.text.strip():
                got_response = True
                ca = extract_ca_from_pdb(r.text, chain_s)
                if not ca and chain and chain != chain_s:
                    ca = extract_ca_from_pdb(r.text, chain)
                if not ca:
                    ca = extract_ca_from_pdb(r.text, "")
        except Exception:
            pass

        # --- 2. Fallback: mmCIF (handles large / newer structures) ---------------
        if not ca:
            try:
                r = sess.get(f"https://files.rcsb.org/download/{pid.lower()}.cif", timeout=TIMEOUT)
                if r.status_code == 200 and r.text.strip():
                    got_response = True
                    ca = extract_ca_from_cif(r.text, chain_s)
                    if not ca and chain and chain != chain_s:
                        ca = extract_ca_from_cif(r.text, chain)
                    if not ca:
                        ca = extract_ca_from_cif(r.text, "")
            except Exception:
                pass

        if not ca:
            status = "no_ca" if got_response else "failed"
            self._bump(status)
            return status, 0

        output_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(ca, columns=["x", "y", "z"]).to_csv(output_file, index=False)
        self._bump("downloaded")
        self._bump("total_atoms", len(ca))
        return "downloaded", len(ca)

    # ------------------------------------------------------------------
    def download_category(self, protein_list: List[str], category_name: str, base_dir: Path):
        if not protein_list:
            return

        clean = config.sanitize_class_name(category_name)
        cat_dir = base_dir / clean
        cat_dir.mkdir(parents=True, exist_ok=True)

        cat = {k: 0 for k in ("total","existing","downloaded","failed","no_ca","total_atoms")}
        cat["total"] = len(protein_list)
        cat_lock = threading.Lock()

        def worker(pc: str):
            pid, ch = pc.split("_", 1) if "_" in pc else (pc, "")
            out = cat_dir / make_filename(pid, ch)
            status, n = self.download_one(pid, ch, out)
            with cat_lock:
                cat[status] += 1
                cat["total_atoms"] += n
            time.sleep(REQUEST_DELAY)

        desc = f"  {category_name[:38]:<38}"
        with tqdm(total=len(protein_list), desc=desc, unit="prot",
                  bar_format="{l_bar}{bar:25}{r_bar}") as pbar:
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
                futs = {ex.submit(worker, p): p for p in protein_list}
                for f in as_completed(futs):
                    try:
                        f.result()
                    except Exception as e:
                        tqdm.write(f"    ⚠ worker error: {e}")
                    pbar.update(1)
                    with cat_lock:
                        pbar.set_postfix(
                            new=cat["downloaded"], exist=cat["existing"],
                            fail=cat["failed"], no_ca=cat["no_ca"],
                            refresh=False,
                        )

        tqdm.write(
            f"   {category_name}: "
            f"new={cat['downloaded']} exist={cat['existing']} "
            f"failed={cat['failed']} no_ca={cat['no_ca']} atoms={cat['total_atoms']:,}"
        )

        with self._lock:
            self.stats["by_category"][category_name] = cat
            self.stats["total"] += len(protein_list)

    # ------------------------------------------------------------------
    def print_summary(self):
        print("\n" + "=" * 60)
        print(" DOWNLOAD SUMMARY")
        print("=" * 60)
        s = self.stats
        print(f"  Total processed : {s['total']}")
        print(f"  Already existing: {s['existing']}")
        print(f"  Newly downloaded: {s['downloaded']}")
        print(f"  Failed          : {s['failed']}")
        print(f"  No CA atoms     : {s['no_ca']}")
        print(f"  Total CA atoms  : {s['total_atoms']:,}")

        stats_file = config.PROCESSED_DATA_DIR / "download_stats.json"
        with open(stats_file, "w") as f:
            json.dump(s, f, indent=2)
        print(f"\n Stats saved to: {stats_file}")



# MAIN

def main():
    print("=" * 60)
    print("SCRIPT 3: DOWNLOAD PROTEIN STRUCTURES (PARALLEL)")
    print("=" * 60)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Workers: {NUM_WORKERS}")

    input_file = config.SIMILAR_CHAINS_DIR / "similar_chains_simple.csv"
    if not input_file.exists():
        print(f"\n {input_file} not found. Run script1b first.")
        return

    df = pd.read_csv(input_file)
    print(f"\n Loaded {len(df)} proteins from {input_file.name}")

    selected = get_selected_classes()
    if selected:
        df = df[df["class"].isin(selected)]
        print(f" Filtering to selected classes: {', '.join(selected)}")

    print(f"   Processing {len(df)} proteins across {df['class'].nunique()} class(es)\n")

    dl = ProteinDownloader()

    # --- PART 1: Original proteins grouped by class -----------------------------
    print("=" * 50)
    print("PART 1: ORIGINAL PROTEINS (by class)")
    print("=" * 50)
    for cls in sorted(df["class"].unique()):
        protein_list = df[df["class"] == cls]["protein_id_chain"].tolist()
        dl.download_category(protein_list, cls, config.COORDINATES_DIR)

    # --- PART 2: Homologs grouped by source class × similarity type ----
    print("\n" + "=" * 50)
    print("PART 2: HOMOLOGS BY CLASS * SIMILARITY TYPE")
    print("=" * 50)

    sim_cols = [
        ("similar_chains_knotprot",  "KnotProt"),
        ("similar_chains_unknotted", "Unknotted"),
        ("similar_chains_pdb",       "PDB"),
    ]

    for cls in sorted(df["class"].unique()):
        class_df = df[df["class"] == cls]
        for col, sim_name in sim_cols:
            homologs: Set[str] = set()
            for val in class_df[col]:
                homologs.update(parse_bracket_list(val))
            if homologs:
                dl.download_category(
                    sorted(homologs),
                    f"{cls}_{sim_name}_Homologs",
                    config.COORDINATES_DIR,
                )

    dl.print_summary()
    print(f"\n Done.  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
