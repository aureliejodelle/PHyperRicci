"""
Script 1a: Fetch Protein IDs Only
---------------------------------
Fetches all protein IDs from KnotProt and PDB.
Output: CSV with columns: protein_id | chain | class
"""

import os
import sys
import time
import csv
import re
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from config import config


def get_selected_classes():
    """Read PIPELINE_CLASSES env var set by run_pipeline.py.
    Returns a list of class names, or None (= all classes)."""
    raw = os.environ.get("PIPELINE_CLASSES", "").strip()
    return [c.strip() for c in raw.split(",") if c.strip()] if raw else None


class ProteinIDFetcher:
    """Fetches protein IDs from various sources."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
    
    def fetch_knotprot_ids(self, knot_type: str) -> List[Tuple[str, str]]:
        """
        Fetch protein IDs and chains from KnotProt for a specific knot type.
        Parses the raw data block (pdbid|chain|...) which is reliable and
        unambiguous across all pages - avoids the HTML link duplication bug.
        """
        print(f"\n  Fetching {knot_type} from KnotProt...")

        knot_config = config.KNOT_TYPES.get(knot_type)
        if not knot_config:
            print(f"Unknown knot type: {knot_type}")
            return []

        proteins = []
        seen = set()
        pages = knot_config["pages"]

        for page in range(1, pages + 1):
            url = config.get_knotprot_url(knot_type, page)

            try:
                response = self.session.get(url, timeout=config.TIMEOUT)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # Strategy 1: parse the raw data <pre> block
                # The page contains a block like:
                #   6ymb|A||S|+3.1;
                #   6ufd|A||S|+3.1;
                # This is the most reliable source - one entry per protein,
                # no duplicates from repeated table links.
                page_proteins = []
                pre_tags = soup.find_all("pre")
                for pre in pre_tags:
                    text = pre.get_text()
                    for line in text.splitlines():
                        line = line.strip().rstrip(";")
                        parts = line.split("|")
                        if len(parts) >= 2:
                            pdb_id = parts[0].strip().upper()
                            chain  = parts[1].strip()
                            if re.match(r"^[A-Z0-9]{1,4}$", pdb_id) and len(pdb_id) == 4:
                                key = (pdb_id, chain)
                                if key not in seen:
                                    seen.add(key)
                                    page_proteins.append(key)

                # Strategy 2: fallback to /view/ links 
                # Only used if the raw block was empty.
                if not page_proteins:
                    view_links = soup.find_all("a", href=re.compile(r"^/view/"))
                    for link in view_links:
                        href = link.get("href", "")
                        parts = href.split("/")
                        if len(parts) >= 4:
                            pdb_id = parts[2].upper()
                            chain  = parts[3]
                            if re.match(r"^[A-Z0-9]{4}$", pdb_id):
                                key = (pdb_id, chain)
                                if key not in seen:
                                    seen.add(key)
                                    page_proteins.append(key)

                proteins.extend(page_proteins)
                print(f"    Page {page}: Found {len(page_proteins)} new proteins")

                time.sleep(config.DOWNLOAD_DELAY)

            except Exception as e:
                print(f"Error fetching page {page}: {e}")

        print(f"Total unique proteins: {len(proteins)}")
        return proteins
    
    def fetch_otcase_ids(self) -> List[Tuple[str, str]]:
        """
        Fetch OTCase protein IDs from RCSB PDB (EC 2.1.3.3).
        """
        print("\n  Fetching OTCase proteins from RCSB PDB...")
        
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_polymer_entity.rcsb_ec_lineage.id",
                            "operator": "exact_match",
                            "value": "2.1.3.3"
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "exptl.method",
                            "operator": "exact_match",
                            "value": "X-RAY DIFFRACTION"
                        }
                    }
                ]
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": 10000}
            }
        }
        
        try:
            response = self.session.post(config.RCSB_SEARCH_URL, json=query)
            response.raise_for_status()
            result = response.json()
            
            proteins = []
            if "result_set" in result:
                for item in result["result_set"]:
                    proteins.append((item["identifier"].upper(), ""))
            
            print(f"Found {len(proteins)} OTCase proteins")
            return proteins
            
        except Exception as e:
            print(f"Error: {e}")
            return []


def main():
    """Main execution."""
    print("=" * 60)
    print("SCRIPT 1a: FETCH PROTEIN IDs ONLY")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize fetcher
    fetcher = ProteinIDFetcher()
    all_proteins = []
    
    # Fetch KnotProt proteins
    selected = get_selected_classes()
    knot_classes = selected if selected else list(config.KNOT_TYPES.keys())

    if selected:
        print(f"\n Fetching selected classes: {', '.join(selected)}")
    else:
        print("\n Fetching all KnotProt classes...")

    for knot_type in knot_classes:
        proteins = fetcher.fetch_knotprot_ids(knot_type)
        for pdb_id, chain in proteins:
            all_proteins.append({
                "protein_id": pdb_id,
                "chain": chain,
                "class": knot_type
            })
    
    # Fetch OTCase proteins (skipped if --classes set and OTCase not included)
    if not selected or "OTCase" in selected:
        otcase_proteins = fetcher.fetch_otcase_ids()
        for pdb_id, chain in otcase_proteins:
            all_proteins.append({
                "protein_id": pdb_id,
                "chain": chain,
                "class": "OTCase"
            })
    else:
        print("\n  Skipping OTCase (not in selected classes)")
    
    print(f"\n Total proteins collected: {len(all_proteins)}")
    
    # Save results - append to existing CSV when running partial classes
    output_file = config.PROTEIN_LISTS_DIR / "all_protein_ids.csv"
    df = pd.DataFrame(all_proteins)

    if selected and output_file.exists():
        existing = pd.read_csv(output_file)
        # Drop rows for classes we are re-fetching, then append fresh data
        existing = existing[~existing["class"].isin(selected + (["OTCase"] if "OTCase" in selected else []))]
        df = pd.concat([existing, df], ignore_index=True)
        print(f"\n Updated {output_file} (merged with existing data)")
    else:
        print(f"\n Saved to {output_file}")

    df.to_csv(output_file, index=False)
    
    # Print summary
    print("\n Summary by class:")
    summary = df.groupby("class").size()
    for class_name, count in summary.items():
        print(f"  {class_name}: {count} proteins")
    
    print(f"\n Script 1a completed successfully!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
