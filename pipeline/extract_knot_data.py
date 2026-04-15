#!/usr/bin/env python3
"""
Script 2: Extract Knot Core Data
--------------------------------
For all trefoil proteins, extracts:
- Full chain sequence
- Knot core range and sequence
- N-terminal and C-terminal ranges and sequences
- Other knot parameters

Input:  config.get_protein_ids_path()
Output: config.get_knot_data_path()
"""

import sys
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
from typing import Dict, Optional, List
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from config import config
import os


def get_selected_classes():
    """Read PIPELINE_CLASSES env var. Returns list of classes or None (= all)."""
    raw = os.environ.get("PIPELINE_CLASSES", "").strip()
    return [c.strip() for c in raw.split(",") if c.strip()] if raw else None


class KnotDataExtractor:
    """Extracts knot core data from KnotProt."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (PhD Research Project)'
        })
    
    def extract_for_protein(self, pdb_id: str, chain: str) -> Dict[str, str]:
        """
        Extract all knot data for a protein.
        
        Args:
            pdb_id: PDB ID
            chain: Chain identifier
        
        Returns:
            Dictionary with all knot data
        """
        url = f"https://knotprot.cent.uw.edu.pl/view/{pdb_id.lower()}/{chain}/"
        
        result = {
            "chain_sequence": "",
            "sequence_length": 0,
            "knot_core_range": "",
            "knot_core_sequence": "",
            "knot_core_length": "",
            "knot_tails_range": "",
            "n_end_range": "",
            "n_end_sequence": "",
            "n_end_length": "",
            "c_end_range": "",
            "c_end_sequence": "",
            "c_end_length": "",
            "slipknot_tails_range": "",
            "slipknot_loops_range": "",
            "knot_type": ""
        }
        
        try:
            response = self.session.get(url, timeout=config.TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract sequence
            chain_sequence = self._extract_sequence(soup)
            
            if not chain_sequence:
                return result
            
            result["chain_sequence"] = self._format_sequence(chain_sequence)
            result["sequence_length"] = len(chain_sequence)
            
            # Extract knot data from table
            knot_table = self._extract_knot_table(soup)
            result.update(knot_table)
            
            # Calculate end ranges and extract subsequences
            if result["knot_core_range"] and result["sequence_length"] > 0:
                n_end_range, c_end_range = self._calculate_end_ranges(
                    result["knot_core_range"], result["sequence_length"]
                )
                
                result["n_end_range"] = n_end_range
                result["c_end_range"] = c_end_range
                
                # Extract subsequences
                result["knot_core_sequence"] = self._format_sequence(
                    self._extract_subsequence(chain_sequence, result["knot_core_range"])
                )
                result["n_end_sequence"] = self._format_sequence(
                    self._extract_subsequence(chain_sequence, n_end_range)
                )
                result["c_end_sequence"] = self._format_sequence(
                    self._extract_subsequence(chain_sequence, c_end_range)
                )
            
        except Exception as e:
            print(f" Error: {e}")
        
        return result
    
    def _extract_sequence(self, soup: BeautifulSoup) -> str:
        """Extract protein sequence from the page.
        Uses the same strategy as script 9 which was confirmed working:
        longest match first, then pre tags, then sequence div.
        """
        # Strategy 1 (from script 9): scan full page text for longest
        # amino-acid string — most reliable across different page layouts
        text = soup.get_text()
        sequences = re.findall(r'[ACDEFGHIKLMNPQRSTVWY\-]{30,}', text)
        if sequences:
            return max(sequences, key=len)

        # Strategy 2: pre tags
        pre_tags = soup.find_all('pre')
        for pre in pre_tags:
            t = pre.get_text(strip=True)
            if len(t) > 50 and re.match(r'^[ACDEFGHIKLMNPQRSTVWY\-]+$', t):
                return t

        # Strategy 3: sequence div with spans
        sequence_div = soup.find("div", {"id": "sequence"})
        if sequence_div:
            spans = sequence_div.find_all("span")
            full_sequence = "".join(span.get_text(strip=True) for span in spans)
            cleaned = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY\-]', '', full_sequence)
            if cleaned:
                return cleaned

        return ""
    
    def _extract_knot_table(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract knot data from the table."""
        rows = soup.find_all("tr")
        
        # Find the row with "Knot core range"
        header_row_idx = None
        for i, tr in enumerate(rows):
            if "Knot core range" in tr.get_text():
                header_row_idx = i
                break
        
        if header_row_idx is None:
            return {}
        
        # Get headers
        header_cells = rows[header_row_idx].find_all("th")
        headers = [h.get_text(strip=True) for h in header_cells]
        
        # Get data row
        if header_row_idx + 1 >= len(rows):
            return {}
        
        data_cells = rows[header_row_idx + 1].find_all("td")
        
        # Map headers to data
        result = {}
        header_mapping = {
            "Knot core range": "knot_core_range",
            "Knot core length": "knot_core_length",
            "Knot tails range": "knot_tails_range",
            "Slipknot tails range": "slipknot_tails_range",
            "Slipknot loops range": "slipknot_loops_range",
            "N-end length": "n_end_length",
            "C-end length": "c_end_length",
            "Type": "knot_type"
        }
        
        for i, header in enumerate(headers):
            if i < len(data_cells) and header in header_mapping:
                result[header_mapping[header]] = data_cells[i].get_text(strip=True)
        
        return result
    
    def _calculate_end_ranges(self, knot_core_range: str, sequence_length: int) -> tuple:
        """Calculate N-end and C-end ranges."""
        try:
            knot_start, knot_end = map(int, knot_core_range.split('-'))
            n_end = f"1-{knot_start - 1}" if knot_start > 1 else ""
            c_end = f"{knot_end + 1}-{sequence_length}" if knot_end < sequence_length else ""
            return n_end, c_end
        except:
            return "", ""
    
    def _extract_subsequence(self, sequence: str, range_str: str) -> str:
        """Extract subsequence from range."""
        if not sequence or not range_str or '-' not in range_str:
            return ""
        try:
            start, end = map(int, range_str.split('-'))
            return sequence[start-1:end]
        except:
            return ""
    
    def _format_sequence(self, sequence: str, width: int = 50) -> str:
        """Format sequence with line breaks."""
        if not sequence:
            return ""
        return '\n'.join(sequence[i:i+width] for i in range(0, len(sequence), width))


def main():
    """Main execution for Script 2."""
    print("=" * 60)
    print("SCRIPT 2: EXTRACT KNOT CORE DATA")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input: {config.get_protein_ids_path()}")
    print(f"Output: {config.get_knot_data_path()}")
    
    # Load protein list
    print("\n Loading protein list...")
    protein_file = config.get_protein_ids_path()
    
    if not protein_file.exists():
        print(f" Error: {protein_file} not found. Run script1a first.")
        return
    
    df_proteins = pd.read_csv(protein_file)
    
    # Filter for KnotProt proteins (excluding OTCase)
    knotprot_proteins = df_proteins[df_proteins["class"] != "OTCase"]

    # Further filter by selected classes if --classes was passed
    selected = get_selected_classes()
    if selected:
        knotprot_proteins = knotprot_proteins[knotprot_proteins["class"].isin(selected)]
        print(f" Filtering to selected classes: {', '.join(selected)}")

    print(f" Loaded {len(knotprot_proteins)} proteins to process")
    
    # Initialize extractor
    extractor = KnotDataExtractor()

    # Column order for output
    column_order = [
        "protein_id", "chain", "class", "protein_id_chain",
        "chain_sequence", "sequence_length",
        "knot_core_range", "knot_core_sequence", "knot_core_length",
        "knot_tails_range",
        "n_end_range", "n_end_sequence", "n_end_length",
        "c_end_range", "c_end_sequence", "c_end_length",
        "slipknot_tails_range", "slipknot_loops_range",
        "knot_type"
    ]

    output_file = config.get_knot_data_path()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Check how many already done (resume support)
    already_done = set()
    if output_file.exists():
        try:
            done_df = pd.read_csv(output_file)
            already_done = set(done_df["protein_id_chain"].dropna().tolist())
            print(f" Resuming — {len(already_done)} proteins already in CSV, skipping them.")
        except Exception:
            already_done = set()

    # Open CSV in append mode so each row is written immediately
    write_header = not output_file.exists() or len(already_done) == 0
    csv_file = open(output_file, "a", newline="", encoding="utf-8")
    import csv as csv_mod
    writer = csv_mod.DictWriter(csv_file, fieldnames=column_order,
                                extrasaction="ignore")
    if write_header:
        writer.writeheader()
        csv_file.flush()

    # Extract knot data
    print("\n Extracting knot data...")
    print(f" Writing incrementally to: {output_file}")

    total_proteins = len(knotprot_proteins)
    n_done = n_skipped = n_no_seq = 0
    start_time = time.time()

    for i, (idx, row) in enumerate(knotprot_proteins.iterrows(), 1):
        pdb_id = row["protein_id"]
        chain = row["chain"]
        protein_class = row["class"]
        protein_id_chain = f"{pdb_id}_{chain}"

        # Skip already processed
        if protein_id_chain in already_done:
            n_skipped += 1
            continue

        print(f"  [{i}/{total_proteins}] {protein_id_chain} ({protein_class})...", flush=True)

        knot_data = extractor.extract_for_protein(pdb_id, chain)

        # Add identifiers
        knot_data["protein_id"]       = pdb_id
        knot_data["chain"]            = chain
        knot_data["class"]            = protein_class
        knot_data["protein_id_chain"] = protein_id_chain

        # Write row immediately to disk
        writer.writerow(knot_data)
        csv_file.flush()
        n_done += 1

        # Print summary
        if knot_data.get("chain_sequence"):
            seq_len   = knot_data.get('sequence_length', '?')
            core_range = knot_data.get('knot_core_range', '')
            print(f" Sequence: {seq_len} aa, Knot core: {core_range}")
        else:
            n_no_seq += 1
            print(f" No sequence found")

        # Progress estimate every 10 proteins
        if i % 10 == 0 or i == total_proteins:
            elapsed = time.time() - start_time
            if n_done > 1 and elapsed > 0:
                avg_time = elapsed / n_done
                remaining = (total_proteins - i) * avg_time
                print(f" Est. remaining: {remaining/60:.1f} minutes")

        time.sleep(config.DOWNLOAD_DELAY)

    csv_file.close()
    print(f"\n Saved incrementally to {output_file}")
    print(f"   Processed this run : {n_done}")
    print(f"   Skipped (resumed)  : {n_skipped}")
    print(f"   No sequence found  : {n_no_seq}")

    # Summary statistics from the final CSV
    print("\n Summary:")
    try:
        df_knot = pd.read_csv(output_file)
        print(f"  Total rows in CSV  : {len(df_knot)}")
        print(f"  With sequences     : {df_knot['chain_sequence'].astype(bool).sum()}")
        print(f"  With knot cores    : {df_knot['knot_core_range'].astype(bool).sum()}")
        print("\n  By protein class:")
        for protein_class in sorted(df_knot["class"].dropna().unique()):
            class_df = df_knot[df_knot["class"] == protein_class]
            with_seq  = class_df['chain_sequence'].astype(bool).sum()
            with_core = class_df['knot_core_range'].astype(bool).sum()
            print(f"    {protein_class}: {len(class_df)} proteins "
                  f"(seq: {with_seq}, core: {with_core})")
    except Exception as e:
        print(f"  Could not read summary: {e}")

    print(f"\n Script 2 completed successfully!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    start_time = time.time()
    main()
