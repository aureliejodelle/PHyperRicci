#!/usr/bin/env python3

import os
import sys
import csv
import time
import threading
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(str(Path(__file__).resolve().parent))
from config import config

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options


# CLASS FILTER (set by run_pipeline.py via env var)

def get_selected_classes():
    """Read PIPELINE_CLASSES env var. Returns list of classes or None (= all)."""
    raw = os.environ.get("PIPELINE_CLASSES", "").strip()
    return [c.strip() for c in raw.split(",") if c.strip()] if raw else None


# PATHS  (from config - single source of truth)

input_file  = config.get_protein_ids_path()
output_file = config.get_similar_chains_path()


# CONFIG

NUM_WORKERS = 4        # Number of parallel browser instances - increase if your machine can handle it
MAX_RETRIES = 2        # Retry failed proteins this many times
PAGE_TIMEOUT = 40      # Max seconds to wait for "loading..." to disappear


# THREAD-LOCAL DRIVER STORAGE
# Each thread gets its own Chrome instance

thread_local = threading.local()

def get_driver():
    if not hasattr(thread_local, "driver"):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        # Disable images/css to speed up page load
        prefs = {
            "profile.managed_default_content_settings.images": 2,
            "profile.managed_default_content_settings.stylesheets": 2,
        }
        chrome_options.add_experimental_option("prefs", prefs)
        thread_local.driver = webdriver.Chrome(options=chrome_options)
    return thread_local.driver


def quit_driver():
    if hasattr(thread_local, "driver"):
        try:
            thread_local.driver.quit()
        except:
            pass
        del thread_local.driver


# HELPERS

def simplify_chain(chain):
    """Convert AAA -> A, BBB -> B, etc."""
    if len(chain) == 3 and all(c == chain[0] for c in chain):
        return chain[0]
    return chain


def format_list_with_brackets(items_list):
    if not items_list:
        return "[]"
    return "[" + ", ".join(items_list) + "]"


def parse_span_text(text):
    """Parse semicolon/newline separated 'PROTID CHAIN' entries from a span."""
    results = []
    if not text:
        return results
    raw_entries = text.replace(";", "\n").split("\n")
    for entry in raw_entries:
        entry = entry.strip()
        if not entry or entry.startswith("#"):
            continue
        parts = entry.split()
        if len(parts) >= 2:
            pid = parts[0].strip()
            ch = parts[1].strip()
            # Simplify chain: AAA -> A
            if len(ch) == 3 and all(c == ch[0] for c in ch):
                ch = ch[0]
            results.append(f"{pid}_{ch}")
    return results



# CORE EXTRACTION (runs in each thread)

def extract_similar_chains(protein_id, chain, attempt=0):
    url = f"https://knotprot.cent.uw.edu.pl/view/{protein_id.lower()}/{chain}/"
    driver = get_driver()

    similar_chains = {
        "knotprot_similar": [],
        "unknotted_similar": [],
        "pdb_similar": []
    }

    try:
        driver.get(url)

        # Click simseq tab - use JS click to avoid interactability issues
        try:
            simseq_tab = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a[href='#simseq']"))
            )
            driver.execute_script("arguments[0].click();", simseq_tab)
        except:
            pass

        # Wait for "loading similar chains, please wait" to disappear
        # This is the main wait — no fixed sleep needed
        try:
            WebDriverWait(driver, PAGE_TIMEOUT).until(
                lambda d: "loading similar chains, please wait" not in d.page_source.lower()
            )
        except:
            # Timed out — page may still have partial data, continue anyway
            pass

        # Force-expand the rawdata collapse div so innerText is populated
        driver.execute_script(
            "var el = document.getElementById('rawdata'); if(el) { el.classList.add('in'); el.style.display='block'; }"
        )

        # Small fixed wait just for DOM to reflect the expansion
        time.sleep(0.5)

        span_map = {
            "locRaw":       "knotprot_similar",
            "unknottedRaw": "unknotted_similar",
            "remRaw":       "pdb_similar",
        }

        for span_id, category in span_map.items():
            # Use textContent (always available) rather than innerText (requires visibility)
            text = driver.execute_script(
                f"var el = document.getElementById('{span_id}'); return el ? el.textContent : '';"
            )
            similar_chains[category] = parse_span_text(text)

        return similar_chains

    except Exception as e:
        print(f"\n  Error on {protein_id}_{chain} (attempt {attempt+1}): {e}")
        if attempt < MAX_RETRIES:
            # Recreate driver on error and retry
            quit_driver()
            time.sleep(2)
            return extract_similar_chains(protein_id, chain, attempt + 1)
        return None


# WORKER FUNCTION (called per thread)

def process_row(row_data):
    protein_id, chain, protein_class = row_data

    similar_data = extract_similar_chains(protein_id, chain)

    if similar_data:
        knotprot_list  = similar_data["knotprot_similar"]
        unknotted_list = similar_data["unknotted_similar"]
        pdb_list       = similar_data["pdb_similar"]
        success = True
    else:
        knotprot_list = unknotted_list = pdb_list = []
        success = False

    protein_id_chain = f"{protein_id}_{chain}" if chain else protein_id

    return {
        "protein_id_chain":          protein_id_chain,
        "class":                     protein_class,
        "similar_chains_knotprot":   format_list_with_brackets(knotprot_list),
        "similar_chains_unknotted":  format_list_with_brackets(unknotted_list),
        "similar_chains_pdb":        format_list_with_brackets(pdb_list),
        "_success":                  success,
        "_knotprot_count":           len(knotprot_list),
        "_unknotted_count":          len(unknotted_list),
        "_pdb_count":                len(pdb_list),
    }



# MAIN

if not os.path.exists(input_file):
    raise FileNotFoundError(f"File not found: {input_file}")

print("\n" + "="*60)
print("SCRIPT 1b: FETCH SIMILAR CHAINS (PARALLEL)")
print("="*60)

start_time = datetime.now()
print(f" Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f" Input:      {input_file}")
print(f" Output:     {output_file}")
print(f" Workers:    {NUM_WORKERS}")

df = pd.read_csv(input_file)

selected = get_selected_classes()
if selected:
    df = df[df["class"].isin(selected)]
    print(f"\n Filtering to selected classes: {', '.join(selected)}")

print(f"\n Processing {len(df)} proteins...")
print("\n Proteins by class:")
for class_name, count in df.groupby("class").size().items():
    print(f"  {class_name}: {count}")

# Build work list
work_items = []
for _, row in df.iterrows():
    protein_id = str(row["protein_id"]).upper()
    chain = str(row["chain"]).upper() if not pd.isna(row["chain"]) else ""
    chain = simplify_chain(chain)
    work_items.append((protein_id, chain, row["class"]))


# OPEN OUTPUT FILE EARLY - write progressively so no data is lost on crash
fieldnames = [
    "protein_id_chain",
    "class",
    "similar_chains_knotprot",
    "similar_chains_unknotted",
    "similar_chains_pdb",
]

results        = []
failed_count   = 0
# When filtering classes, merge results back into existing CSV
_write_mode = "w"
_existing_df = None
if selected and os.path.exists(output_file):
    try:
        _existing_df = pd.read_csv(output_file)
        # Remove rows for classes we're re-fetching (will be replaced)
        _existing_df = _existing_df[~_existing_df["class"].isin(selected)]
        print(f"\n Will merge results with existing {output_file}")
    except Exception:
        _existing_df = None

out_f          = open(output_file, "w", newline="", encoding="utf-8")
writer         = csv.DictWriter(out_f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
writer.writeheader()

# Write retained rows from previous run first
if _existing_df is not None and not _existing_df.empty:
    for _, _row in _existing_df.iterrows():
        writer.writerow({k: _row.get(k, "") for k in fieldnames})
write_lock     = threading.Lock()

def save_result(result):
    row = {k: result[k] for k in fieldnames}
    with write_lock:
        writer.writerow(row)
        out_f.flush()

try:
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_row, item): item for item in work_items}

        with tqdm(total=len(work_items), desc="Processing") as pbar:
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    pid, ch, cls = item
                    print(f"\n  ✗ Fatal error for {pid}_{ch}: {e}")
                    result = {
                        "protein_id_chain":         f"{pid}_{ch}",
                        "class":                    cls,
                        "similar_chains_knotprot":  "[]",
                        "similar_chains_unknotted": "[]",
                        "similar_chains_pdb":        "[]",
                        "_success":                 False,
                        "_knotprot_count":           0,
                        "_unknotted_count":          0,
                        "_pdb_count":                0,
                    }

                save_result(result)
                results.append(result)

                if not result["_success"]:
                    failed_count += 1

                pbar.update(1)
                pbar.set_postfix({
                    "last": result["protein_id_chain"],
                    "kp":   result["_knotprot_count"],
                    "un":   result["_unknotted_count"],
                    "pdb":  result["_pdb_count"],
                    "fail": failed_count,
                })

finally:
    out_f.close()
    # Cleanly quit all thread-local drivers
    # (ThreadPoolExecutor threads are still alive briefly - quit via a final sweep)
    for t in threading.enumerate():
        if hasattr(t, "_target"):
            pass
    # Best-effort quit
    try:
        quit_driver()
    except:
        pass



# SUMMARY

end_time = datetime.now()
duration = end_time - start_time

print("\n" + "="*60)
print(" FINAL SUMMARY")
print("="*60)
print(f"\n  Total processed: {len(results)}")
print(f"  Failed/empty:    {failed_count}")
if results:
    print(f"  Success rate:    {((len(results)-failed_count)/len(results))*100:.1f}%")

total_kp = sum(r["_knotprot_count"]  for r in results)
total_un = sum(r["_unknotted_count"] for r in results)
total_pb = sum(r["_pdb_count"]       for r in results)

print(f"\n  KnotProt chains:  {total_kp}")
print(f"  Unknotted chains: {total_un}")
print(f"  PDB chains:       {total_pb}")

print(f"\n Done.  Output: {output_file}")
print(f" End:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total: {duration}")
