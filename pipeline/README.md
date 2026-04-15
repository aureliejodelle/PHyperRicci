# Protein Knot Analysis Pipeline

Automated pipeline for building a topological feature dataset of knotted and
slipknotted proteins from KnotProt.  Starting from a list of PDB IDs, it
downloads structures, computes **persistent homology** (PH), builds
**hypergraphs** from PH cycle representatives, and computes **Forman-Ricci
curvature** on each hypergraph — producing the feature set used for
PHypeRicci distributional tests and kernel classification.

---

## Directory layout

```
phd/
├── AllScripts/
│   └── computation_codes/          ← this directory
│       ├── run_pipeline.py         ← master entry point
│       ├── config.py               ← all paths / constants
│       ├── fetch_protein_ids.py    ← step 1a
│       ├── fetch_protein_ids_similar_chain.py  ← step 1b
│       ├── extract_knot_data.py    ← step 2
│       ├── download_structures.py  ← step 3
│       ├── classify_pdb_homologs.py ← step 3b (optional)
│       ├── persistent_homology.jl  ← step 4  (Julia)
│       ├── compute_hypergraphs.py  ← step 5
│       └── compute_curvature.py    ← step 6
└── Database/
    ├── raw_data/
    │   ├── protein_lists/          ← CSV of all protein IDs (step 1a output)
    │   ├── similar_chains/         ← CSV of homologs (step 1b output)
    │   ├── knot_data/              ← CSV of knot cores (step 2 output)
    │   └── coordinates_data/       ← Cα CSVs organised by class (step 3 output)
    │       ├── K41/
    │       ├── Kplus31/
    │       └── ...
    └── processed_data/
        ├── Persistent_homology/    ← step 4 output (JSON per protein)
        │   └── <class>/
        │       ├── PH_1/           ← full data (barcodes + reps)
        │       ├── barcodes/       ← birth/death pairs only
        │       └── representatives/ ← cycle representatives only
        ├── hypergraphs/            ← step 5 output
        │   └── <class>/
        │       ├── hyperedge_map/
        │       └── summary/
        ├── ricci_curvature/        ← step 6 output (raw F(e) values)
        ├── normalised_ricci_curvature/
        ├── ratio_ricci_curvature/
        └── residualised_ricci_curvature/
```

---

## Prerequisites

### Python packages

```bash
pip install requests beautifulsoup4 pandas tqdm scipy numpy
pip install selenium          # step 1b — browser automation
pip install topoly            # step 3b — knot detection
pip install shap umap-learn scikit-learn  # downstream analysis
```

### ChromeDriver (step 1b)

Step 1b drives a headless Chrome browser.  ChromeDriver must match your
installed Chrome version.

```bash
# Debian / Ubuntu
sudo apt install chromium-driver

# Or download from https://chromedriver.chromium.org/downloads
```

### Julia + Ripserer.jl (step 4)

```bash
# Install Julia via Juliaup (recommended)
curl -fsSL https://install.julialang.org | sh

# Then install required packages inside Julia:
julia -e 'using Pkg; Pkg.add(["Ripserer", "DataFrames", "CSV", "JSON", "OrderedCollections"])'
```

---

## Quick start

```bash
cd phd/AllScripts/computation_codes

# Check current state of the pipeline
python run_pipeline.py --status

# Run the full pipeline (steps 1a → 6)
python run_pipeline.py

# Run the full pipeline, disabling parallel execution of steps 1b and 2
python run_pipeline.py --no-parallel
```

---

## Pipeline steps

| Step | Script | Language | Input | Output |
|------|--------|----------|-------|--------|
| 1a | `fetch_protein_ids.py` | Python | KnotProt / RCSB API | `protein_lists/all_protein_ids.csv` |
| 1b | `fetch_protein_ids_similar_chain.py` | Python + Selenium | `all_protein_ids.csv` | `similar_chains/similar_chains_simple.csv` |
| 2  | `extract_knot_data.py` | Python | `all_protein_ids.csv` | `knot_data/knot_data_full.csv` |
| 3  | `download_structures.py` | Python | `similar_chains_simple.csv` | `coordinates_data/<class>/` |
| 3b | `classify_pdb_homologs.py` | Python + Topoly | `<class>_PDB_Homologs/` | `topology_classification.json` |
| 4  | `persistent_homology.jl` | Julia | `coordinates_data/` | `Persistent_homology/<class>/` |
| 5  | `compute_hypergraphs.py` | Python | `Persistent_homology/` | `hypergraphs/<class>/` |
| 6  | `compute_curvature.py` | Python | `hypergraphs/` | `ricci_curvature/<class>/` |

### Dependency graph

```
Step 1a
  ├──► Step 1b  ─┐
  ├──► Step 2    ├─ run in parallel
  └──► Step 3  ──┘
             └──► Step 4  (Julia, multi-threaded)
                   └──► Step 5
                         └──► Step 6

Optional (run explicitly):
  Step 3b  depends on: 3, 4, 5, 6
```

---

## CLI reference

### `run_pipeline.py`

```
usage: python run_pipeline.py [--step STEP] [--steps STEP [STEP ...]]
                               [--classes CLASS [CLASS ...]]
                               [--force] [--no-parallel] [--dry-run] [--status]
```

| Flag | Description |
|------|-------------|
| *(no flags)* | Run the full pipeline (steps 1a → 6) |
| `--step STEP` | Run a single step (dependencies auto-satisfied) |
| `--steps STEP ...` | Run multiple specific steps sequentially |
| `--classes CLASS ...` | Restrict to specific knot/slip classes (see list below) |
| `--force` | Re-run even if outputs already exist |
| `--no-parallel` | Disable parallel execution of steps 1b and 2 |
| `--dry-run` | Step 3b only: preview without copying files |
| `--status` | Show per-step completion state and exit |

#### Available classes

| Class name | Description |
|------------|-------------|
| `K+3(1)` | Positive trefoil knot |
| `K-3(1)` | Negative trefoil knot |
| `K4(1)` | Figure-eight knot |
| `K-5(2)` | Three-twist knot |
| `S+3(1)` | Positive trefoil slipknot |
| `S-3(1)` | Negative trefoil slipknot |
| `S4(1)` | Figure-eight slipknot |
| `AOTCases` | OTCase proteins (unknotted control) |

#### Examples

```bash
# Run the entire pipeline
python run_pipeline.py

# Run just step 4 (Julia PH) — dependencies auto-checked
python run_pipeline.py --step 4

# Run steps 5 and 6 sequentially
python run_pipeline.py --steps 5 6

# Fetch protein IDs for one class only
python run_pipeline.py --step 1a --classes "K+3(1)"

# Download structures for two classes, force re-download
python run_pipeline.py --step 3 --classes "K4(1)" "K-5(2)" --force

# Run PH only for the figure-eight knot class
python run_pipeline.py --step 4 --classes "K4(1)"

# Classify PDB homologs (preview first, then apply)
python run_pipeline.py --step 3b --dry-run
python run_pipeline.py --step 3b

# Classify PDB homologs for specific classes
python run_pipeline.py --step 3b --classes "K4(1)" "S4(1)"

# Check what is done
python run_pipeline.py --status
```

---

### `classify_pdb_homologs.py`

Can also be run directly with its own argument parser:

```bash
python classify_pdb_homologs.py                       # all classes
python classify_pdb_homologs.py --dry-run             # preview only
python classify_pdb_homologs.py --classes K41 S41     # specific classes
python classify_pdb_homologs.py --base /path/to/Database
```

Classifies proteins in `<class>_PDB_Homologs/` as knotted or unknotted using
the Topoly Alexander polynomial (stochastic TWO_POINTS closure, 200 trials)
and copies coordinate + processed-data files into the appropriate destination:

- Knotted → `<class>_KnotProt_Homologs/`
- Unknotted → `<class>_Unknotted_Homologs/`

Outputs:
- `Database/topology_classification.json` — flat map `{pid: {topology, class}}`
- `Database/classify_pdb_homologs_log.json` — detailed per-class run log

---

## Configuration (`config.py`)

All paths, API endpoints, and pipeline constants live in `config.py`.  
Edit this file to change the database root or add new knot types.

| Setting | Default | Description |
|---------|---------|-------------|
| `BASE_DIR` | `../../Database` | Root of all data |
| `KNOT_TYPES` | dict | Knot/slip class names, page counts, KnotProt URLs |
| `DOWNLOAD_DELAY` | `1.0 s` | Polite delay between KnotProt requests |
| `SELENIUM_DELAY` | `2.0 s` | Wait time for JS rendering in step 1b |
| `MAX_RETRIES` | `3` | HTTP retry count |
| `TIMEOUT` | `30 s` | HTTP request timeout |

To add a new protein class, append an entry to `KNOT_TYPES`:

```python
"K7(4)": {
    "pages": 1,
    "url": "https://knotprot.cent.uw.edu.pl/results/page/{page}/?set=True&bridgeType=probab&knotTypes=74"
}
```

---

## Resuming interrupted runs

All steps support incremental resumption:

- **Steps 1a, 1b, 2**: if `--classes` is passed, rows for those classes are
  replaced in the existing CSV while other rows are preserved.
- **Step 2**: skips proteins already present in `knot_data_full.csv`.
- **Step 3**: skips protein CSV files that already exist in `coordinates_data/`.
- **Step 4**: skips proteins whose three output JSONs already exist.
- **Steps 5, 6**: skip proteins whose output JSON already exists.

To force a full re-run of any step: `python run_pipeline.py --step N --force`

---

## Logs

Each step writes a timestamped log to `Database/logs/step<N>_<timestamp>.log`.
Logs capture both stdout and stderr of the child script.

```bash
ls Database/logs/
# step1a_20260415_143022.log
# step4_20260415_143800.log
# ...
```

---

## Output formats

### Step 4 — Persistent homology (`Persistent_homology/<class>/`)

**`PH_1/<protein_id>.json`** — full data:
```json
{
  "protein_id": "1XD3_A",
  "class":      "Kplus31",
  "dim_0_barcode":   [[0, 2.14], [0, null], ...],
  "dim_1_barcode":   [[1.83, 4.92], ...],
  "representatives": [[[12, 15], [15, 18], ...], ...]
}
```

**`barcodes/<protein_id>.json`** — birth/death pairs only (for visualization).  
**`representatives/<protein_id>.json`** — cycle edge lists only (input to step 5).

### Step 5 — Hypergraphs (`hypergraphs/<class>/`)

**`hyperedge_map/<protein_id>.json`** — `{hyperedge_id: [node_indices]}`:
```json
{"1": [12, 15, 18, 22], "2": [5, 8, 11], ...}
```

**`summary/<protein_id>.json`** — protein-level stats:
```json
{"protein_id": "1XD3_A", "class": "Kplus31", "n_nodes": 47, "n_hyperedges": 8}
```

### Step 6 — Forman-Ricci curvature (`ricci_curvature/<class>/`)

**`<protein_id>.json`** — `{hyperedge_id: F(e)}`:
```json
{"1": -6.0, "2": -4.0, "3": -8.0, ...}
```

`F(e) = 2|e| − D(e)` where `|e|` = hyperedge size, `D(e)` = sum of node degrees.
Values are non-positive for typical protein hypergraphs; more negative ⟹ more
entangled backbone loops.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| Step 1b hangs | ChromeDriver version mismatch | `chromedriver --version` vs `google-chrome --version` — must match |
| Step 1b finds 0 chains | KnotProt layout changed | Check the `rawdata` div IDs on a KnotProt page |
| Step 4 not found | Julia not on PATH | Install via Juliaup or set `JULIA_BIN` manually in `run_pipeline.py` |
| Step 4 OOM | Dataset too large for RAM | Reduce `--threads` or process one class at a time: `--step 4 --classes "K41"` |
| Step 3b: topoly error | Old Topoly version | `pip install --upgrade topoly` |
| Step 3b: no `.csv` in PDB_Homologs | Step 3 not yet run | Run step 3 first |
| `PIPELINE_CLASSES` leaks | Parent shell has env var set | `unset PIPELINE_CLASSES` before running |
