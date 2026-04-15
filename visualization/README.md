# visualization/

Plotting and feature-extraction scripts for PHyperRicci.

## Scripts

| Script | Step | Description |
|--------|------|-------------|
| `run_visualizations.py` | orchestrator | Runs all visualization steps in order with skip-if-done logic |
| `visualize_ph.py` | v1 | Persistence diagrams, barcodes, and most-persistent cycle plots per protein |
| `extract_features.py` | v2 | Extracts 19 numerical features per protein (PH + HG + Ricci) into `protein_features.csv` |
| `statistical_tests.py` | v3 | KS and Levene tests comparing original class vs homolog groups on Curv_median |
| `analysis_proteins_vs_homologs.py` | v4 | Per-class comparison plots: original vs unknotted vs KnotProt homologs |

## Configuration

All scripts import from `pipeline/config.py` (the single shared config).
No separate config file exists in this folder — both pipeline and visualization
scripts share the same `Config` singleton.

## Usage

```bash
# Run all steps in order
python run_visualizations.py

# Run a specific step
python run_visualizations.py --step v1
python run_visualizations.py --step v2

# Step-specific flags
python run_visualizations.py --step v1 --plots barcodes
python run_visualizations.py --step v1 --classes K41 Kplus31
python run_visualizations.py --step v1 --group k
python run_visualizations.py --step v3 --three-way
python run_visualizations.py --step v4 --class AOTCases

# Re-run even if outputs exist
python run_visualizations.py --force

# Show pipeline status
python run_visualizations.py --status

# Run individual scripts directly
python visualize_ph.py
python extract_features.py
python statistical_tests.py
python analysis_proteins_vs_homologs.py
```

## Inputs required

- Step 4 output: `Database/processed_data/Persistent_homology/<class>/`
- Step 5 output: `Database/processed_data/hypergraphs/<class>/`
- Step 6 output: `Database/processed_data/ricci_curvature/<class>/`
- Step 3 output: `Database/raw_data/coordinates_data/<class>/`

## Outputs

- `Database/results/persistent_homology/` — persistence diagram PDFs per protein
- `Database/results/features/protein_features.csv` — flat feature table
- `Database/results/Features_Analysis/` — comparison plots
