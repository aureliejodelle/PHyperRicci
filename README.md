# PHyperRicci

PHyperRicci** is a geometric-topological framework for analysing the structural complexity of knotted proteins. It combines three complementary descriptors:

- Persistent homology (PH) Vietoris-Rips filtration on Cα coordinates (H0 components, H1 cycles)
- Hypergraph structure: each H1 cycle representative forms a hyperedge connecting its Cα atoms
- Forman-Ricci curvature: discrete curvature on hyperedges, capturing local geometric roughness

Together these form a multi-scale fingerprint of protein backbone topology for classification and distributional analysis across knot classes (K+3(1), K-3(1), K4(1), K-5(2)) and slipknot classes (S+3(1), S-3(1), S4(1)).



### Running the pipeline

```bash
cd pipeline/

# Run a single step for specific classes
python run_pipeline.py --step 1a
python run_pipeline.py --step 3 --classes "K+3(1)" "K4(1)"

# Run multiple steps in sequence
python run_pipeline.py --steps 1a 1b 2 3

# Dry-run to preview what would execute
python run_pipeline.py --step 3b --dry-run

# Step 4 (persistent homology) must be run manually — it is heavy
julia step4_persistent_homology.jl --class K41
```

> **Step 4 is never triggered automatically** because it can take hours per class. Always run it explicitly.

### Running visualizations

```bash
cd visualization/

# Run all visualization steps in order (skips already-done steps)
python run_visualizations.py

# Run a specific step
python run_visualizations.py --step v2
```

---

## Knot classes

| Class | Type | Description |
|-------|------|-------------|
| K+3(1) | Knot | Trefoil, positive handedness |
| K-3(1) | Knot | Trefoil, negative handedness |
| K4(1) | Knot | Figure-eight knot |
| K-5(2) | Knot | 5_2 knot, negative |
| S+3(1) | Slipknot | Slipknot with +3(1) sub-knot |
| S-3(1) | Slipknot | Slipknot with -3(1) sub-knot |
| S4(1) | Slipknot | Slipknot with 4(1) sub-knot |

---

## Results summary

Feature extraction produces **19 numerical features** per protein:

| Feature block | Features |
|--------------|----------|
| PH (H0) | `H0_count`, `H0_mean_pers`, `H0_max_pers` |
| PH (H1) | `H1_count`, `H1_mean_pers`, `H1_max6_pers`, `H1_median_pers`, `H1_total_pers` |
| Hypergraph | `HG_num_hyperedges`, `HG_mean_size`, `HG_max_size`, `HG_min_size` |
| Ricci curvature | `Curv_mean`, `Curv_median`, `Curv_std`, `Curv_min`, `Curv_max`, `Curv_skew`, `Curv_kurt` |

Statistical tests (KS, Levene) comparing original knot proteins vs sequence-similar homologs are in `results/Features_Analysis/`.
