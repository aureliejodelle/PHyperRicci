# random_loops

Synthetic random 3D loop dataset for topological analysis benchmarking.

## What this is

Random closed curves (polygonal loops) are generated in 3D space using the
`topoly` library's `generate_loop()` function. Each loop is then classified
as a **knot** or **unknot** by computing its Jones polynomial:

- `poly == '0_1'` → unknot
- `poly != '0_1'` → knot

This gives a labelled synthetic dataset independent of any protein structure,
used to benchmark and validate the PHyperRicci pipeline (PH + hypergraph +
Forman-Ricci curvature) on a controlled setting where ground-truth topology
is known.

## Dataset structure

Loops are generated at 9 different lengths (L = number of segments):

| Length L | Replicates | Knots per replicate | Unknots per replicate |
|----------|------------|---------------------|-----------------------|
| 100 – 500 (step 50) | 10 each | 50 | 50 |

Each replicate is one JSON file. Files without `_ph` contain raw loop
coordinates; files with `_ph` also contain computed persistent homology.

```
outputs/
├── knots_{L}_{j}.json       # 50 knotted loops of length L, replicate j
├── knots_{L}_{j}_ph.json    # same + PH barcodes
├── unknots_{L}_{j}.json     # 50 unknotted loops of length L, replicate j
└── unknots_{L}_{j}_ph.json  # same + PH barcodes
```

Total: 360 JSON files, knots and unknots at L ∈ {100, 150, 200, 250, 300, 350, 400, 450, 500}.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `computations_jodelle_knots.ipynb` | Generate random loops, classify with Jones polynomial, save to `outputs/` |
| `Compute_PH.ipynb` | Compute persistent homology (H0, H1) on saved loops |
| `curvature_knot.ipynb` | Compute Forman-Ricci curvature on loop hypergraphs |
| `curvature_knots_150.ipynb` | Curvature analysis focused on L=150 loops |

## Requirements

```
topoly       # random loop generation + Jones polynomial
gudhi        # persistent homology
networkx     # hypergraph construction
numpy, matplotlib, seaborn
```
