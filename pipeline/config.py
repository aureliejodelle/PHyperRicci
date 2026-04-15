"""
config.py
=========
Single shared configuration for the PHyperRicci pipeline.
Used by both pipeline/ (computation) and visualization/ scripts.

Import from pipeline/ scripts:
    from config import config

Import from visualization/ scripts:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent / "pipeline"))
    from config import config

Directory layout expected on disk
──────────────────────────────────
  phd/
    Database/
      raw_data/
        protein_lists/          <- step 1a output
        similar_chains/         <- step 1b output
        knot_data/              <- step 2 output
        coordinates_data/       <- step 3 output  (<class>/<protein>.csv)
      processed_data/
        Persistent_homology/    <- step 4 output
        hypergraphs/            <- step 5 output
        ricci_curvature/        <- step 6 output (raw)
        normalised_ricci_curvature/
        ratio_ricci_curvature/
        residualised_ricci_curvature/
      results/
        persistent_homology/    <- visualization output
        features/               <- feature CSV output
        Features_Analysis/      <- comparison plots
        logs/
    AllScripts/
      PHyperRicci/
        pipeline/               <- this file lives here
        visualization/
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict

# Anchor to THIS file's location (pipeline/).
# All paths derived from here so the config is portable across machines.
_HERE = Path(__file__).resolve().parent

# Two levels up from pipeline/ -> phd/Database/
_DB = _HERE / "../../Database"


@dataclass
class Config:
    """
    Unified configuration for computation and visualization scripts.

    Computation paths (steps 1-6)
    ------------------------------
    PROTEIN_LISTS_DIR, SIMILAR_CHAINS_DIR, KNOT_DATA_DIR, COORDINATES_DIR
    PH_DIR, HG_DIR, RC_DIR  (and normalised / ratio / residualised variants)

    Visualization / results paths
    ------------------------------
    RESULTS_DIR, VIZ_PH_DIR, VIZ_FEATURES_DIR, VIZ_COMPARE_DIR, LOGS_DIR

    API / request settings
    -----------------------
    RCSB_SEARCH_URL, PDB_DOWNLOAD_URL, DOWNLOAD_DELAY, SELENIUM_DELAY,
    MAX_RETRIES, TIMEOUT

    Knot class registry
    --------------------
    KNOT_TYPES  -- {class_name: {"pages": N, "url": template}}
    SIMILARITY_TYPES  -- {key: folder_suffix}
    """

    # ── Raw data ───────────────────────────────────────────────────────────────
    BASE_DIR:           Path = field(default_factory=lambda: _DB)
    RAW_DATA_DIR:       Path = field(default_factory=lambda: _DB / "raw_data")
    PROCESSED_DATA_DIR: Path = field(default_factory=lambda: _DB / "processed_data")

    PROTEIN_LISTS_DIR:  Path = field(default_factory=lambda: _DB / "raw_data/protein_lists")
    SIMILAR_CHAINS_DIR: Path = field(default_factory=lambda: _DB / "raw_data/similar_chains")
    KNOT_DATA_DIR:      Path = field(default_factory=lambda: _DB / "raw_data/knot_data")
    COORDINATES_DIR:    Path = field(default_factory=lambda: _DB / "raw_data/coordinates_data")

    # ── Processed data ─────────────────────────────────────────────────────────
    PH_DIR:             Path = field(default_factory=lambda: _DB / "processed_data/Persistent_homology")
    HG_DIR:             Path = field(default_factory=lambda: _DB / "processed_data/hypergraphs")
    RC_DIR:             Path = field(default_factory=lambda: _DB / "processed_data/ricci_curvature")
    RC_NORM_DIR:        Path = field(default_factory=lambda: _DB / "processed_data/normalised_ricci_curvature")
    RC_RATIO_DIR:       Path = field(default_factory=lambda: _DB / "processed_data/ratio_ricci_curvature")
    RC_RESID_DIR:       Path = field(default_factory=lambda: _DB / "processed_data/residualised_ricci_curvature")

    # ── Results / visualization ────────────────────────────────────────────────
    RESULTS_DIR:        Path = field(default_factory=lambda: _DB / "results")
    VIZ_PH_DIR:         Path = field(default_factory=lambda: _DB / "results/persistent_homology")
    VIZ_FEATURES_DIR:   Path = field(default_factory=lambda: _DB / "results/features")
    VIZ_COMPARE_DIR:    Path = field(default_factory=lambda: _DB / "results/Features_Analysis")
    LOGS_DIR:           Path = field(default_factory=lambda: _DB / "results/logs")

    # ── Knot class registry ────────────────────────────────────────────────────
    KNOT_TYPES: Dict = field(default_factory=lambda: {
        "K+3(1)": {
            "pages": 6,
            "url": "https://knotprot.cent.uw.edu.pl/results/page/{page}/?set=True&bridgeType=probab&knotTypes=%2B31"
        },
        "K-3(1)": {
            "pages": 2,
            "url": "https://knotprot.cent.uw.edu.pl/results/page/{page}/?set=True&bridgeType=probab&knotTypes=-31"
        },
        "K4(1)": {
            "pages": 2,
            "url": "https://knotprot.cent.uw.edu.pl/results/page/{page}/?set=True&bridgeType=probab&knotTypes=41"
        },
        "K-5(2)": {
            "pages": 2,
            "url": "https://knotprot.cent.uw.edu.pl/results/page/{page}/?set=True&bridgeType=probab&knotTypes=-52"
        },
        "S+3(1)": {
            "pages": 10,
            "url": "https://knotprot.cent.uw.edu.pl/results/page/{page}/?set=True&bridgeType=probab&slipknotTypes=%2B31"
        },
        "S-3(1)": {
            "pages": 2,
            "url": "https://knotprot.cent.uw.edu.pl/results/page/{page}/?set=True&bridgeType=probab&slipknotTypes=-31"
        },
        "S4(1)": {
            "pages": 2,
            "url": "https://knotprot.cent.uw.edu.pl/results/page/{page}/?set=True&bridgeType=probab&slipknotTypes=41"
        },
        "AOTCases": {
            "pages": 2,
            "url": "https://knotprot.cent.uw.edu.pl/results/page/{page}/?pfam=OTCace&set=True&bridgeType="
        },
    })

    SIMILARITY_TYPES: Dict = field(default_factory=lambda: {
        "knotprot":  "similar_chains_knotprot",
        "unknotted": "similar_chains_unknotted",
        "pdb":       "similar_chains_pdb",
    })

    # ── API / request settings ─────────────────────────────────────────────────
    RCSB_SEARCH_URL:  str   = "https://search.rcsb.org/rcsbsearch/v2/query"
    PDB_DOWNLOAD_URL: str   = "https://files.rcsb.org/download/{}.pdb"
    DOWNLOAD_DELAY:   float = 1.0
    SELENIUM_DELAY:   float = 2.0
    MAX_RETRIES:      int   = 3
    TIMEOUT:          int   = 30

    # ── Standard file names ────────────────────────────────────────────────────
    PROTEIN_IDS_FILE:    str = "all_protein_ids.csv"
    SIMILAR_CHAINS_FILE: str = "similar_chains_simple.csv"
    KNOT_DATA_FILE:      str = "knot_data_full.csv"
    MASTER_PROTEIN_LIST: str = "master_protein_list.csv"

    # ──────────────────────────────────────────────────────────────────────────
    def __post_init__(self):
        """Create all output / results directories on first import."""
        for d in [
            self.RESULTS_DIR,
            self.VIZ_PH_DIR,
            self.VIZ_FEATURES_DIR,
            self.VIZ_COMPARE_DIR,
            self.LOGS_DIR,
            self.PROTEIN_LISTS_DIR,
            self.SIMILAR_CHAINS_DIR,
            self.KNOT_DATA_DIR,
            self.COORDINATES_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    # ── Path helpers (visualization) ───────────────────────────────────────────
    def rc_json_dir(self, class_name: str) -> Path:
        """Ricci curvature directory for one class: RC_DIR/<class>/"""
        return self.RC_DIR / class_name

    def hg_json_dir(self, class_name: str) -> Path:
        """Hyperedge map directory for one class: HG_DIR/<class>/hyperedge_map/"""
        return self.HG_DIR / class_name / "hyperedge_map"

    # ── Path helpers (computation) ─────────────────────────────────────────────
    def get_protein_ids_path(self) -> Path:
        return self.PROTEIN_LISTS_DIR / self.PROTEIN_IDS_FILE

    def get_similar_chains_path(self) -> Path:
        return self.SIMILAR_CHAINS_DIR / self.SIMILAR_CHAINS_FILE

    def get_knot_data_path(self) -> Path:
        return self.KNOT_DATA_DIR / self.KNOT_DATA_FILE

    def get_master_list_path(self) -> Path:
        return self.PROCESSED_DATA_DIR / self.MASTER_PROTEIN_LIST

    def get_knotprot_url(self, knot_type: str, page: int = 1) -> str:
        if knot_type not in self.KNOT_TYPES:
            raise ValueError(f"Unknown knot type: {knot_type}")
        url = self.KNOT_TYPES[knot_type]["url"]
        return url.format(page=page) if "{page}" in url else url

    # ── Class name helpers ─────────────────────────────────────────────────────
    @staticmethod
    def sanitize_class_name(class_name: str) -> str:
        """'K+3(1)' -> 'Kplus31',  'S-3(1)' -> 'Sminus31'"""
        return (
            class_name
            .replace("+", "plus").replace("-", "minus")
            .replace("(", "").replace(")", "").replace(" ", "_")
        )

    def folder_to_class(self) -> dict:
        """Return {folder_name: original_class_name} for all known classes."""
        return {self.sanitize_class_name(c): c for c in self.KNOT_TYPES}

    def resolve_classes(self, selected: list) -> list:
        """Translate class names -> sanitized folder names."""
        return [self.sanitize_class_name(c) for c in selected]

    # ── Summary ────────────────────────────────────────────────────────────────
    def summary(self) -> str:
        lines = [
            "PHyperRicci -- Configuration",
            "=" * 50,
            f"  PH_DIR       : {self.PH_DIR}",
            f"  HG_DIR       : {self.HG_DIR}",
            f"  RC_DIR       : {self.RC_DIR}",
            f"  COORDINATES  : {self.COORDINATES_DIR}",
            f"  RESULTS      : {self.RESULTS_DIR}",
            f"  LOGS         : {self.LOGS_DIR}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def __repr__(self):
        return f"Config(base={self.BASE_DIR})"


# ── Singleton ──────────────────────────────────────────────────────────────────
config = Config()


# ── Argparse stubs (backward compatibility for visualization scripts) ──────────
def add_dataset_arg(parser):
    """No-op stub kept for API compatibility."""
    pass


def apply_dataset_from_args(args):
    """No-op stub kept for API compatibility."""
    return "default"
