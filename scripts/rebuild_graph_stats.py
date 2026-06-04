"""
rebuild_graph_stats.py
----------------------
Reconstruct the full data/processed/graph_stats.csv file by loading all
precomputed PyTorch Geometric .pt graph files from data/processed/graphs/.
This is useful if a Stage 5 precomputation run was interrupted and resumed,
causing the original CSV to be truncated.

Run from project root:
    python scripts/rebuild_graph_stats.py
"""

import csv
import logging
import sys
from pathlib import Path
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rebuild_graph_stats")

GRAPHS_DIR = PROJECT_ROOT / "data" / "processed" / "graphs"
STATS_CSV  = PROJECT_ROOT / "data" / "processed" / "graph_stats.csv"


def rebuild_stats():
    if not GRAPHS_DIR.exists():
        log.error("Graphs directory not found at %s", GRAPHS_DIR)
        return

    # Find all .pt files
    pt_files = sorted(list(GRAPHS_DIR.glob("**/*.pt")))
    log.info("Found %d graph .pt files to process.", len(pt_files))

    if not pt_files:
        log.warning("No .pt files found in %s", GRAPHS_DIR)
        return

    all_stats = []

    for path in tqdm(pt_files, desc="Reading graphs"):
        try:
            # Load PyG data object
            # PyG Data contains custom structures, load with weights_only=False
            data = torch.load(str(path), map_location="cpu", weights_only=False)
            
            stats = getattr(data, "graph_stats", {})
            if not stats:
                log.warning("No graph_stats found in %s", path.name)
                continue
            
            # Recreate stem and split from file path
            stats["stem"] = path.stem
            stats["split"] = path.parent.name
            
            all_stats.append(stats)
        except Exception as e:
            log.error("Failed to read %s: %s", path.name, e)

    if not all_stats:
        log.warning("No stats extracted.")
        return

    # Flatten nested dicts (e.g. tier_counts → tier_counts_flat, etc.)
    def _flatten_stats(row):
        flat = {}
        for k, v in row.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat[f"{k}_{k2}"] = v2
            else:
                flat[k] = v
        return flat

    flat_rows = [_flatten_stats(row) for row in all_stats]

    # Ensure output directory exists
    STATS_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(STATS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_rows[0].keys())
        writer.writeheader()
        writer.writerows(flat_rows)
        
    log.info("Successfully rebuilt and saved %d records to %s", len(flat_rows), STATS_CSV)


if __name__ == "__main__":
    rebuild_stats()
