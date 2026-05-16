"""
tile_dataset.py
---------------
Stage 1 end-to-end runner.

Reads mars_terrain_vault.csv, tiles all 27 DEM pairs, assigns tiles to
train/val/test_in/test_ood splits by DEM location (never by individual tile),
and writes split .txt files.

Blueprint split rules (§8):
  - All tiles from ONE DEM location go into exactly ONE split.
  - Reserve ONE complete DEM location from a different geological region as OOD.
  - From remaining locations: 70% train, 15% val, 15% test_in.
  - OOD location must be from a terrain type that has ≥2 locations (so its
    removal doesn't eliminate that terrain from training entirely).

OOD selection heuristic (deterministic):
  - Candidate: the terrain type with the highest count gets one location
    withheld as OOD. This ensures OOD terrain still appears in train.
  - Tie-break: alphabetical by alias.

PERF-02: The tiling loop is parallelised with ProcessPoolExecutor.
Each pair's tiling is I/O-bound (reading GeoTIFFs, writing .npy files) and
completely independent.

Run from the pa-gnn/ project root:
    python scripts/tile_dataset.py [--overwrite] [--workers 4]

Outputs:
    data/processed/tiles/          — .npy tile quad files
    data/splits/train.txt
    data/splits/val.txt
    data/splits/test_in.txt
    data/splits/test_ood.txt
    data/processed/tile_manifest.csv — full per-tile record

Expected yield: 5,000 to 15,000 accepted 512×512 tiles across all 27 pairs.
"""

import argparse
import csv
import logging
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dem_loader import get_dem_pairs
from src.data.tiling import tile_dem_pair

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tile_dataset")

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

VAULT_CSV   = PROJECT_ROOT / "data" / "raw" / "mars_terrain_vault.csv"
DEM_DIR     = PROJECT_ROOT / "data" / "raw" / "dem"
BROWSE_DIR  = PROJECT_ROOT / "data" / "raw" / "hirise_browse"
TIF_CACHE   = PROJECT_ROOT / "data" / "processed" / "tif_cache"
ALIGNED_DIR = PROJECT_ROOT / "data" / "processed" / "aligned"
LABELS_DIR  = PROJECT_ROOT / "data" / "processed" / "labels"
TILES_DIR   = PROJECT_ROOT / "data" / "processed" / "tiles"
SPLITS_DIR  = PROJECT_ROOT / "data" / "splits"
MANIFEST    = PROJECT_ROOT / "data" / "processed" / "tile_manifest.csv"

# Split ratios (excluding OOD)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test_in = 1.0 - TRAIN_RATIO - VAL_RATIO = 0.15

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def assign_splits(
    pairs: list[dict],
    seed: int = RANDOM_SEED,
) -> dict[str, list[str]]:
    """Assign DEM location aliases to train/val/test_in/test_ood splits.

    Parameters
    ----------
    pairs : list[dict]
        Vault pairs from ``get_dem_pairs()``.  Must have 'alias' and 'terrain'.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys "train", "val", "test_in", "test_ood" mapping to
    lists of alias strings.
    """
    rng = random.Random(seed)

    # Group aliases by terrain type
    terrain_groups: dict[str, list[str]] = {}
    for p in pairs:
        terrain_groups.setdefault(p["terrain"], []).append(p["alias"])

    # Select OOD: from the terrain type with most locations (most budget to spare),
    # reserve the alphabetically-last alias as OOD.
    # Ensure the selected terrain has ≥ 2 locations so it still appears in training.
    eligible = {t: aliases for t, aliases in terrain_groups.items() if len(aliases) >= 2}
    if not eligible:
        raise ValueError("No terrain type has ≥2 DEM locations — cannot select OOD without "
                         "eliminating a terrain type from training.")

    ood_terrain = max(eligible, key=lambda t: len(eligible[t]))
    ood_alias = sorted(eligible[ood_terrain])[-1]   # deterministic: alphabetically last
    log.info("OOD selection: terrain='%s', alias='%s'", ood_terrain, ood_alias)

    # Remaining aliases (all except OOD)
    remaining = [p["alias"] for p in pairs if p["alias"] != ood_alias]
    rng.shuffle(remaining)

    n = len(remaining)
    n_train = math.floor(n * TRAIN_RATIO)
    n_val   = math.floor(n * VAL_RATIO)

    train_aliases   = remaining[:n_train]
    val_aliases     = remaining[n_train:n_train + n_val]
    test_in_aliases = remaining[n_train + n_val:]

    log.info(
        "Splits: train=%d, val=%d, test_in=%d, test_ood=%d locations",
        len(train_aliases), len(val_aliases), len(test_in_aliases), 1,
    )
    return {
        "train":    train_aliases,
        "val":      val_aliases,
        "test_in":  test_in_aliases,
        "test_ood": [ood_alias],
    }


def write_split_files(splits: dict[str, list[str]], splits_dir: Path) -> None:
    """Write one .txt file per split containing one alias per line."""
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, aliases in splits.items():
        out = splits_dir / f"{split_name}.txt"
        with open(out, "w") as f:
            f.write("\n".join(aliases) + "\n")
        log.info("Written: %s (%d locations)", out.name, len(aliases))


# ---------------------------------------------------------------------------
# Per-pair tiling worker (must be at module level for pickling by PERF-02)
# ---------------------------------------------------------------------------

def _tile_one_pair(args_dict: dict) -> list[dict]:
    """Tile a single DEM pair.  Runs in a child process.

    Parameters
    ----------
    args_dict : dict
        Keys: alias, split, pair (vault dict), overwrite (bool)

    Returns
    -------
    list of tile record dicts (with 'split' key injected)
    """
    # Re-configure logging in child process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    _log = logging.getLogger("tile_dataset.worker")

    alias    = args_dict["alias"]
    split    = args_dict["split"]
    pair     = args_dict["pair"]
    overwrite = args_dict["overwrite"]

    # Check required inputs exist
    aligned_tif  = ALIGNED_DIR / (pair["ortho_tif"].stem + "_aligned.tif")
    risk_tif     = LABELS_DIR / f"{alias}_risk.tif"
    hazard_tif   = LABELS_DIR / f"{alias}_hazard.tif"
    validity_tif = LABELS_DIR / f"{alias}_validity.tif"

    missing = [p for p in [aligned_tif, risk_tif, hazard_tif, validity_tif]
               if not p.exists()]
    if missing:
        _log.error(
            "Missing Stage 1 label outputs for [%s]. "
            "Run `python scripts/process_dems.py` first.\n  Missing: %s",
            alias, [str(p) for p in missing],
        )
        return []

    try:
        records = tile_dem_pair(
            alias=alias,
            aligned_browse_tif=aligned_tif,
            risk_tif=risk_tif,
            hazard_tif=hazard_tif,
            validity_tif=validity_tif,
            output_dir=TILES_DIR / split,   # organise tiles by split subfolder
            overwrite=overwrite,
        )

        for rec in records:
            rec["split"] = split
        _log.info("[%s] → %d tiles in '%s'", alias, len(records), split)
        return records

    except Exception as exc:
        _log.exception("Tiling failed for [%s]: %s", alias, exc)
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(overwrite: bool = False, max_workers: int = 4) -> None:
    log.info("=" * 60)
    log.info("PA-GNN Stage 1 — Tile Dataset Builder")
    log.info("Workers: %d", max_workers)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load vault
    # ------------------------------------------------------------------
    pairs = get_dem_pairs(
        vault_csv=VAULT_CSV,
        dem_dir=DEM_DIR,
        browse_dir=BROWSE_DIR,
        tif_cache_dir=TIF_CACHE,
    )
    log.info("Vault: %d pairs loaded", len(pairs))

    # ------------------------------------------------------------------
    # Step 2: Assign splits (before tiling — determines which tiles go where)
    # ------------------------------------------------------------------
    splits = assign_splits(pairs)
    alias_to_split = {
        alias: split
        for split, aliases in splits.items()
        for alias in aliases
    }

    write_split_files(splits, SPLITS_DIR)

    # ------------------------------------------------------------------
    # Step 3: Tile each pair
    #         PERF-02: parallelised via ProcessPoolExecutor
    # ------------------------------------------------------------------
    all_records: list[dict] = []
    TILES_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-create split subdirectories
    for split_name in splits:
        (TILES_DIR / split_name).mkdir(parents=True, exist_ok=True)

    # Build work items
    work_items = []
    for pair in pairs:
        alias = pair["alias"]
        split = alias_to_split.get(alias, "unknown")
        work_items.append({
            "alias": alias,
            "split": split,
            "pair": pair,
            "overwrite": overwrite,
        })

    t0 = time.time()

    if max_workers <= 1:
        # Sequential fallback
        for i, item in enumerate(work_items, 1):
            log.info("--- %d/%d [%s] split=%s ---",
                     i, len(work_items), item["alias"], item["split"])
            all_records.extend(_tile_one_pair(item))
    else:
        log.info("Tiling %d pairs with %d workers...",
                 len(work_items), max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_tile_one_pair, item): item["alias"]
                for item in work_items
            }
            for future in as_completed(futures):
                alias = futures[future]
                try:
                    records = future.result()
                    all_records.extend(records)
                except Exception as exc:
                    log.exception("Worker crashed for [%s]: %s", alias, exc)

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Step 4: Write tile manifest
    # ------------------------------------------------------------------
    if all_records:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "split", "alias", "row", "col",
            "image_npy", "risk_npy", "hazard_npy", "valid_npy",
            "nodata_frac", "sat_frac", "hazardous_frac",
        ]
        with open(MANIFEST, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_records)
        log.info("Tile manifest written: %s", MANIFEST)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    by_split: dict[str, int] = {}
    for rec in all_records:
        by_split[rec["split"]] = by_split.get(rec["split"], 0) + 1

    log.info("")
    log.info("=" * 60)
    log.info("Tiling complete (%.1fs, %d workers). Tile counts by split:",
             elapsed, max_workers)
    for split_name in ["train", "val", "test_in", "test_ood"]:
        log.info("  %-10s: %d tiles", split_name, by_split.get(split_name, 0))
    log.info("  TOTAL     : %d tiles", sum(by_split.values()))
    log.info("  Expected  : 5,000 – 15,000 (blueprint target)")
    log.info("=" * 60)

    if sum(by_split.values()) < 1000:
        log.warning("Tile count is very low (<1000). Check that process_dems.py ran "
                    "successfully and that aligned/ and labels/ directories are populated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile all DEM pairs and generate split files.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-tile even if .npy files already exist.",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: 4). Use 1 for sequential.",
    )
    args = parser.parse_args()
    main(overwrite=args.overwrite, max_workers=args.workers)
