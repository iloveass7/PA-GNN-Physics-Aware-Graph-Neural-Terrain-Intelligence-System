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
import numpy as np
import rasterio

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
def _hazard_fraction(alias: str, labels_dir: Path, thresh: float = 0.7) -> float:
    """Fraction of valid pixels with risk > thresh, read from the relabeled risk tif.

    Used to stratify splits by hazard density so val/test aren't accidentally
    composed of low-hazard terrain. Falls back to 0.0 if the tif is missing.
    """
    risk_p = Path(labels_dir) / f"{alias}_risk.tif"
    if not risk_p.exists():
        log.warning("Risk tif missing for [%s] — hazard fraction defaults to 0.0", alias)
        return 0.0
    with rasterio.open(str(risk_p)) as ds:
        r = ds.read(1).ravel()
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    return float((r > thresh).mean())


# Acceptance criteria for a split assignment
SPLIT_MEAN_TOL = 0.08        # each split mean must be within ±8% of dataset mean
VAL_MAX_TERRAIN_FRAC = 0.67  # no single terrain may exceed this fraction of val
SEED_SEARCH_MAX = 200        # how many seeds to try before giving up


def _evaluate_assignment(buckets, haz, terrain_of, global_mean):
    """Return (ok, reason) for a candidate split assignment."""
    for s in ("train", "val", "test_in"):
        m = float(np.mean([haz[a] for a in buckets[s]])) if buckets[s] else 0.0
        if abs(m - global_mean) > SPLIT_MEAN_TOL:
            return False, f"{s} mean {m:.3f} off global {global_mean:.3f}"
    # no single terrain may dominate val
    val_terrains = [terrain_of[a] for a in buckets["val"]]
    if val_terrains:
        from collections import Counter
        top = Counter(val_terrains).most_common(1)[0][1]
        if top / len(val_terrains) > VAL_MAX_TERRAIN_FRAC:
            return False, f"val dominated by one terrain ({top}/{len(val_terrains)})"
    return True, "ok"


def _assign_once(ranked, haz, capacity, global_mean, rng):
    """One balanced-mean assignment pass.

    Walk locations high→low hazard; place each in the split (with remaining
    capacity) whose resulting mean would be CLOSEST to the dataset mean. This
    pulls all splits toward the same average instead of concentrating hazard.
    """
    buckets = {"train": [], "val": [], "test_in": []}
    sums = {"train": 0.0, "val": 0.0, "test_in": 0.0}

    for alias in ranked:
        open_splits = [s for s in ("train", "val", "test_in")
                       if len(buckets[s]) < capacity[s]]
        def resulting_gap(s):
            n = len(buckets[s]) + 1
            m = (sums[s] + haz[alias]) / n
            # tiny jitter breaks ties differently across seeds
            return (abs(m - global_mean), rng.random())
        target = min(open_splits, key=resulting_gap)
        buckets[target].append(alias)
        sums[target] += haz[alias]
    return buckets


def assign_splits(
    pairs: list[dict],
    seed: int = RANDOM_SEED,
    labels_dir: Path = LABELS_DIR,
) -> dict[str, list[str]]:
    """Assign DEM locations to train/val/test_in/test_ood so that every split
    MIRRORS THE DATASET hazard distribution.

    Method
    ------
    1. Select OOD (unchanged heuristic).
    2. Balanced-mean assignment: each location is placed in the split whose
       resulting mean hazard stays closest to the dataset mean.
    3. Seed search: retry with successive seeds until the assignment satisfies
       (a) every split mean within ±SPLIT_MEAN_TOL of the dataset mean, and
       (b) val not dominated by a single terrain. Guarantees a balanced,
       terrain-diverse val instead of relying on one lucky pass.

    Constraints preserved: location-level splits (no tile leakage); OOD terrain
    retains ≥2 locations. Fully deterministic given `seed`.
    """
    terrain_groups: dict[str, list[str]] = {}
    for p in pairs:
        terrain_groups.setdefault(p["terrain"], []).append(p["alias"])
    terrain_of = {p["alias"]: p["terrain"] for p in pairs}

    # --- OOD selection (unchanged) ---
    eligible = {t: a for t, a in terrain_groups.items() if len(a) >= 2}
    if not eligible:
        raise ValueError("No terrain type has ≥2 DEM locations — cannot select OOD.")
    ood_terrain = max(eligible, key=lambda t: len(eligible[t]))
    ood_alias = sorted(eligible[ood_terrain])[-1]
    log.info("OOD selection: terrain='%s', alias='%s'", ood_terrain, ood_alias)

    remaining = [p["alias"] for p in pairs if p["alias"] != ood_alias]
    haz = {a: _hazard_fraction(a, labels_dir) for a in remaining}
    haz[ood_alias] = _hazard_fraction(ood_alias, labels_dir)   # for logging only
    global_mean = float(np.mean([haz[a] for a in remaining]))
    log.info("Dataset mean hazard (non-OOD, risk>0.7): %.1f%%", 100 * global_mean)

    n = len(remaining)
    capacity = {
        "train": math.floor(n * TRAIN_RATIO),
        "val":   math.floor(n * VAL_RATIO),
    }
    capacity["test_in"] = n - capacity["train"] - capacity["val"]

    # --- Seed search for a balanced, terrain-diverse assignment ---
    chosen, used_seed = None, None
    for s in range(seed, seed + SEED_SEARCH_MAX):
        rng = random.Random(s)
        ranked = sorted(remaining, key=lambda a: (haz[a], rng.random()), reverse=True)
        buckets = _assign_once(ranked, haz, capacity, global_mean, rng)
        ok, reason = _evaluate_assignment(buckets, haz, terrain_of, global_mean)
        if ok:
            chosen, used_seed = buckets, s
            break

    if chosen is None:
        log.warning("No seed in [%d,%d) met both criteria — using best-effort "
                    "balanced pass at seed %d. Inspect the split manually.",
                    seed, seed + SEED_SEARCH_MAX, seed)
        rng = random.Random(seed)
        ranked = sorted(remaining, key=lambda a: (haz[a], rng.random()), reverse=True)
        chosen = _assign_once(ranked, haz, capacity, global_mean, rng)
        used_seed = seed
    else:
        log.info("Balanced assignment found at seed=%d", used_seed)

    def _mean(al): return float(np.mean([haz[x] for x in al])) if al else 0.0
    log.info("Splits (dataset-mirrored): train=%d (%.1f%%), val=%d (%.1f%%), "
             "test_in=%d (%.1f%%), test_ood=1 (%.1f%%)  [global %.1f%%]",
             len(chosen["train"]), 100*_mean(chosen["train"]),
             len(chosen["val"]),   100*_mean(chosen["val"]),
             len(chosen["test_in"]), 100*_mean(chosen["test_in"]),
             100*haz[ood_alias], 100*global_mean)
    for sp in ("train", "val", "test_in"):
        detail = ", ".join(f"{a}[{terrain_of[a][:4]}]({100*haz[a]:.0f}%)"
                           for a in sorted(chosen[sp], key=lambda x: -haz[x]))
        log.info("  %s: %s", sp, detail)

    return {"train": chosen["train"], "val": chosen["val"],
            "test_in": chosen["test_in"], "test_ood": [ood_alias]}
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
