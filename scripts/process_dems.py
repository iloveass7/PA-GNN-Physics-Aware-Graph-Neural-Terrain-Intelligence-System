"""
process_dems.py
---------------
Stage 1 pipeline runner: reads mars_terrain_vault.csv, converts all .IMG and .JP2
files to GeoTIFF (once, then cached), aligns browse images to their paired DEM,
and generates slope / roughness / risk / hazard label GeoTIFFs for every pair.

PERF-01: The per-pair loop is parallelised with ProcessPoolExecutor.
Each DEM pair (convert → align → label) is completely independent, so workers
can safely overlap.  gdalwarp subprocess calls dominate wall time (PERF-05);
parallelism hides this by overlapping multiple gdalwarp invocations.

Run from the pa-gnn/ project root:
    python scripts/process_dems.py [--workers 4]

Outputs are written to:
    data/processed/tif_cache/dem/    — converted DEM GeoTIFFs
    data/processed/tif_cache/browse/ — converted browse GeoTIFFs
    data/processed/aligned/          — browse GeoTIFFs aligned to DEM grid
    data/processed/labels/           — slope, roughness, risk, hazard GeoTIFFs
    data/processed/stage1_report.csv — summary of all processed pairs
"""
import argparse
import csv
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# --- Add project root to sys.path so src/ imports work ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dem_loader import get_dem_pairs, load_dem, validate_vault_files
from src.data.hirise_loader import align_browse_to_dem, load_browse
from src.data.dem_processing import process_dem_pair

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("process_dems")

# ---------------------------------------------------------------------------
# Path configuration (relative to pa-gnn/ project root)
# ---------------------------------------------------------------------------

VAULT_CSV   = PROJECT_ROOT / "data" / "raw" / "mars_terrain_vault.csv"
DEM_DIR     = PROJECT_ROOT / "data" / "raw" / "dem"
BROWSE_DIR  = PROJECT_ROOT / "data" / "raw" / "hirise_browse"

TIF_CACHE   = PROJECT_ROOT / "data" / "processed" / "tif_cache"
ALIGNED_DIR = PROJECT_ROOT / "data" / "processed" / "aligned"
LABELS_DIR  = PROJECT_ROOT / "data" / "processed" / "labels"
REPORT_CSV  = PROJECT_ROOT / "data" / "processed" / "stage1_report.csv"


# ---------------------------------------------------------------------------
# Per-pair worker function (must be at module level for pickling)
# ---------------------------------------------------------------------------

def _process_one_pair(pair: dict) -> dict:
    """Process a single DEM pair: convert → align → label.

    Designed to run in a child process via ProcessPoolExecutor.
    Returns a result dict suitable for the Stage 1 report CSV.
    """
    # Re-configure logging in child process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    _log = logging.getLogger("process_dems.worker")

    alias = pair["alias"]
    try:
        # 1. Convert .IMG → GeoTIFF (idempotent)
        with load_dem(
            dem_filename=pair["dem_img"].name,
            dem_dir=DEM_DIR,
            tif_cache_dir=TIF_CACHE / "dem",
        ) as _:
            pass

        # 2. Convert .JP2 → GeoTIFF (idempotent)
        with load_browse(
            browse_filename=pair["ortho_jp2"].name,
            browse_dir=BROWSE_DIR,
            tif_cache_dir=TIF_CACHE / "browse",
        ) as _:
            pass

        # 3. Align browse image to DEM pixel grid (idempotent)
        # NOTE (PERF-05): gdalwarp is CPU-only and dominates per-pair time.
        # Parallelism (PERF-01) overlaps multiple gdalwarp calls to mitigate.
        aligned_tif = align_browse_to_dem(
            dem_tif=pair["dem_tif"],
            browse_tif=pair["ortho_tif"],
            output_dir=ALIGNED_DIR,
        )

        # 4. Compute slope, roughness, risk, hazard labels
        result = process_dem_pair(
            alias=alias,
            dem_tif=pair["dem_tif"],
            output_dir=LABELS_DIR,
        )

        _log.info("✓ %s complete", alias)
        return {
            "alias": alias,
            "terrain": pair["terrain"],
            "scale": pair["scale"],
            "dem_tif": str(pair["dem_tif"]),
            "browse_tif": str(pair["ortho_tif"]),
            "aligned_tif": str(aligned_tif),
            "slope_tif": str(result["slope_tif"]),
            "roughness_tif": str(result["roughness_tif"]),
            "risk_tif": str(result["risk_tif"]),
            "hazard_tif": str(result["hazard_tif"]),
            "validity_tif": str(result["validity_tif"]),
            "pixel_size_m": result.get("pixel_size_m", ""),
            "nodata_fraction": f"{result.get('nodata_fraction', 0):.4f}",
            "hazardous_fraction": f"{result.get('hazardous_fraction', 0):.4f}",
            "status": "OK",
        }

    except Exception as exc:
        _log.exception("✗ Failed for %s: %s", alias, exc)
        return {
            "alias": alias,
            "terrain": pair.get("terrain", ""),
            "status": f"ERROR: {exc}",
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(max_workers: int = 4):
    log.info("=" * 60)
    log.info("PA-GNN Stage 1 — DEM Processing Pipeline")
    log.info("Project root : %s", PROJECT_ROOT)
    log.info("Vault CSV    : %s", VAULT_CSV)
    log.info("Workers      : %d", max_workers)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 0: Validate that all vault files exist on disk
    # ------------------------------------------------------------------
    log.info("Step 0: Validating vault file presence...")
    validation = validate_vault_files(VAULT_CSV, DEM_DIR, BROWSE_DIR)

    if validation["missing"]:
        log.error("Missing files detected:")
        for m in validation["missing"]:
            if m["dem_missing"]:
                log.error("  [DEM missing]  %s", m["dem_path"])
            if m["jp2_missing"]:
                log.error("  [JP2 missing]  %s", m["jp2_path"])
        log.error("Fix missing files before running Stage 1. Aborting.")
        sys.exit(1)

    log.info("All %d vault pairs validated ✓", len(validation["ok"]))

    # ------------------------------------------------------------------
    # Step 1: Load vault pairs (converts .IMG and .JP2 to GeoTIFF on demand)
    # ------------------------------------------------------------------
    log.info("Step 1: Loading vault pairs and converting to GeoTIFF...")
    pairs = get_dem_pairs(
        vault_csv=VAULT_CSV,
        dem_dir=DEM_DIR,
        browse_dir=BROWSE_DIR,
        tif_cache_dir=TIF_CACHE,
    )
    log.info("Loaded %d pairs from vault", len(pairs))

    # Ensure output directories exist before spawning workers
    (TIF_CACHE / "dem").mkdir(parents=True, exist_ok=True)
    (TIF_CACHE / "browse").mkdir(parents=True, exist_ok=True)
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 2: For each pair: convert, align, process
    #         PERF-01: parallelised via ProcessPoolExecutor
    # ------------------------------------------------------------------
    results: list[dict] = []
    t0 = time.time()

    if max_workers <= 1:
        # Sequential fallback (useful for debugging)
        for i, pair in enumerate(pairs, 1):
            log.info("--- Pair %d/%d: %s [%s] ---",
                     i, len(pairs), pair["alias"], pair["terrain"])
            results.append(_process_one_pair(pair))
    else:
        log.info("Step 2: Processing %d pairs with %d workers...",
                 len(pairs), max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_process_one_pair, pair): pair["alias"]
                for pair in pairs
            }
            for future in as_completed(futures):
                alias = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    log.exception("Worker crashed for %s: %s", alias, exc)
                    results.append({
                        "alias": alias,
                        "status": f"ERROR: {exc}",
                    })

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Step 3: Write summary report
    # ------------------------------------------------------------------
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "alias", "terrain", "scale", "dem_tif", "browse_tif", "aligned_tif",
        "slope_tif", "roughness_tif", "risk_tif", "hazard_tif", "validity_tif",
        "pixel_size_m", "nodata_fraction", "hazardous_fraction", "status",
    ]

    with open(REPORT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    ok_count = sum(1 for r in results if r.get("status") == "OK")
    fail_count = len(results) - ok_count
    log.info("")
    log.info("=" * 60)
    log.info("Stage 1 complete: %d OK, %d failed  (%.1fs, %d workers)",
             ok_count, fail_count, elapsed, max_workers)
    log.info("Report written to: %s", REPORT_CSV)
    log.info("=" * 60)

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1 — DEM Processing Pipeline (convert, align, label)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: 4). Use 1 for sequential."
    )
    args = parser.parse_args()
    main(max_workers=args.workers)
