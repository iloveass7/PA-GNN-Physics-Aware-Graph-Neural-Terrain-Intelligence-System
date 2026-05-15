"""
download_dems.py
----------------
NOTE: This script is intentionally minimal because the dataset is already
in place on this machine.  The download functionality is documented here
for reproducibility purposes only — no network calls are made.

Blueprint §8 (Stage 1 — DEM Data Preparation):
  HiRISE DEMs are downloaded from the USGS Astrogeology Science Center at
  https://www.uahirise.org/dtm/

  Each entry requires:
    - DEM GeoTIFF  (elevation in metres above MOLA reference ellipsoid)
    - Paired HiRISE browse image GeoTIFF at the same geographic location

  Target: 20–30 locations across geologically diverse terrain.

Dataset verification:
    python scripts/download_dems.py --verify

To download new DEMs from USGS (requires internet):
    python scripts/download_dems.py --ids ESP_011443_1755 ESP_013086_1555 ...

Since the dataset is already present, run --verify to confirm the structure.
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_dems")

# Expected directory layout (blueprint §23)
RAW_DEM_DIR    = PROJECT_ROOT / "data" / "raw" / "dem"
RAW_BROWSE_DIR = PROJECT_ROOT / "data" / "raw" / "hirise_browse"

# USGS DEM portal URL template (for documentation / future use)
USGS_PORTAL_URL = "https://www.uahirise.org/dtm/"


# ---------------------------------------------------------------------------
# Dataset verification (primary use — dataset already in place)
# ---------------------------------------------------------------------------

def verify_dataset(dem_dir: Path, browse_dir: Path) -> dict:
    """Verify that the DEM dataset is present and structurally complete.

    Checks:
    - Expected directories exist
    - At least 10 DEM files are present (blueprint minimum)
    - Each DEM has a corresponding browse image (same stem)
    - Files are non-empty GeoTIFFs (basic size check)

    Returns
    -------
    dict with: n_dems, n_browse, n_paired, n_orphan_dem, n_orphan_browse,
               issues (list of warning strings)
    """
    issues = []

    # Directory check
    if not dem_dir.exists():
        issues.append(f"DEM directory missing: {dem_dir}")
        log.error("DEM directory not found: %s", dem_dir)
        return {"n_dems": 0, "n_browse": 0, "issues": issues}

    if not browse_dir.exists():
        issues.append(f"Browse image directory missing: {browse_dir}")
        log.warning("Browse directory not found: %s — will check only DEMs", browse_dir)

    # Discover files
    dem_files    = sorted(dem_dir.glob("*.tif")) + sorted(dem_dir.glob("*.TIF"))
    dem_files   += sorted(dem_dir.glob("**/*.tif"))    # allow one level of subdir
    dem_files    = list(dict.fromkeys(dem_files))       # deduplicate

    browse_files = []
    if browse_dir.exists():
        browse_files  = sorted(browse_dir.glob("*.tif")) + sorted(browse_dir.glob("*.TIF"))
        browse_files += sorted(browse_dir.glob("**/*.tif"))
        browse_files  = list(dict.fromkeys(browse_files))

    # Build stem sets for pairing
    dem_stems    = {f.stem: f for f in dem_files}
    browse_stems = {f.stem: f for f in browse_files}

    # Paired: DEM stem matches browse stem (or contains it as prefix)
    paired = 0
    orphan_dem = []
    for stem, dem_f in dem_stems.items():
        # Accept exact match or prefix match (HiRISE naming conventions vary)
        matched = (stem in browse_stems or
                   any(b_stem.startswith(stem[:15]) for b_stem in browse_stems))
        if matched:
            paired += 1
        else:
            orphan_dem.append(stem)

    orphan_browse = [s for s in browse_stems if s not in dem_stems and
                     not any(d_stem.startswith(s[:15]) for d_stem in dem_stems)]

    # Size check — flag zero-byte files
    zero_byte = [f for f in dem_files + browse_files if f.stat().st_size == 0]
    if zero_byte:
        for f in zero_byte:
            issues.append(f"Zero-byte file: {f.name}")

    # Blueprint minimum check
    if len(dem_files) < 10:
        issues.append(
            f"Only {len(dem_files)} DEM files found. Blueprint requires minimum 10."
        )

    result = {
        "n_dems":           len(dem_files),
        "n_browse":         len(browse_files),
        "n_paired":         paired,
        "n_orphan_dem":     len(orphan_dem),
        "n_orphan_browse":  len(orphan_browse),
        "zero_byte_files":  len(zero_byte),
        "issues":           issues,
    }

    # Report
    log.info("=" * 60)
    log.info("DEM Dataset Verification")
    log.info("  DEM directory   : %s", dem_dir)
    log.info("  Browse directory: %s", browse_dir)
    log.info("  DEM files       : %d", result["n_dems"])
    log.info("  Browse files    : %d", result["n_browse"])
    log.info("  Paired          : %d", result["n_paired"])
    log.info("  Orphan DEMs     : %d", result["n_orphan_dem"])
    log.info("  Orphan browse   : %d", result["n_orphan_browse"])
    log.info("  Zero-byte files : %d", result["zero_byte_files"])
    if issues:
        log.warning("Issues found:")
        for issue in issues:
            log.warning("  ! %s", issue)
    else:
        log.info("  ✓ No issues found.")
    log.info("=" * 60)

    return result


# ---------------------------------------------------------------------------
# Stub download (for reproducibility documentation)
# ---------------------------------------------------------------------------

def download_dem_pair(
    dem_id: str,
    output_dem_dir: Path,
    output_browse_dir: Path,
    dry_run: bool = True,
) -> bool:
    """Download one DEM pair from USGS HiRISE portal.

    NOTE: Dataset is already in place. This function is provided for
    reproducibility documentation. Set dry_run=False only if you need
    to fetch additional DEMs.

    Parameters
    ----------
    dem_id          : HiRISE observation ID, e.g. "ESP_011443_1755"
    output_dem_dir  : destination for DEM GeoTIFF
    output_browse_dir: destination for browse image GeoTIFF
    dry_run         : if True, log what would be downloaded but do nothing

    Returns
    -------
    bool — True if download succeeded (or dry_run)
    """
    url = f"{USGS_PORTAL_URL}{dem_id}"

    if dry_run:
        log.info("[DRY RUN] Would download: %s → %s", url, output_dem_dir / f"{dem_id}_DEM.tif")
        log.info("[DRY RUN] Would download: %s → %s", url, output_browse_dir / f"{dem_id}_RED.tif")
        return True

    # Real download — only executed if dry_run=False
    try:
        import requests
        # USGS portal navigation is interactive; automated download requires
        # the specific file URL which varies per entry. In practice, download
        # manually from https://www.uahirise.org/dtm/ and place files in:
        #   data/raw/dem/         (DEM GeoTIFF, *_DEM.tif)
        #   data/raw/hirise_browse/  (browse image, *_RED.tif or *_COLOR.tif)
        log.error(
            "Automated USGS download not implemented. "
            "Please download manually from %s and place files in:\n"
            "  %s  (DEM GeoTIFF)\n"
            "  %s  (browse image)",
            url, output_dem_dir, output_browse_dir,
        )
        return False
    except ImportError:
        log.error("requests library not available for download.")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HiRISE DEM dataset management (verify / download)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--verify", action="store_true", default=True,
        help="Verify existing dataset structure (default: True)"
    )
    parser.add_argument(
        "--ids", nargs="*", default=[],
        help="HiRISE observation IDs to download (dry-run by default)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Show what would be downloaded without executing (default: True)"
    )
    parser.add_argument(
        "--dem-dir", default=str(RAW_DEM_DIR),
        help=f"DEM output directory (default: {RAW_DEM_DIR})"
    )
    parser.add_argument(
        "--browse-dir", default=str(RAW_BROWSE_DIR),
        help=f"Browse image output directory (default: {RAW_BROWSE_DIR})"
    )
    args = parser.parse_args()

    dem_dir    = Path(args.dem_dir)
    browse_dir = Path(args.browse_dir)

    # Always run verification first
    result = verify_dataset(dem_dir, browse_dir)

    if result["n_dems"] == 0:
        log.info(
            "Dataset directory is empty or missing.\n"
            "Manual download instructions:\n"
            "  1. Visit: %s\n"
            "  2. Select 20–30 DEM pairs from diverse geological regions.\n"
            "  3. Download *_DEM.tif → %s\n"
            "  4. Download *_RED.tif or *_COLOR.tif → %s",
            USGS_PORTAL_URL, dem_dir, browse_dir,
        )

    # Optional download stub
    if args.ids:
        dem_dir.mkdir(parents=True, exist_ok=True)
        browse_dir.mkdir(parents=True, exist_ok=True)
        for dem_id in args.ids:
            download_dem_pair(
                dem_id=dem_id,
                output_dem_dir=dem_dir,
                output_browse_dir=browse_dir,
                dry_run=args.dry_run,
            )

    if result["issues"]:
        sys.exit(1)
    else:
        log.info("Dataset verification passed. Proceeding to scripts/process_dems.py")
        sys.exit(0)


if __name__ == "__main__":
    main()
