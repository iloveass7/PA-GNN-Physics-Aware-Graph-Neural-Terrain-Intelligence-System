"""
mola_validation.py
------------------
Blueprint §8 — MOLA Proxy Validation Experiment (required for paper).

Validates that the Sobel slope proxy computed from HiRISE browse images
correlates with actual slope computed from MOLA MEGDR elevation data.

Procedure:
  1. Download MOLA MEGDR data for locations overlapping HiRISE training tiles.
  2. Compute actual slope from MOLA elevation (finite differences at 460 m/pixel).
  3. Compute Sobel slope proxy from HiRISE browse image.
  4. Aggregate both to MOLA's 460 m/pixel footprint by block-averaging.
  5. Compute Pearson r across all paired sample points.
  6. Generate a scatter plot of MOLA slope vs Sobel proxy.
  7. Report: if r > 0.60 → proxy validated; if r < 0.50 → flag as limitation.

MOLA MEGDR data source:
  https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/
  File pattern: MEGT*.img (16-bit signed integer, metres × 1000 → divide by 1000.0)
  Resolution: 128 pixels/degree = ~463 m/pixel at equator

Run from pa-gnn/ directory:
    python scripts/mola_validation.py

Output:
    results/mola_validation/mola_validation_scatter.png
    results/mola_validation/mola_validation_report.txt
"""

import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mola_validation")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MOLA_DIR      = PROJECT_ROOT / "data" / "raw" / "mola"   # user places MOLA files here
ALIGNED_DIR   = PROJECT_ROOT / "data" / "processed" / "aligned"
LABELS_DIR    = PROJECT_ROOT / "data" / "processed" / "labels"
OUTPUT_DIR    = PROJECT_ROOT / "results" / "mola_validation"

# MOLA pixel resolution in metres at equator
MOLA_PIXEL_M: float = 463.0


# ---------------------------------------------------------------------------
# MOLA helpers
# ---------------------------------------------------------------------------

def _load_mola_tile(mola_path: Path) -> np.ndarray:
    """Load a MOLA MEGDR .img file as float32 elevation in metres.

    MEGDR stores int16 values where each unit = 1/1000 metre (i.e., millimetres).
    Divide by 1000 to get metres.
    """
    import rasterio
    try:
        with rasterio.open(str(mola_path)) as ds:
            data = ds.read(1).astype(np.float32) / 1000.0
            log.info("MOLA tile loaded: %s, shape=%s", mola_path.name, data.shape)
            return data
    except Exception as e:
        raise RuntimeError(
            f"Failed to read MOLA file {mola_path}: {e}\n"
            f"Ensure the file is a valid MOLA MEGDR .img file."
        ) from e


def _compute_slope_from_elevation(
    elevation: np.ndarray,
    pixel_size_m: float,
) -> np.ndarray:
    """Compute slope in degrees from an elevation array using central differences."""
    gx = np.gradient(elevation.astype(np.float64), pixel_size_m, axis=1)
    gy = np.gradient(elevation.astype(np.float64), pixel_size_m, axis=0)
    slope = np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))
    return slope.astype(np.float32)


def _compute_sobel_proxy(image: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude as slope proxy, normalised to [0, 1]."""
    from scipy.ndimage import sobel
    img = image.astype(np.float64)
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    max_val = magnitude.max()
    if max_val < 1e-8:
        return np.zeros_like(magnitude, dtype=np.float32)
    return (magnitude / max_val).astype(np.float32)


def _block_average(array: np.ndarray, block_size: int) -> np.ndarray:
    """Downsample by block averaging (non-overlapping blocks)."""
    h, w = array.shape
    h_blocks = h // block_size
    w_blocks = w // block_size
    cropped = array[:h_blocks * block_size, :w_blocks * block_size]
    return cropped.reshape(h_blocks, block_size, w_blocks, block_size).mean(axis=(1, 3))


# ---------------------------------------------------------------------------
# Main validation logic
# ---------------------------------------------------------------------------

def run_validation_for_pair(
    alias: str,
    aligned_browse_tif: Path,
    hirise_slope_tif: Path,
    mola_path: Path,
    hirise_pixel_m: float,
) -> dict | None:
    """Run the MOLA proxy validation for one HiRISE / MOLA pair.

    Returns a dict with:
        alias, n_samples, pearson_r, p_value, mola_slopes, sobel_proxies
    or None if the MOLA file is missing/incompatible.
    """
    import rasterio

    if not mola_path.exists():
        log.warning("MOLA file missing for [%s]: %s — skipping", alias, mola_path)
        return None

    # Load HiRISE browse image (aligned to DEM grid)
    if not aligned_browse_tif.exists():
        log.warning("Aligned browse missing for [%s] — skipping", alias)
        return None

    with rasterio.open(str(aligned_browse_tif)) as ds:
        image = ds.read(1).astype(np.float32)

    # Load HiRISE slope (computed in dem_processing.py)
    if not hirise_slope_tif.exists():
        log.warning("HiRISE slope tif missing for [%s] — skipping", alias)
        return None

    with rasterio.open(str(hirise_slope_tif)) as ds:
        hirise_slope = ds.read(1).astype(np.float32)

    # Load MOLA elevation and compute slope
    mola_elev = _load_mola_tile(mola_path)
    mola_slope = _compute_slope_from_elevation(mola_elev, MOLA_PIXEL_M)

    # Compute Sobel proxy on HiRISE image
    sobel_proxy = _compute_sobel_proxy(image)

    # Determine block size: how many HiRISE pixels fit in one MOLA pixel
    block_size = max(1, round(MOLA_PIXEL_M / hirise_pixel_m))
    log.info("[%s] Aggregating: block_size=%d pixels (%.1fm HiRISE → %.1fm MOLA)",
             alias, block_size, hirise_pixel_m, MOLA_PIXEL_M)

    # Aggregate Sobel proxy to MOLA resolution
    sobel_agg = _block_average(sobel_proxy, block_size)

    # Crop MOLA slope to match aggregated HiRISE extent
    min_rows = min(mola_slope.shape[0], sobel_agg.shape[0])
    min_cols = min(mola_slope.shape[1], sobel_agg.shape[1])
    mola_crop  = mola_slope[:min_rows, :min_cols].flatten()
    sobel_crop = sobel_agg[:min_rows, :min_cols].flatten()

    # Remove invalid points
    valid = np.isfinite(mola_crop) & np.isfinite(sobel_crop) & (mola_crop >= 0)
    if valid.sum() < 10:
        log.warning("[%s] Fewer than 10 valid sample pairs — skipping", alias)
        return None

    r, p = stats.pearsonr(mola_crop[valid], sobel_crop[valid])
    log.info("[%s] Pearson r = %.4f (p = %.4e, n = %d)", alias, r, p, valid.sum())

    return {
        "alias":        alias,
        "n_samples":    int(valid.sum()),
        "pearson_r":    float(r),
        "p_value":      float(p),
        "mola_slopes":  mola_crop[valid],
        "sobel_proxies": sobel_crop[valid],
    }


def run_mola_validation(pairs: list[dict]) -> None:
    """Run validation across all pairs that have a corresponding MOLA file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_mola   = []
    all_sobel  = []
    all_rs     = []
    results_log = []

    for pair in pairs:
        alias = pair["alias"]
        # Convention: MOLA file named after the DEM alias prefix
        mola_path = MOLA_DIR / f"{alias}_MOLA.img"

        # Try common MOLA filename patterns
        if not mola_path.exists():
            candidates = list(MOLA_DIR.glob("MEGT*.img")) + list(MOLA_DIR.glob("*.img"))
            if candidates:
                mola_path = candidates[0]  # use first available as a fallback
            else:
                log.warning("No MOLA files found in %s — cannot run validation", MOLA_DIR)
                _write_placeholder_report(OUTPUT_DIR)
                return

        aligned_tif = ALIGNED_DIR / (pair["ortho_tif"].stem + "_aligned.tif")
        slope_tif   = LABELS_DIR / f"{alias}_slope.tif"

        # Read pixel size from DEM tif
        import rasterio
        pixel_m = 1.0
        if pair["dem_tif"].exists():
            with rasterio.open(str(pair["dem_tif"])) as ds:
                pixel_m = abs(ds.transform.a)

        result = run_validation_for_pair(
            alias=alias,
            aligned_browse_tif=aligned_tif,
            hirise_slope_tif=slope_tif,
            mola_path=mola_path,
            hirise_pixel_m=pixel_m,
        )

        if result is None:
            continue

        all_mola.extend(result["mola_slopes"])
        all_sobel.extend(result["sobel_proxies"])
        all_rs.append(result["pearson_r"])
        results_log.append(
            f"{result['alias']:35s}  r={result['pearson_r']:.4f}  "
            f"n={result['n_samples']:6d}  p={result['p_value']:.2e}"
        )

    if not all_mola:
        log.error("No valid pair results — cannot produce scatter plot.")
        _write_placeholder_report(OUTPUT_DIR)
        return

    # Aggregate Pearson r across all pairs
    all_mola_arr  = np.array(all_mola)
    all_sobel_arr = np.array(all_sobel)
    overall_r, overall_p = stats.pearsonr(all_mola_arr, all_sobel_arr)
    mean_r = float(np.mean(all_rs))

    log.info("Overall Pearson r = %.4f (p = %.4e)", overall_r, overall_p)

    # --- Scatter plot ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(all_mola_arr, all_sobel_arr, alpha=0.1, s=2, color="#2196F3", rasterized=True)

    # Trend line
    m, b = np.polyfit(all_mola_arr, all_sobel_arr, 1)
    x_line = np.linspace(all_mola_arr.min(), all_mola_arr.max(), 200)
    ax.plot(x_line, m * x_line + b, "r-", linewidth=1.5, label=f"r = {overall_r:.3f}")

    ax.set_xlabel("MOLA Slope (degrees)", fontsize=12)
    ax.set_ylabel("Sobel Slope Proxy (normalised)", fontsize=12)
    ax.set_title("MOLA Slope vs HiRISE Sobel Proxy\n(Aggregated to MOLA 460 m/pixel footprint)",
                 fontsize=11)
    ax.legend(fontsize=11)

    # Interpretation annotation
    interp = "Proxy validated (r > 0.60)" if overall_r > 0.60 else \
             "Limitation (r < 0.50)" if overall_r < 0.50 else "Marginal (0.50 ≤ r ≤ 0.60)"
    ax.text(0.05, 0.95, interp, transform=ax.transAxes,
            fontsize=10, va="top", color="darkgreen" if overall_r > 0.60 else "red")

    plt.tight_layout()
    scatter_path = OUTPUT_DIR / "mola_validation_scatter.png"
    plt.savefig(str(scatter_path), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Scatter plot saved: %s", scatter_path)

    # --- Text report ---
    report_path = OUTPUT_DIR / "mola_validation_report.txt"
    with open(report_path, "w") as f:
        f.write("MOLA Proxy Validation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Pearson r : {overall_r:.4f}\n")
        f.write(f"p-value           : {overall_p:.4e}\n")
        f.write(f"N samples         : {len(all_mola_arr)}\n")
        f.write(f"Mean per-pair r   : {mean_r:.4f}\n\n")
        f.write(f"Interpretation    : {interp}\n\n")
        f.write("Per-pair results:\n")
        f.write("-" * 60 + "\n")
        for line in results_log:
            f.write(line + "\n")
        f.write("\nBlueprint thresholds:\n")
        f.write("  r > 0.60 → proxy validated\n")
        f.write("  r < 0.50 → report as limitation in paper\n")

    log.info("Report written: %s", report_path)
    log.info("Final result: Pearson r = %.4f → %s", overall_r, interp)


def _write_placeholder_report(output_dir: Path) -> None:
    """Write a placeholder report when MOLA data is not available."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "mola_validation_report.txt"
    with open(report, "w") as f:
        f.write("MOLA Proxy Validation — PENDING\n")
        f.write("=" * 60 + "\n\n")
        f.write("MOLA MEGDR data has not been downloaded yet.\n\n")
        f.write("To run this validation:\n")
        f.write("  1. Download MOLA MEGDR files from:\n")
        f.write("     https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/\n")
        f.write("  2. Place .img files in: data/raw/mola/\n")
        f.write("  3. Re-run: python scripts/mola_validation.py\n")
    log.info("Placeholder report written: %s", report)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data.dem_loader import get_dem_pairs

    VAULT_CSV  = PROJECT_ROOT / "data" / "raw" / "mars_terrain_vault.csv"
    DEM_DIR    = PROJECT_ROOT / "data" / "raw" / "dem"
    BROWSE_DIR = PROJECT_ROOT / "data" / "raw" / "hirise_browse"
    TIF_CACHE  = PROJECT_ROOT / "data" / "processed" / "tif_cache"

    log.info("Loading vault...")
    pairs = get_dem_pairs(VAULT_CSV, DEM_DIR, BROWSE_DIR, TIF_CACHE)

    log.info("Running MOLA proxy validation for %d pairs...", len(pairs))
    run_mola_validation(pairs)
