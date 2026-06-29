"""
dem_processing.py
-----------------
Stage 1: DEM Data Preparation and Label Generation.

Given a georeferenced DEM GeoTIFF (float32, metres above MOLA datum) and its
geometrically aligned HiRISE browse image GeoTIFF, this module:

  1. Reads the DEM and computes pixel-accurate slope (degrees) and roughness (metres)
  2. Generates a per-pixel risk score in [0.05, 0.95] for CNN training supervision
  3. Builds a binary hazard mask for evaluation (slope > 15° OR roughness > threshold)
  4. Writes all outputs to a processed output directory as GeoTIFFs

Outputs per DEM pair:
    <alias>_slope.tif      — slope in degrees, float32
    <alias>_roughness.tif  — roughness in metres, float32
    <alias>_risk.tif       — DEM-derived risk label [0.05, 0.95], float32
    <alias>_hazard.tif     — binary hazard mask {0, 1}, uint8
    <alias>_validity.tif   — valid pixel mask (1=valid, 0=NoData), uint8

Usage:
    from src.data.dem_processing import process_dem_pair

    result = process_dem_pair(
        alias="Craters_003125",
        dem_tif=Path("...dem/DTEEC_003125.tif"),
        browse_tif=Path("...browse/PSP_003125_aligned.tif"),
        output_dir=Path("data/processed/labels/"),
    )
    # result["risk_tif"]  -> Path to the generated risk label GeoTIFF
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import uniform_filter

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blueprint constants (Section 8 of pagnn_final_blueprint_v4.md)
# ---------------------------------------------------------------------------

# Risk score formula — fixed physical scales (blueprint §8, corrected)
SLOPE_LIMIT_DEG: float = 30.0          # rover slope traversal limit, degrees
ROUGHNESS_SCALE_M: float = 3.0         # rover-relevant obstacle roughness scale, metres

# Hazard threshold for evaluation binary mask
HAZARD_SLOPE_THRESH_DEG: float = 15.0  # degrees
HAZARD_ROUGHNESS_THRESH_M: float = 1.0 # metres — fixed physical scale (was 0.6 × tile_max)

# Label smoothing: clamp to [RISK_MIN, RISK_MAX] to avoid sigmoid saturation
RISK_MIN: float = 0.05
RISK_MAX: float = 0.95

# Roughness window size in pixels (7×7 sliding window)
ROUGHNESS_WINDOW: int = 7

# NoData sentinel
DEM_NODATA: float = -3.40282e+38
DEM_NODATA_THRESHOLD: float = -1e+30  # any value below this is treated as NoData

# GeoTIFF write options
_GTIFF_OPTIONS = dict(
    driver="GTiff",
    compress="LZW",
    tiled=True,
    blockxsize=512,
    blockysize=512,
)


# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------

def compute_slope(
    elevation: np.ndarray,
    valid_mask: np.ndarray,
    pixel_size_m: float,
) -> np.ndarray:
    """Compute slope in degrees from an elevation array.

    Uses finite differences (central differences) in both spatial directions:
        Gx = (z[i, j+1] - z[i, j-1]) / (2 * dx)
        Gy = (z[i+1, j] - z[i-1, j]) / (2 * dy)
        slope = arctan(sqrt(Gx² + Gy²))

    NoData pixels are excluded from gradient computation by filling with local
    mean before differencing, then masking the output.

    Parameters
    ----------
    elevation : np.ndarray (H, W), float32
        Elevation in metres.  NoData pixels have arbitrary large-negative values.
    valid_mask : np.ndarray (H, W), bool
        True where the pixel is valid (not NoData).
    pixel_size_m : float
        Pixel size in metres (DEM ground sampling distance).

    Returns
    -------
    slope_deg : np.ndarray (H, W), float32
        Slope in degrees.  Invalid pixels are set to 0.
    """
    elev = elevation.astype(np.float64)
    # Fill NoData with local mean to avoid gradient artefacts at NoData boundaries
    fill_value = np.nanmean(elev[valid_mask]) if valid_mask.any() else 0.0
    elev[~valid_mask] = fill_value

    # Central differences
    gx = np.gradient(elev, pixel_size_m, axis=1)  # east-west
    gy = np.gradient(elev, pixel_size_m, axis=0)  # north-south

    slope_rad = np.arctan(np.sqrt(gx ** 2 + gy ** 2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    slope_deg[~valid_mask] = 0.0

    log.debug("Slope: min=%.2f°, max=%.2f°, mean=%.2f°",
              slope_deg[valid_mask].min(),
              slope_deg[valid_mask].max(),
              slope_deg[valid_mask].mean())
    return slope_deg


def compute_roughness(
    elevation: np.ndarray,
    valid_mask: np.ndarray,
    window: int = ROUGHNESS_WINDOW,
) -> np.ndarray:
    """Compute local roughness as std-dev of elevation in a sliding window.

    Roughness = std(elevation) in a (window × window) neighbourhood.
    This is computed as sqrt(E[z²] - E[z]²) using uniform_filter for speed.

    Parameters
    ----------
    elevation : np.ndarray (H, W), float32
    valid_mask : np.ndarray (H, W), bool
    window : int
        Sliding window size in pixels (default: 7×7 from blueprint).

    Returns
    -------
    roughness_m : np.ndarray (H, W), float32
        Roughness in metres.  Invalid pixels are set to 0.
    """
    elev = elevation.astype(np.float64)
    fill_value = np.nanmean(elev[valid_mask]) if valid_mask.any() else 0.0
    elev[~valid_mask] = fill_value

    mean_z = uniform_filter(elev, size=window)
    mean_z2 = uniform_filter(elev ** 2, size=window)
    variance = np.maximum(mean_z2 - mean_z ** 2, 0.0)  # clamp floating-point negatives
    roughness = np.sqrt(variance).astype(np.float32)
    roughness[~valid_mask] = 0.0

    log.debug("Roughness: min=%.4fm, max=%.4fm, mean=%.4fm",
              roughness[valid_mask].min(),
              roughness[valid_mask].max(),
              roughness[valid_mask].mean())
    return roughness


def compute_risk_label(
    slope_deg: np.ndarray,
    roughness_m: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Generate the DEM-derived risk score used as CNN training supervision.

    Formula (blueprint §8, corrected):
        slope_norm     = clamp(slope_deg / 20.0, 0, 1)
        roughness_norm = clamp(roughness_m / 2.0, 0, 1)   # fixed physical scale, metres
        risk           = max(slope_norm, roughness_norm)  # either signal can flag hazard
        risk           = clamp(risk, 0.05, 0.95)

    Rationale: the previous weighted sum (0.6*slope + 0.4*roughness) capped the
    slope term at 0.6, placing genuinely steep terrain below the 0.7 hazard
    threshold, while per-tile-max roughness normalisation was dominated by
    NoData-boundary outliers (max-combine of two independently, *physically*
    normalised signals fixes both: either a steep OR a rough pixel can reach high
    risk, and fixed scales make labels comparable across tiles and terrain types).

    Parameters
    ----------
    slope_deg : np.ndarray (H, W), float32
    roughness_m : np.ndarray (H, W), float32
    valid_mask : np.ndarray (H, W), bool

    Returns
    -------
    risk : np.ndarray (H, W), float32 — values in [RISK_MIN, RISK_MAX].
          Invalid pixels are set to NaN so they can be masked in training.
    """
    slope_norm = np.clip(slope_deg / SLOPE_LIMIT_DEG, 0.0, 1.0)
    roughness_norm = np.clip(roughness_m / ROUGHNESS_SCALE_M, 0.0, 1.0)

    risk = np.maximum(slope_norm, roughness_norm).astype(np.float32)

    # Label smoothing
    risk = np.clip(risk, RISK_MIN, RISK_MAX)

    # Mark invalid pixels as NaN (excluded from loss computation in training)
    risk[~valid_mask] = np.nan

    valid_risk = risk[valid_mask]
    log.debug("Risk label: min=%.3f, max=%.3f, mean=%.3f, hazardous(>0.7)=%.1f%%",
              float(np.nanmin(risk)),
              float(np.nanmax(risk)),
              float(np.nanmean(risk)),
              100.0 * (valid_risk > 0.7).mean())
    return risk


def compute_hazard_mask(
    slope_deg: np.ndarray,
    roughness_m: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Compute binary hazard mask for evaluation (not used as training label).

    A pixel is hazardous if:
        slope > HAZARD_SLOPE_THRESH_DEG (15°)  OR
        roughness > HAZARD_ROUGHNESS_THRESH_M (1.0 m)

    Uses fixed physical thresholds (not per-tile-max) so the binary mask is
    consistent with the continuous risk label and comparable across tiles.

    Parameters
    ----------
    slope_deg, roughness_m : np.ndarray (H, W), float32
    valid_mask : np.ndarray (H, W), bool

    Returns
    -------
    hazard : np.ndarray (H, W), uint8 — {0, 1}.  Invalid pixels = 0.
    """
    hazard = (
        (slope_deg > HAZARD_SLOPE_THRESH_DEG) |
        (roughness_m > HAZARD_ROUGHNESS_THRESH_M)
    ).astype(np.uint8)

    hazard[~valid_mask] = 0

    log.debug("Hazard mask: %.1f%% hazardous pixels",
              100.0 * hazard[valid_mask].mean())
    return hazard


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _write_single_band_tif(
    array: np.ndarray,
    reference_ds: rasterio.DatasetReader,
    out_path: Path,
    dtype: str,
    nodata: Optional[float] = None,
) -> Path:
    """Write a single-band array as a GeoTIFF, copying CRS/transform from reference."""
    profile = reference_ds.profile.copy()
    profile.update(
        **_GTIFF_OPTIONS,
        dtype=dtype,
        count=1,
        nodata=nodata,
    )
    # Remove JP2-specific keys that don't apply to GTiff
    for key in ["quality", "reversible", "numthreads"]:
        profile.pop(key, None)

    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)

    return out_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_dem_pair(
    alias: str,
    dem_tif: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict:
    """Run Stage 1 processing for one DEM pair.

    Reads the DEM GeoTIFF, computes slope, roughness, risk label, and hazard
    mask, and writes four output GeoTIFFs to output_dir.

    Parameters
    ----------
    alias : str
        Vault alias (e.g. ``"Craters_003125"``).  Used as filename prefix.
    dem_tif : Path
        Converted DEM GeoTIFF (float32 metres, MOLA-referenced).
    output_dir : Path
        Directory for output GeoTIFFs (created if it doesn't exist).
    overwrite : bool
        If True, re-compute even if outputs already exist.

    Returns
    -------
    dict with keys:
        alias, slope_tif, roughness_tif, risk_tif, hazard_tif, validity_tif,
        pixel_size_m, nodata_fraction, hazardous_fraction
    """
    dem_tif = Path(dem_tif)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output paths
    slope_tif = output_dir / f"{alias}_slope.tif"
    roughness_tif = output_dir / f"{alias}_roughness.tif"
    risk_tif = output_dir / f"{alias}_risk.tif"
    hazard_tif = output_dir / f"{alias}_hazard.tif"
    validity_tif = output_dir / f"{alias}_validity.tif"

    if (not overwrite and risk_tif.exists()):
        log.info("Stage 1 outputs already exist for %s — skipping (use overwrite=True)", alias)
        return {
            "alias": alias,
            "slope_tif": slope_tif,
            "roughness_tif": roughness_tif,
            "risk_tif": risk_tif,
            "hazard_tif": hazard_tif,
            "validity_tif": validity_tif,
        }

    log.info("=== Stage 1: Processing DEM pair [%s] ===", alias)

    with rasterio.open(str(dem_tif)) as ds:
        elevation = ds.read(1).astype(np.float32)
        transform = ds.transform
        crs = ds.crs
        nodata_val = ds.nodata

        # Pixel size in metres (use absolute values; transform.a can be negative)
        pixel_size_m = abs(transform.a)
        log.info("DEM: shape=%s, pixel_size=%.2fm, CRS=%s",
                 elevation.shape, pixel_size_m, crs)

        # Build valid pixel mask
        if nodata_val is not None:
            valid_mask = elevation != nodata_val
        else:
            valid_mask = elevation > DEM_NODATA_THRESHOLD

        nodata_fraction = 1.0 - valid_mask.mean()
        log.info("NoData fraction: %.1f%%", 100.0 * nodata_fraction)

        if nodata_fraction > 0.5:
            log.warning("More than 50%% of pixels are NoData for %s — "
                        "this DEM may be partially downloaded", alias)

        # --- Compute physics quantities ---
        slope_deg = compute_slope(elevation, valid_mask, pixel_size_m)
        roughness_m = compute_roughness(elevation, valid_mask)
        risk = compute_risk_label(slope_deg, roughness_m, valid_mask)
        hazard = compute_hazard_mask(slope_deg, roughness_m, valid_mask)
        validity = valid_mask.astype(np.uint8)

        # --- Write outputs ---
        _write_single_band_tif(slope_deg, ds, slope_tif, "float32", nodata=0.0)
        _write_single_band_tif(roughness_m, ds, roughness_tif, "float32", nodata=0.0)
        _write_single_band_tif(risk, ds, risk_tif, "float32", nodata=np.nan)
        _write_single_band_tif(hazard, ds, hazard_tif, "uint8", nodata=None)
        _write_single_band_tif(validity, ds, validity_tif, "uint8", nodata=None)

    hazardous_fraction = float(hazard[valid_mask].mean()) if valid_mask.any() else 0.0
    log.info("Stage 1 complete [%s]: hazardous=%.1f%%, nodata=%.1f%%",
             alias, 100.0 * hazardous_fraction, 100.0 * nodata_fraction)

    return {
        "alias": alias,
        "slope_tif": slope_tif,
        "roughness_tif": roughness_tif,
        "risk_tif": risk_tif,
        "hazard_tif": hazard_tif,
        "validity_tif": validity_tif,
        "pixel_size_m": pixel_size_m,
        "nodata_fraction": nodata_fraction,
        "hazardous_fraction": hazardous_fraction,
    }
