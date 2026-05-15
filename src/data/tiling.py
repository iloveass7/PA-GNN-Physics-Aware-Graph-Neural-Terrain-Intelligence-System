"""
tiling.py
---------
Stage 1 — Tiling Step.

Cuts an aligned (image, risk_label, hazard_mask, validity_mask) GeoTIFF quad
into 512×512 tiles with a 256-pixel stride (50% overlap).

Blueprint rules (§8):
  - Tile size: 512×512 pixels
  - Stride: 256 pixels (50% overlap)
  - REJECT tile if: > 10% of DEM pixels are NoData
  - REJECT tile if: > 30% of image pixels are near-saturated
    (near-saturated = within 5% of the pixel bit-depth max)
  - Log every rejection with reason

Outputs per tile:
  <alias>_r<row>_c<col>_image.npy   — float32 image, normalised to [0, 1], shape (512, 512)
  <alias>_r<row>_c<col>_risk.npy    — float32 risk label [0.05, 0.95], shape (512, 512)
  <alias>_r<row>_c<col>_hazard.npy  — uint8 hazard mask {0, 1}, shape (512, 512)
  <alias>_r<row>_c<col>_valid.npy   — uint8 validity mask {0, 1}, shape (512, 512)

Returns a list of tile records (dicts) for split assignment.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TILE_SIZE: int = 512
STRIDE: int = 256

# Rejection thresholds (blueprint §8)
MAX_NODATA_FRACTION: float = 0.10        # reject if > 10% DEM NoData
MAX_SATURATION_FRACTION: float = 0.30    # reject if > 30% image pixels near-saturated
SATURATION_MARGIN: float = 0.05          # "within 5% of bit-depth max" → top 5% of [0,1]

# Image value above this is considered near-saturated
_SAT_THRESHOLD: float = 1.0 - SATURATION_MARGIN   # = 0.95


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pad_to_multiple(array: np.ndarray, tile: int, stride: int) -> np.ndarray:
    """Reflect-pad array so its H and W are large enough for at least one tile
    and the sliding window covers the full image.

    Without padding, tiles near the right/bottom edge would be cropped.
    Reflect padding is used to avoid introducing artificial edge artefacts.
    """
    h, w = array.shape[:2]

    # Minimum size to guarantee complete tile coverage
    min_h = tile + (max(0, h - tile) + stride - 1) // stride * stride
    min_w = tile + (max(0, w - tile) + stride - 1) // stride * stride

    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)

    if array.ndim == 2:
        return np.pad(array, ((0, pad_h), (0, pad_w)), mode="reflect")
    else:
        return np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")


def _normalise_image(image_raw: np.ndarray) -> np.ndarray:
    """Normalise a raw uint16 (or uint8) image tile to float32 [0, 1].

    HiRISE browse images are typically uint16 with values spread across the
    full 0–65535 range.  Per-tile normalisation is used (not global) to
    handle varying illumination across DEM locations.
    """
    img = image_raw.astype(np.float32)
    tile_min = img.min()
    tile_max = img.max()
    if tile_max - tile_min < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return (img - tile_min) / (tile_max - tile_min + 1e-8)


def _is_saturated(image_norm: np.ndarray) -> bool:
    """Return True if more than MAX_SATURATION_FRACTION of pixels are
    near-saturated (≥ _SAT_THRESHOLD in the normalised [0,1] range)."""
    sat_frac = float((image_norm >= _SAT_THRESHOLD).mean())
    return sat_frac > MAX_SATURATION_FRACTION


def _is_nodata_heavy(validity: np.ndarray) -> bool:
    """Return True if more than MAX_NODATA_FRACTION of DEM pixels are NoData."""
    nodata_frac = 1.0 - float(validity.mean())
    return nodata_frac > MAX_NODATA_FRACTION


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tile_dem_pair(
    alias: str,
    aligned_browse_tif: Path,
    risk_tif: Path,
    hazard_tif: Path,
    validity_tif: Path,
    output_dir: Path,
    tile_size: int = TILE_SIZE,
    stride: int = STRIDE,
    overwrite: bool = False,
) -> list[dict]:
    """Slice one DEM pair into 512×512 tiles and save as .npy arrays.

    Parameters
    ----------
    alias : str
        Vault alias (e.g. ``"Craters_003125"``). Used as filename prefix.
    aligned_browse_tif : Path
        Browse image GeoTIFF aligned to DEM grid.
    risk_tif : Path
        Risk label GeoTIFF from ``dem_processing.process_dem_pair()``.
    hazard_tif : Path
        Binary hazard mask GeoTIFF.
    validity_tif : Path
        NoData validity mask GeoTIFF.
    output_dir : Path
        Directory where .npy tile files are saved (created if needed).
    tile_size : int
        Tile width and height in pixels (default: 512).
    stride : int
        Sliding window stride in pixels (default: 256 → 50% overlap).
    overwrite : bool
        If True, re-tile even if output files already exist.

    Returns
    -------
    List of dicts, one per **accepted** tile:
        alias, row, col, image_npy, risk_npy, hazard_npy, valid_npy,
        nodata_frac, sat_frac, hazardous_frac
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Tiling [%s] → %s", alias, output_dir)

    # --- Load all four rasters ---
    with rasterio.open(str(aligned_browse_tif)) as ds_img:
        image_raw = ds_img.read(1)      # (H, W) uint16

    with rasterio.open(str(risk_tif)) as ds_risk:
        risk_full = ds_risk.read(1)     # (H, W) float32, NaN at NoData

    with rasterio.open(str(hazard_tif)) as ds_haz:
        hazard_full = ds_haz.read(1)    # (H, W) uint8

    with rasterio.open(str(validity_tif)) as ds_val:
        validity_full = ds_val.read(1)  # (H, W) uint8

    # Verify spatial consistency
    h, w = image_raw.shape
    if risk_full.shape != (h, w):
        raise ValueError(
            f"[{alias}] Shape mismatch: image={image_raw.shape} "
            f"risk={risk_full.shape}. Was gdalwarp alignment run?"
        )

    # Pad to ensure complete tile coverage
    image_pad   = _pad_to_multiple(image_raw,     tile_size, stride)
    risk_pad    = _pad_to_multiple(risk_full,      tile_size, stride)
    hazard_pad  = _pad_to_multiple(hazard_full,   tile_size, stride)
    validity_pad = _pad_to_multiple(validity_full, tile_size, stride)

    H, W = image_pad.shape
    rows = range(0, H - tile_size + 1, stride)
    cols = range(0, W - tile_size + 1, stride)
    total = len(rows) * len(cols)
    accepted = []
    rejected_nodata = 0
    rejected_sat = 0

    for row in rows:
        for col in cols:
            r0, r1 = row, row + tile_size
            c0, c1 = col, col + tile_size

            img_tile  = image_pad[r0:r1, c0:c1]
            risk_tile = risk_pad[r0:r1, c0:c1]
            haz_tile  = hazard_pad[r0:r1, c0:c1]
            val_tile  = validity_pad[r0:r1, c0:c1]

            # --- Rejection checks ---
            if _is_nodata_heavy(val_tile):
                nodata_frac = 1.0 - float(val_tile.mean())
                log.debug("REJECT [%s] r%d c%d — NoData %.1f%%",
                          alias, row, col, 100.0 * nodata_frac)
                rejected_nodata += 1
                continue

            img_norm = _normalise_image(img_tile)

            if _is_saturated(img_norm):
                sat_frac = float((img_norm >= _SAT_THRESHOLD).mean())
                log.debug("REJECT [%s] r%d c%d — Saturated %.1f%%",
                          alias, row, col, 100.0 * sat_frac)
                rejected_sat += 1
                continue

            # --- Accept tile ---
            prefix = f"{alias}_r{row:05d}_c{col:05d}"
            image_npy  = output_dir / f"{prefix}_image.npy"
            risk_npy   = output_dir / f"{prefix}_risk.npy"
            hazard_npy = output_dir / f"{prefix}_hazard.npy"
            valid_npy  = output_dir / f"{prefix}_valid.npy"

            if overwrite or not image_npy.exists():
                np.save(str(image_npy),  img_norm.astype(np.float32))
                # Replace NaN risk at NoData pixels with 0.05 (min label)
                risk_clean = np.where(np.isnan(risk_tile), 0.05, risk_tile).astype(np.float32)
                np.save(str(risk_npy),   risk_clean)
                np.save(str(hazard_npy), haz_tile.astype(np.uint8))
                np.save(str(valid_npy),  val_tile.astype(np.uint8))

            nodata_frac    = 1.0 - float(val_tile.mean())
            sat_frac       = float((img_norm >= _SAT_THRESHOLD).mean())
            hazardous_frac = float(haz_tile[val_tile == 1].mean()) if val_tile.any() else 0.0

            accepted.append({
                "alias":          alias,
                "row":            row,
                "col":            col,
                "image_npy":      str(image_npy),
                "risk_npy":       str(risk_npy),
                "hazard_npy":     str(hazard_npy),
                "valid_npy":      str(valid_npy),
                "nodata_frac":    round(nodata_frac, 4),
                "sat_frac":       round(sat_frac, 4),
                "hazardous_frac": round(hazardous_frac, 4),
            })

    log.info(
        "[%s] Tiles: %d total → %d accepted, %d rejected (NoData), %d rejected (Sat)",
        alias, total, len(accepted), rejected_nodata, rejected_sat,
    )
    return accepted
