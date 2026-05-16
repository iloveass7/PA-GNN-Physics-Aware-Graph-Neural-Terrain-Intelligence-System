"""
hirise_loader.py
----------------
Converts HiRISE browse ortho .JP2 images to GeoTIFF format and aligns them to
their paired DEM GeoTIFF using GDAL's gdalwarp.

PERF-05 Note:
    gdalwarp is CPU-only and dominates per-pair processing time (~60-80% of
    wall clock per DEM pair).  There is no GPU warp path in GDAL.  The correct
    mitigation is to parallelise the OUTER per-pair loop (PERF-01 in
    process_dems.py) so that multiple gdalwarp subprocesses overlap.  Do NOT
    attempt to call gdalwarp with multithreading flags from within a child
    process — let the OS scheduler handle CPU core allocation across workers.

Usage:
    from src.data.hirise_loader import load_browse, align_browse_to_dem

    # Simple load (JP2 → GeoTIFF conversion + cache)
    with load_browse("PSP_003125_1665_RED_A_01_ORTHO.JP2", browse_dir, tif_cache) as ds:
        image = ds.read(1)              # uint8 or uint16 grayscale
        transform = ds.transform

    # Alignment: reproject browse to match DEM pixel grid exactly
    aligned_tif = align_browse_to_dem(dem_tif_path, browse_tif_path, output_dir)
"""

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

log = logging.getLogger(__name__)

# GeoTIFF creation options
_GTIFF_OPTIONS = {
    "driver": "GTiff",
    "compress": "LZW",
    "tiled": True,
    "blockxsize": 512,
    "blockysize": 512,
    "bigtiff": "IF_SAFER",
}

# HiRISE ortho images are 16-bit unsigned
_BROWSE_DTYPE = "uint16"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _jp2_to_tif(jp2_path: Path, tif_path: Path) -> Path:
    """Convert a HiRISE .JP2 ortho image to a tiled, compressed GeoTIFF.

    JP2 files can be very large (~350 MB). The function reads in blocks to
    avoid loading the entire image into RAM at once.

    Parameters
    ----------
    jp2_path : Path  — source .JP2 file
    tif_path : Path  — destination .tif file (created, parent dirs created)

    Returns
    -------
    Path to the created GeoTIFF.
    """
    tif_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Converting %s → %s", jp2_path.name, tif_path.name)

    with rasterio.open(str(jp2_path)) as src:
        profile = src.profile.copy()
        profile.update(
            **_GTIFF_OPTIONS,
            dtype=_BROWSE_DTYPE,
            count=1,
        )
        # JP2 files may report no nodata — that is fine for browse imagery
        profile.pop("nodata", None)

        with rasterio.open(str(tif_path), "w", **profile) as dst:
            # Copy block-by-block to limit RAM usage
            for ji, window in src.block_windows(1):
                data = src.read(1, window=window).astype(_BROWSE_DTYPE)
                dst.write(data, 1, window=window)

    log.info("Conversion complete: %s (%.1f MB)", tif_path.name,
             tif_path.stat().st_size / 1e6)
    return tif_path


def _resolve_tif_path(raw_path: Path, tif_dir: Path) -> Path:
    return tif_dir / (raw_path.stem + ".tif")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_browse(
    browse_filename: str,
    browse_dir: Path,
    tif_cache_dir: Path,
    force_reconvert: bool = False,
) -> rasterio.DatasetReader:
    """Open a HiRISE browse image, converting from .JP2 to GeoTIFF if needed.

    Parameters
    ----------
    browse_filename : str
        Bare filename as in the vault CSV ``ortho_path`` column,
        e.g. ``"PSP_003125_1665_RED_A_01_ORTHO.JP2"``.
    browse_dir : Path
        Directory containing the raw .JP2 files.
    tif_cache_dir : Path
        Directory for cached .tif files.
    force_reconvert : bool
        Re-convert even if the .tif already exists.

    Returns
    -------
    rasterio.DatasetReader  (open — caller must close or use ``with`` block)
    """
    raw_path = Path(browse_dir) / browse_filename
    tif_path = _resolve_tif_path(raw_path, Path(tif_cache_dir))

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Browse image not found: {raw_path}\n"
            f"Check browse_dir='{browse_dir}' and the vault CSV."
        )

    if force_reconvert or not tif_path.exists():
        _jp2_to_tif(raw_path, tif_path)
    else:
        log.debug("Using cached GeoTIFF: %s", tif_path.name)

    return rasterio.open(str(tif_path))


def align_browse_to_dem(
    dem_tif: Path,
    browse_tif: Path,
    output_dir: Path,
    resampling: Resampling = Resampling.bilinear,
    use_gdal_warp: bool = True,
) -> Path:
    """Reproject and align the browse image to exactly match the DEM pixel grid.

    After alignment:
    - Same CRS as the DEM
    - Same spatial extent as the DEM
    - Same pixel resolution and dimensions as the DEM
    - Every pixel in the browse image corresponds 1-to-1 with its DEM pixel

    This is mandatory before slope/roughness label generation (Stage 1).

    Parameters
    ----------
    dem_tif : Path
        Path to the converted DEM GeoTIFF (the reference grid).
    browse_tif : Path
        Path to the converted browse GeoTIFF (to be warped).
    output_dir : Path
        Directory where the aligned browse GeoTIFF is saved.
    resampling : Resampling
        Rasterio resampling method. Bilinear is appropriate for browse imagery.
    use_gdal_warp : bool
        If True (default), delegates to the ``gdalwarp`` CLI which handles edge
        cases (datum shifts, PDS CRS quirks) more robustly than rasterio's
        Python warp.  Falls back to rasterio if gdalwarp is not on PATH.

    Returns
    -------
    Path to the aligned browse GeoTIFF.
    """
    dem_tif = Path(dem_tif)
    browse_tif = Path(browse_tif)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aligned_name = browse_tif.stem + "_aligned.tif"
    aligned_path = output_dir / aligned_name

    if aligned_path.exists():
        log.debug("Aligned browse already exists: %s", aligned_path.name)
        return aligned_path

    if use_gdal_warp and _gdalwarp_available():
        aligned_path = _align_with_gdalwarp(
            dem_tif, browse_tif, aligned_path, resampling
        )
    else:
        log.warning("gdalwarp not found on PATH — falling back to rasterio warp")
        aligned_path = _align_with_rasterio(
            dem_tif, browse_tif, aligned_path, resampling
        )

    return aligned_path


def _gdalwarp_available() -> bool:
    """Check whether gdalwarp is accessible on the system PATH."""
    return shutil.which("gdalwarp") is not None


def _resampling_to_gdalwarp_str(resampling: Resampling) -> str:
    """Convert rasterio Resampling enum to gdalwarp -r string."""
    _map = {
        Resampling.nearest: "near",
        Resampling.bilinear: "bilinear",
        Resampling.cubic: "cubic",
        Resampling.lanczos: "lanczos",
    }
    return _map.get(resampling, "bilinear")


def _align_with_gdalwarp(
    dem_tif: Path,
    browse_tif: Path,
    aligned_path: Path,
    resampling: Resampling,
) -> Path:
    """Use gdalwarp to align browse to DEM grid (preferred method)."""
    log.info("Aligning %s → %s via gdalwarp", browse_tif.name, aligned_path.name)

    # Read DEM metadata for the target grid
    with rasterio.open(str(dem_tif)) as dem_ds:
        crs_wkt = dem_ds.crs.to_wkt()
        width = dem_ds.width
        height = dem_ds.height
        transform = dem_ds.transform
        xmin = transform.c
        ymax = transform.f
        xmax = xmin + transform.a * width
        ymin = ymax + transform.e * height

    cmd = [
        "gdalwarp",
        "-t_srs", crs_wkt,
        "-te", str(xmin), str(ymin), str(xmax), str(ymax),
        "-ts", str(width), str(height),
        "-r", _resampling_to_gdalwarp_str(resampling),
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        "-co", "BLOCKXSIZE=512",
        "-co", "BLOCKYSIZE=512",
        "-overwrite",
        str(browse_tif),
        str(aligned_path),
    ]

    log.debug("gdalwarp command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"gdalwarp failed for {browse_tif.name}:\n{result.stderr}"
        )

    log.info("Alignment complete: %s", aligned_path.name)
    return aligned_path


def _align_with_rasterio(
    dem_tif: Path,
    browse_tif: Path,
    aligned_path: Path,
    resampling: Resampling,
) -> Path:
    """Fallback: use rasterio.warp.reproject to align browse to DEM."""
    log.info("Aligning %s → %s via rasterio", browse_tif.name, aligned_path.name)

    with rasterio.open(str(dem_tif)) as dem_ds:
        dst_crs = dem_ds.crs
        dst_transform = dem_ds.transform
        dst_width = dem_ds.width
        dst_height = dem_ds.height

    with rasterio.open(str(browse_tif)) as src:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            crs=dst_crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            compress="LZW",
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )

        with rasterio.open(str(aligned_path), "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
            )

    log.info("Alignment complete: %s", aligned_path.name)
    return aligned_path
