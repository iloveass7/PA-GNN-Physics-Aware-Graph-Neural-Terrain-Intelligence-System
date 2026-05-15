"""
dem_loader.py
-------------
Loads HiRISE stereo DEMs from PDS .IMG format using the mars_terrain_vault.csv index.
Converts to Cloud-Optimised GeoTIFF on first access and caches the result so subsequent
loads are fast.  All downstream pipeline code should call `load_dem()` — never read
.IMG files directly.

Usage:
    from src.data.dem_loader import load_dem, get_dem_pairs

    # Load a single DEM as a rasterio DatasetReader
    with load_dem("DTEEC_003125_1665_003191_1665_A01.IMG") as ds:
        elevation = ds.read(1)          # float32 metres
        transform = ds.transform
        crs = ds.crs

    # Iterate all vault pairs
    for pair in get_dem_pairs(vault_csv, dem_dir, browse_dir):
        alias      = pair["alias"]
        dem_path   = pair["dem_tif"]    # converted .tif path
        ortho_path = pair["ortho_tif"]  # converted .tif path (from hirise_loader)
        terrain    = pair["terrain"]
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# GDAL PDS driver name for .IMG files
_PDS_DRIVER = "PDS"

# GeoTIFF creation options — tiled + LZW compressed for fast windowed reads
_GTIFF_OPTIONS = {
    "driver": "GTiff",
    "compress": "LZW",
    "tiled": True,
    "blockxsize": 512,
    "blockysize": 512,
    "bigtiff": "IF_SAFER",  # auto-switch to BigTIFF if >4 GB
}

# NoData sentinel used by HiRISE DEMs
HIRISE_DEM_NODATA = -3.40282e+38


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _img_to_tif(img_path: Path, tif_path: Path) -> Path:
    """Convert a PDS .IMG DEM to a tiled, compressed GeoTIFF.

    Parameters
    ----------
    img_path : Path  — source .IMG file
    tif_path : Path  — destination .tif file (will be created)

    Returns
    -------
    Path to the created GeoTIFF.
    """
    tif_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Converting %s → %s", img_path.name, tif_path.name)

    with rasterio.open(str(img_path)) as src:
        profile = src.profile.copy()
        profile.update(
            **_GTIFF_OPTIONS,
            dtype="float32",
            count=1,
            nodata=HIRISE_DEM_NODATA,
        )

        data = src.read(1).astype("float32")

        # Replace any alternative NoData values with the canonical one
        if src.nodata is not None and src.nodata != HIRISE_DEM_NODATA:
            data[data == src.nodata] = HIRISE_DEM_NODATA

        with rasterio.open(str(tif_path), "w", **profile) as dst:
            dst.write(data, 1)

    log.info("Conversion complete: %s (%.1f MB)", tif_path.name,
             tif_path.stat().st_size / 1e6)
    return tif_path


def _resolve_tif_path(raw_path: Path, tif_dir: Path) -> Path:
    """Return the expected .tif path for a given raw file."""
    return tif_dir / (raw_path.stem + ".tif")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dem(
    dem_filename: str,
    dem_dir: Path,
    tif_cache_dir: Path,
    force_reconvert: bool = False,
) -> rasterio.DatasetReader:
    """Open a HiRISE DEM, converting from .IMG to GeoTIFF if needed.

    The converted GeoTIFF is cached in *tif_cache_dir* so conversion only
    runs once per file.  The returned dataset is the caller's responsibility
    to close (use as a context manager).

    Parameters
    ----------
    dem_filename : str
        Bare filename as it appears in the vault CSV ``dtm_path`` column,
        e.g. ``"DTEEC_003125_1665_003191_1665_A01.IMG"``.
    dem_dir : Path
        Directory containing the raw .IMG files.
    tif_cache_dir : Path
        Directory where converted .tif files are stored (created automatically).
    force_reconvert : bool
        If True, re-run conversion even if the .tif already exists.

    Returns
    -------
    rasterio.DatasetReader  (open — caller must close or use ``with`` block)
    """
    raw_path = Path(dem_dir) / dem_filename
    tif_path = _resolve_tif_path(raw_path, Path(tif_cache_dir))

    if not raw_path.exists():
        raise FileNotFoundError(
            f"DEM source file not found: {raw_path}\n"
            f"Check that dem_dir='{dem_dir}' is correct and the file is present."
        )

    if force_reconvert or not tif_path.exists():
        _img_to_tif(raw_path, tif_path)
    else:
        log.debug("Using cached GeoTIFF: %s", tif_path.name)

    return rasterio.open(str(tif_path))


def get_dem_pairs(
    vault_csv: Path,
    dem_dir: Path,
    browse_dir: Path,
    tif_cache_dir: Path,
    terrain_filter: list[str] | None = None,
    status_filter: str = "Verified",
) -> list[dict]:
    """Parse the mars_terrain_vault.csv and return all DEM+ortho pairs.

    Parameters
    ----------
    vault_csv : Path
        Path to ``mars_terrain_vault.csv``.
    dem_dir : Path
        Directory containing the raw .IMG DEM files.
    browse_dir : Path
        Directory containing the raw .JP2 ortho/browse files.
    tif_cache_dir : Path
        Root directory for cached GeoTIFFs. Subdirs ``dem/`` and ``browse/``
        are created automatically.
    terrain_filter : list[str] | None
        If given, only return pairs whose ``terrain`` column matches one of
        the listed strings (case-insensitive).  E.g. ``["Craters", "Volcanic"]``.
    status_filter : str
        Only return pairs whose ``status`` column matches this value.
        Default: ``"Verified"``.

    Returns
    -------
    List of dicts with keys:
        alias, dem_id, terrain, scale,
        dem_img   (Path to raw .IMG),
        ortho_jp2 (Path to raw .JP2),
        dem_tif   (Path to cached or converted DEM GeoTIFF),
        ortho_tif (Path to cached or converted browse GeoTIFF),
        status
    """
    vault_csv = Path(vault_csv)
    dem_dir = Path(dem_dir)
    browse_dir = Path(browse_dir)
    tif_cache_dir = Path(tif_cache_dir)

    dem_tif_dir = tif_cache_dir / "dem"
    browse_tif_dir = tif_cache_dir / "browse"
    dem_tif_dir.mkdir(parents=True, exist_ok=True)
    browse_tif_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(vault_csv)

    # Apply filters
    if status_filter:
        df = df[df["status"].str.strip() == status_filter]
    if terrain_filter:
        tf_lower = [t.lower() for t in terrain_filter]
        df = df[df["terrain"].str.lower().isin(tf_lower)]

    pairs = []
    for _, row in df.iterrows():
        dem_img = dem_dir / row["dtm_path"].strip()
        ortho_jp2 = browse_dir / row["ortho_path"].strip()

        dem_tif = _resolve_tif_path(dem_img, dem_tif_dir)
        ortho_tif = _resolve_tif_path(ortho_jp2, browse_tif_dir)

        pairs.append(
            {
                "alias": row["alias"].strip(),
                "dem_id": row["dem_id"].strip(),
                "terrain": row["terrain"].strip(),
                "scale": float(row["scale"]),
                "dem_img": dem_img,
                "ortho_jp2": ortho_jp2,
                "dem_tif": dem_tif,
                "ortho_tif": ortho_tif,
                "status": row["status"].strip(),
            }
        )

    log.info("Vault loaded: %d pairs (filter: terrain=%s, status=%s)",
             len(pairs), terrain_filter, status_filter)
    return pairs


def validate_vault_files(
    vault_csv: Path,
    dem_dir: Path,
    browse_dir: Path,
) -> dict:
    """Check that every file listed in the vault CSV exists on disk.

    Returns a dict with keys:
        "ok"      : list of aliases where both files exist
        "missing" : list of dicts with alias + which file(s) are missing
    """
    df = pd.read_csv(vault_csv)
    ok = []
    missing = []

    for _, row in df.iterrows():
        alias = row["alias"].strip()
        dem_path = Path(dem_dir) / row["dtm_path"].strip()
        jp2_path = Path(browse_dir) / row["ortho_path"].strip()

        dem_ok = dem_path.exists()
        jp2_ok = jp2_path.exists()

        if dem_ok and jp2_ok:
            ok.append(alias)
        else:
            missing.append({
                "alias": alias,
                "dem_missing": not dem_ok,
                "jp2_missing": not jp2_ok,
                "dem_path": str(dem_path),
                "jp2_path": str(jp2_path),
            })

    return {"ok": ok, "missing": missing}
