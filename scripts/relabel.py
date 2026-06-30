"""
relabel.py — Fast risk/hazard relabel from cached slope/roughness tifs.

Reads existing data/processed/labels/<alias>_slope.tif and _roughness.tif
(and _validity.tif), recomputes ONLY risk and hazard with the corrected
formula, and overwrites <alias>_risk.tif and <alias>_hazard.tif.

Does NOT touch slope, roughness, validity, or any upstream artifact.
Run from pa-gnn/ root:
    python relabel.py
"""
import sys
from pathlib import Path

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dem_processing import (
    compute_risk_label,
    compute_hazard_mask,
    _write_single_band_tif,
    _GTIFF_OPTIONS,
)

LABELS = PROJECT_ROOT / "data" / "processed" / "labels"


def load(path):
    with rasterio.open(str(path)) as ds:
        return ds.read(1)


def relabel_one(alias):
    slope_p   = LABELS / f"{alias}_slope.tif"
    rough_p   = LABELS / f"{alias}_roughness.tif"
    valid_p   = LABELS / f"{alias}_validity.tif"
    risk_p    = LABELS / f"{alias}_risk.tif"
    hazard_p  = LABELS / f"{alias}_hazard.tif"

    if not (slope_p.exists() and rough_p.exists() and valid_p.exists()):
        print(f"  SKIP {alias}: missing slope/roughness/validity")
        return None

    slope = load(slope_p).astype(np.float32)
    rough = load(rough_p).astype(np.float32)
    valid = load(valid_p).astype(bool)

    risk   = compute_risk_label(slope, rough, valid)
    hazard = compute_hazard_mask(slope, rough, valid)

    # Reference dataset for CRS/transform (use the slope tif — same grid)
    with rasterio.open(str(slope_p)) as ref:
        _write_single_band_tif(risk,   ref, risk_p,   "float32", nodata=np.nan)
        _write_single_band_tif(hazard, ref, hazard_p, "uint8",   nodata=None)

    haz_frac = float(hazard[valid].mean()) if valid.any() else 0.0
    vr = risk[valid]
    print(f"  {alias:28s} hazard={haz_frac*100:5.1f}%  "
          f"risk>0.7={100.0*(vr>0.7).mean():5.1f}%  "
          f"risk_p50={np.nanpercentile(risk,50):.3f}")
    return haz_frac


def main():
    # Discover aliases from existing slope tifs
    aliases = sorted(p.name[:-len("_slope.tif")]
                     for p in LABELS.glob("*_slope.tif"))
    if not aliases:
        print(f"No *_slope.tif found in {LABELS}. Run Stage 1 first.")
        sys.exit(1)

    print(f"Relabeling {len(aliases)} DEMs (risk + hazard only)...\n")
    ok = 0
    for a in aliases:
        if relabel_one(a) is not None:
            ok += 1
    print(f"\nDone: {ok}/{len(aliases)} relabeled. "
          f"slope/roughness/validity untouched.")


if __name__ == "__main__":
    main()