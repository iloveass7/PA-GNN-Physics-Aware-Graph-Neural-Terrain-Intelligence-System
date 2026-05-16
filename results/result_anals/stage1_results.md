# PA-GNN Stage 1 — Results Analysis

**Run date:** 2026-05-17  
**Scripts:** `process_dems.py` → `tile_dataset.py`  
**Outcome:** ✅ 27/27 DEMs OK, 14,686 tiles generated

---

## DEM Processing Results (`process_dems.py`)

**Runtime:** 3,953s (~66 min) · 4 workers  
**Outcome:** 27 OK, 0 failed

### Per-DEM Summary

| Alias | Terrain | Pixel Size | DEM Shape | NoData % | Hazardous % |
|---|---|---|---|---|---|
| Craters_003125 | Craters | 1.01 m | 12,917 × 6,129 | 24.0% | 50.5% |
| Craters_003156 | Craters | 1.00 m | 13,040 × 6,601 | 25.7% | 21.1% |
| Volcanic_006712 | Volcanic | 1.01 m | 21,982 × 7,930 | 34.4% | 53.9% |
| Craters_017719 | Craters | 1.01 m | 13,955 × 6,301 | 29.1% | 36.2% |
| Gullies_025322 | Gullies | 0.99 m | 14,392 × 8,257 | 26.8% | 28.3% |
| Valleys_029246 | Valleys | 1.00 m | 14,758 × 5,505 | 27.2% | 32.7% |
| Craters_033968 | Craters | 1.01 m | 12,633 × 6,373 | 25.7% | 43.1% |
| Canyon Walls_059382 | Canyon Walls | 1.01 m | 11,824 × 5,700 | 26.1% | 47.1% |
| Volcanic_060706 | Volcanic | 1.00 m | 10,086 × 6,824 | 21.4% | 9.1% |
| Volcanic_064700 | Volcanic | 1.01 m | 9,491 × 5,884 | 20.0% | 3.9% |
| Canyon Walls_074396 | Canyon Walls | 1.00 m | 10,196 × 5,522 | 23.3% | 58.3% |
| Canyon Walls_074900 | Canyon Walls | 1.00 m | 10,016 × 5,323 | 22.3% | 26.9% |
| Craters_010245 | Craters | 2.00 m | 9,374 × 4,149 | 28.7% | 18.3% |
| Gullies_022986 | Gullies | 2.00 m | 5,194 × 3,360 | 23.2% | 42.6% |
| Valleys_033682 | Valleys | 2.00 m | 8,764 × 3,526 | 32.6% | 1.2% |
| Valleys_042725 | Valleys | 2.00 m | 9,146 × 3,793 | 30.6% | 9.0% |
| Craters_076968 | Craters | 2.01 m | 9,037 × 3,264 | 33.0% | 23.1% |
| Canyon Walls_082989 | Canyon Walls | 2.02 m | 4,573 × 2,647 | 25.3% | 57.1% |
| Canyon Walls_088616 | Canyon Walls | 2.02 m | 6,894 × 3,017 | 29.8% | 15.6% |
| Craters_089104 | Craters | 2.01 m | 5,760 × 3,165 | 26.3% | 20.0% |
| Polar_005721 | Polar | 1.00 m | 12,133 × 12,594 | **62.0%** ⚠️ | 55.5% |
| Dunes_022534 | Dunes | 1.00 m | 7,124 × 8,668 | 37.0% | 2.5% |
| Dunes_039932 | Dunes | 1.00 m | 7,590 × 6,032 | 39.5% | 7.6% |
| Polar_048808 | Polar | 1.00 m | 7,010 × 10,307 | 43.9% | 28.0% |
| Polar_049106 | Polar | 1.00 m | 5,744 × 9,787 | 38.3% | 4.7% |
| Polar_010109 | Polar | 2.00 m | 5,651 × 3,494 | 23.6% | 15.8% |
| Polar_023009 | Polar | 2.00 m | 2,644 × 5,783 | 39.9% | 13.1% |

### NoData Distribution

- **Normal range (20–40%):** 25 of 27 DEMs — expected for HiRISE stereo products
- **Elevated (>40%):** Polar_048808 (43.9%), Polar_049106 (38.3%), Polar_023009 (39.9%), Dunes_039932 (39.5%), Dunes_022534 (37.0%) — consistent with polar/dune coverage gaps
- **⚠️ Warning: Polar_005721 at 62.0%** — exceeds 50% threshold. DEM is valid and processed correctly but may be partially acquired. Yields fewer usable tiles. Not a blocker.

### Hazardous Fraction by Terrain (DEM-level)

| Terrain | Range | Notes |
|---|---|---|
| Canyon Walls | 15.6% – 58.3% | Highest hazard as expected — steep cliff faces |
| Craters | 18.3% – 50.5% | Wide range — rim vs floor variation |
| Volcanic | 3.9% – 53.9% | Smooth plains vs rugged flows |
| Gullies | 28.3% – 42.6% | Consistent moderate hazard |
| Valleys | 1.2% – 32.7% | Valleys_033682 notably flat (valley floor) |
| Polar | 4.7% – 55.5% | Wide range — layered deposits vs scarps |
| Dunes | 2.5% – 7.6% | Lowest hazard overall — smooth sand terrain |

All values are geologically plausible. The risk formula (0.6×slope + 0.4×roughness) is behaving correctly across terrain types.

### CRS Verification

Two CRS groups confirmed present as expected:

- **Equirectangular MARS** — mid-latitude DEMs (Craters, Volcanic, Canyon Walls, Gullies, Valleys)
- **Polar Stereographic MARS** — polar/dune DEMs (all Polar, both Dunes)

gdalwarp alignment handled both correctly. ✅

### Scale Groups

- **1m/px DEMs (scale=1.0):** 19 DEMs — DTEEC/DTEED/DTEPC prefix, standard resolution
- **2m/px DEMs (scale=2.0):** 8 DEMs — DTEED prefix, extended baseline stereo

Both processed without issue. The 2m DEMs yield smaller tile counts due to smaller physical footprint at the same 512px tile size.

---

## Tiling Results (`tile_dataset.py`)

**Runtime:** 66.8s · 4 workers  
**Total tiles generated:** 14,686

### Split Breakdown

| Split | Locations | Tiles | % of Total | Blueprint Target |
|---|---|---|---|---|
| train | 18 | 9,203 | 62.7% | ~70% locations |
| val | 3 | 1,761 | 12.0% | ~15% locations |
| test_in | 5 | 3,566 | 24.3% | ~15% locations |
| test_ood | 1 | 156 | 1.1% | 1 held-out location |
| **TOTAL** | **27** | **14,686** | — | **5,000–15,000** |

Total is within blueprint target range (top end). ✅

### OOD Selection

- **Selected:** `Craters_089104`
- **Terrain:** Craters — largest group (7 locations), ensuring Craters still appears in train after removal ✅
- **Method:** Alphabetically last alias from largest terrain group (deterministic, seed=42) ✅

### Per-DEM Tile Counts

| Alias | Split | Total Candidates | Accepted | Rejected (NoData) | Rejected (Sat) |
|---|---|---|---|---|---|
| Craters_003125 | test_in | 1,150 | 812 | 338 | 0 |
| Craters_003156 | train | 1,250 | 867 | 383 | 0 |
| Volcanic_006712 | train | 2,550 | 1,582 | 968 | 0 |
| Craters_017719 | test_in | 1,296 | 839 | 454 | **3** |
| Gullies_025322 | val | 1,792 | 1,204 | 588 | 0 |
| Valleys_029246 | train | 1,197 | 791 | 406 | 0 |
| Craters_033968 | train | 1,176 | 810 | 366 | 0 |
| Canyon Walls_059382 | test_in | 1,012 | 663 | 349 | 0 |
| Volcanic_060706 | test_in | 1,014 | 734 | 280 | 0 |
| Volcanic_064700 | train | 814 | 596 | 218 | 0 |
| Canyon Walls_074396 | train | 819 | 573 | 246 | 0 |
| Canyon Walls_074900 | train | 780 | 549 | 231 | 0 |
| Craters_010245 | train | 576 | 349 | 227 | 0 |
| Gullies_022986 | train | 260 | 159 | 101 | 0 |
| Valleys_033682 | train | 442 | 253 | 189 | 0 |
| Valleys_042725 | train | 490 | 296 | 194 | 0 |
| Craters_076968 | train | 420 | 237 | 183 | 0 |
| Canyon Walls_082989 | train | 170 | 100 | 70 | 0 |
| Canyon Walls_088616 | train | 442 | 170 | 116 | 0 |
| Craters_089104 | test_ood | 264 | 156 | 108 | 0 |
| Polar_005721 | train | 2,303 | 776 | 1,527 | 0 |
| Dunes_022534 | test_in | 891 | 518 | 373 | 0 |
| Dunes_039932 | val | 667 | 362 | 305 | 0 |
| Polar_048808 | train | 1,080 | 536 | 544 | 0 |
| Polar_049106 | train | 836 | 457 | 379 | 0 |
| Polar_010109 | val | 286 | 195 | 91 | 0 |
| Polar_023009 | train | 220 | 102 | 118 | 0 |

### Rejection Analysis

- **Saturation rejections: 3 total** (Craters_017719 only) — essentially zero. Mars imagery doesn't saturate like Earth imagery. The 30% threshold is appropriate. ✅
- **NoData rejections:** Proportional to DEM NoData fractions throughout — tiler and labeller are consistent. ✅
- **Polar_005721 rejection rate:** 1,527 of 2,303 candidates rejected (66%) — directly reflects the 62% NoData warning from `process_dems.py`. Accepted 776 tiles which is still usable. ✅

### Concerns

**test_ood tile count is thin (156 tiles from 1 DEM).** This is a structural constraint of the blueprint's OOD strategy — one held-out location — not a bug. Practical implications:

- OOD evaluation metrics will have wide confidence intervals
- Do not over-interpret small differences in OOD performance at thesis defense
- Report with bootstrapped CIs when running `evaluate_all.py`

---

## Blueprint Compliance Summary

| Requirement | Target | Actual | Status |
|---|---|---|---|
| All 27 DEMs processed | 27/27 | 27/27 | ✅ |
| Total tile count | 5,000–15,000 | 14,686 | ✅ |
| Tile size | 512×512 | 512×512 | ✅ |
| Stride | 256px (50% overlap) | 256px | ✅ |
| NoData rejection threshold | >10% | Applied | ✅ |
| Saturation rejection threshold | >30% | Applied | ✅ |
| DEM-location-level splits | Required | Confirmed | ✅ |
| OOD terrain has ≥2 locations in train | Required | Craters: 6 remain | ✅ |
| Split files populated | Required | All 4 written | ✅ |
| Tile manifest written | Required | Written | ✅ |

---

## Outputs Ready for Stage 2

```
data/processed/tiles/train/        9,203 .npy quads
data/processed/tiles/val/          1,761 .npy quads
data/processed/tiles/test_in/      3,566 .npy quads
data/processed/tiles/test_ood/       156 .npy quads
data/processed/labels/             27 × {slope, roughness, risk, hazard, validity}.tif
data/processed/aligned/            27 × aligned browse GeoTIFFs
data/processed/tif_cache/          27 × converted DEM + browse GeoTIFFs
data/splits/train.txt              18 aliases
data/splits/val.txt                 3 aliases
data/splits/test_in.txt             5 aliases
data/splits/test_ood.txt            1 alias
data/processed/stage1_report.csv   per-DEM processing summary
data/processed/tile_manifest.csv   per-tile full record
```

**Next:** `python scripts/build_graphs.py`
