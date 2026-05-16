# PA-GNN Dataset Reference

**Blueprint:** `pagnn_final_blueprint_v4.md` §5 / §8  
**Audit date:** 2026-05-16

---

## Quick-Status Overview

| Dataset | Location | Status | Files | Size |
|---|---|---|---|---|
| HiRISE DEMs | `data/raw/dem/` | ✅ Present | 27 `.IMG` | 6,545 MB |
| HiRISE Browse | `data/raw/hirise_browse/` | ✅ Present | 27 `.JP2` | 7,018 MB |
| CTX Orbital Tiles | `data/raw/ctx/` | ✅ Present | 17,298 `.png` | 3,593 MB |
| HiRISE Map-Proj-v3 | `data/raw/hirise_v3/` | ✅ Present | 7,495 `.jpg` + labels | 102 MB |
| Master index | `data/raw/mars_terrain_vault.csv` | ✅ Present | 27 rows | — |
| DEM tiles (processed) | `data/processed/dem_tiles/` | ❌ Empty | — | — |
| Graphs (processed) | `data/processed/graphs/` | ❌ Empty | — | — |
| CTX pretrain dir | `data/processed/ctx_pretrain/` | ✅ Redirected → raw | — | — |
| CTX demo tiles | `data/processed/ctx_demo/` | ❌ Empty (optional) | — | — |
| Split files | `data/splits/*.txt` | ❌ Empty (4 files) | — | — |

> [!IMPORTANT]
> Three preprocessing steps must run **before training**:
> 1. `python scripts/process_dems.py` — converts .IMG → 512×512 tiles, derives labels, fills splits
> 2. `python scripts/build_graphs.py` — builds PyG graphs from processed tiles
> 3. `python scripts/download_dems.py --verify` — confirms dataset integrity

---

## Dataset 1 — HiRISE DEMs

**Location:** `data/raw/dem/`  **Blueprint:** §5.1, §8

### What it is
27 Digital Terrain Models from the USGS Astrogeology HiRISE instrument. Each encodes per-pixel elevation (metres above MOLA ellipsoid). These are the **source of all hazard labels** — slope, roughness, and discontinuity are derived from these.

### Format
- **Extension:** `.IMG` (USGS PDS/ISIS raw binary raster)
- **Read with:** `rasterio` (GDAL PDS driver) — **not** PIL/cv2
- **Paired browse:** matching `.JP2` in `data/raw/hirise_browse/`

> [!WARNING]
> `.IMG` files cannot be opened with PIL or OpenCV. Use `rasterio.open()` or `gdal.Open()`. The GDAL PDS driver must be compiled in (standard in conda-forge GDAL builds).

### Full terrain catalogue (`mars_terrain_vault.csv` — all 27 Verified)

| Alias | Terrain | DEM File | Scale |
|---|---|---|---|
| Craters_003125 | Craters | DTEEC_003125_1665_003191_1665_A01.IMG | 1.0 |
| Craters_003156 | Craters | DTEEC_003156_1280_002655_1280_A01.IMG | 1.0 |
| Volcanic_006712 | Volcanic | DTEEC_006712_2020_005855_2020_A01.IMG | 1.0 |
| Craters_017719 | Craters | DTEEC_017719_1890_017218_1890_A01.IMG | 1.0 |
| Gullies_025322 | Gullies | DTEEC_025322_2400_026021_2400_G01.IMG | 1.0 |
| Valleys_029246 | Valleys | DTEEC_029246_1345_020543_1345_A01.IMG | 1.0 |
| Craters_033968 | Craters | DTEEC_033968_2065_033823_2065_A01.IMG | 1.0 |
| Canyon Walls_059382 | Canyon Walls | DTEEC_059382_1830_059237_1830_A01.IMG | 1.0 |
| Volcanic_060706 | Volcanic | DTEEC_060706_2195_060416_2195_A01.IMG | 1.0 |
| Volcanic_064700 | Volcanic | DTEEC_064700_1975_064555_1975_A01.IMG | 1.0 |
| Canyon Walls_074396 | Canyon Walls | DTEEC_074396_1425_074251_1425_A01.IMG | 1.0 |
| Canyon Walls_074900 | Canyon Walls | DTEEC_074900_1245_074821_1245_G01.IMG | 1.0 |
| Polar_005721 | Polar | DTEPC_005721_0910_005735_0890_A01.IMG | 1.0 |
| Dunes_022534 | Dunes | DTEPC_022534_1120_022521_1120_A01.IMG | 1.0 |
| Dunes_039932 | Dunes | DTEPC_039932_1055_040064_1055_A01.IMG | 1.0 |
| Polar_048808 | Polar | DTEPC_048808_1000_048650_1000_A01.IMG | 1.0 |
| Polar_049106 | Polar | DTEPC_049106_0970_049080_0970_A01.IMG | 1.0 |
| Polar_010109 | Polar | DTEPD_010109_2655_009991_2655_A01.IMG | 1.0 |
| Polar_023009 | Polar | DTEPD_023009_0930_022839_0930_A01.IMG | 1.0 |
| Craters_010245 | Craters | DTEED_010245_2305_009744_2305_G01.IMG | 2.0 |
| Gullies_022986 | Gullies | DTEED_022986_1300_022841_1300_G01.IMG | 2.0 |
| Valleys_033682 | Valleys | DTEED_033682_2235_033603_2235_A01.IMG | 2.0 |
| Valleys_042725 | Valleys | DTEED_042725_2210_042435_2210_A01.IMG | 2.0 |
| Craters_076968 | Craters | DTEED_076968_1475_076823_1475_A01.IMG | 2.0 |
| Canyon Walls_082989 | Canyon Walls | DTEED_082989_1630_083055_1630_A01.IMG | 2.0 |
| Canyon Walls_088616 | Canyon Walls | DTEED_088616_1750_088471_1750_A01.IMG | 2.0 |
| Craters_089104 | Craters | DTEED_089104_2190_089592_2190_A01.IMG | 2.0 |

**Distribution:** Craters (7) · Canyon Walls (5) · Polar (5) · Volcanic (3) · Valleys (3) · Gullies (2) · Dunes (2)

### Pipeline usage

| Stage | Script | Use |
|---|---|---|
| Stage 1 | `scripts/process_dems.py` | Read .IMG → compute physics features → derive binary hazard labels → tile to 512×512 → write `dem_tiles/` and `splits/` |
| Stage 2 | Physics Engine runtime | Slope, roughness, discontinuity computed from tiled elevation |
| Stage 3/4 | CNN / Fusion training | DEM-derived hazard maps as training labels |
| Stage 6 | GNN training | Per-superpixel mean risk from DEM tiles as weak labels |
| Evaluation | `scripts/evaluate_all.py` | Ground truth for HCR, mIoU, hazard recall |

---

## Dataset 2 — HiRISE Browse Images

**Location:** `data/raw/hirise_browse/`  **Blueprint:** §5.1, §8

### What it is
27 orthorectified HiRISE RED-channel images (JPEG2000) co-registered with the DEMs above. These are the **visual input** to the CNN — the model sees the browse image and the DEM provides the label.

### Format
- **Extension:** `.JP2` (JPEG2000, lossless)
- **Read with:** `rasterio` (GDAL JP2OpenJPEG driver)
- **Size per file:** 52–679 MB

### Pipeline usage

| Stage | Use |
|---|---|
| Stage 1 `process_dems.py` | Co-register browse ↔ DEM → extract 512×512 image tiles → `dem_tiles/<alias>/tile_XXXX.png` |
| Stage 3 CNN | Processed image tiles as model input `(B, 3, 512, 512)` |
| Stage 4 Fusion | Same tiles |

---

## Dataset 3 — MurrayLab CTX Orbital Tiles

**Location:** `data/raw/ctx/`  **Blueprint:** §5.2, §7

### What it is
17,298 non-overlapping 512×512 PNG tiles from the MurrayLab CTX global Mars mosaic (~6 m/pixel). **Unlabelled** — used only for MAE self-supervised pretraining (Stage 0).

### Format & structure
```
data/raw/ctx/
  sliced_tiles_1/    8,649 × 512×512 PNG  (~1.8 GB)
  sliced_tiles_2/    8,649 × 512×512 PNG  (~1.8 GB)
```
- Already at correct 512×512 resolution ✅
- Naming: `tile_x{COL}_y{ROW}_pos({C},{R}).png`
- Mean size: ~208 KB each

### Blueprint target vs actual

| Metric | Blueprint | Actual | Status |
|---|---|---|---|
| Total tiles | 17,298 | 17,298 | ✅ Match |
| Tile size | 512×512 | 512×512 | ✅ Match |
| Format | PNG | PNG | ✅ Match |

### Pipeline usage

| Stage | Use |
|---|---|
| Stage 0 MAE | All 17,298 tiles as unlabelled input. Config: `configs/mae.yaml → ctx_dir = data/raw/ctx` |
| CTX Demo | 3–5 quality tiles selected by `src/evaluation/demo_ctx.py` for qualitative figures |

> [!TIP]
> `configs/mae.yaml` already points to `data/raw/ctx` (updated to skip unnecessary copy step). The MAE DataLoader recursively scans subdirectories.

---

## Dataset 4 — HiRISE Map-Proj-v3

**Location:** `data/raw/hirise_v3/`  **Blueprint:** §5.3, §19

### What it is
Wagstaff et al. (2021) HiRISE landmark classification dataset. 73,031 image crops at 227×227 pixels, labelled with one of 8 terrain classes. Used **exclusively for zero-shot cross-domain evaluation** — no training or fine-tuning on this data.

### File structure
```
data/raw/hirise_v3/
  map-proj-v3/                       7,495 JPEG files (227×227 px)
  labels-map-proj-v3.txt             73,031 lines: "<filename> <class_id>"
  landmarks_map-proj-v3_classmap.csv  8 lines: "<id>,<name>"
```

> [!NOTE]
> 7,495 image files are present on disk. 73,031 labels exist because each image has 7 augmented variants (r90, r180, r270, brt, drk, fh, fv) listed in the labels file but generated on-the-fly. For evaluation, use **originals only** (10,433 entries, every 7th line).

### Class map, risk scores, and label counts

| ID | Class | Risk Score | Hazardous? | Total Crops | Originals |
|---|---|---|---|---|---|
| 0 | other | 0.15 | No | 61,054 | ~8,722 |
| 1 | crater | **0.90** | **Yes** | 4,900 | ~700 |
| 2 | dark dune | **0.85** | **Yes** | 1,141 | ~163 |
| 3 | slope streak | **0.80** | **Yes** | 2,331 | ~333 |
| 4 | bright dune | 0.50 | No | 1,750 | ~250 |
| 5 | impact ejecta | 0.55 | No | 231 | ~33 |
| 6 | swiss cheese | **0.85** | **Yes** | 1,148 | ~164 |
| 7 | spider | 0.45 | No | 476 | ~68 |
| — | **Total** | — | — | **73,031** | **10,433** |

Risk scores defined in `src/data/label_remap.py::HIRISE_V3_RISK_MAP`.

### Blueprint target vs actual

| Metric | Blueprint | Actual | Status |
|---|---|---|---|
| Total crops | 73,031 | 73,031 | ✅ |
| Original crops | 10,433 | 10,433 | ✅ |
| Image size | 227×227 | 227×227 | ✅ |
| Classes | 8 | 8 | ✅ |

### Pipeline usage

| Stage | Use |
|---|---|
| Evaluation §19 | Zero-shot cross-domain test. Images resized 227→512. Metrics: recall, precision, mIoU, ECE, AUC-ROC per class |
| Ablation §20 | Tier-stratified hazard recall |

> [!WARNING]
> Evaluate on **originals only** (`originals_only=True` in `evaluate_hirise_v3()`). Augmented crops are near-duplicates and inflate all metrics.

---

## Processed Directories (Script-Generated)

### `data/processed/dem_tiles/` ❌ Run `scripts/process_dems.py`
```
dem_tiles/
  <alias>/
    tile_0000.png          512×512 uint8 browse image
    tile_0000_risk.npy     (512,512) float32 risk label [0,1]
    tile_0000_valid.npy    (512,512) bool validity mask
```
Used by: Stage 3 CNN, Stage 4 Fusion, Stage 6 GNN, all eval splits.

### `data/processed/graphs/` ❌ Run `scripts/build_graphs.py`
```
graphs/
  <alias>_tile_0000.pt    PyG Data object (torch_geometric.data.Data)
```
Node features: 14-dim vector per §12. Used by `src/data/graph_dataset.py`.

### `data/splits/` ❌ Populated by `scripts/process_dems.py`

| File | Role | Strategy |
|---|---|---|
| `train.txt` | Training (~70%) | Mixed terrain types |
| `val.txt` | Validation (~10%) | Mixed terrain types |
| `test_in.txt` | In-dist test (~10%) | Same terrain types as train |
| `test_ood.txt` | OOD test (~10%) | **Held-out terrain locations** (no leakage) |

**Blueprint §8:** Split at DEM-location level, not tile level, to prevent geographic data leakage.

---

## Pre-Pipeline Checklist

```bash
# Step 1 — Verify raw data integrity
python scripts/download_dems.py --verify

# Step 2 — Process DEMs (must run before anything else)
python scripts/process_dems.py

# Step 3 — Verify splits are populated
python -c "
from pathlib import Path
for f in ['train.txt','val.txt','test_in.txt','test_ood.txt']:
    n = len(Path('data/splits/'+f).read_text().splitlines())
    print(f'{f}: {n} entries')
"

# Step 4 — Begin training pipeline
python scripts/train_mae.py       # Stage 0: ~8-12 hrs on RTX 3060 Ti
python scripts/train_cnn.py       # Stage 3
python scripts/train_fusion.py    # Stage 4
python scripts/build_graphs.py    # Stage 5 (needs fusion checkpoint)
python scripts/train_gnn.py       # Stage 6
```

---

## Format Reference

| Format | Ext | Read with | Notes |
|---|---|---|---|
| USGS PDS raster | `.IMG` | `rasterio`, `gdal` | PDS driver required |
| JPEG2000 | `.JP2` | `rasterio`, `gdal` | JP2OpenJPEG driver |
| PNG tile | `.png` | `PIL`, `torchvision` | 512×512 uint8 |
| JPEG crop | `.jpg` | `PIL`, `cv2` | 227×227, HiRISE v3 |
| PyG graph | `.pt` | `torch.load()` | `torch_geometric.data.Data` |
| NumPy array | `.npy` | `np.load()` | Risk/validity maps |

## Total Footprint

| Category | Size |
|---|---|
| Raw data (all 4 datasets) | ~17.3 GB |
| Processed tiles (estimated) | ~2–5 GB |
| Precomputed graphs (estimated) | ~1–3 GB |
| **Total** | **~20–25 GB** |
