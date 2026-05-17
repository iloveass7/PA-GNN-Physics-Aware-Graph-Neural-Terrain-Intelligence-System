# PA-GNN Progression State

This document tracks the progression and implementation status of the PA-GNN (Physics-Aware Graph Neural Network) pipeline, matching the Thesis Blueprint v4 specifications.

## Stage 0: Self-Supervised Pretraining (MAE) — 🟢 **COMPLETE**
*   **Goal:** Learn terrain representation from unlabelled MurrayLab CTX tiles.
*   **Outputs:** MAE encoder checkpoint (`checkpoints/mae_best.pt`) for Stage 3 init.
*   **Execution Results:** 200 epochs completed (Best Epoch: 195, Loss: 0.351469). Batch size reduced to 16 due to VRAM limits. AMP enabled (~70-75s/epoch).
*   **Files Implemented:**
    *   `src/models/encoder.py`: MobileNetV3-Large + patch embedding + 75% random masking.
    *   `src/models/decoder.py`: 4-layer MLP + masked patch MSE loss.
    *   `src/models/mae.py`: Full MaskedAutoencoder assembly.
    *   `src/data/ctx_loader.py`: Dataset loader for 512×512 MurrayLab CTX tiles.
    *   `scripts/train_mae.py`: Pretraining runner (AdamW, 200 epochs).

## Stage 1: Data Prep & Label Generation — 🟢 **COMPLETE**
*   **Goal:** Convert DEMs to risk labels, align with HiRISE browse images, slice into tiles, and split.
*   **Outputs:** 512×512 `.npy` tile pairs, `tile_manifest.csv`, training splits.
*   **Files Implemented:**
    *   `src/data/dem_loader.py`: PDS .IMG to GeoTIFF conversion + vault CSV parsing.
    *   `src/data/hirise_loader.py`: JP2 to GeoTIFF conversion + GDAL projection alignment.
    *   `src/data/dem_processing.py`: Physics labels (slope, roughness, risk scores).
    *   `src/data/tiling.py`: 512×512 sliding window, 50% overlap, NoData/Saturation rejection.
    *   `src/data/augmentations.py`: Spatial and intensity transforms.
    *   `src/data/normalize.py`: Per-tile minmax + 3-channel grayscale expansion.
    *   `src/data/label_generation.py`: `TilePair` PyTorch Dataset.
    *   `scripts/process_dems.py`: Label and alignment pipeline orchestration.
    *   `scripts/tile_dataset.py`: Tiling runner and DEM-location-based split logic.
    *   `scripts/mola_validation.py`: Required thesis MOLA MEGDR Pearson r correlation.

## Stage 2: Physics Feature Engine — 🟢 **COMPLETE**
*   **Goal:** Compute domain-invariant proxy maps ($S$, $R$, $D$) directly from pixels.
*   **Outputs:** $H_{physics}$ arrays per tile (`data/processed/physics/`).
*   **Files Implemented:**
    *   `src/physics/slope.py`: Sobel gradient magnitude ($S$).
    *   `src/physics/roughness.py`: 7×7 sliding window std via $E[x^2]-E[x]^2$ ($R$).
    *   `src/physics/discontinuity.py`: 13×13 LoG ($\sigma=2.0$) with zero-sum ($D$).
    *   `src/physics/combine.py`: `PhysicsFeatureEngine` assembly ($w_1=0.4, w_2=0.3, w_3=0.3$).
    *   `configs/physics.yaml`: Ablation and weight parameters.
    *   `scripts/validate_dataset.py`: Speed benchmark and risk-ordering sanity checks.
    *   `scripts/precompute_graphs.py`: Pre-caching $H_{physics}$ for Stage 5 graph use.

## Stage 3: CNN Semantic Risk Estimator — 🟢 **COMPLETE**
*   **Goal:** Train a supervised CNN model to predict hazard masks using Stage 0 MAE init.
*   **Outputs:** Trained CNN checkpoint (`checkpoints/cnn_best.pt`).
*   **Files Implemented:**
    *   `src/models/risk_model.py`: MobileNetV3-Large backbone + DeepLabV3+ Decoder (ASPP).
    *   `src/models/ffn_module.py`: Shared FFN block utility.
    *   `src/training/losses.py`: `RiskLoss` (Weighted BCE + 0.5×Dice + 0.1×TV).
    *   `src/training/trainer.py`: Epoch training and validation runner loops.
    *   `configs/cnn.yaml`: Blueprint hyperparameter definitions (batch 8, 60 epochs).
    *   `scripts/train_cnn.py`: Full training orchestration with early stopping and ablation modes.

## Stage 4: Spatial Adaptive Fusion — 🟢 **COMPLETE**
*   **Goal:** Learn a per-pixel $\alpha(x,y)$ mask to dynamically fuse $H_{physics}$ and $H_{learned}$ into $H_{final}$.
*   **Architecture:** 3-layer CNN (Conv(3→16,3×3) → Conv(16→8,3×3) → Conv(8→1,1×1)) with reflect padding and Sigmoid output.
*   **Fusion formula:** $H_{final} = \alpha \cdot H_{learned} + (1-\alpha) \cdot H_{physics}$
*   **Outputs:** Trained fusion checkpoint (`checkpoints/fusion_best.pt`), $\alpha$ maps, $H_{final}$ maps.
*   **Files Implemented:**
    *   `src/models/fusion.py`: `AdaptiveFusion` (3-layer α-prediction CNN, ~1.6K params), `EndToEndFusionModel` (CNN + physics + fusion wrapper with frozen-CNN mode), `fuse_risk_maps()`, `static_fusion()` (B5 baseline), `build_fusion_model()` factory.
    *   `configs/fusion.yaml`: Training config (AdamW, lr=1e-4, 40 epochs, patience 10), `joint_with_cnn: false` (mandatory), same compound loss as Stage 3, α diagnostic thresholds.
    *   `scripts/train_fusion.py`: Training runner with custom `train_one_epoch_fusion()` and `validate_one_epoch_fusion()` (tracks H_final recall vs H_learned/H_physics recall), 6-column visualisation (Image | Target | H_physics | H_learned | α | H_final), α-map degeneration diagnostic, early stopping, resume support.

## Stage 5: Adaptive-Resolution Superpixel Graph — 🟢 **COMPLETE**
*   **Goal:** Build a terrain-complexity-adaptive superpixel graph where node density is driven by $H_{physics}$.
*   **Architecture:** Hierarchical two-pass SLIC + physics-KNN edges (K=5) + 14-dim node features → PyG Data objects.
*   **Node budget:** Flat (<0.25): 5/block, Complex (0.25–0.60): 15/block, Hazard (>0.60): 30–50/block (linear).
*   **Expected nodes:** 120–200 (flat tiles), 300–350 (average), 450–700+ (hazardous tiles).
*   **Outputs:** Precomputed PyG `.pt` graph files (`data/processed/graphs/`), graph statistics CSV.
*   **Files Implemented:**
    *   `src/graph/__init__.py`: Package init with module documentation.
    *   `src/graph/adaptive_slic.py`: `compute_terrain_complexity()` (16×16 block means), `assign_tier_budget()` (3-tier allocation), `adaptive_slic_segmentation()` (two-pass SLIC with hazard refinement + connectivity guarantee).
    *   `src/graph/node_features.py`: `extract_node_features()` — 14-dim feature vector per superpixel (centroid, slope/roughness/disc, H_physics/H_learned/H_final/α, area, intensity stats, hazard flag, neighbour count placeholder).
    *   `src/graph/edges.py`: `build_physics_knn_edges()` (KDTree KNN in combined spatial+physics space + RAG connectivity guarantee), `compute_edge_weights()` (blueprint formula: 0.6×avg_risk + 0.25×norm_dist + 0.15×slope_diff), `build_rag_edges()` (pixel-boundary adjacency fallback).
    *   `src/graph/graph_builder.py`: `build_graph()` orchestrator (SLIC → features → KNN → weights → hazard neighbour count → PyG Data), `build_graph_from_npy()` (file-based entry point), `validate_graph()` (8-point integrity check).
    *   `src/data/graph_dataset.py`: `PrecomputedGraphDataset` (lazy .pt loader), `build_graph_datasets()` factory.
    *   `configs/gnn.yaml`: Graph construction parameters + Stage 6 GATv2 architecture config.
    *   `scripts/precompute_graphs.py`: Full Phase 3a precomputation script (Stage 2→3→4→5 per tile, validation, statistics, bridging frequency monitoring).

## Stage 6: Physics-Aware GATv2 with FFN Module — 🟢 **COMPLETE**
*   **Goal:** Predict node safety states using physics-aware attention message passing.
*   **Outputs:** Trained GATv2+FFN checkpoint (`checkpoints/gnn_best.pt`).
*   **Files Implemented:**
    *   `src/models/gatv2_physics.py`: Custom `PhysicsAwareGATv2Conv` — physics similarity injected into attention logit before softmax ($\lambda \times \exp(-|\Delta S| - |\Delta R|)$), learnable $\lambda$ initialised to 0.1.
    *   `src/models/gnn_model.py`: Full `PhysicsAwareGNN` assembly — 2-layer GATv2 (4 heads) + `GNNFFNBlock` (BatchNorm1d + GELU + residual) + sigmoid head.
    *   `configs/gnn.yaml`: Blueprint §13 hyperparameters (Adam 1e-3, SmoothL1, patience 15, batch 32).
    *   `scripts/train_gnn.py`: Full training orchestration with PyG DataLoader, early stopping, AUC-ROC, CSV logging.


## Stage 7: Uncertainty Estimation (MC Dropout) — 🟢 **COMPLETE**
*   **Goal:** Produce epistemic uncertainty map $U(x,y)$ expressing where the model lacks confidence. High uncertainty triggers conservative routing.
*   **Outputs:** Per-node `risk_mean` + `risk_var`; pixel-space $U(x,y) \in [0,1]^{512 \times 512}$.
*   **Files Implemented:**
    *   `src/uncertainty/mc_dropout.py`: `MCDropoutEstimator` with `mc_dropout_mode` context manager — Dropout layers set to train mode, BatchNorm1d stays in eval mode. N=5 MC passes, per-node variance, vectorised pixel projection via `pixel_membership`.

## Stage 8: Path Planning (A* + D*) — 🟢 **COMPLETE**
*   **Goal:** Route over the GNN-classified graph avoiding hazards with uncertainty-informed costs.
*   **Outputs:** Planned trajectories with per-waypoint risk attribution, PLR, and HCR metrics.
*   **Files Implemented:**
    *   `src/planning/astar.py`: `PhysicsAwareAStar` — $C(i,j) = \exp(3 \times risk_{ij}) \times [0.6 \times risk + 0.25 \times dist + 0.15 \times |\Delta S|]$, uncertainty penalty $(1 + 2U_i)$ when $U_i > 0.3$, no hard node deactivation. Returns `Trajectory` with per-waypoint `dominant_signal` attribution ("physics"/"cnn").
    *   `src/planning/heuristics.py`: Physics-aware $h(n) = d(n, goal) \times (1 + 0.4 \hat{p}_n + 0.1 S_n)$ and baseline Euclidean heuristic for B1.
    *   `src/planning/dstar.py`: `DStarLite` incremental replanner for dynamic edge cost updates during active traversal.

---

## Changes — Stage 0 Execution Fixes & Optimisations

All changes applied **2026-05-16**, during the execution of Stage 0.

### Bug Fixes (Critical — pipeline would crash or silently corrupt without these)

| # | File | Change | Rationale |
|---|---|---|---|
| BF-1 | `src/models/decoder.py` | `PATCH_DIM` = `16×16×1 = 256` (was `16×16×3 = 768`) | CTX tiles are single-channel `(1, 512, 512)`. `patchify()` produces `(B, 1024, 256)` but decoder output was `(B, 1024, 768)`. Shape mismatch crashes `mae_loss()` on first forward pass. |
| BF-2 | `src/models/encoder.py` | Moved `_proj_to_1ch` Conv2d from lazy `forward()` init to `__init__()` | Bare attribute assignment inside `forward()` meant the layer was not registered as a submodule — absent from `state_dict()`, not moved by `.to(device)`, and not included in the optimizer. Backbone received random-projected input for all 200 epochs (silent corruption). |
| BF-3 | `scripts/train_mae.py` | `unpatchify(..., channels=1)` in two places + fixed misleading comments | `unpatchify` was called with `channels=3` but images are single-channel. Crashes during the verification visualisation step at epoch 50/100/150/200. |
| BF-4 | `src/models/decoder.py` | `patchify` used `flatten(2)` instead of `reshape(B, h*w, -1)` | Shape mismatch in `mae_loss` caused crash during training. |
| BF-5 | `scripts/train_mae.py` | De-normalise raw predictions before compositing | Visualisation produced black/white noise due to raw logits composited against `[0,1]` original images. |

### Performance & Configuration Adjustments

| # | File | Change | Expected Gain / Rationale |
|---|---|---|---|
| PF-1 | `scripts/train_mae.py` | Added AMP (`torch.amp.autocast` + `GradScaler`) | ~40–50% faster on CUDA (FP16 tensor cores). Dropped epoch time to ~70-75s. |
| PF-2 | `scripts/train_mae.py` | `num_workers` default 4 → 8 | Eliminates DataLoader I/O stalls. |
| PF-3 | `src/models/encoder.py` | Upsample to 256×256 instead of 512×512 before backbone | ~10–15% less backbone compute. Handled VRAM limits alongside batch reduction. |
| PF-4 | `scripts/train_mae.py` | Batch size reduced 64 → 16 | Prevented OOM crash on RTX 3060 Ti due to VRAM limits. |
| PF-5 | `scripts/train_mae.py` | Removed `torch.compile` | Triton is not supported on Windows; removed to prevent crashes. |

---

## Changes — Stage 1 Pre-Run Fixes & Performance Enhancements

All changes applied **2026-05-16**, before the execution of Stage 1.

### Bug Fixes

| # | File | Change | Rationale |
|---|---|---|---|
| BUG-01 | `src/data/label_generation.py` | `build_dataset()` now auto-appends `split` subdirectory to `tiles_dir` (e.g. `tiles/train/`) with fallback to flat layout | `tile_dataset.py` writes tiles into `TILES_DIR / split` but `build_dataset()` searched only the root `tiles_dir` — zero tiles would be found in all downstream scripts (Stage 3/4/eval). |
| BUG-01b | `scripts/evaluate_all.py` | Fixed import: `from src.data.label_generation import build_dataset` (was `src.data.dem_loader` — doesn't exist there). Fixed path: `tiles` (was `dem_tiles`). | Script would crash with `ImportError` before even reaching the tiles_dir bug. |
| BUG-01c | `scripts/run_ablations.py` | Same fix as BUG-01b. | Same crash scenario. |
| BASE-01 | `configs/base.yaml` | Updated `dem_tiles` path from `data/processed/dem_tiles` to `data/processed/tiles` | Alignment between config and actual implementation. |

### Minor Fixes

| # | File | Change | Rationale |
|---|---|---|---|
| MINOR-01 | `src/data/label_generation.py` | Removed unused `from src.data.normalize import to_cnn_input` | Dead import — `to_cnn_input` is never called in this module. |
| MINOR-03a | `configs/datasets/dem.yaml` | Populated with blueprint §8 parameters (tiling, labels, hazard thresholds, splits) | Was 0 bytes. Required for downstream config-driven usage. |
| MINOR-03b | `configs/datasets/ctx.yaml` | Populated with blueprint §5.2 parameters (paths, properties, demo selection, pretraining) | Was 0 bytes. |
| MINOR-03c | `configs/datasets/hirise_v3.yaml` | Populated with blueprint §5.3 parameters (paths, classes, risk remapping, evaluation properties) | Was 0 bytes. |

### Performance Enhancements

| # | File | Change | Expected Gain / Rationale |
|---|---|---|---|
| PERF-01 | `scripts/process_dems.py` | Parallelised per-pair loop with `ProcessPoolExecutor` (default 4 workers). Extracted `_process_one_pair()` as a module-level function for pickling. Added `--workers` CLI flag. | ~3-4× speedup on 27 DEM pairs. Each pair (convert→align→label) is independent. gdalwarp subprocess calls overlap across workers. `--workers 1` provides sequential fallback for debugging. |
| PERF-02 | `scripts/tile_dataset.py` | Parallelised per-pair tiling loop with `ProcessPoolExecutor` (default 4 workers). Extracted `_tile_one_pair()` as a module-level function. Added `--workers` CLI flag. | ~3-4× speedup. Each pair's tiling reads GeoTIFFs and writes .npy files independently. Split subdirectories are pre-created before workers start. |
| PERF-05 | `src/data/hirise_loader.py` | Added detailed PERF-05 docstring documenting gdalwarp as the CPU-only bottleneck and noting that PERF-01 outer-loop parallelism is the correct mitigation. | Awareness / documentation. Prevents future developers from attempting GPU or inner-loop threading approaches. |

---

## Stage 1 Execution Results

**Run date:** 2026-05-17  
**Status:** ✅ **PASSED — all outputs verified on disk**

### DEM Processing (`process_dems.py`)

- **Runtime:** ~66 min (4 workers)
- **Outcome:** 27/27 DEMs processed successfully, 0 failures
- **Labels generated:** 135 files (27 × {slope, roughness, risk, hazard, validity}.tif) ✅
- **Aligned browse images:** 27 GeoTIFFs ✅
- **TIF cache:** 27 DEMs + 27 browse images ✅
- **stage1_report.csv:** 28 lines (header + 27 entries, all status=OK) ✅
- **CRS groups:** Equirectangular MARS (mid-latitude) + Polar Stereographic MARS (polar/dune) — both handled correctly
- **Scale groups:** 19 DEMs at ~1m/px, 8 DEMs at ~2m/px

### Tiling (`tile_dataset.py`)

- **Runtime:** ~67s (4 workers)
- **Total tiles:** 14,686 (within blueprint target 5,000–15,000) ✅

| Split | Locations | Tiles | Files on Disk (×4 .npy) | Verified |
|---|---|---|---|---|
| train | 18 | 9,203 | 36,812 | ✅ |
| val | 3 | 1,761 | 7,044 | ✅ |
| test_in | 5 | 3,566 | 14,264 | ✅ |
| test_ood | 1 | 156 | 624 | ✅ |
| **TOTAL** | **27** | **14,686** | **58,744** | ✅ |

- **OOD selection:** `Craters_089104` — 6 Craters remain in train ✅
- **tile_manifest.csv:** 14,687 lines (header + 14,686 entries) ✅
- **Split files:** 4 files in `data/splits/` — all populated with correct location counts ✅
- **Saturation rejections:** 3 total (negligible) ✅
- **NoData rejections:** Proportional to per-DEM NoData fractions ✅

### Known Non-Blocking Concerns

1. **Polar_005721:** 62% NoData — exceeded 50% threshold but still yielded 776 usable tiles. Not a blocker.
2. **test_ood set:** Only 156 tiles from 1 DEM — structural constraint of the blueprint's OOD strategy. Report with bootstrapped CIs at evaluation.

### Full Analysis

See [`stage1_results.md`](file:///d:/Physics%20Aware%20-%20Graphical%20Neural%20Network%20for%20Planetary%20Path%20Planning/pa-gnn/results/result_anals/stage1_results.md) for detailed per-DEM breakdowns, hazard fraction analysis, and blueprint compliance table.

---

## Changes — Stage 2 Pre-Run Fixes

All changes applied **2026-05-17**, before the execution of Stage 2.

### Bug Fixes

| # | File | Change | Rationale |
|---|---|---|---|
| BUG-02a | `scripts/precompute_graphs.py` | Added `gnn.yaml` loading to correctly define `allocation_mode`, `gamma`, `n_min`, `n_max`, and `compactness` | Variables were referenced in the `build_graph()` call but never defined in the script, leading to a `NameError` crash at Stage 5. |
| BUG-02b | `src/graph/graph_builder.py` | Added 6 missing parameters (`allocation_mode`, `gamma`, `n_min`, `n_max`, `edge_mode`, `edge_scorer`) to `build_graph_from_npy()` signature with defaults. | `build_graph_from_npy` was passing these variables to `build_graph()` but they were undefined in its scope, causing the same `NameError` crash at Stage 5. |

### Minor Fixes

| # | File | Change | Rationale |
|---|---|---|---|
| MINOR-02 | `scripts/validate_dataset.py` | Added `random.seed(42)` at the start of `main()` | Ensures reproducible random sampling across runs for the Stage 2 sanity checks, visualizations, and especially the baseline Pearson r correlation metric reported in the thesis. |

