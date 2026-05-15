# PA-GNN Progression State

This document tracks the progression and implementation status of the PA-GNN (Physics-Aware Graph Neural Network) pipeline, matching the Thesis Blueprint v4 specifications.

## Stage 0: Self-Supervised Pretraining (MAE) — 🟢 **COMPLETE**
*   **Goal:** Learn terrain representation from unlabelled MurrayLab CTX tiles.
*   **Outputs:** MAE encoder checkpoint (`checkpoints/mae_best.pt`) for Stage 3 init.
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

## Stage 5: Adaptive-Resolution Superpixel Graph — 🔴 **PENDING**
*   **Goal:** Build a hierarchical SLIC graph where node density is controlled by $H_{physics}$.
*   **Requirements:**
    *   Adaptive block logic (flat=5, complex=15, hazard=30-50 nodes).
    *   14-dimensional node feature vectors.
    *   Distance/Cosine edge connections.

## Stage 6: Graph Neural Network (GATv2) — 🔴 **PENDING**
*   **Goal:** Predict node safety states using message passing.
*   **Requirements:**
    *   GATv2 architecture with 4 attention heads.
    *   Weakly-supervised label generation (10% subset supervision).

## Stage 7: Path Planning (A*) — 🔴 **PENDING**
*   **Goal:** Route over the GNN-classified graph avoiding hazards.
*   **Requirements:**
    *   Physics-aware cost function ($w_{dist} + w_{risk}$).
    *   Interpolation/smoothing back to image space.
