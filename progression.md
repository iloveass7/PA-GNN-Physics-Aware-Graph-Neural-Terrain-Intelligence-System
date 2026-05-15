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

## Stage 4: Spatial Adaptive Fusion — 🔴 **PENDING**
*   **Goal:** Learn a per-pixel $\alpha(x,y)$ mask to dynamically fuse $H_{physics}$ and $H_{learned}$.
*   **Requirements:**
    *   Freeze Stage 3 CNN backbone.
    *   Small 3-layer CNN for $\alpha(x,y)$ prediction.
    *   Compound loss on final output $H_{final}$.

## Stage 5: Adaptive-Resolution Superpixel Graph — 🔴 **PENDING**
*   **Goal:** Build a hierarchical SLIC graph where node density is controlled by $H_{physics}$.
*   **Requirements:**
    *   Adaptive block logic (flat=5, complex=15, hazard=30-50 nodes).
    *   14-dimensional node feature vectors.
    *   Distance/Cosine edge connections.

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
