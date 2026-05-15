"""
node_features.py
----------------
Stage 5 — Step 4: 14-Dimensional Node Feature Extraction.

Blueprint §12 Step 4:
  Every superpixel node carries a 14-dimensional feature vector.
  This vector is identical in structure across all nodes regardless
  of superpixel size.

  | Idx | Feature                          | Source   |
  |-----|----------------------------------|----------|
  |  0  | Centroid x (normalised 0–1)      | SLIC     |
  |  1  | Centroid y (normalised 0–1)      | SLIC     |
  |  2  | Mean slope S                     | Stage 2  |
  |  3  | Mean roughness R                 | Stage 2  |
  |  4  | Mean discontinuity D             | Stage 2  |
  |  5  | Mean H_physics                   | Stage 2  |
  |  6  | Mean H_learned                   | Stage 3  |
  |  7  | Mean H_final                     | Stage 4  |
  |  8  | Mean α                           | Stage 4  |
  |  9  | Node area (normalised 0–1)       | SLIC     |
  | 10  | Mean pixel intensity             | Image    |
  | 11  | Std pixel intensity              | Image    |
  | 12  | Hazardous flag (H_final > 0.7)   | Stage 4  |
  | 13  | Hazardous neighbour count        | Graph    |

  Note on Dimension 9 (from blueprint):
    In the adaptive-resolution graph, node area is CRITICAL. A 600-pixel
    node carries averaged information from 6× as many pixels as a 100-pixel
    node. Including normalised area allows the attention mechanism to
    account for node size disparity.

  Dimension 13 is computed AFTER edge construction (in graph_builder.py)
  because it requires the adjacency structure.

Exports:
    extract_node_features  — compute 14-dim features for all superpixels
    NODE_FEATURE_DIM       — constant = 14

Usage:
    from src.graph.node_features import extract_node_features

    features, centroids, areas, node_labels = extract_node_features(
        labels, image, h_physics, slope, roughness, disc,
        h_learned, h_final, alpha, risk_target,
    )
"""

import logging

import numpy as np

log = logging.getLogger(__name__)

NODE_FEATURE_DIM: int = 14

# Blueprint §12 Step 4 / §11: hazard threshold
HAZARD_THRESHOLD: float = 0.7


def extract_node_features(
    labels: np.ndarray,
    image: np.ndarray,
    h_physics: np.ndarray,
    slope: np.ndarray,
    roughness: np.ndarray,
    disc: np.ndarray,
    h_learned: np.ndarray,
    h_final: np.ndarray,
    alpha: np.ndarray,
    risk_target: np.ndarray | None = None,
    tier_map: np.ndarray | None = None,
) -> dict:
    """Extract 14-dimensional feature vector for each superpixel node.

    Parameters
    ----------
    labels : (H, W) int32
        Superpixel label map from adaptive SLIC (contiguous from 0).
    image : (H, W) float32 in [0, 1]
        Original grayscale image.
    h_physics : (H, W) float32 in [0, 1]
        Combined physics risk map (Stage 2).
    slope : (H, W) float32 in [0, 1]
        Slope proxy S (Stage 2).
    roughness : (H, W) float32 in [0, 1]
        Roughness proxy R (Stage 2).
    disc : (H, W) float32 in [0, 1]
        Discontinuity proxy D (Stage 2).
    h_learned : (H, W) float32 in [0, 1]
        CNN risk prediction (Stage 3).
    h_final : (H, W) float32 in [0, 1]
        Fused risk map (Stage 4).
    alpha : (H, W) float32 in [0, 1]
        Fusion trust map (Stage 4).
    risk_target : (H, W) float32 in [0, 1] or None
        DEM-derived risk label for GNN training target.
        If None, uses h_final as fallback (inference mode).
    tier_map : (grid_h, grid_w) int32 or None
        Tier map from adaptive SLIC. Used to assign tier per node.

    Returns
    -------
    dict with keys:
        features   : (N, 14) float32 — node feature matrix
                     (dimension 13 is initialised to 0; filled by graph_builder
                      after edge construction)
        centroids  : (N, 2)  float32 — centroid (y, x) in pixel coords
        areas      : (N,)    int32   — pixel count per superpixel
        node_labels: (N,)    float32 — DEM risk per node (GNN target)
        tiers      : (N,)    int32   — tier assignment per node (0/1/2)
        pixel_membership : (H, W) int32 — labels map (same as input)
    """
    H, W = labels.shape
    tile_diagonal = np.sqrt(H ** 2 + W ** 2)
    unique_ids = np.unique(labels)
    N = len(unique_ids)

    # Pre-allocate arrays
    features = np.zeros((N, NODE_FEATURE_DIM), dtype=np.float32)
    centroids = np.zeros((N, 2), dtype=np.float32)
    areas = np.zeros(N, dtype=np.int32)
    node_labels_arr = np.zeros(N, dtype=np.float32)
    tiers = np.zeros(N, dtype=np.int32)

    # Block size for tier lookup
    block_size = 32

    # Compute max area for normalisation
    max_area = 0
    for i, sp_id in enumerate(unique_ids):
        mask = (labels == sp_id)
        areas[i] = mask.sum()
        if areas[i] > max_area:
            max_area = areas[i]

    max_area = max(max_area, 1)  # guard against zero

    for i, sp_id in enumerate(unique_ids):
        mask = (labels == sp_id)
        ys, xs = np.where(mask)

        # --- Centroid (normalised 0–1) ---
        cy = ys.mean()
        cx = xs.mean()
        centroids[i] = [cy, cx]

        # Feature 0, 1: normalised centroid
        features[i, 0] = cx / max(W - 1, 1)
        features[i, 1] = cy / max(H - 1, 1)

        # Feature 2: mean slope S
        features[i, 2] = slope[mask].mean()

        # Feature 3: mean roughness R
        features[i, 3] = roughness[mask].mean()

        # Feature 4: mean discontinuity D
        features[i, 4] = disc[mask].mean()

        # Feature 5: mean H_physics
        features[i, 5] = h_physics[mask].mean()

        # Feature 6: mean H_learned
        features[i, 6] = h_learned[mask].mean()

        # Feature 7: mean H_final
        features[i, 7] = h_final[mask].mean()

        # Feature 8: mean α
        features[i, 8] = alpha[mask].mean()

        # Feature 9: node area (normalised 0–1)
        features[i, 9] = areas[i] / max_area

        # Feature 10: mean pixel intensity (albedo proxy)
        features[i, 10] = image[mask].mean()

        # Feature 11: std pixel intensity (texture variance)
        features[i, 11] = image[mask].std() if areas[i] > 1 else 0.0

        # Feature 12: hazardous flag (based on threshold)
        features[i, 12] = 1.0 if features[i, 7] > HAZARD_THRESHOLD else 0.0

        # Feature 13: hazardous neighbour count (placeholder — filled by graph_builder)
        features[i, 13] = 0.0

        # --- Node label (GNN training target) ---
        if risk_target is not None:
            node_labels_arr[i] = risk_target[mask].mean()
        else:
            node_labels_arr[i] = h_final[mask].mean()

        # --- Tier assignment ---
        if tier_map is not None:
            # Use centroid to look up which block → which tier
            block_y = min(int(cy) // block_size, tier_map.shape[0] - 1)
            block_x = min(int(cx) // block_size, tier_map.shape[1] - 1)
            tiers[i] = tier_map[block_y, block_x]
        else:
            # Fallback: assign based on mean H_physics
            hp_val = features[i, 5]
            if hp_val < 0.25:
                tiers[i] = 0
            elif hp_val > 0.60:
                tiers[i] = 2
            else:
                tiers[i] = 1

    log.debug(
        "Node features: N=%d, area range=[%d, %d], "
        "mean_features=[%.3f, %.3f, ...], hazardous_nodes=%d",
        N, areas.min(), areas.max(),
        features[:, 0].mean(), features[:, 1].mean(),
        int(features[:, 12].sum()),
    )

    # --- Phase 8: Feature Normalization (Z-score) ---
    # We want to normalize physics and image features across the graph
    # to stabilize GNN training. Features 0, 1, 9, 12 are already 0-1.
    # We don't normalize 13 yet because it's computed later.
    features_to_normalize = [2, 3, 4, 5, 6, 7, 8, 10, 11]
    for f_idx in features_to_normalize:
        feat_col = features[:, f_idx]
        f_mean = feat_col.mean()
        f_std = feat_col.std()
        if f_std > 1e-6:
            features[:, f_idx] = (feat_col - f_mean) / f_std
        else:
            features[:, f_idx] = feat_col - f_mean # Center if zero variance

    return {
        "features":         features,            # (N, 14) float32
        "centroids":        centroids,           # (N, 2)  float32 — (y, x)
        "areas":            areas,               # (N,)    int32
        "node_labels":      node_labels_arr,     # (N,)    float32 — GNN target
        "tiers":            tiers,               # (N,)    int32
        "pixel_membership": labels,              # (H, W)  int32
    }
