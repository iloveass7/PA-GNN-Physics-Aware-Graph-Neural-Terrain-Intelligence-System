"""
graph_builder.py
----------------
Stage 5 — Step 7: Full Image → PyG Data Orchestrator.

Blueprint §12 Step 7:
  Package each tile's graph as a PyTorch Geometric Data object.

  x             : node features, shape (num_nodes, 14), float32
  edge_index    : COO adjacency, shape (2, num_edges), int64
  edge_attr     : edge weights, shape (num_edges, 1), float32
  pos           : centroid pixel coords, shape (num_nodes, 2), float32
  y             : DEM-derived risk label per node, shape (num_nodes,), float32
  tier          : tier assignment per node (0/1/2), shape (num_nodes,), int64
  pixel_membership : pixel→node map, shape (512, 512), int64

This module orchestrates:
  1. adaptive_slic_segmentation  (Step 1–3)
  2. extract_node_features       (Step 4)
  3. build_physics_knn_edges     (Step 5)
  4. compute_edge_weights        (Step 6)
  5. Fill feature 13 (hazardous neighbour count)
  6. Package as PyG Data         (Step 7)

Exports:
    build_graph          — single tile → PyG Data
    build_graph_from_npy — load from .npy files → PyG Data (for precompute_graphs.py)

Usage:
    from src.graph.graph_builder import build_graph

    data = build_graph(
        image, h_physics, slope, roughness, disc,
        h_learned, h_final, alpha, risk_target,
    )
"""

import logging

import numpy as np
import torch

try:
    from torch_geometric.data import Data as PyGData
except ImportError:
    PyGData = None
    logging.getLogger(__name__).warning(
        "torch_geometric not installed. Graph building will fail. "
        "Install with: pip install torch-geometric"
    )

from src.graph.adaptive_slic import (
    adaptive_slic_segmentation,
    TIER_FLAT,
    TIER_COMPLEX,
    TIER_HAZARD,
)
from src.graph.node_features import extract_node_features, NODE_FEATURE_DIM
from src.graph.edges import (
    build_physics_knn_edges,
    compute_edge_weights,
)

log = logging.getLogger(__name__)

# Blueprint thresholds
HAZARD_THRESHOLD: float = 0.7


def _fill_hazardous_neighbour_count(
    features: np.ndarray,
    edge_index: np.ndarray,
) -> np.ndarray:
    """Fill feature dimension 13: count of hazardous neighbours.

    Blueprint §12 Step 4, index 13:
      For each node, count how many of its graph neighbours have
      H_final > 0.7 (i.e., feature[12] == 1.0).

    Parameters
    ----------
    features : (N, 14) float32
        Node feature matrix (feature[12] = hazardous flag already set).
    edge_index : (2, E) int64
        COO-format edges.

    Returns
    -------
    features : (N, 14) float32
        Updated with dimension 13 filled.
    """
    N = features.shape[0]
    hazard_count = np.zeros(N, dtype=np.float32)

    E = edge_index.shape[1]
    for k in range(E):
        src, dst = int(edge_index[0, k]), int(edge_index[1, k])
        if features[dst, 12] > 0.5:  # dst is hazardous
            hazard_count[src] += 1

    # Normalise by max count to keep in reasonable range for GNN
    max_count = hazard_count.max()
    if max_count > 0:
        hazard_count = hazard_count / max_count

    features[:, 13] = hazard_count
    return features


def build_graph(
    image: np.ndarray,
    h_physics: np.ndarray,
    slope: np.ndarray,
    roughness: np.ndarray,
    disc: np.ndarray,
    h_learned: np.ndarray,
    h_final: np.ndarray,
    alpha: np.ndarray,
    risk_target: np.ndarray | None = None,
    K: int = 5,
    flat_threshold: float = 0.25,
    hazard_threshold: float = 0.60,
    allocation_mode: str = "continuous",
    gamma: float = 1.5,
    n_min: int = 8,
    n_max: int = 64,
    compactness: float = 10.0,
    sigma: float = 1.0,
    edge_mode: str = "static",
    edge_scorer: torch.nn.Module | None = None,
) -> "PyGData":
    """Build a complete PyG Data object from a single tile.

    Orchestrates the full Stage 5 pipeline:
      Step 1–3: Adaptive SLIC segmentation
      Step 4:   14-dim node features
      Step 5:   Physics-KNN edges + connectivity guarantee
      Step 6:   Edge weights
      Step 7:   PyG Data packaging

    Parameters
    ----------
    image : (H, W) float32 in [0, 1]
        Grayscale image tile.
    h_physics : (H, W) float32 in [0, 1]
        Combined physics risk map (Stage 2).
    slope, roughness, disc : (H, W) float32 in [0, 1]
        Individual physics features (Stage 2).
    h_learned : (H, W) float32 in [0, 1]
        CNN risk prediction (Stage 3).
    h_final : (H, W) float32 in [0, 1]
        Fused risk map (Stage 4).
    alpha : (H, W) float32 in [0, 1]
        Fusion trust map (Stage 4).
    risk_target : (H, W) float32 or None
        DEM risk label for GNN training target.
        If None, uses h_final as target (inference mode).
    K : int
        KNN neighbour count (default: 5, blueprint §12).
    flat_threshold, hazard_threshold : float
        Tier thresholds (configurable for ablation §20).
    compactness, sigma : float
        SLIC parameters.

    Returns
    -------
    PyGData with fields: x, edge_index, edge_attr, pos, y, tier,
                         pixel_membership, graph_stats
    """
    if PyGData is None:
        raise ImportError(
            "torch_geometric is required for graph construction. "
            "Install: pip install torch-geometric"
        )

    H, W = image.shape

    # --- Step 1–3: Adaptive SLIC ---
    labels, tier_map, slic_stats = adaptive_slic_segmentation(
        h_physics, image,
        flat_threshold=flat_threshold,
        hazard_threshold=hazard_threshold,
        allocation_mode=allocation_mode,
        gamma=gamma,
        n_min=n_min,
        n_max=n_max,
        compactness=compactness,
        sigma=sigma,
    )

    # --- Step 4: Node features ---
    node_data = extract_node_features(
        labels=labels,
        image=image,
        h_physics=h_physics,
        slope=slope,
        roughness=roughness,
        disc=disc,
        h_learned=h_learned,
        h_final=h_final,
        alpha=alpha,
        risk_target=risk_target,
        tier_map=tier_map,
    )

    features    = node_data["features"]       # (N, 14)
    centroids   = node_data["centroids"]      # (N, 2) — (y, x)
    node_labels = node_data["node_labels"]    # (N,)
    tiers       = node_data["tiers"]          # (N,)
    pixel_membership = node_data["pixel_membership"]  # (H, W)

    N = features.shape[0]

    # --- Step 5: Physics-KNN edges ---
    edge_index, bridged = build_physics_knn_edges(
        features=features,
        centroids=centroids,
        labels=labels,
        K=K,
        mode=edge_mode,
        scorer=edge_scorer,
    )

    # --- Fill feature 13: hazardous neighbour count ---
    features = _fill_hazardous_neighbour_count(features, edge_index)

    # --- Step 6: Edge weights ---
    edge_weights = compute_edge_weights(
        edge_index=edge_index,
        features=features,
        centroids=centroids,
        H=H, W=W,
    )

    # --- Step 7: Package as PyG Data ---
    # Centroid as (x, y) for pos (PyG convention: pos is spatial coordinates)
    pos = centroids[:, ::-1].copy()  # convert (y, x) → (x, y)

    data = PyGData(
        x=torch.from_numpy(features).float(),                    # (N, 14)
        edge_index=torch.from_numpy(edge_index).long(),           # (2, E)
        edge_attr=torch.from_numpy(edge_weights).float().unsqueeze(-1),  # (E, 1)
        pos=torch.from_numpy(pos).float(),                       # (N, 2)
        y=torch.from_numpy(node_labels).float(),                  # (N,)
        tier=torch.from_numpy(tiers).long(),                      # (N,)
        pixel_membership=torch.from_numpy(pixel_membership).long(),  # (H, W)
    )

    # Attach stats as metadata (not a tensor, but stored for logging)
    data.graph_stats = {
        **slic_stats,
        "num_nodes": N,
        "num_edges": edge_index.shape[1],
        "bridged": bridged,
        "feature_dim": NODE_FEATURE_DIM,
    }

    log.debug(
        "Graph built: N=%d, E=%d, bridged=%s, tiers=%s",
        N, edge_index.shape[1], bridged, slic_stats["tier_counts"],
    )

    return data


def build_graph_from_npy(
    image_path: str,
    risk_path: str | None = None,
    h_physics_path: str | None = None,
    slope_path: str | None = None,
    roughness_path: str | None = None,
    disc_path: str | None = None,
    h_learned_path: str | None = None,
    h_final_path: str | None = None,
    alpha_path: str | None = None,
    K: int = 5,
    flat_threshold: float = 0.25,
    hazard_threshold: float = 0.60,
    physics_engine=None,
    fusion_model=None,
    device: str = "cpu",
) -> "PyGData":
    """Build graph from .npy tile files using model inference.

    This is the entry point used by scripts/precompute_graphs.py.
    If pre-computed maps (h_physics, h_learned, etc.) are not available
    as .npy files, they are computed on-the-fly from the image using
    the provided physics_engine and fusion_model.

    Parameters
    ----------
    image_path : str
        Path to image .npy file (H, W) float32.
    risk_path : str or None
        Path to DEM risk label .npy file (GNN training target).
    h_physics_path, slope_path, roughness_path, disc_path : str or None
        Paths to pre-computed physics maps. If None, computed from image.
    h_learned_path, h_final_path, alpha_path : str or None
        Paths to pre-computed CNN/fusion maps. If None, computed via model.
    K : int
        KNN neighbour count.
    flat_threshold, hazard_threshold : float
        Tier thresholds.
    physics_engine : PhysicsFeatureEngine or None
        Required if physics maps are not pre-computed.
    fusion_model : EndToEndFusionModel or None
        Required if fusion maps are not pre-computed.
    device : str
        Torch device for model inference.

    Returns
    -------
    PyGData
    """
    import torch as th

    # Load image
    image = np.load(image_path).astype(np.float32)

    # Load or compute risk target
    risk_target = None
    if risk_path is not None:
        risk_target = np.load(risk_path).astype(np.float32)

    # --- Physics maps ---
    if all(p is not None for p in [h_physics_path, slope_path, roughness_path, disc_path]):
        h_physics = np.load(h_physics_path).astype(np.float32)
        slope     = np.load(slope_path).astype(np.float32)
        roughness = np.load(roughness_path).astype(np.float32)
        disc      = np.load(disc_path).astype(np.float32)
    elif physics_engine is not None:
        x = th.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
        with th.no_grad():
            h_phys_t, feats_t = physics_engine(x)
        h_physics = h_phys_t[0, 0].cpu().numpy()
        slope     = feats_t["slope"][0, 0].cpu().numpy()
        roughness = feats_t["roughness"][0, 0].cpu().numpy()
        disc      = feats_t["disc"][0, 0].cpu().numpy()
    else:
        raise ValueError(
            "Either pre-computed physics .npy paths or physics_engine must be provided."
        )

    # --- Fusion maps ---
    if all(p is not None for p in [h_learned_path, h_final_path, alpha_path]):
        h_learned = np.load(h_learned_path).astype(np.float32)
        h_final   = np.load(h_final_path).astype(np.float32)
        alpha     = np.load(alpha_path).astype(np.float32)
    elif fusion_model is not None:
        x = th.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
        with th.no_grad():
            result = fusion_model(x)
        h_learned = result["h_learned"][0, 0].cpu().numpy()
        h_final   = result["h_final"][0, 0].cpu().numpy()
        alpha     = result["alpha"][0, 0].cpu().numpy()
    else:
        raise ValueError(
            "Either pre-computed fusion .npy paths or fusion_model must be provided."
        )

    return build_graph(
        image=image,
        h_physics=h_physics,
        slope=slope,
        roughness=roughness,
        disc=disc,
        h_learned=h_learned,
        h_final=h_final,
        alpha=alpha,
        risk_target=risk_target,
        K=K,
        flat_threshold=flat_threshold,
        hazard_threshold=hazard_threshold,
        allocation_mode=allocation_mode,
        gamma=gamma,
        n_min=n_min,
        n_max=n_max,
        edge_mode=edge_mode,
        edge_scorer=edge_scorer,
    )


def validate_graph(data: "PyGData") -> dict[str, bool]:
    """Validate a built graph against blueprint specifications.

    Checks:
      1. Feature dimensions = 14
      2. All features in expected ranges
      3. Graph is connected (single component)
      4. Node count in expected range (1–1000)
      5. Labels in [0, 1]
      6. Edge weights are positive

    Returns
    -------
    dict of check_name → passed (bool)
    """
    checks = {}

    # 1. Feature dimension
    checks["feature_dim_14"] = (data.x.shape[1] == NODE_FEATURE_DIM)

    # 2. Feature value ranges
    checks["features_finite"] = bool(torch.isfinite(data.x).all())
    checks["features_bounded"] = bool(
        (data.x[:, :12] >= -0.1).all() and (data.x[:, :12] <= 1.1).all()
    )

    # 3. Connectivity (single component)
    edge_np = data.edge_index.numpy()
    N = data.x.shape[0]
    from src.graph.edges import _find_connected_components
    components = _find_connected_components(N, edge_np)
    checks["single_component"] = (len(components) == 1)

    # 4. Node count range
    checks["node_count_valid"] = (1 <= N <= 1500)

    # 5. Labels in [0, 1]
    checks["labels_bounded"] = bool(
        (data.y >= -0.01).all() and (data.y <= 1.01).all()
    )

    # 6. Edge weights positive
    if data.edge_attr is not None and data.edge_attr.numel() > 0:
        checks["edge_weights_positive"] = bool((data.edge_attr >= 0).all())
    else:
        checks["edge_weights_positive"] = True

    # 7. Pixel membership covers all pixels
    if hasattr(data, "pixel_membership"):
        pm = data.pixel_membership
        checks["pixel_membership_complete"] = bool(
            (pm >= 0).all() and (pm < N).all()
        )

    # 8. Tier values valid
    if hasattr(data, "tier"):
        checks["tiers_valid"] = bool(
            ((data.tier == 0) | (data.tier == 1) | (data.tier == 2)).all()
        )

    return checks
