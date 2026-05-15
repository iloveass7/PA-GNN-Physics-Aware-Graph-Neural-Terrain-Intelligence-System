"""
edges.py
--------
Stage 5 — Step 5 + 6: Physics-Similarity KNN Edge Construction + Edge Weights.

Blueprint §12 Step 5 — Physics-Similarity KNN Edge Construction:
  For each node, connect to K=5 nearest neighbours in combined spatial
  and physics distance space.

  Spatial distance:  Euclidean between centroid coords, normalised by tile diagonal.
  Physics distance:  Euclidean between [S, R, D, H_physics], normalised by
                     tile max pairwise physics distance.
  Combined distance: 0.5 × spatial + 0.5 × physics

  KNN: 5 nearest neighbours by combined distance. Edges added in both directions
  (undirected).

  Connectivity guarantee:
    After KNN construction, check graph is fully connected. If disconnected
    components exist, add minimum RAG edges (pixels sharing a boundary) between
    disconnected components. Log bridging frequency.
    If bridging occurs in >20% of tiles, increase K from 5 to 7.

Blueprint §12 Step 6 — Edge Weights:
  w(i,j) = 0.6 × avg(H_final_i, H_final_j)
         + 0.25 × norm_centroid_distance(i,j)
         + 0.15 × |S_i − S_j|

Exports:
    build_physics_knn_edges   — Step 5: KNN edge construction + connectivity
    compute_edge_weights      — Step 6: edge weight computation
    build_rag_edges           — fallback RAG edge construction for bridging

Usage:
    from src.graph.edges import build_physics_knn_edges, compute_edge_weights

    edge_index, bridged = build_physics_knn_edges(features, centroids, labels, K=5)
    edge_attr = compute_edge_weights(edge_index, features, centroids, H, W)
"""

import logging
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 5 — Physics-Similarity KNN Edge Construction
# ---------------------------------------------------------------------------

def _find_connected_components(N: int, edge_index: np.ndarray) -> list[set[int]]:
    """Find connected components via BFS on undirected edge list.

    Parameters
    ----------
    N : int
        Number of nodes.
    edge_index : (2, E) int64
        COO-format edge list.

    Returns
    -------
    List of sets, each containing node indices of one connected component.
    """
    adj = {i: set() for i in range(N)}
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        adj[u].add(v)
        adj[v].add(u)

    visited = set()
    components = []

    for start in range(N):
        if start in visited:
            continue
        component = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for nb in adj[node]:
                if nb not in visited:
                    queue.append(nb)
        components.append(component)

    return components


def build_rag_edges(
    labels: np.ndarray,
) -> list[tuple[int, int]]:
    """Build Region Adjacency Graph edges from pixel-level label map.

    Two superpixels are adjacent if they share at least one pixel boundary
    (4-connectivity).

    Parameters
    ----------
    labels : (H, W) int32
        Superpixel label map.

    Returns
    -------
    List of (node_i, node_j) pairs where i < j.
    """
    H, W = labels.shape
    edges = set()

    for y in range(H):
        for x in range(W):
            current = labels[y, x]
            # Check right neighbour
            if x + 1 < W:
                right = labels[y, x + 1]
                if current != right:
                    edge = (min(current, right), max(current, right))
                    edges.add(edge)
            # Check bottom neighbour
            if y + 1 < H:
                bottom = labels[y + 1, x]
                if current != bottom:
                    edge = (min(current, bottom), max(current, bottom))
                    edges.add(edge)

    return list(edges)


def build_physics_knn_edges(
    features: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
    K: int = 5,
    spatial_weight: float = 0.5,
    physics_weight: float = 0.5,
) -> tuple[np.ndarray, bool]:
    """Build physics-similarity KNN edges with connectivity guarantee.

    Blueprint §12 Step 5.

    Parameters
    ----------
    features : (N, 14) float32
        Node feature matrix.
    centroids : (N, 2) float32
        Centroid (y, x) in pixel coords.
    labels : (H, W) int32
        Superpixel label map (for RAG fallback).
    K : int
        Number of nearest neighbours (default: 5).
    spatial_weight : float
        Weight for spatial distance (default: 0.5).
    physics_weight : float
        Weight for physics distance (default: 0.5).

    Returns
    -------
    edge_index : (2, E) int64
        COO-format edges (undirected: both directions included).
    bridged : bool
        True if RAG bridging was needed to connect disconnected components.
    """
    N = features.shape[0]

    if N <= 1:
        return np.zeros((2, 0), dtype=np.int64), False

    # --- Spatial distance ---
    H, W = labels.shape
    tile_diagonal = np.sqrt(H ** 2 + W ** 2)

    # Normalise centroids by tile diagonal
    centroids_norm = centroids / max(tile_diagonal, 1e-8)

    # --- Physics distance ---
    # Physics sub-vector: [S, R, D, H_physics] = features[:, 2:6]
    physics_vec = features[:, 2:6].copy()

    # Normalise by max pairwise physics distance
    if N > 1:
        # Compute max pairwise distance efficiently using min/max range
        phys_range = physics_vec.max(axis=0) - physics_vec.min(axis=0)
        max_phys_dist = np.sqrt((phys_range ** 2).sum())
        if max_phys_dist > 1e-8:
            physics_vec_norm = physics_vec / max_phys_dist
        else:
            physics_vec_norm = physics_vec
    else:
        physics_vec_norm = physics_vec

    # --- Combined distance for KNN ---
    # Concatenate scaled spatial + physics into a single vector for KDTree
    # We scale each by its weight so Euclidean distance in joint space
    # approximates: weight_s * d_spatial + weight_p * d_physics
    spatial_scaled = centroids_norm * np.sqrt(spatial_weight)
    physics_scaled = physics_vec_norm * np.sqrt(physics_weight)
    combined_vec = np.hstack([spatial_scaled, physics_scaled])  # (N, 6)

    # --- KNN via KDTree ---
    tree = cKDTree(combined_vec)
    # Query K+1 because the first result is the node itself (distance=0)
    k_query = min(K + 1, N)
    _, indices = tree.query(combined_vec, k=k_query)

    # Build edge set (undirected)
    edge_set = set()
    for i in range(N):
        for j_idx in range(k_query):
            j = indices[i, j_idx]
            if i != j:
                # Add both directions
                edge_set.add((i, j))
                edge_set.add((j, i))

    # Convert to COO
    if edge_set:
        edges = np.array(list(edge_set), dtype=np.int64).T  # (2, E)
    else:
        edges = np.zeros((2, 0), dtype=np.int64)

    # --- Connectivity guarantee ---
    components = _find_connected_components(N, edges)
    bridged = False

    if len(components) > 1:
        log.info(
            "Graph has %d disconnected components — adding RAG bridge edges.",
            len(components),
        )
        bridged = True

        # Build RAG for bridging
        rag_edges = build_rag_edges(labels)

        # For each pair of components, find the RAG edge that connects them
        node_to_component = {}
        for comp_idx, comp in enumerate(components):
            for node in comp:
                node_to_component[node] = comp_idx

        bridge_edges = set()
        for u, v in rag_edges:
            comp_u = node_to_component.get(u, -1)
            comp_v = node_to_component.get(v, -1)
            if comp_u != comp_v and comp_u >= 0 and comp_v >= 0:
                bridge_edges.add((u, v))
                bridge_edges.add((v, u))

        # If RAG didn't bridge all components, add direct edges between
        # nearest nodes of each component pair
        if bridge_edges:
            bridge_arr = np.array(list(bridge_edges), dtype=np.int64).T
            edges = np.hstack([edges, bridge_arr]) if edges.shape[1] > 0 else bridge_arr
        else:
            # Fallback: connect nearest nodes between components
            for i in range(1, len(components)):
                comp_a = list(components[0])
                comp_b = list(components[i])

                # Find nearest pair
                centroids_a = centroids[comp_a]
                centroids_b = centroids[comp_b]
                tree_a = cKDTree(centroids_a)
                dists, idxs = tree_a.query(centroids_b, k=1)
                best_b_local = np.argmin(dists)
                best_a_local = idxs[best_b_local]

                u = comp_a[best_a_local]
                v = comp_b[best_b_local]
                new_edges = np.array([[u, v], [v, u]], dtype=np.int64).T
                edges = np.hstack([edges, new_edges]) if edges.shape[1] > 0 else new_edges

                # Merge components for subsequent iterations
                components[0] = components[0] | components[i]

        # Verify connectivity after bridging
        components_after = _find_connected_components(N, edges)
        if len(components_after) > 1:
            log.warning(
                "Graph STILL has %d components after bridging. "
                "Consider increasing K from %d to %d.",
                len(components_after), K, K + 2,
            )

    return edges, bridged


# ---------------------------------------------------------------------------
# Step 6 — Edge Weights
# ---------------------------------------------------------------------------

def compute_edge_weights(
    edge_index: np.ndarray,
    features: np.ndarray,
    centroids: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """Compute edge weights for all edges.

    Blueprint §12 Step 6:
      w(i,j) = 0.6 × avg(H_final_i, H_final_j)
             + 0.25 × norm_centroid_distance(i,j)
             + 0.15 × |S_i − S_j|

    Parameters
    ----------
    edge_index : (2, E) int64
        COO-format edge list.
    features : (N, 14) float32
        Node feature matrix.
    centroids : (N, 2) float32
        Centroid (y, x) in pixel coords.
    H, W : int
        Tile dimensions (for distance normalisation).

    Returns
    -------
    edge_weights : (E,) float32
        Weight per edge.
    """
    E = edge_index.shape[1]
    if E == 0:
        return np.zeros(0, dtype=np.float32)

    src = edge_index[0]  # (E,)
    dst = edge_index[1]  # (E,)

    # H_final is at feature index 7
    h_final_src = features[src, 7]
    h_final_dst = features[dst, 7]
    avg_risk = 0.5 * (h_final_src + h_final_dst)

    # Normalised centroid distance
    tile_diagonal = np.sqrt(H ** 2 + W ** 2)
    centroid_diff = centroids[src] - centroids[dst]  # (E, 2)
    centroid_dist = np.sqrt((centroid_diff ** 2).sum(axis=1))
    norm_dist = centroid_dist / max(tile_diagonal, 1e-8)

    # Slope discontinuity: |S_i - S_j| where S is at feature index 2
    slope_diff = np.abs(features[src, 2] - features[dst, 2])

    # Blueprint formula
    weights = 0.6 * avg_risk + 0.25 * norm_dist + 0.15 * slope_diff

    return weights.astype(np.float32)
