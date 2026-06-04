"""
adaptive_slic.py
----------------
Stage 5 — Step 1–3: Terrain-Complexity-Adaptive SLIC Segmentation.

Blueprint §12:

  Step 1 — Compute Terrain Complexity Map:
    Divide 512×512 tile into 16×16 grid of 32×32 blocks.
    Each block's complexity = mean(H_physics) across pixels in that block.

  Step 2 — Assign Node Budget Per Block:
    Flat   (complexity < 0.25):  5 nodes/block
    Complex (0.25 ≤ complexity ≤ 0.60): 15 nodes/block
    Hazard (complexity > 0.60):  30–50 nodes/block (linear scale)

  Step 3 — Adaptive SLIC Segmentation:
    First pass:  coarse SLIC with n_segments = max(80, floor(total_budget × 0.4))
    Second pass: for each hazard-tier superpixel with >200 pixels, re-run SLIC
                 on that superpixel's region to refine.
    Connectivity guarantee: every pixel assigned to exactly one superpixel.

  Expected total nodes: 120–200 (flat), 300–350 (average), 450–700+ (hazardous).

Exports:
    compute_terrain_complexity  — Step 1: block-level complexity from H_physics
    assign_tier_budget          — Step 2: per-block tier + node budget
    adaptive_slic_segmentation  — Step 3: full adaptive segmentation pipeline
    TIER_FLAT, TIER_COMPLEX, TIER_HAZARD — tier constants

Usage:
    from src.graph.adaptive_slic import adaptive_slic_segmentation

    labels, tier_map, stats = adaptive_slic_segmentation(
        h_physics_np,  # (512, 512) float32 in [0,1]
        image_np,      # (512, 512) float32 in [0,1] — grayscale
    )
    # labels:   (512, 512) int32 — superpixel label per pixel
    # tier_map: (16, 16)   int32 — tier per 32×32 block
    # stats:    dict with total_nodes, tier_counts, budget_total
"""

import logging
from typing import NamedTuple

import numpy as np
from skimage.segmentation import slic
from skimage.measure import label as connected_components

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier constants (used in node_features.py and graph_builder.py)
# ---------------------------------------------------------------------------

TIER_FLAT:    int = 0
TIER_COMPLEX: int = 1
TIER_HAZARD:  int = 2


# ---------------------------------------------------------------------------
# Tier thresholds — configurable for ablation (blueprint §20)
# ---------------------------------------------------------------------------

DEFAULT_FLAT_THRESHOLD:    float = 0.25
DEFAULT_HAZARD_THRESHOLD:  float = 0.60

# Hazard-tier node budget linear range
HAZARD_NODES_MIN: int = 30
HAZARD_NODES_MAX: int = 50

# Refinement threshold: only refine superpixels larger than this
REFINE_PIXEL_THRESHOLD: int = 200

# Block dimensions
BLOCK_SIZE: int = 32
GRID_SIZE:  int = 16    # 512 / 32 = 16


# ---------------------------------------------------------------------------
# Step 1 — Compute Terrain Complexity Map
# ---------------------------------------------------------------------------

def compute_terrain_complexity(
    h_physics: np.ndarray,
    block_size: int = BLOCK_SIZE,
) -> np.ndarray:
    """Compute per-block terrain complexity from H_physics.

    Blueprint §12 Step 1:
      Divide 512×512 tile into non-overlapping 32×32 blocks (16×16 grid).
      Each block's complexity score = mean(H_physics) across all pixels.

    Parameters
    ----------
    h_physics : (H, W) float32 in [0, 1]
        Physics risk map from Stage 2.
    block_size : int
        Block side length (default: 32).

    Returns
    -------
    complexity : (grid_h, grid_w) float32
        Mean H_physics per block.  Values in [0, 1].
    """
    H, W = h_physics.shape
    grid_h = H // block_size
    grid_w = W // block_size

    # Reshape into (grid_h, block_size, grid_w, block_size) and average
    # Handle tiles that aren't exactly divisible (truncate excess)
    h_trunc = grid_h * block_size
    w_trunc = grid_w * block_size
    blocks = h_physics[:h_trunc, :w_trunc].reshape(
        grid_h, block_size, grid_w, block_size
    )
    complexity = blocks.mean(axis=(1, 3))  # (grid_h, grid_w)

    return complexity.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 2 — Assign Node Budget Per Block
# ---------------------------------------------------------------------------

class TierBudget(NamedTuple):
    """Per-block tier assignment and node budget."""
    tier_map: np.ndarray        # (grid_h, grid_w) int32 — 0/1/2
    budget_map: np.ndarray      # (grid_h, grid_w) int32 — nodes per block
    total_budget: int           # sum of all block budgets


def continuous_node_budget(
    complexity: np.ndarray,
    n_min: int = 8,
    n_max: int = 64,
    gamma: float = 1.5,
) -> np.ndarray:
    """Continuous power-law node budget allocation.

    N_b = N_min + (N_max - N_min) * H_physics^gamma

    This is the PRIMARY allocation method — the core scientific contribution.
    Replaces discrete tier thresholds with a smooth, physically-motivated
    density function. Higher gamma concentrates more nodes in high-complexity
    terrain.

    Parameters
    ----------
    complexity : (grid_h, grid_w) float32 in [0, 1]
    n_min      : minimum nodes per block (flat terrain)
    n_max      : maximum nodes per block (extreme hazard)
    gamma      : power-law exponent (1.0 = linear, >1 = concave, <1 = convex)

    Returns
    -------
    budget_map : (grid_h, grid_w) int32
    """
    # Clamp complexity to [0, 1] for safety
    c = np.clip(complexity, 0.0, 1.0)
    budget = n_min + (n_max - n_min) * np.power(c, gamma)
    return np.round(budget).astype(np.int32)


def _assign_tier_labels(
    complexity: np.ndarray,
    flat_threshold: float = DEFAULT_FLAT_THRESHOLD,
    hazard_threshold: float = DEFAULT_HAZARD_THRESHOLD,
) -> np.ndarray:
    """Assign tier labels (for evaluation stratification only).

    Post-hoc tier assignment used for tier-stratified metrics,
    NOT for node budget allocation.
    """
    tier_map = np.full(complexity.shape, TIER_COMPLEX, dtype=np.int32)
    tier_map[complexity < flat_threshold] = TIER_FLAT
    tier_map[complexity > hazard_threshold] = TIER_HAZARD
    return tier_map


def assign_tier_budget(
    complexity: np.ndarray,
    flat_threshold: float = DEFAULT_FLAT_THRESHOLD,
    hazard_threshold: float = DEFAULT_HAZARD_THRESHOLD,
    allocation_mode: str = "continuous",
    n_min: int = 8,
    n_max: int = 64,
    gamma: float = 1.5,
) -> TierBudget:
    """Map block complexity to tier assignment and node budget.

    Supports two allocation modes:
      - "continuous" (PRIMARY): N_b = N_min + (N_max - N_min) * H^gamma
        Smooth, physically-motivated density. Core scientific contribution.
      - "discrete"  (ABLATION): Original 3-tier heuristic from blueprint §12
        Flat: 5, Complex: 15, Hazard: 30-50 (linear)

    Tier labels are always assigned for evaluation stratification regardless
    of allocation mode.

    Parameters
    ----------
    complexity       : (grid_h, grid_w) float32
    flat_threshold   : float (default 0.25, ablation §20)
    hazard_threshold : float (default 0.60, ablation §20)
    allocation_mode  : "continuous" or "discrete"
    n_min            : min nodes/block for continuous mode (default 8)
    n_max            : max nodes/block for continuous mode (default 64)
    gamma            : power-law exponent for continuous mode (default 1.5)

    Returns
    -------
    TierBudget with tier_map, budget_map, total_budget
    """
    # Tier labels (always computed for stratified evaluation)
    tier_map = _assign_tier_labels(complexity, flat_threshold, hazard_threshold)

    if allocation_mode == "continuous":
        # --- PRIMARY: Continuous power-law allocation ---
        budget_map = continuous_node_budget(complexity, n_min, n_max, gamma)
    elif allocation_mode == "discrete":
        # --- ABLATION: Original discrete 3-tier system ---
        budget_map = np.full(complexity.shape, 15, dtype=np.int32)

        # Flat tier
        flat_mask = complexity < flat_threshold
        budget_map[flat_mask] = 5

        # Hazard tier — linear interpolation from 30 at threshold to 50 at 1.0
        hazard_mask = complexity > hazard_threshold
        range_denom = max(1.0 - hazard_threshold, 1e-6)
        hazard_scores = complexity[hazard_mask]
        linear_budget = HAZARD_NODES_MIN + (
            (hazard_scores - hazard_threshold) / range_denom
        ) * (HAZARD_NODES_MAX - HAZARD_NODES_MIN)
        budget_map[hazard_mask] = np.clip(
            linear_budget.astype(np.int32), HAZARD_NODES_MIN, HAZARD_NODES_MAX
        )
    else:
        raise ValueError(
            f"allocation_mode must be 'continuous' or 'discrete', got '{allocation_mode}'"
        )

    total_budget = int(budget_map.sum())

    return TierBudget(
        tier_map=tier_map,
        budget_map=budget_map,
        total_budget=total_budget,
    )


# ---------------------------------------------------------------------------
# Step 3 — Adaptive SLIC Segmentation
# ---------------------------------------------------------------------------

def _relabel_contiguous(labels: np.ndarray) -> np.ndarray:
    """Relabel a segmentation map to contiguous integers starting from 0."""
    unique_labels = np.unique(labels)
    remap = np.zeros(unique_labels.max() + 1, dtype=np.int32) - 1
    for new_id, old_id in enumerate(unique_labels):
        remap[old_id] = new_id
    return remap[labels]


def _ensure_connectivity(labels: np.ndarray) -> np.ndarray:
    """Ensure every superpixel is spatially connected.

    Blueprint §12 Step 3 connectivity guarantee:
      After construction, verify the superpixel map is fully connected.
      Run label cleaning if needed.

    For each superpixel ID, check if it forms one connected component.
    If a superpixel has disconnected fragments, merge each fragment
    into its nearest spatial neighbour superpixel.
    """
    cleaned = labels.copy()
    unique_ids = np.unique(cleaned)

    for sp_id in unique_ids:
        mask = (cleaned == sp_id)
        cc_labels, n_components = connected_components(
            mask, return_num=True, connectivity=1
        )
        if n_components <= 1:
            continue

        # Keep the largest component, reassign smaller ones
        component_sizes = []
        for cc_id in range(1, n_components + 1):
            component_sizes.append((cc_id, (cc_labels == cc_id).sum()))
        component_sizes.sort(key=lambda x: x[1], reverse=True)
        largest_cc = component_sizes[0][0]

        for cc_id, _ in component_sizes[1:]:
            fragment_mask = (cc_labels == cc_id)
            # Find the most common neighbouring superpixel ID
            # (excluding the current superpixel)
            ys, xs = np.where(fragment_mask)
            neighbour_ids = []
            for y, x in zip(ys, xs):
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < cleaned.shape[0] and 0 <= nx < cleaned.shape[1]:
                        nid = cleaned[ny, nx]
                        if nid != sp_id:
                            neighbour_ids.append(nid)
            if neighbour_ids:
                # Merge into the most common neighbour
                merge_target = max(set(neighbour_ids), key=neighbour_ids.count)
                cleaned[fragment_mask] = merge_target

    return _relabel_contiguous(cleaned)


def adaptive_slic_segmentation(
    h_physics: np.ndarray,
    image: np.ndarray,
    flat_threshold: float = DEFAULT_FLAT_THRESHOLD,
    hazard_threshold: float = DEFAULT_HAZARD_THRESHOLD,
    allocation_mode: str = "continuous",
    gamma: float = 1.5,
    n_min: int = 8,
    n_max: int = 64,
    compactness: float = 10.0,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Full adaptive SLIC segmentation pipeline.

    Blueprint §12 Step 3:
      First pass:  SLIC with n_segments = max(80, floor(total_budget × 0.4))
      Second pass: refine hazard-zone superpixels with >200 pixels

    Parameters
    ----------
    h_physics : (H, W) float32 in [0, 1]
        Physics risk map (Stage 2 output).
    image : (H, W) float32 in [0, 1]
        Original grayscale image tile.
    flat_threshold : float
        Below this → flat tier (default 0.25).
    hazard_threshold : float
        Above this → hazard tier (default 0.60).
    allocation_mode : str
        "continuous" or "discrete".
    gamma : float
        Power-law exponent for continuous mode.
    n_min : int
        Min nodes per block.
    n_max : int
        Max nodes per block.
    compactness : float
        SLIC compactness parameter (higher = more regular shape).
    sigma : float
        SLIC Gaussian smoothing sigma.

    Returns
    -------
    labels : (H, W) int32
        Superpixel label per pixel, contiguous from 0.
    tier_map : (grid_h, grid_w) int32
        Tier assignment per 32×32 block.
    stats : dict
        total_nodes, tier_counts (flat/complex/hazard count),
        budget_total, n_refined, bridged_fragments.
    """
    H, W = h_physics.shape

    # --- Step 1: Terrain complexity ---
    complexity = compute_terrain_complexity(h_physics)

    # --- Step 2: Tier + budget ---
    tier_result = assign_tier_budget(
        complexity,
        flat_threshold=flat_threshold,
        hazard_threshold=hazard_threshold,
        allocation_mode=allocation_mode,
        gamma=gamma,
        n_min=n_min,
        n_max=n_max,
    )
    tier_map = tier_result.tier_map
    budget_map = tier_result.budget_map
    total_budget = tier_result.total_budget

    # --- Step 3a: First pass — coarse SLIC ---
    n_segments_coarse = max(80, int(total_budget * 0.4))

    # SLIC expects (H, W) or (H, W, C) with values in [0, 1]
    labels = slic(
        image,
        n_segments=n_segments_coarse,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        enforce_connectivity=True,
        channel_axis=None,  # single-channel
    ).astype(np.int32)

    # --- Step 3b: Second pass — refine hazard-zone superpixels ---
    n_refined = 0
    grid_h, grid_w = tier_map.shape
    next_label = labels.max() + 1

    for bi in range(grid_h):
        for bj in range(grid_w):
            if tier_map[bi, bj] != TIER_HAZARD:
                continue

            # Block pixel bounds
            y0, y1 = bi * BLOCK_SIZE, min((bi + 1) * BLOCK_SIZE, H)
            x0, x1 = bj * BLOCK_SIZE, min((bj + 1) * BLOCK_SIZE, W)

            # Find superpixels whose centroid falls within this block
            block_labels = labels[y0:y1, x0:x1]
            unique_sp_in_block = np.unique(block_labels)

            for sp_id in unique_sp_in_block:
                sp_mask_global = (labels == sp_id)
                sp_pixel_count = sp_mask_global.sum()

                # Check centroid is in this block
                ys, xs = np.where(sp_mask_global)
                cy, cx = ys.mean(), xs.mean()
                if not (y0 <= cy < y1 and x0 <= cx < x1):
                    continue

                # Only refine if too coarse for hazard zone
                if sp_pixel_count <= REFINE_PIXEL_THRESHOLD:
                    continue

                # Compute this superpixel's share of the block's node budget
                block_budget = budget_map[bi, bj]
                n_sp_in_block = len(unique_sp_in_block)
                sub_segments = max(2, block_budget // max(n_sp_in_block, 1))

                # Extract the bounding box of this superpixel for efficient SLIC
                y_min, y_max = ys.min(), ys.max() + 1
                x_min, x_max = xs.min(), xs.max() + 1

                # Create a masked image region for SLIC
                region_img = image[y_min:y_max, x_min:x_max].copy()
                region_mask = sp_mask_global[y_min:y_max, x_min:x_max]

                # Only run SLIC if the region is large enough
                if region_img.size < sub_segments * 4:
                    continue

                try:
                    sub_labels = slic(
                        region_img,
                        n_segments=sub_segments,
                        compactness=compactness * 1.5,  # slightly more compact for sub-regions
                        sigma=sigma * 0.5,
                        start_label=0,
                        enforce_connectivity=True,
                        channel_axis=None,
                        mask=region_mask,
                    ).astype(np.int32)
                except (ValueError, RuntimeError):
                    # SLIC can fail on very small or degenerate regions — skip
                    continue

                # Merge refined sub-superpixels back into global label map
                sub_unique = np.unique(sub_labels[region_mask])
                sub_unique = sub_unique[sub_unique >= 0]

                if len(sub_unique) <= 1:
                    continue  # refinement produced no split

                # Remap sub-labels to new global IDs
                for sub_id in sub_unique:
                    sub_mask_local = (sub_labels == sub_id) & region_mask
                    if sub_mask_local.sum() == 0:
                        continue

                    # Place into global labels
                    full_sub_mask = np.zeros_like(labels, dtype=bool)
                    full_sub_mask[y_min:y_max, x_min:x_max] = sub_mask_local
                    labels[full_sub_mask] = next_label
                    next_label += 1

                n_refined += 1

    # --- Connectivity guarantee ---
    labels = _ensure_connectivity(labels)

    # --- Final relabel to contiguous ---
    labels = _relabel_contiguous(labels)

    # --- Collect statistics ---
    total_nodes = len(np.unique(labels))
    n_flat    = int((tier_map == TIER_FLAT).sum())
    n_complex = int((tier_map == TIER_COMPLEX).sum())
    n_hazard  = int((tier_map == TIER_HAZARD).sum())

    stats = {
        "total_nodes": total_nodes,
        "tier_counts": {
            "flat": n_flat,
            "complex": n_complex,
            "hazard": n_hazard,
        },
        "budget_total": total_budget,
        "n_refined": n_refined,
        "n_segments_coarse": n_segments_coarse,
    }

    log.debug(
        "Adaptive SLIC: %d nodes (budget=%d, refined=%d sp) | "
        "tiers: flat=%d, complex=%d, hazard=%d",
        total_nodes, total_budget, n_refined,
        n_flat, n_complex, n_hazard,
    )

    return labels, tier_map, stats
