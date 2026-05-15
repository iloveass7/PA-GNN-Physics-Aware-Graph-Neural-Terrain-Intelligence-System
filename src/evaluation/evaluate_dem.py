"""
evaluate_dem.py
---------------
In-distribution and OOD DEM evaluation runner.

Blueprint §19 (Evaluation Protocol — In-distribution DEM test and OOD DEM test):
  Runs the full pipeline on held-out DEM tiles and computes all metrics.
  Reports separately for in-distribution (test_in) and out-of-distribution
  (test_ood) splits.

Usage:
    from src.evaluation.evaluate_dem import evaluate_dem_split
    results = evaluate_dem_split("test_in", model, fusion, gnn, physics, cfg)

Or via scripts/evaluate_all.py CLI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import (
    aggregate_seed_results,
    expected_calibration_error,
    hazard_crossing_rate,
    node_auc_roc,
    node_mae,
    path_length_ratio,
    segmentation_metrics,
    tier_stratified_hcr,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tile-level evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_tile(
    image: torch.Tensor,            # (1, 3, H, W) on CPU
    target: torch.Tensor,           # (1, H, W) DEM risk label
    validity: torch.Tensor | None,  # (1, H, W) {0,1}
    pipeline_fn: Callable,          # callable(image) → PipelineResult
    hazard_threshold: float = 0.7,
) -> dict:
    """Run the pipeline on one tile and compute tile-level metrics.

    Parameters
    ----------
    image        : pre-processed image tensor
    target       : DEM risk label tensor
    validity     : DEM validity mask (None = all valid)
    pipeline_fn  : callable that takes image and returns a namespace/dict with
                   .h_final, .path_waypoints, .gnn_node_preds, .gnn_node_labels,
                   .path_waypoint_nodes
    hazard_threshold: threshold for hazard classification

    Returns
    -------
    dict of metric values for this tile
    """
    result = pipeline_fn(image)

    # Pixel-level segmentation
    h_final = result.get("h_final") if isinstance(result, dict) else getattr(result, "h_final", None)
    seg_metrics: dict = {}
    if h_final is not None:
        h_np = h_final.squeeze().cpu().numpy() if isinstance(h_final, torch.Tensor) else h_final
        t_np = target.squeeze().cpu().numpy()
        v_np = validity.squeeze().cpu().numpy() if validity is not None else None
        seg_metrics = segmentation_metrics(h_np, t_np, v_np,
                                           hazard_threshold=hazard_threshold)
        seg_metrics["ece"] = expected_calibration_error(h_np.ravel(), t_np.ravel(),
                                                         hazard_threshold=hazard_threshold)

    # Path metrics
    path_metrics: dict = {}
    waypoints = (result.get("path_waypoints") if isinstance(result, dict)
                 else getattr(result, "path_waypoints", None))
    if waypoints and h_final is not None:
        coords = [(int(w["row"]), int(w["col"])) for w in waypoints
                  if "row" in w and "col" in w]
        h_np = h_final.squeeze().cpu().numpy() if isinstance(h_final, torch.Tensor) else h_final
        path_metrics["hcr"]  = hazard_crossing_rate(coords, h_np, hazard_threshold)
        path_metrics["plr"]  = path_length_ratio(coords)
        path_metrics["path_found"] = 1.0

        # Tier-stratified HCR
        waypoint_nodes = (result.get("path_waypoint_nodes") if isinstance(result, dict)
                          else getattr(result, "path_waypoint_nodes", None))
        if waypoint_nodes:
            strat = tier_stratified_hcr(waypoint_nodes, hazard_threshold)
            path_metrics.update(strat)
    else:
        path_metrics["path_found"] = 0.0
        path_metrics["hcr"]  = float("nan")
        path_metrics["plr"]  = float("nan")

    # GNN node metrics
    gnn_metrics: dict = {}
    gnn_preds  = (result.get("gnn_node_preds") if isinstance(result, dict)
                  else getattr(result, "gnn_node_preds", None))
    gnn_labels = (result.get("gnn_node_labels") if isinstance(result, dict)
                  else getattr(result, "gnn_node_labels", None))
    if gnn_preds is not None and gnn_labels is not None:
        gnn_metrics["gnn_mae"]     = node_mae(gnn_preds, gnn_labels)
        gnn_metrics["gnn_auc_roc"] = node_auc_roc(gnn_preds, gnn_labels,
                                                    hazard_threshold=hazard_threshold)

    return {**seg_metrics, **path_metrics, **gnn_metrics}


# ---------------------------------------------------------------------------
# Split-level evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_dem_split(
    split: str,
    pipeline_fn: Callable,
    dataset,
    device: torch.device,
    hazard_threshold: float = 0.7,
    max_tiles: int | None = None,
) -> dict[str, float]:
    """Evaluate pipeline on an entire DEM split.

    Parameters
    ----------
    split        : "test_in" or "test_ood"
    pipeline_fn  : callable(image_tensor) → dict/namespace with pipeline outputs
    dataset      : PyTorch Dataset yielding dicts with "image", "risk", "validity"
    device       : torch.device
    hazard_threshold: hazard classification threshold
    max_tiles    : if set, limit number of tiles (useful for quick sanity checks)

    Returns
    -------
    dict of mean metric values across the split.
    """
    log.info("Evaluating split '%s' (%d tiles)...", split, len(dataset))

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    all_metrics: list[dict] = []
    n_tiles = 0

    for batch in loader:
        if max_tiles is not None and n_tiles >= max_tiles:
            break

        image    = batch["image"].to(device)
        target   = batch["risk"]
        validity = batch.get("validity")

        try:
            tile_metrics = evaluate_tile(
                image, target, validity, pipeline_fn, hazard_threshold
            )
            all_metrics.append(tile_metrics)
        except Exception as exc:
            log.warning("Tile %d failed: %s", n_tiles, exc)

        n_tiles += 1

    if not all_metrics:
        log.warning("No tiles evaluated for split '%s'", split)
        return {}

    # Aggregate — nanmean to handle tiles where path was not found
    agg: dict[str, float] = {}
    all_keys = set(k for m in all_metrics for k in m)
    for key in sorted(all_keys):
        vals = np.array([m[key] for m in all_metrics if key in m and not
                         (isinstance(m[key], float) and np.isnan(m[key]))])
        if vals.size > 0:
            agg[key] = float(vals.mean())

    # Success rate — fraction where path_found == 1
    path_found_vals = [m.get("path_found", 0.0) for m in all_metrics]
    agg["success_rate"] = float(np.mean(path_found_vals))
    agg["n_tiles"] = float(n_tiles)

    log.info(
        "[%s] HCR=%.4f | PLR=%.4f | recall=%.4f | mIoU=%.4f | "
        "success_rate=%.4f | n=%d",
        split,
        agg.get("hcr", float("nan")),
        agg.get("plr", float("nan")),
        agg.get("hazard_recall", float("nan")),
        agg.get("mIoU", float("nan")),
        agg.get("success_rate", 0.0),
        n_tiles,
    )

    return agg


# ---------------------------------------------------------------------------
# Multi-seed evaluation (blueprint §19)
# ---------------------------------------------------------------------------

def evaluate_dem_multi_seed(
    split: str,
    pipeline_factory: Callable,   # callable(seed) → (pipeline_fn, dataset)
    seeds: list[int],
    device: torch.device,
    hazard_threshold: float = 0.7,
    max_tiles: int | None = None,
) -> dict[str, dict[str, float]]:
    """Run evaluate_dem_split for each seed, return mean ± std.

    Blueprint §19: run with 3 seeds, report mean ± std.

    Parameters
    ----------
    split            : "test_in" or "test_ood"
    pipeline_factory : callable(seed) → (pipeline_fn, dataset)
    seeds            : list of integer seeds (blueprint: [42, 123, 7])
    device           : torch.device

    Returns
    -------
    dict: {metric_name: {"mean": ..., "std": ...}}
    """
    per_seed: list[dict[str, float]] = []

    for seed in seeds:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        pipeline_fn, dataset = pipeline_factory(seed)
        result = evaluate_dem_split(
            split, pipeline_fn, dataset, device, hazard_threshold, max_tiles
        )
        if result:
            per_seed.append(result)

    return aggregate_seed_results(per_seed)
