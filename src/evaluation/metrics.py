"""
metrics.py
----------
All evaluation metrics used across the three evaluation contexts.

Blueprint §19 (Evaluation Protocol):
  - Hazard Crossing Rate (HCR) — path safety metric
  - Path Length Ratio (PLR) — efficiency metric
  - Hazard Recall, Precision, F1, mIoU — segmentation metrics
  - AUC-ROC — GNN binary classification quality
  - MAE — GNN regression quality
  - Expected Calibration Error (ECE) — risk score calibration
  - Tier-stratified HCR — per-tier path quality breakdown

Blueprint §20 (Ablation Study): tier-stratified hazard recall.
Blueprint §13: AUC-ROC, MAE for GNN validation.

All metrics accept numpy arrays or torch tensors unless stated otherwise.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Array = np.ndarray | torch.Tensor


def _to_numpy(x: Array) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# §10 / §19 — Pixel-level segmentation metrics
# ---------------------------------------------------------------------------

def hazard_recall(
    pred: Array,
    target: Array,
    validity: Array | None = None,
    hazard_threshold: float = 0.7,
    pred_threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Fraction of ground-truth hazardous pixels correctly predicted hazardous.

    Primary safety metric. Blueprint §19.

    Parameters
    ----------
    pred      : predicted risk map [0,1], any shape
    target    : DEM-derived risk label [0,1], same shape
    validity  : binary mask {0,1} — 1=valid pixel, None=all valid
    hazard_threshold : target threshold to define hazardous ground-truth
    pred_threshold   : prediction threshold to binarise model output

    Returns
    -------
    float in [0, 1]
    """
    p = _to_numpy(pred).ravel()
    t = _to_numpy(target).ravel()

    if validity is not None:
        v = _to_numpy(validity).ravel().astype(bool)
        p, t = p[v], t[v]

    pred_bin  = p > pred_threshold
    target_haz = t > hazard_threshold

    tp = (pred_bin & target_haz).sum()
    fn = (~pred_bin & target_haz).sum()
    return float(tp / (tp + fn + eps))


def hazard_precision(
    pred: Array,
    target: Array,
    validity: Array | None = None,
    hazard_threshold: float = 0.7,
    pred_threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Fraction of predicted hazardous pixels that are truly hazardous."""
    p = _to_numpy(pred).ravel()
    t = _to_numpy(target).ravel()

    if validity is not None:
        v = _to_numpy(validity).ravel().astype(bool)
        p, t = p[v], t[v]

    pred_bin   = p > pred_threshold
    target_haz = t > hazard_threshold

    tp = (pred_bin & target_haz).sum()
    fp = (pred_bin & ~target_haz).sum()
    return float(tp / (tp + fp + eps))


def mean_iou(
    pred: Array,
    target: Array,
    validity: Array | None = None,
    hazard_threshold: float = 0.7,
    pred_threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Mean IoU over hazard and safe classes. Blueprint §19."""
    p = _to_numpy(pred).ravel()
    t = _to_numpy(target).ravel()

    if validity is not None:
        v = _to_numpy(validity).ravel().astype(bool)
        p, t = p[v], t[v]

    pred_bin   = p > pred_threshold
    target_haz = t > hazard_threshold

    tp = (pred_bin & target_haz).sum()
    fp = (pred_bin & ~target_haz).sum()
    fn = (~pred_bin & target_haz).sum()
    tn = (~pred_bin & ~target_haz).sum()

    iou_haz  = tp  / (tp + fp + fn + eps)
    iou_safe = tn  / (tn + fn + fp + eps)
    return float((iou_haz + iou_safe) / 2.0)


def segmentation_metrics(
    pred: Array,
    target: Array,
    validity: Array | None = None,
    hazard_threshold: float = 0.7,
    pred_threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Compute all segmentation metrics in one pass.

    Returns
    -------
    dict with keys: hazard_recall, hazard_precision, hazard_f1,
                    safe_recall, mIoU
    """
    p = _to_numpy(pred).ravel()
    t = _to_numpy(target).ravel()

    if validity is not None:
        v = _to_numpy(validity).ravel().astype(bool)
        p, t = p[v], t[v]

    pred_bin   = p > pred_threshold
    target_haz = t > hazard_threshold

    tp = float((pred_bin & target_haz).sum())
    fp = float((pred_bin & ~target_haz).sum())
    fn = float((~pred_bin & target_haz).sum())
    tn = float((~pred_bin & ~target_haz).sum())

    recall    = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    f1        = 2 * tp / (2 * tp + fp + fn + eps)
    safe_rec  = tn / (tn + fp + eps)
    iou_haz   = tp / (tp + fp + fn + eps)
    iou_safe  = tn / (tn + fn + fp + eps)
    miou      = (iou_haz + iou_safe) / 2.0

    return {
        "hazard_recall":    recall,
        "hazard_precision": precision,
        "hazard_f1":        f1,
        "safe_recall":      safe_rec,
        "mIoU":             miou,
    }


# ---------------------------------------------------------------------------
# §19 — Path planning metrics
# ---------------------------------------------------------------------------

def hazard_crossing_rate(
    waypoint_coords: list[tuple[int, int]],
    hazard_map: Array,
    hazard_threshold: float = 0.7,
) -> float:
    """Fraction of waypoints in DEM-labelled hazardous terrain.

    Blueprint §19: Target < 5%.

    Parameters
    ----------
    waypoint_coords : list of (row, col) integer pixel positions
    hazard_map      : (H, W) array — DEM-derived risk or H_final map
    hazard_threshold: threshold above which a pixel is hazardous

    Returns
    -------
    float in [0, 1]
    """
    if not waypoint_coords:
        return 0.0

    hmap = _to_numpy(hazard_map)
    n_hazard = 0
    for r, c in waypoint_coords:
        r = int(np.clip(r, 0, hmap.shape[0] - 1))
        c = int(np.clip(c, 0, hmap.shape[1] - 1))
        if hmap[r, c] > hazard_threshold:
            n_hazard += 1
    return n_hazard / len(waypoint_coords)


def path_length_ratio(
    waypoint_coords: list[tuple[int, int]],
) -> float:
    """Path length / straight-line start-goal distance.

    Blueprint §19: Target < 1.35.

    Parameters
    ----------
    waypoint_coords : ordered list of (row, col) integer positions,
                      including start and goal.

    Returns
    -------
    float >= 1.0 (or 0.0 if fewer than 2 waypoints)
    """
    if len(waypoint_coords) < 2:
        return 0.0

    # Actual path length (sum of Euclidean steps)
    path_len = 0.0
    for (r1, c1), (r2, c2) in zip(waypoint_coords[:-1], waypoint_coords[1:]):
        path_len += np.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2)

    # Straight-line distance
    r_start, c_start = waypoint_coords[0]
    r_goal,  c_goal  = waypoint_coords[-1]
    straight = np.sqrt((r_goal - r_start) ** 2 + (c_goal - c_start) ** 2)

    if straight < 1e-8:
        return 1.0
    return path_len / straight


def tier_stratified_hcr(
    waypoint_nodes: list[dict],
    hazard_threshold: float = 0.7,
) -> dict[str, float]:
    """HCR broken down by tier (flat / complex / hazard).

    Blueprint §19 (Tier-stratified HCR).

    Parameters
    ----------
    waypoint_nodes : list of node dicts, each with:
        - "tier"    : int  0=flat, 1=complex, 2=hazard
        - "risk"    : float  GNN p̂_i
        - "coords"  : (row, col)

    Returns
    -------
    dict with keys: "flat_hcr", "complex_hcr", "hazard_hcr", "overall_hcr"
    """
    by_tier: dict[int, list[float]] = {0: [], 1: [], 2: []}
    for node in waypoint_nodes:
        tier = int(node.get("tier", 0))
        risk = float(node.get("risk", 0.0))
        by_tier.get(tier, by_tier[0]).append(risk)

    def _hcr(risks: list[float]) -> float:
        if not risks:
            return 0.0
        return sum(1 for r in risks if r > hazard_threshold) / len(risks)

    all_risks = [n.get("risk", 0.0) for n in waypoint_nodes]
    return {
        "flat_hcr":    _hcr(by_tier[0]),
        "complex_hcr": _hcr(by_tier[1]),
        "hazard_hcr":  _hcr(by_tier[2]),
        "overall_hcr": _hcr(all_risks),
    }


# ---------------------------------------------------------------------------
# §19 — GNN regression / classification metrics
# ---------------------------------------------------------------------------

def node_mae(pred: Array, target: Array) -> float:
    """Mean Absolute Error for GNN node risk regression.

    Blueprint §19 (GNN Metrics). Target: MAE decreasing below 0.15.
    """
    p = _to_numpy(pred).ravel()
    t = _to_numpy(target).ravel()
    return float(np.abs(p - t).mean())


def node_auc_roc(
    pred: Array,
    target: Array,
    hazard_threshold: float = 0.7,
) -> float:
    """AUC-ROC for binary hazard classification from GNN node scores.

    Blueprint §19 (GNN Metrics).

    Parameters
    ----------
    pred   : (N,) predicted risk scores per node
    target : (N,) DEM-derived risk labels per node

    Returns
    -------
    float AUC-ROC or 0.0 if only one class present.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")

    p = _to_numpy(pred).ravel()
    t = _to_numpy(target).ravel()
    labels = (t > hazard_threshold).astype(int)

    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.0

    try:
        return float(roc_auc_score(labels, p))
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# §19 — Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

def expected_calibration_error(
    pred: Array,
    target: Array,
    n_bins: int = 10,
    hazard_threshold: float = 0.7,
) -> float:
    """10-bin ECE for risk score calibration.

    Blueprint §19 (ECE). Lower is better.

    Parameters
    ----------
    pred   : (N,) predicted risk scores in [0,1]
    target : (N,) ground-truth risk labels in [0,1]
    n_bins : number of calibration bins (blueprint: 10)

    Returns
    -------
    float ECE in [0, 1]
    """
    p = _to_numpy(pred).ravel()
    t = (_to_numpy(target).ravel() > hazard_threshold).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(p)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        bin_conf = p[mask].mean()
        bin_acc  = t[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)

    return float(ece)


# ---------------------------------------------------------------------------
# §19 — Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval.

    Blueprint §19: 95% bootstrap CI for HCR comparisons.

    Parameters
    ----------
    values     : list/array of per-tile metric values
    n_bootstrap: number of resamples
    confidence : confidence level (default 0.95)
    seed       : random seed

    Returns
    -------
    (lower, upper) confidence interval bounds
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = 1.0 - confidence
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# §18 / §20 — GNN delta metric
# ---------------------------------------------------------------------------

def gnn_delta(hcr_no_gnn: float, hcr_proposed: float) -> float:
    """GNN delta = HCR_no-GNN − HCR_proposed.

    Blueprint §19 (GNN Metrics). Positive value means GNN reduces HCR.
    """
    return hcr_no_gnn - hcr_proposed


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def aggregate_seed_results(
    per_seed_results: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Aggregate metric dicts from multiple seeds into mean ± std.

    Blueprint §19: run with 3 seeds, report mean ± std.

    Parameters
    ----------
    per_seed_results : list of metric dicts, one per seed run

    Returns
    -------
    dict: {metric_name: {"mean": ..., "std": ...}}
    """
    if not per_seed_results:
        return {}

    all_keys = set()
    for d in per_seed_results:
        all_keys.update(d.keys())

    aggregated = {}
    for key in sorted(all_keys):
        vals = np.array([d[key] for d in per_seed_results if key in d])
        aggregated[key] = {
            "mean": float(vals.mean()),
            "std":  float(vals.std()),
        }
    return aggregated
