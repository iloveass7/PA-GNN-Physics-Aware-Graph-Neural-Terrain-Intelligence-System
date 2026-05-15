"""
evaluate_hirise.py
------------------
Zero-shot cross-domain evaluation on HiRISE Map-Proj-v3 dataset.

Blueprint §5.3 / §19 (HiRISE v3 cross-domain evaluation):
  - 73,031 crops total; use only the 10,433 originals (no augments).
  - Each crop is 227×227 — resize to 512×512 with bilinear interpolation.
  - Integer class labels remapped to risk scores via HIRISE_V3_RISK_MAP.
  - System is trained on DEM tiles; NO fine-tuning on HiRISE v3.
  - Measures generalisation across image distributions.

Blueprint §20 (Ablation): hazard recall per tier is reported separately.

Usage:
    from src.evaluation.evaluate_hirise import evaluate_hirise_v3
    results = evaluate_hirise_v3(pipeline_fn, dataset, device)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.label_remap import build_risk_label_tensor, HIRISE_V3_RISK_MAP
from src.evaluation.metrics import (
    aggregate_seed_results,
    expected_calibration_error,
    hazard_crossing_rate,
    node_auc_roc,
    segmentation_metrics,
)

log = logging.getLogger(__name__)

# Blueprint §5.3: original crops are every 7th sample (original + 6 augments)
AUGMENT_FACTOR = 7   # dataset ordering: original then 6 augments


# ---------------------------------------------------------------------------
# Original-only crop selection
# ---------------------------------------------------------------------------

def get_original_indices(total_crops: int, augment_factor: int = AUGMENT_FACTOR) -> list[int]:
    """Return indices of original (non-augmented) crops.

    Blueprint §5.3: Use only the 10,433 originals for evaluation.
    Augmented crops are near-duplicates and inflate metrics.
    Dataset is ordered in blocks of 7 (original then 6 augments).

    Parameters
    ----------
    total_crops    : total number of entries in the dataset (73,031)
    augment_factor : block size (7 = 1 original + 6 augments)

    Returns
    -------
    list of integer indices into the full dataset
    """
    return list(range(0, total_crops, augment_factor))


# ---------------------------------------------------------------------------
# Crop-level evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_crop(
    image: torch.Tensor,         # (1, 3, 512, 512) on device
    class_id: int,               # integer HiRISE v3 class id
    pipeline_fn: Callable,       # callable(image) → dict/namespace with .h_final
    hazard_threshold: float = 0.7,
    pred_threshold: float = 0.5,
) -> dict:
    """Evaluate one HiRISE v3 crop.

    The crop has a single image-level class label. The pipeline produces a
    512×512 risk map; we take the mean as the image-level risk score.

    Parameters
    ----------
    image        : preprocessed image tensor (1, 3, 512, 512)
    class_id     : integer class id in [0, 7]
    pipeline_fn  : full inference pipeline callable
    hazard_threshold: threshold to classify label as hazardous
    pred_threshold  : threshold to classify prediction as hazardous

    Returns
    -------
    dict with: class_id, gt_risk, pred_risk, predicted_hazardous,
               gt_hazardous, correct_hazard_class
    """
    from src.data.label_remap import remap_label_id
    gt_risk = remap_label_id(class_id)

    result = pipeline_fn(image)
    h_final = (result.get("h_final") if isinstance(result, dict)
               else getattr(result, "h_final", None))

    if h_final is not None:
        h_np = h_final.squeeze().cpu().numpy() if isinstance(h_final, torch.Tensor) else h_final
        pred_risk = float(h_np.mean())
    else:
        pred_risk = 0.0

    gt_hazardous   = gt_risk > hazard_threshold
    pred_hazardous = pred_risk > pred_threshold

    return {
        "class_id":            class_id,
        "gt_risk":             gt_risk,
        "pred_risk":           pred_risk,
        "predicted_hazardous": int(pred_hazardous),
        "gt_hazardous":        int(gt_hazardous),
        "correct_hazard_class": int(gt_hazardous == pred_hazardous),
    }


# ---------------------------------------------------------------------------
# Full dataset evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_hirise_v3(
    pipeline_fn: Callable,
    dataset,
    device: torch.device,
    hazard_threshold: float = 0.7,
    pred_threshold: float = 0.5,
    originals_only: bool = True,
    max_crops: int | None = None,
) -> dict[str, float]:
    """Zero-shot cross-domain evaluation on HiRISE v3.

    Blueprint §19 (HiRISE v3 cross-domain): no fine-tuning.

    Parameters
    ----------
    pipeline_fn   : callable(image_tensor) → dict/namespace with h_final
    dataset       : HiRISEv3Dataset yielding {"image", "label", "class_name"}
    device        : torch.device
    hazard_threshold: threshold on GT risk for hazardous label
    pred_threshold  : threshold on predicted risk for hazardous label
    originals_only  : if True, skip augmented crops (blueprint requirement)
    max_crops       : optional cap on number of evaluated crops

    Returns
    -------
    dict of evaluation metrics:
      hazard_recall, hazard_precision, hazard_f1, mIoU,
      per_class_recall (dict), overall_accuracy, ece, n_crops
    """
    total = len(dataset)
    if originals_only:
        eval_indices = get_original_indices(total)
        log.info("HiRISE v3: using %d / %d original crops (skip augments)",
                 len(eval_indices), total)
    else:
        eval_indices = list(range(total))

    if max_crops is not None:
        eval_indices = eval_indices[:max_crops]

    all_gt_risks: list[float] = []
    all_pred_risks: list[float] = []
    per_class_results: dict[str, list[int]] = {}   # class_name → [correct, ...]

    n_evaluated = 0
    for idx in eval_indices:
        sample = dataset[idx]
        image    = sample["image"].unsqueeze(0).to(device)   # (1, 3, 512, 512)
        class_id = int(sample["label"])
        class_name = sample.get("class_name", str(class_id))

        try:
            crop_result = evaluate_crop(
                image, class_id, pipeline_fn, hazard_threshold, pred_threshold
            )
        except Exception as exc:
            log.debug("Crop %d failed: %s", idx, exc)
            continue

        all_gt_risks.append(crop_result["gt_risk"])
        all_pred_risks.append(crop_result["pred_risk"])

        per_class_results.setdefault(class_name, []).append(
            crop_result["correct_hazard_class"]
        )
        n_evaluated += 1

    if n_evaluated == 0:
        log.warning("No HiRISE v3 crops evaluated.")
        return {}

    gt  = np.array(all_gt_risks,   dtype=np.float32)
    prd = np.array(all_pred_risks, dtype=np.float32)

    # Segmentation-style metrics at crop level (each crop is one "pixel")
    seg = segmentation_metrics(prd, gt,
                               hazard_threshold=hazard_threshold,
                               pred_threshold=pred_threshold)
    ece = expected_calibration_error(prd, gt, hazard_threshold=hazard_threshold)

    # Per-class recall
    per_class_recall = {
        name: float(np.mean(correct_list))
        for name, correct_list in per_class_results.items()
    }

    # AUC-ROC
    auc = node_auc_roc(prd, gt, hazard_threshold=hazard_threshold)

    result = {
        **seg,
        "ece":           ece,
        "auc_roc":       auc,
        "n_crops":       float(n_evaluated),
        "per_class_recall": per_class_recall,
    }

    # Log per-class breakdown
    log.info("HiRISE v3 evaluation complete (%d crops):", n_evaluated)
    log.info("  Hazard recall=%.4f  Precision=%.4f  mIoU=%.4f  ECE=%.4f",
             seg["hazard_recall"], seg["hazard_precision"], seg["mIoU"], ece)
    for cname, recall_val in sorted(per_class_recall.items()):
        gt_risk_val = HIRISE_V3_RISK_MAP.get(cname, "?")
        log.info("  %-16s  recall=%.4f  (gt_risk=%.2f)", cname, recall_val, gt_risk_val)

    return result


# ---------------------------------------------------------------------------
# Multi-seed wrapper (blueprint §19)
# ---------------------------------------------------------------------------

def evaluate_hirise_multi_seed(
    pipeline_factory: Callable,     # callable(seed) → (pipeline_fn, dataset)
    seeds: list[int],
    device: torch.device,
    hazard_threshold: float = 0.7,
    originals_only: bool = True,
) -> dict[str, dict[str, float]]:
    """Run evaluate_hirise_v3 across multiple seeds and aggregate.

    Blueprint §19: run with 3 seeds, report mean ± std.
    """
    per_seed: list[dict[str, float]] = []

    for seed in seeds:
        import random
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)
        torch.manual_seed(seed)

        pipeline_fn, dataset = pipeline_factory(seed)
        result = evaluate_hirise_v3(
            pipeline_fn, dataset, device,
            hazard_threshold=hazard_threshold,
            originals_only=originals_only,
        )
        # Remove non-scalar fields before aggregation
        scalar_result = {k: v for k, v in result.items()
                         if isinstance(v, (int, float))}
        if scalar_result:
            per_seed.append(scalar_result)

    return aggregate_seed_results(per_seed)
