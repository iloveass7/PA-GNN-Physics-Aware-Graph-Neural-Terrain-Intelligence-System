"""
trainer.py
----------
Stage 3 — CNN Risk Estimator trainer utility.

Encapsulates one training epoch and one validation epoch to keep train_cnn.py
clean and testable. Used only by train_cnn.py.

Design:
  - Accepts model, dataloader, optimizer, loss_fn as arguments.
  - Returns per-epoch metric dicts.
  - Gradient clipping (max_norm=1.0) on every step.
  - Accumulates and logs loss components separately (bce, dice, tv).
  - During validation: computes all metrics from losses.py.
"""

import logging
from typing import Callable

import torch
from torch.utils.data import DataLoader

from src.training.losses import RiskLoss, compute_metrics

log = logging.getLogger(__name__)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: RiskLoss,
    device: torch.device,
    grad_clip: float = 1.0,
    scaler: "torch.cuda.amp.GradScaler | None" = None,
) -> dict[str, float]:
    """Run one training epoch.

    Parameters
    ----------
    model     : RiskEstimator in train mode
    loader    : DataLoader yielding dicts with 'image', 'risk', 'valid'
    optimizer : AdamW
    loss_fn   : RiskLoss
    device    : torch.device
    grad_clip : max gradient norm
    scaler    : Optional AMP GradScaler for mixed precision

    Returns
    -------
    dict with: loss, bce, dice, tv  (mean over epoch)
    """
    model.train()
    accum = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "tv": 0.0}
    n_batches = 0

    for batch in loader:
        images   = batch["image"].to(device, non_blocking=True)    # (B, 1, 512, 512)
        targets  = batch["risk"].to(device, non_blocking=True)     # (B, 512, 512)
        validity = batch["valid"].to(device, non_blocking=True)    # (B, 512, 512)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                pred = model(images)                                # (B, 1, 512, 512)
                loss, comps = loss_fn(pred, targets, validity)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(images)
            loss, comps = loss_fn(pred, targets, validity)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        accum["loss"] += comps["total"]
        accum["bce"]  += comps["bce"]
        accum["dice"] += comps["dice"]
        accum["tv"]   += comps["tv"]
        n_batches += 1

    n_batches = max(n_batches, 1)
    return {k: v / n_batches for k, v in accum.items()}


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: RiskLoss,
    device: torch.device,
) -> dict[str, float]:
    """Run one validation epoch.

    Returns
    -------
    dict with: loss, bce, dice, tv, hazard_recall, hazard_precision,
               hazard_f1, safe_recall, mIoU
    """
    model.eval()
    loss_accum = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "tv": 0.0}
    metric_accum = {
        "hazard_recall": 0.0,
        "hazard_precision": 0.0,
        "hazard_f1": 0.0,
        "safe_recall": 0.0,
        "mIoU": 0.0,
    }
    n_batches = 0

    for batch in loader:
        images   = batch["image"].to(device, non_blocking=True)
        targets  = batch["risk"].to(device, non_blocking=True)
        validity = batch["valid"].to(device, non_blocking=True)

        pred = model(images)
        loss, comps = loss_fn(pred, targets, validity)
        metrics = compute_metrics(pred, targets, validity)

        loss_accum["loss"] += comps["total"]
        loss_accum["bce"]  += comps["bce"]
        loss_accum["dice"] += comps["dice"]
        loss_accum["tv"]   += comps["tv"]

        for k, v in metrics.items():
            metric_accum[k] = metric_accum.get(k, 0.0) + v

        n_batches += 1

    n_batches = max(n_batches, 1)
    result = {k: v / n_batches for k, v in loss_accum.items()}
    result.update({k: v / n_batches for k, v in metric_accum.items()})
    return result
