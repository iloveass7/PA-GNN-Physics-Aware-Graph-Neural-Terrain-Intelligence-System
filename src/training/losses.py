"""
losses.py
---------
Stage 3 compound loss function.

Blueprint §10:
  L = L_BCE_weighted + 0.5·L_Dice + 0.1·L_TV

  L_BCE_weighted:
    Hazardous pixels (target > 0.7) get weight 3.0.
    Safe/uncertain pixels get weight 1.0.
    Pixels where DEM is NoData are excluded via validity mask.

  L_Dice:
    Applied to hazardous region (target > 0.7).
    Standard soft Dice coefficient loss.

  L_TV:
    Total variation regulariser for spatial smoothness.
    Applied to the prediction (not the target).

Also exports evaluation metrics used during validation:
  hazard_recall  — fraction of true hazardous pixels correctly flagged
  hazard_precision
  mIoU           — mean intersection over union
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hyperparameters matching blueprint §10
# ---------------------------------------------------------------------------

HAZARD_THRESHOLD: float = 0.7    # above this → hazardous pixel
HAZARD_WEIGHT:    float = 5.0    # BCE weight for hazardous pixels (increased from 3.0 for ~95:5 imbalance)
DICE_COEFF:       float = 0.5    # L = L_BCE + DICE_COEFF × L_Dice + TV_COEFF × L_TV
TV_COEFF:         float = 0.1


# ---------------------------------------------------------------------------
# Individual loss components
# ---------------------------------------------------------------------------

def weighted_bce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    validity: torch.Tensor | None,
    hazard_threshold: float = HAZARD_THRESHOLD,
    hazard_weight: float = HAZARD_WEIGHT,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Weighted binary cross-entropy loss.

    Parameters
    ----------
    pred     : (B, 1, H, W) or (B, H, W) float32 in [0, 1]  — sigmoid output
    target   : (B, 1, H, W) or (B, H, W) float32 in [0, 1]  — DEM risk label
    validity : (B, 1, H, W) or (B, H, W) float32 {0,1}       — DEM validity mask
               None → all pixels included
    hazard_threshold : float — pixels with target > threshold are hazardous
    hazard_weight    : float — BCE weight for hazardous pixels

    Returns
    -------
    scalar loss
    """
    pred   = pred.squeeze(1) if pred.ndim == 4 else pred        # (B, H, W)
    target = target.squeeze(1) if target.ndim == 4 else target  # (B, H, W)

    # Clamp predictions for numerical stability
    pred = pred.clamp(eps, 1.0 - eps)

    # Per-pixel BCE (without reduction)
    # F.binary_cross_entropy is unconditionally blocked under autocast —
    # must locally disable autocast and cast to float32 explicitly.
    with torch.amp.autocast('cuda', enabled=False):
        bce = F.binary_cross_entropy(pred.float(), target.float(), reduction="none")   # (B, H, W)

    # Per-pixel weight map: hazardous → 3.0, otherwise → 1.0
    weights = torch.where(target > hazard_threshold,
                          torch.full_like(target, hazard_weight),
                          torch.ones_like(target))

    loss = bce * weights   # (B, H, W)

    # Apply validity mask (exclude NoData pixels from loss)
    if validity is not None:
        validity = validity.squeeze(1).float()   # (B, H, W)
        valid_sum = validity.sum().clamp(min=1.0)
        loss = (loss * validity).sum() / valid_sum
    else:
        loss = loss.mean()

    return loss


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    validity: torch.Tensor | None,
    hazard_threshold: float = HAZARD_THRESHOLD,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Soft Dice loss on hazardous region (target > hazard_threshold).

    Parameters
    ----------
    pred, target, validity : same as weighted_bce_loss
    smooth : smoothing constant to avoid division by zero

    Returns
    -------
    scalar loss in [0, 1]
    """
    pred   = pred.squeeze(1) if pred.ndim == 4 else pred
    target = target.squeeze(1) if target.ndim == 4 else target

    # Binary hazard map for Dice
    target_haz = (target > hazard_threshold).float()

    if validity is not None:
        validity = validity.squeeze(1).float()
        pred   = pred   * validity
        target_haz = target_haz * validity

    # Flatten spatial dimensions
    pred_flat = pred.flatten(1)          # (B, H*W)
    targ_flat = target_haz.flatten(1)    # (B, H*W)

    intersection = (pred_flat * targ_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + targ_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1.0 - dice).mean()


def total_variation_loss(pred: torch.Tensor) -> torch.Tensor:
    """Total variation regularisation for spatial smoothness.

    TV(x) = mean(|x[i,j] - x[i,j+1]| + |x[i,j] - x[i+1,j]|)

    Parameters
    ----------
    pred : (B, 1, H, W) or (B, H, W) float32

    Returns
    -------
    scalar loss
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)   # (B, 1, H, W)

    diff_h = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs()
    diff_w = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs()
    return diff_h.mean() + diff_w.mean()


# ---------------------------------------------------------------------------
# Compound loss (as used in Stage 3 training)
# ---------------------------------------------------------------------------

class RiskLoss(nn.Module):
    """Compound loss: L_BCE_weighted + DICE_COEFF×L_Dice + TV_COEFF×L_TV.

    Blueprint §10:
      L = L_BCE_weighted + 0.5·L_Dice + 0.1·L_TV

    Parameters
    ----------
    hazard_threshold : float  — binary hazard boundary (default: 0.7)
    hazard_weight    : float  — BCE weight for hazardous pixels (default: 5.0)
    dice_coeff       : float  — Dice loss coefficient (default: 0.5)
    tv_coeff         : float  — TV loss coefficient (default: 0.1)
    """

    def __init__(
        self,
        hazard_threshold: float = HAZARD_THRESHOLD,
        hazard_weight:    float = HAZARD_WEIGHT,
        dice_coeff:       float = DICE_COEFF,
        tv_coeff:         float = TV_COEFF,
    ):
        super().__init__()
        self.hazard_threshold = hazard_threshold
        self.hazard_weight    = hazard_weight
        self.dice_coeff       = dice_coeff
        self.tv_coeff         = tv_coeff

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        validity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Parameters
        ----------
        pred     : (B, 1, H, W) or (B, H, W) — sigmoid risk prediction
        target   : (B, 1, H, W) or (B, H, W) — DEM risk label [0.05, 0.95]
        validity : (B, 1, H, W) or (B, H, W) — {0,1} NoData mask, or None

        Returns
        -------
        total_loss : scalar
        components : dict with 'bce', 'dice', 'tv', 'total'
        """
        l_bce  = weighted_bce_loss(pred, target, validity,
                                   self.hazard_threshold, self.hazard_weight)
        l_dice = dice_loss(pred, target, validity, self.hazard_threshold)
        l_tv   = total_variation_loss(pred)

        total = l_bce + self.dice_coeff * l_dice + self.tv_coeff * l_tv

        return total, {
            "bce":   l_bce.item(),
            "dice":  l_dice.item(),
            "tv":    l_tv.item(),
            "total": total.item(),
        }


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    validity: torch.Tensor | None = None,
    hazard_threshold: float = HAZARD_THRESHOLD,
    pred_threshold:   float = 0.5,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Compute hazard recall, precision, F1, and mIoU.

    Parameters
    ----------
    pred          : (B, 1, H, W) or (B, H, W) — sigmoid output
    target        : (B, 1, H, W) or (B, H, W) — DEM risk label
    validity      : (B, 1, H, W) or (B, H, W) — {0,1} mask or None
    hazard_threshold : threshold on TARGET to define ground truth hazard
    pred_threshold   : threshold on PREDICTION to binarise

    Returns
    -------
    dict with: hazard_recall, hazard_precision, hazard_f1, safe_recall, mIoU
    """
    pred   = pred.squeeze(1) if pred.ndim == 4 else pred
    target = target.squeeze(1) if target.ndim == 4 else target

    if validity is not None:
        validity = validity.squeeze(1).bool()
    else:
        validity = torch.ones_like(target, dtype=torch.bool)

    # Binarise
    pred_bin   = (pred   > pred_threshold).bool()
    target_haz = (target > hazard_threshold).bool()

    # Restrict to valid pixels
    pred_bin   = pred_bin[validity]
    target_haz = target_haz[validity]

    # True/False Positives/Negatives for hazardous class
    tp = (pred_bin & target_haz).sum().float()
    fp = (pred_bin & ~target_haz).sum().float()
    fn = (~pred_bin & target_haz).sum().float()
    tn = (~pred_bin & ~target_haz).sum().float()

    hazard_recall    = tp / (tp + fn + eps)
    hazard_precision = tp / (tp + fp + eps)
    hazard_f1        = 2 * tp / (2 * tp + fp + fn + eps)
    safe_recall      = tn / (tn + fp + eps)

    # mIoU: IoU for hazard class + IoU for safe class, averaged
    iou_haz  = tp / (tp + fp + fn + eps)
    iou_safe = tn / (tn + fn + fp + eps)
    miou     = (iou_haz + iou_safe) / 2.0

    return {
        "hazard_recall":    hazard_recall.item(),
        "hazard_precision": hazard_precision.item(),
        "hazard_f1":        hazard_f1.item(),
        "safe_recall":      safe_recall.item(),
        "mIoU":             miou.item(),
    }
