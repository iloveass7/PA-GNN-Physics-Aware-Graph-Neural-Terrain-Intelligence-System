"""
augmentations.py
----------------
Stage 1 — Data Augmentation for training tiles only.

Blueprint §8:
  Spatial (apply identically to image AND label/mask):
    - Horizontal flip  p = 0.5
    - Vertical flip    p = 0.5
    - Rotation ±15°    with reflect fill

  Intensity (apply to image ONLY — labels are physical measurements):
    - Brightness ±20%
    - Contrast   ±20%
    - Gaussian noise  σ ~ U(0, 0.02)

All augmentations operate on torch.Tensors so they integrate directly with
the DataLoader.  Spatial transforms are applied identically to the image
and label tensors via a shared random state.

Usage:
    from src.data.augmentations import TrainAugmentation, ValAugmentation

    aug = TrainAugmentation()
    image_aug, risk_aug, hazard_aug = aug(image, risk, hazard)
"""

import math
import random

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hflip(t: torch.Tensor) -> torch.Tensor:
    """Horizontal flip of a (..., H, W) tensor."""
    return t.flip(-1)


def _vflip(t: torch.Tensor) -> torch.Tensor:
    """Vertical flip of a (..., H, W) tensor."""
    return t.flip(-2)


def _rotate(t: torch.Tensor, angle_deg: float, mode: str = "bilinear") -> torch.Tensor:
    """Rotate a (C, H, W) or (H, W) tensor by angle_deg using reflect padding.

    Reflect padding prevents black borders from rotation.
    """
    squeeze = t.ndim == 2
    if squeeze:
        t = t.unsqueeze(0)      # (H, W) → (1, H, W)
    t = t.unsqueeze(0)          # (C, H, W) → (1, C, H, W)

    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Affine matrix for rotation about the centre
    theta = torch.tensor(
        [[cos_a, -sin_a, 0.0],
         [sin_a,  cos_a, 0.0]],
        dtype=torch.float32,
    ).unsqueeze(0)  # (1, 2, 3)

    grid = F.affine_grid(theta, t.shape, align_corners=False)
    rotated = F.grid_sample(
        t.float(), grid,
        mode=mode,
        padding_mode="reflection",
        align_corners=False,
    )

    rotated = rotated.squeeze(0)    # back to (C, H, W)
    if squeeze:
        rotated = rotated.squeeze(0)  # back to (H, W)
    return rotated


# ---------------------------------------------------------------------------
# Intensity augmentations (image only)
# ---------------------------------------------------------------------------

def _random_brightness(image: torch.Tensor, max_delta: float = 0.20) -> torch.Tensor:
    """Additive brightness shift in [-max_delta, +max_delta], clamp to [0, 1]."""
    delta = random.uniform(-max_delta, max_delta)
    return (image + delta).clamp(0.0, 1.0)


def _random_contrast(image: torch.Tensor, max_factor: float = 0.20) -> torch.Tensor:
    """Multiplicative contrast change in [1 - max_factor, 1 + max_factor].

    Applied around the per-tile mean to preserve overall brightness.
    """
    factor = random.uniform(1.0 - max_factor, 1.0 + max_factor)
    mean = image.mean()
    return ((image - mean) * factor + mean).clamp(0.0, 1.0)


def _gaussian_noise(image: torch.Tensor, max_sigma: float = 0.02) -> torch.Tensor:
    """Add Gaussian noise with σ ~ U(0, max_sigma), clamp to [0, 1]."""
    sigma = random.uniform(0.0, max_sigma)
    noise = torch.randn_like(image) * sigma
    return (image + noise).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Public augmentation classes
# ---------------------------------------------------------------------------

class TrainAugmentation:
    """Full augmentation pipeline for training tiles.

    Applies spatial transforms identically to image + labels, then applies
    intensity transforms to the image only.

    Parameters
    ----------
    hflip_p : float      Probability of horizontal flip (default: 0.5)
    vflip_p : float      Probability of vertical flip (default: 0.5)
    max_rot_deg : float  Maximum rotation in degrees (default: 15.0)
    brightness : float   Maximum brightness shift (default: 0.20)
    contrast : float     Maximum contrast factor (default: 0.20)
    noise_sigma : float  Maximum Gaussian noise std (default: 0.02)
    """

    def __init__(
        self,
        hflip_p: float = 0.5,
        vflip_p: float = 0.5,
        max_rot_deg: float = 15.0,
        brightness: float = 0.20,
        contrast: float = 0.20,
        noise_sigma: float = 0.02,
    ):
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.max_rot_deg = max_rot_deg
        self.brightness = brightness
        self.contrast = contrast
        self.noise_sigma = noise_sigma

    def __call__(
        self,
        image: torch.Tensor,
        risk: torch.Tensor,
        hazard: torch.Tensor,
        validity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply augmentation.

        Parameters
        ----------
        image   : (H, W) float32 in [0, 1]
        risk    : (H, W) float32 in [0.05, 0.95]
        hazard  : (H, W) float32 or uint8 {0, 1}
        validity: (H, W) uint8 {0, 1}, optional

        Returns
        -------
        (image_aug, risk_aug, hazard_aug, validity_aug)
        All same shapes as inputs.
        """
        # ---- Spatial transforms (same random state for all arrays) ----

        # Horizontal flip
        if random.random() < self.hflip_p:
            image   = _hflip(image)
            risk    = _hflip(risk)
            hazard  = _hflip(hazard)
            if validity is not None:
                validity = _hflip(validity)

        # Vertical flip
        if random.random() < self.vflip_p:
            image   = _vflip(image)
            risk    = _vflip(risk)
            hazard  = _vflip(hazard)
            if validity is not None:
                validity = _vflip(validity)

        # Rotation ±15°
        angle = random.uniform(-self.max_rot_deg, self.max_rot_deg)
        image   = _rotate(image,  angle, mode="bilinear")
        risk    = _rotate(risk,   angle, mode="bilinear")
        # Nearest-neighbour for binary masks to keep {0,1} values
        hazard  = _rotate(hazard, angle, mode="nearest").round()
        if validity is not None:
            validity = _rotate(validity, angle, mode="nearest").round()

        # ---- Intensity transforms (image only) ----
        image = _random_brightness(image, self.brightness)
        image = _random_contrast(image,   self.contrast)
        image = _gaussian_noise(image,    self.noise_sigma)

        return image, risk, hazard, validity


class ValAugmentation:
    """No-op augmentation for validation and test tiles.

    Keeps the same call signature as TrainAugmentation so code can swap
    between the two without conditional logic.
    """

    def __call__(
        self,
        image: torch.Tensor,
        risk: torch.Tensor,
        hazard: torch.Tensor,
        validity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return image, risk, hazard, validity
