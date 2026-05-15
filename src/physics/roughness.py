"""
roughness.py
------------
Stage 2 — Feature 2: Roughness Proxy (R).

Blueprint §9:
  Sliding window standard deviation of pixel intensities.
  Window size: 7×7 pixels.
  Normalise per tile to [0,1] with ε=1e-8.

Implementation:
  - nn.Module, all operations batched, F.conv2d with reflect padding.
  - Operates on (B, 1, H, W) float32 tensors, values in [0,1].
  - Output: (B, 1, H, W) roughness proxy, normalised per tile to [0,1].

Physical basis:
  Rough terrain (boulders, fractured rock) produces high local intensity
  variance because adjacent pixels capture different facets, shadow zones,
  and illuminated surfaces. Smooth terrain (compacted regolith, fine sand)
  produces low variance.

Failure mode (documented for paper):
  Fine-grained sand has low roughness (smooth appearance) despite being a
  traversal hazard for wheeled rovers due to sinkage. Stage 3 CNN corrects.

Sliding window std derivation:
  std(x) = sqrt(E[x²] - E[x]²)
  Both E[x] and E[x²] are computed via uniform depthwise convolution,
  which is equivalent to box filtering — O(HW) per tile, not O(HWK²).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoughnessProxy(nn.Module):
    """Sliding window standard deviation roughness proxy.

    Uses the identity: Var(x) = E[x²] - E[x]²
    Both expectations computed via uniform box filter (depthwise conv2d).

    Forward pass:
        x : (B, 1, H, W)  float32, values in [0, 1]
        → (B, 1, H, W)    float32 roughness proxy, normalised per tile to [0, 1]

    Parameters
    ----------
    window_size : int
        Sliding window width and height (blueprint: 7).
    eps : float
        Small constant for numerical stability (blueprint: 1e-8).
    """

    def __init__(self, window_size: int = 7, eps: float = 1e-8):
        super().__init__()
        self.window_size = window_size
        self.eps = eps
        self.pad = window_size // 2

        # Uniform box filter kernel (non-trainable)
        kernel_val = 1.0 / (window_size * window_size)
        kernel = torch.full((1, 1, window_size, window_size), kernel_val,
                             dtype=torch.float32)
        self.register_buffer("box_kernel", kernel)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, H, W) float32 in [0, 1]

        Returns
        -------
        roughness : (B, 1, H, W) float32 in [0, 1], normalised per tile
        """
        # Reflect padding to preserve spatial dimensions
        x_pad = F.pad(x, (self.pad,) * 4, mode="reflect")

        # E[x] and E[x²] via box filter
        mean_x  = F.conv2d(x_pad,      self.box_kernel)   # (B, 1, H, W)
        mean_x2 = F.conv2d(x_pad ** 2, self.box_kernel)   # (B, 1, H, W)

        # Variance (clamp to avoid negative values from floating point)
        variance = (mean_x2 - mean_x ** 2).clamp(min=0.0)
        std = torch.sqrt(variance + self.eps)              # (B, 1, H, W)

        # Per-tile normalisation to [0, 1]
        B = std.shape[0]
        tile_max = std.flatten(1).max(dim=1).values.view(B, 1, 1, 1)
        roughness = std / (tile_max + self.eps)

        return roughness.clamp(0.0, 1.0)
