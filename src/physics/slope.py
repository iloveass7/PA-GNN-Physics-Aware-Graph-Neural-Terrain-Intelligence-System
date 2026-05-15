"""
slope.py
--------
Stage 2 — Feature 1: Slope Proxy (S).

Blueprint §9:
  Apply Sobel operators in horizontal and vertical directions.
  Gradient magnitude = sqrt(Gx² + Gy²).
  Normalise per tile to [0,1] with ε=1e-8.

Implementation:
  - nn.Module, all operations batched, F.conv2d with reflect padding.
  - Operates on (B, 1, H, W) float32 tensors, values in [0,1].
  - Output: (B, 1, H, W) slope proxy, normalised to [0,1] per tile.

Physical basis:
  Steep slopes produce strong brightness gradients in orbital imagery due to
  differential illumination and shadow casting under fixed solar geometry.
  Flat terrain (smooth regolith) produces near-zero gradient magnitude.

Failure mode (documented for paper):
  Also responds to albedo contrast boundaries on flat terrain (e.g., dark
  basalt adjacent to bright dust deposits). Stage 3 CNN corrects for this.

Performance target (blueprint §9):
  < 5ms per tile on GPU, < 100ms on CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sobel kernels
# ---------------------------------------------------------------------------

# 3×3 Sobel kernel for horizontal gradient (Gx)
_SOBEL_X = torch.tensor([
    [-1.,  0.,  1.],
    [-2.,  0.,  2.],
    [-1.,  0.,  1.],
], dtype=torch.float32).reshape(1, 1, 3, 3)

# 3×3 Sobel kernel for vertical gradient (Gy)
_SOBEL_Y = torch.tensor([
    [-1., -2., -1.],
    [ 0.,  0.,  0.],
    [ 1.,  2.,  1.],
], dtype=torch.float32).reshape(1, 1, 3, 3)


class SlopeProxy(nn.Module):
    """Sobel gradient magnitude slope proxy.

    Forward pass:
        x : (B, 1, H, W)  float32, values in [0, 1]
        → (B, 1, H, W)    float32 slope proxy, normalised per tile to [0, 1]

    All Sobel kernels are registered as non-trainable buffers.
    F.conv2d with reflect padding (pad=1) preserves spatial dimensions.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.register_buffer("sobel_x", _SOBEL_X)
        self.register_buffer("sobel_y", _SOBEL_Y)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, H, W) float32 in [0, 1]

        Returns
        -------
        slope : (B, 1, H, W) float32 in [0, 1], normalised per tile
        """
        # Reflect padding to preserve spatial dimensions
        x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")

        gx = F.conv2d(x_pad, self.sobel_x)   # (B, 1, H, W)
        gy = F.conv2d(x_pad, self.sobel_y)   # (B, 1, H, W)

        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + self.eps)   # (B, 1, H, W)

        # Per-tile normalisation to [0, 1]
        B = magnitude.shape[0]
        tile_max = magnitude.flatten(1).max(dim=1).values.view(B, 1, 1, 1)
        slope = magnitude / (tile_max + self.eps)

        return slope.clamp(0.0, 1.0)
