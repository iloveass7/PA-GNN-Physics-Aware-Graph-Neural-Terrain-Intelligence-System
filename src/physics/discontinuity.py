"""
discontinuity.py
----------------
Stage 2 — Feature 3: Discontinuity Proxy (D).

Blueprint §9:
  Laplacian of Gaussian (LoG) with sigma=2.0.
  Take absolute value.
  Normalise per tile to [0,1] with ε=1e-8.

Implementation:
  - nn.Module, all operations batched, F.conv2d with reflect padding.
  - Operates on (B, 1, H, W) float32 tensors, values in [0,1].
  - Output: (B, 1, H, W) discontinuity proxy, normalised per tile to [0,1].

Physical basis:
  LoG responds to sharp intensity changes at the scale set by sigma.
  Crater rims, rock edges, and scarp margins produce strong responses.
  sigma=2.0 captures hazard-scale features (~6 pixel radius) without
  responding to single-pixel noise.

Failure mode (documented for paper):
  Also responds to illumination boundaries and atmospheric haze edges.
  The weighted combination in the Physics Risk Map mitigates this.

LoG kernel construction:
  LoG(x,y) = -1/(π·σ⁴) · (1 - (x²+y²)/(2σ²)) · exp(-(x²+y²)/(2σ²))
  Discretised over a (2·ceil(3σ)+1)² grid. Zero-summed to prevent DC bias.
  For σ=2.0: kernel size = 2·ceil(6)+1 = 13×13.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_log_kernel(sigma: float = 2.0, device: torch.device = None) -> torch.Tensor:
    """Build a zero-normalised Laplacian of Gaussian kernel.

    Parameters
    ----------
    sigma : float  — Gaussian standard deviation (blueprint: 2.0)
    device : torch.device or None

    Returns
    -------
    kernel : (1, 1, K, K) float32 tensor, zero-summed
    """
    # Kernel radius: 3σ captures 99.7% of Gaussian energy
    radius = math.ceil(3 * sigma)
    size = 2 * radius + 1

    y_coords, x_coords = torch.meshgrid(
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        indexing="ij",
    )

    r2 = x_coords ** 2 + y_coords ** 2
    sigma2 = sigma ** 2

    # LoG formula: -1/(π·σ⁴) · (1 - r²/(2σ²)) · exp(-r²/(2σ²))
    gauss = torch.exp(-r2 / (2 * sigma2))
    log_kernel = -(1.0 / (math.pi * sigma2 ** 2)) * (1.0 - r2 / (2 * sigma2)) * gauss

    # Zero-sum (DC correction) to prevent response to uniform regions
    log_kernel = log_kernel - log_kernel.mean()

    kernel = log_kernel.reshape(1, 1, size, size)
    if device is not None:
        kernel = kernel.to(device)

    return kernel


class DiscontinuityProxy(nn.Module):
    """Laplacian of Gaussian (LoG) discontinuity / edge proxy.

    Forward pass:
        x : (B, 1, H, W)  float32, values in [0, 1]
        → (B, 1, H, W)    float32 discontinuity proxy, normalised per tile to [0, 1]

    Parameters
    ----------
    sigma : float
        Gaussian sigma for LoG kernel (blueprint: 2.0).
    eps : float
        Normalisation epsilon (blueprint: 1e-8).
    """

    def __init__(self, sigma: float = 2.0, eps: float = 1e-8):
        super().__init__()
        self.sigma = sigma
        self.eps = eps

        kernel = _build_log_kernel(sigma=sigma)
        self.pad = kernel.shape[-1] // 2   # half kernel width for same-size output
        self.register_buffer("log_kernel", kernel)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, H, W) float32 in [0, 1]

        Returns
        -------
        disc : (B, 1, H, W) float32 in [0, 1], normalised per tile
        """
        # Reflect padding to handle borders without black edge artefacts
        x_pad = F.pad(x, (self.pad,) * 4, mode="reflect")

        # LoG convolution
        response = F.conv2d(x_pad, self.log_kernel)   # (B, 1, H, W)

        # Absolute value: both positive (peaks) and negative (troughs) are hazards
        response = response.abs()

        # Per-tile normalisation to [0, 1]
        B = response.shape[0]
        tile_max = response.flatten(1).max(dim=1).values.view(B, 1, 1, 1)
        disc = response / (tile_max + self.eps)

        return disc.clamp(0.0, 1.0)
