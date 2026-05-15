"""
ffn_module.py
-------------
Shared feed-forward network (FFN) utility used across stages.

Provides:
  1. FFNBlock   — a single (Linear → GELU → LayerNorm → Dropout) block
  2. FFN        — a stack of FFNBlocks
  3. ConvFFN    — convolutional equivalent for spatial feature maps

These are used by:
  - MAE decoder (Stage 0)   — projection layers
  - GATv2 message passing   — node feature update MLP
  - Stage 4 fusion          — small MLP within alpha predictor

Design follows blueprint requirement of GELU activation throughout.
"""

import torch
import torch.nn as nn


class FFNBlock(nn.Module):
    """Single FFN block: Linear → GELU → LayerNorm → Dropout.

    Parameters
    ----------
    in_dim  : int   — input feature dimension
    out_dim : int   — output feature dimension
    dropout : float — dropout probability (default: 0.1)
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FFN(nn.Module):
    """Multi-layer FFN: stack of FFNBlocks with optional residual connection.

    Parameters
    ----------
    in_dim   : int         — input dimension
    hidden_dim : int       — hidden dimension for intermediate layers
    out_dim  : int         — output dimension
    n_layers : int         — total number of linear layers (default: 2)
    dropout  : float       — dropout probability
    residual : bool        — add residual connection if in_dim == out_dim
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        dropout: float = 0.1,
        residual: bool = False,
    ):
        super().__init__()
        assert n_layers >= 1, "n_layers must be at least 1"

        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        layers = []
        for i in range(n_layers - 1):
            layers.append(FFNBlock(dims[i], dims[i + 1], dropout))
        # Last layer: no dropout
        layers.append(nn.Sequential(
            nn.Linear(dims[-2], dims[-1]),
            nn.GELU(),
            nn.LayerNorm(dims[-1]),
        ))

        self.net = nn.Sequential(*layers)
        self.residual = residual and (in_dim == out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.residual:
            out = out + x
        return out


class ConvFFN(nn.Module):
    """Convolutional FFN for spatial feature maps.

    Equivalent to FFN but uses 1×1 convolutions, preserving (B, C, H, W) shape.
    Used in Stage 4 fusion alpha predictor convolution layers.

    Parameters
    ----------
    in_channels  : int
    hidden_channels : int
    out_channels : int
    kernel_size  : int  — 1 for pointwise, 3 for spatial
    padding_mode : str  — "reflect" (blueprint requirement)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding_mode: str = "reflect",
    ):
        super().__init__()
        pad = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size,
                      padding=pad, padding_mode=padding_mode, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
