"""
gatv2_physics.py
----------------
Stage 6 — Physics-Aware GATv2 Convolution Layer.

Compatibility: PyG 2.7 on Python 3.10.
  Type hints in message() must use Optional[X] not X | Y
  (PyG inspector uses __qualname__ which 3.10 UnionType lacks).

Blueprint §13:
  Injects terrain physics similarity directly into the GATv2 attention
  scoring function before softmax normalisation.

  Attention logit:
    e_ij = LeakyReLU(aᵀ [W h_i || W h_j]) + λ × exp(−|S_i − S_j| − |R_i − R_j|)

  Where:
    W        — learnable weight matrix
    a        — learnable attention vector
    S_i, R_i — slope and roughness from node feature indices 2 and 3
    λ        — learnable scalar initialised to 0.1

  The exp(−physics_distance) term is 1.0 when nodes are physically
  identical and approaches 0 as physics features diverge. Adding this
  before softmax preserves correct normalisation.

References:
  Brody et al. "How Attentive are Graph Attention Networks?" ICLR 2022
  (GATv2 dynamic attention)
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


class PhysicsAwareGATv2Conv(MessagePassing):
    """GATv2 convolution with physics-similarity attention boosting.

    Parameters
    ----------
    in_channels  : int   — input node feature dimension
    out_channels : int   — output dimension per attention head
    heads        : int   — number of attention heads (default: 4)
    concat       : bool  — if True, output is heads × out_channels;
                           if False, output is mean-pooled to out_channels
    dropout      : float — dropout on attention coefficients
    negative_slope : float — LeakyReLU slope (default: 0.2)
    physics_lambda_init : float — initial value for learnable λ (default: 0.1)
    physics_indices : list[int] — indices of physics features [S, R, D, U, α, area]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.3,
        negative_slope: float = 0.2,
        physics_lambda_init: float = 0.1,
        physics_indices: list[int] | None = None,
    ):
        super().__init__(aggr="add", node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        # Default features: S(2), R(3), D(4), H_physics(5), alpha(8), area(9)
        self.physics_indices = physics_indices if physics_indices is not None else [2, 3, 4, 5, 8, 9]

        # --- Learnable parameters ---

        # Weight matrix W: projects input to (heads × out_channels)
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Attention vector a: applied to [W h_i || W h_j] → scalar per head
        # GATv2 applies LeakyReLU BEFORE the dot product with a
        self.att = nn.Parameter(torch.empty(1, heads, 2 * out_channels))

        # Learnable physics similarity scaling factor λ
        self.physics_lambda = nn.Parameter(
            torch.tensor(physics_lambda_init, dtype=torch.float32)
        )

        # Optional bias
        self.bias = nn.Parameter(torch.empty(heads * out_channels if concat else out_channels))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (N, in_channels) — node features
        edge_index : (2, E) — COO edge indices

        Returns
        -------
        out : (N, heads * out_channels) if concat else (N, out_channels)
        """
        N = x.size(0)
        H, C = self.heads, self.out_channels

        # Project node features: (N, in_channels) → (N, H, C)
        x_proj = self.W(x).view(N, H, C)

        # Extract physics features for attention boosting
        # Shape: (N, len(physics_indices))
        physics_feats = x[:, self.physics_indices]

        # Propagate messages
        out = self.propagate(
            edge_index,
            x_proj=x_proj,
            physics_feats=physics_feats,
            size=None,
        )

        if self.concat:
            out = out.view(N, H * C)  # (N, H*C)
        else:
            out = out.mean(dim=1)     # (N, C)

        out = out + self.bias

        return out

    def message(
        self,
        x_proj_i: torch.Tensor,
        x_proj_j: torch.Tensor,
        physics_feats_i: torch.Tensor,
        physics_feats_j: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        """Compute physics-aware attention and message for each edge.

        GATv2 attention: LeakyReLU is applied BEFORE the dot product.
        Physics boost: λ × exp(−sum(|ΔP|)) added to logit before softmax.
        """
        # --- Standard GATv2 attention logit ---
        # Concatenate source and target projections: (E, H, 2C)
        x_cat = torch.cat([x_proj_i, x_proj_j], dim=-1)

        # GATv2: apply LeakyReLU before dot product (key difference from GAT)
        x_cat = F.leaky_relu(x_cat, negative_slope=self.negative_slope)

        # Dot product with attention vector: (E, H, 2C) × (1, H, 2C) → (E, H)
        alpha = (x_cat * self.att).sum(dim=-1)

        # --- Physics similarity boost ---
        # Compute physics distance across all provided indices (L1 norm)
        physics_dist = (physics_feats_i - physics_feats_j).abs().sum(dim=-1)

        # Physics boost: λ × exp(−physics_distance)
        # Shape: (E,) → (E, 1) for broadcasting across heads
        physics_boost = self.physics_lambda * torch.exp(-physics_dist).unsqueeze(-1)

        # Add physics boost to attention logit (before softmax)
        alpha = alpha + physics_boost

        # --- Softmax normalisation across neighbours ---
        alpha = softmax(alpha, index, ptr, size_i)

        # Dropout on attention coefficients
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weight messages by attention: (E, H, C)
        return x_proj_j * alpha.unsqueeze(-1)

    def aggregate(
        self,
        inputs: torch.Tensor,
        index: torch.Tensor,
        ptr: torch.Tensor | None = None,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        """Sum aggregation of attended messages."""
        return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)
