"""
gnn_model.py
------------
Stage 6 — Full Physics-Aware GATv2 + FFN assembly.

Blueprint §13:
  Layer 1: PhysicsAwareGATv2Conv(14→32, heads=4, concat=True → 128)
           + ELU + Dropout(0.3) + FFN(128→512→128)
  Layer 2: PhysicsAwareGATv2Conv(128→32, heads=4, concat=False → 32)
           + ELU + Dropout(0.2) + FFN(32→128→32)
  Output:  Linear(32→1) + Sigmoid → p̂_i ∈ [0,1]

FFN Diversity Module (Han et al., NeurIPS 2022):
  BatchNorm1d → Linear(D→4D) → GELU → Dropout(0.1) → Linear(4D→D) + residual

  Prevents over-smoothing by applying a per-node transformation that is not
  constrained to be an average of neighbour values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gatv2_physics import PhysicsAwareGATv2Conv


class GNNFFNBlock(nn.Module):
    """FFN diversity module for GNN layers.

    Blueprint §13:
      BatchNorm1d(D) → Linear(D→4D) → GELU → Dropout → Linear(4D→D) + residual

    Uses BatchNorm1d (not LayerNorm) as specified for the GNN context.

    Parameters
    ----------
    dim     : int   — input/output feature dimension
    hidden  : int   — hidden dimension (typically 4×dim)
    dropout : float — dropout rate (default: 0.1)
    """

    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, D) node features

        Returns
        -------
        (N, D) with residual connection
        """
        residual = x
        out = self.norm(x)
        out = self.fc1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out + residual


class PhysicsAwareGNN(nn.Module):
    """Full 2-layer Physics-Aware GATv2 + FFN model.

    Architecture matches blueprint §13 exactly:
      Layer 1: GATv2(14→32×4=128) → ELU → Drop(0.3) → FFN(128,512)
      Layer 2: GATv2(128→32, mean) → ELU → Drop(0.2) → FFN(32,128)
      Head:    Linear(32→1) → Sigmoid

    Parameters
    ----------
    in_features         : int   — node feature dimension (default: 14)
    hidden_dim          : int   — GATv2 output per head (default: 32)
    heads               : int   — attention heads (default: 4)
    physics_lambda_init : float — initial λ for physics attention (default: 0.1)
    dropout_l1          : float — dropout for layer 1 (default: 0.3)
    dropout_l2          : float — dropout for layer 2 (default: 0.2)
    ffn_dropout         : float — dropout inside FFN blocks (default: 0.1)
    """

    def __init__(
        self,
        in_features: int = 14,
        hidden_dim: int = 32,
        heads: int = 4,
        physics_lambda_init: float = 0.1,
        dropout_l1: float = 0.3,
        dropout_l2: float = 0.2,
        ffn_dropout: float = 0.1,
    ):
        super().__init__()

        # Layer 1: concat=True → output is heads × hidden_dim = 128
        self.conv1 = PhysicsAwareGATv2Conv(
            in_channels=in_features,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout_l1,
            physics_lambda_init=physics_lambda_init,
        )
        dim_after_l1 = heads * hidden_dim  # 4 × 32 = 128
        self.dropout1 = nn.Dropout(dropout_l1)
        self.ffn1 = GNNFFNBlock(
            dim=dim_after_l1,
            hidden=dim_after_l1 * 4,  # 128 → 512 → 128
            dropout=ffn_dropout,
        )

        # Layer 2: concat=False → output is hidden_dim = 32 (mean-pooled)
        self.conv2 = PhysicsAwareGATv2Conv(
            in_channels=dim_after_l1,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,
            dropout=dropout_l2,
            physics_lambda_init=physics_lambda_init,
        )
        self.dropout2 = nn.Dropout(dropout_l2)
        self.ffn2 = GNNFFNBlock(
            dim=hidden_dim,
            hidden=hidden_dim * 4,  # 32 → 128 → 32
            dropout=ffn_dropout,
        )

        # Output head: per-node risk score
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (N, 14)  — node features
        edge_index : (2, E)   — graph connectivity

        Returns
        -------
        risk : (N,) — per-node risk scores in [0, 1]
        """
        # Layer 1: GATv2 → ELU → Dropout → FFN
        h = self.conv1(x, edge_index)       # (N, 128)
        h = F.elu(h)
        h = self.dropout1(h)
        h = self.ffn1(h)                    # (N, 128)

        # Layer 2: GATv2 → ELU → Dropout → FFN
        h = self.conv2(h, edge_index)       # (N, 32)
        h = F.elu(h)
        h = self.dropout2(h)
        h = self.ffn2(h)                    # (N, 32)

        # Output head
        out = self.head(h).squeeze(-1)      # (N,)
        out = torch.sigmoid(out)

        return out

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return 32-dim node embeddings before the output head.

        Useful for visualisation and analysis.
        """
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = self.dropout1(h)
        h = self.ffn1(h)

        h = self.conv2(h, edge_index)
        h = F.elu(h)
        h = self.dropout2(h)
        h = self.ffn2(h)

        return h  # (N, 32)
