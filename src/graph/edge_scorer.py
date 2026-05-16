"""
edge_scorer.py
--------------
Phase 7 Publication Upgrade: Learnable Edge Affinity MLP.

Replaces the fixed 0.5 * spatial + 0.5 * physics heuristic weighting
with a small learnable MLP that determines edge probability/affinity
based on spatial distance and physical differences (slope, roughness, uncertainty).
"""

import torch
import torch.nn as nn

class EdgeAffinityMLP(nn.Module):
    """Tiny MLP for edge scoring.

    Takes a pair of node features and computes a scalar affinity score in [0, 1].

    Features (4D):
      1. Spatial distance (normalized to [0,1])
      2. |Slope_i - Slope_j|
      3. |Roughness_i - Roughness_j|
      4. |Uncertainty_i - Uncertainty_j| (if available, else 0)
    """

    def __init__(self, in_dim: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Initialize to act roughly like the old heuristic initially:
        # High distance/differences = low affinity
        with torch.no_grad():
            self.mlp[0].weight.data.normal_(0, 0.1)
            self.mlp[0].bias.data.fill_(0)
            self.mlp[2].weight.data.fill_(-0.5) # Negative weight -> larger diff = lower score
            self.mlp[2].bias.data.fill_(1.0)    # Base positive bias

    def forward(
        self,
        spatial_dist: torch.Tensor,
        slope_diff: torch.Tensor,
        roughness_diff: torch.Tensor,
        uncertainty_diff: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        spatial_dist     : (E,) float32
        slope_diff       : (E,) float32
        roughness_diff   : (E,) float32
        uncertainty_diff : (E,) float32

        Returns
        -------
        affinity : (E,) float32 in [0, 1]
        """
        features = torch.stack([
            spatial_dist,
            slope_diff,
            roughness_diff,
            uncertainty_diff
        ], dim=-1) # (E, 4)
        
        return self.mlp(features).squeeze(-1) # (E,)
