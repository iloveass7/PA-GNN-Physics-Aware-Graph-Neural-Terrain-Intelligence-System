"""
mc_dropout.py
-------------
Stage 7 (blueprint §14) — Monte Carlo Dropout uncertainty estimation.

Produces epistemic uncertainty expressing where the model lacks confidence.
High uncertainty triggers conservative routing in the path planner.

Method (Gal & Ghahramani, ICML 2016):
  1. Enable dropout at inference time.
  2. Run N=5 forward passes with different dropout masks.
  3. Per-node risk_mean = mean(p̂_i) and risk_var = var(p̂_i).
  4. Project node uncertainty to pixel space via pixel_membership.

Implementation detail:
  Dropout layers are set to training mode while BatchNorm1d stays in eval mode.
  This ensures uncertainty comes from dropout stochasticity, not batch statistics.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import numpy as np

if TYPE_CHECKING:
    from torch_geometric.data import Data


@contextmanager
def mc_dropout_mode(model: nn.Module):
    """Context manager that enables Dropout layers while keeping BatchNorm in eval.

    This is the correct way to do MC dropout inference:
    - Dropout layers must be in training mode to sample different masks
    - BatchNorm layers must stay in eval mode to use running statistics

    Yields the model and restores original mode on exit.
    """
    # Save original training state for each module
    original_states: dict[nn.Module, bool] = {}
    for module in model.modules():
        original_states[module] = module.training

    # Set model to eval (sets everything to eval)
    model.eval()

    # Re-enable only Dropout layers
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.AlphaDropout)):
            module.train()

    try:
        yield model
    finally:
        # Restore original states
        for module, was_training in original_states.items():
            module.training = was_training


class MCDropoutEstimator:
    """Monte Carlo Dropout uncertainty estimator for the GNN.

    Parameters
    ----------
    model      : PhysicsAwareGNN — trained GNN model
    n_passes   : int — number of MC forward passes (default: 5)
    device     : torch.device — computation device
    """

    def __init__(
        self,
        model: nn.Module,
        n_passes: int = 5,
        device: torch.device | None = None,
    ):
        self.model = model
        self.n_passes = n_passes
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def estimate_node_uncertainty(
        self,
        data: "Data",
    ) -> dict[str, torch.Tensor]:
        """Run MC dropout on a single graph and compute per-node statistics.

        Parameters
        ----------
        data : PyG Data object with .x and .edge_index

        Returns
        -------
        dict with:
          risk_mean : (N,) — mean risk across MC passes (final prediction)
          risk_var  : (N,) — variance across MC passes (epistemic uncertainty)
          all_preds : (n_passes, N) — all individual predictions
        """
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)

        predictions = []

        with mc_dropout_mode(self.model):
            for _ in range(self.n_passes):
                pred = self.model(x, edge_index)  # (N,)
                predictions.append(pred)

        # Stack: (n_passes, N)
        all_preds = torch.stack(predictions, dim=0)

        risk_mean = all_preds.mean(dim=0)   # (N,)
        risk_var = all_preds.var(dim=0)     # (N,)

        return {
            "risk_mean": risk_mean.cpu(),
            "risk_var": risk_var.cpu(),
            "all_preds": all_preds.cpu(),
        }

    @torch.no_grad()
    def estimate_pixel_uncertainty(
        self,
        data: "Data",
        tile_size: int = 512,
    ) -> dict[str, np.ndarray]:
        """Run MC dropout and project node uncertainty to pixel space.

        Parameters
        ----------
        data      : PyG Data object with .x, .edge_index, .pixel_membership
        tile_size : int — spatial dimension of the tile (default: 512)

        Returns
        -------
        dict with:
          risk_map       : (H, W) — mean risk projected to pixel space
          uncertainty_map : (H, W) — variance projected to pixel space
          node_risk_mean : (N,) — per-node mean risk
          node_risk_var  : (N,) — per-node variance
        """
        node_results = self.estimate_node_uncertainty(data)
        risk_mean = node_results["risk_mean"]  # (N,)
        risk_var = node_results["risk_var"]    # (N,)

        # Project to pixel space via pixel_membership
        if hasattr(data, "pixel_membership") and data.pixel_membership is not None:
            membership = data.pixel_membership  # (H, W) int64
            if isinstance(membership, torch.Tensor):
                membership = membership.numpy()

            H, W = membership.shape
            risk_map = np.zeros((H, W), dtype=np.float32)
            uncertainty_map = np.zeros((H, W), dtype=np.float32)

            risk_np = risk_mean.numpy()
            var_np = risk_var.numpy()

            # Vectorised projection: each pixel gets its node's value
            risk_map = risk_np[membership]
            uncertainty_map = var_np[membership]
        else:
            # Fallback: return node-level only, no pixel projection
            risk_map = None
            uncertainty_map = None

        return {
            "risk_map": risk_map,
            "uncertainty_map": uncertainty_map,
            "node_risk_mean": risk_mean.numpy(),
            "node_risk_var": risk_var.numpy(),
        }

    def deterministic_forward(
        self,
        data: "Data",
    ) -> torch.Tensor:
        """Single deterministic forward pass (standard eval mode).

        Used for the final risk prediction after MC dropout analysis.

        Returns
        -------
        risk : (N,) — per-node risk scores
        """
        self.model.eval()
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)

        with torch.no_grad():
            return self.model(x, edge_index).cpu()
