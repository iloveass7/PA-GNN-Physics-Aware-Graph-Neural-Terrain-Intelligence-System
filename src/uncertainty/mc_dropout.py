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
        calibrator: nn.Module | None = None,
        max_time_ms: float = 100.0,
    ):
        self.model = model
        self.n_passes = n_passes
        self.device = device or next(model.parameters()).device
        self.calibrator = calibrator
        self.max_time_ms = max_time_ms

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

        import time
        from src.utils.profiler import CUDATimer
        
        start_time = time.time()
        timer = CUDATimer(device=self.device)
        
        with timer:
            with mc_dropout_mode(self.model):
                for i in range(self.n_passes):
                    # Auto-fallback: if we are running out of time, stop early
                    elapsed_ms = (time.time() - start_time) * 1000.0
                    if i >= 3 and elapsed_ms > self.max_time_ms:
                        import logging
                        logging.getLogger(__name__).warning("MC Dropout auto-fallback: stopped at pass %d due to time limit (%.1fms)", i, elapsed_ms)
                        break
                        
                    pred = self.model(x, edge_index)  # (N,)
                    
                    # Apply temperature scaling if available
                    if self.calibrator is not None:
                        # model outputs probs after sigmoid, we need logits for temperature scaling
                        # but TemperatureScaling expects logits. If model outputs probs, we can't easily invert if it's 0 or 1.
                        # Wait, PhysicsAwareGNN outputs probabilities (sigmoid is inside the head).
                        # Actually, if we apply temperature scaling, it's better to do it inside the model or pass logits.
                        # For post-hoc, if model outputs probs, we can just invert it:
                        logits = torch.log(pred.clamp(1e-7, 1-1e-7) / (1 - pred.clamp(1e-7, 1-1e-7)))
                        pred = self.calibrator(logits)
                        
                    predictions.append(pred)

        # Stack: (n_passes, N)
        stacked = torch.stack(predictions, dim=0)

        # Compute stats
        mean_pred = stacked.mean(dim=0)
        var_pred = stacked.var(dim=0, unbiased=True) if len(predictions) > 1 else torch.zeros_like(mean_pred)

        return {
            "risk_mean": mean_pred.cpu(),
            "risk_var": var_pred.cpu(),
            "all_preds": stacked.cpu(),
            "latency_ms": timer.get_elapsed_ms()
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
