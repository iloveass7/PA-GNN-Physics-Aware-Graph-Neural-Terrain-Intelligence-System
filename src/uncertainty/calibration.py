"""
calibration.py
--------------
Phase 10 Publication Upgrade: Calibration Framework.

Provides Temperature Scaling (Guo et al., 2017) to post-hoc calibrate GNN outputs
before MC Dropout uncertainty estimation. Also provides calibration metrics
(ECE, MCE, Brier Score) and Reliability Diagram generation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class TemperatureScaling(nn.Module):
    """Post-hoc calibration via Temperature Scaling.
    
    p_i = sigmoid(logit_i / T)
    T is a learned scalar parameter (T > 0).
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5) # start slightly > 1 to soften

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : (N,) float32 un-sigmoid logit predictions.
        
        Returns
        -------
        probs : (N,) float32 calibrated probabilities in [0, 1].
        """
        return torch.sigmoid(logits / self.temperature)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 50):
        """Fit temperature to validation set via NLL."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_func():
            optimizer.zero_grad()
            # BCEWithLogitsLoss combines sigmoid and BCE safely
            loss = F.binary_cross_entropy_with_logits(logits / self.temperature, labels)
            loss.backward()
            return loss
            
        optimizer.step(eval_func)
        return self


def compute_calibration_stats(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bin accuracies, confidences, and bin counts for ECE/MCE."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs, bins, right=True) - 1
    # bin_indices in [0, n_bins-1]
    
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for b in range(n_bins):
        mask = (bin_indices == b)
        count = mask.sum()
        bin_counts[b] = count
        if count > 0:
            bin_accs[b] = labels[mask].mean()
            bin_confs[b] = probs[mask].mean()
            
    return bin_accs, bin_confs, bin_counts


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    bin_accs, bin_confs, bin_counts = compute_calibration_stats(probs, labels, n_bins)
    N = len(probs)
    if N == 0:
        return 0.0
    ece = np.sum((bin_counts / N) * np.abs(bin_accs - bin_confs))
    return float(ece)


def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Maximum Calibration Error (MCE)."""
    bin_accs, bin_confs, bin_counts = compute_calibration_stats(probs, labels, n_bins)
    # Only consider non-empty bins
    valid = bin_counts > 0
    if not np.any(valid):
        return 0.0
    mce = np.max(np.abs(bin_accs[valid] - bin_confs[valid]))
    return float(mce)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier Score: Mean squared error of probabilities."""
    return float(np.mean((probs - labels) ** 2))


def plot_reliability_diagram(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10, save_path: str = None):
    """Plot a reliability diagram (calibration curve)."""
    if plt is None:
        return
        
    bin_accs, bin_confs, bin_counts = compute_calibration_stats(probs, labels, n_bins)
    valid = bin_counts > 0
    
    ece = expected_calibration_error(probs, labels, n_bins)
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label="Perfectly Calibrated")
    plt.plot(bin_confs[valid], bin_accs[valid], 's-', label=f"Model (ECE = {ece:.3f})")
    
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Fraction of True Hazards (Accuracy)")
    plt.title("Reliability Diagram")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
