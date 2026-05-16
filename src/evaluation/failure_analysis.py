"""
failure_analysis.py
-------------------
Phase 13 Publication Upgrade: Failure Mode Analysis.

Implements automated detection of common failure modes in the PA-GNN pipeline.
Useful for curating edge cases for thesis/paper figures.
"""

import numpy as np


class FailureDetector:
    """Categorize and log failure modes from fusion and graph stages."""
    
    CATEGORIES = [
        "crater_shadow_confusion",
        "graph_disconnection",
        "dune_misclassification",
        "alpha_collapse",
        "uncertainty_saturation",
    ]
    
    def __init__(self):
        self.counts = {c: 0 for c in self.CATEGORIES}
        
    def detect_crater_shadow(self, h_physics: np.ndarray, h_learned: np.ndarray, alpha: np.ndarray) -> bool:
        """High slope + low CNN risk + low alpha → shadow confusion.
        
        Physics engine sees high slope (crater rim), CNN sees dark patch and thinks it's flat/safe.
        If alpha is low, the fusion model successfully caught it. If alpha is high, it failed.
        We detect cases where the CNN was wrong by a large margin but physics caught it.
        """
        # Physics hazard, CNN safe
        mask = (h_physics > 0.7) & (h_learned < 0.3)
        if mask.sum() > 50: # Arbitrary pixel threshold
            self.counts["crater_shadow_confusion"] += 1
            return True
        return False
        
    def detect_alpha_collapse(self, alpha_map: np.ndarray, threshold: float = 0.02) -> bool:
        """std(alpha) < 0.02 across tile → fusion degeneration.
        
        If the alpha map is completely uniform, the fusion network has collapsed
        and is just taking a static average instead of adapting to local terrain.
        """
        if np.std(alpha_map) < threshold:
            self.counts["alpha_collapse"] += 1
            return True
        return False
        
    def detect_uncertainty_saturation(self, uncertainty_map: np.ndarray, threshold: float = 0.8, pct_limit: float = 0.5) -> bool:
        """>50% of nodes with U > 0.8 → OOD terrain.
        
        If the majority of the map is highly uncertain, the model is likely out-of-distribution
        (e.g., exposed to a completely unseen terrain morphology).
        """
        high_u_pct = (uncertainty_map > threshold).mean()
        if high_u_pct > pct_limit:
            self.counts["uncertainty_saturation"] += 1
            return True
        return False

    def detect_graph_disconnection(self, num_nodes: int, num_edges: int) -> bool:
        """If the graph is extremely sparse, RAG fallback may have failed."""
        if num_edges < num_nodes * 1.5:
            self.counts["graph_disconnection"] += 1
            return True
        return False
        
    def get_report(self) -> dict:
        """Return failure mode counts."""
        return self.counts
