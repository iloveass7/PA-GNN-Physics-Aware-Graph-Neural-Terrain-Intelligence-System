"""
combine.py
----------
Stage 2 — Physics Feature Engine: Combined Physics Risk Map.

Blueprint §9:
  H_physics = w1×S + w2×R + w3×D
  Default weights: w1=0.4 (slope), w2=0.3 (roughness), w3=0.3 (discontinuity)
  Output normalised to [0,1].

This is the top-level Stage 2 module. It assembles SlopeProxy, RoughnessProxy,
and DiscontinuityProxy into a single nn.Module and returns:
  1. H_physics  — combined risk map (B, 1, H, W), used by Stage 5 for adaptive
                   node allocation and as a physics baseline in Stage 4 fusion
  2. features   — dict of individual maps {slope, roughness, disc} for ablations

The weights w1, w2, w3 are NOT trained — they are ablation parameters loaded
from configs/physics.yaml. Grid search over {w1, w2, w3} is performed in
scripts/run_ablations.py as required by the blueprint.

Implementation requirements (blueprint §9):
  - Single nn.Module
  - All operations batched
  - F.conv2d with reflect padding throughout (delegated to sub-modules)
  - All under torch.no_grad() at inference
  - Target: < 5ms per tile on GPU, < 100ms on CPU

Usage:
    from src.physics.combine import PhysicsFeatureEngine

    engine = PhysicsFeatureEngine(w1=0.4, w2=0.3, w3=0.3).eval()
    with torch.no_grad():
        h_phys, feats = engine(image_tensor)   # (B, 1, H, W)
"""

import torch
import torch.nn as nn

from src.physics.discontinuity import DiscontinuityProxy
from src.physics.roughness import RoughnessProxy
from src.physics.slope import SlopeProxy


class PhysicsFeatureEngine(nn.Module):
    """Combined physics risk engine for Stage 2.

    Assembles three feature extractors into one batched module.

    Parameters
    ----------
    w1 : float  — slope weight      (default: 0.4, blueprint initial)
    w2 : float  — roughness weight  (default: 0.3)
    w3 : float  — discontinuity weight (default: 0.3)
    sigma : float  — LoG sigma for discontinuity (default: 2.0)
    window_size : int  — roughness sliding window (default: 7)
    eps : float  — normalisation epsilon (default: 1e-8)

    Notes
    -----
    Weights are stored as plain floats, NOT nn.Parameters — they are ablation
    hyperparameters, not learned parameters.  To tune, run grid search in
    scripts/run_ablations.py.
    """

    def __init__(
        self,
        w1: float = 0.4,
        w2: float = 0.3,
        w3: float = 0.3,
        sigma: float = 2.0,
        window_size: int = 7,
        eps: float = 1e-8,
    ):
        super().__init__()

        # Validate weights
        if not abs(w1 + w2 + w3 - 1.0) < 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0: w1={w1}, w2={w2}, w3={w3} "
                f"(sum={w1 + w2 + w3:.4f})"
            )

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self.slope_module       = SlopeProxy(eps=eps)
        self.roughness_module   = RoughnessProxy(window_size=window_size, eps=eps)
        self.discontinuity_module = DiscontinuityProxy(sigma=sigma, eps=eps)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute physics risk map and individual feature maps.

        Parameters
        ----------
        x : (B, 1, H, W) float32 in [0, 1]
            Single-channel normalised image tile (grayscale).
            If input has 3 channels, the mean is taken automatically.

        Returns
        -------
        h_physics : (B, 1, H, W) float32 in [0, 1]
            Combined physics risk map.  Higher = more hazardous.
        features  : dict with keys 'slope', 'roughness', 'disc'
            Each is (B, 1, H, W) float32 in [0, 1].
            Used for ablation analysis and visualisation.
        """
        # Handle 3-channel inputs (grayscale replicated to 3ch)
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)

        # Compute individual features
        slope    = self.slope_module(x)        # S: (B, 1, H, W)
        roughness = self.roughness_module(x)   # R: (B, 1, H, W)
        disc     = self.discontinuity_module(x) # D: (B, 1, H, W)

        # Weighted sum
        h_physics = self.w1 * slope + self.w2 * roughness + self.w3 * disc
        # Clamp to [0, 1] (already should be, but guard against floating point)
        h_physics = h_physics.clamp(0.0, 1.0)

        features = {
            "slope":     slope,
            "roughness": roughness,
            "disc":      disc,
        }

        return h_physics, features

    def set_weights(self, w1: float, w2: float, w3: float) -> None:
        """Update combination weights for ablation grid search.

        Parameters
        ----------
        w1, w2, w3 : float  — must sum to 1.0
        """
        if not abs(w1 + w2 + w3 - 1.0) < 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0: got {w1:.3f} + {w2:.3f} + {w3:.3f} "
                f"= {w1 + w2 + w3:.4f}"
            )
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def __repr__(self) -> str:
        return (
            f"PhysicsFeatureEngine(\n"
            f"  w1(slope)={self.w1}, w2(roughness)={self.w2}, w3(disc)={self.w3}\n"
            f"  {self.slope_module}\n"
            f"  {self.roughness_module}\n"
            f"  {self.discontinuity_module}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Factory: load from config file
# ---------------------------------------------------------------------------

def build_physics_engine_from_config(config_path: str | None = None) -> PhysicsFeatureEngine:
    """Build PhysicsFeatureEngine from configs/physics.yaml.

    Falls back to blueprint defaults if config is absent.
    """
    from pathlib import Path
    import logging
    log = logging.getLogger(__name__)

    defaults = dict(w1=0.4, w2=0.3, w3=0.3, sigma=2.0, window_size=7, eps=1e-8)

    if config_path is None:
        # Auto-resolve from project root
        config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "physics.yaml"
    else:
        config_path = Path(config_path)

    if config_path.exists() and config_path.stat().st_size > 0:
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            if cfg:
                defaults.update(cfg)
                defaults.pop("ablation", None)
                log.info("Physics config loaded from %s", config_path)
        except Exception as e:
            log.warning("Failed to load physics config, using defaults: %s", e)
    else:
        log.info("Physics config not found, using blueprint defaults: %s", defaults)

    return PhysicsFeatureEngine(**defaults)
