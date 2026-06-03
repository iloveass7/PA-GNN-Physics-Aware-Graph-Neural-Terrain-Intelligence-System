"""
fusion.py
---------
Stage 4 — Spatial Adaptive Fusion.

Blueprint §11:
  Purpose : Combine H_physics and H_learned into H_final using a learned
            per-pixel trust map α(x,y).

  Architecture (AdaptiveFusion):
    Input : three channels concatenated — H_physics, H_learned, and the
            original grayscale image.  Shape (B, 3, 512, 512).
    Layer 1 : Conv(3→16, 3×3) + ReLU + reflect padding
    Layer 2 : Conv(16→8, 3×3) + ReLU + reflect padding
    Layer 3 : Conv(8→1, 1×1) + Sigmoid
    Output  : α(x,y) ∈ [0,1]^{512×512}
              α near 1 → trust CNN (H_learned)
              α near 0 → trust physics (H_physics)

  Fusion:
    H_final(x,y) = α(x,y) × H_learned(x,y) + (1 − α(x,y)) × H_physics(x,y)

  Total parameters : ~12,000.  Intentionally small to prevent overfitting.

  Training (two-phase, mandatory):
    Phase 1 — Train CNN (Stage 3) to convergence.
    Phase 2 — Freeze all CNN weights.  Train fusion network only.
              Loss applied to H_final against DEM-derived labels using
              the same compound loss as Stage 3.
              joint_with_cnn=false in configuration.

  Diagnostic:
    If the α map shows no spatial structure (near-uniform values across
    the tile), fusion training has degenerated.

Exports:
    AdaptiveFusion         — the lightweight 3-layer CNN producing α
    EndToEndFusionModel    — CNN + physics engine + fusion, frozen-CNN mode

Usage:
    from src.models.fusion import AdaptiveFusion, EndToEndFusionModel

    # Standalone α-network
    fusion = AdaptiveFusion()
    alpha  = fusion(h_physics, h_learned, grayscale)  # (B, 1, 512, 512)

    # Full end-to-end wrapper
    model  = EndToEndFusionModel(cnn, physics_engine, fusion, freeze_cnn=True)
    result = model(image_3ch)
    # result["h_final"], result["h_learned"], result["h_physics"], result["alpha"]
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.risk_model import RiskEstimator
from src.physics.combine import PhysicsFeatureEngine

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AdaptiveFusion — the α-prediction network
# ---------------------------------------------------------------------------

class AdaptiveFusion(nn.Module):
    """Lightweight 3-layer CNN that predicts a per-pixel trust map α(x,y).

    Blueprint §11:
      Layer 1 : Conv(3→16, 3×3, padding=1, reflect) + ReLU
      Layer 2 : Conv(16→8, 3×3, padding=1, reflect) + ReLU
      Layer 3 : Conv(8→1, 1×1) + Sigmoid

    α(x,y) ∈ [0,1].  Near 1 = trust CNN.  Near 0 = trust physics.

    Total parameters: ~12,000.
    """

    def __init__(self):
        super().__init__()

        # Layer 1: Conv(3→16, 3×3) + ReLU, reflect padding
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=0, bias=True)
        # Layer 2: Conv(16→8, 3×3) + ReLU, reflect padding
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=0, bias=True)
        # Layer 3: Conv(8→1, 1×1) + Sigmoid
        self.conv3 = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal init for conv layers, zero bias."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialise conv3 bias to 0.0 so that initial α ≈ 0.5 (sigmoid(0) = 0.5)
        # This ensures neither signal dominates at the start of training.
        nn.init.zeros_(self.conv3.bias)

    def forward(
        self,
        h_physics: torch.Tensor,
        h_learned: torch.Tensor,
        grayscale: torch.Tensor,
    ) -> torch.Tensor:
        """Predict per-pixel trust map α.

        Parameters
        ----------
        h_physics : (B, 1, H, W) float32 in [0, 1]  — physics risk map
        h_learned : (B, 1, H, W) float32 in [0, 1]  — CNN risk map
        grayscale : (B, 1, H, W) float32 in [0, 1]  — original grayscale image

        Returns
        -------
        alpha : (B, 1, H, W) float32 in [0, 1]
            Per-pixel trust in H_learned.
        """
        # Concatenate: (B, 3, H, W)
        x = torch.cat([h_physics, h_learned, grayscale], dim=1)

        # Layer 1: reflect pad + conv + ReLU
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        x = F.relu(self.conv1(x), inplace=True)

        # Layer 2: reflect pad + conv + ReLU
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        x = F.relu(self.conv2(x), inplace=True)

        # Layer 3: 1×1 conv + sigmoid (no padding needed for 1×1)
        alpha = torch.sigmoid(self.conv3(x))

        return alpha  # (B, 1, H, W)


def fuse_risk_maps(
    h_physics: torch.Tensor,
    h_learned: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Apply the fusion formula.

    Blueprint §11:
      H_final(x,y) = α(x,y) × H_learned(x,y) + (1 − α(x,y)) × H_physics(x,y)

    Parameters
    ----------
    h_physics : (B, 1, H, W)
    h_learned : (B, 1, H, W)
    alpha     : (B, 1, H, W)  — trust in H_learned

    Returns
    -------
    h_final   : (B, 1, H, W) float32 in [0, 1]
    """
    return alpha * h_learned + (1.0 - alpha) * h_physics


def alpha_regularization(alpha: torch.Tensor, beta: float = 0.01) -> torch.Tensor:
    """Alpha diversity regularization to prevent fusion collapse.

    L_alpha = beta * mean(alpha * (1 - alpha))

    Maximises alpha entropy: penalises alpha maps that are entirely 0 or 1.
    When alpha is pushed toward 0 or 1 everywhere (collapse/degeneration),
    the product alpha*(1-alpha) approaches 0, so the penalty encourages
    the network to maintain a spatially varying trust map.

    Parameters
    ----------
    alpha : (B, 1, H, W)  fusion trust map in [0, 1]
    beta  : float          regularization strength (default 0.01)

    Returns
    -------
    scalar regularization loss
    """
    return beta * (alpha * (1.0 - alpha)).mean()


# ---------------------------------------------------------------------------
# EndToEndFusionModel — wraps CNN + Physics + Fusion
# ---------------------------------------------------------------------------

class EndToEndFusionModel(nn.Module):
    """Stage 4 end-to-end fusion model.

    Wraps:
      - RiskEstimator (Stage 3 CNN)      → H_learned
      - PhysicsFeatureEngine (Stage 2)   → H_physics
      - AdaptiveFusion                   → α map
      - Fusion formula                   → H_final

    Parameters
    ----------
    cnn : RiskEstimator
        The Stage 3 CNN model.  Must be initialised with trained weights.
    physics_engine : PhysicsFeatureEngine
        The Stage 2 physics feature extractor.
    fusion : AdaptiveFusion
        The Stage 4 α-prediction network.
    freeze_cnn : bool
        If True (blueprint requirement), freeze all CNN parameters so only
        the fusion network trains.  Default: True.
    """

    def __init__(
        self,
        cnn: RiskEstimator,
        physics_engine: PhysicsFeatureEngine,
        fusion: AdaptiveFusion,
        freeze_cnn: bool = True,
        alpha_reg_beta: float = 0.01,
    ):
        super().__init__()
        self.cnn = cnn
        self.physics_engine = physics_engine
        self.fusion = fusion
        self.freeze_cnn = freeze_cnn
        self.alpha_reg_beta = alpha_reg_beta

        if freeze_cnn:
            self._freeze_cnn()
        self._freeze_physics()

    def _freeze_cnn(self) -> None:
        """Freeze all CNN parameters — blueprint §11 requirement."""
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.eval()
        log.info("CNN frozen: %d parameters set to requires_grad=False",
                 sum(1 for p in self.cnn.parameters()))

    def _freeze_physics(self) -> None:
        """Freeze physics engine (it has no learnable params, but be explicit)."""
        for param in self.physics_engine.parameters():
            param.requires_grad = False
        self.physics_engine.eval()

    def train(self, mode: bool = True) -> "EndToEndFusionModel":
        """Override train() to keep CNN in eval mode when frozen.

        Blueprint §11 mandates that the CNN stays frozen during fusion training.
        BatchNorm layers in the CNN must stay in eval mode so running statistics
        are not updated, which would corrupt the pretrained CNN.
        """
        super().train(mode)
        if self.freeze_cnn:
            self.cnn.eval()
        # Physics engine has no learnable params; always eval
        self.physics_engine.eval()
        return self

    def forward(
        self,
        image: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the full Stage 2→3→4 pipeline.

        Parameters
        ----------
        image : (B, 1, H, W) float32 in [0, 1]
            Single-channel grayscale image.

        Returns
        -------
        dict with keys:
            h_final    : (B, 1, H, W) — fused risk map
            h_learned  : (B, 1, H, W) — CNN risk prediction
            h_physics  : (B, 1, H, W) — physics risk map
            alpha      : (B, 1, H, W) — per-pixel trust map
            alpha_reg  : scalar       — alpha diversity regularization loss
            features   : dict         — individual physics features {slope, roughness, disc}
        """
        # --- Stage 3: CNN (frozen if freeze_cnn=True) ---
        if self.freeze_cnn:
            with torch.no_grad():
                h_learned = self.cnn(image)  # (B, 1, H, W)
        else:
            h_learned = self.cnn(image)

        # --- Stage 2: Physics features ---
        with torch.no_grad():
            h_physics, features = self.physics_engine(image)  # (B, 1, H, W)

        # --- Extract grayscale channel for fusion input ---
        # Defensive slice: works whether image is 1-ch or 3-ch
        grayscale = image[:, :1, :, :]  # (B, 1, H, W)

        # --- Stage 4: Fusion ---
        alpha = self.fusion(h_physics, h_learned, grayscale)  # (B, 1, H, W)
        h_final = fuse_risk_maps(h_physics, h_learned, alpha)  # (B, 1, H, W)

        # --- Alpha regularization ---
        alpha_reg = alpha_regularization(alpha, beta=self.alpha_reg_beta)

        return {
            "h_final":   h_final,
            "h_learned": h_learned,
            "h_physics": h_physics,
            "alpha":     alpha,
            "alpha_reg": alpha_reg,
            "features":  features,
        }

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        """Return only the trainable parameters (fusion network only).

        Used by the optimizer: only the fusion network's ~12K params train.
        """
        return list(self.fusion.parameters())

    def count_params(self) -> dict[str, int]:
        """Count parameters by component."""
        cnn_total = sum(p.numel() for p in self.cnn.parameters())
        cnn_train = sum(p.numel() for p in self.cnn.parameters() if p.requires_grad)
        fusion_total = sum(p.numel() for p in self.fusion.parameters())
        fusion_train = sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)
        physics_total = sum(p.numel() for p in self.physics_engine.parameters())

        return {
            "cnn_total":     cnn_total,
            "cnn_trainable": cnn_train,
            "fusion_total":  fusion_total,
            "fusion_trainable": fusion_train,
            "physics_total": physics_total,
            "total":         cnn_total + fusion_total + physics_total,
            "trainable":     cnn_train + fusion_train,
        }


# ---------------------------------------------------------------------------
# Static fusion baseline (B5 in the blueprint)
# ---------------------------------------------------------------------------

def static_fusion(
    h_physics: torch.Tensor,
    h_learned: torch.Tensor,
    alpha_fixed: float = 0.5,
) -> torch.Tensor:
    """Baseline B5: fixed α fusion (no adaptive learning).

    H_final = alpha_fixed × H_learned + (1 − alpha_fixed) × H_physics

    Parameters
    ----------
    h_physics : (B, 1, H, W)
    h_learned : (B, 1, H, W)
    alpha_fixed : float — fixed trust in CNN (default: 0.5)

    Returns
    -------
    h_final : (B, 1, H, W)
    """
    return alpha_fixed * h_learned + (1.0 - alpha_fixed) * h_physics


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_fusion_model(
    cnn_checkpoint: str | None = None,
    mae_checkpoint: str | None = None,
    pretrained_imagenet: bool = False,
    freeze_cnn: bool = True,
    physics_w1: float = 0.4,
    physics_w2: float = 0.3,
    physics_w3: float = 0.3,
    alpha_reg_beta: float = 0.01,
) -> EndToEndFusionModel:
    """Build the complete Stage 4 EndToEndFusionModel.

    Parameters
    ----------
    cnn_checkpoint : str or None
        Path to the trained CNN checkpoint from Stage 3.
        If provided, CNN weights are loaded from this checkpoint.
    mae_checkpoint : str or None
        Path to MAE checkpoint (used only if cnn_checkpoint is None,
        to build a CNN from MAE init — useful for testing).
    pretrained_imagenet : bool
        Use ImageNet init if no other checkpoint given.
    freeze_cnn : bool
        Freeze CNN parameters during fusion training (blueprint requirement).
    physics_w1, physics_w2, physics_w3 : float
        Physics feature combination weights (default: blueprint values).

    Returns
    -------
    EndToEndFusionModel ready for Stage 4 training.
    """
    from pathlib import Path
    from src.models.risk_model import build_risk_model

    # Build CNN
    cnn = build_risk_model(
        mae_checkpoint=mae_checkpoint,
        pretrained_imagenet=pretrained_imagenet,
    )

    # Load trained CNN weights if checkpoint provided
    if cnn_checkpoint is not None:
        ckpt_path = Path(cnn_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"CNN checkpoint not found: {ckpt_path}\n"
                "Run `python scripts/train_cnn.py` first (Stage 3)."
            )
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cnn.load_state_dict(ckpt["model"])
        log.info("CNN loaded from %s (epoch %d, recall=%.4f)",
                 ckpt_path.name,
                 ckpt.get("epoch", "?"),
                 ckpt.get("val_hazard_recall", -1))

    # Build physics engine
    physics_engine = PhysicsFeatureEngine(w1=physics_w1, w2=physics_w2, w3=physics_w3)

    # Build fusion network
    fusion = AdaptiveFusion()

    # Assemble
    model = EndToEndFusionModel(
        cnn=cnn,
        physics_engine=physics_engine,
        fusion=fusion,
        freeze_cnn=freeze_cnn,
        alpha_reg_beta=alpha_reg_beta,
    )

    params = model.count_params()
    log.info(
        "EndToEndFusionModel built:\n"
        "  CNN:     %d params (%d trainable)\n"
        "  Fusion:  %d params (%d trainable)\n"
        "  Physics: %d params\n"
        "  Total trainable: %d",
        params["cnn_total"], params["cnn_trainable"],
        params["fusion_total"], params["fusion_trainable"],
        params["physics_total"],
        params["trainable"],
    )

    return model
