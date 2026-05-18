"""
risk_model.py
-------------
Stage 3 — CNN Semantic Risk Estimator.

Architecture (blueprint §10):
  Encoder : MobileNetV3-Large (MAE pretrained from Stage 0)
            - Stride-32 features → ASPP
            - Stride-4 features  → skip connection (projected to 48 ch)
  Decoder : DeepLabV3+
            - ASPP: atrous rates 6, 12, 18 + 1×1 conv + global avg pool
            - All 5 branches concat → project to 256 ch
            - 4× upsample + skip connection (48 ch) → concat → 3×3 conv → 256 ch
            - 4× upsample → 512×512
  Head    : 1×1 Conv 256→1, Sigmoid → H_learned ∈ [0,1]^{512×512}

Parameters: ~11.7M (matches blueprint)

Usage:
    from src.models.risk_model import RiskEstimator, build_risk_model

    model = build_risk_model(mae_checkpoint="checkpoints/mae_best.pt")
    h_learned = model(image_tensor)   # (B, 1, 512, 512)
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from src.models.encoder import adapt_first_conv

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ASPP Module (Atrous Spatial Pyramid Pooling)
# ---------------------------------------------------------------------------

class ASPPConv(nn.Module):
    """Single ASPP branch: dilated conv + BN + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ASPPPooling(nn.Module):
    """Global average pooling branch for ASPP."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pooled = self.block(x)
        return F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling.

    Blueprint §10:
      Rates: 6, 12, 18 (atrous convolutions) + 1×1 conv + global avg pool
      All branches concatenated → projected to 256 channels.
    """

    def __init__(self, in_channels: int, out_channels: int = 256,
                 rates: tuple = (6, 12, 18)):
        super().__init__()

        # 1×1 conv branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Atrous conv branches
        self.atrous_convs = nn.ModuleList([
            ASPPConv(in_channels, out_channels, rate) for rate in rates
        ])

        # Global average pooling branch
        self.global_pool = ASPPPooling(in_channels, out_channels)

        # Projection: 5 branches × out_channels → out_channels
        n_branches = 1 + len(rates) + 1   # 1×1 + atrous + pool = 5
        self.project = nn.Sequential(
            nn.Conv2d(n_branches * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [self.conv_1x1(x)]
        for conv in self.atrous_convs:
            branches.append(conv(x))
        branches.append(self.global_pool(x))

        out = torch.cat(branches, dim=1)
        return self.project(out)


# ---------------------------------------------------------------------------
# DeepLabV3+ Decoder
# ---------------------------------------------------------------------------

class DeepLabV3PlusDecoder(nn.Module):
    """DeepLabV3+ decoder.

    Blueprint §10:
      1. ASPP on stride-32 features → 256 ch
      2. 4× bilinear upsample
      3. Concatenate with stride-4 skip (projected to 48 ch)
      4. 3×3 conv → 256 ch
      5. 4× bilinear upsample → 512×512
      6. 1×1 conv 256→1 + Sigmoid → H_learned

    Parameters
    ----------
    low_level_channels : int   — channels at stride-4 skip connection
    aspp_in_channels : int     — channels from backbone at stride-32
    """

    def __init__(
        self,
        low_level_channels: int = 16,
        aspp_in_channels: int = 960,
    ):
        super().__init__()

        # ASPP on stride-32 features
        self.aspp = ASPP(in_channels=aspp_in_channels, out_channels=256,
                         rates=(6, 12, 18))

        # Skip connection projection: stride-4 → 48 ch
        self.skip_proj = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # Decoder convolution: ASPP (256) + skip (48) → 256 ch
        self.decode_conv = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Output head: 256 → 1, sigmoid
        self.head = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        stride32: torch.Tensor,   # (B, C32, H/32, W/32)
        stride4:  torch.Tensor,   # (B, C4,  H/4,  W/4)
        output_size: tuple = (512, 512),
    ) -> torch.Tensor:
        """
        Returns
        -------
        H_learned : (B, 1, 512, 512) float32 in [0, 1]
        """
        # Step 1: ASPP on stride-32 features
        aspp_out = self.aspp(stride32)                                # (B, 256, H/32, W/32)

        # Step 2: 4× upsample ASPP output
        target_h = stride4.shape[-2]
        target_w = stride4.shape[-1]
        aspp_up = F.interpolate(aspp_out, size=(target_h, target_w),
                                mode="bilinear", align_corners=False)  # (B, 256, H/4, W/4)

        # Step 3: Skip connection
        skip = self.skip_proj(stride4)                                 # (B, 48, H/4, W/4)

        # Step 4: Concatenate and refine
        fused = torch.cat([aspp_up, skip], dim=1)                      # (B, 304, H/4, W/4)
        decoded = self.decode_conv(fused)                              # (B, 256, H/4, W/4)

        # Step 5: Upsample to full resolution
        out = F.interpolate(decoded, size=output_size,
                            mode="bilinear", align_corners=False)      # (B, 256, 512, 512)

        # Step 6: Output head
        return self.head(out)                                          # (B, 1, 512, 512)


# ---------------------------------------------------------------------------
# Full Risk Estimator (Encoder + Decoder)
# ---------------------------------------------------------------------------

class RiskEstimator(nn.Module):
    """Stage 3 CNN Semantic Risk Estimator.

    MobileNetV3-Large backbone + DeepLabV3+ decoder.
    Input:  (B, 1, 512, 512)
    Output: (B, 1, 512, 512) — H_learned ∈ [0, 1]

    Parameters
    ----------
    pretrained_imagenet : bool
        If True and no mae_checkpoint is given, initialise from ImageNet.
        Blueprint requires MAE init; this is the ablation baseline.
    """

    def __init__(self, pretrained_imagenet: bool = False):
        super().__init__()

        # --- Encoder: MobileNetV3-Large ---
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained_imagenet else None
        backbone = mobilenet_v3_large(weights=weights)

        # Adapt first conv from 3-channel → 1-channel via weight averaging
        # (must happen before assigning self.features to avoid stale-reference fragility)
        adapt_first_conv(backbone, in_channels=1)

        # Keep the feature extraction layers only (drop classifier)
        self.features = backbone.features

        # Detect channel dimensions at stride-4 and stride-32
        # MobileNetV3-Large:
        #   features[0]    : stride-2  (first conv),                out=16,  256×256
        #   features[1]    : stride-2  (InvertedResidual, stride=1), out=16,  256×256
        #   features[2]    : stride-4  (InvertedResidual, stride=2), out=24,  128×128  ← skip
        #   features[-1]   : stride-32 (last conv),                  out=960, 16×16
        self._stride4_channels  = 24
        self._stride32_channels = 960

        # Hooks to capture intermediate features
        self._stride4_feat  = None
        self._stride32_feat = None
        self.features[2].register_forward_hook(self._hook_stride4)
        self.features[-1].register_forward_hook(self._hook_stride32)

        # --- Decoder: DeepLabV3+ ---
        self.decoder = DeepLabV3PlusDecoder(
            low_level_channels=self._stride4_channels,
            aspp_in_channels=self._stride32_channels,
        )

    def _hook_stride4(self, module, input, output):
        self._stride4_feat = output

    def _hook_stride32(self, module, input, output):
        self._stride32_feat = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, 512, 512) float32 in [0, 1]

        Returns
        -------
        H_learned : (B, 1, 512, 512) float32 in [0, 1]
        """
        # Run backbone. Hooks on features[2] and features[-1] capture the
        # stride-4 and stride-32 feature maps used by the DeepLabV3+ decoder.
        # NOTE: gradient checkpointing is intentionally NOT used here.
        # checkpoint_sequential re-runs the forward pass during backward,
        # which causes hooks to fire multiple times and overwrite each other,
        # producing stale/mismatched decoder inputs and broken gradients.
        _ = self.features(x)

        return self.decoder(
            stride32=self._stride32_feat,
            stride4=self._stride4_feat,
            output_size=(x.shape[-2], x.shape[-1]),
        )

    def load_mae_encoder(self, checkpoint_path: str | Path) -> None:
        """Load MAE-pretrained encoder weights into this model's backbone.

        Called by build_risk_model() to initialise from Stage 0 checkpoint.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"MAE checkpoint not found: {checkpoint_path}\n"
                "Run `python scripts/train_mae.py` first (Stage 0)."
            )

        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        enc_state = ckpt.get("encoder_state_dict", ckpt)

        # Extract only backbone.features weights — self.features is the Sequential,
        # so its state_dict keys start at "0.0.weight" (no "features." prefix).
        # Checkpoint keys are "backbone.features.0.0.weight" → strip "backbone.features."
        backbone_state = {}
        for k, v in enc_state.items():
            if k.startswith("backbone.features."):
                backbone_state[k.replace("backbone.features.", "")] = v

        if not backbone_state:
            raise RuntimeError(
                f"MAE checkpoint '{checkpoint_path.name}' contains no keys "
                f"starting with 'backbone.features.'. "
                f"Found key prefixes: {sorted(set(k.split('.')[0] for k in enc_state))}. "
                f"Cannot load MAE encoder — would silently train from random init."
            )

        missing, unexpected = self.features.load_state_dict(backbone_state, strict=False)
        n_loaded = len(backbone_state) - len(unexpected)
        log.info(
            "MAE encoder loaded: %d/%d weights from %s "
            "(%d missing, %d unexpected)",
            n_loaded, len(backbone_state), checkpoint_path.name,
            len(missing), len(unexpected),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_risk_model(
    mae_checkpoint: str | Path | None = None,
    pretrained_imagenet: bool = False,
) -> RiskEstimator:
    """Build Stage 3 RiskEstimator with appropriate weight initialisation.

    Priority:
      1. MAE checkpoint (blueprint requirement)
      2. ImageNet weights (ablation baseline: "ImageNet pretrained")
      3. Random init (ablation baseline: "random init")

    Parameters
    ----------
    mae_checkpoint : path to MAE checkpoint from Stage 0, or None
    pretrained_imagenet : if True and no MAE ckpt, use ImageNet weights

    Returns
    -------
    RiskEstimator ready for Stage 3 training
    """
    model = RiskEstimator(pretrained_imagenet=pretrained_imagenet and mae_checkpoint is None)

    if mae_checkpoint is not None:
        mae_checkpoint = Path(mae_checkpoint)
        if mae_checkpoint.exists():
            model.load_mae_encoder(mae_checkpoint)
            log.info("Initialised from MAE checkpoint: %s", mae_checkpoint.name)
        else:
            log.warning(
                "MAE checkpoint not found at %s. "
                "Using ImageNet=%s init instead.",
                mae_checkpoint, pretrained_imagenet,
            )

    n_params = sum(p.numel() for p in model.parameters())
    log.info("RiskEstimator: %.2fM parameters", n_params / 1e6)
    return model
