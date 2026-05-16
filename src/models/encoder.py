"""
encoder.py
----------
MobileNetV3-Large encoder adapted for Stage 0 MAE pretraining and Stage 3 CNN risk.

Blueprint §7 + Publication Refactor:
  - MobileNetV3-Large as the encoder backbone
  - Adapted with a patch embedding stem for MAE pretraining
  - Initialised from MAE checkpoint for Stage 3 supervised training
  - Input: single-channel grayscale (B, 1, 512, 512)
  - MobileNetV3's first conv layer adapted via weight averaging:
      new_weight = old_weight.mean(dim=1, keepdim=True)
    This is standard practice in medical/SAR/remote-sensing imaging.

Two export modes:
  1. MAEEncoder   — used during Stage 0 pretraining (includes patch embed stem)
  2. load_pretrained_encoder() — loads MAE checkpoint into MobileNetV3 for Stage 3

Architecture notes:
  MobileNetV3-Large stride-32 output (32× downsampled) feeds the ASPP module.
  Stride-4 output (4× downsampled) provides the DeepLabV3+ skip connection.
  Both are extracted via forward hooks — no surgery to the backbone is needed.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

log = logging.getLogger(__name__)

# Blueprint: patch size 16×16, image 512×512 → 1024 patches
PATCH_SIZE: int = 16
IMAGE_SIZE: int = 512
NUM_PATCHES: int = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 1024


# ---------------------------------------------------------------------------
# Patch embedding stem
# ---------------------------------------------------------------------------

def adapt_first_conv(backbone: nn.Module, in_channels: int = 1) -> None:
    """Adapt MobileNetV3's first conv layer from 3-channel to single-channel.

    Uses weight averaging: new_weight = old_weight.mean(dim=1, keepdim=True).
    This is accepted practice in medical imaging, SAR, and remote sensing.
    """
    old_conv = backbone.features[0][0]  # MobileNetV3 first conv
    new_conv = nn.Conv2d(
        in_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
    if old_conv.bias is not None:
        new_conv.bias.data = old_conv.bias.data.clone()
    backbone.features[0][0] = new_conv


class PatchEmbedding(nn.Module):
    """Convolutional patch tokeniser: (B, 1, H, W) → (B, N, D).

    Replaces the first stage of MobileNetV3's feature extractor with a
    patch-level projection that the MAE decoder can reconstruct against.

    Patch size 16×16, stride 16 (non-overlapping) → N = (512/16)² = 1024 patches.
    Embedding dim matches MobileNetV3-Large's first feature width (16 channels → 64D).
    """

    def __init__(self, in_channels: int = 1, patch_size: int = PATCH_SIZE,
                 embed_dim: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, H, W)
        returns : (B, N, D)  where N = (H/P) * (W/P)
        """
        x = self.proj(x)            # (B, D, H/P, W/P)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # (B, N, D)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# MAE-adapted MobileNetV3 encoder
# ---------------------------------------------------------------------------

class MAEEncoder(nn.Module):
    """MobileNetV3-Large with a learnable mask token for MAE pretraining.

    During pretraining:
      1. Input image is divided into P×P patches.
      2. 75% of patch indices are randomly masked.
      3. Masked patches are replaced with a learnable mask token.
      4. The encoder processes ALL positions (visible + masked tokens).
         This is the ViT-style encoder; decoder only reconstructs masked patches.

    During Stage 3:
      Only the MobileNetV3 backbone weights are kept; the patch embedding
      and mask token are discarded.  See load_pretrained_encoder().
    """

    MASK_RATIO: float = 0.75

    def __init__(self, pretrained_imagenet: bool = True):
        super().__init__()

        # MobileNetV3-Large backbone
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained_imagenet else None
        self.backbone = mobilenet_v3_large(weights=weights)

        # Adapt first conv from 3-channel → 1-channel via weight averaging
        adapt_first_conv(self.backbone, in_channels=1)

        # Patch embedding for MAE (single-channel input)
        self.patch_embed = PatchEmbedding(in_channels=1, patch_size=PATCH_SIZE, embed_dim=64)

        # Learnable mask token (replaces masked patches)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 64))
        nn.init.normal_(self.mask_token, std=0.02)

        # Learnable positional embedding for 1024 patches
        self.pos_embed = nn.Parameter(torch.zeros(1, NUM_PATCHES, 64))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Register hooks to extract stride-4 and stride-32 features for Stage 3
        self._stride4_feat = None
        self._stride32_feat = None
        self._register_feature_hooks()

    def _register_feature_hooks(self):
        """Hook into MobileNetV3 to capture stride-4 and stride-32 features."""
        # Stride-4: after features[1] (first InvertedResidual block)
        def hook_stride4(module, input, output):
            self._stride4_feat = output

        # Stride-32: after features[-2] (last major block before classifier)
        def hook_stride32(module, input, output):
            self._stride32_feat = output

        self.backbone.features[1].register_forward_hook(hook_stride4)
        self.backbone.features[-2].register_forward_hook(hook_stride32)

    def random_masking(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply random masking: keep (1 - MASK_RATIO) fraction of patches.

        Parameters
        ----------
        x : (B, N, D)  patch embeddings (all patches, before masking)

        Returns
        -------
        x_masked  : (B, N, D)  — patches with masked positions replaced by mask token
        mask      : (B, N)     — 1 = masked, 0 = visible (for loss computation)
        ids_restore: (B, N)    — permutation to restore original order in decoder
        """
        B, N, D = x.shape
        num_masked = int(N * self.MASK_RATIO)

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)         # (B, N)
        ids_restore = torch.argsort(ids_shuffle, dim=1)   # (B, N)

        # Build mask: 1 = masked, 0 = visible
        mask = torch.ones(B, N, device=x.device)
        mask[:, :N - num_masked] = 0
        mask = torch.gather(mask, 1, ids_restore)

        # Replace masked patches with mask token
        mask_tokens = self.mask_token.expand(B, N, -1)
        mask_bool = mask.unsqueeze(-1).bool()
        x_masked = torch.where(mask_bool, mask_tokens, x)

        return x_masked, mask, ids_restore

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = MASK_RATIO,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for MAE pretraining.

        Parameters
        ----------
        x : (B, 1, 512, 512)  — normalised single-channel grayscale

        Returns
        -------
        backbone_feat : (B, C, H', W')  — stride-32 backbone features for decoder
        patch_embeds  : (B, N, D)       — full patch embedding (for decoder input)
        mask          : (B, N)          — 1 = masked
        ids_restore   : (B, N)          — patch order restore indices
        stride4_feat  : (B, C4, H4, W4) — stride-4 skip features (for Stage 3)
        """
        # Step 1: Compute patch embeddings and add positional encoding
        patches = self.patch_embed(x)           # (B, N, 64)
        patches = patches + self.pos_embed      # (B, N, 64)

        # Step 2: Apply random masking
        patches_masked, mask, ids_restore = self.random_masking(patches)

        # Step 3: Project masked patch tokens back to image for backbone
        # Reshape (B, N, 64) → (B, 64, H/P, W/P) → bilinear upsample → (B, 3, H, W)
        B, N, D = patches_masked.shape
        h_patches = w_patches = int(N ** 0.5)
        feat_map = patches_masked.transpose(1, 2).reshape(B, D, h_patches, w_patches)
        # Upsample to 512×512 for backbone
        feat_upsampled = nn.functional.interpolate(
            feat_map, size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear", align_corners=False
        )
        # Project 64 → 1 channel for single-channel MobileNetV3
        if not hasattr(self, "_proj_to_1ch"):
            self._proj_to_1ch = nn.Conv2d(64, 1, 1, bias=False).to(x.device)
        feat_1ch = self._proj_to_1ch(feat_upsampled)

        # Step 4: Pass through MobileNetV3 backbone (hooks capture stride-4 and stride-32)
        _ = self.backbone.features(feat_1ch)

        return (
            self._stride32_feat,   # for MAE decoder
            patches,               # full patch embeddings (before masking)
            mask,
            ids_restore,
            self._stride4_feat,    # for Stage 3 skip connection
        )

    def encode_image(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference-time encoder: no masking. Returns (stride32, stride4) features.

        Used by Stage 3 CNN to get multi-scale features for DeepLabV3+.
        """
        _ = self.backbone.features(x)
        return self._stride32_feat, self._stride4_feat


# ---------------------------------------------------------------------------
# Utility: load MAE pretrained weights into a fresh MobileNetV3 for Stage 3
# ---------------------------------------------------------------------------

def load_pretrained_encoder(checkpoint_path: str | Path) -> nn.Module:
    """Load MAE-pretrained MobileNetV3 backbone weights for Stage 3.

    Extracts only the backbone weights from the MAE checkpoint.
    The patch embedding stem and mask token are discarded.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the MAE checkpoint saved by train_mae.py.

    Returns
    -------
    MAEEncoder with backbone weights loaded from checkpoint.
    Call .encode_image(x) to get (stride32, stride4) features.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"MAE checkpoint not found: {checkpoint_path}\n"
            f"Run `python scripts/train_mae.py` first."
        )

    encoder = MAEEncoder(pretrained_imagenet=False)

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    state = ckpt.get("encoder_state_dict", ckpt)

    # Filter to backbone weights only
    backbone_state = {
        k.replace("backbone.", ""): v
        for k, v in state.items()
        if k.startswith("backbone.")
    }

    missing, unexpected = encoder.backbone.load_state_dict(backbone_state, strict=False)
    if missing:
        log.warning("Missing keys when loading MAE encoder: %s", missing[:5])
    if unexpected:
        log.warning("Unexpected keys in MAE checkpoint: %s", unexpected[:5])

    log.info("MAE pretrained encoder loaded from %s", checkpoint_path.name)
    return encoder
