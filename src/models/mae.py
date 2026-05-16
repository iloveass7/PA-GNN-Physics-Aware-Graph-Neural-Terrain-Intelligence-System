"""
mae.py
------
Full Masked Autoencoder (MAE) model for Stage 0 self-supervised pretraining.

Assembles MAEEncoder + MAEDecoder into a single forward-pass model.
Loss: MSE on masked patches only (He et al. MAE, CVPR 2022).

Usage:
    from src.models.mae import MaskedAutoencoder

    model = MaskedAutoencoder()
    loss, pred, mask = model(images)   # images: (B, 3, 512, 512)
"""

import torch
import torch.nn as nn

from src.models.decoder import MAEDecoder, mae_loss, patchify
from src.models.encoder import MAEEncoder, PATCH_SIZE


class MaskedAutoencoder(nn.Module):
    """Full MAE: encoder + decoder.

    Blueprint §7 configuration:
      Patch size   : 16×16
      Image size   : 512×512  → 1024 patches
      Mask ratio   : 75%      → 768 patches masked per forward pass
      Encoder      : MobileNetV3-Large with patch embedding stem
      Decoder      : 4-layer MLP reconstruction head
      Loss         : MSE on masked patches, with per-patch normalisation

    After pretraining, call .encoder to get the backbone for Stage 3.
    """

    def __init__(
        self,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
        pretrained_imagenet: bool = True,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        self.encoder = MAEEncoder(pretrained_imagenet=pretrained_imagenet)
        self.decoder = MAEDecoder(encoder_dim=64, hidden_dim=256)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for MAE pretraining.

        Parameters
        ----------
        x : (B, 1, 512, 512) — normalised grayscale tile, single-channel

        Returns
        -------
        loss : scalar
        pred : (B, N, P²×C)  — decoder reconstruction (all patch positions)
        mask : (B, N)         — 1 = masked
        """
        # Encoder forward: get patch embeddings + masking info
        _, patches, mask, ids_restore, _ = self.encoder(x, mask_ratio=self.mask_ratio)

        # Decoder: reconstruct all patch positions
        pred = self.decoder(patches, ids_restore)

        # Target: patchified original image
        target = patchify(x, patch_size=PATCH_SIZE)   # (B, N, 768)

        # Loss on masked patches only
        loss = mae_loss(pred, target, mask, norm_pix_loss=self.norm_pix_loss)

        return loss, pred, mask

    def save_encoder_checkpoint(self, path: str) -> None:
        """Save only the encoder state dict (backbone + patch embed) for Stage 3."""
        torch.save(
            {"encoder_state_dict": self.encoder.state_dict()},
            path,
        )
