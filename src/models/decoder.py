"""
decoder.py
----------
Lightweight 4-layer MLP reconstruction head for Stage 0 MAE pretraining.

Blueprint §7:
  - 4-layer MLP decoder
  - Reconstructs masked patches from encoder features
  - Discarded after pretraining — only encoder weights are kept

Input:  (B, N, D) patch embeddings from the encoder
Output: (B, N, P*P*C) pixel values for each masked patch
        where P = patch_size (16), C = channels (3)

Loss:   Mean squared error on masked patches only (as per He et al. MAE, CVPR 2022).
"""

import torch
import torch.nn as nn

from src.models.encoder import NUM_PATCHES, PATCH_SIZE

# Reconstruction target: each patch is P×P×C = 16×16×3 = 768 values
PATCH_DIM: int = PATCH_SIZE * PATCH_SIZE * 3   # 768


class MAEDecoder(nn.Module):
    """4-layer MLP reconstruction head.

    Takes encoder output tokens (visible + masked with positional embed)
    and reconstructs pixel values for each patch position.

    Architecture:
        Linear(encoder_dim → hidden_dim) + GELU + LayerNorm
        Linear(hidden_dim → hidden_dim)  + GELU + LayerNorm
        Linear(hidden_dim → hidden_dim)  + GELU + LayerNorm
        Linear(hidden_dim → patch_dim)   — no activation (raw pixel logits)

    Parameters
    ----------
    encoder_dim : int
        Dimensionality of encoder output tokens (default: 64 from PatchEmbedding).
    hidden_dim : int
        Hidden dimension of MLP layers (default: 256 — lightweight as per blueprint).
    patch_dim : int
        Output dimension = PATCH_SIZE² × C = 768 (16×16×3).
    """

    def __init__(
        self,
        encoder_dim: int = 64,
        hidden_dim: int = 256,
        patch_dim: int = PATCH_DIM,
    ):
        super().__init__()

        self.decoder = nn.Sequential(
            # Layer 1
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            # Layer 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            # Layer 3
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            # Layer 4 — output
            nn.Linear(hidden_dim, patch_dim),
        )

        # Learned positional embedding for decoder (same size as encoder pos_embed)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, NUM_PATCHES, encoder_dim))
        nn.init.normal_(self.decoder_pos_embed, std=0.02)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        patch_embeds: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct all patch positions from encoder token sequence.

        Parameters
        ----------
        patch_embeds : (B, N, D)  — encoder output patches (may be in shuffled order)
        ids_restore  : (B, N)     — indices to restore original patch order

        Returns
        -------
        pred : (B, N, P²×C)  — reconstructed pixel values for ALL patch positions.
               Loss is computed only on masked positions.
        """
        B, N, D = patch_embeds.shape

        # Add decoder positional embedding
        x = patch_embeds + self.decoder_pos_embed

        # MLP reconstruction
        pred = self.decoder(x)   # (B, N, PATCH_DIM)

        return pred


# ---------------------------------------------------------------------------
# MAE loss function
# ---------------------------------------------------------------------------

def mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    norm_pix_loss: bool = True,
) -> torch.Tensor:
    """Compute MAE reconstruction loss on masked patches only.

    Parameters
    ----------
    pred   : (B, N, P²×C)  — decoder predictions
    target : (B, N, P²×C)  — patchified target image values
    mask   : (B, N)         — 1 = masked (compute loss here), 0 = visible
    norm_pix_loss : bool
        If True, normalise target to zero mean / unit variance per patch
        (recommended by He et al. for better reconstruction quality).

    Returns
    -------
    loss : scalar tensor
    """
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var  = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()

    # Per-element squared error
    loss = (pred - target) ** 2             # (B, N, P²×C)
    loss = loss.mean(dim=-1)               # (B, N)

    # Apply mask: only compute loss on masked patches
    loss = (loss * mask).sum() / mask.sum()
    return loss


# ---------------------------------------------------------------------------
# Patchify / unpatchify utilities
# ---------------------------------------------------------------------------

def patchify(imgs: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """Convert (B, C, H, W) images to (B, N, P²×C) patch sequences.

    N = (H/P) * (W/P), P = patch_size
    """
    B, C, H, W = imgs.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        f"Image size ({H},{W}) must be divisible by patch_size {patch_size}"

    h = H // patch_size
    w = W // patch_size

    x = imgs.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1)   # (B, h, w, P, P, C)
    x = x.flatten(2)                    # (B, h*w, P*P*C)
    return x


def unpatchify(patches: torch.Tensor, patch_size: int = PATCH_SIZE,
               img_size: int = 512, channels: int = 3) -> torch.Tensor:
    """Convert (B, N, P²×C) patch sequences back to (B, C, H, W) images."""
    B, N, _ = patches.shape
    h = w = img_size // patch_size

    x = patches.reshape(B, h, w, patch_size, patch_size, channels)
    x = x.permute(0, 5, 1, 3, 2, 4)   # (B, C, h, P, w, P)
    x = x.reshape(B, channels, img_size, img_size)
    return x
