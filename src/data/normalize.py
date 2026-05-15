"""
normalize.py
------------
Per-tile normalisation utilities for HiRISE browse imagery.

Blueprint §8: images are normalised per tile to [0, 1].
Blueprint §10: CNN input is grayscale replicated to 3 channels.

Two normalisation strategies are provided:
  1. MinMax  — (x - min) / (max - min)  [used at tiling time; already in tiling.py]
  2. ZScore  — (x - mean) / std          [optional; not required by blueprint]

The primary export is `to_cnn_input()` which converts a (512, 512) float32 [0,1]
numpy tile into a (3, 512, 512) float32 torch.Tensor ready for MobileNetV3.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Numpy-level normalisation (applied during tiling, kept here for reference)
# ---------------------------------------------------------------------------

def minmax_normalise(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalise a 2D image tile to [0, 1] using per-tile min/max.

    Parameters
    ----------
    image : np.ndarray (H, W), any numeric dtype
    eps : float
        Small constant to avoid division by zero on constant tiles.

    Returns
    -------
    np.ndarray (H, W), float32, values in [0, 1]
    """
    img = image.astype(np.float32)
    tile_min = img.min()
    tile_max = img.max()
    return (img - tile_min) / (tile_max - tile_min + eps)


def zscore_normalise(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalise a 2D image tile to zero mean, unit variance.

    Not used in the primary pipeline but available for ablation studies.
    """
    img = image.astype(np.float32)
    return (img - img.mean()) / (img.std() + eps)


# ---------------------------------------------------------------------------
# Torch-level utilities (used by the DataLoader / training)
# ---------------------------------------------------------------------------

def to_cnn_input(
    image_tile: np.ndarray | torch.Tensor,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Convert a (H, W) grayscale float32 tile to a (3, H, W) CNN-ready tensor.

    MobileNetV3 expects RGB input (3 channels).  The single grayscale channel
    is replicated three times as specified in blueprint §10.

    Parameters
    ----------
    image_tile : np.ndarray (H, W) float32 in [0, 1]  OR  torch.Tensor (H, W)
    device : str or torch.device

    Returns
    -------
    torch.Tensor (3, H, W), float32
    """
    if isinstance(image_tile, np.ndarray):
        tensor = torch.from_numpy(image_tile.astype(np.float32))
    else:
        tensor = image_tile.float()

    # (H, W) → (1, H, W) → (3, H, W)
    tensor = tensor.unsqueeze(0).expand(3, -1, -1)
    return tensor.to(device)


def batch_to_cnn_input(
    image_batch: np.ndarray | torch.Tensor,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Convert a (B, H, W) batch of grayscale tiles to (B, 3, H, W).

    Parameters
    ----------
    image_batch : np.ndarray (B, H, W) float32  OR  torch.Tensor (B, H, W)

    Returns
    -------
    torch.Tensor (B, 3, H, W), float32
    """
    if isinstance(image_batch, np.ndarray):
        tensor = torch.from_numpy(image_batch.astype(np.float32))
    else:
        tensor = image_batch.float()

    # (B, H, W) → (B, 1, H, W) → (B, 3, H, W)
    tensor = tensor.unsqueeze(1).expand(-1, 3, -1, -1)
    return tensor.to(device)
