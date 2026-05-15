"""
ctx_loader.py
-------------
PyTorch Dataset for MurrayLab CTX tiles used in Stage 0 MAE pretraining.

Blueprint §5.2:
  - 17,298 512×512 tiles total (2 sets of ~8,649)
  - PNG format, grayscale, exactly 512×512
  - No labels — unlabelled pretraining corpus
  - Normalised per tile to [0, 1]
  - No augmentation beyond MAE random masking (done inside the model)
  - 3-channel replication for MobileNetV3 (done here, not in the model)

Data location: data/raw/ctx/sliced_tiles_1/ and data/raw/ctx/sliced_tiles_2/
  Tiles matching pattern: tile_x*_y*.png

Usage:
    from src.data.ctx_loader import CTXDataset, build_ctx_dataloader

    dataset = CTXDataset(ctx_dirs=[...])
    loader  = build_ctx_dataloader(dataset, batch_size=64)
"""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

# Expected tile size from blueprint
EXPECTED_SIZE: int = 512


class CTXDataset(Dataset):
    """Dataset of MurrayLab CTX 512×512 grayscale tiles.

    Scans one or more directories for PNG tiles matching the MurrayLab
    naming convention (tile_x*_y*.png).  Any 512×512 PNG file is accepted.

    Each item returned is a (3, 512, 512) float32 tensor in [0, 1],
    with the single grayscale channel replicated to 3 channels as
    required by MobileNetV3.

    Parameters
    ----------
    ctx_dirs : list of str or Path
        Directories containing MurrayLab CTX tile PNG files.
    strict_size : bool
        If True, skip tiles that are not exactly 512×512.  Default: True.
    """

    def __init__(
        self,
        ctx_dirs: list[str | Path],
        strict_size: bool = True,
    ):
        self.strict_size = strict_size
        self.tile_paths: list[Path] = []

        for d in ctx_dirs:
            d = Path(d)
            if not d.exists():
                log.warning("CTX directory not found: %s", d)
                continue
            # Accept any PNG (MurrayLab tiles are named tile_x*_y*.png)
            found = sorted(d.glob("*.png"))
            log.info("Found %d PNG tiles in %s", len(found), d)
            self.tile_paths.extend(found)

        if not self.tile_paths:
            raise FileNotFoundError(
                f"No CTX tiles found in directories: {ctx_dirs}\n"
                f"Expected PNG files in data/raw/ctx/sliced_tiles_1/ and sliced_tiles_2/"
            )

        if strict_size:
            self.tile_paths = self._filter_by_size(self.tile_paths)

        log.info("CTXDataset: %d tiles ready for MAE pretraining", len(self.tile_paths))

    def _filter_by_size(self, paths: list[Path]) -> list[Path]:
        """Remove tiles that are not exactly 512×512."""
        valid = []
        rejected = 0
        for p in paths:
            try:
                with Image.open(p) as img:
                    if img.size == (EXPECTED_SIZE, EXPECTED_SIZE):
                        valid.append(p)
                    else:
                        rejected += 1
            except Exception as e:
                log.warning("Failed to open %s: %s", p.name, e)
                rejected += 1
        if rejected:
            log.info("Filtered out %d tiles not matching %dx%d",
                     rejected, EXPECTED_SIZE, EXPECTED_SIZE)
        return valid

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load one tile, normalise to [0,1], replicate to 3 channels.

        Returns
        -------
        torch.Tensor : (3, 512, 512) float32
        """
        path = self.tile_paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("L")              # ensure grayscale
                arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
        except Exception as e:
            log.error("Failed to load tile %s: %s — returning zeros", path.name, e)
            arr = np.zeros((EXPECTED_SIZE, EXPECTED_SIZE), dtype=np.float32)

        # Per-tile normalisation (already [0,1] from /255, but clip to be safe)
        arr = np.clip(arr, 0.0, 1.0)

        tensor = torch.from_numpy(arr).unsqueeze(0)   # (1, 512, 512)
        tensor = tensor.expand(3, -1, -1)              # (3, 512, 512)
        return tensor


def build_ctx_dataloader(
    dataset: CTXDataset,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """Build a DataLoader for MAE pretraining.

    Blueprint: batch size 64, all 17,298 tiles.
    No augmentation beyond MAE random masking (handled inside the model).
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,     # avoid partial batch at end of epoch
        persistent_workers=num_workers > 0,
    )
