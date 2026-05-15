"""
label_generation.py
-------------------
Consolidated label API used by the training DataLoader.

This module owns three responsibilities:
  1. Reading a saved .npy tile quad (image, risk, hazard, validity)
  2. Applying the correct augmentation pipeline based on the split
  3. Returning a dict of torch.Tensors ready for the model

This is the single entry point all training/validation/test code uses to get
labelled tiles — no script should read .npy files directly.

Usage:
    from src.data.label_generation import TilePair, build_dataset

    dataset = build_dataset(split="train", splits_dir=..., tiles_dir=...)
    sample  = dataset[0]
    # sample["image"]   → (3, 512, 512) float32 CNN input
    # sample["risk"]    → (512, 512)    float32 training label
    # sample["hazard"]  → (512, 512)    float32 evaluation mask
    # sample["valid"]   → (512, 512)    float32 loss validity mask
    # sample["alias"]   → str
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augmentations import TrainAugmentation, ValAugmentation
from src.data.normalize import to_cnn_input

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class TilePair(Dataset):
    """PyTorch Dataset over pre-tiled .npy quad files.

    Each item is one 512×512 tile from one DEM pair.  The dataset reads the
    split .txt file to discover which tiles belong to this split, then loads
    them on demand.

    Parameters
    ----------
    tile_records : list[dict]
        Each dict must have keys: image_npy, risk_npy, hazard_npy, valid_npy, alias.
        Produced by ``label_generation.build_dataset()`` or directly from
        ``tiling.tile_dem_pair()``.
    split : str
        One of "train", "val", "test_in", "test_ood".  Controls augmentation.
    """

    _AUGMENTATIONS = {
        "train":    TrainAugmentation(),
        "val":      ValAugmentation(),
        "test_in":  ValAugmentation(),
        "test_ood": ValAugmentation(),
    }

    def __init__(self, tile_records: list[dict], split: str = "val"):
        assert split in self._AUGMENTATIONS, \
            f"split must be one of {list(self._AUGMENTATIONS)}, got '{split}'"
        self.records = tile_records
        self.split = split
        self.augment = self._AUGMENTATIONS[split]
        log.info("TilePair dataset: split=%s, tiles=%d", split, len(tile_records))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        image   = torch.from_numpy(np.load(rec["image_npy"]).astype(np.float32))   # (H, W)
        risk    = torch.from_numpy(np.load(rec["risk_npy"]).astype(np.float32))    # (H, W)
        hazard  = torch.from_numpy(np.load(rec["hazard_npy"]).astype(np.float32))  # (H, W)
        valid   = torch.from_numpy(np.load(rec["valid_npy"]).astype(np.float32))   # (H, W)

        image, risk, hazard, valid = self.augment(image, risk, hazard, valid)

        # Replicate grayscale → 3-channel for MobileNetV3
        image_3ch = image.unsqueeze(0).expand(3, -1, -1)  # (3, H, W)

        return {
            "image":   image_3ch,           # (3, 512, 512) float32
            "risk":    risk,                # (512, 512)    float32 — training label
            "hazard":  hazard,              # (512, 512)    float32 — eval mask
            "valid":   valid,               # (512, 512)    float32 — loss mask
            "alias":   rec["alias"],
            "row":     rec.get("row", -1),
            "col":     rec.get("col", -1),
        }


# ---------------------------------------------------------------------------
# Split file utilities
# ---------------------------------------------------------------------------

def read_split_file(split_txt: Path) -> list[str]:
    """Read a split .txt file and return list of DEM aliases."""
    split_txt = Path(split_txt)
    if not split_txt.exists() or split_txt.stat().st_size == 0:
        raise FileNotFoundError(
            f"Split file is empty or missing: {split_txt}\n"
            f"Run `python scripts/tile_dataset.py` first to generate split files."
        )
    with open(split_txt) as f:
        return [line.strip() for line in f if line.strip()]


def build_dataset(
    split: str,
    splits_dir: Path,
    tiles_dir: Path,
) -> "TilePair":
    """Build a TilePair dataset for a given split.

    Reads the split .txt file, finds all matching tile .npy files in tiles_dir,
    and returns a TilePair dataset.

    Parameters
    ----------
    split : str
        "train", "val", "test_in", or "test_ood"
    splits_dir : Path
        Directory containing train.txt, val.txt, test_in.txt, test_ood.txt.
    tiles_dir : Path
        Directory containing .npy tile files (output of tile_dataset.py).

    Returns
    -------
    TilePair dataset
    """
    split_txt = Path(splits_dir) / f"{split}.txt"
    aliases = read_split_file(split_txt)

    log.info("Building '%s' dataset from %d DEM locations...", split, len(aliases))

    tiles_dir = Path(tiles_dir)
    records = []

    for alias in aliases:
        # Find all image tiles for this alias
        image_files = sorted(tiles_dir.glob(f"{alias}_r*_c*_image.npy"))
        if not image_files:
            log.warning("No tiles found for alias '%s' in %s", alias, tiles_dir)
            continue

        for img_npy in image_files:
            stem = img_npy.stem.replace("_image", "")
            risk_npy   = tiles_dir / f"{stem}_risk.npy"
            hazard_npy = tiles_dir / f"{stem}_hazard.npy"
            valid_npy  = tiles_dir / f"{stem}_valid.npy"

            if not all(p.exists() for p in [risk_npy, hazard_npy, valid_npy]):
                log.warning("Incomplete tile quad for %s — skipping", stem)
                continue

            # Parse row/col from filename
            parts = stem.split("_")
            row = int(parts[-2][1:]) if len(parts) >= 2 else -1
            col = int(parts[-1][1:]) if len(parts) >= 1 else -1

            records.append({
                "alias":      alias,
                "row":        row,
                "col":        col,
                "image_npy":  str(img_npy),
                "risk_npy":   str(risk_npy),
                "hazard_npy": str(hazard_npy),
                "valid_npy":  str(valid_npy),
            })

    log.info("'%s' dataset: %d tiles from %d DEM locations", split, len(records), len(aliases))
    return TilePair(records, split=split)
