"""
precompute_graphs.py
--------------------
Stage 2 → Stage 5 bridge: pre-compute and cache H_physics maps for all tiles.

Purpose:
  The adaptive graph construction in Stage 5 requires H_physics to be available
  at graph-build time. Pre-computing and caching them avoids recomputing physics
  features on every training iteration, reducing Stage 5 DataLoader latency.

Output:
  For each tile in the manifest, saves:
    data/processed/physics/{split}/{alias}_r{row}_c{col}_hphysics.npy
      — (512, 512) float32 array, H_physics in [0,1]

Also saves individual feature maps when --save_features is set:
    {stem}_slope.npy, {stem}_roughness.npy, {stem}_disc.npy

Run from the pa-gnn/ directory:
    python scripts/precompute_graphs.py [--split all] [--overwrite]
    python scripts/precompute_graphs.py --split train --save_features
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.combine import build_physics_engine_from_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("precompute_graphs")

MANIFEST_CSV  = PROJECT_ROOT / "data" / "processed" / "tile_manifest.csv"
PHYSICS_DIR   = PROJECT_ROOT / "data" / "processed" / "physics"


def load_manifest(manifest_csv: Path) -> list[dict]:
    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Tile manifest not found: {manifest_csv}\n"
            f"Run `python scripts/tile_dataset.py` first."
        )
    with open(manifest_csv) as f:
        return list(csv.DictReader(f))


def precompute(
    split_filter: str = "all",
    save_features: bool = False,
    overwrite: bool = False,
    batch_size: int = 8,
) -> None:
    log.info("=" * 60)
    log.info("Stage 2 — Precomputing H_physics maps")
    log.info("=" * 60)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # --- Engine ---
    engine = build_physics_engine_from_config().to(device).eval()
    log.info("Physics engine ready: w1=%.2f, w2=%.2f, w3=%.2f",
             engine.w1, engine.w2, engine.w3)

    # --- Load manifest ---
    records = load_manifest(MANIFEST_CSV)
    if split_filter != "all":
        records = [r for r in records if r.get("split") == split_filter]
    log.info("Processing %d tiles (split=%s)", len(records), split_filter)

    if not records:
        log.error("No tiles found for split '%s'. Check manifest.", split_filter)
        return

    PHYSICS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Process in batches ---
    skipped = 0
    written = 0
    failed = 0

    # Process one at a time (tiles are large, batching is optional)
    for rec in tqdm(records, desc="H_physics precompute"):
        image_npy = Path(rec["image_npy"])
        if not image_npy.exists():
            log.debug("Missing image npy: %s", image_npy)
            failed += 1
            continue

        split = rec.get("split", "unknown")
        alias = rec.get("alias", "unknown")
        row   = rec.get("row", "0")
        col   = rec.get("col", "0")

        stem = f"{alias}_r{row}_c{col}"
        out_dir = PHYSICS_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)

        hphys_path = out_dir / f"{stem}_hphysics.npy"

        if hphys_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            img = np.load(str(image_npy)).astype(np.float32)
            x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

            with torch.no_grad():
                h_phys, feats = engine(x)

            # Save H_physics
            np.save(str(hphys_path), h_phys[0, 0].cpu().numpy())

            # Optionally save individual feature maps
            if save_features:
                for feat_name, feat_tensor in feats.items():
                    feat_path = out_dir / f"{stem}_{feat_name}.npy"
                    if overwrite or not feat_path.exists():
                        np.save(str(feat_path), feat_tensor[0, 0].cpu().numpy())

            written += 1

        except Exception as e:
            log.error("Failed for %s: %s", stem, e)
            failed += 1

    log.info("")
    log.info("=" * 60)
    log.info("Precompute complete:")
    log.info("  Written : %d", written)
    log.info("  Skipped : %d (already exist, use --overwrite to redo)", skipped)
    log.info("  Failed  : %d", failed)
    log.info("  Output  : %s", PHYSICS_DIR)
    log.info("=" * 60)

    if written + skipped > 0:
        log.info("✓ H_physics maps ready for Stage 5 graph construction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute H_physics for all tiles")
    parser.add_argument(
        "--split", default="all",
        choices=["all", "train", "val", "test_in", "test_ood"],
        help="Which split to process (default: all)"
    )
    parser.add_argument(
        "--save_features", action="store_true",
        help="Also save individual slope/roughness/disc feature maps"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-compute even if output files already exist"
    )
    args = parser.parse_args()

    precompute(
        split_filter=args.split,
        save_features=args.save_features,
        overwrite=args.overwrite,
    )
