"""
precompute_graphs.py
--------------------
Phase 3a — Graph Precomputation: Build and cache PyG Data .pt files for all tiles.

Blueprint §17 Phase 3a:
  Input:  All tiles (train/val/test/OOD). Frozen Phase 1 and Phase 2 checkpoints.
  Process: Run Stages 2 through 5 on every tile. Save PyG Data objects as .pt files.
  Duration: ~3 hours for 15,000 tiles.
  Verify:  Node count in expected range (120–700+), all 14 features within value
           bounds, no disconnected graphs.

This script:
  1. Loads the trained fusion model (Stage 4 checkpoint) and physics engine
  2. For each tile .npy, runs the full Stage 2→3→4→5 pipeline
  3. Saves the resulting PyG Data as a .pt file
  4. Validates each graph and reports statistics

Output:
  data/processed/graphs/{split}/{alias}_r{row}_c{col}.pt

CRITICAL (blueprint §12):
  If the fusion model is retrained, ALL precomputed graphs must be regenerated
  because H_final and α values are baked into node features.

Run from project root:
    python scripts/precompute_graphs.py --cnn_ckpt checkpoints/cnn_best.pt
    python scripts/precompute_graphs.py --fusion_ckpt checkpoints/fusion_best.pt
    python scripts/precompute_graphs.py --split train --overwrite
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import yaml

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.graph_builder import build_graph, validate_graph
from src.models.fusion import build_fusion_model
from src.physics.combine import build_physics_engine_from_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("precompute_graphs")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MANIFEST_CSV = PROJECT_ROOT / "data" / "processed" / "tile_manifest.csv"
TILES_DIR    = PROJECT_ROOT / "data" / "processed" / "tiles"
GRAPHS_DIR   = PROJECT_ROOT / "data" / "processed" / "graphs"
STATS_CSV    = PROJECT_ROOT / "data" / "processed" / "graph_stats.csv"


def load_manifest(manifest_csv: Path) -> list[dict]:
    """Load tile manifest CSV."""
    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Tile manifest not found: {manifest_csv}\n"
            f"Run `python scripts/tile_dataset.py` first."
        )
    with open(manifest_csv) as f:
        return list(csv.DictReader(f))


def precompute_graphs(
    cnn_ckpt: str | None = None,
    fusion_ckpt: str | None = None,
    split_filter: str = "all",
    overwrite: bool = False,
    K: int = 5,
    flat_threshold: float = 0.25,
    hazard_threshold: float = 0.60,
    device_str: str = "auto",
) -> None:
    """Precompute PyG graphs for all tiles.

    Parameters
    ----------
    cnn_ckpt : str or None
        Path to Stage 3 CNN checkpoint. Default: checkpoints/cnn_best.pt.
    fusion_ckpt : str or None
        Path to Stage 4 fusion checkpoint. Default: checkpoints/fusion_best.pt.
    split_filter : str
        Which split to process: "all", "train", "val", "test_in", "test_ood".
    overwrite : bool
        Re-compute even if output .pt files already exist.
    K : int
        KNN neighbour count (default: 5).
    flat_threshold, hazard_threshold : float
        Tier thresholds (default: blueprint values).
    device_str : str
        "auto", "cuda", or "cpu".
    """
    log.info("=" * 70)
    log.info("Phase 3a — Graph Precomputation (Stage 5)")
    log.info("=" * 70)

    # --- Load GNN config (graph construction parameters) ---
    gnn_config_path = PROJECT_ROOT / "configs" / "gnn.yaml"
    if gnn_config_path.exists():
        with open(gnn_config_path) as f:
            gnn_config = yaml.safe_load(f) or {}
        log.info("GNN config loaded from %s", gnn_config_path)
    else:
        gnn_config = {}
        log.warning("GNN config not found at %s, using defaults", gnn_config_path)

    graph_cfg = gnn_config.get("graph", {})
    compactness = graph_cfg.get("compactness", 10.0)

    # --- Device ---
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    log.info("Device: %s", device)

    # --- Resolve checkpoint paths ---
    if cnn_ckpt is None:
        cnn_ckpt = str(PROJECT_ROOT / "checkpoints" / "cnn_best.pt")
    if fusion_ckpt is None:
        fusion_ckpt = str(PROJECT_ROOT / "checkpoints" / "fusion_best.pt")

    # --- Build fusion model (includes CNN + physics engine) ---
    log.info("Loading fusion model...")
    model = build_fusion_model(
        cnn_checkpoint=cnn_ckpt,
        freeze_cnn=True,
    )

    # Load fusion weights if checkpoint exists
    fusion_ckpt_path = Path(fusion_ckpt)
    if fusion_ckpt_path.exists():
        ckpt = torch.load(str(fusion_ckpt_path), map_location="cpu", weights_only=False)
        model.fusion.load_state_dict(ckpt["fusion_model"])
        log.info(
            "Fusion checkpoint loaded: epoch=%s, recall=%.4f",
            ckpt.get("epoch", "?"),
            ckpt.get("val_hazard_recall", -1),
        )
    else:
        log.warning(
            "Fusion checkpoint not found at %s. Using untrained fusion weights. "
            "This is acceptable for testing but not for final precomputation.",
            fusion_ckpt_path,
        )

    model = model.to(device).eval()

    # Also need the physics engine separately for individual features
    physics_engine = build_physics_engine_from_config().to(device).eval()

    # --- Load manifest ---
    records = load_manifest(MANIFEST_CSV)
    if split_filter != "all":
        records = [r for r in records if r.get("split") == split_filter]
    log.info("Processing %d tiles (split=%s)", len(records), split_filter)

    if not records:
        log.error("No tiles found for split '%s'.", split_filter)
        return

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Process tiles ---
    written = 0
    skipped = 0
    failed  = 0
    bridged_count = 0
    all_stats = []

    t_start = time.time()

    for rec in tqdm(records, desc="Building graphs"):
        image_npy = Path(rec["image_npy"])
        if not image_npy.exists():
            log.debug("Missing image: %s", image_npy)
            failed += 1
            continue

        split = rec.get("split", "unknown")
        alias = rec.get("alias", "unknown")
        row   = rec.get("row", "0")
        col   = rec.get("col", "0")
        stem  = f"{alias}_r{row}_c{col}"

        # Output path
        out_dir = GRAPHS_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}.pt"

        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            # Load image tile
            image = np.load(str(image_npy)).astype(np.float32)

            # Load DEM risk target if available
            risk_npy = Path(rec.get("risk_npy", ""))
            risk_target = None
            if risk_npy.exists():
                risk_target = np.load(str(risk_npy)).astype(np.float32)

            # Run fusion model to get all maps
            x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, H, W)

            with torch.no_grad():
                result = model(x)
                _, feats = physics_engine(x)

            h_physics = result["h_physics"][0, 0].cpu().numpy()
            h_learned = result["h_learned"][0, 0].cpu().numpy()
            h_final   = result["h_final"][0, 0].cpu().numpy()
            alpha     = result["alpha"][0, 0].cpu().numpy()
            slope     = feats["slope"][0, 0].cpu().numpy()
            roughness = feats["roughness"][0, 0].cpu().numpy()
            disc      = feats["disc"][0, 0].cpu().numpy()

            # Build graph
            data = build_graph(
                image=image,
                h_physics=h_physics,
                slope=slope,
                roughness=roughness,
                disc=disc,
                h_learned=h_learned,
                h_final=h_final,
                alpha=alpha,
                risk_target=risk_target,
                K=K,
                flat_threshold=flat_threshold,
                hazard_threshold=hazard_threshold,
                allocation_mode=graph_cfg.get("allocation_mode", "continuous"),
                gamma=graph_cfg.get("gamma", 1.5),
                n_min=graph_cfg.get("n_min", 8),
                n_max=graph_cfg.get("n_max", 64),
                compactness=compactness,
            )

            # Validate
            checks = validate_graph(data)
            failed_checks = [k for k, v in checks.items() if not v]
            if failed_checks:
                log.warning(
                    "Graph %s failed validation: %s", stem, failed_checks
                )

            # Save
            torch.save(data, str(out_path))
            written += 1

            # Track stats
            stats = data.graph_stats if hasattr(data, "graph_stats") else {}
            stats["stem"] = stem
            stats["split"] = split
            all_stats.append(stats)

            if stats.get("bridged", False):
                bridged_count += 1

        except Exception as e:
            log.error("Failed for %s: %s", stem, e, exc_info=True)
            failed += 1

    elapsed = time.time() - t_start

    # --- Save stats CSV ---
    if all_stats:
        with open(STATS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_stats[0].keys())
            writer.writeheader()
            for row in all_stats:
                # Flatten nested dicts
                flat_row = {}
                for k, v in row.items():
                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            flat_row[f"{k}_{k2}"] = v2
                    else:
                        flat_row[k] = v
                writer.writerows([flat_row])
        log.info("Graph stats saved: %s", STATS_CSV)

    # --- Summary ---
    total = written + skipped + failed
    log.info("")
    log.info("=" * 70)
    log.info("Graph precomputation complete:")
    log.info("  Written  : %d / %d", written, total)
    log.info("  Skipped  : %d (use --overwrite to redo)", skipped)
    log.info("  Failed   : %d", failed)
    log.info("  Bridged  : %d / %d tiles required RAG bridging",
             bridged_count, written)
    log.info("  Time     : %.1f minutes", elapsed / 60)
    log.info("  Output   : %s", GRAPHS_DIR)

    if all_stats:
        node_counts = [s.get("num_nodes", s.get("total_nodes", 0)) for s in all_stats]
        if node_counts:
            import statistics
            log.info("  Node count: mean=%.0f, std=%.0f, min=%d, max=%d",
                     statistics.mean(node_counts),
                     statistics.stdev(node_counts) if len(node_counts) > 1 else 0,
                     min(node_counts), max(node_counts))

    # Blueprint §12 bridging warning
    if written > 0 and bridged_count / max(written, 1) > 0.20:
        log.warning(
            "⚠️  Bridging occurred in %.0f%% of tiles (>20%%). "
            "Consider increasing K from %d to %d.",
            100 * bridged_count / written, K, K + 2,
        )

    log.info("=" * 70)
    if written + skipped > 0:
        log.info("✓ Graphs ready for Stage 6 GNN training.")
        log.info("  Next: python scripts/train_gnn.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3a — Precompute adaptive graphs for all tiles"
    )
    parser.add_argument(
        "--cnn_ckpt", default=None,
        help="Path to Stage 3 CNN checkpoint (default: checkpoints/cnn_best.pt)",
    )
    parser.add_argument(
        "--fusion_ckpt", default=None,
        help="Path to Stage 4 fusion checkpoint (default: checkpoints/fusion_best.pt)",
    )
    parser.add_argument(
        "--split", default="all",
        choices=["all", "train", "val", "test_in", "test_ood"],
        help="Which split to process (default: all)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-compute even if output .pt files exist",
    )
    parser.add_argument(
        "--K", type=int, default=5,
        help="KNN neighbour count (default: 5)",
    )
    parser.add_argument(
        "--flat_threshold", type=float, default=0.25,
        help="Flat tier threshold (default: 0.25)",
    )
    parser.add_argument(
        "--hazard_threshold", type=float, default=0.60,
        help="Hazard tier threshold (default: 0.60)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="'auto', 'cuda', or 'cpu'",
    )
    args = parser.parse_args()

    precompute_graphs(
        cnn_ckpt=args.cnn_ckpt,
        fusion_ckpt=args.fusion_ckpt,
        split_filter=args.split,
        overwrite=args.overwrite,
        K=args.K,
        flat_threshold=args.flat_threshold,
        hazard_threshold=args.hazard_threshold,
        device_str=args.device,
    )
