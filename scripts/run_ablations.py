"""
run_ablations.py
----------------
Automated runner for all ablation studies defined in the blueprint.

Blueprint §20 (Ablation Study Design):

  A — Adaptive Resolution Ablation (Most Important)
      B7 (fixed-300 nodes) vs Proposed (adaptive) — isolates Contribution 3.
      Also vary tier thresholds: <0.20/0.55/>0.55 and <0.30/0.65/>0.65.

  B — GNN Architecture 2×2 Ablation
      Edge type (RAG vs Physics-KNN) × architecture (GATv2 vs GATv2+FFN).

  C — MAE Pretraining Ablation
      Random init vs ImageNet vs MAE.

  D — Physics Weight Sensitivity
      Grid search over w1, w2, w3 that sum to 1.

  E — K-Neighbour Sensitivity
      Vary K from 3 to 9. Report HCR on validation set.

Run from pa-gnn/ directory:
    python scripts/run_ablations.py                        # all ablations
    python scripts/run_ablations.py --ablation A           # adaptive resolution only
    python scripts/run_ablations.py --ablation C           # MAE pretraining only
    python scripts/run_ablations.py --max-tiles 30         # quick debug run
    python scripts/run_ablations.py --dry-run              # list conditions, no execution

Outputs (per ablation):
    results/tables/ablation_A_adaptive_resolution.csv
    results/tables/ablation_B_gnn_architecture.csv
    results/tables/ablation_C_mae_pretraining.csv
    results/tables/ablation_D_physics_weights.csv
    results/tables/ablation_E_knn_sensitivity.csv
"""

import argparse
import csv
import itertools
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_ablations")

TABLES_DIR  = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


# ---------------------------------------------------------------------------
# Ablation condition specification
# ---------------------------------------------------------------------------

@dataclass
class AblationCondition:
    """One ablation variant."""
    name:        str
    description: str
    params:      dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Ablation A — Adaptive Resolution
# ---------------------------------------------------------------------------

ABLATION_A_CONDITIONS = [
    AblationCondition(
        name="Fixed-300 (B7)",
        description="Fixed 300 nodes (baseline B7). Isolates adaptive resolution contribution.",
        params={"graph_mode": "fixed", "n_nodes_fixed": 300},
    ),
    AblationCondition(
        name="Adaptive (proposed)",
        description="Adaptive resolution — proposed system. Tier thresholds: <0.25 / 0.25-0.60 / >0.60",
        params={"graph_mode": "adaptive", "flat_thresh": 0.25, "hazard_thresh": 0.60},
    ),
    AblationCondition(
        name="Adaptive (thresh-low)",
        description="Adaptive, alternative thresholds: <0.20 / 0.20-0.55 / >0.55",
        params={"graph_mode": "adaptive", "flat_thresh": 0.20, "hazard_thresh": 0.55},
    ),
    AblationCondition(
        name="Adaptive (thresh-high)",
        description="Adaptive, alternative thresholds: <0.30 / 0.30-0.65 / >0.65",
        params={"graph_mode": "adaptive", "flat_thresh": 0.30, "hazard_thresh": 0.65},
    ),
]


# ---------------------------------------------------------------------------
# Ablation B — GNN Architecture 2×2
# ---------------------------------------------------------------------------

ABLATION_B_CONDITIONS = [
    AblationCondition(
        name="RAG-GATv2",
        description="RAG edges + GATv2 only (no FFN)",
        params={"edge_type": "rag", "use_ffn": False},
    ),
    AblationCondition(
        name="RAG-GATv2+FFN",
        description="RAG edges + GATv2 + FFN",
        params={"edge_type": "rag", "use_ffn": True},
    ),
    AblationCondition(
        name="PhysicsKNN-GATv2",
        description="Physics-KNN edges + GATv2 only (no FFN)",
        params={"edge_type": "physics_knn", "use_ffn": False},
    ),
    AblationCondition(
        name="PhysicsKNN-GATv2+FFN (proposed)",
        description="Physics-KNN edges + GATv2 + FFN — proposed system",
        params={"edge_type": "physics_knn", "use_ffn": True},
    ),
]


# ---------------------------------------------------------------------------
# Ablation C — MAE Pretraining
# ---------------------------------------------------------------------------

ABLATION_C_CONDITIONS = [
    AblationCondition(
        name="Random Init",
        description="Random weight initialisation (no pretraining).",
        params={"init_mode": "random"},
    ),
    AblationCondition(
        name="ImageNet Pretrained",
        description="ImageNet pretrained encoder (standard baseline).",
        params={"init_mode": "imagenet"},
    ),
    AblationCondition(
        name="MAE Pretrained (proposed)",
        description="MAE self-supervised pretraining on CTX tiles — proposed system.",
        params={"init_mode": "mae"},
    ),
]


# ---------------------------------------------------------------------------
# Ablation D — Physics Weight Sensitivity
# ---------------------------------------------------------------------------

def _physics_weight_grid() -> list[AblationCondition]:
    """Generate physics weight grid conditions (w1+w2+w3 = 1.0)."""
    conditions = []
    for w1 in np.arange(0.2, 0.7, 0.1):
        for w2 in np.arange(0.1, 0.5, 0.1):
            w3 = 1.0 - w1 - w2
            if 0.05 <= w3 <= 0.5:
                w1r, w2r, w3r = round(float(w1), 2), round(float(w2), 2), round(float(w3), 2)
                conditions.append(AblationCondition(
                    name=f"w1={w1r}_w2={w2r}_w3={w3r}",
                    description=f"Physics weights: slope={w1r}, roughness={w2r}, disc={w3r}",
                    params={"physics_w1": w1r, "physics_w2": w2r, "physics_w3": w3r},
                ))
    return conditions

ABLATION_D_CONDITIONS = _physics_weight_grid()


# ---------------------------------------------------------------------------
# Ablation E — K-Neighbour Sensitivity
# ---------------------------------------------------------------------------

ABLATION_E_CONDITIONS = [
    AblationCondition(
        name=f"K={k}",
        description=f"Physics-KNN with K={k} neighbours.",
        params={"graph_k_neighbours": k},
    )
    for k in [3, 5, 7, 9]   # Blueprint: K=5 proposed; test 3–9
]


# ---------------------------------------------------------------------------
# Evaluation helper (lightweight — runs on precomputed graphs where possible)
# ---------------------------------------------------------------------------

def _evaluate_condition(
    condition: AblationCondition,
    args,
    device: torch.device,
    split: str = "val",
) -> dict[str, float]:
    """Evaluate one ablation condition on the validation split.

    For ablations that require different model weights (C), this function
    loads the appropriate checkpoint. For ablations that only change
    inference parameters (A, D, E), it reuses the main checkpoint.

    Returns
    -------
    dict of metrics (hazard_recall, mIoU, hcr, plr, inference_time_s, ...)
    """
    from src.evaluation.metrics import segmentation_metrics, hazard_crossing_rate, path_length_ratio

    params = condition.params
    init_mode = params.get("init_mode", "mae")

    # Determine which checkpoint to use for ablation C
    if "init_mode" in params:
        if init_mode == "mae":
            cnn_ckpt = PROJECT_ROOT / args.checkpoints / "cnn_best.pt"
        elif init_mode == "imagenet":
            cnn_ckpt = PROJECT_ROOT / args.checkpoints / f"cnn_imagenet.pt"
        else:
            cnn_ckpt = PROJECT_ROOT / args.checkpoints / f"cnn_random.pt"
    else:
        cnn_ckpt = PROJECT_ROOT / args.checkpoints / "cnn_best.pt"

    from src.pipeline import PAGNNPipeline, PipelineConfig

    cfg = PipelineConfig(
        cnn_checkpoint=str(cnn_ckpt),
        fusion_checkpoint=str(PROJECT_ROOT / args.checkpoints / "fusion_best.pt"),
        gnn_checkpoint=str(PROJECT_ROOT / args.checkpoints / "gnn_best.pt"),
        mc_passes=args.mc_passes,
        device=args.device,
        graph_k_neighbours=int(params.get("graph_k_neighbours", 5)),
        graph_flat_thresh=float(params.get("flat_thresh", 0.25)),
        graph_hazard_thresh=float(params.get("hazard_thresh", 0.60)),
        physics_w1=float(params.get("physics_w1", 0.4)),
        physics_w2=float(params.get("physics_w2", 0.3)),
        physics_w3=float(params.get("physics_w3", 0.3)),
    )

    try:
        pipeline = PAGNNPipeline(cfg)
    except Exception as exc:
        log.warning("Pipeline init failed for '%s': %s", condition.name, exc)
        return {"error": str(exc)}

    # Load validation dataset
    try:
        from src.data.label_generation import build_dataset
        splits_dir = PROJECT_ROOT / "data" / "splits"
        tiles_dir  = PROJECT_ROOT / "data" / "processed" / "tiles"
        dataset = build_dataset(split, splits_dir, tiles_dir)
    except Exception as exc:
        log.warning("Could not load dataset for ablation '%s': %s", condition.name, exc)
        return {"error": str(exc)}

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    all_recalls, all_mious, all_hcrs, all_plrs, all_times = [], [], [], [], []
    n_tiles = 0

    for batch in loader:
        if args.max_tiles is not None and n_tiles >= args.max_tiles:
            break

        image  = batch["image"].to(device)
        target = batch["risk"].cpu().numpy().squeeze()

        t0 = time.perf_counter()
        try:
            result = pipeline(image)
        except Exception as exc:
            log.debug("Tile failed: %s", exc)
            n_tiles += 1
            continue
        elapsed = time.perf_counter() - t0

        h_final = result.get("h_final")
        if h_final is not None:
            if isinstance(h_final, torch.Tensor):
                h_np = h_final.squeeze().cpu().numpy()
            else:
                h_np = np.asarray(h_final).squeeze()
            seg = segmentation_metrics(h_np, target, hazard_threshold=0.7)
            all_recalls.append(seg["hazard_recall"])
            all_mious.append(seg["mIoU"])

        waypoints = result.get("path_waypoints", [])
        if waypoints and h_final is not None:
            coords = [(int(w.get("row", 0)), int(w.get("col", 0))) for w in waypoints]
            all_hcrs.append(hazard_crossing_rate(coords, h_np if h_final is not None else target))
            all_plrs.append(path_length_ratio(coords))

        all_times.append(elapsed)
        n_tiles += 1

    def _safe_mean(lst):
        return float(np.mean(lst)) if lst else float("nan")

    return {
        "hazard_recall":     _safe_mean(all_recalls),
        "mIoU":              _safe_mean(all_mious),
        "hcr":               _safe_mean(all_hcrs),
        "plr":               _safe_mean(all_plrs),
        "inference_time_s":  _safe_mean(all_times),
        "n_tiles":           n_tiles,
    }


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def run_ablation_group(
    label: str,
    conditions: list[AblationCondition],
    args,
    device: torch.device,
    output_csv: Path,
) -> list[dict]:
    """Run one ablation group and save results to CSV."""
    log.info("=" * 60)
    log.info("Ablation %s: %d conditions", label, len(conditions))
    log.info("=" * 60)

    rows = []
    for i, condition in enumerate(conditions, 1):
        log.info("[%d/%d] %s — %s", i, len(conditions), condition.name, condition.description)

        if args.dry_run:
            log.info("  [DRY RUN] would run with params: %s", condition.params)
            rows.append({"name": condition.name, "description": condition.description,
                         **condition.params, "dry_run": True})
            continue

        t0 = time.time()
        result = _evaluate_condition(condition, args, device)
        elapsed = time.time() - t0

        row = {
            "ablation":    label,
            "name":        condition.name,
            "description": condition.description,
            **condition.params,
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in result.items()},
            "elapsed_s":   round(elapsed, 1),
        }
        rows.append(row)

        if "error" not in result:
            log.info(
                "  recall=%.4f  mIoU=%.4f  hcr=%.4f  plr=%.4f  time=%.2fs",
                result.get("hazard_recall", float("nan")),
                result.get("mIoU", float("nan")),
                result.get("hcr", float("nan")),
                result.get("plr", float("nan")),
                elapsed,
            )
        else:
            log.warning("  Error: %s", result["error"])

    if rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        log.info("Results saved: %s", output_csv)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PA-GNN ablation study runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ablation",
        choices=["A", "B", "C", "D", "E", "all"],
        default="all",
        help="Which ablation group to run (default: all)"
    )
    parser.add_argument(
        "--max-tiles", type=int, default=None,
        help="Max validation tiles per condition (omit for full eval)"
    )
    parser.add_argument(
        "--checkpoints", default="checkpoints",
        help="Checkpoint directory (default: checkpoints)"
    )
    parser.add_argument(
        "--mc-passes", type=int, default=3,
        help="MC Dropout passes (default: 3 for ablation speed)"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device: auto | cuda | cpu"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List conditions without executing"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)

    log.info("Device: %s | Dry-run: %s | Max tiles: %s",
             device, args.dry_run, args.max_tiles)

    all_run = args.ablation == "all"
    all_results = {}

    # ── Ablation A — Adaptive Resolution ─────────────────────────────────
    if all_run or args.ablation == "A":
        rows = run_ablation_group(
            label="A", conditions=ABLATION_A_CONDITIONS, args=args, device=device,
            output_csv=TABLES_DIR / "ablation_A_adaptive_resolution.csv",
        )
        all_results["A"] = rows

    # ── Ablation B — GNN Architecture ────────────────────────────────────
    if all_run or args.ablation == "B":
        rows = run_ablation_group(
            label="B", conditions=ABLATION_B_CONDITIONS, args=args, device=device,
            output_csv=TABLES_DIR / "ablation_B_gnn_architecture.csv",
        )
        all_results["B"] = rows

    # ── Ablation C — MAE Pretraining ─────────────────────────────────────
    if all_run or args.ablation == "C":
        rows = run_ablation_group(
            label="C", conditions=ABLATION_C_CONDITIONS, args=args, device=device,
            output_csv=TABLES_DIR / "ablation_C_mae_pretraining.csv",
        )
        all_results["C"] = rows

    # ── Ablation D — Physics Weight Sensitivity ───────────────────────────
    if all_run or args.ablation == "D":
        rows = run_ablation_group(
            label="D", conditions=ABLATION_D_CONDITIONS, args=args, device=device,
            output_csv=TABLES_DIR / "ablation_D_physics_weights.csv",
        )
        all_results["D"] = rows

    # ── Ablation E — K-Neighbour Sensitivity ─────────────────────────────
    if all_run or args.ablation == "E":
        rows = run_ablation_group(
            label="E", conditions=ABLATION_E_CONDITIONS, args=args, device=device,
            output_csv=TABLES_DIR / "ablation_E_knn_sensitivity.csv",
        )
        all_results["E"] = rows

    # ── Save combined JSON ────────────────────────────────────────────────
    combined_json = TABLES_DIR / "ablations_combined.json"
    combined_json.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("All ablation results saved to %s", TABLES_DIR)

    # ── Summary ──────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Ablation run complete.")
    for label, rows in all_results.items():
        log.info("  Ablation %s: %d conditions evaluated", label, len(rows))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
