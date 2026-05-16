"""
evaluate_all.py
---------------
Run all three evaluation protocols and compile results.

Blueprint §19 (Evaluation Protocol — Three Evaluation Contexts):
  1. In-distribution DEM test  (test_in split)
  2. OOD DEM test              (test_ood split)
  3. HiRISE v3 cross-domain    (zero-shot, no fine-tuning)

Blueprint §19 (Statistical Requirements):
  Run with 3 seeds. Report mean ± std. 95% bootstrap CI for HCR.

Blueprint §19 (Domain Gap Analysis):
  Three rows (B2/B4/PA-GNN) × three cols (in-dist / OOD / gap).
  This is the paper's headline experimental claim.

Run from pa-gnn/ directory:
    python scripts/evaluate_all.py                        # all protocols, all seeds
    python scripts/evaluate_all.py --split test_in        # one split only
    python scripts/evaluate_all.py --seeds 42             # one seed (quick sanity)
    python scripts/evaluate_all.py --max-tiles 50         # limit tiles (debug)

Outputs:
    results/tables/eval_dem_in.csv
    results/tables/eval_dem_ood.csv
    results/tables/eval_hirise_v3.csv
    results/tables/domain_gap.csv
    results/figures/fig9_domain_gap.png
    results/figures/fig8_main_results.png   (requires baseline results)
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate_all")

# ── Output paths ─────────────────────────────────────────────────────────────
TABLES_DIR  = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
LOGS_DIR    = PROJECT_ROOT / "results" / "logs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        log.warning("No rows to write to %s", path)
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    log.info("Saved: %s", path)


def _format_result(result: dict, seed: int | None = None) -> dict:
    """Flatten a result dict into a CSV-friendly row."""
    row = {}
    if seed is not None:
        row["seed"] = seed
    for k, v in result.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, (int, float)):
                    row[f"{k}_{sub_k}"] = round(float(sub_v), 6)
        elif isinstance(v, (int, float)):
            row[k] = round(float(v), 6)
    return row


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Pipeline factory (builds a callable for each seed)
# ---------------------------------------------------------------------------

def _build_pipeline(args, seed: int):
    """Build the PA-GNN pipeline for one seed run."""
    _set_seed(seed)

    from src.pipeline import PAGNNPipeline, PipelineConfig

    cfg = PipelineConfig(
        cnn_checkpoint=str(PROJECT_ROOT / args.checkpoints / "cnn_best.pt"),
        fusion_checkpoint=str(PROJECT_ROOT / args.checkpoints / "fusion_best.pt"),
        gnn_checkpoint=str(PROJECT_ROOT / args.checkpoints / "gnn_best.pt"),
        mc_passes=args.mc_passes,
        device=args.device,
    )
    pipeline = PAGNNPipeline(cfg)
    return pipeline


# ---------------------------------------------------------------------------
# DEM evaluation runner
# ---------------------------------------------------------------------------

def run_dem_evaluation(
    split: str,
    args,
    seeds: list[int],
    device: torch.device,
) -> list[dict]:
    """Evaluate on DEM split (test_in or test_ood) across seeds.

    Returns list of per-seed result dicts.
    """
    from src.data.label_generation import build_dataset
    from src.evaluation.evaluate_dem import evaluate_dem_split

    rows = []
    splits_dir = PROJECT_ROOT / "data" / "splits"
    tiles_dir  = PROJECT_ROOT / "data" / "processed" / "tiles"

    # Check data exists
    try:
        dataset = build_dataset(split, splits_dir, tiles_dir)
    except Exception as exc:
        log.error("Could not build %s dataset: %s", split, exc)
        return []

    log.info("=" * 60)
    log.info("DEM Evaluation: split=%s, tiles=%d, seeds=%s",
             split, len(dataset), seeds)
    log.info("=" * 60)

    for seed in seeds:
        log.info("Seed %d / %s ...", seed, seeds)
        _set_seed(seed)
        pipeline = _build_pipeline(args, seed)

        t0 = time.time()
        result = evaluate_dem_split(
            split=split,
            pipeline_fn=pipeline,
            dataset=dataset,
            device=device,
            hazard_threshold=0.7,
            max_tiles=args.max_tiles,
        )
        elapsed = time.time() - t0

        row = _format_result(result, seed=seed)
        row["split"]      = split
        row["elapsed_s"]  = round(elapsed, 1)
        rows.append(row)
        log.info("[%s] seed=%d done in %.1fs", split, seed, elapsed)

    return rows


# ---------------------------------------------------------------------------
# HiRISE v3 cross-domain evaluation
# ---------------------------------------------------------------------------

def run_hirise_evaluation(args, seeds: list[int], device: torch.device) -> list[dict]:
    """Zero-shot cross-domain evaluation on HiRISE v3."""
    from src.data.hirise_loader import HiRISEv3Dataset
    from src.evaluation.evaluate_hirise import evaluate_hirise_v3

    hirise_dir = PROJECT_ROOT / "data" / "raw" / "hirise_v3"
    if not hirise_dir.exists():
        log.error("HiRISE v3 directory not found: %s", hirise_dir)
        return []

    rows = []

    log.info("=" * 60)
    log.info("HiRISE v3 Cross-Domain Evaluation (zero-shot)")
    log.info("=" * 60)

    for seed in seeds:
        _set_seed(seed)

        try:
            dataset = HiRISEv3Dataset(
                root_dir=hirise_dir,
                target_size=512,
            )
        except Exception as exc:
            log.error("Could not load HiRISE v3 dataset: %s", exc)
            continue

        pipeline = _build_pipeline(args, seed)

        t0 = time.time()
        result = evaluate_hirise_v3(
            pipeline_fn=pipeline,
            dataset=dataset,
            device=device,
            hazard_threshold=0.7,
            originals_only=True,
            max_crops=args.max_tiles,
        )
        elapsed = time.time() - t0

        row = _format_result(result, seed=seed)
        row["split"]     = "hirise_v3"
        row["elapsed_s"] = round(elapsed, 1)
        rows.append(row)
        log.info("HiRISE v3 seed=%d done in %.1fs", seed, elapsed)

    return rows


# ---------------------------------------------------------------------------
# Domain gap analysis (blueprint §19)
# ---------------------------------------------------------------------------

def compute_domain_gap(
    in_rows:  list[dict],
    ood_rows: list[dict],
    metric:   str = "hazard_recall",
) -> dict:
    """Compute domain gap = in_dist_metric − ood_metric.

    Blueprint §19 (Domain Gap Analysis): Three rows (B2/B4/PA-GNN) × 3 cols.
    This function handles the PA-GNN row.

    Returns
    -------
    dict with in_dist, ood, gap (as mean values across seeds)
    """
    def _mean(rows, key):
        vals = [r[key] for r in rows if key in r and isinstance(r[key], (int, float))]
        return float(np.mean(vals)) if vals else float("nan")

    in_val  = _mean(in_rows,  metric)
    ood_val = _mean(ood_rows, metric)
    gap     = in_val - ood_val if not (np.isnan(in_val) or np.isnan(ood_val)) else float("nan")

    return {
        "in_dist_recall": in_val,
        "ood_recall":     ood_val,
        "domain_gap":     gap,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PA-GNN — Full evaluation across all three protocols",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--split", choices=["test_in", "test_ood", "hirise", "all"],
        default="all",
        help="Which evaluation to run (default: all)"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 123, 7],
        help="Random seeds (blueprint: 3 seeds; default: 42 123 7)"
    )
    parser.add_argument(
        "--max-tiles", type=int, default=None,
        help="Max tiles/crops per split (omit for full evaluation)"
    )
    parser.add_argument(
        "--checkpoints", default="checkpoints",
        help="Checkpoint directory (default: checkpoints)"
    )
    parser.add_argument(
        "--mc-passes", type=int, default=5,
        help="MC Dropout passes (default: 5; use 3 if inference > 5s)"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device: auto | cuda | cpu (default: auto)"
    )
    parser.add_argument(
        "--figures", action="store_true",
        help="Generate publication figures (Fig 8, Fig 9)"
    )
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    device = _device(args.device)
    log.info("Device: %s | Seeds: %s | Max tiles: %s",
             device, args.seeds, args.max_tiles)

    all_results = {}
    t_total = time.time()

    # ── In-distribution DEM ──────────────────────────────────────────────
    if args.split in ("test_in", "all"):
        in_rows = run_dem_evaluation("test_in", args, args.seeds, device)
        if in_rows:
            _save_csv(in_rows, TABLES_DIR / "eval_dem_in.csv")
            all_results["test_in"] = in_rows

    # ── OOD DEM ──────────────────────────────────────────────────────────
    if args.split in ("test_ood", "all"):
        ood_rows = run_dem_evaluation("test_ood", args, args.seeds, device)
        if ood_rows:
            _save_csv(ood_rows, TABLES_DIR / "eval_dem_ood.csv")
            all_results["test_ood"] = ood_rows

    # ── HiRISE v3 cross-domain ───────────────────────────────────────────
    if args.split in ("hirise", "all"):
        hirise_rows = run_hirise_evaluation(args, args.seeds, device)
        if hirise_rows:
            _save_csv(hirise_rows, TABLES_DIR / "eval_hirise_v3.csv")
            all_results["hirise_v3"] = hirise_rows

    # ── Domain gap analysis ──────────────────────────────────────────────
    if "test_in" in all_results and "test_ood" in all_results:
        gap_result = compute_domain_gap(
            all_results["test_in"],
            all_results["test_ood"],
            metric="hazard_recall",
        )
        gap_row = {
            "system":          "PA-GNN (Proposed)",
            "in_dist_recall":  gap_result["in_dist_recall"],
            "ood_recall":      gap_result["ood_recall"],
            "domain_gap":      gap_result["domain_gap"],
        }
        _save_csv([gap_row], TABLES_DIR / "domain_gap_pagnn.csv")
        log.info(
            "Domain gap: in_dist=%.4f  OOD=%.4f  gap=%.4f",
            gap_result["in_dist_recall"],
            gap_result["ood_recall"],
            gap_result["domain_gap"],
        )

        if args.figures:
            try:
                from src.visualization import fig9_domain_gap_table
                fig9_data = {
                    "PA-GNN (Proposed)": gap_result,
                    # B2 and B4 domain gap entries would be added by run_ablations.py
                }
                fig9_domain_gap_table(fig9_data, FIGURES_DIR / "fig9_domain_gap.png")
            except Exception as exc:
                log.warning("Figure 9 generation failed: %s", exc)

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    log.info("=" * 60)
    log.info("Full evaluation complete in %.1fs", elapsed)
    log.info("Results saved to: %s", TABLES_DIR)
    log.info("=" * 60)

    # Persist all results as JSON for downstream figure generation
    results_json = LOGS_DIR / "evaluate_all_results.json"
    serialisable = {}
    for split, rows in all_results.items():
        serialisable[split] = rows
    with open(results_json, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
    log.info("Full results JSON: %s", results_json)


if __name__ == "__main__":
    main()
