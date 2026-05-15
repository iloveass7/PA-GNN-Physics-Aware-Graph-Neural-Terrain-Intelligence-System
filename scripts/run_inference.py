"""
run_inference.py
----------------
Single-image CLI inference with per-stage timing.

Blueprint §23 (scripts/run_inference.py):
  CLI entry point — one action: run the full PA-GNN pipeline on a single
  image tile and produce all outputs (risk map, uncertainty map, path).

Run from pa-gnn/ directory:
    python scripts/run_inference.py --image path/to/tile.png \\
                                    --start 50,50 --goal 460,460 \\
                                    --output results/inference/

Outputs:
    results/inference/<tile_stem>/
        risk_comparison.png   — Figure 3 row
        uncertainty.png       — Figure 6 row
        path_overlay.png      — Figure 7 row
        result_summary.json   — timing + metrics
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_inference")


# ---------------------------------------------------------------------------
# Image loader
# ---------------------------------------------------------------------------

def load_image(path: str | Path, target_size: int = 512) -> torch.Tensor:
    """Load a single image tile as a (1, 3, 512, 512) float tensor.

    Supports: PNG, JPG, TIFF (single-channel or multi-channel).
    Grayscale channels are replicated to 3 channels for model compatibility.

    Parameters
    ----------
    path        : image file path
    target_size : resize target (blueprint: 512×512)

    Returns
    -------
    torch.Tensor of shape (1, 3, H, W), float32, values in [0, 1]
    """
    from PIL import Image
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path)

    # Convert to grayscale then replicate to 3 channels
    img = img.convert("L")   # (H, W) single channel
    arr = np.array(img, dtype=np.float32) / 255.0

    # Normalise per-tile to [0, 1]
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)

    # To tensor and resize
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
    if arr.shape[0] != target_size or arr.shape[1] != target_size:
        t = F.interpolate(t, size=(target_size, target_size),
                          mode="bilinear", align_corners=False)

    # Replicate to 3 channels (blueprint §10)
    return t.repeat(1, 3, 1, 1)   # (1, 3, 512, 512)


# ---------------------------------------------------------------------------
# Output saver
# ---------------------------------------------------------------------------

def save_inference_outputs(
    result_dict: dict,
    tile_name: str,
    output_dir: Path,
    image_tensor: torch.Tensor,
) -> None:
    """Save all visualisation outputs for one inference run."""
    from src.visualization import (
        fig3_risk_comparison,
        fig6_uncertainty_maps,
        fig7_path_comparison,
    )

    tile_dir = output_dir / tile_name
    tile_dir.mkdir(parents=True, exist_ok=True)

    image_np = image_tensor[0, 0].cpu().numpy()   # (H, W) grayscale

    # Figure 3: risk comparison
    samples_f3 = [{
        "image":     image_np,
        "dem_gt":    None,
        "h_physics": result_dict.get("h_physics"),
        "h_learned": result_dict.get("h_learned"),
        "alpha":     result_dict.get("alpha"),
        "h_final":   result_dict.get("h_final"),
        "label":     tile_name,
    }]
    if any(v is not None for v in samples_f3[0].values()):
        try:
            fig3_risk_comparison(samples_f3, tile_dir / "risk_comparison.png", dpi=150)
        except Exception as exc:
            log.warning("Figure 3 failed: %s", exc)

    # Figure 6: uncertainty
    unc = result_dict.get("uncertainty")
    h_final = result_dict.get("h_final")
    if unc is not None and h_final is not None:
        samples_f6 = [{
            "image": image_np,
            "h_final": h_final if isinstance(h_final, np.ndarray) else h_final.numpy(),
            "uncertainty": unc if isinstance(unc, np.ndarray) else np.array(unc),
            "label": tile_name,
        }]
        try:
            fig6_uncertainty_maps(samples_f6, tile_dir / "uncertainty.png", dpi=150)
        except Exception as exc:
            log.warning("Figure 6 failed: %s", exc)

    # Figure 7: path overlay (PA-GNN path only)
    waypoints = result_dict.get("path_waypoints", [])
    if waypoints and h_final is not None:
        hfn = h_final if isinstance(h_final, np.ndarray) else h_final.numpy()
        samples_f7 = [{
            "h_final":    hfn,
            "path_b1":    [],       # Not computed in single-image mode
            "path_b4":    [],
            "path_pagnn": waypoints,
            "label":      tile_name,
        }]
        try:
            fig7_path_comparison(samples_f7, tile_dir / "path_overlay.png", dpi=150)
        except Exception as exc:
            log.warning("Figure 7 failed: %s", exc)

    # Save JSON summary
    summary = {
        "tile":          tile_name,
        "path_found":    result_dict.get("path_found", False),
        "n_waypoints":   len(waypoints),
        "timings":       result_dict.get("timings", {}),
        "total_time_s":  sum(result_dict.get("timings", {}).values()),
    }

    with open(tile_dir / "result_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Outputs saved to %s", tile_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PA-GNN single-image inference with timing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to input image tile (PNG/JPG/TIFF)"
    )
    parser.add_argument(
        "--start", default="50,50",
        help="Path start position as 'row,col' (default: 50,50)"
    )
    parser.add_argument(
        "--goal", default="460,460",
        help="Path goal position as 'row,col' (default: 460,460)"
    )
    parser.add_argument(
        "--output", default="results/inference",
        help="Output directory (default: results/inference)"
    )
    parser.add_argument(
        "--checkpoints", default="checkpoints",
        help="Checkpoint directory (default: checkpoints)"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device: auto | cuda | cpu (default: auto)"
    )
    parser.add_argument(
        "--mc-passes", type=int, default=5,
        help="MC Dropout passes for uncertainty (default: 5)"
    )
    parser.add_argument(
        "--no-path", action="store_true",
        help="Skip path planning (faster; stages 2–6 only)"
    )
    parser.add_argument(
        "--image-size", type=int, default=512,
        help="Target image size (default: 512)"
    )
    args = parser.parse_args()

    # ── Parse start / goal ────────────────────────────────────────────────
    try:
        start = tuple(int(x) for x in args.start.split(","))
        goal  = tuple(int(x) for x in args.goal.split(","))
        assert len(start) == 2 and len(goal) == 2
    except Exception:
        log.error("--start and --goal must be 'row,col' integer pairs")
        sys.exit(1)

    # ── Load image ────────────────────────────────────────────────────────
    log.info("Loading image: %s", args.image)
    t_load = time.perf_counter()
    image = load_image(args.image, target_size=args.image_size)
    log.info("Image loaded in %.3fs  shape=%s", time.perf_counter() - t_load, tuple(image.shape))

    # ── Build pipeline ────────────────────────────────────────────────────
    from src.pipeline import PAGNNPipeline, PipelineConfig

    cfg = PipelineConfig(
        cnn_checkpoint=str(PROJECT_ROOT / args.checkpoints / "cnn_best.pt"),
        fusion_checkpoint=str(PROJECT_ROOT / args.checkpoints / "fusion_best.pt"),
        gnn_checkpoint=str(PROJECT_ROOT / args.checkpoints / "gnn_best.pt"),
        mc_passes=args.mc_passes,
        device=args.device,
    )
    pipeline = PAGNNPipeline(cfg)

    # ── Run inference ─────────────────────────────────────────────────────
    log.info("Running inference  start=%s  goal=%s ...", start, goal)
    t_inf = time.perf_counter()
    result = pipeline(image, start=start, goal=goal,
                      run_path_planning=not args.no_path)
    total_s = time.perf_counter() - t_inf

    # ── Report timings ────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Inference complete in %.2fs", total_s)
    if isinstance(result, dict):
        timings = result.get("timings", {})
        for stage, t in timings.items():
            log.info("  %-20s %.3fs", stage, t)
        log.info("Path found: %s  (%d waypoints)",
                 result.get("path_found", False),
                 len(result.get("path_waypoints", [])))
    log.info("=" * 60)

    # Enforce 5-second target (blueprint §19)
    if total_s > 5.0:
        log.warning(
            "Inference time %.2fs exceeds 5s target. "
            "Consider reducing --mc-passes to 3.", total_s
        )

    # ── Save outputs ──────────────────────────────────────────────────────
    output_dir = PROJECT_ROOT / args.output
    tile_name  = Path(args.image).stem

    if isinstance(result, dict):
        save_inference_outputs(result, tile_name, output_dir, image)

    log.info("Done. Outputs in: %s/%s/", output_dir, tile_name)


if __name__ == "__main__":
    main()
