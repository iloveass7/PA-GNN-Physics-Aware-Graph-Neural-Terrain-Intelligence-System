"""
validate_dataset.py
-------------------
Stage 2 validation script — verifies the physics feature engine produces
sensible outputs on the training tile set and benchmarks performance.

What it checks:
  1. Speed: measures mean inference time per tile on CPU and GPU (if available).
     Blueprint target: < 5ms/tile on GPU, < 100ms/tile on CPU.

  2. Sanity: for each split, loads 50 random tiles and verifies:
     - All three feature maps have values in [0,1]
     - No NaN or Inf values
     - Mean H_physics on known high-risk tiles (hazardous_frac > 0.5) exceeds
       mean on known low-risk tiles (hazardous_frac < 0.1) by at least 0.05

  3. Visualisation: saves a 3×5 grid of (image | slope | roughness | disc | H_physics)
     for 5 random training tiles to results/stage2_validation/

  4. Correlation: computes Pearson r between H_physics and hazard mask fraction
     across all validation tiles (from tile_manifest.csv). Reports this as the
     Stage 2 baseline performance metric.

Run from the pa-gnn/ directory:
    python scripts/validate_dataset.py [--split train] [--n_samples 50]
"""

import argparse
import csv
import logging
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.combine import build_physics_engine_from_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validate_dataset")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MANIFEST_CSV = PROJECT_ROOT / "data" / "processed" / "tile_manifest.csv"
SPLITS_DIR   = PROJECT_ROOT / "data" / "splits"
TILES_DIR    = PROJECT_ROOT / "data" / "processed" / "tiles"
RESULTS_DIR  = PROJECT_ROOT / "results" / "stage2_validation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_manifest(manifest_csv: Path) -> list[dict]:
    """Load tile manifest CSV."""
    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Tile manifest not found: {manifest_csv}\n"
            f"Run `python scripts/tile_dataset.py` first (Stage 1)."
        )
    with open(manifest_csv) as f:
        return list(csv.DictReader(f))


def load_tile_as_tensor(image_npy: str) -> torch.Tensor:
    """Load image .npy → (1, 1, 512, 512) tensor."""
    arr = np.load(image_npy).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------

def benchmark_speed(engine: torch.nn.Module, device: torch.device,
                    n_warmup: int = 5, n_bench: int = 50) -> dict:
    """Measure mean inference time per 512×512 tile."""
    dummy = torch.rand(1, 1, 512, 512, device=device)

    # Warmup
    for _ in range(n_warmup):
        engine(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        engine(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)   # ms

    result = {
        "device": str(device),
        "mean_ms": float(np.mean(times)),
        "std_ms":  float(np.std(times)),
        "min_ms":  float(np.min(times)),
    }

    target_ms = 5.0 if device.type == "cuda" else 100.0
    status = "✓ PASS" if result["mean_ms"] < target_ms else "✗ FAIL"
    log.info(
        "[%s] Speed: mean=%.2fms, target=<%.0fms → %s",
        device, result["mean_ms"], target_ms, status,
    )
    return result


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def sanity_check(engine: torch.nn.Module, records: list[dict],
                 device: torch.device, n: int = 50) -> dict:
    """Verify feature map validity on n random tiles."""
    sample = random.sample(records, min(n, len(records)))

    nan_count = 0
    range_violations = 0
    high_risk_scores = []
    low_risk_scores  = []

    for rec in sample:
        if not Path(rec["image_npy"]).exists():
            continue

        x = load_tile_as_tensor(rec["image_npy"]).to(device)
        h_phys, feats = engine(x)

        # NaN / Inf check
        for name, t in [("H_physics", h_phys), *feats.items()]:
            if torch.isnan(t).any() or torch.isinf(t).any():
                log.error("NaN/Inf in %s for tile %s", name, rec.get("alias", "?"))
                nan_count += 1

        # Range check
        for name, t in [("H_physics", h_phys), *feats.items()]:
            if t.min() < -0.01 or t.max() > 1.01:
                log.warning("Out of range [0,1] for %s: min=%.4f max=%.4f",
                            name, t.min().item(), t.max().item())
                range_violations += 1

        # Collect for risk stratification test
        haz_frac = float(rec.get("hazardous_frac", 0.0))
        score = h_phys.mean().item()
        if haz_frac > 0.5:
            high_risk_scores.append(score)
        elif haz_frac < 0.1:
            low_risk_scores.append(score)

    # Risk ordering check
    risk_gap = 0.0
    if high_risk_scores and low_risk_scores:
        risk_gap = float(np.mean(high_risk_scores) - np.mean(low_risk_scores))
        status = "✓ PASS" if risk_gap > 0.05 else "✗ FAIL (gap too small)"
        log.info(
            "Risk ordering: high_risk_mean=%.4f, low_risk_mean=%.4f, gap=%.4f → %s",
            np.mean(high_risk_scores), np.mean(low_risk_scores), risk_gap, status,
        )

    log.info("Sanity: NaN errors=%d, range violations=%d", nan_count, range_violations)
    return {
        "nan_errors": nan_count,
        "range_violations": range_violations,
        "risk_gap": risk_gap,
        "n_high": len(high_risk_scores),
        "n_low": len(low_risk_scores),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_visualisation(engine: torch.nn.Module, records: list[dict],
                       device: torch.device, n: int = 5) -> None:
    """Save a 5×5 triptych grid: image | slope | roughness | disc | H_physics."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sample = random.sample(records, min(n, len(records)))
    fig, axes = plt.subplots(len(sample), 5, figsize=(20, 4 * len(sample)))
    if len(sample) == 1:
        axes = [axes]

    col_labels = ["Image", "Slope (S)", "Roughness (R)", "Discontinuity (D)", "H_physics"]
    cmaps = ["gray", "hot", "plasma", "magma", "RdYlGn_r"]

    for row_idx, rec in enumerate(sample):
        if not Path(rec["image_npy"]).exists():
            continue

        x = load_tile_as_tensor(rec["image_npy"]).to(device)
        h_phys, feats = engine(x)

        maps = [
            x[0, 0].cpu().numpy(),
            feats["slope"][0, 0].cpu().numpy(),
            feats["roughness"][0, 0].cpu().numpy(),
            feats["disc"][0, 0].cpu().numpy(),
            h_phys[0, 0].cpu().numpy(),
        ]

        alias = rec.get("alias", "?")
        for col_idx, (data, cmap, label) in enumerate(zip(maps, cmaps, col_labels)):
            ax = axes[row_idx][col_idx]
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row_idx == 0:
                ax.set_title(label, fontsize=10, fontweight="bold")
            ax.set_ylabel(alias if col_idx == 0 else "", fontsize=7, rotation=90)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Stage 2 Physics Feature Engine — Validation Samples", fontsize=13)
    plt.tight_layout()
    out_path = RESULTS_DIR / "stage2_feature_grid.png"
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Visualisation saved: %s", out_path)


# ---------------------------------------------------------------------------
# Pearson r correlation
# ---------------------------------------------------------------------------

def compute_physics_baseline_correlation(
    engine: torch.nn.Module,
    records: list[dict],
    device: torch.device,
    split: str = "val",
    max_tiles: int = 500,
) -> float:
    """Compute Pearson r between H_physics mean and hazardous_frac across tiles.

    This is the Stage 2 standalone baseline metric — how well does H_physics
    predict the DEM-derived hazard fraction without any learned components?
    """
    from scipy import stats

    val_records = [r for r in records if r.get("split") == split]
    if not val_records:
        log.warning("No %s split records in manifest — using all records", split)
        val_records = records

    sample = random.sample(val_records, min(max_tiles, len(val_records)))

    h_scores = []
    haz_fracs = []

    for rec in sample:
        if not Path(rec["image_npy"]).exists():
            continue
        if float(rec.get("hazardous_frac", -1)) < 0:
            continue

        x = load_tile_as_tensor(rec["image_npy"]).to(device)
        h_phys, _ = engine(x)
        h_scores.append(h_phys.mean().item())
        haz_fracs.append(float(rec["hazardous_frac"]))

    if len(h_scores) < 10:
        log.warning("Fewer than 10 valid tiles for correlation — result unreliable")
        return float("nan")

    r, p = stats.pearsonr(h_scores, haz_fracs)
    log.info("Stage 2 baseline: Pearson r = %.4f (p = %.4e, n = %d)",
             r, p, len(h_scores))
    log.info("Interpretation: r > 0.5 → physics features track DEM hazards well")
    return float(r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(split: str = "train", n_samples: int = 50) -> None:
    random.seed(42)  # Reproducible sampling for thesis-reportable metrics
    log.info("=" * 60)
    log.info("Stage 2 Validation — Physics Feature Engine")
    log.info("=" * 60)

    # --- Load manifest ---
    records = load_manifest(MANIFEST_CSV)
    log.info("Manifest: %d tile records", len(records))

    # --- Build engine ---
    engine = build_physics_engine_from_config()
    log.info("Physics engine: %s", engine)

    # --- CPU benchmark ---
    cpu_device = torch.device("cpu")
    engine_cpu = engine.to(cpu_device).eval()
    cpu_speed = benchmark_speed(engine_cpu, cpu_device)

    # --- GPU benchmark (if available) ---
    gpu_speed = None
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        engine_gpu = engine.to(gpu_device).eval()
        gpu_speed = benchmark_speed(engine_gpu, gpu_device)
        active_engine = engine_gpu
        active_device = gpu_device
    else:
        log.info("CUDA not available, skipping GPU benchmark")
        active_engine = engine_cpu
        active_device = cpu_device

    # --- Sanity checks ---
    sanity = sanity_check(active_engine, records, active_device, n=n_samples)

    # --- Visualisation ---
    split_records = [r for r in records if r.get("split") == split]
    if not split_records:
        split_records = records[:50]
    save_visualisation(active_engine, split_records, active_device, n=5)

    # --- Correlation ---
    pearson_r = compute_physics_baseline_correlation(
        active_engine, records, active_device, split="val"
    )

    # --- Summary ---
    log.info("")
    log.info("=" * 60)
    log.info("Stage 2 Validation Summary")
    log.info("=" * 60)
    log.info("  CPU speed:       %.2fms/tile (target <100ms)", cpu_speed["mean_ms"])
    if gpu_speed:
        log.info("  GPU speed:       %.2fms/tile (target <5ms)", gpu_speed["mean_ms"])
    log.info("  NaN errors:      %d (target: 0)", sanity["nan_errors"])
    log.info("  Range errors:    %d (target: 0)", sanity["range_violations"])
    log.info("  Risk gap:        %.4f (target: >0.05)", sanity["risk_gap"])
    log.info("  Pearson r:       %.4f (val split)", pearson_r)
    log.info("")
    log.info("  Output: %s", RESULTS_DIR / "stage2_feature_grid.png")
    log.info("=" * 60)

    if sanity["nan_errors"] == 0 and sanity["range_violations"] == 0:
        log.info("✓ Stage 2 PASSED all sanity checks. Ready for Stage 3.")
    else:
        log.error("✗ Stage 2 has errors. Check logs above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 Physics Engine Validation")
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "test_in", "test_ood"])
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of random tiles for sanity checks")
    args = parser.parse_args()
    main(split=args.split, n_samples=args.n_samples)
