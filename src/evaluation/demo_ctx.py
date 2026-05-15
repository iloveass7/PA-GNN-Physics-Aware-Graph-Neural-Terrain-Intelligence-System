"""
demo_ctx.py
-----------
Qualitative pipeline demo on CTX tiles.

Blueprint §5.2 (MurrayLab CTX Tiles — Pretraining and Demo):
  After training, run the full pipeline on 3–5 selected tiles and generate
  risk maps, uncertainty maps, α maps, and planned paths as publication
  figures.

Blueprint §21 (Required Figures):
  - Figure 1: Physics Feature Grid (5 cols × 3–5 rows)
  - Figure 2: Adaptive Graph Resolution Illustration
  - Figure 3: H_physics vs H_learned vs H_final vs α
  - Figure 5: Before and After GATv2 Refinement
  - Figure 6: Uncertainty Map
  - Figure 7: Path Comparison

Tile selection criteria (blueprint §5.2):
  - At least one smooth and one rough/high-contrast region per tile.
  - Reject tiles where >30% pixels are within 5% of tile min/max.

Usage:
    from src.evaluation.demo_ctx import run_ctx_demo
    run_ctx_demo(pipeline_fn, tile_paths, output_dir)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tile quality filter (blueprint §5.2)
# ---------------------------------------------------------------------------

def passes_quality_filter(
    tile: np.ndarray,
    saturation_fraction: float = 0.30,
    saturation_band: float = 0.05,
) -> bool:
    """Return True if tile passes the saturation quality filter.

    Reject tiles where more than `saturation_fraction` of pixels are
    within `saturation_band` of the tile min or max (edge-of-mosaic artefacts).

    Blueprint §5.2: reject tiles where >30% pixels within 5% of min/max.

    Parameters
    ----------
    tile                : (H, W) grayscale array, values in [0, 1]
    saturation_fraction : fraction threshold (blueprint: 0.30)
    saturation_band     : pixel range near min/max considered saturated (0.05)

    Returns
    -------
    bool — True if tile is usable for demo
    """
    lo = tile.min()
    hi = tile.max()

    near_lo = (tile - lo) < saturation_band * (hi - lo + 1e-8)
    near_hi = (hi - tile) < saturation_band * (hi - lo + 1e-8)
    near_either = near_lo | near_hi

    fraction_saturated = near_either.mean()
    return float(fraction_saturated) <= saturation_fraction


def select_demo_tiles(
    ctx_dir: str | Path,
    n_tiles: int = 5,
    max_candidates: int = 500,
    seed: int = 42,
) -> list[Path]:
    """Select n_tiles CTX tiles that pass the quality filter.

    Blueprint §5.2: choose tiles with visually diverse terrain — at least
    one smooth and one rough region. Selection is randomised with a fixed
    seed for reproducibility.

    Parameters
    ----------
    ctx_dir       : directory containing CTX .png tiles
    n_tiles       : target number of demo tiles (blueprint: 3–5)
    max_candidates: how many candidates to scan before giving up
    seed          : random seed for reproducibility

    Returns
    -------
    list of Path objects for selected tiles
    """
    from PIL import Image

    ctx_dir = Path(ctx_dir)
    all_tiles = sorted(ctx_dir.glob("*.png"))

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(all_tiles))[:max_candidates]
    candidates = [all_tiles[i] for i in indices]

    selected = []
    for tile_path in candidates:
        if len(selected) >= n_tiles:
            break
        try:
            img = Image.open(tile_path).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
            if passes_quality_filter(arr):
                selected.append(tile_path)
        except Exception as exc:
            log.debug("Skipping %s: %s", tile_path.name, exc)

    log.info("Selected %d / %d demo tiles from %s", len(selected), n_tiles, ctx_dir)
    return selected


# ---------------------------------------------------------------------------
# Main demo runner
# ---------------------------------------------------------------------------

def run_ctx_demo(
    pipeline_fn: Callable,
    tile_paths: list[str | Path],
    output_dir: str | Path,
    device: torch.device | None = None,
    start_frac: float = 0.1,
    goal_frac: float = 0.9,
) -> list[dict]:
    """Run the full pipeline on each CTX tile and save visualisations.

    Blueprint §5.2 Role 2: qualitative demo after training.

    Parameters
    ----------
    pipeline_fn  : callable(image_tensor) → dict with all pipeline outputs
    tile_paths   : list of CTX tile paths
    output_dir   : directory to save demo figures
    device       : torch.device (defaults to auto-detect)
    start_frac   : path start position as fraction of image size (row, col)
    goal_frac    : path goal position as fraction of image size

    Returns
    -------
    list of result dicts, one per tile
    """
    from PIL import Image
    import torch
    import torch.nn.functional as F

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for tile_idx, tile_path in enumerate(tile_paths):
        tile_path = Path(tile_path)
        log.info("Demo tile %d/%d: %s", tile_idx + 1, len(tile_paths), tile_path.name)

        try:
            # Load and preprocess
            img = Image.open(tile_path).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0

            # Resize to 512×512
            arr_t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            arr_t = F.interpolate(arr_t, size=(512, 512), mode="bilinear",
                                  align_corners=False)
            # Replicate grayscale → 3 channels (blueprint §10)
            image = arr_t.repeat(1, 3, 1, 1).to(device)  # (1,3,512,512)

            # Set start/goal in pixel coords
            H, W = 512, 512
            start = (int(start_frac * H), int(start_frac * W))
            goal  = (int(goal_frac  * H), int(goal_frac  * W))

            # Run pipeline
            result = pipeline_fn(image, start=start, goal=goal)
            result_dict = result if isinstance(result, dict) else vars(result)

            # Save all visualisations
            tile_out_dir = output_dir / f"tile_{tile_idx:02d}_{tile_path.stem}"
            tile_out_dir.mkdir(parents=True, exist_ok=True)

            _save_tile_visualisations(
                tile_idx=tile_idx,
                tile_name=tile_path.stem,
                image_arr=arr,
                result=result_dict,
                output_dir=tile_out_dir,
            )

            results.append({
                "tile": tile_path.name,
                "tile_idx": tile_idx,
                "output_dir": str(tile_out_dir),
                **{k: v for k, v in result_dict.items()
                   if isinstance(v, (int, float, str))},
            })

        except Exception as exc:
            log.error("Demo tile %s failed: %s", tile_path.name, exc, exc_info=True)

    log.info("CTX demo complete. Saved %d tile outputs to %s", len(results), output_dir)
    return results


# ---------------------------------------------------------------------------
# Visualisation helpers (internally used by run_ctx_demo)
# ---------------------------------------------------------------------------

def _save_tile_visualisations(
    tile_idx: int,
    tile_name: str,
    image_arr: np.ndarray,
    result: dict,
    output_dir: Path,
) -> None:
    """Save individual figures for a single demo tile.

    Produces subsets of the required Figures (§21) for this tile:
      - physics_features.png     (Figure 1 row)
      - risk_comparison.png      (Figure 3 row)
      - uncertainty.png          (Figure 6 row)
      - path_overlay.png         (Figure 7 row)
      - graph_viz.png            (Figure 2 row — if graph data available)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping visualisation for tile %s", tile_name)
        return

    cmap_risk  = "RdYlGn_r"
    cmap_unc   = "YlOrRd"
    cmap_gray  = "gray"
    cmap_alpha = "RdBu"

    # ── Figure 3: risk comparison ──────────────────────────────────────────
    h_physics = _get_map(result, "h_physics")
    h_learned = _get_map(result, "h_learned")
    h_final   = _get_map(result, "h_final")
    alpha_map = _get_map(result, "alpha")

    if h_physics is not None and h_final is not None:
        fig, axes = plt.subplots(1, 6, figsize=(24, 4))
        _imshow(axes[0], image_arr,  "Original",      cmap_gray)
        _imshow(axes[1], None,       "DEM GT",         cmap_risk, note="(from eval)")
        _imshow(axes[2], h_physics,  "H_physics",      cmap_risk)
        _imshow(axes[3], h_learned,  "H_learned",      cmap_risk)
        _imshow(axes[4], alpha_map,  "α (CNN trust)",  cmap_alpha)
        _imshow(axes[5], h_final,    "H_final",        cmap_risk)
        plt.suptitle(f"Tile {tile_idx}: Risk Map Comparison", fontsize=11)
        plt.tight_layout()
        plt.savefig(str(output_dir / "risk_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # ── Figure 6: uncertainty ─────────────────────────────────────────────
    uncertainty = _get_map(result, "uncertainty")
    if uncertainty is not None and h_final is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        _imshow(axes[0], image_arr,   "Original",         cmap_gray)
        _imshow(axes[1], h_final,     "H_final (risk)",   cmap_risk)
        _imshow(axes[2], uncertainty, "U(x,y) uncertainty", cmap_unc)
        plt.suptitle(f"Tile {tile_idx}: Uncertainty Map", fontsize=11)
        plt.tight_layout()
        plt.savefig(str(output_dir / "uncertainty.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # ── Figure 7: path overlay ────────────────────────────────────────────
    waypoints = result.get("path_waypoints")
    if waypoints and h_final is not None:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(h_final, cmap=cmap_risk, vmin=0, vmax=1)
        if waypoints:
            coords = np.array([[w.get("row", w.get("y", 0)),
                                w.get("col", w.get("x", 0))]
                               for w in waypoints])
            risks  = [w.get("risk", 0.5) for w in waypoints]
            sc = ax.scatter(coords[:, 1], coords[:, 0],
                            c=risks, cmap=cmap_risk, vmin=0, vmax=1,
                            s=30, zorder=5)
            ax.plot(coords[:, 1], coords[:, 0], "w-", lw=1.5, alpha=0.7)
            plt.colorbar(sc, ax=ax, label="GNN risk")
        ax.set_title(f"Tile {tile_idx}: PA-GNN Path", fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(str(output_dir / "path_overlay.png"), dpi=150, bbox_inches="tight")
        plt.close()

    log.debug("Visualisations saved: %s", output_dir)


def _get_map(result: dict, key: str) -> np.ndarray | None:
    val = result.get(key)
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return val.squeeze().detach().cpu().float().numpy()
    return np.asarray(val, dtype=np.float32).squeeze()


def _imshow(ax, data, title: str, cmap: str, note: str = "") -> None:
    """Helper to display one panel."""
    if data is not None:
        ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
    else:
        ax.text(0.5, 0.5, note or "N/A", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
