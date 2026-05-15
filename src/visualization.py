"""
visualization.py
----------------
Risk maps, uncertainty maps, graph visualisations, and path overlays.

Blueprint §21 (Required Figures):
  Fig 1 — Physics Feature Grid (5 cols × 3–5 rows)
  Fig 2 — Adaptive Graph Resolution Illustration
  Fig 3 — H_physics vs H_learned vs H_final vs α
  Fig 4 — MAE Pretraining Evidence
  Fig 5 — Before/After GATv2 Refinement
  Fig 6 — Uncertainty Map
  Fig 7 — Path Comparison (B1/B4/PA-GNN)
  Fig 8 — Main Results Table (all 10 baselines)
  Fig 9 — Domain Gap Table

All functions save to disk (300 DPI for publication figures) and return
the figure path.  They accept numpy arrays or torch tensors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Defer matplotlib import to avoid errors if headless
_MPL_LOADED = False


def _import_mpl():
    global _MPL_LOADED
    if not _MPL_LOADED:
        import matplotlib
        matplotlib.use("Agg")
        _MPL_LOADED = True
    import matplotlib.pyplot as plt
    return plt


def _to_np(x) -> np.ndarray:
    if x is None:
        return None
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
    except ImportError:
        pass
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# Figure 1 — Physics Feature Grid (§21)
# ---------------------------------------------------------------------------

def fig1_physics_feature_grid(
    samples: list[dict],
    output_path: str | Path,
    dpi: int = 300,
) -> Path:
    """Figure 1: physics feature grid.

    Blueprint §21 Figure 1:
      5 columns (original, S, R, D, H_physics) × N rows (N terrain types).

    Parameters
    ----------
    samples : list of dicts, each with:
        "image"    : (H, W) grayscale array
        "slope"    : (H, W) S feature
        "roughness": (H, W) R feature
        "discont"  : (H, W) D feature
        "h_physics": (H, W) combined physics risk
        "label"    : optional row label string
    output_path : output file path
    dpi         : figure DPI (300 for publication)

    Returns
    -------
    Path to saved figure.
    """
    plt = _import_mpl()
    n_rows = len(samples)
    col_titles = ["Original", "Slope (S)", "Roughness (R)", "Discontinuity (D)", "H_physics"]
    col_keys   = ["image", "slope", "roughness", "discont", "h_physics"]
    col_cmaps  = ["gray", "hot", "hot", "hot", "RdYlGn_r"]

    fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for row, sample in enumerate(samples):
        for col, (key, cmap, title) in enumerate(zip(col_keys, col_cmaps, col_titles)):
            ax = axes[row][col]
            data = _to_np(sample.get(key))
            if data is not None:
                ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes)
            if row == 0:
                ax.set_title(title, fontsize=11, fontweight="bold")
            if col == 0 and "label" in sample:
                ax.set_ylabel(sample["label"], fontsize=9)
            ax.axis("off")

    plt.suptitle("Figure 1 — Physics Feature Grid", fontsize=13)
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info("Figure 1 saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Figure 2 — Adaptive Graph Resolution (§21)
# ---------------------------------------------------------------------------

def fig2_adaptive_graph_resolution(
    samples: list[dict],
    output_path: str | Path,
    dpi: int = 300,
) -> Path:
    """Figure 2: adaptive graph resolution illustration.

    Blueprint §21 Figure 2:
      3 cols (tiles) × 3 rows:
        Row 1: original orbital image
        Row 2: H_physics complexity map with tier boundaries overlaid
        Row 3: superpixel graph, nodes coloured by tier, size ∝ area

    Parameters
    ----------
    samples : list of 3 dicts, each with:
        "image"        : (H, W) grayscale array
        "h_physics"    : (H, W) physics risk / complexity map
        "tier_map"     : (H, W) int array 0/1/2
        "node_centroids": (N, 2) array of (row, col) centroid coords
        "node_areas"   : (N,) normalised areas
        "node_tiers"   : (N,) int tier assignments
        "n_nodes"      : int — used in caption
        "label"        : optional column label
    output_path : output file path

    Returns
    -------
    Path to saved figure.
    """
    plt = _import_mpl()
    TIER_COLORS = {0: "#3498DB", 1: "#F39C12", 2: "#E74C3C"}  # blue/yellow/red
    TIER_LABELS = {0: "Flat", 1: "Complex", 2: "Hazard"}

    n_cols = len(samples)
    fig, axes = plt.subplots(3, n_cols, figsize=(7 * n_cols, 18))
    if n_cols == 1:
        axes = [[axes[r]] for r in range(3)]

    row_titles = ["Original Image", "H_physics + Tier Boundaries", "Superpixel Graph"]

    for col, sample in enumerate(samples):
        # Row 0: original image
        axes[0][col].imshow(_to_np(sample.get("image")), cmap="gray")
        axes[0][col].axis("off")
        if col == 0:
            axes[0][col].set_ylabel(row_titles[0], fontsize=11)
        n_nodes = sample.get("n_nodes", "?")
        axes[0][col].set_title(f"{sample.get('label', f'Tile {col}')} (N={n_nodes})",
                                fontsize=10)

        # Row 1: H_physics with tier overlay
        h_phys = _to_np(sample.get("h_physics"))
        ax1 = axes[1][col]
        if h_phys is not None:
            ax1.imshow(h_phys, cmap="RdYlGn_r", vmin=0, vmax=1, alpha=0.85)
        tier_map = _to_np(sample.get("tier_map"))
        if tier_map is not None:
            # Draw tier boundary contours
            try:
                from matplotlib.colors import ListedColormap
                import matplotlib.patches as mpatches
                tier_img = np.zeros((*tier_map.shape, 4), dtype=np.float32)
                for tier_id, color in TIER_COLORS.items():
                    mask = tier_map == tier_id
                    rgba = plt.matplotlib.colors.to_rgba(color, alpha=0.25)
                    tier_img[mask] = rgba
                ax1.imshow(tier_img)
                patches = [mpatches.Patch(color=c, label=TIER_LABELS[t])
                           for t, c in TIER_COLORS.items()]
                ax1.legend(handles=patches, loc="lower right", fontsize=7)
            except Exception:
                pass
        ax1.axis("off")
        if col == 0:
            ax1.set_ylabel(row_titles[1], fontsize=11)

        # Row 2: superpixel graph
        ax2 = axes[2][col]
        if h_phys is not None:
            ax2.imshow(h_phys, cmap="gray", alpha=0.3, vmin=0, vmax=1)
        centroids = sample.get("node_centroids")
        areas     = sample.get("node_areas")
        tiers     = sample.get("node_tiers")
        if centroids is not None and tiers is not None:
            centroids = np.asarray(centroids)
            tiers     = np.asarray(tiers)
            areas     = np.asarray(areas) if areas is not None else np.ones(len(tiers))
            for t, color in TIER_COLORS.items():
                mask = tiers == t
                if mask.sum() == 0:
                    continue
                sizes = 300 * areas[mask] + 20
                ax2.scatter(centroids[mask, 1], centroids[mask, 0],
                            s=sizes, c=color, alpha=0.8, edgecolors="white",
                            linewidths=0.5, label=TIER_LABELS[t])
            ax2.legend(loc="lower right", fontsize=7)
        ax2.axis("off")
        if col == 0:
            ax2.set_ylabel(row_titles[2], fontsize=11)

    plt.suptitle("Figure 2 — Adaptive Graph Resolution (node density ∝ terrain complexity)",
                 fontsize=12)
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info("Figure 2 saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Figure 3 — Risk Map Comparison (§21)
# ---------------------------------------------------------------------------

def fig3_risk_comparison(
    samples: list[dict],
    output_path: str | Path,
    dpi: int = 300,
) -> Path:
    """Figure 3: H_physics vs H_learned vs H_final vs α.

    Blueprint §21 Figure 3:
      6 columns: (original, DEM GT, H_physics, H_learned, α, H_final) × rows.
    """
    plt = _import_mpl()
    col_keys   = ["image", "dem_gt", "h_physics", "h_learned", "alpha", "h_final"]
    col_titles = ["Original", "DEM Ground Truth", "H_physics", "H_learned", "α", "H_final"]
    col_cmaps  = ["gray", "RdYlGn_r", "RdYlGn_r", "RdYlGn_r", "RdBu", "RdYlGn_r"]

    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 6, figsize=(24, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for row, sample in enumerate(samples):
        for col, (key, cmap, title) in enumerate(zip(col_keys, col_cmaps, col_titles)):
            ax = axes[row][col]
            data = _to_np(sample.get(key))
            if data is not None:
                ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
            if row == 0:
                ax.set_title(title, fontsize=10, fontweight="bold")
            if col == 0 and "label" in sample:
                ax.set_ylabel(sample["label"], fontsize=9)
            ax.axis("off")

    plt.suptitle("Figure 3 — Risk Map Comparison", fontsize=13)
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info("Figure 3 saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Figure 4 — MAE Pretraining Evidence (§21)
# ---------------------------------------------------------------------------

def fig4_mae_pretraining(
    loss_history: list[float],
    reconstruction_samples: list[dict],
    output_path: str | Path,
    dpi: int = 300,
) -> Path:
    """Figure 4: MAE pretraining evidence.

    Blueprint §21 Figure 4:
      Panel 1: reconstruction loss curve over 200 epochs.
      Panel 2: 3 example tiles — masked input / reconstructed / original.

    Parameters
    ----------
    loss_history            : list of per-epoch reconstruction losses
    reconstruction_samples  : list of dicts with "masked", "reconstructed", "original"
    output_path             : save path
    """
    plt = _import_mpl()
    n_tiles = len(reconstruction_samples)
    fig = plt.figure(figsize=(20, 4 + 4 * n_tiles))

    # Loss curve
    ax_loss = fig.add_subplot(n_tiles + 1, 1, 1)
    ax_loss.plot(range(1, len(loss_history) + 1), loss_history, color="#2196F3", lw=2)
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Reconstruction MSE", fontsize=11)
    ax_loss.set_title("MAE Pretraining Loss Curve (200 epochs)", fontsize=12)
    ax_loss.grid(True, alpha=0.3)

    # Reconstruction examples
    for i, sample in enumerate(reconstruction_samples):
        for col, (key, title) in enumerate([("masked", "Masked Input (25% visible)"),
                                             ("reconstructed", "Reconstructed"),
                                             ("original", "Original")]):
            ax = fig.add_subplot(n_tiles + 1, 3, 3 + 3 * i + col + 1)
            data = _to_np(sample.get(key))
            if data is not None:
                ax.imshow(data, cmap="gray")
            ax.set_title(f"{title}" if i == 0 else "", fontsize=9)
            ax.axis("off")

    plt.suptitle("Figure 4 — MAE Self-Supervised Pretraining Evidence", fontsize=13)
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info("Figure 4 saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Figure 5 — Before/After GATv2 (§21)
# ---------------------------------------------------------------------------

def fig5_gnn_refinement(
    samples: list[dict],
    output_path: str | Path,
    dpi: int = 300,
) -> Path:
    """Figure 5: before and after GATv2 refinement.

    Blueprint §21 Figure 5:
      Node-coloured graph plots for 3 tiles.
      Left: H_final values (pre-GNN). Right: p̂_i values (post-GNN).

    Parameters
    ----------
    samples : list of dicts with:
        "h_physics"     : background map (H,W)
        "centroids"     : (N,2) centroid coords
        "h_final_node"  : (N,) pre-GNN node risk (mean H_final)
        "gnn_pred"      : (N,) post-GNN node risk p̂_i
        "label"         : optional tile label
    """
    plt = _import_mpl()
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    if n == 1:
        axes = [axes]

    for row, sample in enumerate(samples):
        bg    = _to_np(sample.get("h_physics"))
        cents = np.asarray(sample.get("centroids", []))
        pre   = np.asarray(sample.get("h_final_node", []))
        post  = np.asarray(sample.get("gnn_pred", []))
        label = sample.get("label", f"Tile {row}")

        for col, (vals, title) in enumerate([(pre,  f"H_final (pre-GNN)  — {label}"),
                                              (post, f"p̂_i (post-GNN)     — {label}")]):
            ax = axes[row][col]
            if bg is not None:
                ax.imshow(bg, cmap="gray", alpha=0.4, vmin=0, vmax=1)
            if len(cents) > 0 and len(vals) > 0:
                sc = ax.scatter(cents[:, 1], cents[:, 0],
                                c=vals, cmap="RdYlGn_r", vmin=0, vmax=1,
                                s=40, edgecolors="white", linewidths=0.3, zorder=5)
                plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    plt.suptitle("Figure 5 — GATv2 Refinement (Before vs After)", fontsize=13)
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info("Figure 5 saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Figure 6 — Uncertainty Map (§21)
# ---------------------------------------------------------------------------

def fig6_uncertainty_maps(
    samples: list[dict],
    output_path: str | Path,
    dpi: int = 300,
) -> Path:
    """Figure 6: uncertainty map.

    Blueprint §21 Figure 6:
      Side-by-side for 3 tiles: H_final risk map and U(x,y) uncertainty map.

    Parameters
    ----------
    samples : list of dicts with "image", "h_final", "uncertainty", "label"
    """
    plt = _import_mpl()
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = [axes]

    for row, sample in enumerate(samples):
        label = sample.get("label", f"Tile {row}")
        for col, (key, title, cmap) in enumerate([
            ("image",       f"Original — {label}",   "gray"),
            ("h_final",     "H_final (risk)",         "RdYlGn_r"),
            ("uncertainty", "U(x,y) uncertainty",     "YlOrRd"),
        ]):
            ax = axes[row][col]
            data = _to_np(sample.get(key))
            if data is not None:
                ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes)
            ax.set_title(title if row == 0 else "", fontsize=10)
            ax.axis("off")

    plt.suptitle("Figure 6 — Uncertainty Maps (MC Dropout Variance)", fontsize=13)
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info("Figure 6 saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Figure 7 — Path Comparison (§21)
# ---------------------------------------------------------------------------

def fig7_path_comparison(
    samples: list[dict],
    output_path: str | Path,
    dpi: int = 300,
) -> Path:
    """Figure 7: path comparison across 3 baselines.

    Blueprint §21 Figure 7:
      3 tiles × 3 paths (B1 Euclidean, B4 CNN-only, PA-GNN).
      Waypoints colour-coded by GNN risk score.

    Parameters
    ----------
    samples : list of dicts with:
        "h_final"      : (H,W) risk map
        "path_b1"      : list of {row, col, risk} for B1 (Euclidean A*)
        "path_b4"      : list of {row, col, risk} for B4 (CNN-MAE)
        "path_pagnn"   : list of {row, col, risk} for PA-GNN
        "label"        : tile label
    """
    plt = _import_mpl()
    path_styles = [
        ("path_b1",    "B1 Euclidean A*",  "#E74C3C"),
        ("path_b4",    "B4 CNN-MAE",       "#F39C12"),
        ("path_pagnn", "PA-GNN (Proposed)","#2ECC71"),
    ]

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(18, 6 * n))
    if n == 1:
        axes = [axes]

    for row, sample in enumerate(samples):
        h_final = _to_np(sample.get("h_final"))
        for col, (key, name, color) in enumerate(path_styles):
            ax = axes[row][col]
            if h_final is not None:
                ax.imshow(h_final, cmap="RdYlGn_r", vmin=0, vmax=1)
            path = sample.get(key, [])
            if path:
                coords = np.array([[w.get("row", 0), w.get("col", 0)] for w in path])
                risks  = [w.get("risk", 0.5) for w in path]
                ax.plot(coords[:, 1], coords[:, 0], "-", color=color, lw=2, alpha=0.8)
                sc = ax.scatter(coords[:, 1], coords[:, 0],
                                c=risks, cmap="RdYlGn_r", vmin=0, vmax=1,
                                s=25, zorder=5, edgecolors="white", linewidths=0.3)
                plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{name}\n{sample.get('label', '')}", fontsize=9)
            ax.axis("off")

    plt.suptitle("Figure 7 — Path Comparison: B1 vs B4 vs PA-GNN", fontsize=13)
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info("Figure 7 saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Figure 8 / 9 — Results Tables (§21)
# ---------------------------------------------------------------------------

def fig8_main_results_table(
    results: dict[str, dict[str, Any]],
    output_path: str | Path,
    dpi: int = 300,
) -> Path:
    """Figure 8: main results table (all 10 baselines).

    Blueprint §21 Figure 8: HCR±CI, PLR±std, success_rate, inference_time.
    Bold best value per column.

    Parameters
    ----------
    results : dict {baseline_name: {metric_name: value_or_(mean,ci)}}
    output_path : save path

    Returns
    -------
    Path to saved figure.
    """
    plt = _import_mpl()

    baseline_order = [
        "B1 Euclidean A*", "B2 Physics-only", "B3 CNN-ImageNet",
        "B4 CNN-MAE", "B5 Static Fusion", "B6 No-GNN",
        "B7 Fixed-300", "B8 RAG-GNN", "B9 No-uncertainty", "PA-GNN (Proposed)",
    ]
    metric_cols = ["HCR (%)", "PLR", "Success Rate (%)", "Inference (s)"]
    metric_keys = ["hcr", "plr", "success_rate", "inference_time_s"]

    row_labels = [b for b in baseline_order if b in results]
    table_data = []
    for bl in row_labels:
        row_vals = []
        for mk in metric_keys:
            v = results[bl].get(mk, "—")
            if isinstance(v, tuple):
                row_vals.append(f"{v[0]*100:.2f}±{v[1]*100:.2f}")
            elif isinstance(v, float):
                if mk in ("hcr", "success_rate"):
                    row_vals.append(f"{v*100:.2f}")
                elif mk == "plr":
                    row_vals.append(f"{v:.3f}")
                else:
                    row_vals.append(f"{v:.2f}")
            else:
                row_vals.append(str(v))
        table_data.append(row_vals)

    fig, ax = plt.subplots(figsize=(14, max(4, 0.55 * len(row_labels) + 2)))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=metric_cols,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    # Bold last row (Proposed)
    n_cols = len(metric_cols)
    n_rows = len(row_labels)
    for col_idx in range(n_cols):
        tbl[n_rows, col_idx + 1].set_text_props(fontweight="bold")
        tbl[n_rows, col_idx + 1].set_facecolor("#D5E8D4")

    ax.set_title("Figure 8 — Main Results Table (All Baselines)", pad=20, fontsize=12)
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info("Figure 8 saved: %s", out)
    return out


def fig9_domain_gap_table(
    results: dict[str, dict[str, float]],
    output_path: str | Path,
    dpi: int = 300,
) -> Path:
    """Figure 9: domain gap table.

    Blueprint §21 Figure 9:
      3 rows (physics-only B2, CNN-MAE B4, PA-GNN) ×
      3 cols (in-dist recall, OOD recall, domain gap = in−out).

    Parameters
    ----------
    results : dict with keys "B2", "B4", "PA-GNN", each with
              "in_dist_recall", "ood_recall" float values.
    """
    plt = _import_mpl()

    rows = ["B2 Physics-only", "B4 CNN-MAE", "PA-GNN (Proposed)"]
    col_labels = ["In-dist Recall", "OOD Recall", "Domain Gap (In−OOD)"]

    table_data = []
    for row in rows:
        entry = results.get(row, {})
        in_r  = entry.get("in_dist_recall", float("nan"))
        ood_r = entry.get("ood_recall",     float("nan"))
        gap   = in_r - ood_r if not (np.isnan(in_r) or np.isnan(ood_r)) else float("nan")
        table_data.append([f"{in_r:.4f}", f"{ood_r:.4f}", f"{gap:.4f}"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_data,
        rowLabels=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2)

    # Highlight domain gap column
    for row_idx in range(1, len(rows) + 1):
        tbl[row_idx, 3].set_facecolor("#FFF2CC")

    # Bold proposed row
    for col_idx in range(4):
        tbl[len(rows), col_idx].set_text_props(fontweight="bold")

    ax.set_title("Figure 9 — Domain Gap Analysis", pad=20, fontsize=12)
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info("Figure 9 saved: %s", out)
    return out
