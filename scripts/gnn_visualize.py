"""
visualize_gnn.py — Stage 6 GNN output visualizer.

Loads the trained GNN (gnn_best.pt), runs it on precomputed graphs, projects the
per-node risk predictions back to pixel space via pixel_membership, and shows a
4-panel row per tile:

    1. node target risk (y)      — superpixel nodes colored by DEM risk label
    2. node predicted risk (p̂)   — superpixel nodes colored by GNN prediction
    3. predicted risk projected to pixels (the "risk map" the GNN produces)
    4. error |p̂ - y| per node     — where the GNN is wrong

This is the figure that shows, spatially, what the GNN does — useful for the
showcase even though the held-out AUROC is modest.

Run from project root:
    python visualize_gnn.py --split test_in --n 5
    python visualize_gnn.py --split val --n 4
    python visualize_gnn.py --ckpt checkpoints/gnn_best.pt --split test_in --n 5
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gnn_model import PhysicsAwareGNN

GRAPHS_DIR = PROJECT_ROOT / "data" / "processed" / "graphs"
DEFAULT_OUT = PROJECT_ROOT / "results" / "stage6"
HAZARD_THRESH = 0.7


def load_model(ckpt_path):
    # match train_gnn defaults / config
    mcfg = {}
    cfg_path = PROJECT_ROOT / "configs" / "gnn.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(open(cfg_path)) or {}
        mcfg = cfg.get("model", {})
    physics_indices = mcfg.get("physics_features", mcfg.get("physics_indices", [2, 3]))
    model = PhysicsAwareGNN(
        in_features=14, hidden_dim=32, heads=4,
        physics_lambda_init=0.1, dropout_l1=0.3, dropout_l2=0.2,
        ffn_dropout=0.1, physics_indices=physics_indices,
    )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"] if "model_state_dict" in ck else ck
    model.load_state_dict(state)
    model.eval()
    return model


def project_to_pixels(node_vals, pixel_membership):
    """Map per-node values onto the (H, W) pixel grid via the membership map."""
    pm = pixel_membership
    if isinstance(pm, torch.Tensor):
        pm = pm.numpy()
    pm = pm.astype(np.int64)
    nv = np.asarray(node_vals)
    # guard: clip membership indices into range
    pm_clipped = np.clip(pm, 0, len(nv) - 1)
    return nv[pm_clipped]


def main():
    ap = argparse.ArgumentParser(description="Visualize Stage 6 GNN predictions")
    ap.add_argument("--split", default="test_in",
                    choices=["train", "val", "test_in", "test_ood"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "checkpoints" / "gnn_best.pt"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.ckpt)
    print(f"Loaded GNN from {args.ckpt}")

    files = sorted(glob.glob(str(GRAPHS_DIR / args.split / "*.pt")))
    if not files:
        print(f"No graphs in {GRAPHS_DIR/args.split}")
        sys.exit(1)
    if len(files) > args.n:
        pick = np.linspace(0, len(files) - 1, args.n).astype(int)
        files = [files[i] for i in pick]

    print(f"Visualizing {len(files)} graph(s) from split='{args.split}'")

    fig, axes = plt.subplots(len(files), 4, figsize=(18, 4.2 * len(files)))
    if len(files) == 1:
        axes = np.array([axes])

    for r, fp in enumerate(files):
        data = torch.load(fp, map_location="cpu", weights_only=False)
        stem = Path(fp).stem

        # node positions (x, y)
        pos = data.pos.numpy() if hasattr(data, "pos") else None
        xs = pos[:, 0] if pos is not None else None
        ys = pos[:, 1] if pos is not None else None

        y_true = data.y.numpy()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).numpy()

        err = np.abs(pred - y_true)
        node_auc = ""
        yb = (y_true > HAZARD_THRESH).astype(int)
        if yb.min() != yb.max():
            try:
                from sklearn.metrics import roc_auc_score
                node_auc = f"  AUC={roc_auc_score(yb, pred):.3f}"
            except Exception:
                pass

        # Panel 1: node target
        ax = axes[r][0]
        if xs is not None:
            sc = ax.scatter(xs, ys, c=y_true, cmap="RdYlGn_r", s=10, vmin=0, vmax=1)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{stem[:20]}…\nnode target (y)", fontsize=9)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")

        # Panel 2: node prediction
        ax = axes[r][1]
        if xs is not None:
            sc = ax.scatter(xs, ys, c=pred, cmap="RdYlGn_r", s=10, vmin=0, vmax=1)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"node predicted (p̂){node_auc}", fontsize=9)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")

        # Panel 3: prediction projected to pixels
        ax = axes[r][2]
        if hasattr(data, "pixel_membership"):
            risk_map = project_to_pixels(pred, data.pixel_membership)
            im = ax.imshow(risk_map, cmap="RdYlGn_r", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title("predicted risk → pixels", fontsize=9)
        else:
            ax.text(0.5, 0.5, "no pixel_membership", ha="center", va="center")
        ax.axis("off")

        # Panel 4: error
        ax = axes[r][3]
        if xs is not None:
            sc = ax.scatter(xs, ys, c=err, cmap="magma", s=10, vmin=0, vmax=1)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"|p̂ − y|  (MAE={err.mean():.3f})", fontsize=9)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")

        print(f"  {stem}: N={len(y_true)}, pred[{pred.min():.2f}–{pred.max():.2f}], MAE={err.mean():.3f}{node_auc}")

    plt.suptitle(f"Stage 6 GNN Predictions — split={args.split}", fontsize=13)
    plt.tight_layout()
    out_path = out_dir / f"gnn_viz_{args.split}.png"
    plt.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()