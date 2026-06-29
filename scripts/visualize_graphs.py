"""
visualize_graphs.py — Stage 5 graph-structure visualizer.

Renders precomputed PyG graph .pt files so you can (a) confirm the graphs are
sane before the GNN run and (b) produce figures for the defense.

For each sampled graph it draws a 4-panel row:
    1. node risk (y)        — superpixel nodes colored by DEM risk label
    2. graph edges          — KNN/RAG edges drawn over node positions
    3. tier map             — nodes colored by tier (flat/complex/hazard)
    4. node H_final (feat)  — nodes colored by a node feature (e.g. H_final), if present

Robust to field-name variation: it reads whatever exists on the Data object and
skips panels it can't build, so it won't crash on a layout mismatch.

Run from project root:
    python visualize_graphs.py --split val --n 5
    python visualize_graphs.py --split test_in --n 4 --out results/stage5
    python visualize_graphs.py --file data/processed/graphs/val/SomeTile_r0_c0.pt
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
from matplotlib.collections import LineCollection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPHS_DIR = PROJECT_ROOT / "data" / "processed" / "graphs"
DEFAULT_OUT = PROJECT_ROOT / "results" / "stage5"

# 14-dim node feature indices (blueprint §12). Used to label the 4th panel.
# 0,1 = pos(x,y) ; 2=S ; 3=R ; 4=D ; 5=H_physics ; 6=H_learned ; 7=H_final ;
# 8=alpha ; 9=area ; ... (exact tail varies; we only need a sensible default)
FEATURE_NAMES = {
    2: "slope", 3: "roughness", 4: "disc", 5: "H_physics",
    6: "H_learned", 7: "H_final", 8: "alpha", 9: "area",
}
DEFAULT_FEATURE_IDX = 7   # H_final — change with --feat


def _to_np(t):
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _get(data, *names):
    """Return the first attribute that exists on the Data object, else None."""
    for n in names:
        if hasattr(data, n):
            v = getattr(data, n)
            if v is not None:
                return v
    return None


def _node_xy(data):
    """Get node (x, y) pixel coordinates. Prefer `pos`; fall back to feats[:, :2]."""
    pos = _to_np(_get(data, "pos"))
    if pos is not None and pos.ndim == 2 and pos.shape[1] >= 2:
        return pos[:, 0], pos[:, 1]
    x = _to_np(_get(data, "x"))
    if x is not None and x.shape[1] >= 2:
        return x[:, 0], x[:, 1]
    return None, None


def visualize_one(data, ax_row, title_prefix=""):
    """Draw a 4-panel row for a single graph onto a list/array of 4 axes."""
    xs, ys = _node_xy(data)
    y_risk = _to_np(_get(data, "y"))
    tier   = _to_np(_get(data, "tier"))
    feats  = _to_np(_get(data, "x"))
    edge_index = _to_np(_get(data, "edge_index"))

    n = 0 if xs is None else len(xs)

    # --- Panel 1: node risk ---
    ax = ax_row[0]
    if xs is not None and y_risk is not None:
        sc = ax.scatter(xs, ys, c=y_risk, cmap="RdYlGn_r", s=12, vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{title_prefix}node risk (y)  N={n}", fontsize=9)
    else:
        ax.text(0.5, 0.5, "no y / pos", ha="center", va="center")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")

    # --- Panel 2: edges ---
    ax = ax_row[1]
    if xs is not None and edge_index is not None and edge_index.shape[0] == 2:
        segs = []
        # subsample edges if huge, to keep the figure readable/fast
        E = edge_index.shape[1]
        idx = np.arange(E)
        if E > 4000:
            idx = np.random.default_rng(0).choice(E, 4000, replace=False)
        for e in idx:
            a, b = edge_index[0, e], edge_index[1, e]
            if a < n and b < n:
                segs.append([(xs[a], ys[a]), (xs[b], ys[b])])
        lc = LineCollection(segs, colors="#3a6ea5", linewidths=0.3, alpha=0.5)
        ax.add_collection(lc)
        ax.scatter(xs, ys, c="#222", s=4)
        ax.set_title(f"edges  E={E}", fontsize=9)
        ax.set_xlim(xs.min() - 5, xs.max() + 5)
        ax.set_ylim(ys.min() - 5, ys.max() + 5)
    else:
        ax.text(0.5, 0.5, "no edge_index", ha="center", va="center")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")

    # --- Panel 3: tier ---
    ax = ax_row[2]
    if xs is not None and tier is not None:
        sc = ax.scatter(xs, ys, c=tier, cmap="viridis", s=12)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("tier (0=flat 1=complex 2=hazard)", fontsize=9)
    else:
        ax.text(0.5, 0.5, "no tier", ha="center", va="center")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")

    # --- Panel 4: a node feature (default H_final) ---
    ax = ax_row[3]
    fidx = visualize_one.feature_idx
    if xs is not None and feats is not None and feats.shape[1] > fidx:
        vals = feats[:, fidx]
        sc = ax.scatter(xs, ys, c=vals, cmap="RdYlGn_r", s=12)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        fname = FEATURE_NAMES.get(fidx, f"feat[{fidx}]")
        ax.set_title(f"node {fname}", fontsize=9)
    else:
        ax.text(0.5, 0.5, "no feats", ha="center", va="center")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")


visualize_one.feature_idx = DEFAULT_FEATURE_IDX


def main():
    ap = argparse.ArgumentParser(description="Visualize Stage 5 precomputed graphs")
    ap.add_argument("--split", default="val",
                    choices=["train", "val", "test_in", "test_ood"])
    ap.add_argument("--n", type=int, default=5, help="How many graphs to sample")
    ap.add_argument("--file", default=None, help="Visualize one specific .pt file")
    ap.add_argument("--feat", type=int, default=DEFAULT_FEATURE_IDX,
                    help="Node feature index for panel 4 (default 7=H_final)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    visualize_one.feature_idx = args.feat
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather files
    if args.file:
        files = [args.file]
    else:
        files = sorted(glob.glob(str(GRAPHS_DIR / args.split / "*.pt")))
        if not files:
            print(f"No .pt files in {GRAPHS_DIR/args.split}. "
                  f"Run precompute_graphs.py --split {args.split} first.")
            sys.exit(1)
        # evenly sample n across the split so we see varied terrain
        if len(files) > args.n:
            pick = np.linspace(0, len(files) - 1, args.n).astype(int)
            files = [files[i] for i in pick]

    print(f"Visualizing {len(files)} graph(s) from split='{args.split}'")

    fig, axes = plt.subplots(len(files), 4, figsize=(18, 4.2 * len(files)))
    if len(files) == 1:
        axes = np.array([axes])

    for r, fp in enumerate(files):
        try:
            data = torch.load(fp, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  failed to load {fp}: {e}")
            for c in range(4):
                axes[r][c].text(0.5, 0.5, "load failed", ha="center"); axes[r][c].axis("off")
            continue
        stem = Path(fp).stem
        nnodes = data.num_nodes if hasattr(data, "num_nodes") else \
                 (data.x.shape[0] if hasattr(data, "x") else "?")
        print(f"  {stem}: {nnodes} nodes")
        visualize_one(data, axes[r], title_prefix=f"{stem[:18]}…\n")

    plt.suptitle(f"Stage 5 Graphs — split={args.split}", fontsize=13)
    plt.tight_layout()
    out_path = out_dir / f"graph_viz_{args.split}.png"
    plt.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
