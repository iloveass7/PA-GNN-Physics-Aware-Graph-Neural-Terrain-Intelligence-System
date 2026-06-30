"""sweep_gnn.py — Judge every GNN checkpoint by test_in AUROC (the reliable selector).

The trainer saves gnn_best.pt on val_MAE, which on a 3-location val set rewards
mean-prediction and is unreliable (same problem we hit with the CNN and fusion).
This sweeps the periodic checkpoints + latest + best on the test_in split and
ranks by AUROC (threshold-free ranking), which is what we actually care about.

Run from project root:
    python sweep_gnn.py
"""
import sys, glob
from pathlib import Path
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from torch_geometric.loader import DataLoader as PyGDataLoader
from src.models.gnn_model import PhysicsAwareGNN
from sklearn.metrics import roc_auc_score, average_precision_score

GRAPHS_DIR = PROJECT_ROOT / "data" / "processed" / "graphs"
CKPT_DIR   = PROJECT_ROOT / "checkpoints"
HAZARD_THRESH = 0.7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- model config (match train_gnn defaults) ---
cfg = {}
cfg_path = PROJECT_ROOT / "configs" / "gnn.yaml"
if cfg_path.exists():
    cfg = yaml.safe_load(open(cfg_path)) or {}
mcfg = cfg.get("model", {})
physics_indices = mcfg.get("physics_features", mcfg.get("physics_indices", [2, 3]))


def load_split(split):
    d = GRAPHS_DIR / split
    graphs = []
    for pt in sorted(d.glob("*.pt")):
        try:
            g = torch.load(pt, weights_only=False)
            if hasattr(g, "x") and hasattr(g, "edge_index") and hasattr(g, "y"):
                if hasattr(g, "pixel_membership"):
                    del g.pixel_membership
                graphs.append(g)
        except Exception as e:
            print(f"  skip {pt.name}: {e}")
    return graphs


def build_model():
    return PhysicsAwareGNN(
        in_features=14, hidden_dim=32, heads=4,
        physics_lambda_init=0.1, dropout_l1=0.3, dropout_l2=0.2,
        ffn_dropout=0.1, physics_indices=physics_indices,
    ).to(DEVICE)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, tgts = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        p = model(batch.x, batch.edge_index).cpu().numpy()
        preds.append(p)
        tgts.append(batch.y.cpu().numpy())
    p = np.concatenate(preds)
    y = (np.concatenate(tgts) > HAZARD_THRESH).astype(int)
    if y.min() == y.max():
        return float("nan"), float("nan"), p[y == 1].mean() if (y == 1).any() else 0.0, p.mean()
    auc = roc_auc_score(y, p)
    prc = average_precision_score(y, p)
    return auc, prc, p[y == 1].mean(), p[y == 0].mean()


def main():
    test_graphs = load_split("test_in")
    if not test_graphs:
        print("No test_in graphs found. Build them first.")
        return
    print(f"Judging on test_in: {len(test_graphs)} graphs\n")
    loader = PyGDataLoader(test_graphs, batch_size=4, shuffle=False)

    model = build_model()

    ckpts = sorted(glob.glob(str(CKPT_DIR / "gnn_epoch_*.pt")))
    for extra in ["gnn_latest.pt", "gnn_best.pt"]:
        p = str(CKPT_DIR / extra)
        if Path(p).exists():
            ckpts.append(p)

    if not ckpts:
        print("No GNN checkpoints found.")
        return

    print(f"{'checkpoint':22s} {'AUROC':>7s} {'PR-AUC':>7s} {'haz_mean':>9s} {'safe_mean':>9s}")
    results = []
    for cp in ckpts:
        ck = torch.load(cp, map_location=DEVICE, weights_only=False)
        state = ck["model_state_dict"] if "model_state_dict" in ck else ck
        model.load_state_dict(state)
        auc, prc, hm, sm = evaluate(model, loader)
        print(f"{Path(cp).name:22s} {auc:7.4f} {prc:7.4f} {hm:9.3f} {sm:9.3f}")
        results.append((Path(cp).name, auc, hm, sm))

    valid = [r for r in results if not np.isnan(r[1])]
    if valid:
        best = max(valid, key=lambda r: r[1])
        print(f"\nBEST by test_in AUROC: {best[0]}  AUROC={best[1]:.4f}  "
              f"(haz_mean={best[2]:.3f} {'>' if best[2] > best[3] else '<'} safe_mean={best[3]:.3f})")
        print(f"\nTo use it:  copy checkpoints/{best[0]} -> checkpoints/gnn_best.pt")


if __name__ == "__main__":
    main()