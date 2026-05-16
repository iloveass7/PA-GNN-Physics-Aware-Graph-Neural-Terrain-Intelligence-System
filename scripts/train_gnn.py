"""
train_gnn.py
------------
Stage 6 — GNN training on precomputed PyG graph files.

Blueprint §13:
  - Target: DEM-derived risk score per node (y field in PyG Data)
  - Loss: SmoothL1 (Huber) — robust to outlier nodes at DEM boundary gaps
  - Optimizer: Adam, LR 1e-3, weight decay 5e-4
  - Max epochs: 100, early stopping patience 15 on val_MAE
  - Batch size: 32 precomputed graphs

Run from project root:
  python scripts/train_gnn.py
  python scripts/train_gnn.py --config configs/gnn.yaml --epochs 100
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError:
    yaml = None

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError:
    raise ImportError("torch_geometric is required. Install via: pip install torch-geometric")

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

from src.models.gnn_model import PhysicsAwareGNN
from src.graph.edge_scorer import EdgeAffinityMLP
from src.evaluation.oversmoothing import log_layer_variances

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_gnn")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "model": {
        "in_features": 14,
        "hidden_dim": 32,
        "heads": 4,
        "physics_lambda_init": 0.1,
        "dropout_l1": 0.3,
        "dropout_l2": 0.2,
        "ffn_dropout": 0.1,
    },
    "training": {
        "optimizer": "Adam",
        "lr": 1e-3,
        "weight_decay": 5e-4,
        "max_epochs": 100,
        "batch_size": 32,
        "grad_clip": 1.0,
    },
    "early_stopping": {
        "patience": 15,
        "monitor": "val_MAE",
        "mode": "min",
    },
    "data": {
        "graph_dir": "data/processed/graphs",
    },
    "checkpoints": {
        "save_dir": "checkpoints",
        "best_name": "gnn_best.pt",
        "periodic_every": 10,
    },
    "logging": {
        "results_dir": "results/stage6",
        "log_csv": "results/stage6/gnn_train_log.csv",
    },
}


def load_config(config_path: str | None) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if config_path and yaml:
        p = Path(config_path)
        if p.exists():
            with open(p) as f:
                user_cfg = yaml.safe_load(f)
            if user_cfg:
                for section, vals in user_cfg.items():
                    if isinstance(vals, dict) and section in cfg:
                        cfg[section].update(vals)
                    else:
                        cfg[section] = vals
    return cfg


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_graph_split(graph_dir: Path, split: str) -> list[Data]:
    """Load all .pt graph files from a split directory."""
    split_dir = graph_dir / split
    if not split_dir.exists():
        log.warning("Split directory not found: %s", split_dir)
        return []

    graphs = []
    pt_files = sorted(split_dir.glob("*.pt"))
    for pt_file in pt_files:
        try:
            data = torch.load(pt_file, weights_only=False)
            if hasattr(data, "x") and hasattr(data, "edge_index") and hasattr(data, "y"):
                graphs.append(data)
        except Exception as e:
            log.debug("Skipping %s: %s", pt_file.name, e)

    log.info("Loaded %d graphs from %s", len(graphs), split_dir)
    return graphs


# ---------------------------------------------------------------------------
# Training and validation
# ---------------------------------------------------------------------------

def dynamic_graph_warmup(batch, edge_scorer, K=5):
    """Rebuild PyG batch edge_index using the learned EdgeAffinityMLP.
    
    Computes dense pairwise affinities within each graph in the batch,
    selects top K edges per node, and returns the new edge_index.
    """
    import torch_geometric.utils as pyg_utils

    N = batch.x.size(0)
    # pos: (N, 2), features: S, R, D, H_physics are at indices 2, 3, 4, 5
    pos = batch.pos
    physics = batch.x[:, 2:6]

    # Compute dense differences
    spatial_dist = torch.cdist(pos, pos)
    slope_diff = torch.abs(physics[:, 0].unsqueeze(1) - physics[:, 0].unsqueeze(0))
    roughness_diff = torch.abs(physics[:, 1].unsqueeze(1) - physics[:, 1].unsqueeze(0))
    uncertainty_diff = torch.abs(physics[:, 3].unsqueeze(1) - physics[:, 3].unsqueeze(0))

    # Mask out edges between different graphs in the batch
    same_graph_mask = (batch.batch.unsqueeze(1) == batch.batch.unsqueeze(0))

    affinities = edge_scorer(spatial_dist, slope_diff, roughness_diff, uncertainty_diff)
    # Zero out affinities between different graphs and self-loops
    affinities = affinities * same_graph_mask.float()
    affinities.fill_diagonal_(0.0)

    # Get top K edges
    k_query = min(K, affinities.size(1) - 1)
    _, topk_indices = affinities.topk(k_query, dim=1)

    # Build new edge_index
    row = torch.arange(N, device=batch.x.device).view(-1, 1).expand(-1, k_query).flatten()
    col = topk_indices.flatten()
    
    # Undirected
    edge_index = torch.stack([row, col], dim=0)
    edge_index = pyg_utils.to_undirected(edge_index)
    
    return edge_index


def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0, edge_scorer=None, warmup=False):
    """Train one epoch. Returns dict with loss and MAE."""
    model.train()
    if edge_scorer:
        edge_scorer.train()
        
    total_loss = 0.0
    total_mae = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        # Phase 7: Dynamic graph building during warmup
        edge_index = batch.edge_index
        if warmup and edge_scorer is not None:
            edge_index = dynamic_graph_warmup(batch, edge_scorer)

        pred = model(batch.x, edge_index)  # (total_nodes,)
        target = batch.y                          # (total_nodes,)

        loss = F.smooth_l1_loss(pred, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if edge_scorer:
            torch.nn.utils.clip_grad_norm_(edge_scorer.parameters(), grad_clip)
            
        optimizer.step()

        n = target.size(0)
        total_loss += loss.item() * n
        total_mae += (pred - target).abs().sum().item()
        total_nodes += n

    total_nodes = max(total_nodes, 1)
    return {
        "loss": total_loss / total_nodes,
        "MAE": total_mae / total_nodes,
    }


@torch.no_grad()
def validate(model, loader, device, hazard_threshold=0.7):
    """Validate. Returns dict with loss, MAE, and AUC-ROC."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_nodes = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index)
        target = batch.y

        loss = F.smooth_l1_loss(pred, target)
        n = target.size(0)
        total_loss += loss.item() * n
        total_mae += (pred - target).abs().sum().item()
        total_nodes += n

        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

    total_nodes = max(total_nodes, 1)
    result = {
        "loss": total_loss / total_nodes,
        "MAE": total_mae / total_nodes,
    }
    
    # Phase 9: Log oversmoothing metrics for the last batch
    if len(loader) > 0:
        smoothing_metrics = log_layer_variances(model, batch)
        result.update(smoothing_metrics)

    # AUC-ROC for binary hazard classification
    if roc_auc_score is not None:
        preds_cat = torch.cat(all_preds).numpy()
        targets_cat = torch.cat(all_targets).numpy()
        binary_targets = (targets_cat > hazard_threshold).astype(np.int32)
        if binary_targets.sum() > 0 and binary_targets.sum() < len(binary_targets):
            try:
                result["AUC"] = roc_auc_score(binary_targets, preds_cat)
            except ValueError:
                result["AUC"] = 0.0
        else:
            result["AUC"] = 0.0

    return result


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 6 — GNN Training")
    parser.add_argument("--config", default="configs/gnn.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    ecfg = cfg["early_stopping"]

    # CLI overrides
    if args.epochs:
        tcfg["max_epochs"] = args.epochs
    if args.batch_size:
        tcfg["batch_size"] = args.batch_size
    if args.lr:
        tcfg["lr"] = args.lr

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=" * 60)
    log.info("Stage 6 — Physics-Aware GATv2 + FFN Training")
    log.info("=" * 60)
    log.info("Device: %s", device)
    log.info("Seed: %d", args.seed)

    # --- Load graphs ---
    graph_dir = PROJECT_ROOT / cfg["data"]["graph_dir"]
    train_graphs = load_graph_split(graph_dir, "train")
    val_graphs = load_graph_split(graph_dir, "val")

    if not train_graphs:
        log.error("No training graphs found in %s/train/", graph_dir)
        log.error("Run graph precomputation first (Stage 5).")
        sys.exit(1)

    # Log graph statistics
    node_counts = [g.x.size(0) for g in train_graphs]
    log.info("Train graphs: %d | Nodes: min=%d, max=%d, mean=%.0f",
             len(train_graphs), min(node_counts), max(node_counts),
             np.mean(node_counts))

    train_loader = PyGDataLoader(train_graphs, batch_size=tcfg["batch_size"],
                                 shuffle=True, drop_last=False)
    val_loader = PyGDataLoader(val_graphs, batch_size=tcfg["batch_size"],
                               shuffle=False) if val_graphs else None

    # --- Model ---
    model = PhysicsAwareGNN(
        in_features=mcfg["in_features"],
        hidden_dim=mcfg["hidden_dim"],
        heads=mcfg["heads"],
        physics_lambda_init=mcfg["physics_lambda_init"],
        dropout_l1=mcfg["dropout_l1"],
        dropout_l2=mcfg["dropout_l2"],
        ffn_dropout=mcfg["ffn_dropout"],
    ).to(device)

    # Phase 7: EdgeAffinityMLP
    edge_scorer = EdgeAffinityMLP().to(device)
    
    # Combine parameters for optimizer
    params = list(model.parameters()) + list(edge_scorer.parameters())

    if tcfg["optimizer"] == "Adam":
        optimizer = Adam(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %d (%.2f K)", n_params, n_params / 1000)

    # --- Checkpointing ---
    ckpt_dir = PROJECT_ROOT / cfg["checkpoints"]["save_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results_dir = PROJECT_ROOT / cfg["logging"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    log_csv_path = PROJECT_ROOT / cfg["logging"]["log_csv"]
    log_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    best_val_mae = float("inf")
    patience_counter = 0
    warmup_epochs = 10  # Phase 7: warmup -> freeze
    history = []

    log.info("Starting training: %d epochs, batch_size=%d, lr=%.1e",
             tcfg["max_epochs"], tcfg["batch_size"], tcfg["lr"])

    for epoch in range(1, tcfg["max_epochs"] + 1):
        is_warmup = epoch <= warmup_epochs
        if epoch == warmup_epochs + 1:
            log.info("Freezing EdgeAffinityMLP and locking graph topology (Stage 2: Freeze).")
            edge_scorer.eval()

        t0 = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            grad_clip=tcfg["grad_clip"],
            edge_scorer=edge_scorer,
            warmup=is_warmup
        )

        # Validate
        val_metrics = {}
        if val_loader is not None:
            val_metrics = validate(model, val_loader, device)

        elapsed = time.time() - t0

        # Log
        log.info(
            "Epoch %3d/%3d | Train: Loss=%.4f MAE=%.4f | Val: Loss=%.4f MAE=%.4f AUC=%.4f | Smooth L1/L2: %.3f/%.3f",
            epoch, tcfg["max_epochs"],
            train_metrics["loss"], train_metrics["MAE"],
            val_metrics.get("loss", 0.0), val_metrics.get("MAE", 0.0), val_metrics.get("AUC", 0.0),
            val_metrics.get("cos_sim_layer1", 0.0), val_metrics.get("cos_sim_layer2", 0.0)
        )

        # Physics lambda monitoring
        for name, param in model.named_parameters():
            if "physics_lambda" in name:
                log.debug("  λ (%s) = %.4f", name, param.item())

        # Save history
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}
        if val_metrics:
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)

        # Early stopping
        if val_metrics:
            monitor_val = val_metrics.get(ecfg["monitor"].replace("val_", ""), 0)
            is_better = (monitor_val < best_val_metric if ecfg["mode"] == "min"
                         else monitor_val > best_val_metric)

            if is_better:
                best_val_metric = monitor_val
                patience_counter = 0
                # Save best checkpoint
                best_path = ckpt_dir / cfg["checkpoints"]["best_name"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_MAE": val_metrics.get("MAE", 0),
                    "val_AUC": val_metrics.get("AUC", 0),
                    "config": cfg,
                }, best_path)
                log.info("  ✓ Best model saved (val_MAE=%.4f)", monitor_val)
            else:
                patience_counter += 1
                if patience_counter >= ecfg["patience"]:
                    log.info("Early stopping at epoch %d (patience=%d)",
                             epoch, ecfg["patience"])
                    break

        # Periodic checkpoint
        periodic = cfg["checkpoints"].get("periodic_every", 10)
        if periodic and epoch % periodic == 0:
            periodic_path = ckpt_dir / f"gnn_epoch_{epoch:04d}.pt"
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict()}, periodic_path)

    # --- Save training log ---
    if history:
        fieldnames = list(history[0].keys())
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        log.info("Training log saved: %s", log_csv_path)

    log.info("=" * 60)
    log.info("Training complete. Best val_MAE: %.4f", best_val_metric)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
