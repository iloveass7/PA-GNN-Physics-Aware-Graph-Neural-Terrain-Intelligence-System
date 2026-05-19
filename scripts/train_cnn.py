"""
train_cnn.py
------------
Stage 3 — CNN Semantic Risk Estimator training runner.

Blueprint §10 configuration:
  Model     : MobileNetV3-Large + DeepLabV3+ (initialised from MAE Stage 0)
  Loss      : Weighted BCE + 0.5×Dice + 0.1×TV
  Optimizer : AdamW, lr=1e-4, weight_decay=1e-4, cosine annealing
  Batch     : 8
  Epochs    : max 60, early stopping patience=10 on val_hazard_recall
  Checkpoint: best val_hazard_recall

Run from the pa-gnn/ directory:
    python scripts/train_cnn.py [--resume] [--init {mae,imagenet,random}]

Outputs:
  checkpoints/cnn_best.pt              — best model (by val_hazard_recall)
  checkpoints/cnn_epoch_NNNN.pt        — periodic full checkpoints
  results/stage3/cnn_loss_curve.png    — training curves
  results/stage3/predictions/          — sample prediction visualisations
  data/processed/cnn_train_log.csv     — epoch-level metric log
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.label_generation import build_dataset
from src.models.risk_model import build_risk_model
from src.training.losses import RiskLoss
from src.training.trainer import train_one_epoch, validate_one_epoch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_cnn")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CONFIG_PATH  = PROJECT_ROOT / "configs" / "cnn.yaml"
SPLITS_DIR   = PROJECT_ROOT / "data" / "splits"
TILES_DIR    = PROJECT_ROOT / "data" / "processed" / "tiles"
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
RESULTS_DIR  = PROJECT_ROOT / "results" / "stage3"
LOG_CSV      = PROJECT_ROOT / "data" / "processed" / "cnn_train_log.csv"
PRED_DIR     = RESULTS_DIR / "predictions"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        log.warning("Config not found at %s, using defaults", config_path)
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_prediction_samples(
    model: torch.nn.Module,
    val_dataset,
    device: torch.device,
    epoch: int,
    n: int = 5,
) -> None:
    """Save image | target | prediction triptychs for visual inspection."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    model.eval()

    import random
    import numpy as np

    indices = random.sample(range(len(val_dataset)), min(n, len(val_dataset)))

    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))
    if len(indices) == 1:
        axes = [axes]

    for row, idx in enumerate(indices):
        sample = val_dataset[idx]
        image  = sample["image"].unsqueeze(0).to(device)    # (1, 1, 512, 512)
        target = sample["risk"].numpy()                     # (512, 512)

        pred = model(image)[0, 0].cpu().numpy()             # (512, 512)
        img_np = image[0, 0].cpu().numpy()                  # grayscale

        for col, (data, title, cmap) in enumerate([
            (img_np,  "Image",      "gray"),
            (target,  "Target (DEM risk)", "RdYlGn_r"),
            (pred,    "H_learned",  "RdYlGn_r"),
        ]):
            ax = axes[row][col]
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title if row == 0 else "", fontsize=10)
            ax.axis("off")

    plt.suptitle(f"Stage 3 Predictions — Epoch {epoch}", fontsize=12)
    plt.tight_layout()
    out = PRED_DIR / f"preds_epoch_{epoch:04d}.png"
    plt.savefig(str(out), dpi=100, bbox_inches="tight")
    plt.close()
    log.info("Predictions saved: %s", out.name)

    model.train()


def save_loss_curves(history: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Loss
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train", color="#2196F3")
    axes[0].plot(epochs, [r["val_loss"]   for r in history], label="val",   color="#FF5722")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Hazard Recall
    axes[1].plot(epochs, [r.get("val_hazard_recall", 0) for r in history], color="#4CAF50")
    axes[1].set_title("Val Hazard Recall (early stopping monitor)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # mIoU
    axes[2].plot(epochs, [r.get("val_mIoU", 0) for r in history], color="#9C27B0")
    axes[2].set_title("Val mIoU")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Stage 3 — CNN Training Curves", fontsize=13)
    plt.tight_layout()
    out = RESULTS_DIR / "cnn_loss_curve.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Loss curves saved: %s", out)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_cnn(
    init_mode: str = "mae",
    resume: bool = False,
    device_str: str = "auto",
) -> None:

    cfg = load_config(CONFIG_PATH)
    train_cfg = cfg.get("training", {})
    loss_cfg  = cfg.get("loss", {})
    dl_cfg    = cfg.get("dataloader", {})
    ckpt_cfg  = cfg.get("checkpoints", {})
    es_cfg    = cfg.get("early_stopping", {})

    # Hyperparameters (with blueprint defaults)
    LR            = float(train_cfg.get("lr", 1e-4))
    WEIGHT_DECAY  = float(train_cfg.get("weight_decay", 1e-4))
    BATCH_SIZE    = int(train_cfg.get("batch_size", 8))
    MAX_EPOCHS    = int(train_cfg.get("max_epochs", 60))
    GRAD_CLIP     = float(train_cfg.get("grad_clip", 1.0))
    USE_AMP       = bool(train_cfg.get("use_amp", True))
    PATIENCE      = int(es_cfg.get("patience", 10))
    NUM_WORKERS   = int(dl_cfg.get("num_workers", 4))
    PERIODIC_N    = int(ckpt_cfg.get("periodic_every", 5))

    # --- Device ---
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    log.info("Device: %s", device)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    log.info("Loading datasets from %s...", TILES_DIR)
    train_ds = build_dataset("train",    SPLITS_DIR, TILES_DIR)
    val_ds   = build_dataset("val",      SPLITS_DIR, TILES_DIR)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=True, persistent_workers=NUM_WORKERS > 0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=NUM_WORKERS > 0)

    log.info("Train: %d tiles  |  Val: %d tiles", len(train_ds), len(val_ds))

    # --- Model ---
    mae_ckpt = PROJECT_ROOT / cfg.get("model", {}).get("mae_checkpoint", "checkpoints/mae_best.pt")
    if init_mode == "mae":
        model = build_risk_model(mae_checkpoint=mae_ckpt)
    elif init_mode == "imagenet":
        model = build_risk_model(pretrained_imagenet=True)
        log.info("Ablation: ImageNet pretrained init")
    else:  # random
        model = build_risk_model()
        log.info("Ablation: Random init")
    model = model.to(device)

    # --- Loss ---
    loss_fn = RiskLoss(
        hazard_threshold = float(loss_cfg.get("hazard_threshold", 0.7)),
        hazard_weight    = float(loss_cfg.get("hazard_weight",    5.0)),  # default synced with losses.py
        dice_coeff       = float(loss_cfg.get("dice_coeff",       0.5)),
        tv_coeff         = float(loss_cfg.get("tv_coeff",         0.1)),
    )

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW([
    {"params": model.features.parameters(), "lr": 1e-5},
    {"params": model.decoder.parameters(),  "lr": 1e-3},
], weight_decay=WEIGHT_DECAY)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS - 5, eta_min=1e-6
    )
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine_scheduler], milestones=[5]
    )

    # --- AMP scaler ---
    scaler = torch.amp.GradScaler('cuda') if USE_AMP and device.type == "cuda" else None

    # --- Resume ---
    start_epoch  = 1
    best_recall  = -1.0
    no_improve   = 0
    history: list[dict] = []

    latest_ckpt = CKPT_DIR / "cnn_latest.pt"
    if resume and latest_ckpt.exists():
        log.info("Resuming from %s", latest_ckpt)
        ckpt = torch.load(str(latest_ckpt), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_recall = ckpt.get("best_recall", -1.0)
        no_improve  = ckpt.get("no_improve", 0)
        history     = ckpt.get("history", [])
        log.info("Resumed from epoch %d, best_recall=%.4f", start_epoch - 1, best_recall)

    # --- Training loop ---
    log.info("=" * 60)
    log.info("Stage 3 CNN Training: init=%s, epochs=%d, batch=%d, lr=%.1e",
             init_mode, MAX_EPOCHS, BATCH_SIZE, LR)
    log.info("=" * 60)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn,
                                        device, GRAD_CLIP, scaler)
        val_metrics   = validate_one_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        elapsed = time.time() - t0
        val_recall = val_metrics["hazard_recall"]
        val_miou   = val_metrics["mIoU"]

        log.info(
            "Epoch %02d/%d | train_loss=%.4f | val_loss=%.4f | "
            "val_recall=%.4f | val_mIoU=%.4f | %.1fs",
            epoch, MAX_EPOCHS,
            train_metrics["loss"], val_metrics["loss"],
            val_recall, val_miou, elapsed,
        )

        # Record
        row = {"epoch": epoch, "init": init_mode,
               "train_loss": train_metrics["loss"],
               "val_loss": val_metrics["loss"],
               **{f"val_{k}": v for k, v in val_metrics.items() if k != "loss"}}
        history.append(row)

        # --- Best checkpoint ---
        if val_recall > best_recall:
            best_recall = val_recall
            no_improve  = 0
            best_path   = CKPT_DIR / "cnn_best.pt"
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_hazard_recall": best_recall,
                "val_mIoU": val_miou,
                "init_mode": init_mode,
            }, str(best_path))
            log.info("  ✓ New best checkpoint (recall=%.4f)", best_recall)
        else:
            no_improve += 1

        # --- Periodic checkpoint ---
        if epoch % PERIODIC_N == 0 or epoch == MAX_EPOCHS:
            periodic_path = CKPT_DIR / f"cnn_epoch_{epoch:04d}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_recall": best_recall,
                "no_improve": no_improve,
                "history": history,
            }, str(periodic_path))
            # Also save as latest
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_recall": best_recall,
                "no_improve": no_improve,
                "history": history,
            }, str(latest_ckpt))

        # --- Prediction visualisation every 10 epochs ---
        if epoch % 10 == 0 or epoch == MAX_EPOCHS:
            save_prediction_samples(model, val_ds, device, epoch, n=5)

        # --- Early stopping ---
        if no_improve >= PATIENCE:
            log.info("Early stopping: no improvement for %d epochs.", PATIENCE)
            break

    # --- Final outputs ---
    save_loss_curves(history)

    with open(LOG_CSV, "w", newline="") as f:
        if history:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
    log.info("Training log: %s", LOG_CSV)

    log.info("=" * 60)
    log.info("Stage 3 training complete.")
    log.info("  Best val_hazard_recall : %.4f", best_recall)
    log.info("  Best checkpoint        : %s", CKPT_DIR / "cnn_best.pt")
    log.info("  Used by Stage 4 (fusion) and Stage 5 (graph).")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3 — CNN Risk Estimator Training")
    parser.add_argument(
        "--init", default="mae",
        choices=["mae", "imagenet", "random"],
        help="Encoder initialisation: mae (blueprint), imagenet (ablation), random (ablation)"
    )
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoints/cnn_latest.pt")
    parser.add_argument("--device", default="auto",
                        help="'auto', 'cuda', or 'cpu'")
    args = parser.parse_args()

    train_cnn(init_mode=args.init, resume=args.resume, device_str=args.device)
