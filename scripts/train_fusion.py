"""
train_fusion.py
---------------
Stage 4 — Spatial Adaptive Fusion training runner.

Blueprint §11 two-phase training:
  Phase 1 : Train CNN (Stage 3) to convergence.  (Done in train_cnn.py)
  Phase 2 : Freeze all CNN weights.  Load Phase 1 checkpoint.
            Train ONLY the fusion network's ~12K parameters.
            Loss applied to H_final against DEM-derived labels using
            the same compound loss as Stage 3.

Run from the project root directory:
    python scripts/train_fusion.py [--cnn_ckpt PATH] [--resume] [--device auto]

Requires:
  - Trained CNN checkpoint from Stage 3 (checkpoints/cnn_best.pt)
  - Processed tile data in data/processed/tiles/
  - Split files in data/splits/

Outputs:
  checkpoints/fusion_best.pt               — best model (by val_hazard_recall)
  checkpoints/fusion_epoch_NNNN.pt         — periodic full checkpoints
  results/stage4/fusion_loss_curve.png     — training curves
  results/stage4/predictions/              — H_final + alpha visualisations
  data/processed/fusion_train_log.csv      — epoch-level metric log
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.label_generation import build_dataset
from src.models.fusion import AdaptiveFusion, EndToEndFusionModel, build_fusion_model
from src.training.losses import RiskLoss, compute_metrics
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_fusion")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CONFIG_PATH  = PROJECT_ROOT / "configs" / "fusion.yaml"
SPLITS_DIR   = PROJECT_ROOT / "data" / "splits"
TILES_DIR    = PROJECT_ROOT / "data" / "processed" / "tiles"
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
RESULTS_DIR  = PROJECT_ROOT / "results" / "stage4"
LOG_CSV      = PROJECT_ROOT / "data" / "processed" / "fusion_train_log.csv"
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
# Fusion-specific training/validation loops
# ---------------------------------------------------------------------------

def train_one_epoch_fusion(
    model: EndToEndFusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: RiskLoss,
    device: torch.device,
    grad_clip: float = 1.0,
    scaler: "torch.cuda.amp.GradScaler | None" = None,
) -> dict[str, float]:
    """Run one training epoch for Stage 4 fusion.

    Only the fusion network's ~12K parameters receive gradient updates.
    The CNN and physics engine remain frozen.

    Parameters
    ----------
    model     : EndToEndFusionModel with freeze_cnn=True
    loader    : DataLoader yielding dicts with 'image', 'risk', 'valid'
    optimizer : AdamW (on fusion params only)
    loss_fn   : RiskLoss (same compound loss as Stage 3)
    device    : torch.device
    grad_clip : max gradient norm
    scaler    : Optional AMP GradScaler

    Returns
    -------
    dict with: loss, bce, dice, tv  (mean over epoch)
    """
    model.train()  # Sets fusion to train, keeps CNN in eval (via override)
    accum = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "tv": 0.0}
    n_batches = 0

    for batch in loader:
        images   = batch["image"].to(device, non_blocking=True)    # (B, 3, 512, 512)
        targets  = batch["risk"].to(device, non_blocking=True)     # (B, 512, 512)
        validity = batch["valid"].to(device, non_blocking=True)    # (B, 512, 512)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                result = model(images)
                h_final = result["h_final"]                        # (B, 1, H, W)
                loss, comps = loss_fn(h_final, targets, validity)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.fusion.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            result = model(images)
            h_final = result["h_final"]
            loss, comps = loss_fn(h_final, targets, validity)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.fusion.parameters(), grad_clip)
            optimizer.step()

        accum["loss"] += comps["total"]
        accum["bce"]  += comps["bce"]
        accum["dice"] += comps["dice"]
        accum["tv"]   += comps["tv"]
        n_batches += 1

    n_batches = max(n_batches, 1)
    return {k: v / n_batches for k, v in accum.items()}


@torch.no_grad()
def validate_one_epoch_fusion(
    model: EndToEndFusionModel,
    loader: DataLoader,
    loss_fn: RiskLoss,
    device: torch.device,
) -> dict[str, float]:
    """Run one validation epoch for Stage 4 fusion.

    Computes loss and metrics on H_final AND separately on H_learned and
    H_physics for comparison.

    Returns
    -------
    dict with keys:
        loss, bce, dice, tv                          — H_final loss components
        hazard_recall, hazard_precision, hazard_f1,
        safe_recall, mIoU                            — H_final metrics
        h_learned_hazard_recall                      — CNN-only recall (for comparison)
        h_physics_hazard_recall                      — physics-only recall (for comparison)
        alpha_mean, alpha_std                        — α-map statistics (diagnostic)
    """
    model.eval()

    loss_accum = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "tv": 0.0}
    metric_accum = {
        "hazard_recall": 0.0,
        "hazard_precision": 0.0,
        "hazard_f1": 0.0,
        "safe_recall": 0.0,
        "mIoU": 0.0,
    }
    comparison_accum = {
        "h_learned_hazard_recall": 0.0,
        "h_physics_hazard_recall": 0.0,
        "alpha_mean": 0.0,
        "alpha_std": 0.0,
    }
    n_batches = 0

    for batch in loader:
        images   = batch["image"].to(device, non_blocking=True)
        targets  = batch["risk"].to(device, non_blocking=True)
        validity = batch["valid"].to(device, non_blocking=True)

        result = model(images)
        h_final   = result["h_final"]
        h_learned = result["h_learned"]
        h_physics = result["h_physics"]
        alpha     = result["alpha"]

        # Loss on H_final
        loss, comps = loss_fn(h_final, targets, validity)
        metrics = compute_metrics(h_final, targets, validity)

        # Comparison metrics for H_learned and H_physics
        learned_metrics = compute_metrics(h_learned, targets, validity)
        physics_metrics = compute_metrics(h_physics, targets, validity)

        # Accumulate
        loss_accum["loss"] += comps["total"]
        loss_accum["bce"]  += comps["bce"]
        loss_accum["dice"] += comps["dice"]
        loss_accum["tv"]   += comps["tv"]

        for k, v in metrics.items():
            metric_accum[k] = metric_accum.get(k, 0.0) + v

        comparison_accum["h_learned_hazard_recall"] += learned_metrics["hazard_recall"]
        comparison_accum["h_physics_hazard_recall"] += physics_metrics["hazard_recall"]
        comparison_accum["alpha_mean"] += alpha.mean().item()
        comparison_accum["alpha_std"]  += alpha.std().item()

        n_batches += 1

    n_batches = max(n_batches, 1)
    result_dict = {k: v / n_batches for k, v in loss_accum.items()}
    result_dict.update({k: v / n_batches for k, v in metric_accum.items()})
    result_dict.update({k: v / n_batches for k, v in comparison_accum.items()})
    return result_dict


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_fusion_samples(
    model: EndToEndFusionModel,
    val_dataset,
    device: torch.device,
    epoch: int,
    n: int = 5,
) -> None:
    """Save 6-column visualisation: Image | Target | H_physics | H_learned | α | H_final.

    Blueprint §21 Figure 3 format.
    """
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    model.eval()

    import random
    indices = random.sample(range(len(val_dataset)), min(n, len(val_dataset)))

    fig, axes = plt.subplots(len(indices), 6, figsize=(24, 4 * len(indices)))
    if len(indices) == 1:
        axes = [axes]

    col_titles = ["Image", "Target (DEM)", "H_physics", "H_learned", "α map", "H_final"]
    cmaps      = ["gray",  "RdYlGn_r",    "RdYlGn_r",  "RdYlGn_r",  "coolwarm", "RdYlGn_r"]

    for row_idx, sample_idx in enumerate(indices):
        sample = val_dataset[sample_idx]
        image_3ch = sample["image"].unsqueeze(0).to(device)  # (1, 3, 512, 512)
        target    = sample["risk"].numpy()                   # (512, 512)

        result = model(image_3ch)
        h_physics = result["h_physics"][0, 0].cpu().numpy()
        h_learned = result["h_learned"][0, 0].cpu().numpy()
        alpha_map = result["alpha"][0, 0].cpu().numpy()
        h_final   = result["h_final"][0, 0].cpu().numpy()
        img_np    = image_3ch[0, 0].cpu().numpy()

        panels = [img_np, target, h_physics, h_learned, alpha_map, h_final]

        for col_idx, (data, title, cmap) in enumerate(zip(panels, col_titles, cmaps)):
            ax = axes[row_idx][col_idx]
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row_idx == 0:
                ax.set_title(title, fontsize=10)
            ax.axis("off")

    plt.suptitle(f"Stage 4 Fusion — Epoch {epoch}", fontsize=13)
    plt.tight_layout()
    out = PRED_DIR / f"fusion_epoch_{epoch:04d}.png"
    plt.savefig(str(out), dpi=100, bbox_inches="tight")
    plt.close()
    log.info("Fusion visualisations saved: %s", out.name)


def save_loss_curves(history: list[dict]) -> None:
    """Save training curves: loss, H_final recall vs H_learned recall, α stats."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in history]

    fig, axes = plt.subplots(1, 4, figsize=(22, 4))

    # (1) Loss
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train", color="#2196F3")
    axes[0].plot(epochs, [r["val_loss"]   for r in history], label="val",   color="#FF5722")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (2) Hazard Recall comparison: H_final vs H_learned vs H_physics
    axes[1].plot(epochs, [r.get("val_hazard_recall", 0) for r in history],
                 label="H_final", color="#4CAF50", linewidth=2)
    axes[1].plot(epochs, [r.get("val_h_learned_hazard_recall", 0) for r in history],
                 label="H_learned (CNN)", color="#FF9800", linestyle="--")
    axes[1].plot(epochs, [r.get("val_h_physics_hazard_recall", 0) for r in history],
                 label="H_physics", color="#9C27B0", linestyle="--")
    axes[1].set_title("Val Hazard Recall (H_final must exceed H_learned)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # (3) mIoU
    axes[2].plot(epochs, [r.get("val_mIoU", 0) for r in history], color="#9C27B0")
    axes[2].set_title("Val mIoU")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    # (4) Alpha statistics (diagnostic)
    axes[3].plot(epochs, [r.get("val_alpha_mean", 0.5) for r in history],
                 label="α mean", color="#2196F3")
    axes[3].fill_between(
        epochs,
        [r.get("val_alpha_mean", 0.5) - r.get("val_alpha_std", 0) for r in history],
        [r.get("val_alpha_mean", 0.5) + r.get("val_alpha_std", 0) for r in history],
        alpha=0.2, color="#2196F3",
    )
    axes[3].set_title("α map statistics (uniform = degenerate)")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylim(0, 1)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.suptitle("Stage 4 — Fusion Training Curves", fontsize=13)
    plt.tight_layout()
    out = RESULTS_DIR / "fusion_loss_curve.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Loss curves saved: %s", out)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_fusion(
    cnn_ckpt: str | None = None,
    resume: bool = False,
    device_str: str = "auto",
) -> None:

    cfg = load_config(CONFIG_PATH)
    train_cfg = cfg.get("training", {})
    loss_cfg  = cfg.get("loss", {})
    dl_cfg    = cfg.get("dataloader", {})
    ckpt_cfg  = cfg.get("checkpoints", {})
    es_cfg    = cfg.get("early_stopping", {})
    phys_cfg  = cfg.get("physics", {})
    diag_cfg  = cfg.get("diagnostic", {})

    # Blueprint §11 enforcement
    joint_with_cnn = cfg.get("joint_with_cnn", False)
    if joint_with_cnn:
        log.error(
            "joint_with_cnn=true is FORBIDDEN by blueprint §11. "
            "This produces degenerate α maps. Overriding to false."
        )
        joint_with_cnn = False

    freeze_cnn = not joint_with_cnn  # Must be True

    # Hyperparameters (with blueprint defaults)
    LR            = float(train_cfg.get("lr", 1e-4))
    WEIGHT_DECAY  = float(train_cfg.get("weight_decay", 1e-4))
    BATCH_SIZE    = int(train_cfg.get("batch_size", 8))
    MAX_EPOCHS    = int(train_cfg.get("max_epochs", 40))
    GRAD_CLIP     = float(train_cfg.get("grad_clip", 1.0))
    USE_AMP       = bool(train_cfg.get("use_amp", True))
    PATIENCE      = int(es_cfg.get("patience", 10))
    NUM_WORKERS   = int(dl_cfg.get("num_workers", 4))
    PERIODIC_N    = int(ckpt_cfg.get("periodic_every", 5))
    ALPHA_STD_THR = float(diag_cfg.get("alpha_std_threshold", 0.02))
    ALPHA_LOG_N   = int(diag_cfg.get("log_alpha_stats_every", 5))

    # CNN checkpoint resolution
    if cnn_ckpt is None:
        cnn_ckpt = str(PROJECT_ROOT / cfg.get("cnn_checkpoint", "checkpoints/cnn_best.pt"))

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
    train_ds = build_dataset("train", SPLITS_DIR, TILES_DIR)
    val_ds   = build_dataset("val",   SPLITS_DIR, TILES_DIR)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=True, persistent_workers=NUM_WORKERS > 0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=NUM_WORKERS > 0)

    log.info("Train: %d tiles  |  Val: %d tiles", len(train_ds), len(val_ds))

    # --- Model ---
    model = build_fusion_model(
        cnn_checkpoint=cnn_ckpt,
        freeze_cnn=freeze_cnn,
        physics_w1=float(phys_cfg.get("w1", 0.4)),
        physics_w2=float(phys_cfg.get("w2", 0.3)),
        physics_w3=float(phys_cfg.get("w3", 0.3)),
    )
    model = model.to(device)

    # --- Loss (same compound loss as Stage 3) ---
    loss_fn = RiskLoss(
        hazard_threshold = float(loss_cfg.get("hazard_threshold", 0.7)),
        hazard_weight    = float(loss_cfg.get("hazard_weight",    3.0)),
        dice_coeff       = float(loss_cfg.get("dice_coeff",       0.5)),
        tv_coeff         = float(loss_cfg.get("tv_coeff",         0.1)),
    )

    # --- Optimizer (fusion params ONLY) ---
    # Blueprint §11: only fusion network trains, CNN is frozen.
    trainable_params = model.get_trainable_params()
    n_trainable = sum(p.numel() for p in trainable_params)
    log.info("Optimizer: AdamW on %d fusion parameters (%.1fK)",
             n_trainable, n_trainable / 1e3)

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=LR * 0.01,
    )

    # --- AMP scaler ---
    scaler = torch.cuda.amp.GradScaler() if USE_AMP and device.type == "cuda" else None

    # --- Resume ---
    start_epoch  = 1
    best_recall  = -1.0
    no_improve   = 0
    history: list[dict] = []

    latest_ckpt = CKPT_DIR / "fusion_latest.pt"
    if resume and latest_ckpt.exists():
        log.info("Resuming from %s", latest_ckpt)
        ckpt = torch.load(str(latest_ckpt), map_location=device, weights_only=False)
        model.fusion.load_state_dict(ckpt["fusion_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_recall = ckpt.get("best_recall", -1.0)
        no_improve  = ckpt.get("no_improve", 0)
        history     = ckpt.get("history", [])
        log.info("Resumed from epoch %d, best_recall=%.4f", start_epoch - 1, best_recall)

    # --- Training loop ---
    log.info("=" * 60)
    log.info("Stage 4 Fusion Training: epochs=%d, batch=%d, lr=%.1e, "
             "freeze_cnn=%s", MAX_EPOCHS, BATCH_SIZE, LR, freeze_cnn)
    log.info("=" * 60)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        t0 = time.time()

        train_metrics = train_one_epoch_fusion(
            model, train_loader, optimizer, loss_fn, device, GRAD_CLIP, scaler,
        )
        val_metrics = validate_one_epoch_fusion(
            model, val_loader, loss_fn, device,
        )
        scheduler.step()

        elapsed = time.time() - t0
        val_recall     = val_metrics["hazard_recall"]
        val_miou       = val_metrics["mIoU"]
        learned_recall = val_metrics["h_learned_hazard_recall"]
        physics_recall = val_metrics["h_physics_hazard_recall"]
        alpha_mean     = val_metrics["alpha_mean"]
        alpha_std      = val_metrics["alpha_std"]

        log.info(
            "Epoch %02d/%d | train_loss=%.4f | val_loss=%.4f | "
            "H_final_recall=%.4f | H_learned_recall=%.4f | H_physics_recall=%.4f | "
            "α_mean=%.3f ± %.3f | %.1fs",
            epoch, MAX_EPOCHS,
            train_metrics["loss"], val_metrics["loss"],
            val_recall, learned_recall, physics_recall,
            alpha_mean, alpha_std, elapsed,
        )

        # --- Blueprint §11 diagnostic check ---
        if epoch % ALPHA_LOG_N == 0 or epoch == 1:
            if alpha_std < ALPHA_STD_THR:
                log.warning(
                    "⚠️  DIAGNOSTIC: α map has very low spatial variance (std=%.4f < %.4f). "
                    "Fusion may be degenerate — α is near-uniform. "
                    "Check that CNN is frozen and producing diverse H_learned. "
                    "Expected: low-α on slope terrain, high-α on crater rims.",
                    alpha_std, ALPHA_STD_THR,
                )

        # Blueprint §11 verification: H_final recall must exceed H_learned recall
        if epoch >= 5 and val_recall < learned_recall:
            log.warning(
                "⚠️  H_final recall (%.4f) is LOWER than H_learned recall (%.4f). "
                "Blueprint §11 requires H_final ≥ H_learned. Debug before Stage 5.",
                val_recall, learned_recall,
            )

        # Record
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "loss"},
        }
        history.append(row)

        # --- Best checkpoint ---
        if val_recall > best_recall:
            best_recall = val_recall
            no_improve  = 0
            best_path   = CKPT_DIR / "fusion_best.pt"
            torch.save({
                "fusion_model": model.fusion.state_dict(),
                "epoch": epoch,
                "val_hazard_recall": best_recall,
                "val_mIoU": val_miou,
                "alpha_mean": alpha_mean,
                "alpha_std": alpha_std,
                "h_learned_recall": learned_recall,
                "h_physics_recall": physics_recall,
            }, str(best_path))
            log.info("  ✓ New best checkpoint (recall=%.4f)", best_recall)
        else:
            no_improve += 1

        # --- Periodic checkpoint ---
        if epoch % PERIODIC_N == 0 or epoch == MAX_EPOCHS:
            periodic_path = CKPT_DIR / f"fusion_epoch_{epoch:04d}.pt"
            save_dict = {
                "fusion_model": model.fusion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_recall": best_recall,
                "no_improve": no_improve,
                "history": history,
            }
            torch.save(save_dict, str(periodic_path))
            torch.save(save_dict, str(latest_ckpt))

        # --- Visualisation every 10 epochs ---
        if epoch % 10 == 0 or epoch == MAX_EPOCHS or epoch == 1:
            save_fusion_samples(model, val_ds, device, epoch, n=5)

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
    log.info("Stage 4 training complete.")
    log.info("  Best val_hazard_recall : %.4f", best_recall)
    log.info("  Best checkpoint        : %s", CKPT_DIR / "fusion_best.pt")
    log.info("  Next: Stage 5 (graph precomputation) — uses H_final from this model.")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 4 — Spatial Adaptive Fusion Training (CNN frozen)"
    )
    parser.add_argument(
        "--cnn_ckpt", default=None,
        help="Path to trained CNN checkpoint (default: checkpoints/cnn_best.pt)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoints/fusion_latest.pt",
    )
    parser.add_argument(
        "--device", default="auto",
        help="'auto', 'cuda', or 'cpu'",
    )
    args = parser.parse_args()

    train_fusion(cnn_ckpt=args.cnn_ckpt, resume=args.resume, device_str=args.device)
