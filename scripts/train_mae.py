"""
train_mae.py
------------
Stage 0 — MAE Self-Supervised Pretraining Runner.

Blueprint §7 configuration:
  - Data:       All 17,298 CTX tiles (sliced_tiles_1/ + sliced_tiles_2/)
  - Optimizer:  AdamW
  - LR:         1.5×10⁻⁴ with cosine annealing
  - Weight dec: 0.05
  - Batch size: 64
  - Epochs:     200
  - Expected:   8–12 hours on RTX 3060 Ti

Verification:
  - Loss decreases monotonically over 200 epochs.
  - Qualitative check: at end of training, 5 reconstructed tiles are saved
    to results/mae_verification/ — these should show terrain structure,
    not noise.

Output:
  - checkpoints/mae_best.pt          — best (lowest loss) encoder checkpoint
  - checkpoints/mae_epoch_{N}.pt     — checkpoint every 10 epochs
  - results/mae_loss_curve.png       — training loss curve
  - results/mae_verification/        — 5 reconstructed tile visualisations
  - data/processed/mae_pretrain_log.csv — epoch-level loss log

Run from the pa-gnn/ directory:
    python scripts/train_mae.py [--epochs 200] [--batch_size 64] [--resume]
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
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ctx_loader import CTXDataset, build_ctx_dataloader
from src.models.decoder import unpatchify
from src.models.mae import MaskedAutoencoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_mae")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CTX_DIRS = [
    PROJECT_ROOT / "data" / "raw" / "ctx" / "sliced_tiles_1",
    PROJECT_ROOT / "data" / "raw" / "ctx" / "sliced_tiles_2",
]
CHECKPOINT_DIR  = PROJECT_ROOT / "checkpoints"
RESULTS_DIR     = PROJECT_ROOT / "results"
VERIFY_DIR      = RESULTS_DIR / "mae_verification"
LOSS_CURVE_PATH = RESULTS_DIR / "mae_loss_curve.png"
LOG_CSV         = PROJECT_ROOT / "data" / "processed" / "mae_pretrain_log.csv"


# ---------------------------------------------------------------------------
# Qualitative verification: save 5 reconstructed tiles
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_reconstruction_samples(
    model: MaskedAutoencoder,
    dataset: CTXDataset,
    device: torch.device,
    n: int = 5,
) -> None:
    """Save original + masked + reconstructed image triptychs for visual inspection."""
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Pick evenly spaced samples across dataset
    step = max(1, len(dataset) // n)
    indices = [min(i * step, len(dataset) - 1) for i in range(n)]

    from src.models.decoder import patchify, PATCH_DIM
    from src.models.encoder import PATCH_SIZE, IMAGE_SIZE

    for sample_idx, idx in enumerate(indices):
        img_tensor = dataset[idx].unsqueeze(0).to(device)  # (1, 3, 512, 512)

        _, pred, mask = model(img_tensor)

        # Reconstruct image from predictions
        # pred: (1, N, P²×C), mask: (1, N)
        target_patches = patchify(img_tensor, PATCH_SIZE)  # (1, N, 768)

        # Compose reconstruction: use prediction for masked, original for visible
        recon_patches = target_patches.clone()
        mask_bool = mask.unsqueeze(-1).bool()               # (1, N, 1)
        recon_patches = torch.where(mask_bool, pred, target_patches)

        recon_img = unpatchify(recon_patches, PATCH_SIZE, IMAGE_SIZE, channels=3)

        # Build masked image for visualisation (mask out 75% of patches)
        masked_patches = target_patches.clone()
        masked_patches[mask.bool()] = 0.0
        masked_img = unpatchify(masked_patches, PATCH_SIZE, IMAGE_SIZE, channels=3)

        # Save triptych
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, data, title in zip(
            axes,
            [img_tensor, masked_img, recon_img],
            ["Original", "Masked (75%)", "Reconstructed"],
        ):
            img_np = data[0, 0].cpu().numpy()  # take first channel (all 3 are identical)
            ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=12)
            ax.axis("off")

        plt.suptitle(f"MAE Verification — Tile #{idx}", fontsize=13)
        plt.tight_layout()
        out_path = VERIFY_DIR / f"verify_{sample_idx:02d}_tile{idx:05d}.png"
        plt.savefig(str(out_path), dpi=100, bbox_inches="tight")
        plt.close()
        log.info("Saved verification: %s", out_path.name)

    model.train()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_mae(
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1.5e-4,
    weight_decay: float = 0.05,
    resume: bool = False,
    device_str: str = "auto",
    num_workers: int = 4,
    checkpoint_every: int = 10,
    verify_every: int = 50,
) -> None:

    # --- Device setup ---
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    log.info("Device: %s", device)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)

    # --- Dataset ---
    log.info("Loading CTX dataset from %d directories...", len(CTX_DIRS))
    dataset = CTXDataset(ctx_dirs=CTX_DIRS)
    loader  = build_ctx_dataloader(dataset, batch_size=batch_size,
                                   num_workers=num_workers)
    log.info("Dataset: %d tiles, %d batches/epoch", len(dataset), len(loader))

    # --- Model ---
    model = MaskedAutoencoder(
        mask_ratio=0.75,
        norm_pix_loss=True,
        pretrained_imagenet=True,
    ).to(device)
    log.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # --- Optimizer & scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # --- Resume ---
    start_epoch = 1
    best_loss = float("inf")
    loss_history: list[dict] = []

    resume_ckpt = CHECKPOINT_DIR / "mae_latest.pt"
    if resume and resume_ckpt.exists():
        log.info("Resuming from %s", resume_ckpt)
        ckpt = torch.load(str(resume_ckpt), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        loss_history = ckpt.get("loss_history", [])
        log.info("Resumed from epoch %d, best_loss=%.6f", start_epoch - 1, best_loss)
    else:
        log.info("Starting fresh training (pretrained_imagenet=True)")

    # --- Training loop ---
    log.info("=" * 60)
    log.info("MAE Pretraining: %d epochs, batch=%d, lr=%.2e", epochs, batch_size, lr)
    log.info("=" * 60)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for batch_idx, images in enumerate(loader):
            images = images.to(device, non_blocking=True)

            loss, _, _ = model(images)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

            if batch_idx % 50 == 0:
                log.debug(
                    "Epoch %d/%d  batch %d/%d  loss=%.6f",
                    epoch, epochs, batch_idx, len(loader), loss.item(),
                )

        scheduler.step()

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_time = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        log.info(
            "Epoch %d/%d — loss=%.6f, lr=%.2e, time=%.1fs",
            epoch, epochs, mean_loss, current_lr, epoch_time,
        )

        loss_history.append({
            "epoch": epoch,
            "loss": mean_loss,
            "lr": current_lr,
            "time_s": epoch_time,
        })

        # --- Checkpoint: best ---
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_path = CHECKPOINT_DIR / "mae_best.pt"
            model.save_encoder_checkpoint(str(best_path))
            log.info("✓ New best encoder checkpoint saved (loss=%.6f)", best_loss)

        # --- Checkpoint: periodic ---
        if epoch % checkpoint_every == 0:
            periodic_path = CHECKPOINT_DIR / f"mae_epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "loss_history": loss_history,
                },
                str(periodic_path),
            )
            # Also save as latest for resuming
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "loss_history": loss_history,
                },
                str(CHECKPOINT_DIR / "mae_latest.pt"),
            )
            log.info("Periodic checkpoint saved: %s", periodic_path.name)

        # --- Qualitative verification ---
        if epoch % verify_every == 0 or epoch == epochs:
            log.info("Saving reconstruction samples for epoch %d...", epoch)
            save_reconstruction_samples(model, dataset, device, n=5)

    # --- Final: loss curve ---
    _save_loss_curve(loss_history)

    # --- Final: CSV log ---
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "lr", "time_s"])
        writer.writeheader()
        writer.writerows(loss_history)
    log.info("Loss log written: %s", LOG_CSV)

    # --- Final verification ---
    log.info("Saving final reconstruction samples...")
    save_reconstruction_samples(model, dataset, device, n=5)

    log.info("=" * 60)
    log.info("MAE Pretraining complete.")
    log.info("Best encoder saved to: %s", CHECKPOINT_DIR / "mae_best.pt")
    log.info("Use this checkpoint to initialise the CNN in Stage 3.")
    log.info("=" * 60)


def _save_loss_curve(loss_history: list[dict]) -> None:
    """Save the training loss curve plot."""
    epochs = [r["epoch"] for r in loss_history]
    losses = [r["loss"] for r in loss_history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, losses, linewidth=1.5, color="#2196F3")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MAE Reconstruction Loss (MSE on masked patches)", fontsize=12)
    ax.set_title("Stage 0 — MAE Pretraining Loss Curve", fontsize=13)
    ax.grid(True, alpha=0.3)

    # Annotate best epoch
    if losses:
        best_ep = epochs[losses.index(min(losses))]
        best_l  = min(losses)
        ax.axvline(best_ep, color="red", linestyle="--", alpha=0.5,
                   label=f"Best: epoch {best_ep} (loss={best_l:.5f})")
        ax.legend()

    plt.tight_layout()
    plt.savefig(str(LOSS_CURVE_PATH), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Loss curve saved: %s", LOSS_CURVE_PATH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 0 — MAE Pretraining")
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1.5e-4)
    parser.add_argument("--weight_decay",type=float, default=0.05)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--device",      type=str,   default="auto",
                        help="'auto', 'cuda', or 'cpu'")
    parser.add_argument("--checkpoint_every", type=int, default=10,
                        help="Save full checkpoint every N epochs")
    parser.add_argument("--verify_every", type=int, default=50,
                        help="Save reconstruction visualisations every N epochs")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoints/mae_latest.pt")
    args = parser.parse_args()

    train_mae(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        resume=args.resume,
        device_str=args.device,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        verify_every=args.verify_every,
    )
