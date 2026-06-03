"""
Check: what is the actual prediction distribution of cnn_best.pt?
If predictions never exceed 0.5, then pred_threshold=0.5 gives recall=0 always.
"""
import sys, torch
from pathlib import Path
PROJECT_ROOT = Path(r"d:\Physics Aware - Graphical Neural Network for Planetary Path Planning\pa-gnn")
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.fusion import build_fusion_model
from src.data.label_generation import build_dataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_fusion_model(
    cnn_checkpoint=str(PROJECT_ROOT / "checkpoints" / "cnn_best.pt"),
    freeze_cnn=True,
).to(device)
model.eval()

val_ds = build_dataset("val", PROJECT_ROOT / "data" / "splits", PROJECT_ROOT / "data" / "processed" / "tiles")
loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

all_preds, all_targets = [], []
with torch.no_grad():
    for i, batch in enumerate(loader):
        if i >= 30: break  # Sample 30 batches = 240 tiles
        img = batch["image"].to(device)
        result = model(img)
        all_preds.append(result["h_learned"].cpu().flatten())
        all_targets.append(batch["risk"].flatten())

preds   = torch.cat(all_preds)
targets = torch.cat(all_targets)
haz_mask = targets > 0.7

print(f"=== CNN (H_learned) Prediction Distribution (240 val tiles) ===")
print(f"  Overall: min={preds.min():.4f}, max={preds.max():.4f}, mean={preds.mean():.4f}, std={preds.std():.4f}")
print(f"  Hazard pixels (target>0.7): {haz_mask.float().mean()*100:.2f}% of total")
print(f"  Pred on hazardous pixels: mean={preds[haz_mask].mean():.4f}, max={preds[haz_mask].max():.4f}")
print(f"  Pred on safe pixels:      mean={preds[~haz_mask].mean():.4f}, max={preds[~haz_mask].max():.4f}")

print(f"\n=== Recall at various thresholds ===")
for thr in [0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.02]:
    tp = ((preds > thr) & haz_mask).float().sum()
    fn = ((preds <= thr) & haz_mask).float().sum()
    fp = ((preds > thr) & ~haz_mask).float().sum()
    recall    = tp / (tp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    f1 = 2*tp / (2*tp + fp + fn + 1e-6)
    print(f"  thr={thr:.2f}  recall={recall:.4f}  precision={precision:.4f}  F1={f1:.4f}")

