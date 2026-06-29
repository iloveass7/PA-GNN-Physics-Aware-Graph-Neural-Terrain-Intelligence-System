"""sweep.py — Judge every CNN checkpoint by test_in AUROC. This is the real selection."""
import sys, glob
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.label_generation import build_dataset
from src.models.risk_model import build_risk_model

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
TILES_DIR  = PROJECT_ROOT / "data" / "processed" / "tiles"
HAZARD_THRESH = 0.7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(0)

# Judge on test_in — the reliable split, NOT val
ds = build_dataset("test_in", SPLITS_DIR, TILES_DIR)
loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
print(f"Judging on test_in: {len(ds)} tiles\n")

model = build_risk_model().to(DEVICE)

ckpts = sorted(glob.glob(str(PROJECT_ROOT/"checkpoints"/"cnn_epoch_*.pt")))
# also evaluate latest (final epoch) and the trainer's mIoU-selected best
for extra in ["cnn_latest.pt", "cnn_best.pt"]:
    p = str(PROJECT_ROOT/"checkpoints"/extra)
    if Path(p).exists():
        ckpts.append(p)
print(f"{'checkpoint':24s} {'AUROC':>7s} {'PR-AUC':>7s} {'haz_mean':>9s} {'safe_mean':>9s}")
results = []
for cp in ckpts:
    ck = torch.load(cp, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for b in loader:
            mm = b["valid"].bool() & torch.isfinite(b["risk"])
            pr = model(b["image"].to(DEVICE))[:, 0].cpu()
            preds.append(pr[mm].numpy()); tgts.append(b["risk"][mm].numpy())
    p = np.concatenate(preds); y = (np.concatenate(tgts) > HAZARD_THRESH)
    if len(y) > 5_000_000:
        ix = rng.choice(len(y), 5_000_000, replace=False); p, y = p[ix], y[ix]
    auc = roc_auc_score(y, p); prc = average_precision_score(y, p)
    hm, sm = p[y].mean(), p[~y].mean()
    print(f"{Path(cp).name:24s} {auc:7.4f} {prc:7.4f} {hm:9.3f} {sm:9.3f}")
    results.append((Path(cp).name, auc, hm, sm))

if results:
    best = max(results, key=lambda r: r[1])
    print(f"\nBEST by test_in AUROC: {best[0]}  AUROC={best[1]:.4f}  "
          f"(haz_mean={best[2]:.3f} {'>' if best[2]>best[3] else '<'} safe_mean={best[3]:.3f})")
    print(f"\nTo use it:  copy checkpoints/{best[0]} -> checkpoints/cnn_best.pt")
else:
    print("\nNo checkpoints found to sweep!")
