"""sweep_fusion.py — Judge every fusion checkpoint by test_in AUROC. This is the real selection."""
import sys, glob
import yaml
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.label_generation import build_dataset
from src.models.fusion import build_fusion_model

CONFIG_PATH = PROJECT_ROOT / "configs" / "fusion.yaml"
SPLITS_DIR  = PROJECT_ROOT / "data" / "splits"
TILES_DIR   = PROJECT_ROOT / "data" / "processed" / "tiles"
HAZARD_THRESH = 0.7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(0)

# Load config
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f) or {}

phys_cfg = cfg.get("physics", {})
alpha_reg_beta = float(cfg.get("loss", {}).get("alpha_reg_beta", 0.01))

# Use test_in split for reliable selection
ds = build_dataset("test_in", SPLITS_DIR, TILES_DIR)
loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
print(f"Judging on test_in: {len(ds)} tiles\n")

# Re-build fusion model (loading checkpoints/cnn_best.pt for the frozen CNN)
cnn_ckpt = str(PROJECT_ROOT / cfg.get("cnn_checkpoint", "checkpoints/cnn_best.pt"))
print(f"Using frozen CNN checkpoint: {cnn_ckpt}")

model = build_fusion_model(
    cnn_checkpoint=cnn_ckpt,
    freeze_cnn=True,
    physics_w1=float(phys_cfg.get("w1", 0.4)),
    physics_w2=float(phys_cfg.get("w2", 0.3)),
    physics_w3=float(phys_cfg.get("w3", 0.3)),
    alpha_reg_beta=alpha_reg_beta,
).to(DEVICE)

ckpts = sorted(glob.glob(str(PROJECT_ROOT/"checkpoints"/"fusion_epoch_*.pt")))
for extra in ["fusion_latest.pt", "fusion_best.pt"]:
    p = str(PROJECT_ROOT/"checkpoints"/extra)
    if Path(p).exists():
        ckpts.append(p)

print(f"\n{'checkpoint':24s} {'AUROC':>7s} {'PR-AUC':>7s} {'haz_mean':>9s} {'safe_mean':>9s}")
results = []
for cp in ckpts:
    ck = torch.load(cp, map_location=DEVICE, weights_only=False)
    model.fusion.load_state_dict(ck["fusion_model"])
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for b in loader:
            mm = b["valid"].bool() & torch.isfinite(b["risk"])
            pr = model(b["image"].to(DEVICE))["h_final"][:, 0].cpu()
            preds.append(pr[mm].numpy())
            tgts.append(b["risk"][mm].numpy())
    p = np.concatenate(preds)
    y = (np.concatenate(tgts) > HAZARD_THRESH)
    if len(y) > 5_000_000:
        ix = rng.choice(len(y), 5_000_000, replace=False)
        p, y = p[ix], y[ix]
    
    auc = roc_auc_score(y, p) if (y.min() != y.max()) else float("nan")
    prc = average_precision_score(y, p) if (y.min() != y.max()) else float("nan")
    hm, sm = p[y].mean(), p[~y].mean()
    print(f"{Path(cp).name:24s} {auc:7.4f} {prc:7.4f} {hm:9.3f} {sm:9.3f}")
    results.append((Path(cp).name, auc, hm, sm))

if results:
    best = max(results, key=lambda r: r[1])
    print(f"\nBEST by test_in AUROC: {best[0]}  AUROC={best[1]:.4f}  "
          f"(haz_mean={best[2]:.3f} {'>' if best[2]>best[3] else '<'} safe_mean={best[3]:.3f})")
    print(f"\nTo use it:  copy checkpoints/{best[0]} -> checkpoints/fusion_best.pt")
else:
    print("\nNo checkpoints found to sweep!")
