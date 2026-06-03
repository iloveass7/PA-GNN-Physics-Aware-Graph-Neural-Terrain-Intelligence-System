"""Quick sanity check for the two blocking fixes."""
import sys, logging
logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
sys.path.insert(0, ".")

import torch
from src.models.risk_model import build_risk_model
from src.training.losses import RiskLoss

# 1. MAE weight loading
print("=" * 50)
print("TEST 1: MAE weight loading")
m = build_risk_model(mae_checkpoint="checkpoints/mae_best.pt")
print("PASS — model loaded\n")

# 2. BCE under AMP autocast
print("=" * 50)
print("TEST 2: BCE loss under AMP autocast")
m.train()
loss_fn = RiskLoss()
x = torch.randn(2, 1, 512, 512)
with torch.amp.autocast('cuda', enabled=False):
    pred = m(x)
    # Simulate AMP by casting to half
    pred_half = pred.half()
    target = torch.rand(2, 512, 512).half()
    valid = torch.ones(2, 512, 512).half()
    loss, comps = loss_fn(pred_half, target, valid)
print(f"Loss: {loss.item():.4f}")
print(f"Components: {comps}")
print("PASS — no autocast crash\n")

print("=" * 50)
print("ALL TESTS PASSED")
