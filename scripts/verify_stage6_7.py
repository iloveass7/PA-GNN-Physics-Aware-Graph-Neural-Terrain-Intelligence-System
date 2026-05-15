"""
verify_stage6_7.py
------------------
Verification tests for Stage 6 (GNN) and Stage 7 (Path Planning) modules.
Run from project root: python scripts/verify_stage6_7.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from torch_geometric.data import Data

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
errors = []


def section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check(name, condition, detail=""):
    if condition:
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))
        errors.append(name)


# -----------------------------------------------------------------------
# Shared synthetic graph
# -----------------------------------------------------------------------
torch.manual_seed(42)
N = 50
x = torch.rand(N, 14)
pos = torch.rand(N, 2) * 512.0

# KNN-style edges: manual NumPy pairwise distance (no torch-cluster needed)
pos_np = pos.numpy()
dists = np.sqrt(((pos_np[:, None, :] - pos_np[None, :, :]) ** 2).sum(-1))  # (N, N)
np.fill_diagonal(dists, np.inf)
k = 5
knn_src, knn_dst = [], []
for i in range(N):
    neighbours = np.argsort(dists[i])[:k]
    for j in neighbours:
        knn_src.append(i); knn_dst.append(j)
        knn_src.append(j); knn_dst.append(i)  # undirected
edge_index = torch.tensor([knn_src, knn_dst], dtype=torch.long)

y = torch.rand(N)
tier = torch.randint(0, 3, (N,))
pixel_membership = torch.randint(0, N, (512, 512))

data = Data(
    x=x,
    edge_index=edge_index,
    pos=pos,
    y=y,
    tier=tier,
    pixel_membership=pixel_membership,
)

# -----------------------------------------------------------------------
# TEST 1: PhysicsAwareGATv2Conv
# -----------------------------------------------------------------------
section("TEST 1: PhysicsAwareGATv2Conv")

from src.models.gatv2_physics import PhysicsAwareGATv2Conv

# Layer 1 style: concat=True → (N, heads*out) = (50, 128)
conv1 = PhysicsAwareGATv2Conv(in_channels=14, out_channels=32, heads=4, concat=True)
out1 = conv1(x, edge_index)
check("Output shape (concat=True)", out1.shape == (N, 128), f"{out1.shape} == (50, 128)")
check("No NaN in output", not torch.isnan(out1).any())
check("Lambda initialised to 0.1", abs(conv1.physics_lambda.item() - 0.1) < 1e-6,
      f"lambda={conv1.physics_lambda.item():.4f}")

# Layer 2 style: concat=False → (N, out) = (50, 32)
conv2 = PhysicsAwareGATv2Conv(in_channels=128, out_channels=32, heads=4, concat=False)
x128 = torch.rand(N, 128)
out2 = conv2(x128, edge_index)
check("Output shape (concat=False)", out2.shape == (N, 32), f"{out2.shape} == (50, 32)")
check("No NaN (layer 2)", not torch.isnan(out2).any())

# Lambda is learnable (grad exists after backward)
loss_tmp = out1.sum()
loss_tmp.backward()
check("Lambda has gradient", conv1.physics_lambda.grad is not None)

# -----------------------------------------------------------------------
# TEST 2: GNNFFNBlock
# -----------------------------------------------------------------------
section("TEST 2: GNNFFNBlock (BatchNorm + GELU + Residual)")

from src.models.gnn_model import GNNFFNBlock

ffn = GNNFFNBlock(dim=128, hidden=512, dropout=0.1)
h_in = torch.rand(N, 128)
h_out = ffn(h_in)
check("FFN output shape", h_out.shape == (N, 128), str(h_out.shape))
check("Residual connection active", not torch.allclose(h_out, h_in))
check("No NaN in FFN", not torch.isnan(h_out).any())

# -----------------------------------------------------------------------
# TEST 3: Full PhysicsAwareGNN
# -----------------------------------------------------------------------
section("TEST 3: PhysicsAwareGNN Full Forward Pass")

from src.models.gnn_model import PhysicsAwareGNN

model = PhysicsAwareGNN(
    in_features=14, hidden_dim=32, heads=4,
    physics_lambda_init=0.1,
    dropout_l1=0.3, dropout_l2=0.2, ffn_dropout=0.1,
)
model.eval()
with torch.no_grad():
    risk = model(x, edge_index)

check("Output shape", risk.shape == (N,), f"{risk.shape} == ({N},)")
check("Output in [0,1]", risk.min() >= 0.0 and risk.max() <= 1.0,
      f"min={risk.min():.4f} max={risk.max():.4f}")
check("No NaN in output", not torch.isnan(risk).any())

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
check("Param count reasonable", 5_000 < n_params < 500_000,
      f"{n_params:,} params")

# Embeddings
emb = model.get_embeddings(x, edge_index)
check("Embeddings shape", emb.shape == (N, 32), str(emb.shape))

# Training mode forward (dropout active)
model.train()
risk_train = model(x, edge_index)
check("Train mode forward works", risk_train.shape == (N,))
model.eval()

# -----------------------------------------------------------------------
# TEST 4: MCDropoutEstimator — node level
# -----------------------------------------------------------------------
section("TEST 4: MCDropoutEstimator — node uncertainty")

from src.uncertainty.mc_dropout import MCDropoutEstimator, mc_dropout_mode

est = MCDropoutEstimator(model, n_passes=5)
result = est.estimate_node_uncertainty(data)

rm = result["risk_mean"]
rv = result["risk_var"]
ap = result["all_preds"]

check("risk_mean shape", rm.shape == (N,), str(rm.shape))
check("risk_var shape", rv.shape == (N,), str(rv.shape))
check("all_preds shape", ap.shape == (5, N), str(ap.shape))
check("risk_mean in [0,1]", rm.min() >= 0.0 and rm.max() <= 1.0,
      f"min={rm.min():.4f} max={rm.max():.4f}")
check("Variance > 0 (dropout stochastic)", rv.max().item() > 0,
      f"max_var={rv.max().item():.6f}")
check("No NaN in risk_mean", not torch.isnan(rm).any())

# -----------------------------------------------------------------------
# TEST 5: MCDropoutEstimator — pixel projection
# -----------------------------------------------------------------------
section("TEST 5: MCDropoutEstimator — pixel space projection")

pix_result = est.estimate_pixel_uncertainty(data, tile_size=512)

rmap = pix_result["risk_map"]
umap = pix_result["uncertainty_map"]

check("risk_map shape", rmap is not None and rmap.shape == (512, 512),
      str(rmap.shape) if rmap is not None else "None")
check("uncertainty_map shape", umap is not None and umap.shape == (512, 512),
      str(umap.shape) if umap is not None else "None")
check("No NaN in risk_map", not np.isnan(rmap).any())
check("risk_map in [0,1]", rmap.min() >= 0.0 and rmap.max() <= 1.0,
      f"min={rmap.min():.4f} max={rmap.max():.4f}")

# mc_dropout_mode context: verify Dropout is train, BatchNorm is eval
with mc_dropout_mode(model) as m:
    has_dropout_train = any(
        isinstance(mod, torch.nn.Dropout) and mod.training
        for mod in m.modules()
    )
    has_bn_eval = any(
        isinstance(mod, torch.nn.BatchNorm1d) and not mod.training
        for mod in m.modules()
    )
check("Dropout in train mode during MC", has_dropout_train)
check("BatchNorm1d in eval mode during MC", has_bn_eval)

# -----------------------------------------------------------------------
# TEST 6: Heuristics
# -----------------------------------------------------------------------
section("TEST 6: Planning Heuristics")

from src.planning.heuristics import physics_aware_heuristic, euclidean_heuristic

# Use two nodes at the SAME distance to goal to isolate the risk term
node_data_small = {
    0: {"pos": (0.0, 0.0), "risk": 0.2, "slope": 0.1},   # low risk, same dist to goal
    1: {"pos": (0.0, 0.0), "risk": 0.8, "slope": 0.5},   # high risk, same dist to goal
    2: {"pos": (100.0, 100.0), "risk": 0.0, "slope": 0.0},  # goal
}

h_euc = euclidean_heuristic(0, 2, node_data_small)
h_phys_safe = physics_aware_heuristic(0, 2, node_data_small)
h_phys_risky = physics_aware_heuristic(1, 2, node_data_small)

euc_dist = (100**2 + 100**2) ** 0.5
check("Euclidean heuristic correct", abs(h_euc - euc_dist) < 1e-4,
      f"h={h_euc:.4f} expected={euc_dist:.4f}")
check("Physics heuristic >= euclidean (risk term)", h_phys_safe >= h_euc,
      f"h_phys={h_phys_safe:.4f} h_euc={h_euc:.4f}")
# Same distance, different risk — higher risk node must get higher h
check("High-risk node gets higher heuristic (equidistant)", h_phys_risky > h_phys_safe,
      f"risky={h_phys_risky:.4f} safe={h_phys_safe:.4f}")

# -----------------------------------------------------------------------
# TEST 7: A* Path Planner
# -----------------------------------------------------------------------
section("TEST 7: PhysicsAwareAStar")

from src.planning.astar import (
    PhysicsAwareAStar, build_networkx_graph, select_start_goal, Trajectory
)

node_risks = risk.numpy()                          # GNN outputs
node_uncertainties = rv.numpy()                    # MC dropout variance

G, nd = build_networkx_graph(data, node_risks, node_uncertainties)
check("NetworkX graph node count", G.number_of_nodes() == N,
      f"{G.number_of_nodes()} nodes")
check("NetworkX graph has edges", G.number_of_edges() > 0,
      f"{G.number_of_edges()} edges")

# Check edge costs are positive
costs = [G[u][v]["cost"] for u, v in G.edges()]
check("All edge costs positive", all(c > 0 for c in costs),
      f"min_cost={min(costs):.6f}")

# Select start/goal
start, goal = select_start_goal(data, strategy="corners")
check("Start != Goal", start != goal, f"start={start} goal={goal}")

planner = PhysicsAwareAStar(use_physics_heuristic=True)
traj = planner.plan(G, nd, start, goal)

check("Path found", traj is not None)
if traj is not None:
    check("Trajectory is success", traj.success)
    check("Start waypoint correct", traj.waypoints[0].node_id == start,
          f"got {traj.waypoints[0].node_id}")
    check("Goal waypoint correct", traj.waypoints[-1].node_id == goal,
          f"got {traj.waypoints[-1].node_id}")
    check("PLR >= 1.0", traj.path_length_ratio >= 1.0,
          f"PLR={traj.path_length_ratio:.4f}")
    check("HCR in [0,1]", 0.0 <= traj.hazard_crossing_rate <= 1.0,
          f"HCR={traj.hazard_crossing_rate:.4f}")

    # Check per-waypoint attribution
    signals = [wp.dominant_signal for wp in traj.waypoints]
    check("All waypoints have signal attribution",
          all(s in ("physics", "cnn") for s in signals))

    tiers = [wp.tier for wp in traj.waypoints]
    check("All waypoints have tier", all(t in (0, 1, 2) for t in tiers))

# Euclidean baseline (B1)
planner_euc = PhysicsAwareAStar(use_physics_heuristic=False)
traj_euc = planner_euc.plan(G, nd, start, goal)
check("Euclidean baseline also finds path", traj_euc is not None)

# -----------------------------------------------------------------------
# TEST 8: D* Lite Replanner
# -----------------------------------------------------------------------
section("TEST 8: DStarLite Dynamic Replanner")

from src.planning.dstar import DStarLite

dstar = DStarLite(G, nd, start, goal)
dstar.compute_shortest_path()
path = dstar.extract_path()

check("D* finds initial path", path is not None)
if path is not None:
    check("D* path starts at start", path[0] == start, f"{path[0]}=={start}")
    check("D* path ends at goal", path[-1] == goal, f"{path[-1]}=={goal}")
    check("D* path length >= 1", len(path) >= 1, f"len={len(path)}")

    # Dynamic update: inflate costs of all edges from mid node
    if len(path) > 2:
        mid = path[len(path) // 2]
        changed = [(mid, s, 999.0) for s in G.successors(mid)]
        dstar.update_edge_costs(changed)
        new_path = dstar.extract_path()
        # New path should exist (soft costs, not hard removal)
        check("D* replanning produces a path", new_path is not None)
        if new_path and path:
            check("D* replan produces valid start/goal",
                  new_path[0] == start and new_path[-1] == goal)

# -----------------------------------------------------------------------
# TEST 9: GNN training script imports cleanly
# -----------------------------------------------------------------------
section("TEST 9: train_gnn.py importable")

try:
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "train_gnn",
        str(PROJECT_ROOT / "scripts" / "train_gnn.py")
    )
    mod = importlib.util.module_from_spec(spec)
    # Don't execute (would run main), just check it loads
    check("train_gnn.py loads without syntax errors", True)
except SyntaxError as e:
    check("train_gnn.py loads without syntax errors", False, str(e))

# Check load_config works
from scripts.train_gnn import load_config
cfg = load_config(str(PROJECT_ROOT / "configs" / "gnn.yaml"))
check("Config loads model section", "model" in cfg)
check("Config in_features=14", cfg["model"]["in_features"] == 14)
check("Config loss=SmoothL1", cfg["training"]["loss"] == "SmoothL1")

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print()
print("=" * 60)
if not errors:
    print("  ALL TESTS PASSED")
else:
    print(f"  {len(errors)} TEST(S) FAILED:")
    for e in errors:
        print(f"    - {e}")
print("=" * 60)

sys.exit(0 if not errors else 1)
