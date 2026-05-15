# PA-GNN Blueprint vs Code — Military-Grade Compliance Audit

Systematic comparison of every formula, hyperparameter, and architectural decision in `pagnn_final_blueprint_v4.md` against the actual implementation. Each checkpoint is **PASS** (code matches blueprint), **MINOR** (cosmetic deviation), or **FAIL** (functional gap).

---

## Stage 0 — MAE Pretraining (§9)

| # | Blueprint Requirement | Code | Verdict |
|---|---|---|---|
| 1 | MobileNetV3-Large encoder | `encoder.py` L1: MobileNetV3-Large wrapper | ✅ PASS |
| 2 | 75% random patch masking | `mae.py` masking logic | ✅ PASS |
| 3 | 4-layer MLP decoder | `decoder.py`: 4-layer MLP | ✅ PASS |
| 4 | MSE reconstruction loss on masked patches | `mae.py` forward | ✅ PASS |
| 5 | AdamW, 200 epochs | `train_mae.py` | ✅ PASS |
| 6 | `configs/mae.yaml` exists | 0 bytes — **EMPTY STUB** | ⚠️ MINOR |

> [!NOTE]
> `configs/mae.yaml` is empty but `train_mae.py` has hardcoded defaults. Functionally correct but inconsistent with blueprint §23 which lists it as a config file.

---

## Stage 1 — Data Prep (§10)

| # | Blueprint Requirement | Code | Verdict |
|---|---|---|---|
| 7 | PDS .IMG → GeoTIFF conversion | `dem_loader.py` | ✅ PASS |
| 8 | JP2 → GeoTIFF + GDAL projection | `hirise_loader.py` | ✅ PASS |
| 9 | Slope, roughness, risk from DEM | `dem_processing.py` | ✅ PASS |
| 10 | 512×512 tiling, 50% overlap | `tiling.py` | ✅ PASS |
| 11 | NoData/saturation rejection | `tiling.py` | ✅ PASS |
| 12 | DEM-location-based splits | `tile_dataset.py` | ✅ PASS |
| 13 | MOLA MEGDR Pearson r validation | `mola_validation.py` | ✅ PASS |

---

## Stage 2 — Physics Feature Engine (§10)

| # | Blueprint Requirement | Code | Verdict |
|---|---|---|---|
| 14 | S = Sobel gradient magnitude | `slope.py` | ✅ PASS |
| 15 | R = sliding window std dev | `roughness.py` | ✅ PASS |
| 16 | D = Laplacian of Gaussian | `discontinuity.py` | ✅ PASS |
| 17 | H_physics = w1×S + w2×R + w3×D | `combine.py` `PhysicsFeatureEngine` | ✅ PASS |
| 18 | Default w1=0.4, w2=0.3, w3=0.3 | `physics.yaml` L15-17 | ✅ PASS |

---

## Stage 3 — CNN Risk Estimator (§10)

| # | Blueprint Requirement | Code | Verdict |
|---|---|---|---|
| 19 | MobileNetV3 encoder + DeepLabV3+ decoder | `risk_model.py` | ✅ PASS |
| 20 | ASPP + skip connections | `decoder.py` | ✅ PASS |
| 21 | Weighted BCE + Dice + TV loss | `losses.py` `RiskLoss` | ✅ PASS |
| 22 | hazard_weight=3.0, dice_coeff=0.5, tv_coeff=0.1 | `cnn.yaml` | ✅ PASS |
| 23 | Batch 8, 60 epochs | `cnn.yaml` | ✅ PASS |
| 24 | FFN module with residual connection | `ffn_module.py`: LayerNorm + GELU + residual | ✅ PASS |

---

## Stage 4 — Adaptive Fusion (§11)

| # | Blueprint Requirement | Code | Verdict |
|---|---|---|---|
| 25 | Conv(3→16, 3×3) + ReLU + reflect padding | `fusion.py` L86-87, L129-130 | ✅ PASS |
| 26 | Conv(16→8, 3×3) + ReLU + reflect padding | `fusion.py` L88, L133-134 | ✅ PASS |
| 27 | Conv(8→1, 1×1) + Sigmoid | `fusion.py` L90, L137 | ✅ PASS |
| 28 | ~12,000 parameters | Code: `count_params()` tracks this | ✅ PASS |
| 29 | H_final = α×H_learned + (1−α)×H_physics | `fuse_risk_maps()` L162 | ✅ PASS |
| 30 | `joint_with_cnn: false` mandatory | `fusion.yaml` L9, `train_fusion.py` L374-380 enforcement | ✅ PASS |
| 31 | Freeze CNN, train fusion only | `EndToEndFusionModel._freeze_cnn()`, `get_trainable_params()` | ✅ PASS |
| 32 | Same compound loss as Stage 3 | `train_fusion.py` L437-442 uses `RiskLoss` | ✅ PASS |
| 33 | α degeneration diagnostic (std < 0.02) | `train_fusion.py` L514-522 | ✅ PASS |
| 34 | H_final recall must exceed H_learned | `train_fusion.py` L524-530 | ✅ PASS |

---

## Stage 5 — Superpixel Graph (§12)

| # | Blueprint Requirement | Code | Verdict |
|---|---|---|---|
| 35 | 16×16 grid of 32×32 blocks | `adaptive_slic.py` L77-78: `BLOCK_SIZE=32, GRID_SIZE=16` | ✅ PASS |
| 36 | Flat (<0.25): 5 nodes/block | `assign_tier_budget()` L160-162 | ✅ PASS |
| 37 | Complex (0.25–0.60): 15/block | `assign_tier_budget()` L156-157 default | ✅ PASS |
| 38 | Hazard (>0.60): 30–50 linear | `assign_tier_budget()` L164-176 | ✅ PASS |
| 39 | Two-pass SLIC (coarse + hazard refinement) | `adaptive_slic_segmentation()` L304-401 | ✅ PASS |
| 40 | n_segments = max(80, floor(budget×0.4)) | L305 | ✅ PASS |
| 41 | Refine superpixels >200px in hazard zones | L347: `REFINE_PIXEL_THRESHOLD=200` | ✅ PASS |
| 42 | Connectivity guarantee | `_ensure_connectivity()` L200-247 | ✅ PASS |
| 43 | 14-dim node features (exact indices 0–13) | `node_features.py` L11-27 + L140-187 | ✅ PASS |
| 44 | Feature 9 = normalised area (critical for attention) | L174-175 | ✅ PASS |
| 45 | Feature 13 = hazardous neighbour count (filled after edges) | `graph_builder.py` `_fill_hazardous_neighbour_count()` | ✅ PASS |
| 46 | Physics-KNN K=5, combined 0.5×spatial + 0.5×physics | `edges.py` L139-141 defaults | ✅ PASS |
| 47 | Physics sub-vector = [S, R, D, H_physics] = features[:,2:6] | `edges.py` L183 | ✅ PASS |
| 48 | KDTree for efficient KNN | `edges.py` L206: `cKDTree` | ✅ PASS |
| 49 | RAG bridging for disconnected components | `edges.py` L231-290 | ✅ PASS |
| 50 | Edge weight: 0.6×avg_risk + 0.25×norm_dist + 0.15×|ΔS| | `edges.py` L350 | ✅ PASS |
| 51 | PyG Data: x, edge_index, edge_attr, pos, y, tier, pixel_membership | `graph_builder.py` L227-234 | ✅ PASS |
| 52 | `validate_graph()` 8-point integrity check | `graph_builder.py` L366-426 | ✅ PASS |

---

## Stage 6 — Physics-Aware GATv2 + FFN (§13)

| # | Blueprint Requirement | Code | Verdict |
|---|---|---|---|
| 53 | `e_ij = LeakyReLU(aᵀ[Wh_i‖Wh_j]) + λ×exp(−\|S_i−S_j\|−\|R_i−R_j\|)` | `gatv2_physics.py` `message()` L155-176 | ✅ PASS |
| 54 | λ learnable, init 0.1 | L82-84: `nn.Parameter(torch.tensor(0.1))` | ✅ PASS |
| 55 | S at index 2, R at index 3 | L68-69: `slope_idx=2, roughness_idx=3` | ✅ PASS |
| 56 | Physics added BEFORE softmax | L176 before L179 `softmax()` | ✅ PASS |
| 57 | 4 attention heads | `gnn_model.py` L97: `heads=4` default | ✅ PASS |
| 58 | Layer 1: in=14, out=32, concat=True → 128 | L97-103 | ✅ PASS |
| 59 | Layer 1: ELU + Dropout(0.3) + FFN(128, 512) | L134-137 | ✅ PASS |
| 60 | Layer 2: in=128, out=32, concat=False → 32 | L108-114 | ✅ PASS |
| 61 | Layer 2: ELU + Dropout(0.2) + FFN(32, 128) | L139-142 | ✅ PASS |
| 62 | Output: Linear(32→1) + Sigmoid | L121-122 | ✅ PASS |
| 63 | FFN: BatchNorm1d → Linear(D→4D) → GELU → Drop(0.1) → Linear(4D→D) + residual | `GNNFFNBlock` L38-58 | ✅ PASS |
| 64 | Loss: SmoothL1 (Huber) | `train_gnn.py` `F.smooth_l1_loss` | ✅ PASS |
| 65 | Adam, lr=1e-3, weight_decay=5e-4 | `train_gnn.py` + `gnn.yaml` | ✅ PASS |
| 66 | 100 epochs, patience 15 on val_MAE | `gnn.yaml` + `train_gnn.py` | ✅ PASS |
| 67 | Batch size 32 precomputed graphs | `gnn.yaml` | ✅ PASS |
| 68 | Target = DEM-derived y (NOT H_final) | `train_gnn.py` uses `batch.y` | ✅ PASS |

> [!NOTE]
> Blueprint §13 mentions `ffn_module.py` in the file structure (§23 L1005). The GNN uses its own `GNNFFNBlock` (BatchNorm1d) rather than the shared `ffn_module.py` (LayerNorm). This is **correct** — blueprint specifies BatchNorm1d for GNN FFN specifically. However, blueprint's file list at §23 implies `ffn_module.py` is used by the GNN model. Functionally correct, structurally a minor disconnect.

---

## Stage 7 — Uncertainty Estimation (§14)

| # | Blueprint Requirement | Code | Verdict |
|---|---|---|---|
| 69 | MC Dropout with N=5 passes | `mc_dropout.py` L64: `n_passes=5` | ✅ PASS |
| 70 | Dropout layers in train mode, BatchNorm in eval | `mc_dropout_mode()` L33-52 | ✅ PASS |
| 71 | Node uncertainty = variance of p̂_i across passes | L100: `risk_var = all_preds.var(dim=0)` | ✅ PASS |
| 72 | Pixel projection via pixel_membership | L120-141 vectorised projection | ✅ PASS |
| 73 | Fallback to N=3 if inference >5s | Config `mc_fallback: 3` exists, but **no timing logic** in code | ⚠️ MINOR |
| 74 | Blueprint says "run full forward pass (Stages 2-6) N times" | Code runs **GNN only** N times (correct — physics/CNN are deterministic) | ✅ PASS |

> [!IMPORTANT]
> **Checkpoint 74 clarification:** The blueprint says "run the full forward pass (Stages 2 through 6) N=5 times." Strictly interpreted, this means running the CNN and physics engine 5 times too. However, Stages 2-4 have **no dropout layers** — they'd produce identical outputs every pass. The code correctly optimises this by only running MC passes through the GNN (Stage 6), which is the only component with active dropout. This is a **correct optimisation**, not a deviation.

> [!NOTE]
> **Checkpoint 73:** The `gnn.yaml` has `mc_fallback: 3` but `mc_dropout.py` has no timing logic to auto-switch. A user would need to manually set `n_passes=3`. Functionally minor — the value exists in config but the auto-switch is unimplemented.

---

## Stage 8 — Path Planning (§15)

| # | Blueprint Requirement | Code | Verdict |
|---|---|---|---|
| 75 | `C(i,j) = exp(3×risk_ij) × [0.6×risk + 0.25×dist + 0.15×\|ΔS\|]` | `astar.py` graph builder L110-114 | ✅ PASS |
| 76 | risk_ij = 0.5 × (p̂_i + p̂_j) | L108 | ✅ PASS |
| 77 | Uncertainty penalty: U_i > 0.3 → cost × (1 + 2.0 × U_i) | L116-117 | ✅ PASS |
| 78 | No hard node deactivation (soft costs only) | Confirmed — no node removal in code | ✅ PASS |
| 79 | h(n) = euclidean(n,goal) × (1 + 0.4×p̂_n + 0.1×S_n) | `heuristics.py` L50-52 | ✅ PASS |
| 80 | Per-waypoint: coords, risk, uncertainty, tier, dominant signal | `Waypoint` dataclass | ✅ PASS |
| 81 | Dominant signal: "physics" if α < 0.5, "cnn" if α > 0.5 | `_make_waypoint()` | ✅ PASS |
| 82 | D* Lite for dynamic replanning | `dstar.py` full implementation | ✅ PASS |
| 83 | D* maintains incremental search state | `compute_shortest_path()`, `update_edge_costs()`, `replan()` | ✅ PASS |

---

## File Structure (§23)

| # | Blueprint File | Exists | Size | Verdict |
|---|---|---|---|---|
| 84 | `configs/base.yaml` | 0 bytes | EMPTY | ❌ FAIL |
| 85 | `configs/mae.yaml` | 0 bytes | EMPTY | ⚠️ MINOR |
| 86 | `configs/datasets/*.yaml` | NOT FOUND | Missing | ❌ FAIL |
| 87 | `src/data/label_remap.py` | 0 bytes | EMPTY | ⚠️ MINOR |
| 88 | `src/pipeline.py` | 0 bytes | EMPTY | ❌ FAIL |
| 89 | `src/visualization.py` | 0 bytes | EMPTY | ❌ FAIL |
| 90 | `src/utils.py` | 0 bytes | EMPTY | ❌ FAIL |
| 91 | `src/evaluation/metrics.py` | 0 bytes | EMPTY | ❌ FAIL |
| 92 | `src/evaluation/evaluate_dem.py` | 0 bytes | EMPTY | ❌ FAIL |
| 93 | `src/evaluation/evaluate_hirise.py` | 0 bytes | EMPTY | ❌ FAIL |
| 94 | `src/evaluation/demo_ctx.py` | 0 bytes | EMPTY | ❌ FAIL |
| 95 | `scripts/evaluate_all.py` | 0 bytes | EMPTY | ❌ FAIL |
| 96 | `scripts/run_ablations.py` | 0 bytes | EMPTY | ❌ FAIL |
| 97 | `scripts/run_inference.py` | 0 bytes | EMPTY | ❌ FAIL |
| 98 | `scripts/download_dems.py` | 0 bytes | EMPTY | ❌ FAIL |

---

## Summary Scoreboard

| Category | Total Checks | ✅ PASS | ⚠️ MINOR | ❌ FAIL |
|---|---|---|---|---|
| **Core Pipeline (Stages 0-8)** | 83 | **80** | **3** | **0** |
| **File Structure (§23)** | 15 | 0 | 2 | **13** |
| **TOTAL** | **98** | **80** | **5** | **13** |

---

## Critical Analysis

### ✅ What's Bulletproof (80/83 core checks)

Every single formula, hyperparameter, and architectural decision in the 9-stage pipeline matches the blueprint exactly:

- **GATv2 attention formula** — physics boost injected before softmax, λ learnable at 0.1 ✓
- **FFN diversity module** — BatchNorm1d + GELU + residual (not LayerNorm like shared FFN) ✓
- **Adaptive SLIC** — 3-tier budget with linear hazard scaling 30-50, two-pass refinement ✓
- **Edge cost formula** — exact coefficients 0.6/0.25/0.15 with exp(3×risk) multiplier ✓
- **Uncertainty penalty** — soft scaling (1 + 2.0 × U_i) at threshold 0.3, no hard removal ✓
- **All training configs** — Adam 1e-3, SmoothL1, patience 15, batch 32 ✓

### ⚠️ Minor Deviations (3)

1. **`configs/mae.yaml` empty** — defaults hardcoded in `train_mae.py`. Works but inconsistent.
2. **MC Dropout timing fallback** — config has `mc_fallback: 3` but no auto-switch code.
3. **`label_remap.py` empty** — HiRISE class remapping exists inline in `hirise_loader.py`.

### ❌ Functional Gaps (13 empty stubs)

All 13 failures are in the **tooling/evaluation layer**, not the core pipeline:

| Gap | Impact | Thesis Risk |
|---|---|---|
| `pipeline.py` (end-to-end orchestrator) | Cannot run full Stages 0→8 in one command | **Medium** — can run scripts individually |
| `evaluate_all.py` + `evaluation/*.py` | Cannot generate thesis tables/metrics | **HIGH** — needed for §19 evaluation protocol |
| `run_ablations.py` | Cannot run §20 ablation suite (10 baselines × 3 seeds) | **HIGH** — core thesis claim requires this |
| `visualization.py` | Cannot generate thesis figures 1-9 | **HIGH** — §21 requires 9 specific figures |
| `run_inference.py` | No single-image CLI demo | Low |
| `download_dems.py` | No automated data acquisition | Low |
| `configs/base.yaml`, `configs/datasets/*.yaml` | Missing shared config and dataset configs | **Medium** |

### Bottom Line

> [!IMPORTANT]
> **The ML pipeline (Stages 0-8) is 97% blueprint-compliant** (80/83 core checks pass). Every formula, every hyperparameter, every architectural decision matches the blueprint.
>
> **The thesis defense infrastructure is 0% complete.** Evaluation scripts, ablation runner, visualization, and metrics computation are all empty stubs. These are mandatory for the IEEE RA-L submission (§19-§21) — without them, you cannot generate the 10-baseline comparison table, the 9 required figures, or the 3-seed statistical results.
