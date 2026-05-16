"""Audit every file claimed in progression.md — checks existence and size."""
import os
from pathlib import Path

root = Path(r"d:\PA-GNN-Physics-Aware-Graph-Neural-Terrain-Intelligence-System")

files = {
    "Stage 0 (MAE) - COMPLETE": [
        "src/models/encoder.py",
        "src/models/decoder.py",
        "src/models/mae.py",
        "src/data/ctx_loader.py",
        "scripts/train_mae.py",
    ],
    "Stage 1 (Data Prep) - COMPLETE": [
        "src/data/dem_loader.py",
        "src/data/hirise_loader.py",
        "src/data/dem_processing.py",
        "src/data/tiling.py",
        "src/data/augmentations.py",
        "src/data/normalize.py",
        "src/data/label_generation.py",
        "scripts/process_dems.py",
        "scripts/tile_dataset.py",
        "scripts/mola_validation.py",
    ],
    "Stage 2 (Physics) - COMPLETE": [
        "src/physics/slope.py",
        "src/physics/roughness.py",
        "src/physics/discontinuity.py",
        "src/physics/combine.py",
        "configs/physics.yaml",
        "scripts/validate_dataset.py",
        "scripts/precompute_graphs.py",
    ],
    "Stage 3 (CNN) - COMPLETE": [
        "src/models/risk_model.py",
        "src/models/ffn_module.py",
        "src/training/losses.py",
        "src/training/trainer.py",
        "configs/cnn.yaml",
        "scripts/train_cnn.py",
    ],
    "Stage 4 (Fusion) - PENDING": [
        "src/models/fusion.py",
        "scripts/train_fusion.py",
        "configs/fusion.yaml",
    ],
    "Stage 5 (Graph) - PENDING": [
        "src/graph/adaptive_slic.py",
        "src/graph/node_features.py",
        "src/graph/edges.py",
        "src/graph/graph_builder.py",
    ],
    "Stage 6 (GNN) - COMPLETE": [
        "src/models/gatv2_physics.py",
        "src/models/gnn_model.py",
        "configs/gnn.yaml",
        "scripts/train_gnn.py",
    ],
    "Stage 7 (Uncertainty) - COMPLETE": [
        "src/uncertainty/mc_dropout.py",
    ],
    "Stage 8 (Path Planning) - COMPLETE": [
        "src/planning/astar.py",
        "src/planning/heuristics.py",
        "src/planning/dstar.py",
    ],
    "Supporting (not in progression)": [
        "src/pipeline.py",
        "src/visualization.py",
        "src/utils.py",
        "src/evaluation/metrics.py",
        "src/evaluation/evaluate_dem.py",
        "src/evaluation/evaluate_hirise.py",
        "src/evaluation/demo_ctx.py",
        "src/data/graph_dataset.py",
        "src/data/label_remap.py",
        "scripts/evaluate_all.py",
        "scripts/run_ablations.py",
        "scripts/run_inference.py",
        "scripts/download_dems.py",
        "configs/base.yaml",
        "configs/mae.yaml",
    ],
}

total_ok = 0
total_stub = 0
total_missing = 0
problems = []

for stage, paths in files.items():
    print()
    print("--- {} ---".format(stage))
    for rel in paths:
        fp = root / rel
        if not fp.exists():
            print("  {:62s}  MISSING".format(rel))
            total_missing += 1
            if "PENDING" not in stage:
                problems.append(("MISSING", stage, rel))
        else:
            size = fp.stat().st_size
            if size < 10:
                print("  {:62s}  EMPTY STUB ({} bytes)".format(rel, size))
                total_stub += 1
                if "PENDING" not in stage and "Supporting" not in stage:
                    problems.append(("EMPTY", stage, rel))
            else:
                print("  {:62s}  {:>6d} bytes  OK".format(rel, size))
                total_ok += 1

print()
print("=" * 80)
print("SUMMARY: {} OK, {} empty stubs, {} missing".format(total_ok, total_stub, total_missing))
print()

if problems:
    print("PROBLEMS (claimed COMPLETE but empty/missing):")
    for kind, stage, rel in problems:
        print("  [{}] {} -> {}".format(kind, stage, rel))
else:
    print("No problems found in COMPLETE stages.")

# Check what data directories exist
print()
print("--- Data directories ---")
for d in ["data/raw", "data/processed", "data/splits", "checkpoints", "results"]:
    dp = root / d
    if dp.exists():
        count = sum(1 for _ in dp.rglob("*") if _.is_file())
        print("  {:40s}  EXISTS ({} files)".format(d, count))
    else:
        print("  {:40s}  NOT FOUND".format(d))
