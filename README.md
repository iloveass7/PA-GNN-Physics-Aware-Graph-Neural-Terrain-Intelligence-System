<div align="center">
  <h1>🚀 PA-GNN: Physics-Aware Graph Neural Terrain Intelligence System</h1>
  <p><strong>A fully autonomous, pre-landing path planning system for Mars rovers using orbital imagery.</strong></p>

  [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
  [![PyG](https://img.shields.io/badge/PyG-%2339729E.svg?style=flat&logo=PyTorchGeometric&logoColor=white)](https://pyg.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

<br>

> **Institution:** Ahsanullah University of Science and Technology  
> **Course:** CSE-4733 Thesis / Project | Group: 4933  
> **Target Publication:** IEEE Robotics and Automation Letters (RA-L)  

## 🌌 Overview

Mars rovers are bottlenecked by manual path-planning on Earth. **PA-GNN** is an end-to-end framework that takes a single satellite photograph of previously unseen Martian terrain and automatically generates a safe, uncertainty-aware driving route—with zero human-in-the-loop and zero manual annotations.

By fusing **explicit physical measurements** (derived from stereo elevation models) with **learned visual semantics** (using a Masked Autoencoder and DeepLabV3+), PA-GNN dynamically constructs a **terrain-adaptive superpixel graph**. It processes this graph using a custom Physics-Aware Graph Attention Network (GATv2) to provide highly precise risk and uncertainty mapping.

---

## ✨ Key Contributions

1. 🌍 **Self-Supervised Pretraining:** Masked Autoencoder (MAE) pretraining on 17,000+ unlabelled CTX Mars orbital tiles.
2. ⛰️ **Zero Human Annotation:** CNN supervised entirely by automated slope and roughness features from real HiRISE DEMs.
3. 🕸️ **Adaptive Graph Resolution:** Superpixel graph density dynamically scales based on terrain complexity (flat = fewer nodes, hazardous = more nodes).
4. 🔗 **Physics-KNN Edges:** Nodes are connected by spatial proximity *and* physical feature similarity.
5. 🧠 **Physics-Aware GATv2:** Attention mechanisms injected with terrain physics priors, backed by an FFN diversity module to prevent over-smoothing.
6. 🛤️ **Uncertainty-Informed Routing:** Monte Carlo dropout maps epistemic uncertainty to penalize unsafe A* routing dynamically.

---

## 🏗️ System Architecture


The pipeline executes through 8 core stages:
1. **Pretraining:** MAE self-supervised learning on orbital imagery.
2. **Physics Engine:** Computes Slope, Roughness, and Discontinuity.
3. **CNN Semantic Estimator:** Predicts physical risk from visual features.
4. **Spatial Fusion:** Learns a per-pixel trust map to combine physics and CNN.
5. **Adaptive Graph:** SLIC-based adaptive node allocation.
6. **GNN:** Physics-Aware GATv2 + FFN refines risk estimates using neighborhood context.
7. **Uncertainty:** MC Dropout generates confidence mapping.
8. **Planning:** A* algorithm generates paths with per-waypoint risk attribution.

---

## 📂 Project Structure

```text
pa-gnn/
├── configs/            # All YAML configuration files
├── data/               # Raw, processed data, and train/test splits
├── src/
│   ├── data/           # Dataset classes and preprocessing logic
│   ├── models/         # Neural network modules (MAE, DeepLab, GATv2, Fusion)
│   ├── physics/        # Physics feature computation (Slope, Roughness)
│   ├── graph/          # Adaptive SLIC and PyG graph construction
│   ├── planning/       # A* and D* path planning
│   ├── uncertainty/    # MC Dropout inference
│   ├── training/       # Loss functions and generic trainers
│   └── evaluation/     # Metrics and demo scripts
├── scripts/            # Executable CLI entry points for pipeline stages
│   └── data_prep/      # Utility scripts for data validation and sync
├── checkpoints/        # Saved `.pth` model weights
└── results/            # Figures, logs, and tables
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and set up the environment:
```bash
git clone https://github.com/iloveass7/PA-GNN-Physics-Aware-Graph-Neural-Terrain-Intelligence-System.git
cd PA-GNN
pip install -r requirements.txt
```

### 2. Dataset Preparation
The system requires HiRISE Stereo DEMs, MurrayLab CTX tiles, and the HiRISE Map-Proj-v3 dataset. 
Place your downloaded data into `data/raw/` and use our management scripts to verify integrity:
```bash
python scripts/data_prep/nuclear.py
python scripts/data_prep/integrity_auditor.py
```

### 3. Execution Pipeline
The pipeline is designed to be executed sequentially via our CLI scripts:
```bash
# Phase 0: Pretraining
python scripts/train_mae.py

# Phase 1 & 2: Supervised Vision & Fusion
python scripts/train_cnn.py
python scripts/train_fusion.py

# Phase 3: Graph Construction & GNN
python scripts/precompute_graphs.py
python scripts/train_gnn.py

# Inference & Evaluation
python scripts/run_inference.py
python scripts/evaluate_all.py
```

---

## 👥 Authors

*   **Syed Abir Hossain** (20220104013)
*   **Ashik Mahmud** (20220104021)
*   **Mahadir Rahaman** (20220104046)

**Supervisor:** Tamanna Tabassum, Assistant Professor, Dept. of CSE, Ahsanullah University of Science and Technology.

---
<div align="center">
  <i>"Ad Astra Per Aspera" — To the stars through difficulties.</i>
</div>
