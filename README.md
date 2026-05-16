# Physics-Aware Graph Neural Navigation (PA-GNN)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**PA-GNN** is a novel, end-to-end Physics-Aware Graph Neural Network system designed for autonomous planetary rover navigation in highly uncertain, extreme terrains (e.g., Martian craters, dune fields). This repository contains the official implementation targeting publication in **IEEE Robotics and Automation Letters (RA-L)**.

By tightly coupling deterministic physics models with deep geometric learning (GATv2), PA-GNN overcomes the fragility of pure end-to-end vision models. It dynamically fuses domain-invariant physics priors with high-level visual semantics, producing a robust, uncertainty-aware topological graph for safe A*/D* path planning.

---

## 🌟 Key Contributions

1. **Spatial Adaptive Fusion (Stage 4):** A learned $\alpha(x,y)$ mechanism that dynamically balances physical constraints ($H_{physics}$) and learned visual semantics ($H_{learned}$) based on localized terrain ambiguity.
2. **Continuous Adaptive Graph Allocation (Stage 5):** Replaces rigid grid structures with a scale-free, power-law driven superpixel allocation. Node density is intrinsically tied to terrain complexity, preserving high-frequency hazard boundaries while saving compute in flat regions.
3. **Physics-Aware GATv2 Attention (Stage 6):** Embeds an $L_1$ physics similarity constraint ($S, R, D, U, \alpha, \text{area}$) directly into the GNN attention mechanism before the softmax operator, enforcing strict physical continuity across learned topological representations.
4. **Calibrated Epistemic Uncertainty (Stage 7):** Provides rigorous out-of-distribution (OOD) detection via Monte Carlo Dropout paired with Temperature Scaling, driving uncertainty-penalized path planning.

---

## 🏗 System Architecture (The 9-Stage Pipeline)

PA-GNN operates on a comprehensive 9-stage pipeline, processing raw orbital imagery into traversable trajectories.

### Pretraining & Data
*   **Stage 0 (Self-Supervised Pretraining):** Masked Autoencoder (MAE) with a MobileNetV3-Large backbone trained on unlabelled MurrayLab CTX tiles to learn robust Martian terrain representations.
*   **Stage 1 (Data Prep & Validation):** Co-registration of HiRISE optical imagery (25cm/px) and DEM elevation data (1m/px), validated against MOLA MEGDR baselines.

### Dual-Stream Perception
*   **Stage 2 (Physics Feature Engine):** Deterministic proxy extraction. Computes Slope ($S$), Roughness ($R$), and Discontinuity ($D$). Combined as $H_{physics} = 0.4S + 0.3R + 0.3D$.
*   **Stage 3 (CNN Risk Estimator):** Weakly-supervised DeepLabV3+ with MAE-initialized MobileNetV3. Predicts a purely semantic hazard mask ($H_{learned}$).

### Geometric Deep Learning
*   **Stage 4 (Adaptive Fusion):** A 3-layer CNN predicts $\alpha \in [0,1]$ to yield the final fused hazard map: $H_{final} = \alpha H_{learned} + (1-\alpha) H_{physics}$.
*   **Stage 5 (Adaptive Superpixel Graph):** Converts $H_{final}$ into a topological graph using continuous power-law SLIC segmentation. Edges are formed via KDTree-based Physics-KNN with RAG connectivity guarantees.
*   **Stage 6 (Physics-Aware GNN):** A 2-layer GATv2 network refines node safety states through physics-constrained message passing and a custom `GNNFFNBlock`.

### Uncertainty & Planning
*   **Stage 7 (Uncertainty Estimation):** Computes $U(x,y)$ using 5 forward MC Dropout passes. Includes auto-fallback timing and Temperature Scaling calibration.
*   **Stage 8 (Path Planning):** Weighted A* and D* Lite algorithms operating on the GNN graph. Edge costs are exponentially scaled by predicted risk and linearly penalized by high epistemic uncertainty ($1 + 2U_i$).

---

## 🚀 Installation & Setup

### Requirements
*   **OS:** Linux (Ubuntu 20.04/22.04 recommended) or Windows 11 / WSL2
*   **CUDA:** 11.8 or newer (required for PyTorch Geometric)
*   **Python:** 3.10+

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-org/PA-GNN.git
cd PA-GNN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install PyTorch (ensure your CUDA version matches)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric dependencies
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install remaining dependencies
pip install -r requirements.txt
```

---

## 📊 Evaluation & Scientific Metrics

The PA-GNN framework includes a rigorous IEEE RA-L compliant evaluation suite located in `src/evaluation/`.

*   **Failure Analysis (`failure_analysis.py`)**: Automatically detects edge cases such as Crater-Shadow Confusion, Alpha Collapse, and Graph Disconnection.
*   **Oversmoothing (`oversmoothing.py`)**: Tracks Dirichlet Energy and Pairwise Cosine Similarity to guarantee GNN layer stability and prevent representation collapse.
*   **Calibration (`calibration.py`)**: Generates Expected Calibration Error (ECE) and Reliability Diagrams for the uncertainty module.
*   **Statistical Validation (`statistics.py`)**: Implements Paired T-tests, Wilcoxon Signed-Rank tests, and 95% Bootstrap CIs for robust ablation comparisons.

---

## 💻 Usage

### 1. Training the Pipeline
The pipeline is designed to be trained sequentially. Configuration parameters for all stages are localized in the `configs/` directory.

```bash
# Train Stage 0 (MAE Pretraining)
python scripts/train_mae.py

# Train Stage 3 (CNN Risk Estimator)
python scripts/train_cnn.py

# Train Stage 4 (Adaptive Fusion)
python scripts/train_fusion.py

# Precompute Graphs (Stage 5)
python scripts/precompute_graphs.py

# Train Stage 6 (GNN)
python scripts/train_gnn.py
```

### 2. Running Inference & Planning
```bash
# Run the full pipeline on a test tile and profile latency
python scripts/run_inference.py --input test_tile.npy --profile
```

---

## 📁 Repository Structure

```text
PA-GNN/
├── configs/                  # YAML configurations (cnn, fusion, gnn, physics)
├── data/                     # Raw and processed Martian terrain data
├── scripts/                  # Executable runners (training, data prep, inference)
└── src/
    ├── data/                 # Data loaders, tiling, and augmentations
    ├── evaluation/           # IEEE RA-L metrics, statistics, and failure analysis
    ├── graph/                # Adaptive SLIC, Graph Builder, Node Features, Edge Scorer
    ├── models/               # MAE, GATv2, Fusion, and CNN architectures
    ├── physics/              # Slope, Roughness, and Discontinuity extraction
    ├── planning/             # Physics-aware A* and D* Lite algorithms
    ├── training/             # Loss functions and Trainer loops
    ├── uncertainty/          # MC Dropout and Temperature Scaling
    └── utils/                # CUDATimer, MemoryTracker, and common utilities
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Citation
If you use PA-GNN or its evaluation suite in your research, please cite our upcoming paper:
*(Citation details to be added post-publication)*
