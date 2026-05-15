# PA-GNN: Physics-Aware Graph Neural Terrain Intelligence System
## Final Thesis Blueprint — Version 4.0
### Includes: Adaptive Graph Resolution + Simplified Structure

**Institution:** Ahsanullah University of Science and Technology
**Department:** Computer Science and Engineering
**Course:** CSE-4733 Thesis / Project | Group: 4933
**Supervisor:** Tamanna Tabassum, Assistant Professor, Dept. of CSE
**Authors:** Syed Abir Hossain (20220104013), Ashik Mahmud (20220104021), Mahadir Rahaman (20220104046)
**Target Publication:** IEEE Robotics and Automation Letters (RA-L)
**Document purpose:** Complete specification for implementation by a coding agent. Every architectural decision, training procedure, dataset, evaluation metric, and paper writing requirement is documented here. No detail is left to assumption.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [The Core Argument](#2-the-core-argument)
3. [Six Novel Contributions](#3-six-novel-contributions)
4. [Related Work and Positioning](#4-related-work-and-positioning)
5. [Datasets](#5-datasets)
6. [System Architecture Overview](#6-system-architecture-overview)
7. [Stage 0 — MAE Self-Supervised Pretraining](#7-stage-0--mae-self-supervised-pretraining)
8. [Stage 1 — DEM Data Preparation and Label Generation](#8-stage-1--dem-data-preparation-and-label-generation)
9. [Stage 2 — Physics Feature Engine](#9-stage-2--physics-feature-engine)
10. [Stage 3 — CNN Semantic Risk Estimator](#10-stage-3--cnn-semantic-risk-estimator)
11. [Stage 4 — Spatial Adaptive Fusion](#11-stage-4--spatial-adaptive-fusion)
12. [Stage 5 — Adaptive-Resolution Superpixel Graph](#12-stage-5--adaptive-resolution-superpixel-graph)
13. [Stage 6 — Physics-Aware GATv2 with FFN Module](#13-stage-6--physics-aware-gatv2-with-ffn-module)
14. [Stage 7 — Uncertainty Estimation](#14-stage-7--uncertainty-estimation)
15. [Stage 8 — Path Planning](#15-stage-8--path-planning)
16. [System Outputs](#16-system-outputs)
17. [Training Pipeline — Four Phases](#17-training-pipeline--four-phases)
18. [Baseline Systems](#18-baseline-systems)
19. [Evaluation Protocol](#19-evaluation-protocol)
20. [Ablation Study Design](#20-ablation-study-design)
21. [Required Figures](#21-required-figures)
22. [Paper Writing Guidance](#22-paper-writing-guidance)
23. [Project File Structure](#23-project-file-structure)
24. [Implementation Sequence](#24-implementation-sequence)
25. [Known Risks and Mitigations](#25-known-risks-and-mitigations)

---

## 1. The Problem

Mars rovers average 144 metres per Martian solar day. Not because of mechanical limits — because every drive segment must be manually approved by operators on Earth examining satellite images that may be months old. The one-way communication delay is 4 to 21 minutes. A round-trip approval takes up to 42 minutes. The rover sits idle.

The requirement: a system that takes a satellite photograph of terrain the rover has never visited and produces a safe driving route automatically, before landing, with no human in the loop and no prior rover data from that site.

Every existing approach fails at least one of these requirements. Physics-only systems miss flat-but-dangerous terrain. Learning systems require human labels or prior rover traversal data and fail on unseen terrain distributions. Probabilistic fusion requires active rover operation. Graph-based planners use ground-level sensors, not orbital imagery. No existing system does all of this together.

---

## 2. The Core Argument

### Physics and Learning Have Complementary Failure Modes

Physics features (slope, roughness, discontinuity) fail on terrain that is physically dangerous but geometrically flat in 2D projection — sand traps, sediment-filled craters, loose surface over subsurface voids. The CNN, trained on physical elevation measurements, catches these through visual pattern recognition.

The CNN fails on terrain underrepresented in training DEMs, or where the visual appearance of a hazard differs across regions. The physics features, being pure mathematics on pixel gradients, catch these regardless.

A learned per-pixel fusion decides which signal to trust at each location. The result is robust to the failure mode of either individual component because the other compensates.

### Physics-Derived Supervision Eliminates Human Labelling

The CNN learns from slope and roughness values computed from real stereo elevation models — physical measurements, not human opinions. No annotation budget. No annotator bias. Scales automatically to new planetary bodies whenever orbital stereo imagery exists.

### Terrain Complexity Should Drive Graph Resolution

A fixed-resolution graph wastes capacity representing flat, safe terrain with the same node density as a complex crater field. Allocating more graph nodes to geometrically complex, high-risk terrain and fewer to flat terrain makes the graph information-density-aware. The GNN receives a richer representation exactly where it matters most — at the boundaries between safe and hazardous terrain where misclassification is most costly.

---

## 3. Six Novel Contributions

Each contribution is absent from all comparison papers. Each is supported by a specific experiment.

**Contribution 1 — MAE self-supervised pretraining on Mars orbital imagery**
Pretraining the CNN encoder on 17,298 unlabelled CTX Mars orbital tiles using a Masked Autoencoder before supervised training. Adapts the encoder to Mars orbital image statistics (single-channel, crater-dominated, fixed solar illumination direction) before any DEM label is introduced. No prior planetary navigation paper uses self-supervised pretraining on the actual operational orbital image distribution.

**Contribution 2 — DEM-derived physics labels for CNN supervision**
Training the CNN on automatically generated slope and roughness risk scores from real stereo DEMs. Zero human annotation. Scalable to any planetary body with orbital stereo imaging capability. No prior Mars navigation paper uses this training strategy.

**Contribution 3 — Terrain-complexity-adaptive graph resolution**
Dynamically allocating superpixel node density based on local terrain complexity as measured by the physics risk map: fewer nodes on flat terrain, progressively more nodes in geometrically complex and high-risk zones. Produces an information-density-aware graph where the GNN receives the highest representational capacity exactly at terrain boundaries and hazard zones. No prior planetary navigation or terrain GNN paper uses physics-driven adaptive graph resolution.

**Contribution 4 — Physics-similarity KNN graph edges**
Connecting superpixel nodes by combined spatial proximity and physics feature similarity instead of spatial adjacency alone. Ensures GNN message passing stays within geologically coherent terrain regions rather than crossing terrain type boundaries. Adapted from Rodrigues and Carbonera (ICEIS 2024) to a planetary terrain context with physics features replacing colour as the similarity metric.

**Contribution 5 — Physics-aware GATv2 with FFN diversity module**
Injecting terrain physics similarity directly into the GATv2 attention scoring function before softmax normalisation. Adding an FFN module after each GATv2 layer (adapted from Han et al., NeurIPS 2022) to prevent over-smoothing of risk signals as depth increases. The combined architecture has not appeared in any planetary navigation paper.

**Contribution 6 — Uncertainty-informed path planning with per-waypoint risk attribution**
Monte Carlo dropout produces an uncertainty map alongside the risk map. Uncertainty inflates edge costs in the path planner, providing conservative routing in regions where the model lacks training data. Each path waypoint is labelled with its dominant risk signal source (physics vs CNN) derived from the fusion α map. No prior planetary navigation system provides both uncertainty-qualified routing and per-waypoint attribution.

---

## 4. Related Work and Positioning

### Papers to Cite and How to Position Against Each

**TRG-Planner — Lee et al., IEEE RA-L 2025**
Risk-weighted graph path planning from LiDAR ground-level sensing. No orbital imagery, no CNN, no GNN, no uncertainty, no adaptive graph resolution.
Position: validates risk-graph approach. You extend it to orbital imagery with all missing capabilities.

**PIETRA — Yang et al., arXiv 2024**
Physics-informed traversability learning via physics-shaped loss functions. Evidential uncertainty. No graph, no GNN, no path output, no orbital input.
Position: shares physics-learning motivation and uncertainty. Your physics features are explicit inference-time signals, not just training-time loss shaping. Your graph component is entirely absent from PIETRA.

**Endo et al. — ICRA 2023, RA-L 2024**
Probabilistic fusion of terrain classifier and slip predictor for planetary rovers. Requires rover traversal data. No orbital imagery, no graph, no GNN.
Position: closest competitor in application domain. Pre-landing capability from orbital imagery alone is your key distinction.

**MarsMapNet — Zhao et al., IEEE TGRS 2024**
Superpixel-guided feature fusion for Martian landform mapping. Orbital input. Maps terrain but does not plan paths. Fixed superpixel resolution.
Position: validates superpixels for Mars orbital imagery. You extend to risk-weighted path planning with adaptive resolution.

**Rodrigues and Carbonera — ICEIS 2024**
Systematic evaluation of graph construction choices for image GCNs. KNN-Combined edges outperform region adjacency. Tested on Earth RGB images with colour similarity.
Position: directly motivates physics-similarity KNN edges. You validate the principle in planetary terrain using physics features as similarity metric.

**Han et al. Vision GNN — NeurIPS 2022**
FFN module after graph convolution prevents over-smoothing. Tested on ImageNet.
Position: motivates FFN addition to GATv2 for terrain risk propagation.

**He et al. MAE — CVPR 2022**
Masked Autoencoder self-supervised pretraining. Cite as the pretraining methodology basis.

**Brody et al. GATv2 — ICLR 2022**
Dynamic attention in graph attention networks. Cite as primary GNN architecture reference.

**Gal and Ghahramani — ICML 2016**
Monte Carlo dropout for uncertainty estimation. Cite as uncertainty methodology basis.

**Achanta et al. SLIC — IEEE TPAMI 2012**
Standard superpixel algorithm. Cite as implementation reference.

### Research Gap Table — Include in Related Work Section of Paper

| Capability | TRG-Planner | PIETRA | Endo et al. | MarsMapNet | **This work** |
|---|:---:|:---:|:---:|:---:|:---:|
| Orbital image input | No | No | No | Yes | **Yes** |
| Zero human annotation | Yes | No | No | No | **Yes** |
| MAE pretraining on target domain | No | No | No | No | **Yes** |
| DEM-supervised CNN | No | No | No | No | **Yes** |
| Adaptive graph resolution | No | No | No | No | **Yes** |
| Physics-similarity KNN edges | No | No | No | No | **Yes** |
| Physics-aware GNN attention | No | No | No | No | **Yes** |
| FFN diversity module | No | No | No | No | **Yes** |
| Uncertainty estimation | No | Yes | Partial | No | **Yes** |
| Per-waypoint risk attribution | No | No | No | No | **Yes** |
| Full path output | Yes | No | Yes | No | **Yes** |
| Pre-landing capable | No | No | No | Partial | **Yes** |

---

## 5. Datasets

Three datasets serve the system in completely non-overlapping roles. No dataset substitutes for another.

### 5.1 HiRISE Stereo DEMs — Training Labels

**What it is:** HiRISE (High Resolution Imaging Science Experiment) produces orbital images of Mars at 25 cm/pixel. Stereo pairs of HiRISE images are processed using photogrammetry to reconstruct 3D elevation — a Digital Elevation Model where each pixel encodes real measured height in metres.

**Source:** USGS Astrogeology Science Center at https://www.uahirise.org/dtm/ — freely downloadable. Each entry provides a DEM GeoTIFF and the corresponding HiRISE browse image GeoTIFF at the same geographic location.

**How many to download:** Minimum 10 DEM locations for a functional system. Recommended target is 20 to 30. Select from geologically diverse terrain: volcanic plains, canyon regions, ancient crater fields, polar layered deposits, fluvial valley networks. Do not download all DEMs from one region.

**Role:** Each DEM location provides one training pair. The HiRISE browse image is the CNN input. The DEM-derived slope and roughness risk map is the training label. No human labels anything.

**Format:** GeoTIFF with embedded coordinate reference system metadata. Requires GDAL and rasterio Python libraries.

**Expected yield:** 5,000 to 15,000 labelled 512×512 training tiles.

### 5.2 MurrayLab CTX Tiles — Pretraining and Demo

**What it is:** Context Camera (CTX) orbital images of Mars at ~6 m/pixel, preprocessed into 512×512 tiles by the MurrayLab group. No terrain labels.

**Source:** MurrayLab CTX dataset. Two tile sets, 8,649 tiles each, 17,298 total. PNG format, grayscale, exactly 512×512. No resize needed.

**Two roles:**
Role 1 — MAE pretraining corpus. 17,298 unlabelled Mars orbital tiles are the pretraining data for the encoder. This adapts the encoder to Mars orbital image statistics before any supervised label is introduced.
Role 2 — Qualitative demo. After training, run the full pipeline on 3 to 5 selected tiles and generate risk maps, uncertainty maps, α maps, and planned paths as figures.

**Tile selection for demo:** Choose tiles with visually diverse terrain — at least one smooth region and one rough or high-contrast region per tile. Reject tiles where more than 30% of pixels are within 5% of the tile minimum or maximum value (edge-of-mosaic padding artefacts).

### 5.3 HiRISE Map-Proj-v3 — Cross-Domain Evaluation

**What it is:** 73,031 image crops from 180 HiRISE browse images. Each crop is 227×227 pixels with one image-level class label. Compiled by Wagstaff et al. at NASA JPL.

**Source:** DOI 10.5281/zenodo.2538136. Contains image directory, label text file (one integer per line matching image order), classmap CSV.

**The eight real classes:** other, crater, dark dune, slope streak, bright dune, impact ejecta, swiss cheese, spider. The class edge_case does not exist in the actual dataset — do not reference it anywhere.

**Augmentation filtering:** Each of the 10,433 original crops was augmented 6 times producing 73,031 total. Use only the 10,433 originals for evaluation. Augmented crops are near-duplicates and inflate metrics. Identify originals by filename suffix or the fact that the dataset is ordered in blocks of seven (original then 6 augments). Write a verification script to confirm the ordering before assuming it.

**Input size:** Crops are 227×227. Pipeline expects 512×512. Resize using bilinear interpolation. Apply consistently. Document in paper.

**Risk remapping:**

| Class | Risk score | Rationale |
|---|---|---|
| other | 0.15 | Generic flat terrain |
| crater | 0.90 | Rim and interior: entrapment and slope hazard |
| dark dune | 0.85 | Fine sand: high slip and entrapment |
| slope streak | 0.80 | Mass movement indicator: unstable terrain |
| bright dune | 0.50 | Variable composition: moderate slip |
| impact ejecta | 0.55 | Scattered rock: passable but rough |
| swiss cheese | 0.85 | CO2 sublimation pits: structurally unsafe |
| spider | 0.45 | Dendritic erosion: rough but often passable |

**Role:** Zero-shot cross-domain evaluation. System trained on DEM tiles, evaluated on HiRISE v3 crops with no fine-tuning. Measures generalisation to a different image distribution and validates that physics features reduce the domain gap.

---

## 6. System Architecture Overview

```
PRETRAINING (runs once before everything)
Stage 0: MAE on 17,298 CTX tiles → Pretrained encoder weights

OPERATIONAL PIPELINE (per tile at inference)

Input: Orbital image tile (512×512, grayscale)
         │
    ┌────┴─────┐
    ▼          ▼
Stage 2      Stage 3
Physics      CNN Semantic
Features     Risk Estimator
S, R, D      MAE-pretrained
H_physics    MobileNetV3 +
[0,1]        DeepLabV3+
             H_learned [0,1]
    │          │
    └────┬─────┘
         ▼
      Stage 4
      Adaptive Fusion
      α map (per-pixel trust)
      H_final = α·H_learned + (1−α)·H_physics
         │
         ▼
      Stage 5                    ← KEY NOVELTY
      Adaptive-Resolution
      Superpixel Graph
      120–700+ nodes per tile
      Physics-KNN edges (K=5)
      14-dim node features
         │
         ▼
      Stage 6
      Physics-Aware GATv2
      + FFN Diversity Module
      p̂_i ∈ [0,1] per node
         │
         ▼
      Stage 7
      MC Dropout ×5
      Uncertainty map U(x,y)
         │
         ▼
      Stage 8
      A* + D* Path Planning
      Uncertainty-weighted costs
      Per-waypoint attribution

OUTPUTS: H_final + U(x,y) + Safe path T with attribution
```

---

## 7. Stage 0 — MAE Self-Supervised Pretraining

### Purpose

MobileNetV3 is ImageNet-pretrained — optimised for RGB Earth photographs of everyday objects. Mars orbital imagery is single-channel grayscale, dominated by crater morphology and regolith texture, illuminated from a fixed solar direction. Starting supervised training from ImageNet weights means the encoder begins in the wrong visual feature space.

MAE pretraining on CTX tiles adapts the encoder to Mars orbital imagery before any label is ever introduced. The encoder learns to reconstruct masked regions of Mars terrain images, developing internal representations of craters, slope-shading gradients, and surface roughness patterns — all without labels.

### How MAE Works

During each training forward pass, 75% of image patches are randomly masked (replaced with a learnable token). The encoder processes the visible 25%. A lightweight decoder reconstructs the full image from the encoder representation. Loss is mean squared pixel error on masked patches only. After pretraining, the decoder is discarded. Only the encoder weights are kept.

### Configuration

**Pretraining data:** All 17,298 CTX tiles. Single-channel, normalised per tile to [0,1]. No labels. No augmentation beyond standard MAE random masking.

**Patch size:** 16×16 pixels. For a 512×512 image this produces 1,024 patches. 75% masking means 768 patches masked per forward pass.

**Encoder:** MobileNetV3-Large adapted with a patch embedding stem.

**Decoder:** Lightweight 4-layer MLP reconstruction head. Discarded after pretraining.

**Training:** AdamW, learning rate 1.5×10⁻⁴, cosine annealing, weight decay 0.05, batch size 64, 200 epochs. Expected time: 8–12 hours on RTX 3060 Ti.

**Verification:** Reconstruction loss should decrease over 200 epochs. Inspect 5 reconstructed tiles qualitatively — decoder output should show terrain structure, not noise.

### Ablation Required

Compare three conditions: random initialisation, ImageNet pretrained weights, MAE pretrained weights (proposed). All other components identical. Report hazard recall and mIoU on the in-distribution DEM test set for each. This directly proves whether MAE pretraining adds measurable value.

---

## 8. Stage 1 — DEM Data Preparation and Label Generation

### DEM Acquisition

Download HiRISE stereo DEM pairs from USGS portal. For each entry: download the DEM GeoTIFF (pixel values = metres of elevation above MOLA reference ellipsoid) and the paired HiRISE browse image GeoTIFF. Target 20 to 30 locations across diverse geological regions.

### Geometric Alignment

Use GDAL's gdalwarp to reproject the DEM to exactly match the browse image's pixel grid — same projection, same origin, same pixel resolution, same dimensions. Verify alignment by overlaying the DEM slope map on the browse image: steep features in the DEM must align with shadow-casting features in the image. All datasets aligned to the MOLA reference frame as the common planetary coordinate system.

### Label Computation

From the aligned DEM, compute three quantities per pixel.

**Slope in degrees:** Finite differences of elevation in east-west and north-south directions, divided by DEM pixel resolution in metres. Slope = arctan(gradient magnitude). This is geometrically correct slope in degrees, not a pixel gradient.

**Roughness in metres:** Standard deviation of elevation within a 7×7 sliding window. Measures local height variation in real units.

**DEM risk score (training label):**
risk = 0.6 × clamp(slope_deg / 20.0, 0, 1) + 0.4 × (roughness / tile_max_roughness)
Clamped to [0.05, 0.95] for label smoothing (prevents sigmoid saturation during training).

20 degrees is the hard traversal limit for a Mars rover — cite from NASA published rover mechanical specifications.

**Hazard threshold for evaluation:** A pixel is labelled hazardous when slope exceeds 15 degrees OR roughness exceeds 0.6 × tile maximum.

### Tiling

Cut image-label pairs into 512×512 tiles with 256-pixel stride (50% overlap to reduce boundary artefacts in graph construction).

Reject tiles where: more than 10% of DEM pixels are NoData, or more than 30% of image pixels are near-saturated. Log all rejections.

### Splits — By DEM Location, Not By Tile

All tiles from one DEM location go into exactly one split. This prevents location-specific feature leakage between training and test.

Reserve one complete DEM location in a different geological region as the out-of-distribution (OOD) test set.

From remaining DEM locations: 70% training, 15% validation, 15% in-distribution test.

### Data Augmentation — Training Tiles Only

**Spatial (apply identically to image and label):** horizontal flip p=0.5, vertical flip p=0.5, rotation ±15° with reflect fill.

**Intensity (apply to image only — labels are physical measurements):** brightness ±20%, contrast ±20%, Gaussian noise σ ~ U(0, 0.02).

### MOLA Proxy Validation Experiment — Required

Download MOLA MEGDR elevation data for 20–30 locations overlapping HiRISE training tiles. Compute actual slope from MOLA elevation. Compute Sobel slope proxy from the HiRISE browse image. Aggregate both to MOLA's 460 m/pixel footprint by averaging. Compute Pearson r. Report as one number with one scatter plot. If r > 0.60: proxy is validated. If r < 0.50: report as a limitation and discuss implications.

---

## 9. Stage 2 — Physics Feature Engine

### Purpose

Compute three terrain geometry features directly from pixel intensity patterns. No training. No labels. No assumptions about terrain visual appearance. Mathematically identical on any planetary body. The domain-invariant safety net.

### Feature 1 — Slope Proxy (S)

Apply Sobel operators in horizontal and vertical directions. Compute gradient magnitude as sqrt(Gx² + Gy²). Normalise per tile to [0,1] with ε=1e-8.

**Basis:** Steep slopes in orbital imagery produce strong local brightness gradients from differential illumination and shadow casting. Flat terrain produces near-zero gradient magnitude.

**Failure mode:** Also responds to albedo contrast boundaries on flat terrain. Stage 3 corrects for this.

### Feature 2 — Roughness (R)

Sliding window standard deviation of pixel intensities. Window size: 7×7 pixels. Normalise per tile to [0,1] with ε=1e-8.

**Basis:** Rough terrain (boulders, fractured rock) produces high local intensity variance. Smooth terrain (compacted regolith, fine sand) produces low variance.

**Failure mode:** Fine-grained sand has low roughness despite being a traversal hazard. Stage 3 corrects for this.

### Feature 3 — Discontinuity Proxy (D)

Laplacian of Gaussian (LoG) with sigma=2.0. Take absolute value. Normalise per tile to [0,1] with ε=1e-8.

**Basis:** LoG responds to sharp intensity changes at the scale set by sigma. Crater rims, rock edges, and scarp margins produce strong responses. Sigma=2.0 captures hazard-scale features without responding to pixel noise.

### Combined Physics Risk Map

H_physics = w1 × S + w2 × R + w3 × D

Initial weights: w1=0.4, w2=0.3, w3=0.3. Tune via grid search on validation set (see Ablation Study). Report optimal weights in paper.

### Implementation Requirements

One PyTorch nn.Module. All operations batched. F.conv2d with reflect padding throughout. All under torch.no_grad() at inference. Target: under 5ms per tile on GPU, under 100ms on CPU.

### Role in Adaptive Graph Resolution

H_physics is the terrain complexity signal that drives adaptive node allocation in Stage 5. It is computed before graph construction and serves as both a direct risk estimate and a density-allocation map. This creates a clean directed dependency: physics features inform graph structure, not the other way around.

---

## 10. Stage 3 — CNN Semantic Risk Estimator

### Purpose

Learn to predict terrain physical danger from visual appearance, using DEM-derived risk scores as supervision. Captures visual hazard patterns that physics features cannot detect.

### Architecture

**Encoder:** MobileNetV3-Large. Initialise from MAE pretraining (Stage 0), not ImageNet. Fine-tune all layers. Retain stride-32 features for ASPP and stride-4 features for the skip connection.

**Decoder:** DeepLabV3+. ASPP module: atrous convolutions at rates 6, 12, 18 plus 1×1 convolution and global average pooling branch, all concatenated and projected to 256 channels. Upsample 4× and concatenate with stride-4 skip (projected to 48 channels). 3×3 convolution to 256 channels. Upsample to 512×512.

**Output head:** 1×1 convolution 256→1. Sigmoid activation. Output is H_learned ∈ [0,1]^{512×512}.

**Input:** (B, 3, 512, 512) — single grayscale channel replicated to 3 channels. Replication required because MobileNetV3 expects 3-channel input.

**Parameters:** ~11.7 million.

### Training Target

DEM-derived risk score per pixel: 0.6 × clamp(slope_deg/20, 0, 1) + 0.4 × normalised_roughness, clamped to [0.05, 0.95]. Pixels where DEM was NoData are excluded via a validity mask — excluded from loss computation using ignore masking.

### Loss Function

Weighted BCE: hazardous pixels (target > 0.7) get weight 3.0. Safe/uncertain pixels get weight 1.0.
Dice: applied to hazardous region (target > 0.7). Coefficient 0.5.
Total Variation: spatial smoothness regulariser. Coefficient 0.1.

Full: L = L_BCE_weighted + 0.5·L_Dice + 0.1·L_TV

### Training Configuration

| Parameter | Value |
|---|---|
| Initialisation | MAE pretrained encoder from Stage 0 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Learning rate | 1e-4, cosine annealing |
| Batch size | 8 |
| Max epochs | 60 |
| Early stopping | Patience 10, monitor val_hazard_recall |
| Checkpoint | Best val_hazard_recall |

---

## 11. Stage 4 — Spatial Adaptive Fusion

### Purpose

Combine H_physics and H_learned into H_final using a learned per-pixel trust map α(x,y). The fusion learns which signal to trust more at each specific location in the image based on local terrain context.

### Architecture

Input: three channels concatenated — H_physics, H_learned, and the original grayscale image. Shape (B, 3, 512, 512). The original image provides local context that neither risk map alone encodes.

Layer 1: Conv(3→16, 3×3) + ReLU + reflect padding.
Layer 2: Conv(16→8, 3×3) + ReLU + reflect padding.
Layer 3: Conv(8→1, 1×1) + Sigmoid.

Output: α(x,y) ∈ [0,1]^{512×512}. Values near 1 = trust CNN. Values near 0 = trust physics.

Fusion: H_final(x,y) = α(x,y) × H_learned(x,y) + (1−α(x,y)) × H_physics(x,y)

Total parameters: ~12,000. Intentionally small to prevent overfitting.

### Training

Two-phase training is mandatory.

Phase 1: Train CNN (Stage 3) to convergence. Save checkpoint.
Phase 2: Freeze all CNN weights. Load Phase 1 checkpoint. Train fusion network only. Loss applied to H_final against DEM-derived labels using the same compound loss as Stage 3.

Set joint_with_cnn=false in configuration. Training jointly without freezing produces degenerate α maps where the CNN learns to be intentionally incorrect because the fusion will compensate.

### Role in Adaptive Graph Resolution

After Stage 4, H_final is the terrain risk estimate used to compute terrain complexity scores for adaptive node allocation in Stage 5. H_final is also directly embedded in node features. The fusion's output therefore serves two roles: direct risk estimation and density-allocation signal.

### Diagnostic Check

If the α map shows no spatial structure (near-uniform values across the tile), fusion training has degenerated. Debug before proceeding to Stage 5. Expected appearance: low-α clusters on slope-dominated terrain, high-α clusters on visually distinctive crater rims and boulder fields.

---

## 12. Stage 5 — Adaptive-Resolution Superpixel Graph

### Purpose and Novelty

This stage converts the 512×512 H_final map into a graph. The novel contribution is that the number of graph nodes per tile is not fixed — it is dynamically determined by local terrain complexity, which is measured by H_physics. Hazardous, geometrically complex terrain receives more nodes. Flat, safe terrain receives fewer. The GNN then receives a representation where node density is highest exactly where terrain classification is hardest — at the boundaries between safe and hazardous terrain and within high-risk zones.

No prior planetary terrain navigation paper uses physics-driven adaptive graph resolution. This contribution is positioned against MarsMapNet (which uses fixed superpixels) and all prior GNN-based terrain systems.

### Step 1 — Compute Terrain Complexity Map

Divide the 512×512 tile into non-overlapping 32×32 blocks (producing a 16×16 grid of 256 blocks total). For each block, compute the mean of H_physics across all pixels in that block. This is the block's terrain complexity score.

### Step 2 — Assign Node Budget Per Block

Map each block's complexity score to a node budget using three tiers.

**Flat tier** (complexity score < 0.25): 5 nodes per block. This allocates approximately 5 × N_flat nodes where N_flat is the number of flat blocks. These regions are geologically simple — more nodes add no information.

**Complex tier** (complexity score 0.25 to 0.60): 15 nodes per block. Moderate terrain complexity — slopes, rough surfaces, transitional zones.

**Hazard tier** (complexity score > 0.60): between 30 and 50 nodes per block, scaled linearly from 30 at score=0.60 to 50 at score=1.0. These are the regions where the GNN's contextual reasoning adds the most value — crater rims, boulder fields, steep slopes.

**Expected total node count across tile:** A tile of mostly flat terrain with a small crater region produces approximately 120 to 200 nodes total. A tile of complex, hazardous terrain produces 450 to 700+ nodes. The average across the training set is expected to be 300 to 350 nodes, similar to the fixed-baseline.

### Step 3 — Adaptive SLIC Segmentation

SLIC does not natively support spatially varying node density. Use a hierarchical splitting approach.

First pass: Run SLIC on the entire tile with a conservative target of n_segments = max(80, floor(total_budget × 0.4)). This produces a coarse baseline segmentation.

Second pass — Refinement: For each block assigned to the hazard tier, identify all SLIC superpixels whose centroid falls within that block. For each such superpixel, if its pixel count exceeds 200 pixels (indicating it is too coarse for a hazard zone), run SLIC again on just that superpixel's pixel set with n_segments proportional to the superpixel's share of the block's node budget. Merge the refined sub-superpixels back into the global label map.

This hierarchical approach achieves spatially adaptive density without a custom SLIC implementation. It uses only standard scikit-image SLIC calls.

**Connectivity guarantee:** After construction, verify the resulting superpixel map is fully connected (every pixel belongs to exactly one superpixel, no isolated pixel groups). Run scikit-image's label cleaning if needed.

### Step 4 — Node Feature Vector — 14 Dimensions

Every superpixel node carries a 14-dimensional feature vector. This vector is identical in structure across all nodes regardless of superpixel size. The GNN sees consistent input dimensionality.

| Index | Feature | Source | Notes |
|---|---|---|---|
| 0 | Centroid x (normalised 0–1) | SLIC | Spatial position |
| 1 | Centroid y (normalised 0–1) | SLIC | Spatial position |
| 2 | Mean slope S | Stage 2 | Physics slope signal |
| 3 | Mean roughness R | Stage 2 | Physics roughness signal |
| 4 | Mean discontinuity D | Stage 2 | Physics edge signal |
| 5 | Mean H_physics | Stage 2 | Combined physics risk |
| 6 | Mean H_learned | Stage 3 | CNN risk signal |
| 7 | Mean H_final | Stage 4 | Fused risk |
| 8 | Mean α | Stage 4 | Local physics-CNN trust |
| 9 | Node area (normalised 0–1) | SLIC | **Critical for adaptive graphs — encodes node size so GNN can account for size disparity** |
| 10 | Mean pixel intensity | Image | Albedo proxy |
| 11 | Std pixel intensity | Image | Texture variance |
| 12 | Hazardous flag (H_final > 0.7) | Stage 4 | Pre-computed indicator |
| 13 | Hazardous neighbour count | Graph | Neighbourhood hazard clustering |

**Note on Dimension 9:** In a fixed-resolution graph, node area is a minor feature. In the adaptive-resolution graph, it is critical. A 600-pixel node (flat terrain) carries averaged information from six times as many pixels as a 100-pixel node (hazard zone). The GNN must learn to weight contributions accordingly. Including normalised area explicitly in the feature vector allows the attention mechanism to account for node size disparity. Without it, the GNN has no way to distinguish a large low-density node from a small high-density node of similar risk score.

### Step 5 — Physics-Similarity KNN Edge Construction

For each superpixel node, connect to its K=5 nearest neighbours in combined spatial and physics distance space.

**Spatial distance:** Euclidean distance between centroid coordinates, normalised by tile diagonal.

**Physics distance:** Euclidean distance between the physics sub-vector [S, R, D, H_physics], normalised by tile maximum pairwise physics distance.

**Combined distance:** 0.5 × spatial_distance + 0.5 × physics_distance.

**KNN:** For each node, find 5 nearest neighbours by combined distance. Add edges in both directions (undirected).

**Connectivity guarantee:** After KNN construction, check graph is fully connected. If disconnected components exist, add minimum RAG edges (pixels sharing a boundary) between the disconnected components to restore a single connected component. Log frequency of bridging. If bridging occurs in more than 20% of tiles, increase K from 5 to 7.

**Why not RAG:** A crater rim node should not exchange risk messages with an adjacent flat regolith node just because they share a pixel boundary. Physics-KNN connects nodes to geologically similar nodes, keeping message passing within terrain-coherent regions.

**Edge size disparity note:** Near the boundary between a low-density flat zone (large nodes) and a high-density hazard zone (small nodes), physics-KNN edges may connect geometrically large and small nodes. The edge weight formula accounts for this by using centroid distance normalised by tile diagonal — large nodes that are far apart receive appropriately large distance penalties.

### Step 6 — Edge Weights

w(i,j) = 0.6 × avg(H_final_i, H_final_j) + 0.25 × norm_centroid_distance(i,j) + 0.15 × |S_i − S_j|

The slope discontinuity term penalises edges crossing slope gradient changes — traversing a terrain type transition is more dangerous than the absolute risk of either endpoint.

### Step 7 — PyG Data Object

Package each tile's graph as a PyTorch Geometric Data object.

x: node features, shape (num_nodes, 14), float32.
edge_index: COO format adjacency, shape (2, num_edges), int64.
edge_attr: edge weights, shape (num_edges, 1), float32.
pos: centroid pixel coordinates, shape (num_nodes, 2), float32.
y: DEM-derived risk label per node (mean DEM risk across superpixel pixels), shape (num_nodes,), float32. This is the GNN training target.
tier: tier assignment per node (0=flat, 1=complex, 2=hazard), shape (num_nodes,), int64. Used for tier-stratified evaluation.
pixel_membership: map from pixel coordinates to node index, shape (512, 512), int64. Used for projecting node predictions back to pixel space.

---

## 13. Stage 6 — Physics-Aware GATv2 with FFN Module

### Purpose

Refine per-node risk estimates using neighbourhood context. A node that appears moderately risky in isolation but is surrounded by confirmed hazardous nodes should be upgraded. The GATv2 learns these corrections through attention-weighted message passing where physics similarity directly influences which neighbours are attended to most strongly.

### Why Standard GATv2 Is Insufficient Alone

Standard GATv2 computes attention from learned node embeddings only. Two geologically similar nodes (both flat regolith, different textures) may receive low attention between them based on embedding distance alone. Physics-aware attention injects a prior: nodes that are physically similar should attend to each other more strongly, independently of embedding distance. The GNN can learn to suppress this prior if data justifies it, but it provides useful structure from the start of training.

### Physics-Aware Attention Formula

For nodes i and j, the attention logit before softmax normalisation:

e_ij = LeakyReLU(aᵀ [W h_i || W h_j]) + λ × exp(−|S_i − S_j| − |R_i − R_j|)

Where W is a learnable weight matrix, a is a learnable attention vector, h_i and h_j are node embeddings, S and R are the node slope and roughness features from the feature vector. λ is a learnable scalar initialised to 0.1.

The exp(−physics_distance) term is 1.0 when nodes are physically identical and approaches 0 as physics features diverge. Adding this term before softmax — not after — means softmax normalisation still applies correctly and attention weights still sum to 1.

### FFN Diversity Module

After each GATv2Conv layer, apply an FFN module to each node independently.

Structure: BatchNorm1d on input → Linear(D → 4D) → GELU → Dropout(0.1) → Linear(4D → D) → add residual connection.

The FFN applies a learned per-node transformation that is not constrained to be an average of neighbour values. This prevents over-smoothing — the progressive convergence of all node features toward the neighbourhood mean, which destroys the risk gradient between hazardous and safe nodes that the path planner needs.

### Full Architecture

Layer 1: Physics-aware GATv2Conv (in=14, out=32, heads=4, concat=True → 128-dim) + ELU + Dropout(0.3) + FFN(D=128, hidden=512).

Layer 2: Physics-aware GATv2Conv (in=128, out=32, heads=4, concat=False → 32-dim) + ELU + Dropout(0.2) + FFN(D=32, hidden=128).

Output head: Linear(32→1) + Sigmoid → p̂_i ∈ [0,1] per node.

### Why Regression on DEM Labels, Not on H_final

If the target is H_final (already stored at node feature index 7), the GNN is learning to smooth its own input. It can achieve good loss by simply outputting a weighted average of visible neighbour H_final values — adding no genuine reasoning. By targeting the external DEM-derived risk label, the GNN must use neighbourhood context meaningfully: nodes adjacent to high-slope regions should have elevated risk even if their own slope is moderate, because the surrounding terrain geometry constrains escape options.

### Training Configuration

| Parameter | Value |
|---|---|
| Target | DEM-derived risk score per node (y field in PyG Data object) |
| Loss | SmoothL1 (Huber) — robust to outlier nodes at DEM boundary gaps |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 5e-4 |
| Max epochs | 100 |
| Early stopping | Patience 15, monitor val_MAE |
| Checkpoint | Best val_MAE |
| Batch size | 32 precomputed graphs |

### Precomputed Graph Strategy

Building graphs on the fly: ~0.8s per tile (SLIC + adaptive density + 14 features + KNN edges). Over 100 epochs × 10,000 tiles = approximately 222 hours. Not feasible.

Precompute all graphs once after Stage 4 completes. Store as serialised PyG Data .pt files. Load from disk during GNN training: ~0.01s per graph. 100 epochs over 10,000 tiles takes approximately 2.8 hours.

Critical: if fusion model is retrained, regenerate all precomputed graphs. They contain H_final and α values baked in as node features.

Storage estimate: variable-node-count graphs average ~320 nodes. Approximately 1.2 GB for 10,000 graphs.

---

## 14. Stage 7 — Uncertainty Estimation

### Purpose

Produce an epistemic uncertainty map U(x,y) expressing where the model lacks confidence. High uncertainty triggers conservative routing even when the point risk estimate is moderate.

### Monte Carlo Dropout

Enable dropout layers at inference time. Run the full forward pass (Stages 2 through 6) N=5 times with different dropout mask realisations. Collect five H_final maps and five sets of p̂_i node scores.

**Node uncertainty:** variance of p̂_i across 5 forward passes.

**Pixel uncertainty map U(x,y):** project node-level uncertainty to pixel space using the pixel_membership map from Stage 5. Each pixel receives the uncertainty of its superpixel node.

**Implementation requirement:** set all Dropout layers to training mode for the N MC passes. Return to eval mode for the final deterministic forward pass used in path planning.

**Inference time impact:** N=5 passes approximately quintuples the CNN and GNN inference time. If total inference exceeds 5 seconds on CPU, reduce to N=3. Report the timing breakdown by stage.

**Note on adaptive resolution and uncertainty:** Nodes in the hazard tier (high-density zones) will generally show lower uncertainty than flat-tier nodes, because hazardous terrain is well-represented in DEM training data. If flat-tier nodes show unexpectedly high uncertainty, it may indicate the training DEM set underrepresents that terrain type. This pattern is itself a diagnostic finding worth reporting.

---

## 15. Stage 8 — Path Planning

### A* with Uncertainty-Weighted Costs

A* search on the NetworkX graph representation of the superpixel graph. Finds minimum-cost path between start and goal nodes.

**Edge cost:**
C(i,j) = exp(3 × risk_ij) × [0.6 × risk_ij + 0.25 × dist_ij + 0.15 × |S_i − S_j|]

where risk_ij = average of p̂_i and p̂_j. The exponential multiplier grows from 1.0 at zero risk to ~20 at maximum risk. This is soft obstacle avoidance — high-risk terrain is expensive to route through but never fully blocked.

**Uncertainty penalty:** If node i has uncertainty U_i > 0.3, multiply all incident edge costs by (1 + 2.0 × U_i). High-uncertainty terrain is routed around conservatively even when the point estimate of risk is moderate.

**No hard node deactivation:** Never remove nodes from the graph. Hard deactivation caused 40% path failures in preliminary work by creating barriers with no route around them. Soft cost scaling achieves safer routing with ~100% path success rate.

**Heuristic:** h(n) = euclidean_distance(n, goal) × (1 + 0.4 × p̂_n + 0.1 × S_n). Slightly inadmissible in theory; guides search toward safer corridors in practice.

### D* for Dynamic Replanning

D* (Dynamic A*) maintains an incremental search structure that updates efficiently when edge costs change. Enables real-time path updates if onboard sensors detect new hazards during traversal.

Implement as a secondary planner. Compute initial path with A*. If new hazard is detected during traversal, update affected edge costs and call D* to replan from current rover position. D* reuses most of the prior search computation.

For pre-landing planning (the primary thesis claim), D* is not required. For the active-traversal extension, D* provides real-time capability.

### Adaptive Graph Resolution and Path Planning

The path planner benefits directly from adaptive resolution. In hazard zones, fine-grained nodes provide precise routing — the planner can find narrow safe corridors between hazard patches that would be invisible to coarse fixed-resolution nodes. In flat terrain, coarse nodes reduce path length ratio by avoiding unnecessary waypoint zigzagging.

### Path Output

**H_final:** 512×512 continuous risk map.

**U(x,y):** 512×512 uncertainty map.

**Trajectory T:** Ordered sequence of node centroid coordinates from start to goal. Each waypoint carries: coordinates, p̂_i, U_i, tier assignment (flat/complex/hazard), and dominant risk signal (physics if α_i < 0.5, CNN if α_i > 0.5).

Per-waypoint dominant risk attribution is a unique output of this system. No prior planetary navigation paper labels each routing decision with its physical or semantic origin. This is directly useful to mission operators who need to understand why a particular segment was chosen.

---

## 16. System Outputs

**Output 1 — Risk Map H_final:** 512×512 continuous hazard field [0,1]. Fused physics and CNN risk estimates.

**Output 2 — Uncertainty Map U(x,y):** 512×512 epistemic uncertainty field [0,1]. MC dropout variance. High values indicate low model confidence.

**Output 3 — Safe Trajectory T:** Ordered waypoints from start to goal. Each waypoint includes coordinates, GNN risk score, uncertainty score, terrain tier (flat/complex/hazard), and dominant risk signal attribution (physics or CNN).

---

## 17. Training Pipeline — Four Phases

Execute in order. Each phase depends on outputs from the previous phase.

### Phase 0 — MAE Pretraining

Input: 17,298 CTX tiles, no labels.
Output: Pretrained encoder checkpoint.
Duration: ~8–12 hours on RTX 3060 Ti.
Verify: Reconstruction loss decreases over 200 epochs. Qualitative inspection of 5 reconstructed tiles shows recognisable terrain structure.

### Phase 1 — CNN Supervised Training

Input: DEM training tiles with DEM-derived labels. MAE encoder weights from Phase 0.
Output: Trained RiskModel checkpoint.
Duration: Up to 60 epochs, ~6–10 hours.
Verify: Validation hazard recall > 0.70 by epoch 30. If not, check label generation pipeline for errors.

### Phase 2 — Fusion Training

Input: DEM training tiles. Frozen Phase 1 CNN checkpoint. Physics feature extractor.
Output: Trained AdaptiveFusion checkpoint.
Duration: Up to 40 epochs, ~3–5 hours.
Verify: H_final validation hazard recall must exceed H_learned recall alone. If H_final recall is lower, debug Phase 2 training before continuing.

### Phase 3a — Graph Precomputation

Input: All tiles (train/val/test/OOD). Frozen Phase 1 and Phase 2 checkpoints.
Process: Run Stages 2 through 5 on every tile. Save PyG Data objects as .pt files.
Duration: ~3 hours for 15,000 tiles.
Verify: Graph statistics per tile — node count in expected range, all 14 features within value bounds, no disconnected graphs.

### Phase 3b — GNN Training

Input: Precomputed .pt graph files.
Output: Trained GATv2+FFN checkpoint.
Duration: ~3–4 hours for 100 epochs.
Verify: Validation MAE decreases consistently. If MAE plateaus above 0.15 after 60 epochs, check that node labels are DEM-derived (not H_final).

---

## 18. Baseline Systems

Ten systems are evaluated. Each differs from the proposed system in exactly one way, isolating each component's contribution independently.

| ID | Name | Pretrain | Graph nodes | Edges | GNN | Uncertainty |
|---|---|---|---|---|---|---|
| B1 | Euclidean A* | — | — | — | None | None |
| B2 | Physics-only | — | Adaptive | Physics-KNN | None | None |
| B3 | CNN-ImageNet | ImageNet | Adaptive | Physics-KNN | None | None |
| B4 | CNN-MAE | MAE | Adaptive | Physics-KNN | None | None |
| B5 | Static fusion | MAE | Adaptive | Physics-KNN | None | None |
| B6 | No-GNN | MAE | Adaptive | Physics-KNN | None | None |
| B7 | Fixed-300 | MAE | Fixed 300 | Physics-KNN | GATv2+FFN | None |
| B8 | RAG-GNN | MAE | Adaptive | RAG | GATv2+FFN | None |
| B9 | No-uncertainty | MAE | Adaptive | Physics-KNN | GATv2+FFN | None |
| **Proposed** | **PA-GNN** | **MAE** | **Adaptive** | **Physics-KNN** | **GATv2+FFN** | **MC Dropout** |

**What each pairwise comparison proves:**

B1→B2: physics features over pure distance routing.
B2→B3: CNN over physics alone.
B3→B4: MAE pretraining over ImageNet initialisation.
B4→B5: adaptive fusion over CNN alone.
B5→B6: GNN over no-GNN on fused graph.
B6→B7: adaptive node count over fixed 300 nodes (the key new ablation).
B7→B8: physics-KNN edges over RAG, GNN architecture held constant.
B8→B9: uncertainty estimation contribution to path quality.
B9→Proposed: combined full system.

**B7 (Fixed-300) is required** to isolate the contribution of adaptive graph resolution. Without it, a reviewer cannot attribute any improvement to node count adaptation versus the GNN or edge construction changes.

---

## 19. Evaluation Protocol

### Three Evaluation Contexts — Never Average Across Them

**In-distribution DEM test:** Tiles from 15% of held-in DEM locations not used in training or validation. Standard benchmark.

**Out-of-distribution DEM test:** All tiles from the completely withheld DEM location in a different geological region. Tests generalisation to unseen terrain. Most important evaluation for the domain gap claim.

**HiRISE v3 cross-domain:** All 1,096 original HiRISE v3 crops. Zero-shot transfer — no fine-tuning. Image-level risk prediction versus remapped landmark class labels.

### Statistical Requirements — Non-Negotiable for RA-L

Run all experiments with 3 random seeds. Report mean ± standard deviation for every metric. Compute 95% bootstrap confidence intervals for HCR comparisons between proposed system and best baseline.

### Segmentation and Risk Metrics

**Hazard recall:** fraction of ground-truth hazardous pixels correctly predicted hazardous. Primary safety metric. Report separately for H_physics, H_learned, H_final.

**Mean IoU:** spatial overlap of predicted and true hazard regions.

**Expected Calibration Error (ECE):** 10-bin calibration of risk scores. Lower is better.

### Adaptive Resolution Specific Metrics

**Tier-stratified HCR:** Report HCR separately for waypoints that pass through flat-tier nodes, complex-tier nodes, and hazard-tier nodes. This shows where the path planning improvement is concentrated.

**Node count distribution:** Report mean, std, min, max node count across the test set. Demonstrates the actual range of adaptation achieved.

**Hazard recall by tier:** Report CNN and GNN hazard recall separately for nodes in each tier. Expected finding: hazard-tier nodes (most nodes, highest complexity) show the largest improvement from adaptive resolution versus fixed-300.

### GNN Metrics

**AUC-ROC:** binary hazard classification using p̂_i.
**MAE:** regression error against DEM-derived node labels.
**GNN delta:** HCR_no-GNN minus HCR_proposed.

### Path Planning Metrics

| Metric | Definition | Target |
|---|---|---|
| Hazard Crossing Rate (HCR) | Fraction of waypoints in DEM-labelled hazardous terrain | < 5% |
| Path Length Ratio (PLR) | Path length / straight-line start-goal distance | < 1.35 |
| Success Rate | Fraction of tiles where A* finds a path | > 95% |
| Inference Time (CPU) | End-to-end wall clock time | < 5 seconds |

### Domain Gap Analysis — The Central Result

Three rows: physics-only (B2), CNN-MAE-only (B4), proposed PA-GNN.
Three columns: in-distribution hazard recall, OOD hazard recall, domain gap (in minus out).

Expected pattern: physics shows minimal gap (domain-invariant by construction). CNN shows larger gap. Hybrid PA-GNN shows gap smaller than CNN-only. This is the paper's headline experimental claim.

---

## 20. Ablation Study Design

### Adaptive Resolution Ablation — Most Important New Ablation

Compare B7 (fixed 300 nodes) against proposed (adaptive nodes) with all other components identical. Report: HCR, PLR, hazard recall by tier, node count statistics, inference time. This is the direct validation of Contribution 3.

Additionally vary the tier thresholds. Current: flat <0.25, complex 0.25–0.60, hazard >0.60. Test alternative split at <0.20/0.20–0.55/>0.55 and <0.30/0.30–0.65/>0.65. Show that the result is robust to threshold choice — this demonstrates that the contribution is not a brittle consequence of one specific parameter.

### GNN Architecture Ablation — 2×2

Crossing edge type (RAG vs Physics-KNN) with architecture (GATv2-only vs GATv2+FFN). Four conditions, isolating each contribution independently.

### MAE Pretraining Ablation

Random init vs ImageNet vs MAE. All other components identical. Report hazard recall and mIoU.

### Physics Weight Sensitivity

Grid search over w1, w2, w3. Report validation hazard recall for each. Show robustness range.

### K-Neighbour Sensitivity

Vary K from 3 to 9. Report HCR on validation set. Validate K=5 choice.

---

## 21. Required Figures

### Figure 1 — Physics Feature Grid

5 columns (original, S, R, D, H_physics) × 3–5 rows (different terrain types). Caption explains each feature and identifies where each fails.

### Figure 2 — Adaptive Graph Resolution Illustration

This is the key new figure for Contribution 3.

Layout: 3 tiles as columns, 3 rows.
Row 1: original orbital image.
Row 2: H_physics terrain complexity map with tier boundaries overlaid (colour-coded: blue=flat, yellow=complex, red=hazard).
Row 3: resulting superpixel graph with nodes colour-coded by tier. Node size in the visualisation proportional to superpixel area.

Caption must explicitly state the node count for each tile and identify how node density correlates with terrain complexity. This is the figure that communicates Contribution 3 immediately.

### Figure 3 — H_physics vs H_learned vs H_final vs α

6 columns (original, DEM ground truth, H_physics, H_learned, α, H_final) × 3–5 rows. At least one row demonstrating H_physics catching a hazard H_learned misses. At least one row demonstrating the reverse.

### Figure 4 — MAE Pretraining Evidence

Two panels: reconstruction loss curve over 200 epochs; 3 example tiles showing masked input (25% visible patches), reconstructed output, and original.

### Figure 5 — Before and After GATv2 Refinement

Node-coloured graph plots for 3 tiles. Left column: H_final values (pre-GNN). Right column: p̂_i values (post-GNN). Annotate nodes where the GNN made corrections.

### Figure 6 — Uncertainty Map

Side-by-side for 3 tiles: H_final risk map and U(x,y) uncertainty map. Identify high-uncertainty regions and describe their terrain type.

### Figure 7 — Path Comparison

3 tiles, 3 paths each (B1 Euclidean, B4 CNN-only, proposed PA-GNN). Waypoints colour-coded by GNN risk score. Caption identifies where proposed path successfully avoids hazards the baselines cross.

### Figure 8 — Main Results Table (All 10 Baselines)

All 10 baselines as rows. HCR ± CI, PLR ± std, success rate, inference time as columns. Bold best value per column.

### Figure 9 — Domain Gap Table

3 rows × 3 columns as described in Section 19. Domain gap column immediately readable.

---

## 22. Paper Writing Guidance

### Abstract (250 words)

Open with the pre-landing navigation problem and the no-annotation requirement. State the system in one sentence listing all six components. State three key results: domain gap reduction, HCR improvement over best baseline, 100% success rate. Close with the scalability implication: applicable to any planetary body with orbital stereo imaging capability.

### Introduction

Required paragraph on zero-annotation claim (include this explicitly):
"The proposed system requires no human annotation at any stage of training. Terrain risk labels are generated automatically from real stereo digital elevation models — physical height measurements produced by photogrammetric processing of orbital image pairs. The CNN encoder is additionally pretrained on unlabelled orbital imagery using a Masked Autoencoder, adapting it to the target visual domain without requiring any labelled examples. The system therefore scales automatically to any planetary body where an orbiter with stereo imaging capability exists, without annotation effort or prior rover mission data."

Required paragraph on adaptive resolution (include this explicitly):
"A key architectural decision distinguishes this system from prior graph-based terrain planners: the number of graph nodes is not fixed. The system dynamically allocates superpixel node density based on local terrain complexity as measured by physics features — assigning as few as 120 nodes to flat, geologically simple terrain and as many as 700+ to complex, hazardous terrain within the same tile. This makes the graph information-density-aware: the GNN receives its highest representational capacity precisely where terrain classification is hardest and where misclassification would be most dangerous."

List all six contributions explicitly. Number them.

### Methodology — Critical Notes

For Stage 5 (adaptive resolution): state explicitly that this is adapted from and extends prior work on content-adaptive superpixels to a physically-motivated density allocation scheme. No prior work uses physics-derived complexity scores to drive planetary terrain graph resolution.

For Stage 5 (KNN edges): explicitly cite Rodrigues and Carbonera (ICEIS 2024). State: "We adapt the finding that restricting GNN graph edges to similar regions improves classification performance, applying this principle to planetary terrain graphs using physics features as the similarity metric."

For Stage 6 (FFN module): explicitly cite Han et al. (NeurIPS 2022) for the FFN module and Brody et al. (ICLR 2022) for GATv2. Explain that physics similarity is injected before softmax to preserve normalisation.

For Stage 7 (uncertainty): cite Gal and Ghahramani (ICML 2016) for MC dropout. Distinguish epistemic (model ignorance) from aleatoric (terrain ambiguity) uncertainty. State aleatoric estimation is future work.

### Limitations Section — Required Content

State explicitly: DEM coverage covers a small fraction of Mars. The MOLA proxy validation is not comprehensive. Uncertainty estimation captures epistemic uncertainty only. Cross-planet generalisation is assumed but not experimentally validated. The adaptive resolution thresholds are validated within the tested range but may require retuning for different planetary bodies or imaging systems.

---

## 23. Project File Structure

```
pa-gnn/
│
├── configs/                    # All YAML configuration files
│   ├── base.yaml               # Paths, seeds, device, image size
│   ├── physics.yaml            # S/R/D feature params and weights
│   ├── mae.yaml                # MAE pretraining config
│   ├── cnn.yaml                # CNN architecture and training
│   ├── fusion.yaml             # Fusion architecture, joint_with_cnn=false
│   ├── gnn.yaml                # GATv2 layers, physics lambda, FFN, K
│   └── datasets/
│       ├── dem.yaml            # DEM paths, tiling, split config
│       ├── ctx.yaml            # CTX tile paths, saturation threshold
│       └── hirise_v3.yaml      # HiRISE v3 paths, class-to-risk remap
│
├── data/
│   ├── raw/                    # Downloaded files — never modified
│   │   ├── dem/                # HiRISE DEM GeoTIFF files
│   │   ├── hirise_browse/      # Browse images paired with DEMs
│   │   ├── ctx/                # MurrayLab CTX tile directories
│   │   └── hirise_v3/          # HiRISE v3 crops + label txt + classmap csv
│   ├── processed/
│   │   ├── dem_tiles/          # 512×512 image-label pairs (train/val/test_in/test_ood)
│   │   ├── graphs/             # Precomputed PyG .pt files (train/val/test_in/test_ood)
│   │   ├── ctx_pretrain/       # CTX tiles prepared for MAE pretraining
│   │   └── ctx_demo/           # 3–5 selected CTX tiles for qualitative demo
│   └── splits/                 # train.txt, val.txt, test_in.txt, test_ood.txt
│
├── src/
│   ├── data/                   # Dataset classes and preprocessing
│   │   ├── dem_loader.py       # DEM tile PyTorch Dataset
│   │   ├── ctx_loader.py       # CTX loader (MAE pretraining and demo)
│   │   ├── hirise_loader.py    # HiRISE v3 evaluation loader
│   │   ├── graph_dataset.py    # PyG precomputed graph loader
│   │   ├── dem_processing.py   # GDAL co-registration + slope/roughness computation
│   │   ├── label_generation.py # DEM elevation → risk score pipeline
│   │   ├── tiling.py           # Tile generation with quality filtering
│   │   ├── normalize.py        # Per-tile min-max normalisation
│   │   ├── augmentations.py    # Joint spatial + image-only intensity augmentation
│   │   └── label_remap.py      # DEM risk and HiRISE landmark class remapping
│   │
│   ├── models/                 # All neural network modules
│   │   ├── mae.py              # MAE encoder + reconstruction decoder
│   │   ├── encoder.py          # MobileNetV3-Large wrapper (stride-4 and stride-32)
│   │   ├── decoder.py          # DeepLabV3+ with ASPP and skip connections
│   │   ├── risk_model.py       # Full CNN assembly (encoder + decoder + sigmoid head)
│   │   ├── fusion.py           # AdaptiveFusion 3-layer CNN + EndToEndFusionModel
│   │   ├── gatv2_physics.py    # Physics-aware GATv2Conv (custom attention scoring)
│   │   ├── ffn_module.py       # FFN diversity module with residual connection
│   │   └── gnn_model.py        # Full GATv2+FFN assembly + output head
│   │
│   ├── physics/                # Physics feature computation
│   │   ├── slope.py            # Sobel gradient magnitude
│   │   ├── roughness.py        # Sliding window standard deviation
│   │   ├── discontinuity.py    # Laplacian of Gaussian
│   │   └── combine.py          # Weighted combination to H_physics
│   │
│   ├── graph/                  # Graph construction
│   │   ├── adaptive_slic.py    # Terrain-complexity-adaptive SLIC segmentation
│   │   ├── node_features.py    # 14-dimensional feature extraction per superpixel
│   │   ├── edges.py            # Physics-KNN + RAG connectivity guarantee
│   │   └── graph_builder.py    # Full image → PyG Data orchestrator
│   │
│   ├── planning/               # Path planning
│   │   ├── astar.py            # A* with soft cost scaling + uncertainty penalty
│   │   ├── dstar.py            # D* for dynamic replanning
│   │   └── heuristics.py       # Physics-aware heuristic function
│   │
│   ├── uncertainty/
│   │   └── mc_dropout.py       # MC dropout: N passes + variance computation
│   │
│   ├── training/
│   │   ├── losses.py           # Weighted BCE + Dice + Total Variation
│   │   └── trainer.py          # Generic training loop + early stopping + checkpointing
│   │
│   ├── evaluation/
│   │   ├── metrics.py          # HCR, PLR, recall, IoU, ECE, AUC, MAE
│   │   ├── evaluate_dem.py     # In-distribution and OOD evaluation
│   │   ├── evaluate_hirise.py  # Cross-domain evaluation
│   │   └── demo_ctx.py         # Qualitative pipeline demo on CTX tiles
│   │
│   ├── pipeline.py             # Full 9-stage inference pipeline with per-stage timing
│   ├── visualization.py        # Risk maps, uncertainty maps, graphs, path overlays
│   └── utils.py                # Config loader, seed, logger, file I/O helpers
│
├── scripts/                    # CLI entry points — one action each
│   ├── download_dems.py        # Batch DEM download from USGS portal
│   ├── process_dems.py         # Co-registration + label generation for all DEMs
│   ├── tile_dataset.py         # Tiling + quality filtering + split assignment
│   ├── validate_dataset.py     # Integrity checks before training
│   ├── train_mae.py            # Stage 0 MAE pretraining
│   ├── train_cnn.py            # Stage 3 supervised CNN training
│   ├── train_fusion.py         # Stage 4 fusion training (CNN frozen)
│   ├── precompute_graphs.py    # One-time adaptive graph precomputation
│   ├── train_gnn.py            # Stage 6 GNN training on precomputed graphs
│   ├── mola_validation.py      # MOLA Pearson correlation experiment
│   ├── run_ablations.py        # All ablation conditions automated
│   ├── evaluate_all.py         # All three evaluation protocols
│   └── run_inference.py        # Single-image CLI inference with timing
│
├── checkpoints/
│   ├── mae_encoder.pth         # MAE pretrained encoder
│   ├── cnn_best.pth            # Trained RiskModel
│   ├── fusion_best.pth         # Trained AdaptiveFusion
│   └── gnn_best.pth            # Trained GATv2+FFN
│
└── results/
    ├── tables/                 # CSV results files
    ├── figures/                # Publication-ready figures (300 DPI)
    └── logs/                   # Training logs + TensorBoard directories
```

---

## 24. Implementation Sequence

### Week 1 — Environment and Data

Days 1–2: Create project structure. Install all dependencies. Write environment verification script (all imports + GPU check). Write dataset path verification script.

Days 3–5: Download 10+ HiRISE DEM pairs from USGS (diverse geological regions). Download MurrayLab CTX tile sets. Download HiRISE v3 dataset. Download MOLA data for 20–30 locations overlapping HiRISE tiles.

Days 6–7: Implement DEM co-registration with GDAL. Verify alignment for 3 DEMs visually. Implement slope, roughness, and risk label computation from DEM. Verify against expected physical properties.

### Week 2 — Data Pipeline

Days 8–10: Implement adaptive tiling with quality filters. Assign tiles to splits by DEM location (not by tile). Run dataset validation script: tile counts, label value ranges, no image-label mismatch, no split overlap.

Days 11–14: Implement all three dataset classes (DEM loader, CTX loader, HiRISE v3 loader). Implement per-tile normalisation and joint augmentation. Verify spatial correspondence is preserved after augmentation on 10 pairs.

### Week 3 — Physics Features and MAE

Days 15–17: Implement PhysicsFeatureExtractor (all operations batched PyTorch). Verify speed targets (< 5ms GPU, < 100ms CPU). Generate Figure 1 for 5 tiles. Visually confirm features respond as expected.

Days 18–21: Implement MAE encoder, reconstruction decoder, training loop. Launch MAE pretraining on CTX tiles. Training runs 200 epochs. At completion generate Figure 4 (reconstruction examples). Save encoder checkpoint.

### Week 4 — CNN Training

Days 22–25: Implement RiskModel (encoder + decoder + sigmoid head). Implement compound loss (weighted BCE + Dice + TV). Implement training loop with early stopping.

Days 26–28: Launch CNN training with MAE pretrained encoder. Monitor validation hazard recall. Save best checkpoint. If recall < 0.70 by epoch 30, debug label pipeline before continuing.

### Week 5 — Fusion and Validation Experiments

Days 29–31: Implement AdaptiveFusion and EndToEndFusionModel. Launch fusion training with CNN frozen. Verify H_final recall exceeds H_learned recall. Save fusion checkpoint.

Days 32–35: Run MOLA correlation experiment. Generate scatter plot. Run MAE pretraining ablation (3 initialisation conditions). Generate preliminary Figures 2, 3 for 5 tiles.

### Week 6 — Adaptive Graph Construction

Days 36–40: Implement terrain complexity scoring (block-level H_physics means). Implement tier assignment (3 tiers with configurable thresholds). Implement adaptive SLIC (hierarchical two-pass segmentation). Implement 14-dim node feature extraction. Implement physics-KNN edge construction with RAG connectivity guarantee. Implement PyG Data packaging.

Days 41–42: Run graph precomputation for all tiles. Verify graph statistics: node count range (120–700+), feature value ranges, no disconnected graphs. Generate Figure 2 (adaptive resolution illustration) for 3 tiles.

### Week 7 — GNN Training

Days 43–46: Implement physics-aware GATv2Conv (custom attention with physics term, verified normalisation). Implement FFN module with residual connection. Assemble GNN model with output head.

Days 47–49: Launch GNN training on precomputed graphs. Monitor validation MAE. Run 2×2 GNN ablation (edge type × FFN). Generate Figure 5 (before/after GNN) for 3 tiles.

### Week 8 — Uncertainty and Planning

Days 50–52: Implement MC dropout inference (N=5 passes, variance computation, pixel projection). Generate Figure 6 (uncertainty maps) for 3 tiles.

Days 53–56: Implement A* with uncertainty-weighted soft cost scaling and physics-aware heuristic. Implement D* for dynamic replanning. Implement per-waypoint dominant risk attribution. Implement pipeline timing instrumentation.

### Week 9 — Full Evaluation

Days 57–63: Run all 10 baseline systems on all 3 evaluation datasets. Run all evaluations with 3 seeds. Compute mean ± std for all metrics. Compute 95% bootstrap CI for HCR comparisons. Run adaptive resolution ablation (vary tier thresholds). Generate Figures 7, 8, 9. Compile all results.

### Weeks 10–12 — Paper Writing

Write in order: abstract, introduction, related work, methodology, experiments, conclusion, limitations. Generate all figures at 300 DPI. Verify all numerical claims match computed results. Submit to IEEE RA-L.

---

## 25. Known Risks and Mitigations

### Risk 1 — Fewer than 10 Usable DEMs

USGS portal yields fewer valid DEM pairs than expected.

Mitigation: Supplement with MOLA co-registered with CTX. MOLA provides coarser but global elevation labels. Treat as a lower-quality secondary training tier with reduced loss weight. Document mixing ratio in paper.

### Risk 2 — MOLA Correlation Below 0.50

Sobel proxy does not correlate well with actual slope.

Mitigation: Report honestly. Adjust Stage 2 claim to "signal correlated with terrain geometry" rather than slope measurement. Stage 3 is unaffected — it is trained on actual DEM measurements.

### Risk 3 — MAE Pretraining Shows No Improvement

The MAE ablation shows no measurable hazard recall improvement over ImageNet initialisation.

Mitigation: Report as a negative result. Keep ImageNet init as default. This is still publishable as an empirical finding about self-supervised pretraining transferability from CTX to DEM-supervised training.

### Risk 4 — Adaptive Graph Produces Disconnected Components

High terrain heterogeneity causes K=5 KNN to produce disconnected components.

Mitigation: The RAG connectivity guarantee described in Stage 5 resolves this. If bridging is required for more than 20% of tiles, increase K from 5 to 7. Log bridging frequency and report.

### Risk 5 — GNN Validation MAE Does Not Converge

MAE plateaus above 0.15 after 100 epochs.

Mitigation: Verify node labels are DEM-derived (not H_final). If confirmed correct, try increasing K to allow more neighbourhood context. Add a third GATv2+FFN layer. If over-smoothing persists, verify FFN residual connections are correctly implemented.

### Risk 6 — Path Success Rate Below 95%

A* fails to find paths for more than 5% of tiles.

Mitigation: Verify no hard node deactivation is applied. Confirm soft cost scaling is the only obstacle representation. If graph disconnection causes failures, verify the RAG connectivity guarantee runs after every graph construction call.

### Risk 7 — Inference Time Exceeds 5 Seconds on CPU

N=5 MC dropout passes approximately quintuple CNN inference time.

Mitigation: Reduce to N=3 passes. Report timing breakdown by stage. Note that the 5-second target applies to worst-case (high-complexity, 700+ node) tiles — average tiles will be significantly faster with adaptive resolution since most tiles have 200–350 nodes.

### Risk 8 — Adaptive Resolution Thresholds Are Brittle

Small changes to tier thresholds produce large changes in HCR.

Mitigation: The ablation in Section 20 tests three threshold configurations. If results vary dramatically across configurations, document this as a hyperparameter sensitivity and provide the threshold selection procedure as part of the method. This is an honest finding, not a flaw.

---

*Document version: 4.0 — Final merged blueprint with adaptive graph resolution*
*Novelty contributions: 6 (MAE pretraining, DEM supervision, adaptive resolution, physics-KNN edges, physics-aware GATv2+FFN, uncertainty attribution)*
*Pipeline: PA-GNN — Physics-Aware Graph Neural Terrain Intelligence System*
*Target: IEEE Robotics and Automation Letters*
*Authors: Syed Abir Hossain, Ashik Mahmud, Mahadir Rahaman — AUST CSE-4733*
*Supervisor: Tamanna Tabassum, Assistant Professor, Dept. of CSE, AUST*
