"""
pipeline.py
-----------
Full 9-stage PA-GNN inference pipeline with per-stage timing.

Blueprint §6 (System Architecture Overview) / §16 (System Outputs):

  Pretraining (once, offline):
    Stage 0 — MAE → pretrained encoder weights

  Operational pipeline (per tile at inference):
    Stage 2 — Physics Feature Engine   → H_physics, S, R, D
    Stage 3 — CNN Risk Estimator       → H_learned
    Stage 4 — Adaptive Fusion          → H_final, α map
    Stage 5 — Adaptive-Resolution SLIC → superpixel graph (PyG Data)
    Stage 6 — Physics-Aware GATv2+FFN  → p̂_i per node
    Stage 7 — MC Dropout               → uncertainty map U(x,y)
    Stage 8 — A* + D* Path Planning    → trajectory T with attribution

  Outputs:
    H_final       — 512×512 continuous risk map [0,1]
    U(x,y)        — 512×512 epistemic uncertainty map [0,1]
    Trajectory T  — ordered waypoints with coordinates, risk, uncertainty,
                    tier, and dominant risk signal attribution (physics / CNN)

Usage:
    from src.pipeline import PipelineConfig, PAGNNPipeline

    cfg     = PipelineConfig.from_checkpoints("checkpoints/")
    pipeline = PAGNNPipeline(cfg)

    result  = pipeline.run(image_tensor, start=(50, 50), goal=(460, 460))
    # result.h_final, result.uncertainty, result.path_waypoints, result.timings
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """All outputs of one PA-GNN inference call.

    Blueprint §16 — System Outputs.
    """
    # Risk maps
    h_physics:  np.ndarray | None = None   # (512, 512) physics risk
    h_learned:  np.ndarray | None = None   # (512, 512) CNN risk
    h_final:    np.ndarray | None = None   # (512, 512) fused risk
    alpha:      np.ndarray | None = None   # (512, 512) α trust map

    # Feature maps
    slope:      np.ndarray | None = None   # (512, 512) S
    roughness:  np.ndarray | None = None   # (512, 512) R
    discont:    np.ndarray | None = None   # (512, 512) D

    # Uncertainty
    uncertainty: np.ndarray | None = None  # (512, 512) U(x,y)

    # GNN node outputs
    gnn_node_preds:  np.ndarray | None = None   # (N,) p̂_i
    gnn_node_labels: np.ndarray | None = None   # (N,) y (if available)
    graph_data:      Any = None                  # PyG Data object

    # Path planning
    path_waypoints:      list[dict] = field(default_factory=list)
    path_waypoint_nodes: list[dict] = field(default_factory=list)
    path_found:          bool = False

    # Timing breakdown (seconds per stage)
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable summary (no large arrays)."""
        return {
            "path_found":      self.path_found,
            "n_waypoints":     len(self.path_waypoints),
            "n_nodes":         int(self.gnn_node_preds.shape[0]) if self.gnn_node_preds is not None else 0,
            "timings":         self.timings,
            "total_time_s":    sum(self.timings.values()),
        }


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Checkpoint paths and hyperparameters for inference.

    Blueprint defaults are encoded here so the pipeline works without a YAML.
    """
    # Checkpoint paths
    cnn_checkpoint:    str = "checkpoints/cnn_best.pt"
    fusion_checkpoint: str = "checkpoints/fusion_best.pt"
    gnn_checkpoint:    str = "checkpoints/gnn_best.pt"

    # Physics weights (blueprint §9 defaults; overridden by ablation)
    physics_w1: float = 0.4
    physics_w2: float = 0.3
    physics_w3: float = 0.3

    # MC Dropout (blueprint §14)
    mc_passes: int = 5

    # Graph construction (blueprint §12)
    graph_k_neighbours: int = 5
    graph_flat_thresh:   float = 0.25
    graph_hazard_thresh: float = 0.60

    # GNN model (blueprint §13)
    gnn_in_features:   int = 14
    gnn_hidden_dim:    int = 32
    gnn_heads:         int = 4
    gnn_lambda_init:   float = 0.1

    # Path planning (blueprint §15)
    hazard_threshold:     float = 0.7
    uncertainty_penalty:  float = 2.0
    uncertainty_thresh:   float = 0.3

    # Performance
    device: str = "auto"

    @classmethod
    def from_checkpoints(cls, checkpoints_dir: str | Path, **kwargs) -> "PipelineConfig":
        """Convenience constructor from a checkpoints directory."""
        d = Path(checkpoints_dir)
        return cls(
            cnn_checkpoint=str(d / "cnn_best.pt"),
            fusion_checkpoint=str(d / "fusion_best.pt"),
            gnn_checkpoint=str(d / "gnn_best.pt"),
            **kwargs,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "PipelineConfig":
        """Load config from a YAML file."""
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Model loader helpers
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _load_fusion_model(cfg: PipelineConfig, device: torch.device) -> nn.Module:
    """Load EndToEndFusionModel (Stages 2+3+4) from checkpoints."""
    from src.models.fusion import build_fusion_model

    model = build_fusion_model(
        cnn_checkpoint=cfg.cnn_checkpoint,
        freeze_cnn=True,
        physics_w1=cfg.physics_w1,
        physics_w2=cfg.physics_w2,
        physics_w3=cfg.physics_w3,
    )

    # Load fusion-specific weights
    fusion_path = Path(cfg.fusion_checkpoint)
    if fusion_path.exists():
        ckpt = torch.load(str(fusion_path), map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt.get("fusion_state_dict", ckpt))
        # Load only fusion sub-module weights if wrapped
        if all(k.startswith("fusion.") for k in state if not k.startswith("cnn.")):
            model.fusion.load_state_dict(
                {k.removeprefix("fusion."): v for k, v in state.items()
                 if k.startswith("fusion.")}
            )
        else:
            try:
                model.fusion.load_state_dict(state)
            except Exception:
                model.load_state_dict(state, strict=False)
        log.info("Fusion weights loaded from %s", fusion_path.name)
    else:
        log.warning("Fusion checkpoint not found: %s — using untrained fusion", fusion_path)

    return model.eval().to(device)


def _load_gnn_model(cfg: PipelineConfig, device: torch.device) -> nn.Module:
    """Load PhysicsAwareGNN from checkpoint."""
    from src.models.gnn_model import PhysicsAwareGNN

    model = PhysicsAwareGNN(
        in_features=cfg.gnn_in_features,
        hidden_dim=cfg.gnn_hidden_dim,
        heads=cfg.gnn_heads,
        physics_lambda_init=cfg.gnn_lambda_init,
    )

    gnn_path = Path(cfg.gnn_checkpoint)
    if gnn_path.exists():
        ckpt = torch.load(str(gnn_path), map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(state, strict=True)
        log.info("GNN weights loaded from %s (epoch %s)", gnn_path.name,
                 ckpt.get("epoch", "?"))
    else:
        log.warning("GNN checkpoint not found: %s — using random weights", gnn_path)

    return model.eval().to(device)


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class PAGNNPipeline:
    """Full PA-GNN inference pipeline.

    Stages executed per tile:
      2 → Physics features
      3 → CNN risk map (H_learned)
      4 → Adaptive fusion (H_final, α)
      5 → Adaptive-resolution superpixel graph
      6 → GATv2+FFN node risk prediction
      7 → MC Dropout uncertainty
      8 → A* path planning with uncertainty-weighted costs

    Parameters
    ----------
    cfg    : PipelineConfig — checkpoint paths and hyperparameters
    device : torch.device or None — defaults to cfg.device
    """

    def __init__(
        self,
        cfg: PipelineConfig | None = None,
        device: torch.device | None = None,
    ):
        self.cfg = cfg or PipelineConfig()
        self.device = device or _resolve_device(self.cfg.device)

        log.info("Initialising PA-GNN Pipeline on %s...", self.device)

        # Load models
        self.fusion_model = _load_fusion_model(self.cfg, self.device)
        self.gnn_model    = _load_gnn_model(self.cfg, self.device)

        # MC Dropout estimator
        from src.uncertainty.mc_dropout import MCDropoutEstimator
        self.mc_estimator = MCDropoutEstimator(
            model=self.gnn_model,
            n_passes=self.cfg.mc_passes,
            device=self.device,
        )

        # Graph builder (lazy import to avoid torch-geometric dependency at import time)
        self._graph_builder = None

        log.info("PA-GNN Pipeline ready.")

    def _get_graph_builder(self):
        if self._graph_builder is None:
            from src.graph.graph_builder import GraphBuilder
            self._graph_builder = GraphBuilder(
                k_neighbours=self.cfg.graph_k_neighbours,
                flat_thresh=self.cfg.graph_flat_thresh,
                hazard_thresh=self.cfg.graph_hazard_thresh,
            )
        return self._graph_builder

    # -----------------------------------------------------------------------
    # Per-stage methods (exposed for testing and ablation)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def run_stages_234(
        self, image: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Stages 2, 3, 4 — physics + CNN + fusion.

        Parameters
        ----------
        image : (1, 3, 512, 512) float32 on self.device

        Returns
        -------
        dict with h_final, h_learned, h_physics, alpha, features
        """
        return self.fusion_model(image)

    def run_stage5(
        self,
        image: torch.Tensor,
        stage234_outputs: dict[str, torch.Tensor],
    ):
        """Stage 5 — build adaptive superpixel graph.

        Returns a PyG Data object.
        """
        builder = self._get_graph_builder()

        # Convert to numpy on CPU for graph construction
        image_np = image[0, 0].cpu().float().numpy()   # (H, W) grayscale
        h_physics_np = stage234_outputs["h_physics"][0, 0].cpu().float().numpy()
        h_learned_np = stage234_outputs["h_learned"][0, 0].cpu().float().numpy()
        h_final_np   = stage234_outputs["h_final"][0, 0].cpu().float().numpy()
        alpha_np     = stage234_outputs["alpha"][0, 0].cpu().float().numpy()
        feats        = stage234_outputs.get("features", {})
        slope_np     = feats.get("slope", stage234_outputs["h_physics"])
        roughness_np = feats.get("roughness", stage234_outputs["h_physics"])
        disc_np      = feats.get("disc", stage234_outputs["h_physics"])

        if isinstance(slope_np, torch.Tensor):
            slope_np     = slope_np[0, 0].cpu().float().numpy()
            roughness_np = roughness_np[0, 0].cpu().float().numpy()
            disc_np      = disc_np[0, 0].cpu().float().numpy()

        graph_data = builder.build_graph(
            image=image_np,
            h_physics=h_physics_np,
            h_learned=h_learned_np,
            h_final=h_final_np,
            alpha=alpha_np,
            slope=slope_np,
            roughness=roughness_np,
            disc=disc_np,
        )
        return graph_data

    @torch.no_grad()
    def run_stage6(self, graph_data) -> np.ndarray:
        """Stage 6 — deterministic GNN forward pass.

        Returns
        -------
        (N,) numpy array of per-node risk scores p̂_i
        """
        self.gnn_model.eval()
        x          = graph_data.x.to(self.device)
        edge_index = graph_data.edge_index.to(self.device)
        preds = self.gnn_model(x, edge_index)
        return preds.cpu().numpy()

    def run_stage7(self, graph_data) -> dict:
        """Stage 7 — MC Dropout uncertainty estimation.

        Returns
        -------
        dict with risk_map (H,W), uncertainty_map (H,W),
        node_risk_mean (N,), node_risk_var (N,)
        """
        return self.mc_estimator.estimate_pixel_uncertainty(graph_data)

    def run_stage8(
        self,
        graph_data,
        gnn_preds: np.ndarray,
        uncertainty_map: np.ndarray | None,
        node_uncertainty: np.ndarray | None,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[dict]:
        """Stage 8 — A* path planning with per-waypoint attribution.

        Parameters
        ----------
        graph_data       : PyG Data with .pos, .x, pixel_membership
        gnn_preds        : (N,) GNN node risk scores
        uncertainty_map  : (H,W) pixel-space uncertainty (or None)
        node_uncertainty : (N,) per-node variance (or None)
        start, goal      : (row, col) pixel positions

        Returns
        -------
        list of waypoint dicts, each with:
            row, col, risk, uncertainty, tier, attribution, node_idx
        """
        try:
            from src.planning.astar import PhysicsAwareAStar
        except ImportError:
            log.warning("A* planner not available; returning empty path.")
            return []

        if not hasattr(graph_data, "pos") or graph_data.pos is None:
            log.warning("Graph has no .pos; skipping path planning.")
            return []

        pos_np = graph_data.pos.numpy()   # (N, 2) where pos[:, 0] is x (col), pos[:, 1] is y (row)
        n_nodes = pos_np.shape[0]

        # Map start/goal pixel coords (row, col) → nearest node indices
        def _nearest_node(r: int, c: int) -> int:
            dists = np.sqrt((pos_np[:, 0] - c) ** 2 + (pos_np[:, 1] - r) ** 2)
            return int(np.argmin(dists))

        start_node = _nearest_node(*start)
        goal_node  = _nearest_node(*goal)

        if start_node == goal_node:
            log.warning("Start and goal map to same node; trivial path.")
            return []

        # Run PhysicsAwareAStar
        planner = PhysicsAwareAStar(use_physics_heuristic=True)
        traj = planner.plan_from_data(
            data=graph_data,
            start=start_node,
            goal=goal_node,
            node_risks=gnn_preds,
            node_uncertainties=node_uncertainty,
            hazard_threshold=self.cfg.hazard_threshold,
        )

        if traj is None or not traj.success:
            return []

        # Build waypoint list with full attribution
        waypoints = []
        for wp in traj.waypoints:
            waypoints.append({
                "node_idx":    wp.node_id,
                "row":         wp.y,  # wp.y is the row coordinate
                "col":         wp.x,  # wp.x is the col coordinate
                "risk":        wp.risk,
                "uncertainty": wp.uncertainty,
                "tier":        wp.tier,
                "alpha":       wp.alpha,
                "attribution": "CNN" if wp.dominant_signal == "cnn" else "physics",
            })

        return waypoints

    # -----------------------------------------------------------------------
    # Full pipeline entry point
    # -----------------------------------------------------------------------

    def run(
        self,
        image: torch.Tensor,
        start: tuple[int, int] = (50, 50),
        goal: tuple[int, int] = (460, 460),
        run_path_planning: bool = True,
    ) -> PipelineResult:
        """Run the full inference pipeline on one tile.

        Parameters
        ----------
        image            : (1, 3, 512, 512) float32 in [0,1] on any device
        start            : (row, col) start position in pixel coords
        goal             : (row, col) goal position in pixel coords
        run_path_planning: if False, skip Stages 7 and 8 (faster ablations)

        Returns
        -------
        PipelineResult with all outputs and per-stage timing.
        """
        timings: dict[str, float] = {}
        image = image.to(self.device)

        result = PipelineResult()

        # ── Stages 2, 3, 4 ───────────────────────────────────────────────
        t0 = time.perf_counter()
        stage234 = self.run_stages_234(image)
        timings["stages_234"] = time.perf_counter() - t0

        # Extract numpy maps
        def _np(key):
            t = stage234.get(key)
            return t[0, 0].cpu().float().numpy() if t is not None else None

        result.h_physics = _np("h_physics")
        result.h_learned = _np("h_learned")
        result.h_final   = _np("h_final")
        result.alpha     = _np("alpha")
        feats = stage234.get("features", {})
        result.slope     = feats.get("slope",     stage234.get("h_physics"))
        result.roughness = feats.get("roughness", stage234.get("h_physics"))
        result.discont   = feats.get("disc",      stage234.get("h_physics"))
        for key in ("slope", "roughness", "discont"):
            attr = getattr(result, key)
            if isinstance(attr, torch.Tensor):
                setattr(result, key, attr[0, 0].cpu().float().numpy())

        # ── Stage 5 — Graph construction ─────────────────────────────────
        t0 = time.perf_counter()
        try:
            graph_data = self.run_stage5(image, stage234)
            result.graph_data = graph_data
        except Exception as exc:
            log.error("Stage 5 (graph construction) failed: %s", exc, exc_info=True)
            timings["stage_5"] = time.perf_counter() - t0
            result.timings = timings
            return result
        timings["stage_5"] = time.perf_counter() - t0

        # ── Stage 6 — GNN ────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            gnn_preds = self.run_stage6(graph_data)
            result.gnn_node_preds = gnn_preds
            if hasattr(graph_data, "y") and graph_data.y is not None:
                result.gnn_node_labels = graph_data.y.cpu().numpy()
        except Exception as exc:
            log.error("Stage 6 (GNN) failed: %s", exc, exc_info=True)
            timings["stage_6"] = time.perf_counter() - t0
            result.timings = timings
            return result
        timings["stage_6"] = time.perf_counter() - t0

        if not run_path_planning:
            result.timings = timings
            return result

        # ── Stage 7 — Uncertainty ─────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            unc_result = self.run_stage7(graph_data)
            result.uncertainty = unc_result.get("uncertainty_map")
            node_unc = unc_result.get("node_risk_var")
        except Exception as exc:
            log.warning("Stage 7 (MC Dropout) failed: %s", exc)
            node_unc = None
            result.uncertainty = None
        timings["stage_7"] = time.perf_counter() - t0

        # ── Stage 8 — Path planning ───────────────────────────────────────
        t0 = time.perf_counter()
        try:
            waypoints = self.run_stage8(
                graph_data=graph_data,
                gnn_preds=gnn_preds,
                uncertainty_map=result.uncertainty,
                node_uncertainty=node_unc,
                start=start,
                goal=goal,
            )
            result.path_waypoints = waypoints
            result.path_found = len(waypoints) > 0

            # Build per-waypoint node dicts for tier-stratified metrics
            result.path_waypoint_nodes = [
                {"tier": w["tier"], "risk": w["risk"],
                 "coords": (w["row"], w["col"]),
                 "attribution": w["attribution"]}
                for w in waypoints
            ]
        except Exception as exc:
            log.error("Stage 8 (path planning) failed: %s", exc, exc_info=True)
            result.path_found = False
        timings["stage_8"] = time.perf_counter() - t0

        result.timings = timings
        total = sum(timings.values())
        log.info(
            "Pipeline complete in %.2fs | "
            "stages_234=%.2fs | graph=%.2fs | gnn=%.2fs | unc=%.2fs | path=%.2fs | "
            "nodes=%d | path_found=%s",
            total,
            timings.get("stages_234", 0),
            timings.get("stage_5", 0),
            timings.get("stage_6", 0),
            timings.get("stage_7", 0),
            timings.get("stage_8", 0),
            int(gnn_preds.shape[0]),
            result.path_found,
        )

        return result

    def __call__(self, image: torch.Tensor, **kwargs) -> dict:
        """Make the pipeline callable (convenience for passing as pipeline_fn).

        Returns result as a dict for compatibility with evaluation modules.
        """
        result = self.run(image, **kwargs)
        return {
            "h_final":           torch.from_numpy(result.h_final) if result.h_final is not None else None,
            "h_learned":         torch.from_numpy(result.h_learned) if result.h_learned is not None else None,
            "h_physics":         torch.from_numpy(result.h_physics) if result.h_physics is not None else None,
            "alpha":             torch.from_numpy(result.alpha) if result.alpha is not None else None,
            "uncertainty":       result.uncertainty,
            "gnn_node_preds":    result.gnn_node_preds,
            "gnn_node_labels":   result.gnn_node_labels,
            "path_waypoints":    result.path_waypoints,
            "path_waypoint_nodes": result.path_waypoint_nodes,
            "path_found":        result.path_found,
            "timings":           result.timings,
            "slope":             result.slope,
            "roughness":         result.roughness,
            "discont":           result.discont,
        }
