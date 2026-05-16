"""
astar.py
--------
Stage 8 (blueprint §15) — A* path planner with uncertainty-weighted costs.

Edge cost:
  C(i,j) = exp(3 × risk_ij) × [0.6 × risk_ij + 0.25 × dist_ij + 0.15 × |S_i − S_j|]
  risk_ij = 0.5 × (p̂_i + p̂_j)

Uncertainty penalty:
  If U_i > 0.3, multiply all incident edge costs by (1 + 2.0 × U_i)

No hard node deactivation — soft cost scaling only.
"""

from __future__ import annotations

import heapq
import math
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

try:
    import networkx as nx
except ImportError:
    nx = None

from src.planning.heuristics import physics_aware_heuristic, euclidean_heuristic

log = logging.getLogger(__name__)


@dataclass
class Waypoint:
    """Single waypoint along the planned path."""
    node_id: int
    x: float
    y: float
    risk: float
    uncertainty: float
    tier: int
    dominant_signal: str
    alpha: float


@dataclass
class Trajectory:
    """Complete planned path from start to goal."""
    waypoints: list[Waypoint]
    total_cost: float
    path_length: float
    straight_line_dist: float
    path_length_ratio: float
    hazard_crossing_rate: float
    success: bool


def build_networkx_graph(
    data: Any,
    node_risks: np.ndarray | None = None,
    node_uncertainties: np.ndarray | None = None,
) -> tuple[Any, dict[int, dict[str, Any]]]:
    """Convert PyG Data to NetworkX DiGraph with edge costs.

    Parameters
    ----------
    data               : PyG Data with .x, .edge_index, .pos
    node_risks         : (N,) GNN risk predictions (optional override)
    node_uncertainties : (N,) MC dropout variances (optional)

    Returns
    -------
    G         : NetworkX DiGraph with 'cost' edge attributes
    node_data : dict mapping node_id → feature dict
    """
    if nx is None:
        raise ImportError("networkx is required for path planning")

    N = data.x.size(0) if isinstance(data.x, torch.Tensor) else len(data.x)
    x_np = data.x.numpy() if isinstance(data.x, torch.Tensor) else np.array(data.x)
    pos_np = data.pos.numpy() if isinstance(data.pos, torch.Tensor) else np.array(data.pos)

    slopes = x_np[:, 2]
    h_final = x_np[:, 7]
    alpha_vals = x_np[:, 8]

    risks = np.asarray(node_risks, dtype=np.float32) if node_risks is not None else h_final
    uncertainties = np.asarray(node_uncertainties, dtype=np.float32) if node_uncertainties is not None else np.zeros(N, dtype=np.float32)

    tiers = np.zeros(N, dtype=np.int64)
    if hasattr(data, "tier") and data.tier is not None:
        tiers = data.tier.numpy() if isinstance(data.tier, torch.Tensor) else np.array(data.tier)

    node_data: dict[int, dict[str, Any]] = {}
    for i in range(N):
        node_data[i] = {
            "pos": (float(pos_np[i, 0]), float(pos_np[i, 1])),
            "risk": float(risks[i]),
            "slope": float(slopes[i]),
            "alpha": float(alpha_vals[i]),
            "uncertainty": float(uncertainties[i]),
            "tier": int(tiers[i]),
        }

    # Tile diagonal for distance normalisation
    tile_diag = max(math.sqrt(
        (pos_np[:, 0].max() - pos_np[:, 0].min()) ** 2 +
        (pos_np[:, 1].max() - pos_np[:, 1].min()) ** 2
    ), 1e-6)

    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    edge_index = data.edge_index.numpy() if isinstance(data.edge_index, torch.Tensor) else np.array(data.edge_index)

    for k in range(edge_index.shape[1]):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])

        risk_ij = 0.5 * (risks[i] + risks[j])
        dx = pos_np[i, 0] - pos_np[j, 0]
        dy = pos_np[i, 1] - pos_np[j, 1]
        dist_ij = math.sqrt(dx * dx + dy * dy) / tile_diag
        slope_diff = abs(slopes[i] - slopes[j])

        cost = math.exp(3.0 * risk_ij) * (
            0.6 * risk_ij + 0.25 * dist_ij + 0.15 * slope_diff
        )

        if uncertainties[i] > 0.3:
            cost *= (1.0 + 2.0 * uncertainties[i])

        G.add_edge(i, j, cost=max(cost, 1e-10))

    return G, node_data


class PhysicsAwareAStar:
    """A* planner with physics-aware cost and uncertainty penalty.

    Parameters
    ----------
    use_physics_heuristic : bool — use physics-aware heuristic (default: True)
    risk_weight           : float — heuristic risk coefficient
    slope_weight          : float — heuristic slope coefficient
    heuristic_weight      : float — WA* weight factor (default: 1.5)
    """

    def __init__(self, use_physics_heuristic: bool = True,
                 risk_weight: float = 0.4, slope_weight: float = 0.1,
                 heuristic_weight: float = 1.5):
        self.use_physics_heuristic = use_physics_heuristic
        self.risk_weight = risk_weight
        self.slope_weight = slope_weight
        self.heuristic_weight = heuristic_weight

    def plan(self, G, node_data, start: int, goal: int,
             hazard_threshold: float = 0.7) -> Trajectory | None:
        """Run A* search from start to goal on NetworkX graph."""
        if start == goal:
            wp = self._make_waypoint(start, node_data)
            return Trajectory([wp], 0.0, 0.0, 0.0, 1.0,
                              1.0 if wp.risk > hazard_threshold else 0.0, True)

        if self.use_physics_heuristic:
            h_func = lambda n: physics_aware_heuristic(
                n, goal, node_data, self.risk_weight, self.slope_weight, self.heuristic_weight)
        else:
            h_func = lambda n: euclidean_heuristic(n, goal, node_data)

        counter = 0
        open_set = [(h_func(start), counter, start)]
        came_from: dict[int, int] = {}
        g_score: dict[int, float] = {start: 0.0}
        closed: set[int] = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct(came_from, current,
                                         g_score[current], node_data,
                                         hazard_threshold)
            if current in closed:
                continue
            closed.add(current)

            for neighbour in G.successors(current):
                if neighbour in closed:
                    continue
                tentative_g = g_score[current] + G[current][neighbour]["cost"]
                if tentative_g < g_score.get(neighbour, float("inf")):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g
                    counter += 1
                    heapq.heappush(open_set, (tentative_g + h_func(neighbour),
                                              counter, neighbour))

        log.warning("A* failed: no path from %d to %d", start, goal)
        return None

    def plan_from_data(self, data, start: int, goal: int,
                       node_risks=None, node_uncertainties=None,
                       hazard_threshold: float = 0.7) -> Trajectory | None:
        """Build graph from PyG Data and run A*."""
        G, node_data = build_networkx_graph(data, node_risks, node_uncertainties)
        return self.plan(G, node_data, start, goal, hazard_threshold)

    def _make_waypoint(self, nid: int, nd_map: dict) -> Waypoint:
        nd = nd_map[nid]
        alpha = nd.get("alpha", 0.5)
        return Waypoint(
            node_id=nid, x=nd["pos"][0], y=nd["pos"][1],
            risk=nd["risk"], uncertainty=nd.get("uncertainty", 0.0),
            tier=nd.get("tier", 0),
            dominant_signal="physics" if alpha < 0.5 else "cnn",
            alpha=alpha,
        )

    def _reconstruct(self, came_from, current, total_cost,
                     node_data, hazard_threshold) -> Trajectory:
        path_ids = [current]
        while current in came_from:
            current = came_from[current]
            path_ids.append(current)
        path_ids.reverse()

        waypoints = [self._make_waypoint(nid, node_data) for nid in path_ids]

        path_length = sum(
            math.sqrt((waypoints[i].x - waypoints[i-1].x)**2 +
                       (waypoints[i].y - waypoints[i-1].y)**2)
            for i in range(1, len(waypoints))
        )

        dx = waypoints[-1].x - waypoints[0].x
        dy = waypoints[-1].y - waypoints[0].y
        straight = math.sqrt(dx*dx + dy*dy)
        plr = path_length / max(straight, 1e-6)
        n_haz = sum(1 for wp in waypoints if wp.risk > hazard_threshold)
        hcr = n_haz / max(len(waypoints), 1)

        return Trajectory(waypoints, total_cost, path_length, straight,
                          plr, hcr, True)


def select_start_goal(data, strategy: str = "corners") -> tuple[int, int]:
    """Select start/goal nodes. 'corners' = top-left → bottom-right."""
    pos = data.pos.numpy() if isinstance(data.pos, torch.Tensor) else np.array(data.pos)
    N = pos.shape[0]

    if strategy == "corners":
        dist_tl = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        start = int(np.argmin(dist_tl))
        dist_br = np.sqrt((pos[:, 0] - pos[:, 0].max())**2 +
                          (pos[:, 1] - pos[:, 1].max())**2)
        goal = int(np.argmin(dist_br))
        if start == goal:
            goal = int(np.argmax(dist_tl))
    elif strategy == "random":
        idx = np.random.default_rng().choice(N, 2, replace=False)
        start, goal = int(idx[0]), int(idx[1])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return start, goal
