"""
heuristics.py
-------------
Stage 8 (blueprint §15) — Physics-aware heuristic for A* search.

  h(n) = euclidean_distance(n, goal) × (1 + 0.4 × p̂_n + 0.1 × S_n)

Slightly inadmissible — guides search toward safer corridors in practice.
The risk and slope terms inflate the distance estimate for high-risk nodes,
biasing the search away from hazardous terrain early in expansion.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def physics_aware_heuristic(
    node_id: int,
    goal_id: int,
    node_data: dict[int, dict[str, Any]],
    risk_weight: float = 0.4,
    slope_weight: float = 0.1,
) -> float:
    """Compute physics-aware A* heuristic from node to goal.

    Parameters
    ----------
    node_id      : int — current node index
    goal_id      : int — goal node index
    node_data    : dict mapping node_id → dict with keys:
                   'pos' (x,y), 'risk' (p̂_i), 'slope' (S_i)
    risk_weight  : float — coefficient for risk term (default: 0.4)
    slope_weight : float — coefficient for slope term (default: 0.1)

    Returns
    -------
    h : float — estimated cost-to-go (slightly inadmissible)
    """
    n = node_data[node_id]
    g = node_data[goal_id]

    # Euclidean distance between centroids
    dx = n["pos"][0] - g["pos"][0]
    dy = n["pos"][1] - g["pos"][1]
    dist = math.sqrt(dx * dx + dy * dy)

    # Inflate by local risk and slope
    risk = n.get("risk", 0.0)
    slope = n.get("slope", 0.0)

    h = dist * (1.0 + risk_weight * risk + slope_weight * slope)
    return h


def euclidean_heuristic(
    node_id: int,
    goal_id: int,
    node_data: dict[int, dict[str, Any]],
) -> float:
    """Pure Euclidean distance heuristic (admissible, for baseline B1).

    Parameters
    ----------
    node_id   : int — current node
    goal_id   : int — goal node
    node_data : dict mapping node_id → dict with 'pos' key

    Returns
    -------
    h : float — Euclidean distance
    """
    n = node_data[node_id]
    g = node_data[goal_id]

    dx = n["pos"][0] - g["pos"][0]
    dy = n["pos"][1] - g["pos"][1]
    return math.sqrt(dx * dx + dy * dy)
