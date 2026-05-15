"""
dstar.py
--------
Stage 8 (blueprint §15) — D* Lite for dynamic replanning.

D* Lite maintains an incremental search structure that updates efficiently
when edge costs change. Enables real-time path updates if onboard sensors
detect new hazards during traversal.

For pre-landing planning (primary thesis claim), D* is not required.
For active-traversal extension, D* provides real-time capability.

Reference:
  Koenig & Likhachev, "D* Lite", AAAI 2002.
"""

from __future__ import annotations

import heapq
import math
import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class DStarLite:
    """D* Lite incremental replanner.

    Uses the same cost function as A* (blueprint §15) but maintains
    search state for efficient replanning when edge costs change.

    Parameters
    ----------
    G         : NetworkX DiGraph with 'cost' edge attributes
    node_data : dict mapping node_id → feature dict
    start     : int — initial start node
    goal      : int — goal node
    """

    def __init__(self, G, node_data: dict[int, dict], start: int, goal: int):
        self.G = G
        self.node_data = node_data
        self.start = start
        self.goal = goal
        self.k_m = 0.0  # Key modifier for consistency after start changes

        # rhs and g values
        self.rhs: dict[int, float] = {}
        self.g: dict[int, float] = {}
        self._counter = 0
        self._open: list[tuple[tuple[float, float], int, int]] = []
        self._open_set: set[int] = set()

        # Initialise all nodes
        for n in G.nodes():
            self.rhs[n] = float("inf")
            self.g[n] = float("inf")

        # D* Lite works backwards: goal is the "start" of search
        self.rhs[self.goal] = 0.0
        self._insert(self.goal)

    def _heuristic(self, a: int, b: int) -> float:
        pa = self.node_data[a]["pos"]
        pb = self.node_data[b]["pos"]
        return math.sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)

    def _calculate_key(self, s: int) -> tuple[float, float]:
        g_val = self.g[s]
        rhs_val = self.rhs[s]
        min_val = min(g_val, rhs_val)
        return (min_val + self._heuristic(self.start, s) + self.k_m, min_val)

    def _insert(self, s: int):
        key = self._calculate_key(s)
        self._counter += 1
        heapq.heappush(self._open, (key, self._counter, s))
        self._open_set.add(s)

    def _predecessors(self, s: int) -> list[int]:
        return list(self.G.predecessors(s))

    def _successors(self, s: int) -> list[int]:
        return list(self.G.successors(s))

    def _cost(self, u: int, v: int) -> float:
        if self.G.has_edge(u, v):
            return self.G[u][v]["cost"]
        return float("inf")

    def _update_vertex(self, u: int):
        if u != self.goal:
            # rhs(u) = min over successors of cost(u,s) + g(s)
            min_val = float("inf")
            for s in self._successors(u):
                val = self._cost(u, s) + self.g[s]
                if val < min_val:
                    min_val = val
            self.rhs[u] = min_val

        # Remove from open if present (lazy deletion via staleness check)
        self._open_set.discard(u)

        if self.g[u] != self.rhs[u]:
            self._insert(u)

    def compute_shortest_path(self):
        """Run D* Lite search until the start node is consistent."""
        start_key = self._calculate_key(self.start)

        while self._open:
            # Peek at top
            top_key, _, top_node = self._open[0]

            if top_key >= start_key and self.rhs[self.start] == self.g[self.start]:
                break

            heapq.heappop(self._open)

            # Skip stale entries
            if top_node not in self._open_set:
                continue
            self._open_set.discard(top_node)

            u = top_node
            new_key = self._calculate_key(u)

            if top_key < new_key:
                # Reinsert with updated key
                self._insert(u)
            elif self.g[u] > self.rhs[u]:
                # Locally overconsistent
                self.g[u] = self.rhs[u]
                for s in self._predecessors(u):
                    self._update_vertex(s)
            else:
                # Locally underconsistent
                self.g[u] = float("inf")
                self._update_vertex(u)
                for s in self._predecessors(u):
                    self._update_vertex(s)

            start_key = self._calculate_key(self.start)

    def extract_path(self) -> list[int] | None:
        """Extract the current shortest path from start to goal.

        Returns
        -------
        list of node IDs from start to goal, or None if unreachable.
        """
        if self.g[self.start] == float("inf"):
            return None

        path = [self.start]
        current = self.start
        visited = {current}

        while current != self.goal:
            best_next = None
            best_cost = float("inf")

            for s in self._successors(current):
                c = self._cost(current, s) + self.g[s]
                if c < best_cost:
                    best_cost = c
                    best_next = s

            if best_next is None or best_next in visited:
                return None

            path.append(best_next)
            visited.add(best_next)
            current = best_next

        return path

    def update_edge_costs(self, changed_edges: list[tuple[int, int, float]]):
        """Update edge costs and trigger incremental replanning.

        Parameters
        ----------
        changed_edges : list of (u, v, new_cost) tuples
        """
        # Update key modifier
        old_start_h = self._heuristic(self.start, self.start)

        for u, v, new_cost in changed_edges:
            if self.G.has_edge(u, v):
                self.G[u][v]["cost"] = max(new_cost, 1e-10)
            else:
                self.G.add_edge(u, v, cost=max(new_cost, 1e-10))

            self._update_vertex(u)

        # Recompute
        self.compute_shortest_path()

    def replan(self, new_start: int) -> list[int] | None:
        """Replan from a new start position (rover has moved).

        Parameters
        ----------
        new_start : int — current rover node

        Returns
        -------
        path from new_start to goal, or None
        """
        self.k_m += self._heuristic(self.start, new_start)
        self.start = new_start
        self.compute_shortest_path()
        return self.extract_path()
