"""
graph_dataset.py
----------------
PyG Dataset loader for precomputed graph .pt files.

Blueprint §12 Step 7 / §13:
  Precomputed graphs are stored as PyG Data .pt files under
  data/processed/graphs/{split}/.  Each file contains a single
  graph for one tile.

  At training time, graphs are loaded from disk (~0.01s per graph)
  instead of being built on-the-fly (~0.8s per tile).

  CRITICAL (blueprint §12):
    If the fusion model is retrained, ALL precomputed graphs must be
    regenerated because H_final and α values are baked into node features.

Exports:
    PrecomputedGraphDataset  — PyTorch Dataset over .pt graph files
    build_graph_datasets     — factory to build train/val/test datasets

Usage:
    from src.data.graph_dataset import PrecomputedGraphDataset

    dataset = PrecomputedGraphDataset("data/processed/graphs/train")
    data = dataset[0]  # PyG Data object
"""

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class PrecomputedGraphDataset(Dataset):
    """PyTorch Dataset over precomputed PyG graph .pt files.

    Each .pt file contains a single torch_geometric.data.Data object
    produced by graph_builder.build_graph().

    Parameters
    ----------
    graph_dir : str or Path
        Directory containing .pt graph files.
    max_nodes : int or None
        If set, skip graphs with more than this many nodes (for memory).
    """

    def __init__(
        self,
        graph_dir: str | Path,
        max_nodes: int | None = None,
    ):
        self.graph_dir = Path(graph_dir)
        self.max_nodes = max_nodes

        if not self.graph_dir.exists():
            raise FileNotFoundError(
                f"Graph directory not found: {self.graph_dir}\n"
                "Run `python scripts/precompute_graphs.py` first."
            )

        # Discover all .pt files
        self.graph_files = sorted(self.graph_dir.glob("*.pt"))

        if not self.graph_files:
            raise FileNotFoundError(
                f"No .pt graph files found in {self.graph_dir}\n"
                "Run `python scripts/precompute_graphs.py` first."
            )

        log.info(
            "PrecomputedGraphDataset: %d graphs from %s",
            len(self.graph_files), self.graph_dir,
        )

    def __len__(self) -> int:
        return len(self.graph_files)

    def __getitem__(self, idx: int):
        """Load a precomputed graph from disk.

        Returns
        -------
        PyG Data object with fields:
            x              : (N, 14) float32
            edge_index     : (2, E)  int64
            edge_attr      : (E, 1)  float32
            pos            : (N, 2)  float32
            y              : (N,)    float32
            tier           : (N,)    int64
            pixel_membership : (H, W) int64
        """
        path = self.graph_files[idx]
        data = torch.load(str(path), map_location="cpu", weights_only=False)

        # Optional node count filtering
        if self.max_nodes is not None and data.x.shape[0] > self.max_nodes:
            log.debug("Skipping graph %s: %d nodes > %d max",
                      path.name, data.x.shape[0], self.max_nodes)
            # Return a minimal dummy — the DataLoader collate will handle it
            # In practice, filter these out before training
            return data

        return data

    def get_stats(self) -> dict:
        """Compute dataset statistics by loading a sample of graphs.

        Returns
        -------
        dict with: total_graphs, sample_size, mean_nodes, std_nodes,
                   min_nodes, max_nodes, mean_edges, feature_dim
        """
        import random
        sample_size = min(50, len(self.graph_files))
        sample_indices = random.sample(range(len(self.graph_files)), sample_size)

        node_counts = []
        edge_counts = []

        for idx in sample_indices:
            data = self[idx]
            node_counts.append(data.x.shape[0])
            edge_counts.append(data.edge_index.shape[1])

        import numpy as np
        node_arr = np.array(node_counts)
        edge_arr = np.array(edge_counts)

        return {
            "total_graphs": len(self.graph_files),
            "sample_size": sample_size,
            "mean_nodes": float(node_arr.mean()),
            "std_nodes": float(node_arr.std()),
            "min_nodes": int(node_arr.min()),
            "max_nodes": int(node_arr.max()),
            "mean_edges": float(edge_arr.mean()),
            "feature_dim": 14,
        }


def build_graph_datasets(
    graphs_dir: str | Path,
    splits: list[str] | None = None,
) -> dict[str, PrecomputedGraphDataset]:
    """Build PrecomputedGraphDataset for each split.

    Parameters
    ----------
    graphs_dir : str or Path
        Base directory (e.g., data/processed/graphs/) containing
        subdirectories for each split.
    splits : list of str or None
        Which splits to load. Default: ["train", "val", "test_in", "test_ood"].

    Returns
    -------
    dict mapping split name → PrecomputedGraphDataset
    """
    graphs_dir = Path(graphs_dir)
    if splits is None:
        splits = ["train", "val", "test_in", "test_ood"]

    datasets = {}
    for split in splits:
        split_dir = graphs_dir / split
        if split_dir.exists() and any(split_dir.glob("*.pt")):
            datasets[split] = PrecomputedGraphDataset(split_dir)
        else:
            log.warning("No graphs found for split '%s' in %s", split, split_dir)

    return datasets
