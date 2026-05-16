"""
oversmoothing.py
----------------
Phase 9 Publication Upgrade: Oversmoothing Analysis.

Metrics to detect representation collapse in Deep GNNs:
- Dirichlet Energy
- Mean Pairwise Cosine Similarity
- Variance Retention Rate
"""

import torch
import torch.nn.functional as F


def dirichlet_energy(embeddings: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Sum of squared differences across all edges.
    
    Approaches 0 as representations collapse (oversmoothing).
    
    Parameters
    ----------
    embeddings : (N, D) float32
    edge_index : (2, E) int64
    
    Returns
    -------
    energy : float
    """
    if edge_index.size(1) == 0:
        return 0.0
        
    src = embeddings[edge_index[0]]
    dst = embeddings[edge_index[1]]
    
    # L2 distance squared
    diff = (src - dst).pow(2).sum(dim=-1)
    
    # Half sum since edges are typically undirected (counted twice)
    return float(diff.sum().item()) / 2.0


def pairwise_cosine_similarity(embeddings: torch.Tensor, max_samples: int = 5000) -> float:
    """Mean pairwise cosine similarity of node representations.
    
    Approaches 1.0 if nodes become indistinguishable (oversmoothing).
    
    Parameters
    ----------
    embeddings : (N, D) float32
    max_samples: If N is large, randomly sample to prevent OOM.
    
    Returns
    -------
    similarity : float
    """
    N = embeddings.size(0)
    if N <= 1:
        return 1.0
        
    if N > max_samples:
        indices = torch.randperm(N)[:max_samples]
        emb = embeddings[indices]
    else:
        emb = embeddings
        
    # Normalize features
    emb_norm = F.normalize(emb, p=2, dim=-1)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(emb_norm, emb_norm.t())
    
    # Exclude self-similarity (diagonal)
    mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
    mean_sim = sim_matrix[mask].mean()
    
    return float(mean_sim.item())


def feature_variance(embeddings: torch.Tensor) -> float:
    """Mean variance across all feature dimensions.
    
    Low variance indicates collapse.
    """
    return float(embeddings.var(dim=0).mean().item())


def log_layer_variances(model, data) -> dict[str, float]:
    """Hook into model to record intermediate layer variances.
    
    Requires model modification or forward hook. 
    Returns dictionary of variance per layer.
    """
    # Assuming standard PhysicsAwareGNN structure:
    # This is a helper function that expects the model to return 
    # intermediate embeddings, or you can register hooks.
    
    metrics = {}
    
    # Register forward hooks on GATv2 layers
    layer_embs = {}
    
    def get_activation(name):
        def hook(model, input, output):
            layer_embs[name] = output.detach()
        return hook
        
    hooks = []
    if hasattr(model, 'conv1'):
        hooks.append(model.conv1.register_forward_hook(get_activation('layer1')))
    if hasattr(model, 'conv2'):
        hooks.append(model.conv2.register_forward_hook(get_activation('layer2')))
        
    # Forward pass
    with torch.no_grad():
        _ = model(data.x, data.edge_index)
        
    # Remove hooks
    for h in hooks:
        h.remove()
        
    # Compute metrics
    metrics['var_input'] = feature_variance(data.x)
    for name, emb in layer_embs.items():
        metrics[f'var_{name}'] = feature_variance(emb)
        metrics[f'dirichlet_{name}'] = dirichlet_energy(emb, data.edge_index)
        metrics[f'cos_sim_{name}'] = pairwise_cosine_similarity(emb)
        
    return metrics
