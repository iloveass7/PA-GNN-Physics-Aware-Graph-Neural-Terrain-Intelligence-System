"""
statistics.py
-------------
Phase 14 Publication Upgrade: Statistical Validation.

Statistical significance testing for model ablations.
Implements paired t-tests, Wilcoxon signed-rank tests, and bootstrap CIs.
"""

import numpy as np
from scipy import stats
import pandas as pd


def paired_t_test(scores_a: np.ndarray, scores_b: np.ndarray) -> tuple[float, float]:
    """Perform a paired t-test between two arrays of metrics.
    
    Parameters
    ----------
    scores_a, scores_b : (N,) float arrays of matching length.
    
    Returns
    -------
    t_stat, p_value
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Arrays must have the same length for paired t-test.")
    
    res = stats.ttest_rel(scores_a, scores_b)
    return res.statistic, res.pvalue


def wilcoxon_signed_rank(scores_a: np.ndarray, scores_b: np.ndarray) -> tuple[float, float]:
    """Non-parametric Wilcoxon signed-rank test.
    
    Useful when metric differences are not normally distributed.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Arrays must have the same length for Wilcoxon test.")
        
    res = stats.wilcoxon(scores_a, scores_b)
    return res.statistic, res.pvalue


def bootstrap_ci(scores: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95) -> tuple[float, float, float]:
    """Calculate the bootstrap confidence interval for the mean.
    
    Returns
    -------
    mean, lower_bound, upper_bound
    """
    mean_val = np.mean(scores)
    
    # Resample with replacement
    indices = np.random.randint(0, len(scores), size=(n_bootstrap, len(scores)))
    bootstrap_means = np.mean(scores[indices], axis=1)
    
    lower_bound = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)
    
    return mean_val, lower_bound, upper_bound


def significance_table(all_results: dict[str, np.ndarray], baseline_key: str) -> pd.DataFrame:
    """Generate a publication-ready comparison table.
    
    Parameters
    ----------
    all_results : dict mapping condition name -> array of trial scores.
    baseline_key : the key to compare all other models against.
    
    Returns
    -------
    DataFrame with Mean ± 95% CI and p-values compared to baseline.
    """
    baseline_scores = all_results[baseline_key]
    
    records = []
    for name, scores in all_results.items():
        mean, lb, ub = bootstrap_ci(scores)
        
        if name == baseline_key:
            p_val = 1.0
            sig = "-"
        else:
            _, p_val = paired_t_test(baseline_scores, scores)
            
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = "ns"
                
        records.append({
            "Model": name,
            "Mean": f"{mean:.4f}",
            "95% CI": f"[{lb:.4f}, {ub:.4f}]",
            "p-value": f"{p_val:.4e}" if p_val < 0.01 else f"{p_val:.3f}",
            "Significance": sig
        })
        
    return pd.DataFrame(records)
