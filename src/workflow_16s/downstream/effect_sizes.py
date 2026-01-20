"""
Effect Size Calculations for Microbiome Differential Abundance

Provides biologically meaningful effect sizes beyond p-values:
- Cohen's d: Standardized mean difference
- Cliff's delta: Non-parametric effect size (robust to outliers)
- Fold-change: Log2 fold-change with pseudocount
- Glass's delta: Using control group SD only
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional
import warnings

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    
    d = (mean1 - mean2) / pooled_SD
    
    Interpretation:
    - Small: |d| = 0.2
    - Medium: |d| = 0.5
    - Large: |d| = 0.8
    
    Parameters
    ----------
    group1, group2 : array-like
        Values for each group
    
    Returns
    -------
    float
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_sd == 0:
        return 0.0
    
    return (mean1 - mean2) / pooled_sd


def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cliff's delta - non-parametric effect size.
    
    More robust to outliers than Cohen's d.
    Based on Mann-Whitney U statistic.
    
    delta = (# pairs where group1 > group2 - # pairs where group1 < group2) / (n1 * n2)
    
    Interpretation:
    - Negligible: |delta| < 0.147
    - Small: |delta| < 0.33
    - Medium: |delta| < 0.474
    - Large: |delta| >= 0.474
    
    Parameters
    ----------
    group1, group2 : array-like
        Values for each group
    
    Returns
    -------
    float
        Cliff's delta effect size (-1 to 1)
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return np.nan
    
    # Count dominance: how many pairs where group1 > group2
    dominance = 0
    for x in group1:
        dominance += np.sum(x > group2) - np.sum(x < group2)
    
    return dominance / (n1 * n2)


def log2_fold_change(group1: np.ndarray, group2: np.ndarray, 
                     pseudocount: float = 1.0) -> float:
    """
    Calculate log2 fold-change with pseudocount.
    
    log2FC = log2((mean1 + pseudocount) / (mean2 + pseudocount))
    
    Interpretation:
    - log2FC = 1: 2-fold increase
    - log2FC = -1: 2-fold decrease
    - log2FC = 2: 4-fold increase
    
    Parameters
    ----------
    group1, group2 : array-like
        Values for each group (typically abundance data)
    pseudocount : float
        Added to avoid log(0), default 1.0
    
    Returns
    -------
    float
        Log2 fold-change
    """
    mean1 = np.mean(group1) + pseudocount
    mean2 = np.mean(group2) + pseudocount
    
    return np.log2(mean1 / mean2)


def glass_delta(group1: np.ndarray, group2: np.ndarray, 
                control_group: int = 2) -> float:
    """
    Calculate Glass's delta - uses only control group SD.
    
    Appropriate when experimental group might have different variance.
    
    delta = (mean1 - mean2) / SD_control
    
    Parameters
    ----------
    group1, group2 : array-like
        Values for each group
    control_group : int
        Which group is control (1 or 2), default 2
    
    Returns
    -------
    float
        Glass's delta effect size
    """
    if control_group == 1:
        control_sd = np.std(group1, ddof=1)
    else:
        control_sd = np.std(group2, ddof=1)
    
    if control_sd == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / control_sd


def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Hedges' g - bias-corrected Cohen's d.
    
    Better for small sample sizes (n < 20).
    
    Parameters
    ----------
    group1, group2 : array-like
        Values for each group
    
    Returns
    -------
    float
        Hedges' g effect size
    """
    n1, n2 = len(group1), len(group2)
    df = n1 + n2 - 2
    
    # Correction factor
    correction = 1 - (3 / (4 * df - 1))
    
    return cohens_d(group1, group2) * correction


def calculate_all_effect_sizes(group1: np.ndarray, group2: np.ndarray,
                                pseudocount: float = 1.0) -> dict:
    """
    Calculate all effect sizes for a feature.
    
    Parameters
    ----------
    group1, group2 : array-like
        Values for each group
    pseudocount : float
        For log2FC calculation
    
    Returns
    -------
    dict
        Dictionary with all effect size metrics
    """
    return {
        'cohens_d': cohens_d(group1, group2),
        'cliffs_delta': cliffs_delta(group1, group2),
        'log2_fold_change': log2_fold_change(group1, group2, pseudocount),
        'hedges_g': hedges_g(group1, group2),
        'mean_group1': np.mean(group1),
        'mean_group2': np.mean(group2),
        'median_group1': np.median(group1),
        'median_group2': np.median(group2),
        'fold_change': np.mean(group1) / (np.mean(group2) + 1e-10),
    }


def interpret_effect_size(effect_size: float, method: str = 'cohens_d') -> str:
    """
    Provide interpretation of effect size magnitude.
    
    Parameters
    ----------
    effect_size : float
        The calculated effect size
    method : str
        Which method was used ('cohens_d', 'cliffs_delta')
    
    Returns
    -------
    str
        Interpretation ('negligible', 'small', 'medium', 'large')
    """
    abs_es = abs(effect_size)
    
    if method == 'cohens_d' or method == 'hedges_g':
        if abs_es < 0.2:
            return 'negligible'
        elif abs_es < 0.5:
            return 'small'
        elif abs_es < 0.8:
            return 'medium'
        else:
            return 'large'
    
    elif method == 'cliffs_delta':
        if abs_es < 0.147:
            return 'negligible'
        elif abs_es < 0.33:
            return 'small'
        elif abs_es < 0.474:
            return 'medium'
        else:
            return 'large'
    
    else:
        return 'unknown'


def effect_size_confidence_interval(group1: np.ndarray, group2: np.ndarray,
                                    method: str = 'cohens_d',
                                    confidence: float = 0.95,
                                    n_bootstrap: int = 1000) -> Tuple[float, float]:
    """
    Calculate confidence interval for effect size using bootstrap.
    
    Parameters
    ----------
    group1, group2 : array-like
        Values for each group
    method : str
        Effect size method to use
    confidence : float
        Confidence level (default 0.95 for 95% CI)
    n_bootstrap : int
        Number of bootstrap iterations
    
    Returns
    -------
    tuple
        (lower_bound, upper_bound) of confidence interval
    """
    n1, n2 = len(group1), len(group2)
    bootstrap_es = []
    
    func_map = {
        'cohens_d': cohens_d,
        'cliffs_delta': cliffs_delta,
        'hedges_g': hedges_g,
    }
    
    if method not in func_map:
        raise ValueError(f"Unknown method: {method}")
    
    es_func = func_map[method]
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx1 = np.random.choice(n1, size=n1, replace=True)
        idx2 = np.random.choice(n2, size=n2, replace=True)
        
        boot_g1 = group1[idx1]
        boot_g2 = group2[idx2]
        
        bootstrap_es.append(es_func(boot_g1, boot_g2))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_es, 100 * alpha / 2)
    upper = np.percentile(bootstrap_es, 100 * (1 - alpha / 2))
    
    return (lower, upper)
