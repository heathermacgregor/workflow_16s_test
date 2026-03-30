"""
Unified Statistics & Effect Sizes Module.

Consolidates effect size calculations (Cohen's d, Cliff's delta, etc.), 
confidence intervals, and statistical test wrappers into a single location.

References:
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
- Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional, Tuple
from workflow_16s.utils.logger import get_logger


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d (Standardized Mean Difference).
    d = (mean1 - mean2) / pooled_SD
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2: return np.nan
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0: return 0.0
    return (mean1 - mean2) / pooled_std

def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cliff's Delta (Non-parametric).
    robust to outliers and ordinal data.
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0: return np.nan
    
    # Count dominance: how many pairs where group1 > group2
    dominance = 0
    for x in group1:
        dominance += np.sum(x > group2) - np.sum(x < group2)
    
    return dominance / (n1 * n2)

def glass_delta(
    group1: np.ndarray, 
    group2: np.ndarray, 
    control_group: int = 2
) -> float:
    """
    Calculate Glass's Delta (uses only control group SD).
    Useful when experimental manipulation alters variance.
    """
    control = group2 if control_group == 2 else group1
    control_std = np.std(control, ddof=1)
    
    if control_std == 0: return 0.0
    return float((np.mean(group1) - np.mean(group2)) / control_std)

def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Hedges' g (Bias-corrected Cohen's d).
    Recommended for small sample sizes (n < 20).
    """
    d = cohens_d(group1, group2)
    n = len(group1) + len(group2)
    correction = 1 - (3 / (4 * n - 9))
    return d * correction

def log2_fold_change(
    group1: np.ndarray, 
    group2: np.ndarray, 
    pseudocount: float = 1.0
) -> float:
    """
    Calculate Log2 Fold Change with pseudocount.
    log2FC = log2((mean1 + P) / (mean2 + P))
    """
    mean1 = np.mean(group1) + pseudocount
    mean2 = np.mean(group2) + pseudocount
    return np.log2(mean1 / mean2)

# ==================================================================================== #
#                            INTERPRETATION & CONFIDENCE
# ==================================================================================== #

def interpret_effect_size(value: float, metric: str = 'cohens_d') -> str:
    """Interpret effect size magnitude based on standard thresholds."""
    abs_val = abs(value)
    
    if metric == 'cliffs_delta':
        if abs_val < 0.147: return 'negligible'
        if abs_val < 0.33:  return 'small'
        if abs_val < 0.474: return 'medium'
        return 'large'
    else: # Cohen's d / Hedges' g / Glass's Delta
        if abs_val < 0.2: return 'negligible'
        if abs_val < 0.5: return 'small'
        if abs_val < 0.8: return 'medium'
        return 'large'
    
def interpret_cliffs_delta(d_val: float) -> str:
    """
    Interprets Cliff's Delta effect size.
    Wrapper for interpret_effect_size to satisfy imports.
    
    Thresholds (Romano et al., 2006):
    |d| < 0.147 : Negligible
    |d| < 0.33  : Small
    |d| < 0.474 : Medium
    |d| >= 0.474: Large
    """
    return interpret_effect_size(d_val, metric='cliffs_delta')

def interpret_cohens_d(d_val: float) -> str:
    """
    Interprets Cohen's d effect size.
    Wrapper for interpret_effect_size to satisfy imports.
    """
    return interpret_effect_size(d_val, metric='cohens_d')

def effect_size_confidence_interval(
    group1: np.ndarray, 
    group2: np.ndarray,
    method: str = 'cohens_d',
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, float]:
    """
    Calculate Bootstrap Confidence Interval for an effect size.
    """
    n1, n2 = len(group1), len(group2)
    func_map = {
        'cohens_d': cohens_d, 
        'cliffs_delta': cliffs_delta, 
        'hedges_g': hedges_g
    }
    
    es_func = func_map.get(method, cohens_d)
    bootstrap_es = []
    
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
    
    return (float(lower), float(upper))

# ==================================================================================== #
#                                  WRAPPERS
# ==================================================================================== #

def calculate_all_effect_sizes(
    group1: np.ndarray, 
    group2: np.ndarray, 
    group_names: Optional[Tuple[str, str]] = None,
    pseudocount: float = 1.0
) -> Dict[str, Any]:
    """Compute all available effect size metrics for a pair of groups."""
    # Remove NaNs for safety
    g1 = group1[~np.isnan(group1)]
    g2 = group2[~np.isnan(group2)]
    
    if len(g1) < 2 or len(g2) < 2:
        return {'error': 'Insufficient samples (need ≥2 per group)'}

    results = {
        'n1': len(g1), 
        'n2': len(g2),
        'mean1': np.mean(g1), 
        'mean2': np.mean(g2),
        'median1': np.median(g1), 
        'median2': np.median(g2),
        'std1': np.std(g1, ddof=1), 
        'std2': np.std(g2, ddof=1),
        'log2_fold_change': log2_fold_change(g1, g2, pseudocount)
    }
    
    if group_names:
        results['group1_name'], results['group2_name'] = group_names

    # Calculate standard metrics with interpretations
    metrics = [
        (cohens_d, 'cohens_d'), 
        (cliffs_delta, 'cliffs_delta'), 
        (hedges_g, 'hedges_g'),
        (glass_delta, 'glass_delta')
    ]
    
    for func, name in metrics:
        try:
            val = func(g1, g2)
            results[name] = val
            results[f'{name}_interpretation'] = interpret_effect_size(val, name)
        except Exception as e:
            logger = get_logger("workflow_16s")
            logger.debug(f"{name} calculation failed: {e}")
            results[name] = np.nan

    # Add biological significance flag based on Cliff's Delta (Robust)
    cd = abs(results.get('cliffs_delta', 0))
    if cd >= 0.33: results['biological_significance'] = 'likely meaningful'
    elif cd >= 0.147: results['biological_significance'] = 'potentially meaningful'
    else: results['biological_significance'] = 'negligible'

    return results

def effect_size_with_stats(
    data: pd.DataFrame,
    value_col: str,
    group_col: str,
    test: str = 'mannwhitneyu'
) -> pd.DataFrame:
    """
    Run a statistical test AND calculate effect sizes in one go.
    Useful for high-throughput screening of taxa.
    """
    groups = data[group_col].dropna().unique()
    
    if len(groups) != 2:
        logger = get_logger("workflow_16s")
        logger.error(f"Effect size requires exactly 2 groups, found {len(groups)}")
        return pd.DataFrame()
    
    g1 = data[data[group_col] == groups[0]][value_col].dropna().values
    g2 = data[data[group_col] == groups[1]][value_col].dropna().values
    
    # Statistical test
    if test == 'mannwhitneyu':
        stat, p_val_raw = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        test_name = 'Mann-Whitney U'
    elif test == 'ttest':
        stat, p_val_raw = stats.ttest_ind(g1, g2, equal_var=False)
        test_name = 'Welch t-test'
    else:
        raise ValueError(f"Unknown test: {test}")
    
    # Ensure p_val is a float
    p_val: float = float(p_val_raw) # type: ignore
    
    # Effect sizes
    effect_sizes = calculate_all_effect_sizes(
        g1, g2, group_names=(str(groups[0]), str(groups[1]))
    )
    
    # Combine results
    results = {
        'test': test_name,
        'statistic': stat,
        'p_value': p_val,
        'significant': p_val < 0.05,
        **effect_sizes
    }
    
    return pd.DataFrame([results])