# ==================================================================================== #
# statistics/effect_sizes.py
# Effect Size Calculations for Microbiome Analyses
# ==================================================================================== #

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from workflow_16s.utils.logger import get_logger

logger = get_logger("workflow_16s")

# ==================================================================================== #

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for two independent groups.
    
    Cohen's d measures the standardized difference between two means,
    expressed in standard deviation units.
    
    Parameters
    ----------
    group1 : np.ndarray
        First group values
    group2 : np.ndarray
        Second group values
        
    Returns
    -------
    float
        Cohen's d effect size
        
    Interpretation
    --------------
    |d| < 0.2  : negligible
    0.2 ≤ |d| < 0.5 : small
    0.5 ≤ |d| < 0.8 : medium
    |d| ≥ 0.8  : large
    
    References
    ----------
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
        
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d


def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cliff's Delta effect size (non-parametric alternative to Cohen's d).
    
    Cliff's Delta measures the probability that a randomly selected value from
    one group is greater than a randomly selected value from another group.
    It's robust to outliers and doesn't assume normality.
    
    Parameters
    ----------
    group1 : np.ndarray
        First group values
    group2 : np.ndarray
        Second group values
        
    Returns
    -------
    float
        Cliff's Delta, ranging from -1 to 1
        
    Interpretation
    --------------
    |δ| < 0.147 : negligible
    0.147 ≤ |δ| < 0.33 : small
    0.33 ≤ |δ| < 0.474 : medium
    |δ| ≥ 0.474 : large
    
    References
    ----------
    Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions.
    """
    n1, n2 = len(group1), len(group2)
    
    if n1 == 0 or n2 == 0:
        return 0.0
    
    # Count dominances
    dominance = 0
    for x1 in group1:
        for x2 in group2:
            if x1 > x2:
                dominance += 1
            elif x1 < x2:
                dominance -= 1
    
    delta = dominance / (n1 * n2)
    return delta


def glass_delta(group1: np.ndarray, group2: np.ndarray, control_group: int = 2) -> float:
    """
    Calculate Glass's Delta (effect size using control group SD).
    
    Unlike Cohen's d, Glass's Delta uses only the control group's standard
    deviation, making it more appropriate when group variances differ substantially.
    
    Parameters
    ----------
    group1 : np.ndarray
        First group (often treatment)
    group2 : np.ndarray
        Second group (often control)
    control_group : int, optional
        Which group is the control (1 or 2), by default 2
        
    Returns
    -------
    float
        Glass's Delta effect size
        
    Interpretation
    --------------
    Same as Cohen's d
    
    References
    ----------
    Glass, G. V., McGaw, B., & Smith, M. L. (1981). Meta-analysis in social research.
    """
    control = group2 if control_group == 2 else group1
    control_std = np.std(control, ddof=1)
    
    if control_std == 0:
        return 0.0
        
    delta = (np.mean(group1) - np.mean(group2)) / control_std
    return delta


def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Hedges' g (bias-corrected Cohen's d for small samples).
    
    Hedges' g applies a correction factor to Cohen's d to reduce bias
    in small sample sizes (n < 20 per group).
    
    Parameters
    ----------
    group1 : np.ndarray
        First group values
    group2 : np.ndarray
        Second group values
        
    Returns
    -------
    float
        Hedges' g effect size
        
    References
    ----------
    Hedges, L. V., & Olkin, I. (1985). Statistical methods for meta-analysis.
    """
    d = cohens_d(group1, group2)
    n = len(group1) + len(group2)
    
    # Correction factor
    correction = 1 - (3 / (4 * n - 9))
    
    g = d * correction
    return g


def interpret_effect_size(value: float, metric: str = 'cohens_d') -> str:
    """
    Interpret effect size magnitude based on established thresholds.
    
    Parameters
    ----------
    value : float
        Effect size value
    metric : str, optional
        Type of effect size ('cohens_d', 'cliffs_delta', 'glass_delta', 'hedges_g')
        
    Returns
    -------
    str
        Interpretation category
    """
    abs_val = abs(value)
    
    if metric in ['cohens_d', 'glass_delta', 'hedges_g']:
        if abs_val < 0.2:
            return 'negligible'
        elif abs_val < 0.5:
            return 'small'
        elif abs_val < 0.8:
            return 'medium'
        else:
            return 'large'
            
    elif metric == 'cliffs_delta':
        if abs_val < 0.147:
            return 'negligible'
        elif abs_val < 0.33:
            return 'small'
        elif abs_val < 0.474:
            return 'medium'
        else:
            return 'large'
    
    return 'unknown'


def calculate_all_effect_sizes(
    group1: np.ndarray,
    group2: np.ndarray,
    group_names: Optional[Tuple[str, str]] = None
) -> Dict[str, Any]:
    """
    Calculate multiple effect size metrics for comprehensive reporting.
    
    Parameters
    ----------
    group1 : np.ndarray
        First group values
    group2 : np.ndarray
        Second group values
    group_names : Optional[Tuple[str, str]], optional
        Names for the two groups, by default None
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing all effect sizes and interpretations
    """
    # Remove NaNs
    g1 = group1[~np.isnan(group1)]
    g2 = group2[~np.isnan(group2)]
    
    if len(g1) < 2 or len(g2) < 2:
        return {
            'error': 'Insufficient samples (need ≥2 per group)',
            'n1': len(g1),
            'n2': len(g2)
        }
    
    results = {
        'n1': len(g1),
        'n2': len(g2),
        'mean1': np.mean(g1),
        'mean2': np.mean(g2),
        'std1': np.std(g1, ddof=1),
        'std2': np.std(g2, ddof=1),
        'median1': np.median(g1),
        'median2': np.median(g2)
    }
    
    if group_names:
        results['group1_name'] = group_names[0]
        results['group2_name'] = group_names[1]
    
    # Calculate effect sizes
    try:
        results['cohens_d'] = cohens_d(g1, g2)
        results['cohens_d_interpretation'] = interpret_effect_size(
            results['cohens_d'], 'cohens_d'
        )
    except Exception as e:
        logger.warning(f"Cohen's d calculation failed: {e}")
        results['cohens_d'] = np.nan
    
    try:
        results['cliffs_delta'] = cliffs_delta(g1, g2)
        results['cliffs_delta_interpretation'] = interpret_effect_size(
            results['cliffs_delta'], 'cliffs_delta'
        )
    except Exception as e:
        logger.warning(f"Cliff's Delta calculation failed: {e}")
        results['cliffs_delta'] = np.nan
    
    try:
        results['glass_delta'] = glass_delta(g1, g2)
        results['glass_delta_interpretation'] = interpret_effect_size(
            results['glass_delta'], 'glass_delta'
        )
    except Exception as e:
        logger.warning(f"Glass's Delta calculation failed: {e}")
        results['glass_delta'] = np.nan
    
    try:
        results['hedges_g'] = hedges_g(g1, g2)
        results['hedges_g_interpretation'] = interpret_effect_size(
            results['hedges_g'], 'hedges_g'
        )
    except Exception as e:
        logger.warning(f"Hedges' g calculation failed: {e}")
        results['hedges_g'] = np.nan
    
    # Add recommendation
    if not np.isnan(results.get('cliffs_delta', np.nan)):
        if abs(results['cliffs_delta']) >= 0.33:
            results['biological_significance'] = 'likely meaningful'
        elif abs(results['cliffs_delta']) >= 0.147:
            results['biological_significance'] = 'potentially meaningful'
        else:
            results['biological_significance'] = 'negligible'
    
    return results


def effect_size_with_stats(
    data: pd.DataFrame,
    value_col: str,
    group_col: str,
    test: str = 'mannwhitneyu'
) -> pd.DataFrame:
    """
    Combine statistical test with effect size calculations.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data containing values and grouping
    value_col : str
        Column name for values to compare
    group_col : str
        Column name for grouping variable
    test : str, optional
        Statistical test ('mannwhitneyu', 'ttest'), by default 'mannwhitneyu'
        
    Returns
    -------
    pd.DataFrame
        Results with p-value and effect sizes
    """
    groups = data[group_col].unique()
    
    if len(groups) != 2:
        logger.error(f"Effect size requires exactly 2 groups, found {len(groups)}")
        return pd.DataFrame()
    
    group1_data = data[data[group_col] == groups[0]][value_col].dropna().values
    group2_data = data[data[group_col] == groups[1]][value_col].dropna().values
    
    # Statistical test
    if test == 'mannwhitneyu':
        stat, p_val = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        test_name = 'Mann-Whitney U'
    elif test == 'ttest':
        stat, p_val = stats.ttest_ind(group1_data, group2_data)
        test_name = 'Independent t-test'
    else:
        raise ValueError(f"Unknown test: {test}")
    
    # Effect sizes
    effect_sizes = calculate_all_effect_sizes(
        group1_data, group2_data, group_names=(groups[0], groups[1])
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
