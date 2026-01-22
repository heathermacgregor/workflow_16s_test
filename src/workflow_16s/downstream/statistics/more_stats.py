"""
Enhanced Statistical Testing Module with Effect Sizes and Biological Significance.

This module provides comprehensive statistical testing for microbiome data with emphasis
on effect sizes and biological interpretation beyond p-values.

Features:
1. Effect size calculations (Cohen's d, Cliff's Delta, Eta-squared, R-squared)
2. Biological significance thresholds
3. Combined statistical and biological significance reporting
4. Multiple testing correction with FDR
5. Power calculations for observed effects
6. Confidence intervals for effect sizes

References:
    Cohen J. (1988). Statistical Power Analysis for the Behavioral Sciences.
    
    Cliff N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions.
    Psychological Bulletin, 114(3), 494-509.
    
    Fritz CO, Morris PE, Richler JJ. (2012). Effect size estimates: current use,
    calculations, and interpretation. Journal of Experimental Psychology: General,
    141(1), 2-18.

Example:
    >>> from workflow_16s.downstream.statistics import (
    ...     calculate_effect_sizes, test_with_effect_size, generate_stats_report
    ... )
    >>> 
    >>> # Calculate effect sizes for all features
    >>> results = calculate_effect_sizes(
    ...     adata, group_col='treatment', method='cliffs_delta'
    ... )
    >>> 
    >>> # Test with biological significance
    >>> sig_results = test_with_effect_size(
    ...     adata, group_col='treatment',
    ...     p_threshold=0.05, effect_threshold=0.5
    ... )
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower

logger = logging.getLogger('workflow_16s')

# =============================================================================
# EFFECT SIZE CALCULATIONS
# =============================================================================

def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
    pooled_sd: bool = True
) -> float:
    """
    Calculate Cohen's d effect size for two groups.
    
    Cohen's d is the standardized mean difference between two groups.
    
    Interpretation:
        |d| < 0.2:  negligible
        0.2 ≤ |d| < 0.5:  small
        0.5 ≤ |d| < 0.8:  medium
        |d| ≥ 0.8:  large
    
    Args:
        group1: First group values
        group2: Second group values
        pooled_sd: Use pooled standard deviation (default: True)
    
    Returns:
        Cohen's d effect size
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    
    if pooled_sd:
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = np.sqrt(pooled_var)
    else:
        pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
    
    if pooled_std == 0:
        return 0.0
    
    return (mean1 - mean2) / pooled_std


def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cliff's Delta effect size (non-parametric).
    
    Cliff's Delta is a robust, non-parametric effect size measure
    based on ordinal comparisons between groups.
    
    Interpretation:
        |δ| < 0.147:  negligible
        0.147 ≤ |δ| < 0.33:  small
        0.33 ≤ |δ| < 0.474:  medium
        |δ| ≥ 0.474:  large
    
    Args:
        group1: First group values
        group2: Second group values
    
    Returns:
        Cliff's Delta effect size (-1 to 1)
    """
    n1, n2 = len(group1), len(group2)
    
    # Count dominances
    dominance_count = 0
    for val1 in group1:
        for val2 in group2:
            if val1 > val2:
                dominance_count += 1
            elif val1 < val2:
                dominance_count -= 1
    
    return dominance_count / (n1 * n2)


def eta_squared(groups: List[np.ndarray]) -> float:
    """
    Calculate Eta-squared (η²) effect size for multiple groups.
    
    Eta-squared represents the proportion of variance explained by
    group membership.
    
    Interpretation:
        η² < 0.01:  negligible
        0.01 ≤ η² < 0.06:  small
        0.06 ≤ η² < 0.14:  medium
        η² ≥ 0.14:  large
    
    Args:
        groups: List of arrays, one per group
    
    Returns:
        Eta-squared effect size (0 to 1)
    """
    # Combine all data
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    
    # Calculate between-group sum of squares
    ss_between = sum(
        len(group) * (np.mean(group) - grand_mean) ** 2
        for group in groups
    )
    
    # Calculate total sum of squares
    ss_total = np.sum((all_data - grand_mean) ** 2)
    
    if ss_total == 0:
        return 0.0
    
    return ss_between / ss_total


def r_squared_from_correlation(r: float) -> float:
    """
    Calculate R² from Spearman or Pearson correlation coefficient.
    
    Interpretation:
        R² < 0.01:  negligible
        0.01 ≤ R² < 0.09:  small
        0.09 ≤ R² < 0.25:  medium
        R² ≥ 0.25:  large
    
    Args:
        r: Correlation coefficient
    
    Returns:
        R-squared (0 to 1)
    """
    return r ** 2


def interpret_effect_size(
    effect_size: float,
    method: str = 'cohens_d'
) -> str:
    """
    Interpret effect size magnitude using standard thresholds.
    
    Args:
        effect_size: Calculated effect size
        method: Method used ('cohens_d', 'cliffs_delta', 'eta_squared', 'r_squared')
    
    Returns:
        Interpretation string
    """
    abs_es = abs(effect_size)
    
    if method == 'cohens_d':
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
    
    elif method in ['eta_squared', 'r_squared']:
        if abs_es < 0.01:
            return 'negligible'
        elif abs_es < 0.06 if method == 'eta_squared' else abs_es < 0.09:
            return 'small'
        elif abs_es < 0.14 if method == 'eta_squared' else abs_es < 0.25:
            return 'medium'
        else:
            return 'large'
    
    return 'unknown'


# =============================================================================
# COMPREHENSIVE STATISTICAL TESTING
# =============================================================================

def calculate_effect_sizes(
    adata: ad.AnnData,
    group_col: str,
    method: str = 'cliffs_delta',
    features: Optional[List[str]] = None,
    min_samples_per_group: int = 3
) -> pd.DataFrame:
    """
    Calculate effect sizes for all features between groups.
    
    Args:
        adata: AnnData object with count data
        group_col: Metadata column defining groups
        method: Effect size method ('cohens_d', 'cliffs_delta', 'eta_squared')
        features: Specific features to test (default: all)
        min_samples_per_group: Minimum samples required per group
    
    Returns:
        DataFrame with feature, groups, effect_size, interpretation
    """
    logger.info(f"Calculating {method} effect sizes for {group_col}")
    
    # Get groups
    groups = adata.obs[group_col].values
    unique_groups = pd.Series(groups).dropna().unique()
    
    if len(unique_groups) < 2:
        raise ValueError(f"Need at least 2 groups, found {len(unique_groups)}")
    
    # Get data
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    if features is None:
        features = adata.var_names.tolist()
    
    feature_indices = [i for i, f in enumerate(adata.var_names) if f in features]
    
    results = []
    
    # For pairwise methods (Cohen's d, Cliff's delta)
    if method in ['cohens_d', 'cliffs_delta'] and len(unique_groups) == 2:
        group1_mask = groups == unique_groups[0]
        group2_mask = groups == unique_groups[1]
        
        # Check sample sizes
        if group1_mask.sum() < min_samples_per_group or group2_mask.sum() < min_samples_per_group:
            logger.warning(
                f"Insufficient samples: {group1_mask.sum()} vs {group2_mask.sum()}"
            )
            return pd.DataFrame()
        
        for idx in feature_indices:
            group1_vals = X[group1_mask, idx]
            group2_vals = X[group2_mask, idx]
            
            # Calculate effect size
            if method == 'cohens_d':
                es = cohens_d(group1_vals, group2_vals)
            else:  # cliffs_delta
                es = cliffs_delta(group1_vals, group2_vals)
            
            interpretation = interpret_effect_size(es, method)
            
            results.append({
                'feature': adata.var_names[idx],
                'group1': unique_groups[0],
                'group2': unique_groups[1],
                'n1': group1_mask.sum(),
                'n2': group2_mask.sum(),
                'effect_size': es,
                'abs_effect_size': abs(es),
                'interpretation': interpretation,
                'method': method
            })
    
    # For multi-group methods (Eta-squared)
    elif method == 'eta_squared':
        # Create groups
        group_masks = {g: groups == g for g in unique_groups}
        
        # Check sample sizes
        if any(mask.sum() < min_samples_per_group for mask in group_masks.values()):
            logger.warning("Some groups have insufficient samples")
        
        for idx in feature_indices:
            group_arrays = [X[mask, idx] for mask in group_masks.values()]
            
            # Calculate eta-squared
            es = eta_squared(group_arrays)
            interpretation = interpret_effect_size(es, method)
            
            results.append({
                'feature': adata.var_names[idx],
                'groups': ', '.join(unique_groups),
                'effect_size': es,
                'interpretation': interpretation,
                'method': method
            })
    
    df = pd.DataFrame(results)
    logger.info(f"Calculated effect sizes for {len(df)} features")
    
    return df


def test_with_effect_size(
    adata: ad.AnnData,
    group_col: str,
    p_threshold: float = 0.05,
    effect_threshold: float = 0.5,
    method: str = 'auto',
    correction: str = 'fdr_bh',
    min_samples_per_group: int = 3
) -> pd.DataFrame:
    """
    Perform statistical testing with effect size filtering.
    
    Identifies features that are both statistically significant (p < threshold)
    AND biologically significant (|effect_size| > threshold).
    
    Args:
        adata: AnnData object
        group_col: Grouping column
        p_threshold: P-value threshold (default: 0.05)
        effect_threshold: Minimum effect size for biological significance
        method: Statistical test ('auto', 'mannwhitneyu', 'kruskal', 't-test')
        correction: Multiple testing correction method
        min_samples_per_group: Minimum samples per group
    
    Returns:
        DataFrame with test results, effect sizes, and significance flags
    """
    logger.info(f"Testing features with p<{p_threshold} and |ES|>{effect_threshold}")
    
    # Get groups
    groups = adata.obs[group_col].values
    unique_groups = pd.Series(groups).dropna().unique()
    n_groups = len(unique_groups)
    
    # Determine statistical test
    if method == 'auto':
        method = 'mannwhitneyu' if n_groups == 2 else 'kruskal'
    
    # Get data
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    results = []
    
    # Perform tests for each feature
    for idx, feature in enumerate(adata.var_names):
        # Get group data
        if n_groups == 2:
            group1_mask = groups == unique_groups[0]
            group2_mask = groups == unique_groups[1]
            
            if group1_mask.sum() < min_samples_per_group or group2_mask.sum() < min_samples_per_group:
                continue
            
            group1_vals = X[group1_mask, idx]
            group2_vals = X[group2_mask, idx]
            
            # Statistical test
            if method == 'mannwhitneyu':
                stat, pval = stats.mannwhitneyu(
                    group1_vals, group2_vals, alternative='two-sided'
                )
            elif method == 't-test':
                stat, pval = stats.ttest_ind(group1_vals, group2_vals)
            else:
                raise ValueError(f"Unknown test: {method}")
            
            # Effect size
            es_cohens = cohens_d(group1_vals, group2_vals)
            es_cliffs = cliffs_delta(group1_vals, group2_vals)
            
            mean1, mean2 = np.mean(group1_vals), np.mean(group2_vals)
            log2_fc = np.log2((mean1 + 1) / (mean2 + 1))
            
            results.append({
                'feature': feature,
                'group1': unique_groups[0],
                'group2': unique_groups[1],
                'n1': group1_mask.sum(),
                'n2': group2_mask.sum(),
                'mean1': mean1,
                'mean2': mean2,
                'log2_fold_change': log2_fc,
                'statistic': stat,
                'p_value': pval,
                'cohens_d': es_cohens,
                'cliffs_delta': es_cliffs,
                'test_method': method
            })
        
        else:  # Multiple groups
            group_arrays = [X[groups == g, idx] for g in unique_groups]
            
            # Statistical test
            if method == 'kruskal':
                stat, pval = stats.kruskal(*group_arrays)
            else:
                raise ValueError(f"Test {method} not supported for >2 groups")
            
            # Effect size
            es_eta = eta_squared(group_arrays)
            
            results.append({
                'feature': feature,
                'n_groups': n_groups,
                'statistic': stat,
                'p_value': pval,
                'eta_squared': es_eta,
                'test_method': method
            })
    
    if not results:
        logger.warning("No features tested")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Multiple testing correction
    if correction:
        reject, pvals_corrected, _, _ = multipletests(
            df['p_value'].values, alpha=p_threshold, method=correction
        )
        df['p_adj'] = pvals_corrected
        df['significant'] = reject
    else:
        df['p_adj'] = df['p_value']
        df['significant'] = df['p_value'] < p_threshold
    
    # Add biological significance
    if n_groups == 2:
        df['biologically_significant'] = df['cliffs_delta'].abs() > effect_threshold
        df['both_significant'] = df['significant'] & df['biologically_significant']
        
        # Interpret effect sizes
        df['cohens_d_interpretation'] = df['cohens_d'].apply(
            lambda x: interpret_effect_size(x, 'cohens_d')
        )
        df['cliffs_delta_interpretation'] = df['cliffs_delta'].apply(
            lambda x: interpret_effect_size(x, 'cliffs_delta')
        )
    else:
        df['biologically_significant'] = df['eta_squared'] > effect_threshold
        df['both_significant'] = df['significant'] & df['biologically_significant']
        df['eta_squared_interpretation'] = df['eta_squared'].apply(
            lambda x: interpret_effect_size(x, 'eta_squared')
        )
    
    # Sort by significance and effect size
    sort_cols = ['both_significant', 'p_adj']
    if n_groups == 2:
        df['abs_effect'] = df['cliffs_delta'].abs()
    else:
        df['abs_effect'] = df['eta_squared']
    
    df = df.sort_values(['both_significant', 'abs_effect'], ascending=[False, False])
    
    logger.info(
        f"Found {df['significant'].sum()} statistically significant features"
    )
    logger.info(
        f"Found {df['biologically_significant'].sum()} biologically significant features"
    )
    logger.info(
        f"Found {df['both_significant'].sum()} features with BOTH significances"
    )
    
    return df


def generate_stats_report(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> str:
    """
    Generate human-readable statistical summary report.
    
    Args:
        results_df: Results from test_with_effect_size()
        output_path: Optional path to save markdown report
    
    Returns:
        Markdown-formatted report string
    """
    n_total = len(results_df)
    n_stat_sig = results_df['significant'].sum()
    n_bio_sig = results_df['biologically_significant'].sum()
    n_both = results_df['both_significant'].sum()
    
    report = f"""# Statistical Testing Report

## Summary

- **Total features tested:** {n_total}
- **Statistically significant (p < 0.05):** {n_stat_sig} ({n_stat_sig/n_total*100:.1f}%)
- **Biologically significant (large effect):** {n_bio_sig} ({n_bio_sig/n_total*100:.1f}%)
- **Both significant:** {n_both} ({n_both/n_total*100:.1f}%)

## Interpretation

Features are categorized into four groups:

1. **Both statistically AND biologically significant:** {n_both} features
   - These are the most reliable and interpretable findings
   - Recommended for follow-up and reporting

2. **Statistically significant only:** {n_stat_sig - n_both} features
   - P-value < 0.05 but small effect size
   - May be due to large sample size detecting trivial differences
   - Interpret with caution

3. **Biologically significant only:** {n_bio_sig - n_both} features
   - Large effect size but p ≥ 0.05
   - May indicate underpowered study
   - Consider increasing sample size

4. **Neither significant:** {n_total - max(n_stat_sig, n_bio_sig)} features
   - No evidence of meaningful difference

## Top 10 Features (Both Significant)

"""
    
    # Top features table
    top_both = results_df[results_df['both_significant']].head(10)
    
    if len(top_both) > 0:
        if 'group1' in top_both.columns:  # Two groups
            report += "| Feature | Log2 FC | Cliff's Δ | p-adj | Interpretation |\n"
            report += "|---------|---------|-----------|-------|----------------|\n"
            
            for _, row in top_both.iterrows():
                report += (
                    f"| {row['feature']} | "
                    f"{row['log2_fold_change']:.3f} | "
                    f"{row['cliffs_delta']:.3f} | "
                    f"{row['p_adj']:.2e} | "
                    f"{row['cliffs_delta_interpretation']} |\n"
                )
        else:  # Multiple groups
            report += "| Feature | η² | p-adj | Interpretation |\n"
            report += "|---------|-----|-------|----------------|\n"
            
            for _, row in top_both.iterrows():
                report += (
                    f"| {row['feature']} | "
                    f"{row['eta_squared']:.3f} | "
                    f"{row['p_adj']:.2e} | "
                    f"{row['eta_squared_interpretation']} |\n"
                )
    else:
        report += "*No features meet both significance criteria*\n"
    
    report += "\n## Recommendations\n\n"
    
    if n_both > 0:
        report += f"- ✅ {n_both} features show strong evidence of meaningful differences\n"
    else:
        report += "- ⚠️ No features meet both significance criteria - consider:\n"
        report += "  - Increasing sample size\n"
        report += "  - Adjusting effect size threshold\n"
        report += "  - Examining biologically significant features\n"
    
    if (n_stat_sig - n_both) > n_both:
        report += "- ⚠️ Many statistically significant features have small effects - possible overpowered study\n"
    
    if (n_bio_sig - n_both) > 5:
        report += "- ⚠️ Multiple biologically significant features lack statistical significance - possible underpowered study\n"
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Saved report to {output_path}")
    
    return report


# =============================================================================
# POWER CALCULATIONS
# =============================================================================

def calculate_achieved_power(
    n_per_group: int,
    effect_size: float,
    alpha: float = 0.05,
    test_type: str = 't-test'
) -> float:
    """
    Calculate achieved statistical power for observed effect size.
    
    Args:
        n_per_group: Sample size per group
        effect_size: Cohen's d
        alpha: Significance level
        test_type: Type of test ('t-test', 'mann-whitney')
    
    Returns:
        Achieved power (0 to 1)
    """
    if test_type == 't-test':
        power_analysis = TTestIndPower()
        power = power_analysis.power(
            effect_size=effect_size,
            nobs1=n_per_group,
            alpha=alpha,
            alternative='two-sided'
        )
        return power
    else:
        # Approximation for non-parametric tests
        # Power is typically ~5-10% lower than parametric
        power_analysis = TTestIndPower()
        parametric_power = power_analysis.power(
            effect_size=effect_size,
            nobs1=n_per_group,
            alpha=alpha,
            alternative='two-sided'
        )
        return parametric_power * 0.95  # Conservative estimate


def required_sample_size(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    test_type: str = 't-test'
) -> int:
    """
    Calculate required sample size per group.
    
    Args:
        effect_size: Expected Cohen's d
        power: Target power (default: 0.8)
        alpha: Significance level
        test_type: Type of test
    
    Returns:
        Required sample size per group
    """
    power_analysis = TTestIndPower()
    n = power_analysis.solve_power(
        effect_size=effect_size,
        power=power,
        alpha=alpha,
        alternative='two-sided'
    )
    return int(np.ceil(n))
