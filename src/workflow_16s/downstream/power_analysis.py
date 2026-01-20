"""
Power Analysis Tools for Microbiome Study Design.

This module provides tools for estimating sample sizes and statistical power
for microbiome studies, helping researchers design adequately powered experiments.

Capabilities:
1. PERMANOVA power estimation
2. Differential abundance power estimation  
3. Alpha diversity power estimation
4. Pilot data-based power calculations
5. Power curves and visualizations

References:
    Kelly BJ, Gross R, Bittinger K, et al. (2015). Power and sample-size estimation
    for microbiome studies using pairwise distances and PERMANOVA.
    Bioinformatics, 31(15), 2461-2468.
    
    La Rosa PS, Brooks JP, Deych E, et al. (2012). Hypothesis testing and power
    calculations for taxonomic-based human microbiome data.
    PLoS ONE, 7(12), e52078.

Example:
    >>> from workflow_16s.downstream.power_analysis import (
    ...     estimate_permanova_power, plot_power_curves
    ... )
    >>> 
    >>> # Estimate required sample size
    >>> power_result = estimate_permanova_power(
    ...     pilot_adata,
    ...     group_col='treatment',
    ...     target_power=0.8,
    ...     effect_size=0.1
    ... )
    >>> 
    >>> # Plot power curves
    >>> fig = plot_power_curves(power_result)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import pairwise_distances
from statsmodels.stats.power import FTestAnovaPower, TTestIndPower

logger = logging.getLogger(__name__)


def estimate_permanova_power(
    adata: ad.AnnData,
    group_col: str,
    distance_metric: str = 'braycurtis',
    target_power: float = 0.8,
    alpha: float = 0.05,
    effect_sizes: Optional[List[float]] = None,
    sample_sizes: Optional[List[int]] = None
) -> Dict:
    """
    Estimate statistical power or required sample size for PERMANOVA.
    
    Uses pilot data to estimate within-group and between-group variance,
    then calculates power for different sample sizes or effect sizes.
    
    Args:
        adata: Pilot AnnData object
        group_col: Column with group labels
        distance_metric: Beta diversity metric
        target_power: Target statistical power (default: 0.8)
        alpha: Significance threshold
        effect_sizes: List of effect sizes (R²) to test
        sample_sizes: List of sample sizes to test
    
    Returns:
        Dictionary with power analysis results
    """
    logger.info("Estimating PERMANOVA power from pilot data")
    
    # Default parameter ranges
    if effect_sizes is None:
        effect_sizes = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25]
    if sample_sizes is None:
        sample_sizes = list(range(10, 201, 10))
    
    # Calculate distance matrix
    abundance = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    distances = pairwise_distances(abundance, metric=distance_metric)
    
    # Get groups
    groups = adata.obs[group_col].values
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    
    if n_groups < 2:
        raise ValueError("Need at least 2 groups for PERMANOVA power analysis")
    
    # Calculate variance components from pilot data
    total_ss = np.sum(distances ** 2) / (2 * len(distances))
    
    # Between-group variance
    group_centroids = []
    for group in unique_groups:
        group_mask = groups == group
        group_distances = distances[np.ix_(group_mask, group_mask)]
        centroid = np.mean(group_distances)
        group_centroids.append(centroid)
    
    between_ss = np.var(group_centroids) * len(distances) / n_groups
    within_ss = total_ss - between_ss
    
    # Estimate R² from pilot data
    pilot_r_squared = between_ss / total_ss if total_ss > 0 else 0
    
    logger.info(f"Pilot data statistics:")
    logger.info(f"  Groups: {n_groups}")
    logger.info(f"  Samples: {len(distances)}")
    logger.info(f"  Estimated R²: {pilot_r_squared:.4f}")
    
    # Calculate power for different scenarios
    power_results = []
    
    for effect_size in effect_sizes:
        for sample_size in sample_sizes:
            # Calculate F-statistic
            # F = (R² / (k-1)) / ((1-R²) / (n-k))
            # where k = number of groups, n = total sample size
            
            n_per_group = sample_size // n_groups
            total_n = n_per_group * n_groups
            
            if total_n < n_groups + 1:
                continue
            
            df_between = n_groups - 1
            df_within = total_n - n_groups
            
            f_stat = (effect_size / df_between) / ((1 - effect_size) / df_within)
            
            # Use F-distribution for power calculation
            f_power = FTestAnovaPower()
            power = f_power.solve_power(
                effect_size=effect_size,
                nobs=total_n,
                alpha=alpha,
                k_groups=n_groups
            )
            
            power_results.append({
                'effect_size': effect_size,
                'sample_size': sample_size,
                'samples_per_group': n_per_group,
                'power': power,
                'f_statistic': f_stat
            })
    
    power_df = pd.DataFrame(power_results)
    
    # Find minimum sample size for target power at pilot effect size
    target_ss_results = power_df[
        (power_df['effect_size'] == min(effect_sizes, key=lambda x: abs(x - pilot_r_squared))) &
        (power_df['power'] >= target_power)
    ]
    
    if len(target_ss_results) > 0:
        min_sample_size = target_ss_results['sample_size'].min()
        min_per_group = target_ss_results['samples_per_group'].min()
        logger.info(f"Minimum sample size for {target_power} power: {min_sample_size} total ({min_per_group} per group)")
    else:
        min_sample_size = None
        logger.warning(f"Target power {target_power} not achievable with tested sample sizes")
    
    return {
        'power_table': power_df,
        'pilot_r_squared': pilot_r_squared,
        'n_groups': n_groups,
        'target_power': target_power,
        'alpha': alpha,
        'min_sample_size': min_sample_size
    }


def estimate_da_power(
    mean_effect_size: float,
    within_group_variance: float,
    sample_sizes: Optional[List[int]] = None,
    alpha: float = 0.05,
    target_power: float = 0.8
) -> Dict:
    """
    Estimate power for differential abundance testing.
    
    Uses effect size and variance estimates to calculate power for
    detecting differential abundance with different sample sizes.
    
    Args:
        mean_effect_size: Expected mean difference (log-scale)
        within_group_variance: Within-group variance
        sample_sizes: Sample sizes to test (per group)
        alpha: Significance threshold
        target_power: Target statistical power
    
    Returns:
        Dictionary with power analysis results
    """
    logger.info("Estimating differential abundance power")
    
    if sample_sizes is None:
        sample_sizes = list(range(5, 101, 5))
    
    # Calculate Cohen's d
    cohens_d = mean_effect_size / np.sqrt(within_group_variance)
    
    logger.info(f"Cohen's d: {cohens_d:.3f}")
    
    # Use t-test power calculator
    power_calc = TTestIndPower()
    
    power_results = []
    for n in sample_sizes:
        power = power_calc.solve_power(
            effect_size=cohens_d,
            nobs1=n,
            alpha=alpha,
            ratio=1.0,  # Equal sample sizes
            alternative='two-sided'
        )
        
        power_results.append({
            'sample_size_per_group': n,
            'total_sample_size': 2 * n,
            'power': power,
            'cohens_d': cohens_d
        })
    
    power_df = pd.DataFrame(power_results)
    
    # Find minimum sample size for target power
    target_results = power_df[power_df['power'] >= target_power]
    
    if len(target_results) > 0:
        min_n = target_results['sample_size_per_group'].min()
        logger.info(f"Minimum sample size per group for {target_power} power: {min_n}")
    else:
        min_n = None
        logger.warning(f"Target power {target_power} not achievable with tested sample sizes")
    
    return {
        'power_table': power_df,
        'cohens_d': cohens_d,
        'target_power': target_power,
        'alpha': alpha,
        'min_sample_size_per_group': min_n
    }


def recommend_sample_size(
    pilot_adata: ad.AnnData,
    group_col: str,
    analysis_type: str = 'permanova',
    target_power: float = 0.8,
    alpha: float = 0.05,
    expected_effect_size: Optional[float] = None
) -> Dict:
    """
    Provide sample size recommendations based on pilot data.
    
    Analyzes pilot data to estimate variance components and effect sizes,
    then recommends sample sizes needed to achieve target statistical power.
    
    Args:
        pilot_adata: Pilot study AnnData object
        group_col: Column defining groups
        analysis_type: Type of analysis ('permanova', 'differential_abundance', 'alpha_diversity')
        target_power: Target statistical power (default: 0.8)
        alpha: Significance threshold (default: 0.05)
        expected_effect_size: Optional expected effect size (uses pilot estimate if None)
    
    Returns:
        Dictionary with sample size recommendations and power curves
    
    Example:
        >>> recommendations = recommend_sample_size(
        ...     pilot_adata,
        ...     group_col='treatment',
        ...     analysis_type='permanova',
        ...     target_power=0.8
        ... )
        >>> print(recommendations['recommendation'])
        "We recommend N=50 samples per group (100 total) to achieve 80% power"
    """
    logger.info(f"Generating sample size recommendations for {analysis_type}")
    
    if analysis_type == 'permanova':
        results = estimate_permanova_power(
            pilot_adata,
            group_col=group_col,
            target_power=target_power,
            alpha=alpha
        )
        
        min_sample_size = results['min_sample_size']
        pilot_r_squared = results['pilot_r_squared']
        n_groups = results['n_groups']
        
        if min_sample_size:
            recommendation = (
                f"Based on pilot data (R² = {pilot_r_squared:.3f}, {n_groups} groups), "
                f"we recommend N={min_sample_size} total samples "
                f"({min_sample_size // n_groups} per group) to achieve {target_power*100:.0f}% power "
                f"for detecting group differences with PERMANOVA at α={alpha}."
            )
        else:
            recommendation = (
                f"Pilot data shows very small effect size (R² = {pilot_r_squared:.3f}). "
                f"Consider increasing pilot sample size or expecting that very large "
                f"samples (N > 200) may be needed to achieve {target_power*100:.0f}% power."
            )
    
    elif analysis_type == 'differential_abundance':
        # Estimate from pilot data
        groups = pilot_adata.obs[group_col].values
        unique_groups = pd.Series(groups).dropna().unique()
        
        if len(unique_groups) != 2:
            raise ValueError("Differential abundance power analysis requires exactly 2 groups")
        
        # Calculate mean effect size and variance from pilot data
        X = pilot_adata.X.toarray() if hasattr(pilot_adata.X, 'toarray') else pilot_adata.X
        group1_mask = groups == unique_groups[0]
        group2_mask = groups == unique_groups[1]
        
        # Mean effect across all features
        mean_diffs = []
        variances = []
        
        for i in range(pilot_adata.n_vars):
            g1 = X[group1_mask, i]
            g2 = X[group2_mask, i]
            
            if len(g1) > 1 and len(g2) > 1:
                mean_diffs.append(np.abs(np.mean(g1) - np.mean(g2)))
                variances.append(np.mean([np.var(g1), np.var(g2)]))
        
        mean_effect = np.median(mean_diffs)
        mean_variance = np.median(variances)
        
        results = estimate_da_power(
            mean_effect_size=mean_effect if not expected_effect_size else expected_effect_size,
            within_group_variance=mean_variance,
            target_power=target_power,
            alpha=alpha
        )
        
        min_n = results['min_sample_size_per_group']
        cohens_d = results['cohens_d']
        
        if min_n:
            recommendation = (
                f"Based on pilot data (median Cohen's d = {cohens_d:.3f}), "
                f"we recommend N={min_n} samples per group ({2*min_n} total) "
                f"to achieve {target_power*100:.0f}% power for detecting differential abundance "
                f"at α={alpha}."
            )
        else:
            recommendation = (
                f"Pilot data shows very small effect size (Cohen's d = {cohens_d:.3f}). "
                f"Consider that very large samples may be needed or that the biological "
                f"difference is minimal."
            )
    
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    results['recommendation'] = recommendation
    results['analysis_type'] = analysis_type
    
    logger.info(f"Recommendation: {recommendation}")
    
    return results


def generate_power_report(
    power_results: Dict,
    output_path: Optional[Path] = None
) -> str:
    """
    Generate human-readable power analysis report.
    
    Args:
        power_results: Results from recommend_sample_size()
        output_path: Optional path to save markdown report
    
    Returns:
        Markdown-formatted report
    """
    analysis_type = power_results.get('analysis_type', 'unknown')
    target_power = power_results.get('target_power', 0.8)
    alpha = power_results.get('alpha', 0.05)
    
    report = f"""# Power Analysis Report

## Analysis Type: {analysis_type.title()}

### Study Design Parameters

- **Target Power:** {target_power*100:.0f}%
- **Significance Level (α):** {alpha}
- **Test Type:** Two-sided

### Sample Size Recommendation

{power_results['recommendation']}

### Interpretation

**Statistical power** is the probability of detecting a true effect when it exists.

Commonly accepted power levels:
- **80%** (0.80): Standard for most studies
- **90%** (0.90): Higher confidence, requires ~30% more samples
- **95%** (0.95): Very high confidence, requires ~60% more samples

**Important Considerations:**

1. **Pilot Data Limitations**: Sample size estimates are based on pilot data and assume
   similar effect sizes in the full study.

2. **Multiple Testing**: If testing many features, consider that multiple testing
   correction will reduce effective power. You may need 20-50% more samples to maintain
   power after correction.

3. **Unbalanced Groups**: If groups are unbalanced, total sample size should be increased
   by 10-25% to maintain power.

4. **Biological Variability**: High biological variability may require larger samples
   than estimated.

### Next Steps

1. **If underpowered** (current N < recommended N):
   - Consider recruiting more samples
   - Focus on largest expected effects
   - Use more efficient analysis methods (e.g., differential abundance vs. univariate tests)

2. **If adequately powered** (current N ≥ recommended N):
   - Proceed with analysis
   - Document power calculations in methods

3. **If overpowered** (current N >> recommended N):
   - Consider stricter significance thresholds
   - Focus on effect sizes, not just p-values
   - Use subset for discovery, remainder for validation

"""
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Saved power report to {output_path}")
    
    return report


def pilot_data_power_analysis(
    adata: ad.AnnData,
    group_col: str,
    target_power: float = 0.8,
    alpha: float = 0.05
) -> Dict:
    """
    Comprehensive power analysis using pilot data.
    
    Analyzes pilot data to estimate effect sizes and variances, then
    calculates power for various sample sizes.
    
    Args:
        adata: Pilot AnnData object
        group_col: Column with group labels
        target_power: Target statistical power
        alpha: Significance threshold
    
    Returns:
        Dictionary with comprehensive power analysis results
    """
    logger.info("="*60)
    logger.info("COMPREHENSIVE POWER ANALYSIS FROM PILOT DATA")
    logger.info("="*60)
    
    results = {}
    
    # PERMANOVA power
    logger.info("\n1. PERMANOVA Power Analysis")
    logger.info("-" * 40)
    
    permanova_power = estimate_permanova_power(
        adata,
        group_col=group_col,
        target_power=target_power,
        alpha=alpha
    )
    results['permanova'] = permanova_power
    
    # Differential abundance power (using CLR-transformed data)
    logger.info("\n2. Differential Abundance Power Analysis")
    logger.info("-" * 40)
    
    from workflow_16s.utils.compositional import clr_table
    
    # CLR transform
    clr_data = clr_table(adata.to_df())
    
    # Get groups
    groups = adata.obs[group_col].unique()
    if len(groups) == 2:
        # Calculate effect sizes for each feature
        group1_data = clr_data[adata.obs[group_col] == groups[0]]
        group2_data = clr_data[adata.obs[group_col] == groups[1]]
        
        mean_diffs = (group2_data.mean() - group1_data.mean()).abs()
        variances = pd.concat([group1_data, group2_data]).var()
        
        # Use median values
        median_effect = mean_diffs.median()
        median_variance = variances.median()
        
        logger.info(f"Median absolute effect size: {median_effect:.3f}")
        logger.info(f"Median variance: {median_variance:.3f}")
        
        da_power = estimate_da_power(
            mean_effect_size=median_effect,
            within_group_variance=median_variance,
            target_power=target_power,
            alpha=alpha
        )
        results['differential_abundance'] = da_power
    else:
        logger.info("Skipping DA power (requires 2 groups)")
        results['differential_abundance'] = None
    
    # Alpha diversity power (if calculated)
    if 'shannon' in adata.obs.columns:
        logger.info("\n3. Alpha Diversity Power Analysis")
        logger.info("-" * 40)
        
        if len(groups) == 2:
            shannon_g1 = adata.obs.loc[adata.obs[group_col] == groups[0], 'shannon']
            shannon_g2 = adata.obs.loc[adata.obs[group_col] == groups[1], 'shannon']
            
            effect = abs(shannon_g2.mean() - shannon_g1.mean())
            pooled_var = ((shannon_g1.var() * len(shannon_g1) + 
                          shannon_g2.var() * len(shannon_g2)) / 
                         (len(shannon_g1) + len(shannon_g2)))
            
            alpha_power = estimate_da_power(
                mean_effect_size=effect,
                within_group_variance=pooled_var,
                target_power=target_power,
                alpha=alpha
            )
            results['alpha_diversity'] = alpha_power
        else:
            results['alpha_diversity'] = None
    
    logger.info("\n" + "="*60)
    logger.info("POWER ANALYSIS SUMMARY")
    logger.info("="*60)
    
    if 'permanova' in results:
        logger.info(f"PERMANOVA min. sample size: {results['permanova']['min_sample_size']}")
    if results.get('differential_abundance'):
        logger.info(f"DA min. sample size per group: {results['differential_abundance']['min_sample_size_per_group']}")
    
    logger.info("="*60)
    
    return results


def plot_power_curves(
    power_results: Dict,
    output_path: Optional[Path] = None
) -> go.Figure:
    """
    Plot power curves from power analysis results.
    
    Args:
        power_results: Output from estimate_permanova_power or pilot_data_power_analysis
        output_path: Optional path to save plot
    
    Returns:
        Plotly figure object
    """
    # Determine what type of results we have
    if 'permanova' in power_results:
        # Comprehensive pilot data analysis
        power_df = power_results['permanova']['power_table']
        title = 'PERMANOVA Power Analysis'
    elif 'power_table' in power_results:
        # Single analysis result
        power_df = power_results['power_table']
        title = 'Power Analysis'
    else:
        raise ValueError("Invalid power_results format")
    
    # Create figure
    fig = go.Figure()
    
    # Plot power curves for each effect size
    if 'effect_size' in power_df.columns:
        for effect_size in power_df['effect_size'].unique():
            subset = power_df[power_df['effect_size'] == effect_size]
            
            fig.add_trace(go.Scatter(
                x=subset['sample_size'],
                y=subset['power'],
                mode='lines+markers',
                name=f'R² = {effect_size:.3f}',
                hovertemplate='Sample size: %{x}<br>Power: %{y:.3f}<extra></extra>'
            ))
    else:
        # Single power curve
        fig.add_trace(go.Scatter(
            x=power_df['sample_size_per_group'] if 'sample_size_per_group' in power_df.columns else power_df['sample_size'],
            y=power_df['power'],
            mode='lines+markers',
            name='Power',
            hovertemplate='Sample size: %{x}<br>Power: %{y:.3f}<extra></extra>'
        ))
    
    # Add reference line at 0.8 power
    fig.add_hline(
        y=0.8,
        line_dash='dash',
        line_color='gray',
        annotation_text='Target power (0.8)',
        annotation_position='right'
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Sample Size (Total)' if 'sample_size' in power_df.columns else 'Sample Size (Per Group)',
        yaxis_title='Statistical Power',
        template='plotly_white',
        height=500,
        width=800,
        hovermode='closest'
    )
    
    if output_path:
        fig.write_html(output_path)
        logger.info(f"Power curve plot saved to {output_path}")
    
    return fig


def minimal_detectable_effect(
    sample_size_per_group: int,
    within_group_variance: float,
    alpha: float = 0.05,
    power: float = 0.8
) -> float:
    """
    Calculate the minimal detectable effect size.
    
    Given a sample size and desired power, calculates the smallest
    effect size that can be reliably detected.
    
    Args:
        sample_size_per_group: Sample size per group
        within_group_variance: Within-group variance
        alpha: Significance threshold
        power: Desired statistical power
    
    Returns:
        Minimal detectable effect size (Cohen's d)
    """
    power_calc = TTestIndPower()
    
    # Solve for effect size
    cohens_d = power_calc.solve_power(
        effect_size=None,
        nobs1=sample_size_per_group,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative='two-sided'
    )
    
    # Convert to absolute effect
    min_effect = cohens_d * np.sqrt(within_group_variance)
    
    logger.info(f"Minimal detectable effect:")
    logger.info(f"  Cohen's d: {cohens_d:.3f}")
    logger.info(f"  Absolute effect: {min_effect:.3f}")
    
    return cohens_d


def power_analysis_report(
    power_results: Dict,
    output_path: Path
) -> None:
    """
    Generate a comprehensive power analysis report.
    
    Args:
        power_results: Output from pilot_data_power_analysis
        output_path: Path to save report (markdown file)
    """
    logger.info(f"Generating power analysis report: {output_path}")
    
    with open(output_path, 'w') as f:
        f.write("# Microbiome Study Power Analysis Report\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        
        f.write("## Summary\n\n")
        
        # PERMANOVA results
        if 'permanova' in power_results and power_results['permanova']:
            perm = power_results['permanova']
            f.write("### PERMANOVA Analysis\n\n")
            f.write(f"- **Pilot R²**: {perm['pilot_r_squared']:.4f}\n")
            f.write(f"- **Number of groups**: {perm['n_groups']}\n")
            f.write(f"- **Target power**: {perm['target_power']}\n")
            f.write(f"- **Alpha**: {perm['alpha']}\n")
            
            if perm['min_sample_size']:
                f.write(f"- **Minimum total sample size**: {perm['min_sample_size']}\n")
                f.write(f"- **Per group**: ~{perm['min_sample_size'] // perm['n_groups']}\n\n")
            else:
                f.write("- **Note**: Target power not achievable with tested sample sizes\n\n")
        
        # DA results
        if 'differential_abundance' in power_results and power_results['differential_abundance']:
            da = power_results['differential_abundance']
            f.write("### Differential Abundance Analysis\n\n")
            f.write(f"- **Cohen's d**: {da['cohens_d']:.3f}\n")
            f.write(f"- **Target power**: {da['target_power']}\n")
            
            if da['min_sample_size_per_group']:
                f.write(f"- **Minimum sample size per group**: {da['min_sample_size_per_group']}\n")
                f.write(f"- **Total sample size**: {da['min_sample_size_per_group'] * 2}\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("Based on the pilot data analysis:\n\n")
        
        # Generate recommendations
        if 'permanova' in power_results and power_results['permanova']['min_sample_size']:
            min_ss = power_results['permanova']['min_sample_size']
            f.write(f"1. For adequate PERMANOVA power, collect at least **{min_ss} samples total**\n")
        
        if 'differential_abundance' in power_results and power_results['differential_abundance']:
            if power_results['differential_abundance']['min_sample_size_per_group']:
                min_per_group = power_results['differential_abundance']['min_sample_size_per_group']
                f.write(f"2. For differential abundance testing, collect at least **{min_per_group} samples per group**\n")
        
        f.write("\n## Notes\n\n")
        f.write("- These estimates assume similar effect sizes to the pilot data\n")
        f.write("- Actual power may vary based on data characteristics\n")
        f.write("- Consider increasing sample size by 10-20% to account for quality control filtering\n")
    
    logger.info(f"Report saved to {output_path}")
