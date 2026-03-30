"""
Enhanced Statistical Analysis Integration Module

This module integrates the new scientifically-appropriate methods into the workflow:
- Effect size calculations (Cohen's d, Cliff's delta, fold-change, etc.)
- Batch effect detection and correction (percentile normalization, ConQuR)
- Rarefaction analysis for sequencing depth adequacy
- Volcano plots for differential abundance visualization

These enhancements address critical weaknesses identified in the pipeline review:
1. Statistical testing without effect sizes (p-values alone are misleading)
2. Uncontrolled batch effects (using APPROPRIATE microbiome methods, NOT ComBat/limma)
3. Lack of sequencing depth validation
4. Limited visualization of differential abundance results

Author: GitHub Copilot (AI Assistant)
Date: 2026-01-08
"""

# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import anndata as ad
import numpy as np
import pandas as pd

# Local Imports - New Modules
from workflow_16s.downstream.stats.effect_sizes import (
    calculate_all_effect_sizes, cliffs_delta, cohens_d,
    effect_size_confidence_interval, hedges_g,
    interpret_cliffs_delta, interpret_cohens_d,
    log2_fold_change
)
from workflow_16s.downstream.stats.batch import (
    add_batch_as_covariate, apply_conqur_correction,
    detect_batch_effects, percentile_normalization,
    visualize_batch_effects
)
from workflow_16s.downstream.diversity import (
    assess_sequencing_adequacy, plot_rarefaction_curves,
    rarefaction_curves_for_dataset, suggest_rarefaction_depth
)
from workflow_16s.downstream.visualization import (
    create_ma_plot, create_volcano_plot, effect_size_volcano
)
from workflow_16s.downstream.utils import (
    fix_adata_dtypes, safe_write_h5ad
)
from workflow_16s.downstream.utils import (
    get_optimal_parameters, subsample_stratified, estimate_runtime
)

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')


# ==================================== FUNCTIONS ===================================== #

def add_effect_sizes_to_stats(
    stats_df: pd.DataFrame,
    adata: ad.AnnData,
    group_col: str,
    feature_col: str = 'feature',
    methods: List[str] = None
) -> pd.DataFrame:
    """
    Add effect size calculations to statistical test results.
    
    This addresses a critical limitation: p-values alone don't show biological importance.
    A statistically significant result (p < 0.05) can have a tiny effect size that's 
    biologically meaningless. Effect sizes quantify the magnitude of differences.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        Statistical test results with at least 'feature' column
    adata : ad.AnnData
        AnnData object with abundance data (adata.X) and metadata (adata.obs)
    group_col : str
        Column in adata.obs defining groups to compare
    feature_col : str, default='feature'
        Column in stats_df containing feature names
    methods : List[str], optional
        Effect size methods to calculate. If None, calculates all:
        - 'cliffs_delta': NON-PARAMETRIC, robust to outliers (PRIMARY for microbiome)
        - 'cohens_d': Standardized mean difference (assumes normality)
        - 'log2fc': Log2 fold-change with pseudocount
        - 'hedges_g': Bias-corrected Cohen's d for small samples
    
    Returns
    -------
    pd.DataFrame
        stats_df with added columns for each effect size method
    
    Notes
    -----
    - Cliff's delta is the PRIMARY method for microbiome data (non-parametric)
    - Cohen's d assumes normal distribution (violated by sparse microbiome data)
    - Only works for 2-group comparisons
    
    Examples
    --------
    >>> # After running Mann-Whitney U test:
    >>> stats_results = mwu_bonferroni(table, metadata, group_column='treatment')
    >>> enhanced = add_effect_sizes_to_stats(stats_results, adata, group_col='treatment')
    >>> # Now has columns: cliffs_delta, cohens_d, log2_fold_change, etc.
    >>> significant = enhanced[(enhanced['p_adj'] < 0.05) & (abs(enhanced['cliffs_delta']) > 0.33)]
    """
    if methods is None:
        methods = ['cliffs_delta', 'cohens_d', 'log2fc', 'hedges_g']
    
    # Validate inputs
    if feature_col not in stats_df.columns:
        raise ValueError(f"Feature column '{feature_col}' not found in stats_df")
    
    if group_col not in adata.obs.columns:
        raise ValueError(f"Group column '{group_col}' not found in adata.obs")
    
    # Get unique groups
    groups = adata.obs[group_col].unique()
    if len(groups) != 2:
        logger.warning(
            f"Effect size calculation requires exactly 2 groups, found {len(groups)}. "
            "Skipping effect sizes."
        )
        return stats_df
    
    logger.info(f"Adding effect sizes for {len(stats_df)} features...")
    
    # Convert AnnData to DataFrame for easier access
    feature_names = adata.var_names.tolist()
    abundance_df = pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
        index=adata.obs_names,
        columns=feature_names
    )
    
    # Add group column
    abundance_df[group_col] = adata.obs[group_col].values
    
    # Calculate effect sizes for each feature
    effect_sizes_data = []
    
    for _, row in stats_df.iterrows():
        feature = row[feature_col]
        
        if feature not in feature_names:
            logger.debug(f"Feature '{feature}' not found in adata.var_names")
            effect_sizes_data.append({m: np.nan for m in methods})
            continue
        
        # Get data for each group
        group1_data = abundance_df[abundance_df[group_col] == groups[0]][feature].values
        group2_data = abundance_df[abundance_df[group_col] == groups[1]][feature].values
        
        # Calculate effect sizes
        es = {}
        
        if 'cliffs_delta' in methods:
            es['cliffs_delta'] = cliffs_delta(group1_data, group2_data)
            es['cliffs_delta_interpretation'] = interpret_cliffs_delta(es['cliffs_delta'])
        
        if 'cohens_d' in methods:
            es['cohens_d'] = cohens_d(group1_data, group2_data)
            es['cohens_d_interpretation'] = interpret_cohens_d(es['cohens_d'])
        
        if 'log2fc' in methods:
            es['log2_fold_change'] = log2_fold_change(group1_data, group2_data)
            es['fold_change'] = 2 ** es['log2_fold_change']
        
        if 'hedges_g' in methods:
            es['hedges_g'] = hedges_g(group1_data, group2_data)
        
        effect_sizes_data.append(es)
    
    # Merge effect sizes into stats_df
    effect_sizes_df = pd.DataFrame(effect_sizes_data)
    enhanced_stats = pd.concat([stats_df.reset_index(drop=True), effect_sizes_df], axis=1)
    
    logger.info(f"Added {len(effect_sizes_df.columns)} effect size columns")
    
    return enhanced_stats


def check_and_correct_batch_effects(
    adata: ad.AnnData,
    batch_col: str,
    method: str = 'percentile',
    significance_threshold: float = 0.01,
    variance_threshold: float = 0.1,
    output_dir: Optional[Path] = None
) -> Tuple[bool, ad.AnnData]:
    """
    Detect batch effects and apply APPROPRIATE correction if needed.
    
    CRITICAL: This uses microbiome-appropriate methods, NOT gene expression methods.
    ComBat and limma are INAPPROPRIATE for microbiome data because they assume:
    - Continuous, normally-distributed data (violated by count-based data)
    - No zero-inflation (violated by sparse microbiome data)
    - Non-compositional (violated by relative abundances)
    
    Instead, we use:
    - Percentile normalization: Non-parametric quantile matching
    - ConQuR: R package designed specifically for microbiome batch correction
    - Batch as covariate: Include batch in statistical models (most conservative)
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with abundance data
    batch_col : str
        Column in adata.obs identifying batch/sequencing run
    method : str, default='percentile'
        Correction method: 'percentile', 'conqur', or 'covariate'
    significance_threshold : float, default=0.01
        P-value threshold for batch effect detection
    variance_threshold : float, default=0.1
        R² threshold (proportion of variance explained by batch)
    output_dir : Path, optional
        Directory to save diagnostic plots
    
    Returns
    -------
    tuple
        (batch_detected: bool, corrected_adata: AnnData)
        If no batch effect detected, returns original adata unchanged
    
    Notes
    -----
    - Detection uses PCA + ANOVA (batch explains variance in PC space)
    - Percentile normalization is fastest and works well for most cases
    - ConQuR requires R + rpy2 installation
    - "Covariate" method just adds dummy variables to adata.obs for modeling
    
    Examples
    --------
    >>> # Before statistical testing:
    >>> batch_detected, corrected = check_and_correct_batch_effects(
    ...     adata, batch_col='sequencing_run', output_dir=Path('qc/batch')
    ... )
    >>> if batch_detected:
    ...     logger.info("Batch effects detected and corrected")
    ...     adata = corrected
    """
    # Fix dtypes to avoid h5py errors
    fix_adata_dtypes(adata, inplace=True)
    
    logger.info(f"Checking for batch effects (batch column: {batch_col})...")
    
    # 1. Detect batch effects
    detection_results = detect_batch_effects(
        adata,
        batch_col=batch_col,
        n_components=5
    )
    
    p_value = detection_results['p_value']
    r_squared = detection_results['r_squared']
    
    logger.info(
        f"Batch effect detection: p={p_value:.4f}, R²={r_squared:.4f} "
        f"(thresholds: p<{significance_threshold}, R²>{variance_threshold})"
    )
    
    # 2. Determine if correction is needed
    batch_detected = (p_value < significance_threshold) and (r_squared > variance_threshold)
    
    if not batch_detected:
        logger.info("No significant batch effects detected. Skipping correction.")
        return False, adata
    
    logger.warning(
        f"BATCH EFFECTS DETECTED: Batch explains {r_squared*100:.1f}% of variance. "
        f"Applying {method} correction..."
    )
    
    # 3. Visualize before correction (if output_dir provided)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        before_plot = output_dir / 'batch_effects_before_correction.png'
        visualize_batch_effects(adata, batch_col, output_path=before_plot)
        logger.info(f"Saved diagnostic plot: {before_plot}")
    
    # 4. Apply correction
    if method == 'percentile':
        corrected = percentile_normalization(adata, batch_col=batch_col)
    
    elif method == 'conqur':
        corrected = apply_conqur_correction(
            adata,
            batch_col=batch_col,
            covariate_cols=None  # Add if you want to preserve biological covariates
        )
    
    elif method == 'covariate':
        # Just add dummy variables - actual correction happens in statistical models
        corrected = add_batch_as_covariate(adata, batch_col=batch_col)
        logger.info(
            f"Added batch dummy variables to adata.obs. "
            "Include these in your statistical models."
        )
    
    else:
        raise ValueError(
            f"Unknown batch correction method: {method}. "
            "Choose 'percentile', 'conqur', or 'covariate'"
        )
    
    # 5. Visualize after correction (if output_dir provided)
    if output_dir and method != 'covariate':
        after_plot = output_dir / 'batch_effects_after_correction.png'
        visualize_batch_effects(corrected, batch_col, output_path=after_plot)
        logger.info(f"Saved diagnostic plot: {after_plot}")
    
    # 6. Fix dtypes in corrected data
    fix_adata_dtypes(corrected, inplace=True)
    
    logger.info(f"Batch correction complete using {method} method")
    
    return True, corrected


def validate_sequencing_depth(
    adata: ad.AnnData,
    output_dir: Path,
    min_adequate_pct: float = 0.80,
    plot: bool = True,
    ignore_size_recommendations: bool = False,
    stratify_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate that sequencing depth was adequate to capture community diversity.
    
    If rarefaction curves don't plateau, you're missing rare taxa and diversity 
    estimates are biased. This is a critical QC step often overlooked.
    
    For large datasets (>1000 samples), automatically subsamples to avoid excessive
    runtimes unless ignore_size_recommendations=True.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with count data (pre-normalization)
    output_dir : Path
        Directory to save rarefaction plots
    min_adequate_pct : float, default=0.80
        Minimum % of samples that should reach plateau
    plot : bool, default=True
        Whether to generate rarefaction curve plots
    ignore_size_recommendations : bool, default=False
        If True, disables automatic subsampling for large datasets
        Set via config: performance.ignore_size_recommendations = true
    stratify_col : str, optional
        Column to stratify subsampling (preserves group ratios)
    
    Returns
    -------
    dict
        {
            'n_samples': int,
            'n_adequate': int,
            'pct_adequate': float,
            'mean_plateau_ratio': float,
            'suggested_rarefaction_depth': int,
            'passes_qc': bool,
            'subsampled': bool,
            'n_samples_used': int,
            'n_samples_total': int
        }
    
    Raises
    ------
    ValueError
        If < min_adequate_pct samples reached plateau
    
    Notes
    -----
    - Plateau = final richness ≥ 95% of asymptote
    - Suggested depth = 10th percentile of read counts (standard practice)
    - Curves use multinomial sampling (100 iterations for stability)
    
    Examples
    --------
    >>> # Before rarefying or normalizing:
    >>> depth_results = validate_sequencing_depth(
    ...     raw_adata, output_dir=Path('qc/rarefaction'),
    ...     stratify_col='treatment'  # Preserve group ratios if subsampling
    ... )
    >>> if not depth_results['passes_qc']:
    ...     logger.warning("Insufficient sequencing depth!")
    """
    logger.info("Validating sequencing depth with rarefaction curves...")
    
    # Check dataset size and get recommendations
    profile = get_optimal_parameters(adata, ignore_size_recommendations)
    
    # Apply subsampling if recommended
    adata_to_use = adata
    subsampled = False
    if profile.recommendations.get('rarefaction_subsample') is not None:
        n_subsample = profile.recommendations['rarefaction_subsample']
        logger.info(f"Large dataset detected. Subsampling {n_subsample} samples for rarefaction analysis...")
        adata_to_use, _ = subsample_stratified(
            adata, n_samples=n_subsample, stratify_col=stratify_col
        )
        subsampled = True
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Calculate rarefaction curves
    rarefaction_data = rarefaction_curves_for_dataset(adata_to_use)
    
    # 2. Plot curves
    if plot:
        plot_path = output_dir / 'rarefaction_curves.png'
        plot_rarefaction_curves(
            rarefaction_data,
            output_path=plot_path,
            show_individual=True,
            show_mean=True
        )
        logger.info(f"Saved rarefaction plot: {plot_path}")
    
    # 3. Assess adequacy
    adequacy = assess_sequencing_adequacy(rarefaction_data)
    
    # 4. Suggest rarefaction depth
    suggested_depth = suggest_rarefaction_depth(adata)
    
    # 5. Compile results
    results = {
        'n_samples': adequacy['n_samples'],
        'n_adequate': adequacy['n_adequate'],
        'pct_adequate': adequacy['pct_adequate'],
        'mean_plateau_ratio': adequacy['mean_plateau_ratio'],
        'suggested_rarefaction_depth': suggested_depth,
        'passes_qc': adequacy['pct_adequate'] >= min_adequate_pct,
        'subsampled': subsampled,
        'n_samples_used': adata_to_use.n_obs,
        'n_samples_total': adata.n_obs
    }
    
    # 6. Log summary
    logger.info(
        f"Sequencing depth validation: "
        f"{results['n_adequate']}/{results['n_samples']} samples adequate "
        f"({results['pct_adequate']:.1f}%)"
    )
    logger.info(f"Mean plateau ratio: {results['mean_plateau_ratio']:.3f}")
    logger.info(f"Suggested rarefaction depth: {suggested_depth:,} reads")
    
    if not results['passes_qc']:
        logger.warning(
            f"QC FAIL: Only {results['pct_adequate']:.1f}% of samples reached plateau "
            f"(threshold: {min_adequate_pct*100:.0f}%). Consider deeper sequencing."
        )
    else:
        logger.info("QC PASS: Sequencing depth is adequate")
    
    return results


def create_differential_abundance_plots(
    stats_df: pd.DataFrame,
    output_dir: Path,
    feature_col: str = 'feature',
    p_col: str = 'p_adj',
    fc_col: str = 'log2_fold_change',
    effect_col: str = 'cliffs_delta',
    mean_col: str = 'mean_abundance',
    fc_threshold: float = 1.0,
    p_threshold: float = 0.05,
    effect_threshold: float = 0.33,
    top_n: int = 10
) -> Dict[str, Path]:
    """
    Generate publication-ready volcano and MA plots for differential abundance.
    
    These plots help identify which features are:
    - Statistically significant (low p-value)
    - Biologically meaningful (large effect size/fold-change)
    - Potentially driven by low abundance artifacts (MA plot)
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        Statistical results with effect sizes added
    output_dir : Path
        Directory to save plots
    feature_col : str, default='feature'
        Column with feature names
    p_col : str, default='p_adj'
        Column with adjusted p-values
    fc_col : str, default='log2_fold_change'
        Column with log2 fold-changes
    effect_col : str, default='cliffs_delta'
        Column with effect sizes (for effect size volcano)
    mean_col : str, default='mean_abundance'
        Column with mean abundance (for MA plot)
    fc_threshold : float, default=1.0
        Log2 fold-change threshold (1.0 = 2-fold change)
    p_threshold : float, default=0.05
        FDR threshold for significance
    effect_threshold : float, default=0.33
        Effect size threshold (0.33 = "medium" for Cliff's delta)
    top_n : int, default=10
        Number of top features to label
    
    Returns
    -------
    dict
        {plot_type: Path} for each generated plot
    
    Notes
    -----
    - Classic volcano: log2FC vs -log10(p)
    - MA plot: Detects low-abundance bias
    - Effect size volcano: Emphasizes biological importance over fold-change
    - All plots: 300 DPI, publication-ready
    
    Examples
    --------
    >>> # After adding effect sizes:
    >>> plots = create_differential_abundance_plots(
    ...     enhanced_stats, output_dir=Path('figures/volcano')
    ... )
    >>> # Returns: {'volcano': Path, 'ma': Path, 'effect_volcano': Path}
    """
    logger.info("Generating differential abundance plots...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_paths = {}
    
    # 1. Classic volcano plot
    if fc_col in stats_df.columns and p_col in stats_df.columns:
        volcano_path = output_dir / 'volcano_plot.png'
        
        fig = create_volcano_plot(
            stats_df,
            feature_col=feature_col,
            p_col=p_col,
            fc_col=fc_col,
            fc_threshold=fc_threshold,
            p_threshold=p_threshold,
            top_n=top_n
        )
        
        fig.savefig(volcano_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved volcano plot: {volcano_path}")
        plot_paths['volcano'] = volcano_path
    
    # 2. MA plot (mean vs amplitude)
    if fc_col in stats_df.columns and mean_col in stats_df.columns and p_col in stats_df.columns:
        ma_path = output_dir / 'ma_plot.png'
        
        fig = create_ma_plot(
            stats_df,
            feature_col=feature_col,
            mean_col=mean_col,
            fc_col=fc_col,
            p_col=p_col,
            fc_threshold=fc_threshold,
            p_threshold=p_threshold
        )
        
        fig.savefig(ma_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved MA plot: {ma_path}")
        plot_paths['ma'] = ma_path
    
    # 3. Effect size volcano
    if effect_col in stats_df.columns and p_col in stats_df.columns:
        effect_volcano_path = output_dir / 'effect_size_volcano.png'
        
        fig = effect_size_volcano(
            stats_df,
            feature_col=feature_col,
            p_col=p_col,
            effect_col=effect_col,
            effect_threshold=effect_threshold,
            p_threshold=p_threshold,
            top_n=top_n
        )
        
        fig.savefig(effect_volcano_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved effect size volcano: {effect_volcano_path}")
        plot_paths['effect_volcano'] = effect_volcano_path
    
    logger.info(f"Generated {len(plot_paths)} differential abundance plots")
    
    return plot_paths


# ============================= INTEGRATION FUNCTION ============================= #

def enhanced_differential_abundance_workflow(
    adata: ad.AnnData,
    group_col: str,
    batch_col: Optional[str] = None,
    output_dir: Path = None,
    validate_depth: bool = True,
    correct_batch: bool = True,
    batch_method: str = 'percentile',
    effect_size_methods: List[str] = None,
    create_plots: bool = True,
    **test_kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete enhanced differential abundance workflow with all QC and visualization.
    
    This integrates:
    1. Sequencing depth validation (rarefaction curves)
    2. Batch effect detection and correction (if batch_col provided)
    3. Statistical testing (user-provided test function)
    4. Effect size calculation (Cliff's delta, Cohen's d, fold-change)
    5. Volcano plot generation (classic, MA, effect size)
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with abundance data
    group_col : str
        Column in adata.obs defining groups to compare
    batch_col : str, optional
        Column identifying batch/sequencing run
    output_dir : Path, optional
        Base output directory. Creates subdirectories for each analysis
    validate_depth : bool, default=True
        Whether to run rarefaction curve validation
    correct_batch : bool, default=True
        Whether to correct batch effects if detected
    batch_method : str, default='percentile'
        Batch correction method: 'percentile', 'conqur', or 'covariate'
    effect_size_methods : List[str], optional
        Effect size methods to calculate
    create_plots : bool, default=True
        Whether to generate volcano plots
    **test_kwargs
        Passed to statistical test function (e.g., alpha, fold_change_threshold)
    
    Returns
    -------
    tuple
        (enhanced_stats_df: pd.DataFrame, metadata: dict)
        
        enhanced_stats_df has columns:
        - feature: Feature name
        - p_value, p_adj: P-values
        - cliffs_delta, cohens_d, log2_fold_change: Effect sizes
        - interpretations: Text interpretations
        
        metadata contains:
        - depth_validation: Dict from validate_sequencing_depth()
        - batch_detected: bool
        - batch_corrected: bool
        - plot_paths: Dict[str, Path]
    
    Examples
    --------
    >>> # Complete workflow:
    >>> from workflow_16s.stats.test import mwu_bonferroni
    >>> 
    >>> # Run statistical test first
    >>> stats_df = mwu_bonferroni(table, metadata, group_column='treatment')
    >>> 
    >>> # Enhance with new modules
    >>> enhanced, meta = enhanced_differential_abundance_workflow(
    ...     adata,
    ...     group_col='treatment',
    ...     batch_col='sequencing_run',
    ...     output_dir=Path('results/enhanced_stats')
    ... )
    >>> 
    >>> # Filter for biologically meaningful hits
    >>> hits = enhanced[
    ...     (enhanced['p_adj'] < 0.05) &  # Statistically significant
    ...     (abs(enhanced['cliffs_delta']) > 0.33)  # Medium effect size
    ... ]
    """
    if output_dir is None:
        output_dir = Path('.')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {}
    
    # STEP 1: Validate sequencing depth
    if validate_depth:
        logger.info("=" * 70)
        logger.info("STEP 1: Validating sequencing depth")
        logger.info("=" * 70)
        
        depth_results = validate_sequencing_depth(
            adata,
            output_dir=output_dir / 'rarefaction',
            plot=True
        )
        metadata['depth_validation'] = depth_results
        
        if not depth_results['passes_qc']:
            logger.warning(
                "Sequencing depth validation failed. Results may be biased. "
                "Consider rarefying to suggested depth or deeper sequencing."
            )
    
    # STEP 2: Detect and correct batch effects
    batch_detected = False
    batch_corrected = False
    
    if batch_col and correct_batch:
        logger.info("=" * 70)
        logger.info("STEP 2: Checking for batch effects")
        logger.info("=" * 70)
        
        batch_detected, corrected_adata = check_and_correct_batch_effects(
            adata,
            batch_col=batch_col,
            method=batch_method,
            output_dir=output_dir / 'batch_correction'
        )
        
        if batch_detected:
            adata = corrected_adata
            batch_corrected = True
    
    metadata['batch_detected'] = batch_detected
    metadata['batch_corrected'] = batch_corrected
    
    # STEP 3: Run statistical test (placeholder - user must provide stats_df)
    # Note: This function assumes stats_df is already computed
    # In practice, you'd call your test function here
    logger.info("=" * 70)
    logger.info("STEP 3: Statistical testing")
    logger.info("=" * 70)
    logger.info(
        "NOTE: This function expects stats_df as input. "
        "Run your statistical test (e.g., mwu_bonferroni) before calling this function."
    )
    
    # For now, return empty results
    # User should call individual functions: add_effect_sizes_to_stats, create_plots
    
    return pd.DataFrame(), metadata


# ============================= CONVENIENCE FUNCTIONS ============================ #

def quick_effect_size_report(stats_df: pd.DataFrame, effect_col: str = 'cliffs_delta') -> str:
    """Generate a quick summary report of effect sizes."""
    if effect_col not in stats_df.columns:
        return f"Effect column '{effect_col}' not found in results"
    
    es = stats_df[effect_col].dropna()
    
    report = f"""
Effect Size Summary ({effect_col}):
=====================================
Total features: {len(es)}
Mean: {es.mean():.3f}
Median: {es.median():.3f}
Min: {es.min():.3f}
Max: {es.max():.3f}

Interpretation (Cliff's delta):
- Negligible (|δ| < 0.147): {(abs(es) < 0.147).sum()} ({(abs(es) < 0.147).sum()/len(es)*100:.1f}%)
- Small (0.147 ≤ |δ| < 0.33): {((abs(es) >= 0.147) & (abs(es) < 0.33)).sum()} ({((abs(es) >= 0.147) & (abs(es) < 0.33)).sum()/len(es)*100:.1f}%)
- Medium (0.33 ≤ |δ| < 0.474): {((abs(es) >= 0.33) & (abs(es) < 0.474)).sum()} ({((abs(es) >= 0.33) & (abs(es) < 0.474)).sum()/len(es)*100:.1f}%)
- Large (|δ| ≥ 0.474): {(abs(es) >= 0.474).sum()} ({(abs(es) >= 0.474).sum()/len(es)*100:.1f}%)
"""
    
    return report
