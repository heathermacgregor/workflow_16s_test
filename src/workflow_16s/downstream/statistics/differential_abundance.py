# ==================================================================================== #
# statistics/differential_abundance.py
# Compositional Differential Abundance Testing
# ==================================================================================== #

from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats
from scipy.sparse import issparse

from workflow_16s.utils.logger import get_logger
from workflow_16s.downstream.statistics.multiple_testing import apply_multiple_testing_correction

# Import effect size calculations
try:
    from workflow_16s.downstream.statistics.effect_sizes import (
        cohens_d,
        cliffs_delta,
        interpret_effect_size
    )
    EFFECT_SIZES_AVAILABLE = True
except ImportError:
    EFFECT_SIZES_AVAILABLE = False

logger = get_logger("workflow_16s")

# ==================================================================================== #

def ancom_bc_wrapper(
    adata: ad.AnnData,
    group_col: str,
    formula: Optional[str] = None,
    output_dir: Optional[Path] = None,
    p_adj_method: str = 'fdr_bh',
    alpha: float = 0.05,
    struc_zero: bool = True,
    neg_lb: bool = True
) -> Optional[pd.DataFrame]:
    """
    Wrapper for ANCOM-BC (Analysis of Compositions of Microbiomes with Bias Correction).
    
    ANCOM-BC is a compositionally aware differential abundance method that:
    1. Corrects for sampling fraction bias
    2. Accounts for library size variation
    3. Controls FDR while maintaining power
    
    NOTE: This is a Python wrapper that requires R with ANCOMBC package installed.
    Install in R: install.packages("BiocManager"); BiocManager::install("ANCOMBC")
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with count data in .X or layers['raw_counts']
    group_col : str
        Metadata column for group comparison
    formula : Optional[str], optional
        Model formula (e.g., "~group + batch"), by default None (uses ~group_col)
    output_dir : Optional[Path], optional
        Directory to save results, by default None
    p_adj_method : str, optional
        FDR correction method, by default 'fdr_bh'
    alpha : float, optional
        Significance threshold, by default 0.05
    struc_zero : bool, optional
        Whether to detect structural zeros, by default True
    neg_lb : bool, optional
        Whether to use negative lower bound for bias correction, by default True
        
    Returns
    -------
    Optional[pd.DataFrame]
        Results table with differential abundance statistics
        
    Notes
    -----
    - Requires rpy2 and R with ANCOMBC package
    - More robust than simple t-tests or Mann-Whitney for compositional data
    - Handles batch effects and confounders when specified in formula
    
    References
    ----------
    Lin, H., & Peddada, S. D. (2020). Analysis of compositions of microbiomes 
    with bias correction. Nature Communications, 11(1), 3514.
    """
    logger.info("=== ANCOM-BC Differential Abundance ===")
    logger.info(f"Group column: {group_col}")
    
    # Check for rpy2
    try:
        from rpy2.robjects import pandas2ri, r
        from rpy2.robjects.packages import importr
        from rpy2.robjects.conversion import localconverter
    except ImportError:
        logger.error(
            "rpy2 not installed. Install with: pip install rpy2\n"
            "Also ensure R is installed with ANCOMBC package."
        )
        return None
    
    # Check if group column exists
    if group_col not in adata.obs.columns:
        logger.error(f"Group column '{group_col}' not found in adata.obs")
        return None
    
    # Prepare count matrix
    if 'raw_counts' in adata.layers:
        counts = adata.layers['raw_counts']
    else:
        counts = adata.X
    
    if issparse(counts):
        counts = counts.toarray()
    
    # Create count DataFrame
    count_df = pd.DataFrame(
        counts.T,  # ANCOM expects features × samples
        index=adata.var_names,
        columns=adata.obs_names
    )
    
    # Create metadata DataFrame
    metadata = adata.obs[[group_col]].copy()
    metadata.index.name = 'sample_id'
    
    # Default formula
    if formula is None:
        formula = f"~{group_col}"
    
    logger.info(f"Formula: {formula}")
    logger.info(f"Count matrix: {count_df.shape[0]} features × {count_df.shape[1]} samples")
    
    try:
        # Activate pandas conversion
        pandas2ri.activate()
        
        # Import R packages
        base = importr('base')
        phyloseq = importr('phyloseq')
        ancombc_pkg = importr('ANCOMBC')
        
        # Convert to R objects
        with localconverter(pandas2ri.converter):
            r_counts = pandas2ri.py2rpy(count_df)
            r_metadata = pandas2ri.py2rpy(metadata)
        
        # Create phyloseq object
        logger.info("Creating phyloseq object...")
        r_script = f"""
        library(phyloseq)
        library(ANCOMBC)
        
        # Create phyloseq object
        otu_mat <- as.matrix(counts_df)
        sample_df <- sample_data(metadata_df)
        
        ps <- phyloseq(otu_table(otu_mat, taxa_are_rows = TRUE),
                       sample_df)
        
        # Run ANCOM-BC
        ancombc_result <- ancombc(
            phyloseq = ps,
            formula = "{formula}",
            p_adj_method = "{p_adj_method}",
            alpha = {alpha},
            struc_zero = {str(struc_zero).upper()},
            neg_lb = {str(neg_lb).upper()},
            global = TRUE
        )
        
        ancombc_result
        """
        
        # Execute R script
        r.assign('counts_df', r_counts)
        r.assign('metadata_df', r_metadata)
        
        logger.info("Running ANCOM-BC (this may take a few minutes)...")
        ancombc_result = r(r_script)
        
        # Extract results
        res = ancombc_result.rx2('res')
        
        # Convert to pandas
        with localconverter(pandas2ri.converter):
            lfc = pandas2ri.rpy2py(res.rx2('lfc'))  # Log fold changes
            se = pandas2ri.rpy2py(res.rx2('se'))    # Standard errors
            W = pandas2ri.rpy2py(res.rx2('W'))      # Test statistics
            p_val = pandas2ri.rpy2py(res.rx2('p_val'))  # P-values
            q_val = pandas2ri.rpy2py(res.rx2('q_val'))  # Adjusted p-values
            diff_abn = pandas2ri.rpy2py(res.rx2('diff_abn'))  # Significance flags
        
        # Combine into results DataFrame
        results_list = []
        
        for i, feature in enumerate(count_df.index):
            feature_result = {
                'feature': feature,
                'log_fold_change': lfc.iloc[i, 0] if lfc.shape[1] > 0 else np.nan,
                'standard_error': se.iloc[i, 0] if se.shape[1] > 0 else np.nan,
                'W_statistic': W.iloc[i, 0] if W.shape[1] > 0 else np.nan,
                'p_value': p_val.iloc[i, 0] if p_val.shape[1] > 0 else np.nan,
                'q_value': q_val.iloc[i, 0] if q_val.shape[1] > 0 else np.nan,
                'significant': diff_abn.iloc[i, 0] if diff_abn.shape[1] > 0 else False
            }
            results_list.append(feature_result)
        
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('q_value')
        
        # Add taxonomy if available
        if 'Taxon' in adata.var.columns:
            results_df = results_df.merge(
                adata.var[['Taxon']], 
                left_on='feature', 
                right_index=True, 
                how='left'
            )
        
        # Log summary
        n_sig = results_df['significant'].sum()
        logger.info(f"\n=== ANCOM-BC Results ===")
        logger.info(f"Significant features: {n_sig}/{len(results_df)} ({n_sig/len(results_df)*100:.1f}%)")
        logger.info(f"Top 10 significant features:")
        logger.info(f"\n{results_df.head(10).to_string(index=False)}")
        
        # Save results
        if output_dir:
            output_dir.mkdir(exist_ok=True, parents=True)
            output_file = output_dir / f"ancombc_results_{group_col}.csv"
            results_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to: {output_file}")
        
        pandas2ri.deactivate()
        
        return results_df
        
    except Exception as e:
        logger.error(f"ANCOM-BC failed: {e}")
        logger.error(
            "Ensure R is installed with ANCOMBC package:\n"
            "  R> install.packages('BiocManager')\n"
            "  R> BiocManager::install('ANCOMBC')"
        )
        pandas2ri.deactivate()
        return None


def simple_compositional_da(
    adata: ad.AnnData,
    group_col: str,
    output_dir: Optional[Path] = None,
    method: str = 'mannwhitneyu',
    fdr_method: str = 'fdr_bh',
    alpha: float = 0.05,
    min_prevalence: float = 0.1,
    pseudocount: float = 1.0
) -> pd.DataFrame:
    """
    Simple compositional differential abundance using CLR transformation.
    
    This is a fallback method when ANCOM-BC is unavailable. It:
    1. CLR-transforms counts to address compositionality
    2. Tests group differences on CLR-transformed abundances
    3. Applies FDR correction
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object
    group_col : str
        Grouping column
    output_dir : Optional[Path], optional
        Output directory
    method : str, optional
        Statistical test ('mannwhitneyu', 'ttest'), by default 'mannwhitneyu'
    fdr_method : str, optional
        FDR correction method, by default 'fdr_bh'
    alpha : float, optional
        Significance threshold, by default 0.05
    min_prevalence : float, optional
        Minimum prevalence filter, by default 0.1
    pseudocount : float, optional
        Pseudocount for CLR, by default 1.0
        
    Returns
    -------
    pd.DataFrame
        Differential abundance results
    """
    logger.info("=== Simple Compositional DA (CLR + {method.upper()}) ===")
    
    if group_col not in adata.obs.columns:
        logger.error(f"Group column '{group_col}' not found")
        return pd.DataFrame()
    
    # Get groups
    groups = adata.obs[group_col].unique()
    if len(groups) != 2:
        logger.error(f"Requires exactly 2 groups, found {len(groups)}")
        return pd.DataFrame()
    
    # CLR transformation
    if 'raw_counts' in adata.layers:
        counts = adata.layers['raw_counts']
    else:
        counts = adata.X
    
    if issparse(counts):
        counts = counts.toarray()
    
    # Add pseudocount and CLR transform
    counts_pseudo = counts + pseudocount
    geo_means = stats.gmean(counts_pseudo, axis=1, keepdims=True)
    clr_data = np.log(counts_pseudo / geo_means)
    
    # Filter by prevalence
    prevalence = (counts > 0).sum(axis=0) / counts.shape[0]
    keep_features = prevalence >= min_prevalence
    
    logger.info(f"Filtering: {np.sum(keep_features)}/{len(keep_features)} features pass {min_prevalence*100}% prevalence")
    
    clr_data = clr_data[:, keep_features]
    feature_names = adata.var_names[keep_features]
    
    # Perform tests
    results = []
    
    for i, feature in enumerate(feature_names):
        group1_data = clr_data[adata.obs[group_col] == groups[0], i]
        group2_data = clr_data[adata.obs[group_col] == groups[1], i]
        
        # Statistical test
        if method == 'mannwhitneyu':
            stat, p_val = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        elif method == 'ttest':
            stat, p_val = stats.ttest_ind(group1_data, group2_data)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Effect size (difference in mean CLR values)
        mean_diff = np.mean(group1_data) - np.mean(group2_data)
        
        # Calculate effect sizes
        effect_size_cohens = np.nan
        effect_size_cliffs = np.nan
        effect_interpretation = 'unknown'
        
        if EFFECT_SIZES_AVAILABLE:
            try:
                # Cohen's d for parametric effect size
                effect_size_cohens = cohens_d(group1_data, group2_data)
                
                # Cliff's delta for non-parametric effect size
                effect_size_cliffs = cliffs_delta(group1_data, group2_data)
                
                # Use Cliff's delta interpretation (more robust)
                effect_interpretation = interpret_effect_size(
                    effect_size_cliffs, 
                    method='cliffs_delta'
                )
            except Exception as e:
                logger.debug(f"Effect size calculation failed for {feature}: {e}")
        
        results.append({
            'feature': feature,
            'mean_clr_group1': np.mean(group1_data),
            'mean_clr_group2': np.mean(group2_data),
            'clr_difference': mean_diff,
            'cohens_d': effect_size_cohens,
            'cliffs_delta': effect_size_cliffs,
            'effect_size_interpretation': effect_interpretation,
            'test_statistic': stat,
            'p_value': p_val
        })
    
    results_df = pd.DataFrame(results)
    
    # FDR correction
    reject, p_adj, _ = apply_multiple_testing_correction(
        results_df['p_value'].values, method=fdr_method, alpha=alpha
    )
    
    results_df['p_value_adjusted'] = p_adj
    results_df['significant'] = reject
    
    # Add biological significance flag (requires both statistical significance AND large effect)
    if EFFECT_SIZES_AVAILABLE and 'cliffs_delta' in results_df.columns:
        results_df['biologically_significant'] = (
            (results_df['cliffs_delta'].abs() >= 0.33) &  # At least medium effect (Cliff's delta)
            (results_df['p_value_adjusted'] < alpha)
        )
        
        n_bio_sig = results_df['biologically_significant'].sum()
        logger.info(f"Biologically significant features: {n_bio_sig}/{len(results_df)} ({n_bio_sig/len(results_df)*100:.1f}%)")
    
    # Add taxonomy
    if 'Taxon' in adata.var.columns:
        taxon_map = adata.var.loc[feature_names, 'Taxon']
        results_df['Taxon'] = results_df['feature'].map(taxon_map)
    
    # Sort by p-value
    results_df = results_df.sort_values('p_value_adjusted')
    
    # Log summary
    n_sig = results_df['significant'].sum()
    logger.info(f"Significant features: {n_sig}/{len(results_df)} ({n_sig/len(results_df)*100:.1f}%)")
    
    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / f"simple_da_results_{group_col}.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to: {output_file}")
    
    return results_df
