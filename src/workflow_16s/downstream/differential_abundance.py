"""
Expanded Differential Abundance Methods for Microbiome Analysis.

This module provides multiple state-of-the-art methods for differential abundance
testing in microbiome data, along with a framework for comparing and combining results
from different methods.

Implemented Methods:
1. ANCOM-BC (Already implemented via R)
2. DESeq2 - RNA-seq method adapted for microbiome (via R)
3. corncob - Beta-binomial regression for count data (via R)
4. LinDA - Linear models with adaptive zero-handling (via R)
5. ALDEx2 - Compositional differential abundance (via R)
6. Wilcoxon/Mann-Whitney U - Non-parametric testing (Python)
7. PERMANOVA - Multivariate differential abundance

References:
    Lin H, Peddada SD. (2020). Analysis of compositions of microbiomes with bias
    correction. Nature Communications, 11(1), 3514. (ANCOM-BC)
    
    Love MI, Huber W, Anders S. (2014). Moderated estimation of fold change and
    dispersion for RNA-seq data with DESeq2. Genome Biology, 15(12), 550.
    
    Martin BD, Witten D, Willis AD. (2020). Modeling microbial abundances and
    dysbiosis with beta-binomial regression. Annals of Applied Statistics, 14(1), 94-115.
    (corncob)
    
    Zhou H, He K, Chen J, Zhang X. (2022). LinDA: linear models for differential
    abundance analysis of microbiome compositional data. Genome Biology, 23(1), 95.
    
    Fernandes AD, Reid JN, Macklaim JM, McMurrough TA, Edgell DR, Gloor GB. (2014).
    Unifying the analysis of high-throughput sequencing datasets: characterizing
    RNA-seq, 16S rRNA gene sequencing and selective growth experiments by
    compositional data analysis. Microbiome, 2, 15. (ALDEx2)

Example:
    >>> from workflow_16s.downstream.differential_abundance import (
    ...     run_deseq2, run_corncob, compare_da_methods
    ... )
    >>> 
    >>> # Run single method
    >>> deseq2_results = run_deseq2(adata, group_col='treatment')
    >>> 
    >>> # Compare multiple methods
    >>> comparison = compare_da_methods(
    ...     adata,
    ...     methods=['deseq2', 'corncob', 'ancom-bc'],
    ...     group_col='treatment'
    ... )
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

# Check for R and rpy2
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, conversion
    from rpy2.robjects.packages import importr
    
    # Use context manager instead of deprecated activate()
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False
    logger.warning("rpy2 not available. R-based DA methods will not work.")


def _check_r_package(package_name: str) -> bool:
    """Check if an R package is installed."""
    if not R_AVAILABLE:
        return False
    try:
        importr(package_name)
        return True
    except Exception:
        return False


def run_deseq2(
    adata: ad.AnnData,
    group_col: str,
    design_formula: Optional[str] = None,
    alpha: float = 0.05,
    lfc_threshold: float = 0,
    min_count: int = 10,
    fit_type: str = 'parametric'
) -> pd.DataFrame:
    """
    Run DESeq2 differential abundance analysis.
    
    DESeq2 was originally developed for RNA-seq but works well for microbiome data.
    It models count data with a negative binomial distribution and includes
    normalization and dispersion estimation.
    
    Args:
        adata: AnnData object with count data
        group_col: Column in adata.obs for group comparison
        design_formula: R formula (default: ~group_col)
        alpha: Significance threshold
        lfc_threshold: Log2 fold-change threshold for significance
        min_count: Minimum total count threshold for features
        fit_type: Dispersion fit type ('parametric', 'local', 'mean')
    
    Returns:
        DataFrame with differential abundance results
        
    Raises:
        RuntimeError: If R or DESeq2 is not available
    """
    if not _check_r_package('DESeq2'):
        raise RuntimeError(
            "DESeq2 R package not available. Install with:\n"
            "  R -e \"BiocManager::install('DESeq2')\""
        )
    
    logger.info(f"Running DESeq2 analysis for group: {group_col}")
    
    # Prepare count matrix
    counts = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    counts = counts.T.astype(int)  # Features x samples, integer counts
    
    # Filter low-count features
    keep_features = counts.sum(axis=1) >= min_count
    counts_filt = counts[keep_features]
    feature_names = adata.var_names[keep_features]
    
    logger.info(f"Filtered {(~keep_features).sum()} low-count features (< {min_count} total counts)")
    
    # Prepare metadata
    coldata = adata.obs[[group_col]].copy()
    coldata.columns = ['group']
    
    # Use context manager for R conversions
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        # Convert to R objects
        r_counts = ro.r.matrix(
            ro.IntVector(counts_filt.flatten()),
            nrow=counts_filt.shape[0],
            ncol=counts_filt.shape[1]
        )
        ro.r.assign('counts', r_counts)
        ro.r.assign('rownames(counts)', ro.StrVector(feature_names))
        ro.r.assign('colnames(counts)', ro.StrVector(adata.obs_names))
        
        r_coldata = pandas2ri.py2rpy(coldata)
        ro.r.assign('coldata', r_coldata)
        
        # Set design formula
        if design_formula is None:
            design_formula = "~ group"
        
        # Run DESeq2
        logger.debug(f"Running DESeq2 with design: {design_formula}, fit_type: {fit_type}")
        
        ro.r(f'''
        library(DESeq2)
        dds <- DESeqDataSetFromMatrix(
            countData = counts,
            colData = coldata,
            design = {design_formula}
        )
        dds <- DESeq(dds, fitType='{fit_type}', quiet=TRUE)
        res <- results(dds, alpha={alpha}, lfcThreshold={lfc_threshold})
        res_df <- as.data.frame(res)
        res_df$feature <- rownames(res_df)
        ''')
        
        # Get results
        results = pandas2ri.rpy2py(ro.r('res_df'))
    
    # Clean up results
    results = results.rename(columns={
        'baseMean': 'mean_abundance',
        'log2FoldChange': 'log2_fold_change',
        'lfcSE': 'lfc_se',
        'stat': 'wald_statistic',
        'pvalue': 'p_value',
        'padj': 'p_adj'
    })
    
    # Add taxonomy if available
    if 'taxonomy' in adata.var.columns:
        tax_dict = dict(zip(adata.var_names, adata.var['taxonomy']))
        results['taxonomy'] = results['feature'].map(tax_dict)
    
    # Sort by adjusted p-value
    results = results.sort_values('p_adj')
    
    # Log summary
    n_sig = (results['p_adj'] < alpha).sum()
    logger.info(f"DESeq2 identified {n_sig}/{len(results)} significant features at alpha={alpha}")
    
    return results


def run_corncob(
    adata: ad.AnnData,
    group_col: str,
    formula: Optional[str] = None,
    phi_formula: Optional[str] = None,
    fdr_cutoff: float = 0.05,
    min_count: int = 10
) -> pd.DataFrame:
    """
    Run corncob beta-binomial differential abundance analysis.
    
    corncob models microbiome count data using a beta-binomial distribution,
    which accounts for both overdispersion and compositionality.
    
    Args:
        adata: AnnData object with count data
        group_col: Column in adata.obs for group comparison
        formula: R formula for mean model (default: ~group_col)
        phi_formula: R formula for dispersion model (default: ~1)
        fdr_cutoff: FDR threshold for significance
        min_count: Minimum total count threshold
    
    Returns:
        DataFrame with differential abundance results
        
    Raises:
        RuntimeError: If R or corncob is not available
    """
    if not _check_r_package('corncob'):
        raise RuntimeError(
            "corncob R package not available. Install with:\n"
            "  R -e \"devtools::install_github('bryandmartin/corncob')\""
        )
    
    logger.info(f"Running corncob analysis for group: {group_col}")
    
    # Prepare data
    counts = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    counts = counts.T.astype(int)
    
    # Filter low-count features
    keep_features = counts.sum(axis=1) >= min_count
    counts_filt = counts[keep_features]
    feature_names = adata.var_names[keep_features]
    
    # Total library sizes
    total_counts = counts.sum(axis=0)
    
    # Prepare metadata
    sample_data = adata.obs[[group_col]].copy()
    sample_data.columns = ['group']
    
    # Use context manager for R conversions
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        # Convert to R
        r_counts = ro.r.matrix(
            ro.IntVector(counts_filt.flatten()),
            nrow=counts_filt.shape[0],
            ncol=counts_filt.shape[1]
        )
        ro.r.assign('W', r_counts)
        ro.r.assign('M', ro.IntVector(total_counts))
        ro.r.assign('rownames(W)', ro.StrVector(feature_names))
        
        r_sample_data = pandas2ri.py2rpy(sample_data)
        ro.r.assign('sample_data', r_sample_data)
        
        # Set formulas
        if formula is None:
            formula = "~ group"
        if phi_formula is None:
            phi_formula = "~ 1"
        
        # Run corncob
        logger.debug(f"Running corncob with formula={formula}, phi_formula={phi_formula}")
        
        ro.r(f'''
        library(corncob)
        results_list <- list()
        
        for (i in 1:nrow(W)) {{
            tryCatch({{
                mod <- bbdml(
                    formula = cbind(W[i,], M - W[i,]) {formula},
                    phi.formula = {phi_formula},
                    data = sample_data,
                    link = "logit",
                    phi.link = "logit"
                )
                
                # Extract coefficient and p-value
                coef_summary <- summary(mod)$coefficients
                group_coef <- coef_summary["groupgroup2", ]  # Assumes binary group
                
                results_list[[i]] <- data.frame(
                    feature = rownames(W)[i],
                    estimate = group_coef["Estimate"],
                    std_error = group_coef["Std. Error"],
                    t_value = group_coef["t value"],
                    p_value = group_coef["Pr(>|t|)"],
                    stringsAsFactors = FALSE
                )
            }}, error = function(e) {{
                results_list[[i]] <- data.frame(
                    feature = rownames(W)[i],
                    estimate = NA,
                    std_error = NA,
                    t_value = NA,
                    p_value = 1.0,
                    stringsAsFactors = FALSE
                )
            }})
        }}
        
        results_df <- do.call(rbind, results_list)
        results_df$p_adj <- p.adjust(results_df$p_value, method="BH")
        ''')
        
        # Get results
        results = pandas2ri.rpy2py(ro.r('results_df'))
    
    # Clean and sort
    results = results.dropna(subset=['p_value'])
    results = results.sort_values('p_adj')
    
    # Add taxonomy if available
    if 'taxonomy' in adata.var.columns:
        tax_dict = dict(zip(adata.var_names, adata.var['taxonomy']))
        results['taxonomy'] = results['feature'].map(tax_dict)
    
    # Log summary
    n_sig = (results['p_adj'] < fdr_cutoff).sum()
    logger.info(f"corncob identified {n_sig}/{len(results)} significant features at FDR={fdr_cutoff}")
    
    return results


def run_linda(
    adata: ad.AnnData,
    group_col: str,
    formula: Optional[str] = None,
    alpha: float = 0.05,
    adaptive: bool = True,
    max_abund_for_win: float = 0.1
) -> pd.DataFrame:
    """
    Run LinDA (Linear models for Differential Abundance analysis).
    
    LinDA uses linear models with adaptive zero-handling, making it robust
    to different data characteristics.
    
    Args:
        adata: AnnData object with count data
        group_col: Column in adata.obs for group comparison
        formula: R formula (default: ~group_col)
        alpha: Significance threshold
        adaptive: Use adaptive zero-handling
        max_abund_for_win: Maximum abundance for winsorization
    
    Returns:
        DataFrame with differential abundance results
        
    Raises:
        RuntimeError: If R or LinDA is not available
    """
    if not _check_r_package('MicrobiomeStat'):
        raise RuntimeError(
            "MicrobiomeStat R package (with LinDA) not available. Install with:\n"
            "  R -e \"devtools::install_github('cafferychen777/MicrobiomeStat')\""
        )
    
    logger.info(f"Running LinDA analysis for group: {group_col}")
    
    # Prepare count data
    counts = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    otu_table = pd.DataFrame(
        counts.T,
        index=adata.var_names,
        columns=adata.obs_names
    )
    
    # Prepare metadata
    meta_data = adata.obs[[group_col]].copy()
    
    # Use context manager for R conversions
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        # Convert to R
        r_otu = pandas2ri.py2rpy(otu_table)
        r_meta = pandas2ri.py2rpy(meta_data)
        
        ro.r.assign('otu_table', r_otu)
        ro.r.assign('meta_data', r_meta)
        
        # Set formula
        if formula is None:
            formula = f"~ {group_col}"
        
        # Run LinDA
        logger.debug(f"Running LinDA with formula={formula}, adaptive={adaptive}")
        
        ro.r(f'''
        library(MicrobiomeStat)
        
        linda_res <- linda(
            otu.tab = otu_table,
            meta = meta_data,
            formula = '{formula}',
            alpha = {alpha},
            adaptive = {'TRUE' if adaptive else 'FALSE'},
            max.abund.for.win = {max_abund_for_win}
        )
        
        # Extract results for first variable (assumes binary comparison)
        var_name <- names(linda_res$output)[1]
        results_df <- linda_res$output[[var_name]]
        results_df$feature <- rownames(results_df)
        ''')
        
        # Get results
        results = pandas2ri.rpy2py(ro.r('results_df'))
    
    # Rename columns
    results = results.rename(columns={
        'log2FoldChange': 'log2_fold_change',
        'lfcSE': 'lfc_se',
        'stat': 'statistic',
        'pvalue': 'p_value',
        'padj': 'p_adj',
        'reject': 'significant',
        'df': 'degrees_freedom'
    })
    
    # Add taxonomy if available
    if 'taxonomy' in adata.var.columns:
        tax_dict = dict(zip(adata.var_names, adata.var['taxonomy']))
        results['taxonomy'] = results['feature'].map(tax_dict)
    
    # Sort by adjusted p-value
    results = results.sort_values('p_adj')
    
    # Log summary
    n_sig = (results['p_adj'] < alpha).sum()
    logger.info(f"LinDA identified {n_sig}/{len(results)} significant features at alpha={alpha}")
    
    return results


def run_aldex2(
    adata: ad.AnnData,
    group_col: str,
    test: str = 'welch',
    mc_samples: int = 128,
    denom: str = 'all',
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Run ALDEx2 compositional differential abundance analysis.
    
    ALDEx2 uses Monte Carlo sampling from the Dirichlet distribution to
    account for compositionality and sampling variation.
    
    Args:
        adata: AnnData object with count data
        group_col: Column in adata.obs for group comparison
        test: Statistical test ('welch', 'wilcox', 'kw')
        mc_samples: Number of Monte Carlo samples
        denom: Denominator for CLR ('all', 'iqlr', 'zero', 'lvha')
        alpha: Significance threshold
    
    Returns:
        DataFrame with differential abundance results
        
    Raises:
        RuntimeError: If R or ALDEx2 is not available
    """
    if not _check_r_package('ALDEx2'):
        raise RuntimeError(
            "ALDEx2 R package not available. Install with:\n"
            "  R -e \"BiocManager::install('ALDEx2')\""
        )
    
    logger.info(f"Running ALDEx2 analysis for group: {group_col}")
    
    # Prepare count data
    counts = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    otu_table = pd.DataFrame(
        counts.T.astype(int),
        index=adata.var_names,
        columns=adata.obs_names
    )
    
    # Prepare group vector
    conditions = adata.obs[group_col].values
    
    # Use context manager for R conversions
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        # Convert to R
        r_counts = pandas2ri.py2rpy(otu_table)
        r_conditions = ro.StrVector(conditions)
        
        ro.r.assign('counts', r_counts)
        ro.r.assign('conditions', r_conditions)
        
        # Run ALDEx2
        logger.debug(f"Running ALDEx2 with test={test}, mc_samples={mc_samples}, denom={denom}")
        
        ro.r(f'''
        library(ALDEx2)
        
        aldex_clr <- aldex.clr(
            counts,
            conditions,
            mc.samples = {mc_samples},
            denom = "{denom}",
            verbose = FALSE
        )
        
        aldex_test <- aldex.{test}(aldex_clr, verbose = FALSE)
        aldex_effect <- aldex.effect(aldex_clr, verbose = FALSE)
        
        # Combine results
        results_df <- data.frame(
            feature = rownames(aldex_test),
            we.ep = aldex_test$we.ep,  # Expected p-value (Welch's t-test)
            we.eBH = aldex_test$we.eBH,  # BH-corrected p-value
            wi.ep = aldex_test$wi.ep,  # Wilcoxon p-value
            wi.eBH = aldex_test$wi.eBH,  # Wilcoxon BH-corrected
            effect = aldex_effect$effect,  # Effect size
            overlap = aldex_effect$overlap,  # Proportion of MC samples with overlap
            diff.btw = aldex_effect$diff.btw,  # Between-group difference
            diff.win = aldex_effect$diff.win,  # Within-group difference
            stringsAsFactors = FALSE
        )
        ''')
        
        # Get results
        results = pandas2ri.rpy2py(ro.r('results_df'))
    
    # Select appropriate p-values based on test
    if test == 'welch':
        results['p_value'] = results['we.ep']
        results['p_adj'] = results['we.eBH']
    elif test == 'wilcox':
        results['p_value'] = results['wi.ep']
        results['p_adj'] = results['wi.eBH']
    
    # Add taxonomy if available
    if 'taxonomy' in adata.var.columns:
        tax_dict = dict(zip(adata.var_names, adata.var['taxonomy']))
        results['taxonomy'] = results['feature'].map(tax_dict)
    
    # Sort by adjusted p-value
    results = results.sort_values('p_adj')
    
    # Log summary
    n_sig = (results['p_adj'] < alpha).sum()
    logger.info(f"ALDEx2 identified {n_sig}/{len(results)} significant features at alpha={alpha}")
    
    return results


def run_wilcoxon(
    adata: ad.AnnData,
    group_col: str,
    alpha: float = 0.05,
    min_prevalence: float = 0.1,
    use_clr: bool = True
) -> pd.DataFrame:
    """
    Run Wilcoxon rank-sum (Mann-Whitney U) test.
    
    Non-parametric test that doesn't assume normal distributions.
    Recommended for compositional data when using CLR-transformed values.
    
    Args:
        adata: AnnData object
        group_col: Column in adata.obs for group comparison
        alpha: Significance threshold
        min_prevalence: Minimum prevalence threshold
        use_clr: Whether to CLR-transform data first
    
    Returns:
        DataFrame with test results
    """
    from workflow_16s.utils.compositional import clr_table
    
    logger.info(f"Running Wilcoxon test for group: {group_col}")
    
    # Get groups
    groups = adata.obs[group_col].unique()
    if len(groups) != 2:
        raise ValueError(f"Wilcoxon test requires exactly 2 groups, found {len(groups)}")
    
    # Prepare data
    if use_clr:
        data_transformed = clr_table(adata.to_df())
    else:
        data_transformed = adata.to_df()
    
    # Filter by prevalence
    prevalence = (adata.to_df() > 0).mean()
    keep_features = prevalence >= min_prevalence
    data_filt = data_transformed.loc[:, keep_features]
    
    logger.info(f"Testing {keep_features.sum()} features (prevalence >= {min_prevalence})")
    
    # Run tests
    results = []
    for feature in data_filt.columns:
        group1_data = data_filt.loc[adata.obs[group_col] == groups[0], feature]
        group2_data = data_filt.loc[adata.obs[group_col] == groups[1], feature]
        
        # Wilcoxon rank-sum test
        statistic, p_value = stats.mannwhitneyu(
            group1_data, group2_data, alternative='two-sided'
        )
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(group1_data), len(group2_data)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        # Median difference
        median_diff = np.median(group2_data) - np.median(group1_data)
        
        results.append({
            'feature': feature,
            'statistic': statistic,
            'p_value': p_value,
            'effect_size_r': effect_size,
            'median_diff': median_diff,
            'median_group1': np.median(group1_data),
            'median_group2': np.median(group2_data),
            'prevalence': prevalence[feature]
        })
    
    results_df = pd.DataFrame(results)
    
    # FDR correction
    _, p_adj, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
    results_df['p_adj'] = p_adj
    
    # Add taxonomy if available
    if 'taxonomy' in adata.var.columns:
        tax_dict = dict(zip(adata.var_names, adata.var['taxonomy']))
        results_df['taxonomy'] = results_df['feature'].map(tax_dict)
    
    # Sort by adjusted p-value
    results_df = results_df.sort_values('p_adj')
    
    # Log summary
    n_sig = (results_df['p_adj'] < alpha).sum()
    logger.info(f"Wilcoxon test identified {n_sig}/{len(results_df)} significant features at alpha={alpha}")
    
    return results_df


def compare_da_methods(
    adata: ad.AnnData,
    methods: List[str],
    group_col: str,
    alpha: float = 0.05,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Compare results from multiple differential abundance methods.
    
    Args:
        adata: AnnData object
        methods: List of methods to compare (e.g., ['deseq2', 'wilcoxon', 'aldex2'])
        group_col: Column for group comparison
        alpha: Significance threshold
        output_dir: Optional directory for output plots
    
    Returns:
        Dictionary with:
            - 'results': Dict of DataFrames from each method
            - 'consensus': DataFrame of features significant in multiple methods
            - 'comparison': Comparison statistics
            - 'venn_data': Data for Venn diagram
    """
    logger.info("="*60)
    logger.info(f"COMPARING {len(methods)} DIFFERENTIAL ABUNDANCE METHODS")
    logger.info("="*60)
    
    # Map method names to functions
    method_functions = {
        'deseq2': run_deseq2,
        'corncob': run_corncob,
        'linda': run_linda,
        'aldex2': run_aldex2,
        'wilcoxon': run_wilcoxon
    }
    
    # Run each method
    results = {}
    significant_features = {}
    
    for method in methods:
        if method not in method_functions:
            logger.warning(f"Unknown method: {method}. Skipping.")
            continue
        
        logger.info(f"\nRunning {method}...")
        try:
            method_results = method_functions[method](adata, group_col, alpha=alpha)
            results[method] = method_results
            
            # Extract significant features
            sig_features = set(
                method_results[method_results['p_adj'] < alpha]['feature']
            )
            significant_features[method] = sig_features
            
            logger.info(f"{method}: {len(sig_features)} significant features")
            
        except Exception as e:
            logger.error(f"Error running {method}: {e}")
            continue
    
    # Find consensus features
    all_features = set.union(*significant_features.values()) if significant_features else set()
    
    consensus_data = []
    for feature in all_features:
        n_methods = sum(1 for sig_set in significant_features.values() if feature in sig_set)
        methods_list = [m for m, sig_set in significant_features.items() if feature in sig_set]
        
        # Collect effect sizes and p-values from each method
        effect_sizes = {}
        p_values = {}
        for method in methods_list:
            df = results[method]
            row = df[df['feature'] == feature].iloc[0]
            
            # Get effect size (method-specific column names)
            if 'log2_fold_change' in row:
                effect_sizes[method] = row['log2_fold_change']
            elif 'effect' in row:
                effect_sizes[method] = row['effect']
            elif 'estimate' in row:
                effect_sizes[method] = row['estimate']
            elif 'median_diff' in row:
                effect_sizes[method] = row['median_diff']
            
            p_values[method] = row['p_adj']
        
        consensus_data.append({
            'feature': feature,
            'n_methods': n_methods,
            'methods': ', '.join(methods_list),
            'mean_effect_size': np.mean(list(effect_sizes.values())),
            'min_p_adj': min(p_values.values()),
            'max_p_adj': max(p_values.values())
        })
    
    consensus_df = pd.DataFrame(consensus_data)
    consensus_df = consensus_df.sort_values(['n_methods', 'min_p_adj'], ascending=[False, True])
    
    # Comparison statistics
    comparison = {
        'total_features_tested': len(adata.var_names),
        'methods_run': list(results.keys()),
        'significant_per_method': {m: len(s) for m, s in significant_features.items()},
        'consensus_features': {
            'all_methods': len([f for f in consensus_data if f['n_methods'] == len(methods)]),
            'majority': len([f for f in consensus_data if f['n_methods'] > len(methods) / 2]),
            'any_method': len(all_features)
        }
    }
    
    # Log summary
    logger.info("\n" + "="*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*60)
    logger.info(f"Methods compared: {', '.join(results.keys())}")
    logger.info(f"Features significant in ALL methods: {comparison['consensus_features']['all_methods']}")
    logger.info(f"Features significant in MAJORITY: {comparison['consensus_features']['majority']}")
    logger.info(f"Features significant in ANY method: {comparison['consensus_features']['any_method']}")
    logger.info("="*60)
    
    # Save results
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual method results
        for method, df in results.items():
            df.to_csv(output_dir / f'{method}_results.csv', index=False)
        
        # Save consensus
        consensus_df.to_csv(output_dir / 'consensus_features.csv', index=False)
        
        # Save comparison summary
        import json
        with open(output_dir / 'comparison_summary.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    return {
        'results': results,
        'consensus': consensus_df,
        'comparison': comparison,
        'significant_features': significant_features
    }


def plot_da_comparison(
    comparison_results: Dict,
    output_path: Optional[Path] = None
) -> go.Figure:
    """
    Create visualization comparing differential abundance methods.
    
    Args:
        comparison_results: Output from compare_da_methods()
        output_path: Optional path to save plot
    
    Returns:
        Plotly figure object
    """
    significant_features = comparison_results['significant_features']
    methods = list(significant_features.keys())
    
    # Create Venn-like bar plot showing overlap
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Significant Features per Method', 'Feature Overlap'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Left panel: Number of significant features
    n_sig = [len(sig_set) for sig_set in significant_features.values()]
    fig.add_trace(
        go.Bar(
            x=methods,
            y=n_sig,
            text=n_sig,
            textposition='outside',
            marker_color='steelblue'
        ),
        row=1, col=1
    )
    
    # Right panel: Overlap counts
    consensus_df = comparison_results['consensus']
    overlap_counts = consensus_df['n_methods'].value_counts().sort_index()
    
    fig.add_trace(
        go.Bar(
            x=[f'{n} methods' for n in overlap_counts.index],
            y=overlap_counts.values,
            text=overlap_counts.values,
            textposition='outside',
            marker_color='coral'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Differential Abundance Method Comparison',
        showlegend=False,
        height=500,
        width=1000,
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text='Method', row=1, col=1)
    fig.update_xaxes(title_text='Agreement', row=1, col=2)
    fig.update_yaxes(title_text='Number of Features', row=1, col=1)
    fig.update_yaxes(title_text='Number of Features', row=1, col=2)
    
    if output_path is not None:
        fig.write_html(output_path)
        logger.info(f"Comparison plot saved to {output_path}")
    
    return fig


def consensus_da_features(
    comparison_results: Dict,
    min_methods: int = 2,
    max_p_adj: float = 0.05
) -> pd.DataFrame:
    """
    Extract consensus differential abundance features.
    
    Args:
        comparison_results: Output from compare_da_methods()
        min_methods: Minimum number of methods that must agree
        max_p_adj: Maximum adjusted p-value threshold
    
    Returns:
        DataFrame with consensus features
    """
    consensus_df = comparison_results['consensus']
    
    # Filter by criteria
    consensus_features = consensus_df[
        (consensus_df['n_methods'] >= min_methods) &
        (consensus_df['min_p_adj'] <= max_p_adj)
    ]
    
    logger.info(
        f"Found {len(consensus_features)} consensus features "
        f"(min_methods={min_methods}, max_p_adj={max_p_adj})"
    )
    
    return consensus_features
