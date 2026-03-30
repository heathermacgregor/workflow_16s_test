"""
Unified Differential Abundance Module.

Combines R-based compositional methods (DESeq2, ANCOM-BC, ALDEx2, LinDA, corncob)
with Python-based non-parametric tests (Wilcoxon/Mann-Whitney) into a single
interface.

Includes consensus voting to identify robust biomarkers across multiple methods.
"""

from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests

from workflow_16s.utils.logger import get_logger


def _to_dense_array(X) -> np.ndarray:
    """Convert various array types (sparse, dense, lazy-loaded) to dense numpy array."""
    if issparse(X):
        return X.toarray()
    elif hasattr(X, 'to_numpy'):
        # Handles pandas Series/DataFrame and some array-like types
        return X.to_numpy()
    elif hasattr(X, '__array__'):
        # Generic array protocol
        return np.asarray(X)
    else:
        # Already a numpy array or array-like
        return np.asarray(X)

def _check_r_package(package_name: str) -> bool:
    """Check if an R package is installed."""
    try:
        from rpy2.robjects.packages import importr
        importr(package_name)
        return True
    except Exception:
        return False

def run_wilcoxon(
    adata: ad.AnnData,
    group_col: str,
    alpha: float = 0.05,
    min_prevalence: float = 0.1,
    use_clr: bool = True
) -> pd.DataFrame:
    """
    Run Wilcoxon rank-sum (Mann-Whitney U) test.
    Non-parametric test, robust for compositional data if CLR transformed.
    """
    logger = get_logger("workflow_16s")
    logger.info(f"Running Wilcoxon test for group: {group_col}")
    
    # Get groups
    groups = adata.obs[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError(f"Wilcoxon test requires exactly 2 groups, found {len(groups)}")
    
    # Prepare data
    if use_clr:
        # Local CLR implementation to avoid circular imports
        counts = _to_dense_array(adata.X)
        pseudocount = 1.0
        gmeans = np.exp(np.mean(np.log(counts + pseudocount), axis=1, keepdims=True))
        data_transformed = pd.DataFrame(
            np.log((counts + pseudocount) / gmeans),
            index=adata.obs_names,
            columns=adata.var_names
        )
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
        g1_data = data_filt.loc[adata.obs[group_col] == groups[0], feature]
        g2_data = data_filt.loc[adata.obs[group_col] == groups[1], feature]
        
        try:
            stat, p_val = stats.mannwhitneyu(g1_data, g2_data, alternative='two-sided')
            
            # Calculate Effect Sizes if available
            cliffs = np.nan
            try:
                from workflow_16s.downstream.stats.effect_sizes import cliffs_delta
                cliffs = cliffs_delta(g1_data.values, g2_data.values)
            except Exception as e:
                logger.warning(f"Error calculating Cliff's delta for feature {feature}: {e}")
                cliffs = np.nan

            results.append({
                'feature': feature,
                'statistic': stat,
                'p_value': p_val,
                'cliffs_delta': cliffs,
                'median_group1': np.median(g1_data),
                'median_group2': np.median(g2_data)
            })
        except Exception as e:
            logger.warning(f"Error running Wilcoxon test for feature {feature}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    
    # FDR correction
    if not results_df.empty:
        _, p_adj, _, _ = multipletests(
            results_df['p_value'].fillna(1.0), 
            method='fdr_bh'
        )
        results_df['p_adj'] = p_adj
    else:
        results_df['p_adj'] = 1.0
        
    # Add taxonomy
    if 'Taxon' in adata.var.columns:
        results_df = results_df.merge(
            adata.var[['Taxon']], 
            left_on='feature', 
            right_index=True, 
            how='left'
        )
        
    return results_df.sort_values('p_adj')

def run_ancombc(
    adata: ad.AnnData,
    group_col: str,
    formula: Optional[str] = None,
    p_adj_method: str = 'fdr_bh',
    alpha: float = 0.05
) -> pd.DataFrame:
    """Wrapper for ANCOM-BC (Bias Correction)."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, conversion
    if not _check_r_package('ANCOMBC'):
        raise RuntimeError("ANCOMBC R package not available.")
    logger = get_logger("workflow_16s")
    logger.info(f"Running ANCOM-BC for group: {group_col}")
    
    # Prepare data
    counts = _to_dense_array(adata.X)
    count_df = pd.DataFrame(counts.T, index=adata.var_names, columns=adata.obs_names)
    metadata = adata.obs[[group_col]].copy()
    
    if formula is None: formula = f"~{group_col}"
    
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        r_counts = pandas2ri.py2rpy(count_df)
        r_meta = pandas2ri.py2rpy(metadata)
        
        ro.globalenv['counts_df'] = r_counts
        ro.globalenv['metadata_df'] = r_meta
        
        r_script = f"""
        library(phyloseq); library(ANCOMBC)
        otu_mat <- as.matrix(counts_df)
        sample_df <- sample_data(metadata_df)
        ps <- phyloseq(otu_table(otu_mat, taxa_are_rows = TRUE), sample_df)
        
        out <- ancombc(phyloseq = ps, formula = "{formula}", 
                       p_adj_method = "{p_adj_method}", alpha = {alpha}, 
                       global = TRUE)
        out$res
        """
        res_obj = ro.r(r_script)
        
        # Extract components from R object before conversion
        lfc = pandas2ri.rpy2py(ro.r('res_obj$lfc') if 'lfc' in dir(res_obj) else ro.r('data.frame()'))
        q_val = pandas2ri.rpy2py(ro.r('res_obj$q_val') if 'q_val' in dir(res_obj) else ro.r('data.frame()'))
        
    # Construct result DataFrame
    results = []
    for i, feature in enumerate(count_df.index):
        results.append({
            'feature': feature,
            'log_fold_change': lfc.iloc[i, 0] if not lfc.empty else 0,
            'p_adj': q_val.iloc[i, 0] if not q_val.empty else 1.0
        })
        
    return pd.DataFrame(results).sort_values('p_adj')

def run_deseq2(
    adata: ad.AnnData,
    group_col: str,
    alpha: float = 0.05
) -> pd.DataFrame:
    """Run DESeq2 differential abundance."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, conversion
    from rpy2.robjects.packages import importr
    if not _check_r_package('DESeq2'):
        raise RuntimeError("DESeq2 R package not available.")
        
    logger = get_logger("workflow_16s")
    logger.info(f"Running DESeq2 for group: {group_col}")
    
    counts = _to_dense_array(adata.X)
    counts = counts.T.astype(int)
    
    # Filter very low counts to help DESeq2
    keep = counts.sum(axis=1) >= 10
    counts = counts[keep]
    features = adata.var_names[keep]
    
    coldata = adata.obs[[group_col]].copy()
    coldata.columns = ['group']
    
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv['counts'] = pandas2ri.py2rpy(pd.DataFrame(counts, index=adata.var_names))
        ro.globalenv['coldata'] = pandas2ri.py2rpy(coldata)
        
        ro.r(f'''
        library(DESeq2)
        counts_matrix <- as.matrix(counts)
        dds <- DESeqDataSetFromMatrix(countData=counts_matrix, colData=coldata, design=~group)
        dds <- DESeq(dds, quiet=TRUE)
        res <- results(dds, alpha={alpha})
        res_df <- as.data.frame(res)
        ''')
        res_df = pandas2ri.rpy2py(ro.r('res_df'))
        
    res_df['feature'] = features
    res_df = res_df.rename(columns={'padj': 'p_adj', 'log2FoldChange': 'log2_fold_change'})
    return res_df.sort_values('p_adj')

def run_aldex2(
    adata: ad.AnnData,
    group_col: str,
    alpha: float = 0.05
) -> pd.DataFrame:
    """Run ALDEx2 (ANOVA-Like Differential Expression)."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, conversion
    from rpy2.robjects.packages import importr
    if not _check_r_package('ALDEx2'): raise RuntimeError("ALDEx2 not available.")
    
    logger = get_logger("workflow_16s")
    logger.info(f"Running ALDEx2 for group: {group_col}")
    
    counts = _to_dense_array(adata.X)
    counts = counts.T.astype(int)
    conds = adata.obs[group_col].astype(str).values
    
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        r_counts = pandas2ri.py2rpy(pd.DataFrame(counts, index=adata.var_names))
        ro.globalenv['counts'] = r_counts
        ro.globalenv['conds'] = ro.StrVector(conds)
        
        ro.r('''
        library(ALDEx2)
        x <- aldex.clr(counts, conds, mc.samples=128, denom="all", verbose=FALSE)
        xt <- aldex.ttest(x, verbose=FALSE)
        xe <- aldex.effect(x, verbose=FALSE)
        res <- data.frame(xt, xe)
        ''')
        res_df = pandas2ri.rpy2py(ro.r('res'))
        
    res_df['feature'] = adata.var_names
    res_df = res_df.rename(columns={
        'we.eBH': 'p_adj', 
        'diff.btw': 'effect_size'
    })
    return res_df.sort_values('p_adj')

def run_linda(adata: ad.AnnData, group_col: str, alpha: float = 0.05) -> pd.DataFrame:
    """Run LinDA (Linear Models for Differential Abundance)."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, conversion
    if not _check_r_package('MicrobiomeStat'): 
        raise RuntimeError("MicrobiomeStat/LinDA not available.")
    
    logger = get_logger("workflow_16s")
    logger.info(f"Running LinDA for group: {group_col}")
    
    counts = _to_dense_array(adata.X)
    otu = pd.DataFrame(counts.T, index=adata.var_names, columns=adata.obs_names)
    meta = adata.obs[[group_col]].copy()
    
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv['otu'] = pandas2ri.py2rpy(otu)
        ro.globalenv['meta'] = pandas2ri.py2rpy(meta)
        
        ro.r(f'''
        library(MicrobiomeStat)
        res <- linda(otu, meta, formula = '~{group_col}', alpha = {alpha})
        out <- res$output[[1]] # First variable results
        ''')
        res_df = pandas2ri.rpy2py(ro.r('out'))
        
    res_df['feature'] = res_df.index
    res_df = res_df.rename(columns={'padj': 'p_adj', 'log2FoldChange': 'log2_fold_change'})
    return res_df.sort_values('p_adj')

def run_corncob(adata: ad.AnnData, group_col: str, alpha: float = 0.05) -> pd.DataFrame:
    """
    Run corncob (Count Regression for Correlated Observations with the Beta-binomial).
    Good for detecting differential variability and abundance.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, conversion
    if not _check_r_package('corncob'): raise RuntimeError("corncob not available.")
    
    logger = get_logger("workflow_16s")
    logger.info(f"Running corncob for group: {group_col}")
    
    counts = _to_dense_array(adata.X)
    otu = pd.DataFrame(counts.T, index=adata.var_names, columns=adata.obs_names)
    meta = adata.obs[[group_col]].copy()
    
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv['otu'] = pandas2ri.py2rpy(otu)
        ro.globalenv['meta'] = pandas2ri.py2rpy(meta)
        
        ro.r(f'''
        library(phyloseq)
        library(corncob)
        
        otu_mat <- as.matrix(otu)
        sample_df <- sample_data(meta)
        ps <- phyloseq(otu_table(otu_mat, taxa_are_rows = TRUE), sample_df)
        
        # Run differential test on all taxa
        da_analysis <- differentialTest(formula = ~ {group_col},
                                        phi.formula = ~ {group_col},
                                        formula_null = ~ 1,
                                        phi.formula_null = ~ {group_col},
                                        test = "Wald", boot = FALSE,
                                        data = ps,
                                        fdr_cutoff = {alpha})
        
        # Extract results
        sig_taxa <- da_analysis$significant_taxa
        p_values <- da_analysis$p
        p_adj <- da_analysis$p_fdr
        
        res_df <- data.frame(
            feature = names(p_values),
            p_value = p_values,
            p_adj = p_adj
        )
        ''')
        res_df = pandas2ri.rpy2py(ro.r('res_df'))
        
    # Clean up results
    res_df = res_df.sort_values('p_adj')
    res_df['significant'] = res_df['p_adj'] < alpha
    return res_df


def compare_da_methods(
    adata: ad.AnnData,
    methods: List[str],
    group_col: str,
    alpha: float = 0.05,
    min_prevalence: float = 0.1,
    output_dir: Optional[Path] = None
) -> Dict:
    """Run multiple DA methods and compare results."""
    logger = get_logger("workflow_16s")
    func_map = {
        'wilcoxon': run_wilcoxon,
        'deseq2': run_deseq2,
        'ancombc': run_ancombc,
        'aldex2': run_aldex2,
        'linda': run_linda,
        'corncob': run_corncob
    }
    
    results = {}
    significant_features = {}
    
    for method in methods:
        if method not in func_map:
            logger.warning(f"Unknown method {method}, skipping.")
            continue
            
        try:
            # Run method
            if method == 'wilcoxon':
                res = func_map[method](adata, group_col, alpha, min_prevalence)
            else:
                res = func_map[method](adata, group_col, alpha=alpha)
                
            results[method] = res
            
            # Identify significant features
            sig = set(res[res['p_adj'] < alpha]['feature'])
            significant_features[method] = sig
            logger.info(f"{method}: {len(sig)} significant features found.")
            
        except Exception as e:
            logger.error(f"Method {method} failed: {e}")
            
    return {
        'results': results, 
        'significant_features': significant_features
    }

def consensus_da_features(
    comparison_results: Dict,
    min_methods: int = 2,
    max_p_adj: float = 0.05
) -> pd.DataFrame:
    """Extract consensus features found by multiple methods."""
    sig_sets = comparison_results['significant_features']
    if not sig_sets: return pd.DataFrame()
    
    all_features = set.union(*sig_sets.values())
    consensus_data = []
    
    for feat in all_features:
        # Which methods found this feature?
        found_by = [m for m, s in sig_sets.items() if feat in s]
        n_found = len(found_by)
        
        if n_found >= min_methods:
            consensus_data.append({
                'feature': feat,
                'n_methods': n_found,
                'methods': ", ".join(found_by)
            })
            
    return pd.DataFrame(consensus_data).sort_values('n_methods', ascending=False)