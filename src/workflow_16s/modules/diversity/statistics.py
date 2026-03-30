# ==================================================================================== #
# diversity/statistics.py
# ==================================================================================== #

import anndata as ad
import multiprocessing
import re
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

from workflow_16s.downstream.utils import AnalysisUtils
from workflow_16s.visualization.utils import PlottingUtils, DEFAULT_HEIGHT
from workflow_16s.utils.logger import get_logger


def _calculate_kruskal(item: Tuple[str, pd.Series], meta_vector_shared: pd.Series) -> Optional[Dict[str, Any]]:
    taxon, abund_series = item; plot_df = pd.DataFrame({'meta': meta_vector_shared, 'abund': abund_series}).dropna()
    if plot_df.shape[0] < 5: return None 
    groups = [d['abund'].values for _, d in plot_df.groupby('meta', observed=True) if not d.empty]
    if len(groups) < 2: return None
    try: stat, p_val = stats.kruskal(*groups); return {'taxon': taxon, 'statistic': stat, 'p_value': p_val}
    except: return None

def run_taxa_metadata_statistics(
    adata: ad.AnnData, 
    analysis_levels: List[str], 
    plot_dir_stats: Path, 
    n_cpus: int,
    max_taxa: int = 500,
    max_categorical: int = 20
):
    logger = get_logger("workflow_16s")
    plot_utils = PlottingUtils(logger)
    logger.info(f"--- Starting Taxa-Metadata Statistics (Using {n_cpus} CPUs) ---")
    if adata.n_obs < 5: return
    metadata = AnalysisUtils.find_plottable_metadata(adata, fullness_threshold=0.25, max_categories=50, admin_noise_columns=[])  # type: ignore
    cat_cols, num_cols = [c for c in metadata['categorical'] if c in adata.obs.columns], [c for c in metadata['numeric'] if c in adata.obs.columns]
    
    # Limit categorical columns to avoid overwhelming computation
    if len(cat_cols) > max_categorical:
        logger.warning(f"Too many categorical columns ({len(cat_cols)}). Limiting to {max_categorical} most complete.")
        # Select columns with highest completeness
        completeness = {col: adata.obs[col].notna().sum() / len(adata.obs) for col in cat_cols}
        cat_cols = sorted(cat_cols, key=lambda x: completeness[x], reverse=True)[:max_categorical]
    
    logger.info(f"Testing {len(cat_cols)} categorical and {len(num_cols)} numeric columns")
    
    for level in analysis_levels:
        logger.info(f"Processing taxonomy level: {level}")
        adata_agg = AnalysisUtils.get_analysis_adata(adata, level=level)
        if adata_agg is None or adata_agg.n_obs < 5: continue
        clr_df = AnalysisUtils._clr_transform(adata_agg, pseudocount=1)
        
        # Limit taxa if too many (sample top N by variance)
        if len(clr_df.columns) > max_taxa:
            logger.warning(f"Too many taxa ({len(clr_df.columns)}). Selecting top {max_taxa} by variance.")
            # Convert sparse columns to dense before computing variance
            if hasattr(clr_df, 'sparse'):
                variances = clr_df.sparse.to_dense().var(axis=0)
            else:
                # Handle case where columns might be individually sparse
                variances = pd.Series({col: clr_df[col].values.var() if hasattr(clr_df[col].values, 'var') else np.var(clr_df[col].to_numpy()) 
                                      for col in clr_df.columns})
            top_taxa = variances.nlargest(max_taxa).index
            clr_df = clr_df[top_taxa]
        
        for i, col in enumerate(cat_cols, 1):
            logger.info(f"  [{i}/{len(cat_cols)}] Testing categorical: {col}")
            meta_vec = adata_agg.obs[col].astype(str).fillna('Unknown')
            if meta_vec.nunique() < 2: continue
            
            # Use chunksize for better memory management
            items = list(clr_df.items())
            with multiprocessing.Pool(processes=n_cpus) as pool:
                res = [r for r in pool.imap_unordered(
                    partial(_calculate_kruskal, meta_vector_shared=meta_vec), 
                    items,
                    chunksize=max(1, len(items) // (n_cpus * 4))
                ) if r]
            
            if not res: continue
            res_df = pd.DataFrame(res); res_df['p_adj'] = multipletests(res_df['p_value'], method='fdr_bh')[1]
            sig_count = (res_df['p_adj'] < 0.05).sum()
            logger.info(f"    Found {sig_count} significant taxa for {col}")
            for _, row in res_df[res_df['p_adj'] < 0.05].sort_values('p_adj').head(10).iterrows():
                # plot_raincloud logic embedded here or called from internal helper...
                pass
        
        for i, col in enumerate(num_cols, 1):
            logger.info(f"  [{i}/{len(num_cols)}] Testing numeric: {col}")
            meta_vec = pd.to_numeric(adata_agg.obs[col], errors='coerce')
            if meta_vec.nunique() < 2: continue
            res_list = []
            for taxon, abund in clr_df.items():
                df_c = pd.DataFrame({'m': meta_vec, 'a': abund}).dropna()
                if df_c.shape[0] >= 5:
                    r, p = stats.spearmanr(df_c['m'], df_c['a'])
                    if not pd.isna(p): res_list.append({'taxon': taxon, 'correlation': r, 'p_value': p})  # type: ignore
            if not res_list: continue
            res_df = pd.DataFrame(res_list); res_df['p_adj'] = multipletests(res_df['p_value'], method='fdr_bh')[1]
            sig_count = (res_df['p_adj'] < 0.05).sum()
            logger.info(f"    Found {sig_count} significant taxa for {col}")
            for _, row in res_df[res_df['p_adj'] < 0.05].sort_values('p_adj').head(10).iterrows():
                # plot_correlation logic embedded here or called from internal helper...
                pass
    
    logger.info("Taxa-Metadata Statistics complete")
    plot_utils.flush_plot_queue()