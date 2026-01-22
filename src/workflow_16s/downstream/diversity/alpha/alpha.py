# ==================================================================================== #
#                           downstream/diversity/alpha.py
# ==================================================================================== #

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import scanpy as sc
import scipy.stats as stats
from scipy.sparse import issparse, csr_matrix
from skbio.diversity import alpha_diversity
from skbio.tree import TreeNode
import anndata as ad

from workflow_16s.downstream.steps.preprocessing import AnalysisUtils
from workflow_16s.downstream.visualization import PlottingUtils, DEFAULT_HEIGHT
from workflow_16s.utils.logger import get_logger

logger = get_logger("workflow_16s")
plot_utils = PlottingUtils(logger)

FACILITY_SHAPE_COLS = {'facility_capacity', 'facility_start_year', 'facility_end_year', 'facility_type', 'facility'}

# --- OPTIMIZED CALCULATION FUNCTIONS (SPARSE MATH) ---

def _calc_shannon_sparse(X):
    """
    Memory-efficient Shannon calculation for sparse matrices.
    Avoids densifying the matrix (which causes 700GB+ RAM usage).
    H = -sum(p * ln(p))
    """
    if not issparse(X):
        X = csr_matrix(X)
    
    # 1. Calculate relative abundances (p)
    # Sum along rows to get read depth per sample
    depths = np.array(X.sum(axis=1)).flatten()
    
    # Handle empty samples to avoid divide-by-zero
    depths[depths == 0] = 1.0
    
    # Efficient sparse multiplication to normalize: (1/depth) * Count
    from scipy.sparse import diags
    norm = diags(1.0 / depths)
    P = norm @ X  # P is sparse relative abundance

    # 2. Calculate p * ln(p) only for non-zero elements
    # Work directly with the linear data array to stay sparse/efficient
    P_data = P.data
    # Filter potential zeros (though sparse shouldn't have them)
    valid_mask = P_data > 0
    P_valid = P_data[valid_mask]
    
    # H_data = p * ln(p)
    H_data = P_valid * np.log(P_valid)
    
    # 3. Sum rows (reconstructing sparse structure temporarily is safest)
    # We map the calculated entropy bits back to their matrix positions
    H_matrix = csr_matrix((H_data, P.indices[valid_mask], P.indptr), shape=P.shape)
    
    # Sum along rows and negate
    shannon_values = -np.array(H_matrix.sum(axis=1)).flatten()
    return shannon_values

def _calc_observed_features_sparse(X):
    """Simple count of non-zero elements per row."""
    if not issparse(X):
        return np.count_nonzero(X, axis=1)
    # For CSR, getting nnz per row is instant
    return np.array((X > 0).sum(axis=1)).flatten()

# --- MAIN FUNCTION ---

def run_alpha_diversity(adata: ad.AnnData, plot_dir_alpha: Path, tree_path: Optional[Path] = None, priority_categorical: Optional[List[str]] = None, priority_numeric: Optional[List[str]] = None):
    """
    Calculate and plot alpha diversity metrics against metadata.
    Optimized for large sparse datasets (fixes OOM) and duplicate indices (fixes Crash).
    """
    
    # ------------------------------------------------------------------
    # [MANUAL OVERRIDE] STRICTLY DISABLE
    # ------------------------------------------------------------------
    # This ensures the function exits immediately, overriding any calling logic.
    # To re-enable in the future, remove or comment out these two lines.
    logger.info("🛑 Alpha Diversity is manually disabled in the script. Skipping...")
    return 
    # ------------------------------------------------------------------

    logger.info("--- Starting Alpha Diversity Analysis ---")

    # CRITICAL FIX: Ensure unique indices to prevent "duplicate labels" crash in pandas
    if not adata.obs_names.is_unique:
        logger.warning("⚠️ Duplicate sample IDs found! Making indices unique (appending -1, -2, etc.) to prevent crashes.")
        adata.obs_names_make_unique()
    
    # 1. Setup Data Matrix
    # We prioritize adata.X to ensure we are using the CURRENT filtered cells,
    # avoiding the 65k vs 37k mismatch from unsliced adata.raw.
    if 'raw_counts' in adata.layers:
        logger.info("Using 'raw_counts' layer.")
        raw_counts = adata.layers['raw_counts']
    else:
        logger.warning("'raw_counts' layer not found. Using adata.X.")
        raw_counts = adata.X

    if adata.n_obs == 0: 
        logger.warning("No samples remaining for alpha diversity."); return
    
    # Ensure Sparse (CSR is fastest for row operations)
    if issparse(raw_counts): 
        X = csr_matrix(raw_counts)
    else: 
        logger.info("Converting dense counts to sparse for efficiency...")
        X = csr_matrix(raw_counts)
    
    sample_ids = adata.obs_names.tolist()
    feature_ids = adata.var_names.tolist()
    metrics_to_plot = []
    
    # Dictionary to hold results
    results_dict = {}

    # 2. Calculate Metrics (Sparse)
    try:
        # --- Observed Features ---
        logger.info("Calculating Observed Features...")
        results_dict['observed_features'] = _calc_observed_features_sparse(X)
        metrics_to_plot.append('observed_features')

        # --- Shannon ---
        logger.info("Calculating Shannon Entropy (Sparse Optimized)...")
        results_dict['shannon'] = _calc_shannon_sparse(X)
        metrics_to_plot.append('shannon')

        # --- Evenness (Pielou) ---
        H = results_dict['shannon']
        S = results_dict['observed_features']
        
        # Avoid div/0 errors
        denom = np.log(S)
        denom[denom == 0] = 1.0 
        results_dict['pielou_evenness'] = H / denom
        metrics_to_plot.append('pielou_evenness')

    except Exception as e:
        logger.error(f"Basic alpha diversity calculation failed: {e}")

    # 3. Faith's PD (Requires Tree + Dense Subset)
    if tree_path:
        if not tree_path.exists(): 
            logger.error(f"Tree file not found at: {tree_path}. Skipping Faith's PD.")
        else:
            try:
                logger.info(f"Loading phylogenetic tree from: {tree_path}")
                tree = TreeNode.read(str(tree_path))
                tree_tips = {tip.name for tip in tree.tips()}
                
                features_in_tree = [f_id for f_id in feature_ids if f_id in tree_tips]
                
                if not features_in_tree:
                    logger.error("No features from data found in the tree. Skipping Faith's PD.")
                else:
                    feature_idx = [feature_ids.index(f_id) for f_id in features_in_tree]
                    
                    # Subset matrix to only tree features (reduces columns from 3M -> ~10-50k)
                    X_tree = X[:, feature_idx]
                    
                    logger.info(f"Calculating Faith's PD for {len(features_in_tree)} tree-aligned features...")
                    
                    # Only compute for samples that actually have counts in the tree subset
                    tree_sample_sums = np.array(X_tree.sum(axis=1)).flatten()
                    valid_mask = tree_sample_sums > 0
                    valid_indices = np.where(valid_mask)[0]
                    valid_ids = [sample_ids[i] for i in valid_indices]
                    
                    if len(valid_ids) > 0:
                        # Safety check for dense conversion
                        n_cells = len(valid_ids) * len(features_in_tree)
                        if n_cells > 500_000_000: # approx 4GB int64 limit
                            logger.warning(f"Faith's PD matrix too large ({n_cells} cells). Skipping to prevent OOM.")
                        else:
                            X_tree_dense = X_tree[valid_indices, :].toarray().astype(int)
                            
                            faith_pd_values = alpha_diversity(
                                'faith_pd', 
                                X_tree_dense, 
                                ids=valid_ids, 
                                tree=tree, 
                                otu_ids=features_in_tree
                            )
                            
                            # Align back to full sample list using Series
                            faith_series = pd.Series(faith_pd_values, index=valid_ids)
                            results_dict['faith_pd'] = faith_series.reindex(sample_ids)
                            metrics_to_plot.append('faith_pd')
                            logger.info(f"Faith's PD computed for {len(valid_ids)} samples.")
                    else:
                        logger.warning("No samples have counts matching the tree. Skipping Faith's PD.")

            except MemoryError:
                logger.error("OOM calculating Faith's PD. Skipped.")
            except Exception as e:
                logger.error(f"Faith's PD calculation failed: {e}")
    else:
        logger.info("No tree_path provided, skipping Faith's PD.")

    # 4. Safe Merge (Fixes Index Mismatch Crash)
    if results_dict:
        # Create DataFrame aligned to CURRENT sample_ids
        alpha_df = pd.DataFrame(results_dict, index=sample_ids)
        
        logger.info(f"Merging {alpha_df.shape[1]} metrics into metadata...")
        
        # Use direct assignment which respects index alignment automatically
        for col in alpha_df.columns:
            adata.obs[col] = alpha_df[col]
            
        logger.info(f"Alpha metrics added: {', '.join(alpha_df.columns)}")
    else:
        logger.warning("No metrics calculated.")
        return

    # 5. Plotting (Existing Logic Preserved)
    logger.info("Plotting alpha diversity...")
    metadata_cols = AnalysisUtils.find_plottable_metadata(adata, admin_noise_columns=None, fullness_threshold=0.25, max_categories=50)
    cat_cols = metadata_cols['categorical']
    num_cols = metadata_cols['numeric']
    
    all_cols_to_plot = (priority_categorical or []) + (priority_numeric or []) + cat_cols + num_cols
    all_cols_to_plot = list(dict.fromkeys(all_cols_to_plot))
    all_cols_to_plot = [c for c in all_cols_to_plot if c in adata.obs.columns]
    
    for metric in metrics_to_plot:
        if metric not in adata.obs.columns or adata.obs[metric].isnull().all(): 
            logger.warning(f"Skipping plots for '{metric}' (no valid data)."); continue 
            
        for col in all_cols_to_plot:
            if col == metric: continue
            
            plot_df = adata.obs[[col, metric]].copy().replace(pd.NA, np.nan).dropna(subset=[metric])
            if plot_df.shape[0] < 2: continue
            
            fig = None
            plot_width = 1200
            plot_kwargs = {}
            hover_dict = {plot_df.index.name or 'SampleID': plot_df.index}
            stat_text = ""
            
            # Handle Facility Match Coloring
            if 'facility_match' in adata.obs.columns:
                # SAFE ACCESS using loc to handle any potential (though now fixed) alignment issues
                fm = adata.obs.loc[plot_df.index, 'facility_match']
                if isinstance(fm.dtype, pd.CategoricalDtype):
                    if 'Unknown' not in fm.cat.categories: fm = fm.cat.add_categories('Unknown')
                    plot_df['facility_match'] = fm.fillna('Unknown')
                else:
                    plot_df['facility_match'] = fm.astype(str).fillna('Unknown')
                
                hover_dict['facility_match'] = True
                if (col in num_cols) and (col in FACILITY_SHAPE_COLS): 
                    plot_kwargs['symbol'] = 'facility_match'
            
            # --- Numeric Plots ---
            if col in num_cols:
                plot_df['numeric_x'] = pd.to_numeric(plot_df[col], errors='coerce')
                hover_dict[col] = True
                try:
                    fig = px.scatter(
                        plot_df, x='numeric_x', y=metric, 
                        color='facility_match' if 'facility_match' in plot_df.columns else None,
                        title=f"Alpha Diversity ({metric}) vs {col}", 
                        hover_data=hover_dict, trendline="ols", 
                        marginal_x="box", marginal_y="box", opacity=0.25, **plot_kwargs
                    )
                    fig.update_layout(xaxis_title=col, yaxis_title=metric.replace('_', ' ').capitalize())
                    
                    valid_stats = plot_df[['numeric_x', metric]].dropna()
                    if len(valid_stats) >= 5:
                        res = stats.spearmanr(valid_stats['numeric_x'], valid_stats[metric])
                        stat_text = f"Spearman r={res.statistic:.3f}, p={res.pvalue:.2e}"
                        pval = res.pvalue
                    else: pval = 1.0
                    
                except Exception as e: logger.error(f"Scatter failed {metric}/{col}: {e}"); continue

            # --- Categorical Plots ---
            elif col in cat_cols or (col in (priority_categorical or [])):
                plot_df['plot_x'] = plot_df[col].astype(str).fillna('Unknown')
                counts = plot_df['plot_x'].value_counts()
                plot_df['plot_x_n'] = plot_df['plot_x'].apply(lambda x: f"{x} (n={counts.get(x,0)})")
                
                color_col = 'plot_x'
                if col in FACILITY_SHAPE_COLS and 'facility_match' in plot_df.columns:
                    plot_df['combined'] = plot_df['plot_x'] + " (" + plot_df['facility_match'].astype(str) + ")"
                    color_col = 'combined'

                try:
                    n_cats = plot_df['plot_x_n'].nunique()
                    fig = px.violin(
                        plot_df, x='plot_x_n', y=metric, color=color_col,
                        title=f"Alpha Diversity ({metric}) by {col}",
                        hover_data=hover_dict, box=True, points='all'
                    )
                    fig.update_layout(xaxis_title=col, yaxis_title=metric, width=max(800, n_cats*60+250))
                    
                    groups = [d[metric].values for _, d in plot_df.groupby('plot_x') if len(d) > 0]
                    if len(groups) == 2:
                        s, p = stats.mannwhitneyu(groups[0], groups[1])
                        stat_text = f"M-W p={p:.2e}"; pval = p
                    elif len(groups) > 2:
                        s, p = stats.kruskal(*groups)
                        stat_text = f"K-W p={p:.2e}"; pval = p
                    else: pval = 1.0
                    
                except Exception as e: logger.error(f"Violin failed {metric}/{col}: {e}"); continue

            if fig:
                safe_col = re.sub(r'[^A-Za-z0-9_]+', '', col)
                prefix = "SIGNIFICANT_" if (not pd.isna(pval) and pval < 0.05) else ""
                if prefix: fig.update_layout(title=f"{fig.layout.title.text}<br><b>{stat_text}</b>")
                
                plot_path = plot_dir_alpha / f"{prefix}{'scatter_' if col in num_cols else 'violin_'}{metric}_vs_{safe_col}"
                plot_utils.save_plotly_fig(fig, plot_path, batch=True)

    plot_utils.flush_plot_queue()