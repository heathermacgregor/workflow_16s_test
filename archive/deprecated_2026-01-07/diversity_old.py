# ==================================================================================== #

import multiprocessing
import re
import textwrap
import subprocess
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os # <-- IMPORT ADDED

import anndata as ad
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scanpy as sc
import scipy.stats as stats
from scipy.sparse import issparse, csr_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, rankdata
from skbio.diversity import alpha_diversity, beta_diversity
from skbio.stats.distance import permanova, MissingIDError, mantel, DistanceMatrix
from skbio.stats.ordination import pcoa, rda
from skbio.tree import TreeNode
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from tqdm.auto import tqdm

from workflow_16s.downstream.preprocessing import AnalysisUtils
from workflow_16s.downstream.plotting import PlottingUtils, DEFAULT_HEIGHT
from workflow_16s.utils.logger import get_logger

# ==================================================================================== #

logger = get_logger("workflow_16s")
plot_utils = PlottingUtils(logger)

EXPECTED_VAR_COLUMNS = {'Taxon', 'Confidence', 'sequence'}
EXPECTED_VAR_DTYPES = {'Taxon': 'string', 'Confidence': 'Float64', 'sequence': 'string'}
FACILITY_SHAPE_COLS = {'facility_capacity', 'facility_start_year', 'facility_end_year', 'facility_type', 'facility'}

sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white', frameon=False)
ad.settings.allow_write_nullable_strings = True

# _CPU_COUNT will be set by the main run_beta_diversity_and_stats function
# This is used to pass the n_cpus from orchestrator.py to the parallel helpers.
_CPU_COUNT = os.cpu_count() or 1

# ==================================================================================== #
#                       Alpha Diversity & Taxa-Stats (Unchanged)
# ==================================================================================== #

def _calculate_kruskal(item: Tuple[str, pd.Series], meta_vector_shared: pd.Series) -> Optional[Dict[str, Any]]:
    """
    Helper function to calculate Kruskal-Wallis test for a single taxon.
    
    Args:
        item: Tuple of (taxon_name, abundance_series)
        meta_vector_shared: Categorical metadata vector for grouping
        
    Returns:
        Dictionary with test results or None if test fails
    """
    taxon, abund_series = item
    plot_df = pd.DataFrame({'meta': meta_vector_shared, 'abund': abund_series}).dropna()
    if plot_df.shape[0] < 5: return None 
    groups = [d['abund'].values for _, d in plot_df.groupby('meta', observed=True) if not d.empty]
    if len(groups) < 2: return None
    try: stat, p_val = stats.kruskal(*groups); return {'taxon': taxon, 'statistic': stat, 'p_value': p_val}
    except (ValueError, Exception): return None


def plot_significant_taxa_heatmap(adata_agg: ad.AnnData, sig_df: pd.DataFrame, plot_dir_stats: Path, tax_level: str):
    """Plots a heatmap of significant taxa."""
    logger.info(f"Generating heatmap for {tax_level}..."); sig_taxa = sig_df['taxon'].unique().tolist()
    if not sig_taxa: logger.info("No significant taxa."); return
    plottable_meta = AnalysisUtils.find_plottable_metadata(adata_agg, admin_noise_columns=None, fullness_threshold=0.25, max_categories=50)
    meta_cols = [c for c in plottable_meta['categorical'] + plottable_meta['numeric'] if c in adata_agg.obs.columns][:5]
    if not meta_cols: logger.warning("No metadata for heatmap annotation.")
    try:
        if meta_cols: 
            sc.pl.heatmap(adata_agg, var_names=sig_taxa, groupby=meta_cols[0], cmap='viridis', dendrogram=True, show=False, standard_scale='var', figsize=(min(len(sig_taxa) * 0.5, 20), min(adata_agg.n_obs * 0.1, 30)))
            target_path = plot_dir_stats / tax_level / "SIGNIFICANT_taxa_heatmap.png"; target_path.parent.mkdir(exist_ok=True, parents=True); plt.savefig(target_path, bbox_inches='tight', dpi=150); plt.close(); logger.info(f"Saved heatmap: {target_path}")
        else: logger.warning("No metadata for heatmap grouping, skipping heatmap generation.")
    except Exception as e: logger.error(f"Failed heatmap for {tax_level}: {e}"); plt.close()


def plot_correlation(x_data, y_data, x_name, y_name, plot_dir_stats, tax_level, corr, p_adj):
    """Plots scatter plot with marginal histograms."""
    plot_df = pd.DataFrame({'x': x_data, 'y': y_data}).dropna()
    if plot_df.shape[0] < 2: logger.warning(f"Skip corr plot {y_name} vs {x_name}"); return
    title = f"Spearman ({tax_level})"; subtitle = f"{y_name} vs. {x_name}"; stats_txt = f"r={corr:.3f}, p_adj={p_adj:.2e}"
    try:
        hover_name = plot_df.index.name if plot_df.index.name else 'SampleID'; hover_dict = {hover_name: plot_df.index, 'x': True, 'y': True}
        fig = px.scatter(plot_df, x='x', y='y', title=f"{title}<br>{subtitle}<br><b>{stats_txt}</b>", labels={'x': x_name, 'y': f"{y_name} (CLR)"}, trendline="ols", trendline_color_override="red", hover_data=hover_dict, marginal_x="histogram", marginal_y="histogram")
        fig.update_layout(margin={'l': 70, 'r': 100, 'b': 70, 't': 120, 'pad': 4})
        safe_x = re.sub(r'[^A-Za-z0-9_]+', '', x_name); safe_y = re.sub(r'[^A-Za-z0-9_]+', '', y_name)
        pdir = plot_dir_stats / tax_level / "Correlation"; ppath = pdir / f"corr_{safe_x}_vs_{safe_y}"; plot_utils.save_plotly_fig(fig, ppath, batch=True)
    except Exception as e: logger.error(f"Failed corr plot {y_name} vs {x_name}: {e}")


def plot_raincloud(x_data, y_data, x_name, y_name, plot_dir_stats, tax_level, p_adj):
    """
    Plots a raincloud (violin+box) plot for categorical comparisons.
    Points removed for clarity with dense data.
    """
    plot_df = pd.DataFrame({'plot_x_original': x_data, 'y': y_data}).dropna(); hover_col_name = plot_df.index.name if plot_df.index.name else 'SampleID'; plot_df[hover_col_name] = plot_df.index
    plot_df['plot_x_original'] = plot_df['plot_x_original'].astype(str).fillna('Unknown'); counts_map = plot_df['plot_x_original'].value_counts().to_dict()
    plot_df['plot_x_with_n'] = plot_df['plot_x_original'].apply(lambda x: f"{x} (n={counts_map.get(x, 0)})"); unique_vals = x_data.dropna().unique()  
    try: numeric_vals = pd.to_numeric(unique_vals, errors='coerce'); nan_mask = pd.isna(numeric_vals); numeric_sorted = sorted(numeric_vals[~nan_mask]); string_vals_sorted = sorted([str(v) for v in unique_vals[nan_mask]]); sorted_known = [str(int(v)) if v == int(v) else str(v) for v in numeric_sorted] + string_vals_sorted; category_order_original = ['Unknown'] + sorted_known
    except (ValueError, TypeError): sorted_known = sorted([str(c) for c in unique_vals]); category_order_original = ['Unknown'] + sorted_known
    category_order_new = [ f"{c_orig} (n={counts_map.get(c_orig, 0)})" for c_orig in category_order_original if c_orig in counts_map and f"{c_orig} (n={counts_map[c_orig]})" in plot_df['plot_x_with_n'].unique()]
    if plot_df.shape[0] < 2 or len(category_order_new) < 2: logger.warning(f"Skip raincloud plot {y_name} vs {x_name}"); return
    title = f"Taxon Abundance ({tax_level})"; subtitle = f"{y_name} vs. {x_name}"; stats_txt = f"Kruskal-Wallis p_adj={p_adj:.2e}"    
    try:
        fig = px.violin(plot_df, x='plot_x_with_n', y='y', color='plot_x_original', title=f"{title}<br>{subtitle}<br><b>{stats_txt}</b>", labels={'y': f"{y_name} (CLR)"}, box=True, points=False, hover_name=hover_col_name, category_orders={'plot_x_with_n': category_order_new})
        n_cats = len(category_order_new); plot_width = max(800, n_cats * 60 + 250)
        fig.update_layout(xaxis_title=x_name, legend_title_text=x_name, height=1000, width=plot_width, margin={'l': 70, 'r': 150, 'b': 150, 't': 120, 'pad': 4}); fig.update_traces(meanline_visible=True)
        safe_x = re.sub(r'[^A-Za-z0-9_]+', '', x_name); safe_y = re.sub(r'[^A-Za-z0-9_]+', '', y_name)
        pdir = plot_dir_stats / tax_level / "Categorical"; ppath = pdir / f"raincloud_{safe_x}_vs_{safe_y}"; plot_utils.save_plotly_fig(fig, ppath, batch=True)
    except Exception as e: logger.error(f"Failed raincloud plot {y_name} vs {x_name}: {e}")


def run_alpha_diversity(adata: ad.AnnData, plot_dir_alpha: Path, tree_path: Optional[Path] = None, priority_categorical: Optional[List[str]] = None, priority_numeric: Optional[List[str]] = None):
    """Calculates and plots alpha diversity metrics against metadata. If a tree_path is provided, also calculates Faith's Phylogenetic Diversity."""
    logger.info("--- Starting Alpha Diversity Analysis ---")
    if 'raw_counts' not in adata.layers: logger.error("'raw_counts' layer not found. Cannot calculate alpha diversity."); return
    if adata.n_obs == 0: logger.warning("No samples remaining for alpha diversity."); return
    raw_counts = adata.layers['raw_counts']
    if issparse(raw_counts): counts_matrix_sparse = csr_matrix(raw_counts)
    else: logger.warning("Converting dense raw_counts to sparse for alpha diversity."); counts_matrix_sparse = csr_matrix(raw_counts)
    sample_ids = adata.obs_names.tolist(); feature_ids = adata.var_names.tolist()
    logger.info("Calculating alpha diversity metrics..."); alpha_div_df = pd.DataFrame(index=sample_ids)
    counts_gt_0 = counts_matrix_sparse.copy(); counts_gt_0.data[counts_gt_0.data > 0] = 1; alpha_div_df['observed_features'] = np.array(counts_gt_0.sum(axis=1)).flatten()
    sample_sums = np.array(counts_matrix_sparse.sum(axis=1)).flatten(); non_zero_mask = sample_sums > 0; metrics_to_plot = ['observed_features']
    if np.any(non_zero_mask):
        try:
            # --- START FIX: Filter ids to match the filtered counts matrix ---
            # Create a list of IDs that correspond to the non-zero rows
            ids_subset = [id for (id, keep) in zip(sample_ids, non_zero_mask) if keep]
            shannon_values = alpha_diversity('shannon', counts_matrix_sparse[non_zero_mask].astype(int), ids=ids_subset)
            # --- END FIX ---
            alpha_div_df['shannon'] = shannon_values.reindex(sample_ids)
            metrics_to_plot.append('shannon')
        except Exception as e: logger.error(f"Shannon calculation failed: {e}"); alpha_div_df['shannon'] = np.nan
    else: alpha_div_df['shannon'] = np.nan
    if tree_path:
        if not tree_path.exists(): logger.error(f"Tree file not found at: {tree_path}. Skipping Faith's PD.")
        else:
            try:
                logger.info(f"Loading phylogenetic tree from: {tree_path}")
                tree = TreeNode.read(str(tree_path)); tree_tips = {tip.name for tip in tree.tips()}; features_in_tree = [f_id for f_id in feature_ids if f_id in tree_tips]
                if not features_in_tree: logger.error("No features from data found in the tree. Check var_names. Skipping Faith's PD.")
                else:
                    feature_idx = [feature_ids.index(f_id) for f_id in features_in_tree]; counts_matrix_filt_sparse = counts_matrix_sparse[:, feature_idx]
                    logger.info(f"Calculating Faith's PD for {len(features_in_tree)} features...")
                    faith_pd_values = alpha_diversity('faith_pd', counts_matrix_filt_sparse.astype(int), ids=sample_ids, tree=tree, otu_ids=features_in_tree); alpha_div_df['faith_pd'] = faith_pd_values.reindex(sample_ids); metrics_to_plot.append('faith_pd')
            except Exception as e: logger.error(f"Faith's PD calculation failed: {e}"); alpha_div_df['faith_pd'] = np.nan
    else: logger.info("No tree_path provided, skipping Faith's PD.")
    adata.obs = adata.obs.join(alpha_div_df); logger.info(f"Alpha metrics added to adata.obs: {', '.join(metrics_to_plot)}")
    logger.info("Plotting alpha diversity...")
    metadata_cols = AnalysisUtils.find_plottable_metadata(adata, admin_noise_columns=None, fullness_threshold=0.25, max_categories=50); cat_cols = metadata_cols['categorical']; num_cols = metadata_cols['numeric']
    all_cols_to_plot = (priority_categorical or []) + (priority_numeric or []) + cat_cols + num_cols; all_cols_to_plot = list(dict.fromkeys(all_cols_to_plot)); all_cols_to_plot = [c for c in all_cols_to_plot if c in adata.obs.columns]
    for metric in metrics_to_plot:
        if metric not in adata.obs.columns or adata.obs[metric].isnull().all(): logger.warning(f"Skipping plots for '{metric}' (no valid data)."); continue 
        for col in all_cols_to_plot:
            if col == metric: logger.debug(f"  -> Skipping plot of {metric} vs itself."); continue
            logger.info(f"  -> Plotting {metric} vs {col}"); 
            
            # --- FIX: Replace pd.NA with np.nan to prevent kaleido save error ---
            plot_df = adata.obs[[col, metric]].copy().replace(pd.NA, np.nan).dropna(subset=[metric])
            # --- END FIX ---
            
            if plot_df.shape[0] < 2: logger.debug(f"Skip plot {col} vs {metric}: < 2 valid points."); continue  
            fig = None; plot_width = 1200; plot_kwargs = {}; hover_name = plot_df.index.name if plot_df.index.name else 'SampleID'; hover_dict: Dict[str, Any] = {str(hover_name): plot_df.index}
            stat_text = ""  # Initialize stat_text to avoid unbound variable errors
            if 'facility_match' in adata.obs.columns:
                fm_data = adata.obs.loc[plot_df.index, 'facility_match']
                if isinstance(fm_data.dtype, pd.CategoricalDtype):
                    if 'Unknown' not in fm_data.cat.categories: fm_data = fm_data.cat.add_categories('Unknown')
                    plot_df['facility_match'] = fm_data.fillna('Unknown')
                else: plot_df['facility_match'] = fm_data.astype(str).fillna('Unknown')
                hover_dict['facility_match'] = True
                if (col in num_cols) and (col in FACILITY_SHAPE_COLS): plot_kwargs['symbol'] = 'facility_match'
            if col in num_cols:
                logger.debug(f"       Plotting '{col}' as numeric scatter plot."); plot_df['numeric_x'] = pd.to_numeric(plot_df[col], errors='coerce'); hover_dict[col] = True
                try:
                    fig = px.scatter(plot_df, x='numeric_x', y=metric, color='facility_match' if 'facility_match' in plot_df.columns else None, title=f"Alpha Diversity ({metric}) vs {col}", hover_data=hover_dict, trendline="ols", marginal_x="box", marginal_y="box", opacity=0.25, **plot_kwargs)
                    fig.update_layout(xaxis_title=col, yaxis_title=metric.replace('_', ' ').capitalize(), legend_title_text='Facility Match' if 'facility_match' in plot_df.columns else col); plot_width = 1000
                except Exception as e: logger.error(f"Failed numeric scatter plot for {metric} vs {col}: {e}"); continue
                # Stats: Spearman for numeric
                stat_text = ""; pval = 1.0; test_name = "Spearman"; valid_stats_df = plot_df[['numeric_x', metric]].dropna()
                if valid_stats_df.shape[0] >= 5:
                    try:
                        result = spearmanr(valid_stats_df['numeric_x'], valid_stats_df[metric]); corr = result[0]; pval = result[1]
                        try: pval_scalar = float(pval) # type: ignore
                        except (TypeError, ValueError): pval_scalar = float(np.nan)
                        if not pd.isna(pval_scalar) and pval_scalar is not None: stat_text = f"{test_name} r={corr:.3f}, p={pval_scalar:.2e}"; logger.info(f"  -> {test_name}: r={corr:.3f}, p={pval_scalar:.4e}")
                        else: stat_text = f"{test_name}: NaN"
                    except ValueError as e: logger.warning(f"Stat test failed {metric} vs {col}: {e}"); pval = 1.0
            elif col in cat_cols or (col in (priority_categorical or [])):
                logger.debug(f"       Plotting '{col}' as categorical raincloud plot."); temp_col_data = plot_df[col]
                if isinstance(temp_col_data.dtype, pd.CategoricalDtype):
                    if 'Unknown' not in temp_col_data.cat.categories: plot_df[col] = temp_col_data.cat.add_categories('Unknown')
                    plot_df['plot_x_original'] = plot_df[col].fillna('Unknown')
                else: plot_df['plot_x_original'] = temp_col_data.astype(str).fillna('Unknown')
                counts_map = plot_df['plot_x_original'].value_counts().to_dict(); plot_df['plot_x_with_n'] = plot_df['plot_x_original'].apply(lambda x: f"{x} (n={counts_map.get(x, 0)})")
                unique_vals = adata.obs[col].dropna().unique()
                try:
                    numeric_vals = pd.to_numeric(unique_vals, errors='coerce'); nan_mask = pd.isna(numeric_vals); numeric_sorted = sorted(numeric_vals[~nan_mask])
                    string_vals_sorted = sorted([str(v) for v in unique_vals[nan_mask]]); sorted_known = [str(int(v)) if v == int(v) else str(v) for v in numeric_sorted] + string_vals_sorted
                    category_order_original = ['Unknown'] + sorted_known
                except (ValueError, TypeError): sorted_known = sorted([str(c) for c in unique_vals]); category_order_original = ['Unknown'] + sorted_known
                category_order_new = [f"{c_orig} (n={counts_map.get(c_orig, 0)})" for c_orig in category_order_original if c_orig in counts_map and f"{c_orig} (n={counts_map[c_orig]})" in plot_df['plot_x_with_n'].unique()]
                if len(category_order_new) < 2: logger.debug(f"Skip plot {col} vs {metric}: < 2 categories after prep."); continue
                color_col_strip = 'plot_x_original'; legend_title = col
                if col in FACILITY_SHAPE_COLS and 'facility_match' in plot_df.columns: plot_df['combined_color'] = plot_df['plot_x_original'].astype(str) + " (" + plot_df['facility_match'].astype(str) + ")"; color_col_strip = 'combined_color'; legend_title = f"{col} (Facility Match)"
                try:
                    n_cats = len(category_order_new); plot_width = max(800, n_cats * 60 + 250)
                    fig = px.violin(plot_df, x='plot_x_with_n', y=metric, color=color_col_strip, title=f"Alpha Diversity ({metric}) by {col}", hover_data=hover_dict, category_orders={'plot_x_with_n': category_order_new}, box=True, points='all')
                    fig.update_layout(xaxis_title=col, yaxis_title=metric.replace('_', ' ').capitalize(), legend_title_text=legend_title, showlegend=(color_col_strip == 'combined_color'), width=plot_width)
                except Exception as e: logger.error(f"Failed violin plot for {metric} vs {col}: {e}"); continue
                # Stats: K-W/M-W for categorical
                groups = [d[metric].values for _, d in plot_df.groupby('plot_x_original', observed=True) if not d.empty]; stat_text = ""; pval = 1.0; test_name = ""
                try:
                    if len(groups) == 2: test_name = "M-W"; stat, pval_result = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided'); pval = float(pval_result)
                    elif len(groups) > 2: test_name = "K-W"; stat, pval_result = stats.kruskal(*groups); pval = float(pval_result)
                    else: pval = 1.0
                    if not pd.isna(pval) and pval is not None: stat_text = f"{test_name}: p={pval:.2e}"; logger.info(f"  -> {test_name}: p={pval:.4e}")
                    else: stat_text = f"{test_name}: NaN"
                except ValueError as e:
                    if "identical" in str(e).lower(): pval = 1.0; test_name = f"{test_name} (skip-ident)"
                    else: logger.warning(f"Stat test failed {metric} vs {col}: {e}"); pval = 1.0
            if fig is None: continue
            safe_col = re.sub(r'[^A-Za-z0-9_]+', '', col); file_prefix = ""; 
            plot_type_prefix = "scatter_box_" if (col in num_cols) else "violin_"
            try: pval_float = float(pval) # type: ignore
            except (TypeError, ValueError): pval_float = float('nan')
            if not pd.isna(pval_float) and pval_float < 0.05: logger.info(f"  -> SIGNIFICANT (p={pval_float:.4e})"); fig.update_layout(title=f"Alpha Div. ({metric}) by {col}<br><b>{stat_text}</b>"); file_prefix = "SIGNIFICANT_"
            plot_path_html = plot_dir_alpha / f"{file_prefix}{plot_type_prefix}{metric}_vs_{safe_col}"; plot_utils.save_plotly_fig(fig, plot_path_html, batch=True)  
    # Flush batched plots
    plot_utils.flush_plot_queue()
        

# ==================================================================================== #
#                       Beta Diversity (Main Function)
# ==================================================================================== #

def run_beta_diversity_and_stats(adata: ad.AnnData, analysis_levels: List[str], plot_dir_beta: Path, tree_path: Optional[Path] = None, n_cpus: Optional[int] = None):
    """
    Runs beta diversity calculations (Bray-Curtis, UniFrac), ordinations (PCoA, UMAP),
    and statistical tests (PERMANOVA, Mantel) for multiple taxonomic levels.
    """
    global _CPU_COUNT # <-- NEW: Set global CPU count for helpers
    _CPU_COUNT = n_cpus if n_cpus is not None else (os.cpu_count() or 1)
    
    logger.info(f"--- Starting Beta Diversity Analysis (Using {_CPU_COUNT} CPUs) ---")
    if adata.n_obs < 3: logger.warning("Not enough samples (< 3) for beta diversity. Skipping."); return
    
    # --- Load Tree (if applicable) ---
    tree = None
    if 'ASV' in analysis_levels:
        if tree_path:
            if not tree_path.exists(): logger.error(f"Tree file not found at: {tree_path}. Skipping UniFrac.")
            else:
                try: logger.info(f"Loading phylogenetic tree from: {tree_path}"); tree = TreeNode.read(str(tree_path))
                except Exception as e: logger.error(f"Failed to load phylogenetic tree: {e}")
        else: logger.warning("ASV level requested but no tree_path provided. Skipping UniFrac.")
        
    # --- Get all plottable metadata ONCE ---
    metadata_cols = AnalysisUtils.find_plottable_metadata(adata, admin_noise_columns=None, fullness_threshold=0.25, max_categories=50)
    plottable_categorical = [c for c in metadata_cols['categorical'] if c in adata.obs.columns]
    plottable_numeric = [c for c in metadata_cols['numeric'] if c in adata.obs.columns]
    if not plottable_categorical: logger.warning("No plottable categorical metadata found.")
    if not plottable_numeric: logger.warning("No plottable numeric metadata found.")
        
    # --- Run analysis for each level ---
    total_plots = 0
    for level in analysis_levels:
        try:
            n_plots = _run_beta_diversity_analysis_level(
                adata=adata, 
                level=level, 
                tree=tree, 
                plottable_categorical=plottable_categorical, 
                plottable_numeric=plottable_numeric, 
                plot_dir_beta=plot_dir_beta
            )
            total_plots += n_plots
        except Exception as e:
            logger.error(f"CRITICAL: Unhandled error in beta diversity for level '{level}': {e}")
            
    logger.info(f"--- Beta Diversity Analysis Complete. Saved {total_plots} total plots. ---")
    # Flush any remaining plots
    plot_utils.flush_plot_queue()


def _run_beta_diversity_analysis_level(adata: ad.AnnData, level: str, tree: Optional[TreeNode], plottable_categorical: List[str], plottable_numeric: List[str], plot_dir_beta: Path) -> int:
    """Helper function to process all beta diversity metrics for a single analysis level."""
    logger.info(f"===== Processing Beta Diversity for Level: {level} ====="); total_plots = 0
    if level == 'ASV':
        logger.info("Using base ASV-level AnnData.")
        adata_agg = adata.copy()
    else:
        adata_agg = AnalysisUtils.get_analysis_adata(adata, level=level)
    
    if adata_agg is None or adata_agg.n_obs < 3 or adata_agg.n_vars < 2: logger.warning(f"Skipping {level}: Not enough samples or features."); return 0
    
    counts_for_beta = adata_agg.layers['raw_counts']
    if hasattr(counts_for_beta, 'toarray'): counts_for_beta = counts_for_beta.toarray() # type: ignore
    
    sample_sums = counts_for_beta.sum(axis=1); zero_mask = (sample_sums == 0)
    if np.any(zero_mask): 
        n_zero = zero_mask.sum()
        logger.warning(f"Filtering {n_zero} zero-count samples for {level}.")
        if n_zero == adata_agg.n_obs: logger.error(f"All samples have zero counts for {level}. Skipping."); return 0
        adata_agg = adata_agg[~zero_mask, :].copy()
        counts_for_beta = counts_for_beta[~zero_mask]
        
    if adata_agg.n_obs < 3: logger.error(f"Not enough samples for {level} after filtering."); return 0
    
    sample_ids = adata_agg.obs_names.tolist(); feature_ids = adata_agg.var_names.tolist()
    
    # --- UMAP Calculation (on CLR) ---
    try:
        adata_clr = adata_agg.copy()
        # Use the CLR transform from preprocessing
        clr_df = AnalysisUtils._clr_transform(adata_clr, pseudocount=1)

        if pd.api.types.is_sparse(clr_df.dtypes.iloc[0]):
            adata_clr.X = clr_df.sparse.to_coo().tocsr()
            logger.debug(f"Converted sparse DataFrame to csr_matrix for UMAP.")
        else:
            logger.warning("CLR data is dense. UMAP/Neighbors may be slow.")
            adata_clr.X = clr_df.values

        sc.pp.neighbors(adata_clr, use_rep='X', n_neighbors=min(15, adata_clr.n_obs - 1))
        sc.tl.umap(adata_clr)
        adata_agg.obsm['X_umap'] = adata_clr.obsm['X_umap']
    except Exception as e: logger.error(f"Failed to compute UMAP for {level}: {e}")
    
    # --- Beta Diversity Metrics ---
    metrics_to_run = ['braycurtis']
    if level == 'ASV' and tree is not None: metrics_to_run.extend(['unweighted_unifrac', 'weighted_unifrac'])
    elif level == 'ASV' and tree is None: logger.warning("ASV level detected but no tree provided. Skipping UniFrac.")
        
    for metric_name in metrics_to_run:
        logger.info(f"--- Processing: {metric_name} ({level}) ---")
        dist_matrix = None
        try:
            if metric_name == 'braycurtis': dist_matrix = beta_diversity("braycurtis", counts_for_beta.astype(int), ids=sample_ids)
            elif metric_name in ['unweighted_unifrac', 'weighted_unifrac'] and tree is not None:
                tree_tips = {tip.name for tip in tree.tips()}; features_in_tree = [f_id for f_id in feature_ids if f_id in tree_tips]
                if not features_in_tree: logger.error("No features in common between tree and data. Skipping UniFrac."); continue
                feature_idx = [feature_ids.index(f_id) for f_id in features_in_tree]
                counts_filt = counts_for_beta[:, feature_idx]
                dist_matrix = beta_diversity(metric_name, counts_filt.astype(int), ids=sample_ids, tree=tree, otu_ids=features_in_tree, validate=False)
            
            if dist_matrix is not None: 
                total_plots += _process_distance_matrix(
                    dist_matrix=dist_matrix, 
                    dist_name=metric_name, 
                    level=level, 
                    adata_agg=adata_agg, 
                    plottable_categorical=plottable_categorical, 
                    plottable_numeric=plottable_numeric, 
                    plot_dir_beta=plot_dir_beta,
                    n_cpus=_CPU_COUNT # <-- PASS CPU COUNT
                )
        except Exception as e: 
            logger.error(f"{metric_name} calculation failed for {level}: {e}")
            
    if 'X_umap' in adata_agg.obsm: 
        total_plots += _plot_ordination(
            adata_agg=adata_agg, 
            ordination_name='UMAP', 
            level=level, 
            plottable_categorical=plottable_categorical, 
            plottable_numeric=plottable_numeric, 
            plot_dir_beta=plot_dir_beta
        )
    return total_plots


# ==================================================================================== #
#                  Beta Diversity Stats (OPTIMIZED & BUG-FIXED)
# ==================================================================================== #

def _run_permanova_parallel(col: str, metadata_df: pd.DataFrame, dist_matrix: DistanceMatrix) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Helper function to run PERMANOVA for a single column."""
    try:
        if col not in metadata_df.columns: 
            return None
        
        grouping_series = metadata_df[col].copy()
        if isinstance(grouping_series.dtype, pd.CategoricalDtype):
            if 'Unknown' not in grouping_series.cat.categories:
                try: grouping_series = grouping_series.cat.add_categories('Unknown')
                except Exception: grouping_series = grouping_series.astype(str)
        
        grouping = grouping_series.astype(str).fillna('Unknown'); group_counts = grouping.value_counts()
        valid_groups = group_counts[group_counts >= 2].index
        if len(valid_groups) < 2: 
            return None
            
        keep_mask = grouping.isin(valid_groups); keep_ids = metadata_df.index[keep_mask].tolist()
        if len(keep_ids) < 3: 
            return None
            
        dm_subset = dist_matrix.filter(ids=keep_ids)
        grouping_subset = grouping[keep_mask]
        if dm_subset.shape[0] < 2 or grouping_subset.nunique() < 2: 
            return None
            
        # Increased from 999 to 9999 for more robust p-value estimation (2026 best practice)
        perm_res = permanova(dm_subset, grouping=grouping_subset, permutations=9999)
        if pd.isna(perm_res['p-value']): 
            logger.warning(f"PERMANOVA NaN p-val for {col}.")
            return None
            
        logger.info(f" -> PERMANOVA '{col}': p={perm_res['p-value']:.4e}, F={perm_res['test statistic']:.4f}")
        return (col, perm_res)
    except (ValueError, KeyError, IndexError, MissingIDError) as e: 
        logger.error(f" -> PERMANOVA '{col}' failed: {e}")
        return None

def _run_mantel_parallel(col: str, metadata_df: pd.DataFrame, dist_matrix: DistanceMatrix) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Helper function to run Mantel test for a single column."""
    try:
        if col not in metadata_df.columns: 
            return None
            
        numeric_vector = metadata_df[col].copy()
        valid_mask = numeric_vector.notna()
        valid_ids = metadata_df.index[valid_mask].tolist()
        if len(valid_ids) < 3: 
            return None
            
        try: dist_subset = dist_matrix.filter(ids=valid_ids)
        except MissingIDError as e: logger.warning(f"Mantel skip '{col}': filter error: {e}"); return None
        if dist_subset.shape[0] < 2: 
            return None
            
        numeric_vec_sub = np.asarray(numeric_vector[valid_mask]).reshape(-1, 1)
        try: num_dist_cond = pdist(numeric_vec_sub, 'euclidean')
        except ValueError as e: logger.warning(f"Mantel skip '{col}': num dist matrix error: {e}"); return None
        if np.all(num_dist_cond == 0): logger.debug(f"Mantel skip '{col}': no variance."); return None
        
        numeric_dm = DistanceMatrix(squareform(num_dist_cond), ids=valid_ids)
        r, p, _ = mantel(dist_subset, numeric_dm, permutations=999)
        
        if pd.isna(p): logger.warning(f"Mantel NaN p-val for {col}."); return None
        
        logger.info(f" -> Mantel '{col}': r={r:.4f}, p={p:.4e}")
        return (col, {'r': r, 'p-value': p})
    except Exception as e: 
        logger.error(f" -> Mantel '{col}' failed: {e}")
        return None


def _process_distance_matrix(dist_matrix: DistanceMatrix, dist_name: str, level: str, adata_agg: ad.AnnData, plottable_categorical: List[str], plottable_numeric: List[str], plot_dir_beta: Path, n_cpus: int) -> int:
    """
    Internal helper function to run PCoA, PERMANOVA, Mantel, and plotting for a given distance matrix.
    NOW PARALLELIZED.
    """
    plots_saved = 0
    try: pcoa_results = pcoa(dist_matrix)
    except Exception as e: logger.error(f"PCoA failed for {dist_name} at {level}: {e}"); return 0
    
    pcoa_samples_df = pcoa_results.samples if isinstance(pcoa_results.samples, pd.DataFrame) else pd.DataFrame(pcoa_results.samples, index=adata_agg.obs_names)
    obsm_key = f'X_pcoa_{dist_name}'; uns_key = f'pcoa_{dist_name}_variance'
    
    adata_agg.obsm[obsm_key] = pcoa_samples_df; adata_agg.uns[uns_key] = pcoa_results.proportion_explained
    var_exp = adata_agg.uns[uns_key]; pc1_var = var_exp[0] if var_exp is not None and len(var_exp) > 0 else 0; pc2_var = var_exp[1] if var_exp is not None and len(var_exp) > 1 else 0
    axis_labels = {'x': f"PC1 ({pc1_var:.2%})", 'y': f"PC2 ({pc2_var:.2%})"}
    
    metadata_df = adata_agg.obs
    
    # --- PARALLEL PERMANOVA ---
    logger.info(f"Running PERMANOVA ({dist_name}, {level}) on {n_cpus} cores...")
    permanova_results = {}
    # Create a partial function to pass the static args (metadata, dist_matrix)
    permanova_func = partial(_run_permanova_parallel, metadata_df=metadata_df, dist_matrix=dist_matrix)
    
    with multiprocessing.Pool(processes=n_cpus) as pool:
        # Use imap_unordered for efficiency (don't wait for slow ones)
        results_iter = pool.imap_unordered(permanova_func, plottable_categorical)
        for result in results_iter:
            if result:
                col, res_data = result
                permanova_results[col] = res_data
    
    # --- PARALLEL MANTEL ---
    logger.info(f"Running Mantel test ({dist_name}, {level}) on {n_cpus} cores...")
    mantel_results = {}
    mantel_func = partial(_run_mantel_parallel, metadata_df=metadata_df, dist_matrix=dist_matrix)

    with multiprocessing.Pool(processes=n_cpus) as pool:
        results_iter = pool.imap_unordered(mantel_func, plottable_numeric)
        for result in results_iter:
            if result:
                col, res_data = result
                mantel_results[col] = res_data

    # --- Plotting (runs after all stats are collected) ---
    plots_saved += _plot_ordination(
        adata_agg=adata_agg, 
        ordination_name='PCoA', 
        dist_name=dist_name, 
        level=level, 
        plottable_categorical=plottable_categorical, 
        plottable_numeric=plottable_numeric, 
        plot_dir_beta=plot_dir_beta, 
        axis_labels=axis_labels, 
        permanova_results=permanova_results, 
        mantel_results=mantel_results
    )
    return plots_saved


def _plot_ordination(adata_agg: ad.AnnData, ordination_name: str, level: str, plottable_categorical: List[str], plottable_numeric: List[str], plot_dir_beta: Path, dist_name: str = "", axis_labels: Optional[Dict[str, str]] = None, permanova_results: Optional[Dict[str, Any]] = None, mantel_results: Optional[Dict[str, Any]] = None) -> int:
    """Internal helper to plot any 2D ordination (PCoA, UMAP, etc.)
    
    Args:
        adata_agg: AnnData object with ordination results in .obsm
        ordination_name: Name of the ordination method ('PCoA', 'UMAP', etc.)
        level: Taxonomic level being analyzed
        plottable_categorical: List of categorical metadata columns to plot
        plottable_numeric: List of numeric metadata columns to plot
        plot_dir_beta: Directory to save plots
        dist_name: Name of the distance metric (for PCoA)
        axis_labels: Optional dict with 'x' and 'y' labels for axes
        permanova_results: Optional dict of PERMANOVA results for annotations
        mantel_results: Optional dict of Mantel test results for annotations
        
    Returns:
        Number of plots saved
    """
    plots_saved = 0
    if permanova_results is None: permanova_results = {}
    if mantel_results is None: mantel_results = {}
    
    if ordination_name == 'PCoA':
        obsm_key = f'X_pcoa_{dist_name}'; plot_title_prefix = f"{dist_name} PCoA"; plot_file_prefix = f"{dist_name}_PCoA"
        if axis_labels is None: axis_labels = {'x': 'PC1', 'y': 'PC2'}
    elif ordination_name == 'UMAP':
        obsm_key = 'X_umap'; plot_title_prefix = "UMAP"; plot_file_prefix = "UMAP"
        if axis_labels is None: axis_labels = {'x': 'UMAP 1', 'y': 'UMAP 2'}
    else: logger.error(f"Unknown ordination_name: {ordination_name}"); return 0
    
    if obsm_key not in adata_agg.obsm: logger.warning(f"Skipping {ordination_name} plots for {level}: '{obsm_key}' not in obsm."); return 0
    
    # --- Prepare data for plotting ---
    adata_agg_norm = adata_agg.copy()
    adata_agg_norm.X = adata_agg_norm.layers['raw_counts'].copy()
    sc.pp.normalize_total(adata_agg_norm, target_sum=1e4)
    sc.pp.log1p(adata_agg_norm)
    
    new_col_names = []
    if adata_agg_norm.n_vars > 0:
        mean_abund = np.array(adata_agg_norm.X.mean(0)).flatten()
        n_top = min(20, len(mean_abund))
        top_indices = np.argsort(mean_abund)[-n_top:][::-1]
        top_taxa_names = adata_agg_norm.var_names[top_indices].tolist()
        abund_data = adata_agg_norm.X[:, top_indices]
        if hasattr(abund_data, 'toarray') and callable(getattr(abund_data, 'toarray', None)): abund_data = abund_data.toarray() # type: ignore
        elif not isinstance(abund_data, np.ndarray): abund_data = np.asarray(abund_data)
        new_col_names = [f"Feature: {name}" for name in top_taxa_names]
        taxa_df = pd.DataFrame(abund_data, index=adata_agg_norm.obs_names, columns=new_col_names)
    else: logger.warning(f"No variables for {level} in {ordination_name} plotting."); taxa_df = pd.DataFrame(index=adata_agg_norm.obs_names)
    
    # Handle 1-component ordinations
    full_ord_data_obj = adata_agg.obsm[obsm_key]

    # Ensure we are working with a NumPy array (PCoA is DataFrame, UMAP is array)
    if isinstance(full_ord_data_obj, pd.DataFrame): full_ord_data = full_ord_data_obj.values
    else: full_ord_data = full_ord_data_obj
    
    n_components = full_ord_data.shape[1]

    if n_components >= 2:
        ord_data = full_ord_data[:, :2]
        col_names = ['Dim1', 'Dim2']
    elif n_components == 1:
        logger.warning(f"Ordination '{obsm_key}' only has 1 component. Plotting with a dummy 2nd axis.")
        # Create a dummy 2nd dimension of all zeros
        ord_data = np.column_stack([full_ord_data[:, 0], np.zeros(full_ord_data.shape[0])])
        col_names = ['Dim1', 'Dim2 (dummy)']
        # Adjust axis labels for the dummy axis
        if axis_labels and 'y' in axis_labels: axis_labels['y'] = 'PC2 (Dummy)'
    else:
        logger.error(f"Cannot plot ordination '{obsm_key}': Has 0 components.")
        return 0 # Return 0 plots saved

    ord_df = pd.DataFrame(ord_data, columns=col_names, index=adata_agg.obs_names)
    ord_df['SampleID'] = ord_df.index
    ord_df = ord_df.join(adata_agg.obs)
    ord_df = ord_df.join(taxa_df) # Join top feature abundances
    
    if 'facility_match' in adata_agg.obs.columns:
        fm_data = adata_agg.obs.loc[ord_df.index, 'facility_match']
        if isinstance(fm_data.dtype, pd.CategoricalDtype):
            if 'Unknown' not in fm_data.cat.categories: fm_data = fm_data.cat.add_categories('Unknown')
            ord_df['facility_match'] = fm_data.fillna('Unknown')
        else: ord_df['facility_match'] = fm_data.astype(str).fillna('Unknown')
            
    all_color_columns = (
        [c for c in plottable_categorical if c in ord_df.columns] + 
        [c for c in plottable_numeric if c in ord_df.columns] + 
        [c for c in new_col_names if c in ord_df.columns]
    )
    logger.info(f"Plotting {ordination_name} for {len(all_color_columns)} metadata columns...")

    # --- Main Plotting Loop ---
    for col in all_color_columns:
        if col not in ord_df.columns: continue
        
        # Replace pd.NA with np.nan to prevent kaleido save error 
        plot_df = ord_df.copy().replace(pd.NA, np.nan).dropna(subset=[col])
        
        if plot_df.shape[0] < 2: continue
        
        plot_kwargs: Dict[str, Any] = {'hover_name': 'SampleID', 'hover_data': {col: True}}
        is_categorical = col in plottable_categorical
        is_feature = col.startswith("Feature:")
        is_numeric = (col in plottable_numeric) or is_feature
        stat_text = ""; file_prefix = ""
        
        # --- Handle color and stats text ---
        if is_categorical:
            plot_kwargs['color'] = col
            if col in permanova_results:
                res = permanova_results[col]
                if res['p-value'] < 0.05:
                    stat_text = f"PERMANOVA: p={res['p-value']:.2e}, F={res['test statistic']:.2f}"
                    file_prefix = "SIGNIFICANT_"
            # Truncate long category names
            if pd.api.types.is_object_dtype(plot_df[col].dtype) or isinstance(plot_df[col].dtype, pd.CategoricalDtype):
                plot_df[col] = plot_df[col].astype(str).str.slice(0, 50)
                
        elif is_numeric:
            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
            plot_df = plot_df.dropna(subset=[col])
            plot_kwargs.update({
                'color': col,
                'color_continuous_scale': 'viridis'
            })
            if col in mantel_results:
                res = mantel_results[col]
                if res['p-value'] < 0.05:
                    stat_text = f"Mantel: p={res['p-value']:.2e}, r={res['r']:.3f}"
                    file_prefix = "SIGNIFICANT_"
                    
        # --- Handle symbol/shape ---
        if 'facility_match' in plot_df.columns:
            plot_kwargs['symbol'] = 'facility_match'
            plot_kwargs['hover_data']['facility_match'] = True
            
        # --- Generate Plot ---
        title = f"{plot_title_prefix} ({level}) by {col}"
        if stat_text: title = f"{title}<br><b>{stat_text}</b>"
        
        try:
            fig = px.scatter(
                plot_df, 
                x='Dim1', 
                y='Dim2', 
                title=title, 
                labels=axis_labels, 
                opacity=0.7, 
                **plot_kwargs
            )
            fig.update_layout(legend_title_text=col, margin={'l': 70, 'r': 150, 'b': 70, 't': 120, 'pad': 4})
            fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
            fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
            
            safe_col = re.sub(r'[^A-Za-z0-9_]+', '', col)
            pdir = plot_dir_beta / level; ppath = pdir / f"{file_prefix}{plot_file_prefix}_vs_{safe_col}"
            plot_utils.save_plotly_fig(fig, ppath, batch=True)
            plots_saved += 1
            
        except Exception as e:
            logger.error(f"Failed {ordination_name} plot for {col}: {e}")

    return plots_saved


# ==================================================================================== #
#                       Other Analysis Modules
# ==================================================================================== #

def run_community_state_typing(adata: ad.AnnData, plot_dir_beta: Path, level: str = 'Genus', max_k: int = 10) -> Optional[str]:
    """
    Performs Community State Typing (CST) using K-Medoids clustering on CLR-transformed data.
    """
    logger.info(f"--- Starting Community State Typing (Level: {level}) ---")
    cst_dir = plot_dir_beta / "CST"; cst_dir.mkdir(exist_ok=True, parents=True)
    
    adata_agg = AnalysisUtils.get_analysis_adata(adata, level=level)
    if adata_agg is None or adata_agg.n_obs < max_k or adata_agg.n_vars < 2:
        logger.warning(f"Skipping CST for {level}: Not enough samples or features."); return None
    
    logger.info(f"Running CLR transform for {level}...")
    clr_df = AnalysisUtils._clr_transform(adata_agg, pseudocount=1)
    if clr_df.empty: logger.error("CLR transform failed for CST."); return None
    
    # Ensure data is dense for K-Medoids
    if hasattr(clr_df, 'sparse'):
        clr_data = clr_df.sparse.to_dense().values
    else:
        clr_data = clr_df.values
        
    if not isinstance(clr_data, np.ndarray):
        clr_data = np.asarray(clr_data)

    logger.info(f"Running K-Medoids and Silhouette scoring for k=2 to {max_k}...")
    silhouette_scores = {}
    k_range = range(2, max_k + 1)
    
    for k in tqdm(k_range, desc=f"Clustering {level} (k=2..{max_k})"):
        if k > clr_data.shape[0]:
            logger.warning(f"Skipping k={k}: k is larger than number of samples ({clr_data.shape[0]})")
            break
        try:
            kmedoids = KMedoids(n_clusters=k, metric='euclidean', method='pam', init='k-medoids++', max_iter=300, random_state=42)
            labels = kmedoids.fit_predict(clr_data)
            score = silhouette_score(clr_data, labels, metric='euclidean')
            silhouette_scores[k] = score
        except Exception as e:
            logger.error(f"Failed K-Medoids for k={k}: {e}")
            
    if not silhouette_scores:
        logger.error("CST clustering failed for all k values. Skipping."); return None

    # --- Plot Silhouette Scores ---
    try:
        score_df = pd.DataFrame.from_dict(silhouette_scores, orient='index', columns=['Silhouette Score']).reset_index().rename(columns={'index': 'k'})
        best_k = score_df.loc[score_df['Silhouette Score'].idxmax(), 'k']
        logger.info(f"Best k by Silhouette score: {best_k} (Score: {silhouette_scores[best_k]:.4f})")
        
        fig = px.line(score_df, x='k', y='Silhouette Score', title=f'Silhouette Score by k ({level})', markers=True)
        fig.add_vline(x=best_k, line_dash="dash", line_color="red", annotation_text=f"Best k={best_k}")
        plot_utils.save_plotly_fig(fig, cst_dir / f"cst_silhouette_score_{level}", batch=False)
    except Exception as e:
        logger.error(f"Failed to plot Silhouette scores: {e}")
        best_k = 2 # Fallback
    
    # --- Final Clustering and Saving ---
    logger.info(f"Running final clustering with k={best_k}...")
    kmedoids_final = KMedoids(n_clusters=best_k, metric='euclidean', method='pam', init='k-medoids++', max_iter=300, random_state=42)
    final_labels = kmedoids_final.fit_predict(clr_data)
    
    cst_col_name = f"{level}_CST"
    adata.obs[cst_col_name] = pd.Series(final_labels, index=adata_agg.obs_names).astype(str).astype('category')
    logger.info(f"CST labels added to adata.obs as '{cst_col_name}'")
    
    # --- Plot PCoA of CSTs (if braycurtis was run) ---
    pcoa_key = f'X_pcoa_braycurtis'
    if pcoa_key in adata_agg.obsm:
        logger.info(f"Plotting CSTs on {level} Bray-Curtis PCoA...")
        adata_agg.obs[cst_col_name] = pd.Series(final_labels, index=adata_agg.obs_names).astype(str).astype('category')
        
        try:
            sc.pl.embedding(
                adata_agg, 
                basis=pcoa_key, 
                color=cst_col_name, 
                title=f'Bray-Curtis PCoA ({level}) colored by CST (k={best_k})', 
                show=False
            )
            pcoa_path = cst_dir / f"cst_pcoa_{level}_k{best_k}.png"
            plt.savefig(pcoa_path, dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Failed to plot CST on PCoA: {e}")

    return cst_col_name


def run_taxa_metadata_statistics(adata: ad.AnnData, analysis_levels: List[str], plot_dir_stats: Path, n_cpus: int):
    """
    Performs statistical tests (Spearman, Kruskal-Wallis) comparing taxa
    abundances (CLR transformed) with sample metadata.
    """
    logger.info(f"--- Starting Taxa-Metadata Statistics (Using {n_cpus} CPUs) ---")
    if adata.n_obs < 5: logger.warning("Not enough samples (< 5) for stats. Skipping."); return
        
    metadata_cols = AnalysisUtils.find_plottable_metadata(adata, admin_noise_columns=None, fullness_threshold=0.25, max_categories=50)
    cat_cols = [c for c in metadata_cols['categorical'] if c in adata.obs.columns]
    num_cols = [c for c in metadata_cols['numeric'] if c in adata.obs.columns]
    
    for level in analysis_levels:
        logger.info(f"===== Processing Stats for Level: {level} =====")
        adata_agg = AnalysisUtils.get_analysis_adata(adata, level=level)
        if adata_agg is None or adata_agg.n_obs < 5 or adata_agg.n_vars < 1: logger.warning(f"Skipping {level}: Not enough samples or features."); continue
            
        logger.info(f"Running CLR transform for {level}...")
        clr_df = AnalysisUtils._clr_transform(adata_agg, pseudocount=1)
        if clr_df.empty: logger.error(f"CLR transform failed for {level}. Skipping stats."); continue
        
        # --- Run Categorical Stats (Kruskal-Wallis) ---
        if cat_cols:
            logger.info(f"Running Kruskal-Wallis tests for {level} ({len(cat_cols)} vars, {clr_df.shape[1]} taxa)...")
            for col in cat_cols:
                meta_vector = adata_agg.obs[col].copy()
                if meta_vector.nunique() < 2: continue
                
                logger.debug(f"  -> K-W vs '{col}'")
                kw_results = []
                
                # Prepare data for multiprocessing
                tasks = [(taxon, clr_df[taxon]) for taxon in clr_df.columns]
                kw_func = partial(_calculate_kruskal, meta_vector_shared=meta_vector)
                
                with multiprocessing.Pool(processes=n_cpus) as pool:
                    results_iter = pool.imap_unordered(kw_func, tasks)
                    for res in results_iter:
                        if res: kw_results.append(res)
                        
                if not kw_results: logger.warning(f"K-W tests failed for {col}."); continue
                
                res_df = pd.DataFrame(kw_results).dropna()
                if res_df.empty: logger.warning(f"No valid K-W results for {col}."); continue
                
                res_df['p_adj'] = multipletests(res_df['p_value'], method='fdr_bh')[1]
                sig_df = res_df[res_df['p_adj'] < 0.05].sort_values('p_adj')
                
                if not sig_df.empty:
                    logger.info(f"  -> Found {len(sig_df)} significant taxa for '{col}'")
                    for _, row in sig_df.head(10).iterrows(): # Plot top 10
                        plot_raincloud(meta_vector, clr_df[row['taxon']], col, row['taxon'], plot_dir_stats, level, row['p_adj'])
                    # plot_significant_taxa_heatmap(adata_agg, sig_df, plot_dir_stats, level)
                else:
                    logger.debug(f"  -> No significant taxa found for '{col}'")

        # --- Run Numeric Stats (Spearman) ---
        if num_cols:
            logger.info(f"Running Spearman correlations for {level} ({len(num_cols)} vars, {clr_df.shape[1]} taxa)...")
            for col in num_cols:
                meta_vector = pd.to_numeric(adata_agg.obs[col], errors='coerce')
                if meta_vector.nunique() < 2: continue
                
                logger.debug(f"  -> Spearman vs '{col}'")
                corr_results = []
                
                for taxon in clr_df.columns:
                    try:
                        df_comb = pd.DataFrame({'meta': meta_vector, 'taxa': clr_df[taxon]}).dropna()
                        if df_comb.shape[0] < 5: continue
                        corr, p_val = spearmanr(df_comb['meta'], df_comb['taxa'])
                        if not pd.isna(corr) and not pd.isna(p_val):
                            corr_results.append({'taxon': taxon, 'correlation': corr, 'p_value': p_val})
                    except Exception:
                        continue # Skip if spearmanr fails
                
                if not corr_results: logger.warning(f"Spearman tests failed for {col}."); continue
                
                res_df = pd.DataFrame(corr_results).dropna()
                if res_df.empty: logger.warning(f"No valid Spearman results for {col}."); continue
                
                res_df['p_adj'] = multipletests(res_df['p_value'], method='fdr_bh')[1]
                sig_df = res_df[res_df['p_adj'] < 0.05].sort_values('p_adj')
                
                if not sig_df.empty:
                    logger.info(f"  -> Found {len(sig_df)} significant taxa for '{col}'")
                    for _, row in sig_df.head(10).iterrows(): # Plot top 10
                        plot_correlation(meta_vector, clr_df[row['taxon']], col, row['taxon'], plot_dir_stats, level, row['correlation'], row['p_adj'])
                else:
                    logger.debug(f"  -> No significant taxa found for '{col}'")

    # Flush all queued plots
    plot_utils.flush_plot_queue()
    logger.info("--- Taxa-Metadata Statistics Complete ---")


def run_constrained_ordination(adata: ad.AnnData, analysis_levels: List[str], plot_dir_beta: Path, priority_vars: List[str]):
    """
    Performs Redundancy Analysis (RDA) on CLR-transformed data.
    
    Args:
        adata: AnnData object with microbiome data
        analysis_levels: List of taxonomic levels to analyze
        plot_dir_beta: Directory to save RDA plots
        priority_vars: List of metadata variables to include in RDA
    """
    logger.info("--- Starting Constrained Ordination (RDA) ---")
    
    # Prepare the environmental (metadata) DataFrame
    env_vars = [v for v in priority_vars if v in adata.obs.columns]
    if not env_vars: logger.warning("No priority variables found for RDA. Skipping."); return
    
    env_df = adata.obs[env_vars].copy()
    
    # One-hot encode categorical variables
    cat_vars = [v for v in env_vars if pd.api.types.is_categorical_dtype(env_df[v]) or pd.api.types.is_object_dtype(env_df[v])]
    if cat_vars:
        logger.debug(f"One-hot encoding for RDA: {cat_vars}")
        env_df = pd.get_dummies(env_df, columns=cat_vars, drop_first=True, dummy_na=True, dtype=float)
        
    # Scale and fill numeric variables
    num_vars = [v for v in env_vars if pd.api.types.is_numeric_dtype(env_df[v])]
    if num_vars:
        logger.debug(f"Scaling numeric vars for RDA: {num_vars}")
        env_df[num_vars] = env_df[num_vars].fillna(env_df[num_vars].mean())
        env_df[num_vars] = StandardScaler().fit_transform(env_df[num_vars])

    # Drop any rows with NaNs that might remain (e.g., from all-NaN columns)
    env_df = env_df.dropna()
    if env_df.empty: logger.error("Metadata is empty after prep for RDA. Skipping."); return

    for level in analysis_levels:
        logger.info(f"===== Processing RDA for Level: {level} =====")
        adata_agg = AnalysisUtils.get_analysis_adata(adata, level=level)
        if adata_agg is None or adata_agg.n_obs < 5 or adata_agg.n_vars < 2: logger.warning(f"Skipping {level}: Not enough samples or features."); continue
            
        logger.info(f"Running CLR transform for {level}...")
        clr_df = AnalysisUtils._clr_transform(adata_agg, pseudocount=1)
        if clr_df.empty: logger.error(f"CLR transform failed for {level}. Skipping RDA."); continue
        
        # Align data: only keep samples present in both dataframes
        common_samples = clr_df.index.intersection(env_df.index)
        if len(common_samples) < 3: logger.warning(f"Skipping {level}: < 3 common samples with metadata."); continue
        
        clr_df_aligned = clr_df.loc[common_samples]
        env_df_aligned = env_df.loc[common_samples]
        
        # Ensure env_df has variance
        env_df_aligned = env_df_aligned.loc[:, env_df_aligned.var() > 0]
        if env_df_aligned.shape[1] == 0: logger.warning(f"Skipping {level}: No metadata variables with variance."); continue

        try:
            logger.info(f"Running RDA with {clr_df_aligned.shape[0]} samples and {env_df_aligned.shape[1]} metadata features.")
            rda_results = rda(y=clr_df_aligned, x=env_df_aligned, scale_Y=False) # Y is already CLR
            
            # --- Plot RDA ---
            # Create a dataframe for plotting
            plot_df = pd.DataFrame(rda_results.samples, index=common_samples, columns=['RDA1', 'RDA2'])
            plot_df = plot_df.join(adata_agg.obs) # Join original metadata for colors
            
            rda1_var = rda_results.proportion_explained[0]
            rda2_var = rda_results.proportion_explained[1]
            title = f"RDA Plot ({level})"
            
            for col in priority_vars:
                if col not in plot_df.columns: continue
                
                safe_col = re.sub(r'[^A-Za-z0-9_]+', '', col)
                pdir = plot_dir_beta / level; ppath = pdir / f"RDA_vs_{safe_col}"
                
                plot_kwargs = {'hover_name': plot_df.index.name if plot_df.index.name else 'SampleID'}
                if col in cat_vars: plot_kwargs['color'] = col
                elif col in num_vars: plot_kwargs['color'] = col; plot_kwargs['color_continuous_scale'] = 'viridis'
                
                fig = px.scatter(
                    plot_df,
                    x='RDA1',
                    y='RDA2',
                    title=f"{title} by {col}<br><sup>RDA1 explains {rda1_var:.2%}, RDA2 explains {rda2_var:.2%}</sup>",
                    labels={'RDA1': f"RDA1 ({rda1_var:.2%})", 'RDA2': f"RDA2 ({rda2_var:.2%})"},
                    **plot_kwargs
                )
                plot_utils.save_plotly_fig(fig, ppath, batch=True)
                
        except Exception as e:
            logger.error(f"RDA failed for {level}: {e}")

    plot_utils.flush_plot_queue()
    logger.info("--- Constrained Ordination Complete ---")


def run_network_analysis(adata: ad.AnnData, analysis_levels: List[str], plot_dir_network: Path):
    """
    Performs co-occurrence network analysis using Spearman correlation.
    
    Args:   
        adata: AnnData object with microbiome data
        analysis_levels: List of taxonomic levels to analyze
        plot_dir_network: Directory to save network plots and files
    """
    logger.info("--- Starting Network Analysis ---")
    
    for level in analysis_levels:
        logger.info(f"===== Processing Network for Level: {level} =====")
        plot_dir_network_level = plot_dir_network / level
        plot_dir_network_level.mkdir(exist_ok=True, parents=True)
        
        adata_agg = AnalysisUtils.get_analysis_adata(adata, level=level)
        if adata_agg is None or adata_agg.n_obs < 10 or adata_agg.n_vars < 5: logger.warning(f"Skipping {level}: Not enough samples (>10) or features (>5)."); continue
            
        # Use log1p normalized data (from scanpy) for network
        adata_net = adata_agg.copy()
        adata_net.X = adata_net.layers['raw_counts'].copy()
        sc.pp.normalize_total(adata_net, target_sum=1e4)
        sc.pp.log1p(adata_net)
        
        # Feature selection: keep top 100 most abundant features
        if adata_net.n_vars > 100:
            logger.info(f"Filtering {level} from {adata_net.n_vars} to 100 most abundant features for network.")
            top_indices = np.argsort(np.array(adata_net.X.mean(0)).flatten())[-100:]
            adata_net = adata_net[:, top_indices].copy()
            
        data_matrix = adata_net.X.toarray() if issparse(adata_net.X) else adata_net.X
        
        try:
            # Calculate Spearman correlation matrix
            corr_matrix, p_matrix = spearmanr(data_matrix, axis=0)
            
            # Set diagonals to 0
            np.fill_diagonal(corr_matrix, 0)
            
            # --- Create NetworkX Graph ---
            G = nx.Graph()
            
            # Define thresholds
            corr_threshold = 0.5
            p_adj_threshold = 0.05
            
            taxa_names = adata_net.var_names
            
            # Adjust p-values (this is slow)
            # Flatten upper triangle for p-value adjustment
            p_values_flat = p_matrix[np.triu_indices(p_matrix.shape[0], k=1)]
            if len(p_values_flat) == 0:
                logger.warning(f"No edges to test for {level}. Skipping network.")
                continue
                
            reject, p_adj, _, _ = multipletests(p_values_flat, method='fdr_bh')
            
            # Reconstruct adjusted p-value matrix
            p_adj_matrix = np.ones_like(p_matrix)
            p_adj_matrix[np.triu_indices(p_adj_matrix.shape[0], k=1)] = p_adj
            p_adj_matrix = p_adj_matrix + p_adj_matrix.T - np.diag(p_adj_matrix.diagonal())

            # Add nodes
            for i, taxon in enumerate(taxa_names):
                G.add_node(taxon, abundance=data_matrix[:, i].mean())
                
            # Add edges based on thresholds
            for i in range(len(taxa_names)):
                for j in range(i + 1, len(taxa_names)):
                    corr = corr_matrix[i, j]
                    p_adj_val = p_adj_matrix[i, j]
                    
                    if p_adj_val < p_adj_threshold and abs(corr) > corr_threshold:
                        weight = corr
                        edge_type = 'positive' if corr > 0 else 'negative'
                        G.add_edge(taxa_names[i], taxa_names[j], weight=abs(weight), type=edge_type, sign=weight)

            if G.number_of_edges() == 0:
                logger.warning(f"No significant edges found for {level} network. Skipping plot."); continue
                
            # --- Save Graph file ---
            gml_path = plot_dir_network_level / f"network_{level}_corr{corr_threshold}_padj{p_adj_threshold}.gml"
            nx.write_gml(G, str(gml_path))
            logger.info(f"Network for {level} saved to: {gml_path} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
            
            # --- Plotting with Plotly ---
            if G.number_of_nodes() > 150:
                logger.warning(f"Skipping network plot for {level}: Too many nodes ({G.number_of_nodes()}).")
                continue
                
            logger.info(f"Generating network plot for {level}...")
            # Get positions from the layout
            pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

            # --- 1. Create Edge Traces ---
            edge_x_pos = []
            edge_y_pos = []
            edge_x_neg = []
            edge_y_neg = []
            
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_type = edge[2].get('type', 'positive')
                
                if edge_type == 'positive':
                    edge_x_pos.extend([x0, x1, None])
                    edge_y_pos.extend([y0, y1, None])
                else:
                    edge_x_neg.extend([x0, x1, None])
                    edge_y_neg.extend([y0, y1, None])

            edge_trace_pos = go.Scatter(
                x=edge_x_pos, y=edge_y_pos,
                line=dict(width=0.5, color='rgba(0,100,255,0.5)'),
                hoverinfo='none',
                mode='lines',
                name='Positive Correlation')

            edge_trace_neg = go.Scatter(
                x=edge_x_neg, y=edge_y_neg,
                line=dict(width=0.5, color='rgba(255,50,0,0.5)'),
                hoverinfo='none',
                mode='lines',
                name='Negative Correlation')
            
            # --- 2. Create Node Trace ---
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            node_info = []

            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                
                # Get abundance for size (with a fallback)
                abundance = G.nodes[node].get('abundance', 0)
                node_size.append(5 + (np.log1p(abundance) * 5)) # Scale size
                
                # Get degree for hover info
                degree = G.degree(node)
                node_info.append(f"{node}<br>Degree: {degree}<br>Mean Abundance: {abundance:.3f}")

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition='top center',
                textfont=dict(size=8),
                hoverinfo='text',
                hovertext=node_info,
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    reversescale=False,
                    color=[G.degree(node) for node in G.nodes()], # Color by degree
                    size=node_size,
                    colorbar=dict(
                        thickness=15,
                        title='Node Degree',
                        xanchor='left',
                        titleside='right'
                    ),
                    line_width=1,
                    line_color='black'
                ),
                name='Taxa')
                
            # --- 3. Create Figure ---
            fig = go.Figure(data=[edge_trace_pos, edge_trace_neg, node_trace],
                 layout=go.Layout(
                    title=f'Co-occurrence Network ({level}) | Corr > {corr_threshold}, p.adj < {p_adj_threshold}',
                    titlefont_size=16,
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Network visualization via Plotly. GML file saved for Cytoscape/Gephi.",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
            
            # --- 4. Save Figure ---
            plot_path = plot_dir_network_level / f"network_plot_{level}.html"
            plot_utils.save_plotly_fig(fig, plot_path, batch=False) # Don't batch network plots
            logger.info(f"Network plot for {level} saved to: {plot_path}")
            
        except Exception as e:
            logger.error(f"Network analysis failed for {level}: {e}")

    logger.info("--- Network Analysis Complete ---")