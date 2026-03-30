# workflow_16s/visualization/metadata.py

import re
from pathlib import Path
from typing import List

import anndata as ad
import plotly.express as px
import pandas as pd
import numpy as np

from workflow_16s.utils.logger import get_logger
from workflow_16s.visualization.utils import PlottingUtils
from workflow_16s.downstream.utils import AnalysisUtils



def plot_stacked_bar(adata: ad.AnnData, cst_col: str, plottable_cat: List[str], target_path: Path):
    """Generates 100% stacked bar charts for a primary categorical column (e.g., CST) against a list of other metadata columns."""
    # Use the parent directory from the target_path
    target_path = Path(target_path)
    plot_dir = target_path.parent
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("workflow_16s")
    if not plottable_cat: 
        logger.warning("No plottable categorical columns provided for stacked bar plot."); return
    logger.info(f"Generating stacked bar plots for '{cst_col}'...")

    for meta_col in plottable_cat:
        if meta_col not in adata.obs.columns: 
            logger.warning(f"Skipping stacked bar plot: '{meta_col}' not in adata.obs."); continue  
        try:
            # Convert both columns to string to handle mixed types 
            cst_data_str = adata.obs[cst_col].astype(str).fillna('Unknown')
            meta_data_str = adata.obs[meta_col].astype(str).fillna('Unknown')
            # 1. Create a contingency table (counts)
            contingency_table = pd.crosstab(meta_data_str, cst_data_str)
            # 2. Normalize to get percentages (100% stacked bar)
            normalized_table = contingency_table.div(contingency_table.sum(axis=1), axis=0)
            # 3. Melt for Plotly (long format)
            plot_df = normalized_table.reset_index().melt(id_vars=meta_col, var_name=cst_col, value_name='Percentage')
            # 4. Create the figure
            fig = px.bar(
                plot_df, x=meta_col, y='Percentage', color=cst_col, 
                title=f'Community State Type Distribution by {meta_col}', 
                labels={
                    meta_col: meta_col.replace('_', ' ').capitalize(), 
                    cst_col: cst_col.replace('_', ' ').capitalize()
                }, text_auto=True
            )
            fig.update_traces(texttemplate='%{y:.1%}')
            fig.update_layout(
                xaxis_title=meta_col.replace('_', ' ').capitalize(), 
                yaxis_title='Proportion of Samples', yaxis_tickformat='.0%'
            )
            # 5. Save the figure using the class's save method
            # We use batch=False to save immediately, since the orchestrator calls flush() right after this.
            safe_col_name = re.sub(r'[^A-Za-z0-9_]+', '', meta_col)
            # We use the plot_dir, not the full target_path
            save_path_stem = plot_dir / f"cst_vs_{safe_col_name}_bar"
            PlottingUtils(logger).save_plotly_fig(fig, save_path_stem, batch=False) 
        except Exception as e: logger.error(f"Failed to generate stacked bar plot for {meta_col}: {e}")
            
                
def plot_metadata_pairplot(adata: ad.AnnData, plot_dir_meta: Path, max_vars: int = 10, save_scale: int = 2):
    """Creates scatter matrix for top numerical metadata."""
    logger = get_logger("workflow_16s")
    logger.info(f"--- Generating Metadata Pair Plot (Top {max_vars} Numeric) ---")
    if adata is None: logger.error("AnnData object not loaded."); return
    meta_vars = AnalysisUtils.find_plottable_metadata(adata, fullness_threshold=0.5); numeric_vars = meta_vars['numeric']
    if len(numeric_vars) < 2: logger.info("Skipping pair plot"); return
    vars_to_plot = sorted(numeric_vars)[:max_vars]; logger.info(f"Plotting pair plot for: {vars_to_plot}")
    # Replace pd.NA with np.nan to prevent kaleido save error 
    plot_df = adata.obs[vars_to_plot].copy().replace(pd.NA, np.nan)
    color_col = 'facility_match' if 'facility_match' in adata.obs.columns else None
    if color_col:
        fm_data = adata.obs[color_col]
        if isinstance(fm_data.dtype, pd.CategoricalDtype):
            if 'Unknown' not in fm_data.cat.categories: fm_data = fm_data.cat.add_categories('Unknown')
            plot_df[color_col] = fm_data.fillna('Unknown')
        else: plot_df[color_col] = fm_data.astype(str).fillna('Unknown')
    logger.debug(f"Pairplot df shape: {plot_df.shape}")
    if plot_df.empty: logger.warning("Pairplot DataFrame is empty before plotting."); return
    try:
        n_vars = len(vars_to_plot); base_size = 900; plot_height = min(max(base_size, n_vars * 175), 2500); plot_width = min(max(base_size, n_vars * 175), 2500)
        fig = px.scatter_matrix(plot_df, dimensions=vars_to_plot, color=color_col, title="Pairwise Relationships of Numerical Metadata")  
        fig.update_traces(diagonal_visible=True, showupperhalf=True, marker=dict(size=5, opacity=0.7), selector=dict(type='scatter'))
        fig.update_layout(height=plot_height, width=plot_width, font_size=max(10, 22 - n_vars), legend_font_size=max(10, 22 - n_vars), title_font_size=max(16, 32 - n_vars))
        plot_path = plot_dir_meta / "metadata_pairplot"; PlottingUtils(logger).save_plotly_fig(fig, plot_path); logger.info(f"Saved metadata pair plot: {plot_path}")
    except Exception as e: logger.error(f"Failed metadata pair plot: {e}")


def plot_metadata_correlation_heatmap(adata: ad.AnnData, plot_dir_meta: Path, save_scale: int = 2):
    """Calculates and plots Spearman correlation heatmap for numerical metadata."""
    logger = get_logger("workflow_16s")
    logger.info("--- Generating Metadata Correlation Heatmap ---")
    if adata is None: logger.error("AnnData object not loaded."); return
    meta_vars = AnalysisUtils.find_plottable_metadata(adata, fullness_threshold=0.5); numeric_vars = meta_vars['numeric']
    if len(numeric_vars) < 2: logger.info("Skipping correlation heatmap"); return
    logger.info(f"Calculating Spearman correlation matrix for {len(numeric_vars)} variables."); 
    # Replace pd.NA with np.nan to prevent kaleido save error 
    numeric_df = adata.obs[numeric_vars].copy().replace(pd.NA, np.nan)
    for col in numeric_df.columns:
        try: numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce').astype(float)
        except Exception as e: logger.warning(f"Could not convert '{col}' to float: {e}. Dropping."); numeric_df = numeric_df.drop(columns=[col])
    if numeric_df.empty or numeric_df.shape[1] < 2: logger.warning("Skipping correlation heatmap."); return
    try: numeric_df.fillna(numeric_df.mean(), inplace=True)
    except Exception as e: logger.error(f"Failed during fillna for heatmap: {e}"); return
    try: corr_matrix = numeric_df.corr(method='spearman')
    except Exception as e: logger.error(f"Failed correlation matrix calculation: {e}"); return
    if isinstance(corr_matrix, np.ndarray): corr_matrix_df = pd.DataFrame(corr_matrix, index=numeric_vars, columns=numeric_vars) 
    else: corr_matrix_df = corr_matrix 
    try:
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', color_continuous_midpoint=0, zmin=-1, zmax=1, title="Spearman Correlation of Numerical Metadata")
        fig.update_traces(texttemplate="%{z:.2f}", textfont_size=max(8, 14 - len(numeric_vars) // 2))
        fig.update_xaxes(side="bottom", tickangle=-90)
        n_vars = len(numeric_vars); plot_height = max(800, n_vars * 50); plot_width = max(900, n_vars * 55)
        fig.update_layout(height=plot_height, width=plot_width, margin=dict(l=200, r=50, b=200, t=100))
        plot_path = plot_dir_meta / "metadata_correlation_heatmap"; PlottingUtils(logger).save_plotly_fig(fig, plot_path); logger.info(f"Saved metadata correlation heatmap: {plot_path}")
    except Exception as e: logger.error(f"Failed correlation heatmap plot: {e}")