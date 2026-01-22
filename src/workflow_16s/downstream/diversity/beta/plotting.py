"""
Plotting utilities for beta diversity ordinations.

Handles visualization of PCoA, UMAP, and RDA results colored by metadata.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd
import plotly.express as px
import scanpy as sc
from scipy.sparse import issparse

from workflow_16s.downstream.visualization import PlottingUtils
from workflow_16s.utils.logger import get_logger

logger = get_logger("workflow_16s")
plot_utils = PlottingUtils(logger)


def plot_ordination(
    adata_agg: ad.AnnData,
    ordination_name: str,
    level: str,
    plottable_categorical: List[str],
    plottable_numeric: List[str],
    plot_dir_beta: Path,
    dist_name: str = "",
    axis_labels: Optional[Dict[str, str]] = None,
    permanova_results: Optional[Dict[str, Any]] = None,
    mantel_results: Optional[Dict[str, Any]] = None
) -> int:
    """
    Generate ordination plots colored by metadata variables.

    Creates scatter plots of ordination results (PCoA or UMAP) with points
    colored by categorical or numeric metadata. Significant statistical
    associations are annotated in plot titles.

    Parameters
    ----------
    adata_agg : ad.AnnData
        Aggregated AnnData object with ordination results in .obsm
    ordination_name : str
        Type of ordination ('PCoA' or 'UMAP')
    level : str
        Taxonomic level being analyzed (e.g., 'Genus')
    plottable_categorical : List[str]
        Categorical metadata columns to plot
    plottable_numeric : List[str]
        Numeric metadata columns to plot
    plot_dir_beta : Path
        Output directory for plots
    dist_name : str, optional
        Distance metric name for PCoA (e.g., 'braycurtis')
    axis_labels : Optional[Dict[str, str]], optional
        Custom axis labels with keys 'x' and 'y'
    permanova_results : Optional[Dict[str, Any]], optional
        PERMANOVA results for categorical variables
    mantel_results : Optional[Dict[str, Any]], optional
        Mantel test results for numeric variables

    Returns
    -------
    int
        Number of plots successfully generated

    Notes
    -----
    - Top 20 most abundant taxa are added to hover data
    - Significant results (p < 0.05) are prefixed with 'SIGNIFICANT_'
    - facility_match is used as point shape if available
    - Plots are batched for efficient saving
    """
    plots_saved = 0
    permanova_results = permanova_results or {}
    mantel_results = mantel_results or {}

    # Configure based on ordination type
    if ordination_name == 'PCoA':
        obsm_key = f'X_pcoa_{dist_name}'
        plot_title_prefix = f"{dist_name} PCoA"
        plot_file_prefix = f"{dist_name}_PCoA"
        if axis_labels is None:
            axis_labels = {'x': 'PC1', 'y': 'PC2'}
    elif ordination_name == 'UMAP':
        obsm_key = 'X_umap'
        plot_title_prefix = "UMAP"
        plot_file_prefix = "UMAP"
        if axis_labels is None:
            axis_labels = {'x': 'UMAP 1', 'y': 'UMAP 2'}
    else:
        return 0

    if obsm_key not in adata_agg.obsm:
        return 0

    # Prepare normalized data for hover information
    adata_agg_norm = adata_agg.copy()
    adata_agg_norm.X = adata_agg_norm.layers['raw_counts'].copy()
    sc.pp.normalize_total(adata_agg_norm, target_sum=1e4)
    sc.pp.log1p(adata_agg_norm)

    # Get top abundant taxa
    new_col_names = []
    if adata_agg_norm.n_vars > 0:
        mean_abund = np.array(adata_agg_norm.X.mean(0)).flatten()
        n_top = min(20, len(mean_abund))
        top_indices = np.argsort(mean_abund)[-n_top:][::-1]
        top_taxa_names = adata_agg_norm.var_names[top_indices].tolist()

        abund_data = (adata_agg_norm.X[:, top_indices].toarray()
                     if issparse(adata_agg_norm.X)
                     else np.asarray(adata_agg_norm.X[:, top_indices]))

        new_col_names = [f"Feature: {name}" for name in top_taxa_names]
        taxa_df = pd.DataFrame(
            abund_data,
            index=adata_agg_norm.obs_names,
            columns=new_col_names
        )
    else:
        taxa_df = pd.DataFrame(index=adata_agg_norm.obs_names)

    # Extract ordination coordinates
    full_ord_data = (adata_agg.obsm[obsm_key].values
                    if isinstance(adata_agg.obsm[obsm_key], pd.DataFrame)
                    else adata_agg.obsm[obsm_key])

    if full_ord_data.shape[1] >= 2:
        ord_data = full_ord_data[:, :2]
        c_names = ['Dim1', 'Dim2']
    else:
        ord_data = np.column_stack([full_ord_data[:, 0], np.zeros(full_ord_data.shape[0])])
        c_names = ['Dim1', 'Dim2 (dummy)']

    # Build plotting dataframe
    ord_df = (pd.DataFrame(ord_data, columns=c_names, index=adata_agg.obs_names)
              .join(adata_agg.obs)
              .join(taxa_df))

    # Generate plots for each metadata variable
    for col in (plottable_categorical + plottable_numeric + new_col_names):
        if col not in ord_df.columns:
            continue

        plot_df = ord_df.copy().replace(pd.NA, np.nan).dropna(subset=[col])
        if plot_df.shape[0] < 2:
            continue

        plot_kwargs: Dict[str, Any] = {
            'hover_name': plot_df.index.name or 'SampleID',
            'hover_data': {col: True}
        }

        stat_text = ""
        file_prefix = ""

        # Handle categorical variables
        if col in plottable_categorical:
            plot_kwargs['color'] = col

            if col in permanova_results:
                res = permanova_results[col]
                if res['p-value'] < 0.05:
                    stat_text = f"PERMANOVA: p={res['p-value']:.2e}, F={res['test statistic']:.2f}"
                    file_prefix = "SIGNIFICANT_"

            # Truncate long category names
            if (pd.api.types.is_object_dtype(plot_df[col].dtype) or
                isinstance(plot_df[col].dtype, pd.CategoricalDtype)):
                plot_df[col] = plot_df[col].astype(str).str.slice(0, 50)

        # Handle numeric variables
        else:
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

        # Add facility_match as symbol if available
        if 'facility_match' in plot_df.columns:
            plot_kwargs['symbol'] = 'facility_match'

        # Create title with statistics
        title = f"{plot_title_prefix} ({level}) by {col}"
        if stat_text:
            title = f"{title}<br><b>{stat_text}</b>"

        # Generate and save plot
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

            plot_utils.save_plotly_fig(
                fig,
                plot_dir_beta / level / f"{file_prefix}{plot_file_prefix}_vs_{re.sub(r'[^A-Za-z0-9_]+', '', col)}",
                batch=True
            )
            plots_saved += 1

        except Exception as e:
            logger.error(f"Plot failed for {col}: {e}")

    return plots_saved