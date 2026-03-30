# ==================================================================================== #
# diversity/alpha/rarefaction.py
# Rarefaction Curves for Alpha Diversity
# ==================================================================================== #

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.sparse import issparse, csr_matrix
from skbio.diversity import alpha_diversity
import anndata as ad

from workflow_16s.visualization.utils import PlottingUtils
from workflow_16s.utils.logger import get_logger

# ==================================================================================== #

def subsample_counts(counts_row: np.ndarray, depth: int, random_state: int = 42) -> np.ndarray:
    """
    Randomly subsample a count vector to a specified depth.
    
    Parameters
    ----------
    counts_row : np.ndarray
        Count vector for a single sample
    depth : int
        Target sequencing depth for subsampling
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Subsampled count vector
    """
    total = counts_row.sum()
    if depth > total:
        return counts_row
    
    np.random.seed(random_state)
    
    # Create pool of feature indices based on counts
    feature_pool = []
    for feature_idx, count in enumerate(counts_row):
        feature_pool.extend([feature_idx] * int(count))
    
    # Randomly sample without replacement
    sampled_indices = np.random.choice(feature_pool, size=depth, replace=False)
    
    # Count occurrences
    subsampled = np.zeros_like(counts_row)
    unique, counts = np.unique(sampled_indices, return_counts=True)
    subsampled[unique] = counts
    
    return subsampled


def calculate_rarefaction_curve(
    counts_row: np.ndarray,
    depths: List[int],
    metric: str = 'observed_features',
    n_iterations: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate rarefaction curve for a single sample.
    
    Parameters
    ----------
    counts_row : np.ndarray
        Count vector for sample
    depths : List[int]
        Sequencing depths to evaluate
    metric : str, optional
        Alpha diversity metric ('observed_features', 'shannon'), by default 'observed_features'
    n_iterations : int, optional
        Number of subsampling iterations per depth, by default 10
    random_state : int, optional
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Rarefaction curve data (depth, metric_mean, metric_std)
    """
    results = []
    
    for depth in depths:
        if depth > counts_row.sum():
            continue
            
        metric_values = []
        for i in range(n_iterations):
            subsampled = subsample_counts(counts_row, depth, random_state + i)
            
            if metric == 'observed_features':
                value = np.count_nonzero(subsampled)
            elif metric == 'shannon':
                # Calculate Shannon index
                value = alpha_diversity(metric, subsampled.reshape(1, -1), ids=['sample'])[0]
            else:
                raise ValueError(f"Unsupported metric: {metric}")
                
            metric_values.append(value)
        
        results.append({
            'depth': depth,
            f'{metric}_mean': np.mean(metric_values),
            f'{metric}_std': np.std(metric_values)
        })
    
    return pd.DataFrame(results)


def generate_rarefaction_curves(
    adata: ad.AnnData,
    output_dir: Path,
    metric: str = 'observed_features',
    n_depths: int = 20,
    n_iterations: int = 10,
    max_samples_to_plot: int = 50,
    group_col: Optional[str] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate rarefaction curves for all samples in AnnData object.
    
    Rarefaction curves assess whether sequencing depth is sufficient to capture
    the full diversity of microbial communities. Curves that plateau indicate
    adequate sampling, while curves that continue increasing suggest deeper
    sequencing would reveal more diversity.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with raw count data
    output_dir : Path
        Directory to save plots
    metric : str, optional
        Alpha diversity metric, by default 'observed_features'
    n_depths : int, optional
        Number of depth points to evaluate, by default 20
    n_iterations : int, optional
        Subsampling iterations per depth, by default 10
    max_samples_to_plot : int, optional
        Maximum number of samples to plot individually, by default 50
    group_col : Optional[str], optional
        Metadata column for grouping samples, by default None
    random_state : int, optional
        Random seed
        
    Returns
    -------
    pd.DataFrame
        All rarefaction curve data
        
    Notes
    -----
    - Requires 'raw_counts' layer in adata
    - Generates individual sample curves and group-averaged curves
    - Saves interactive HTML plots
    """
    logger = get_logger("workflow_16s")
    plot_utils = PlottingUtils(logger)
    logger.info(f"--- Generating Rarefaction Curves ({metric}) ---")
    
    if 'raw_counts' not in adata.layers:
        logger.error("'raw_counts' layer not found. Cannot generate rarefaction curves.")
        return pd.DataFrame()
    
    counts_matrix = adata.layers['raw_counts']
    if issparse(counts_matrix):
        counts_matrix = counts_matrix.toarray() # type: ignore
    
    # Determine depth range
    sample_depths = counts_matrix.sum(axis=1)
    min_depth = int(sample_depths.min())
    max_depth = int(sample_depths.max())
    
    depths = np.linspace(min_depth, max_depth, n_depths, dtype=int)
    
    logger.info(f"Depth range: {min_depth:,} to {max_depth:,} reads")
    logger.info(f"Calculating rarefaction for {adata.n_obs} samples...")
    
    all_curves = []
    
    for sample_idx, sample_id in enumerate(adata.obs_names):
        counts_row = counts_matrix[sample_idx, :]
        
        curve_df = calculate_rarefaction_curve(
            counts_row, depths, metric, n_iterations, random_state # type: ignore
        )
        curve_df['sample_id'] = sample_id
        
        if group_col and group_col in adata.obs.columns:
            curve_df['group'] = adata.obs.loc[sample_id, group_col]
        
        all_curves.append(curve_df)
        
        if (sample_idx + 1) % 100 == 0:
            logger.info(f"  Processed {sample_idx + 1}/{adata.n_obs} samples")
    
    all_curves_df = pd.concat(all_curves, ignore_index=True)
    
    # Plot individual samples (limited to max_samples_to_plot)
    n_samples_to_plot = min(adata.n_obs, max_samples_to_plot)
    samples_to_plot = adata.obs_names[:n_samples_to_plot]
    
    fig = go.Figure()
    
    for sample_id in samples_to_plot:
        sample_data = all_curves_df[all_curves_df['sample_id'] == sample_id]
        
        fig.add_trace(go.Scatter(
            x=sample_data['depth'],
            y=sample_data[f'{metric}_mean'],
            mode='lines',
            name=sample_id,
            line=dict(width=1),
            opacity=0.6,
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"Rarefaction Curves: {metric.replace('_', ' ').title()} ({n_samples_to_plot} samples)",
        xaxis_title="Sequencing Depth (reads)",
        yaxis_title=metric.replace('_', ' ').title(),
        hovermode='closest',
        height=600,
        width=1000
    )
    
    plot_path = output_dir / f"rarefaction_curves_{metric}_individual.html"
    plot_utils.save_plotly_fig(fig, plot_path)
    logger.info(f"Saved individual rarefaction curves: {plot_path.name}")
    
    # Group-averaged curves
    if group_col and group_col in adata.obs.columns:
        fig_grouped = go.Figure()
        
        for group_name in all_curves_df['group'].unique():
            if pd.isna(group_name):
                continue
                
            group_data = all_curves_df[all_curves_df['group'] == group_name]
            
            # Average across samples within each depth
            grouped = group_data.groupby('depth').agg({
                f'{metric}_mean': ['mean', 'std']
            }).reset_index()
            
            grouped.columns = ['depth', 'mean', 'std']
            
            fig_grouped.add_trace(go.Scatter(
                x=grouped['depth'],
                y=grouped['mean'],
                mode='lines+markers',
                name=str(group_name),
                line=dict(width=2),
                error_y=dict(
                    type='data',
                    array=grouped['std'],
                    visible=True
                )
            ))
        
        fig_grouped.update_layout(
            title=f"Rarefaction Curves by {group_col}: {metric.replace('_', ' ').title()}",
            xaxis_title="Sequencing Depth (reads)",
            yaxis_title=f"{metric.replace('_', ' ').title()} (mean ± SD)",
            hovermode='closest',
            height=600,
            width=1000,
            legend=dict(title=group_col)
        )
        
        plot_path_grouped = output_dir / f"rarefaction_curves_{metric}_by_{group_col}.html"
        plot_utils.save_plotly_fig(fig_grouped, plot_path_grouped)
        logger.info(f"Saved grouped rarefaction curves: {plot_path_grouped.name}")
    
    # Summary statistics
    logger.info("\n=== Rarefaction Curve Summary ===")
    logger.info(f"Metric: {metric}")
    logger.info(f"Depth range: {min_depth:,} - {max_depth:,} reads")
    
    # Check for plateau (increase < 5% in last 20% of depth range)
    plateau_threshold = 0.05
    last_20pct_depths = depths[int(0.8 * len(depths)):]
    
    plateau_count = 0
    for sample_id in adata.obs_names:
        sample_data = all_curves_df[all_curves_df['sample_id'] == sample_id]
        sample_data = sample_data.sort_values('depth')
        
        last_20pct_data = sample_data[sample_data['depth'].isin(last_20pct_depths)]
        if len(last_20pct_data) >= 2:
            first_val = last_20pct_data[f'{metric}_mean'].iloc[0]
            last_val = last_20pct_data[f'{metric}_mean'].iloc[-1]
            
            if first_val > 0:
                pct_increase = (last_val - first_val) / first_val
                if pct_increase < plateau_threshold:
                    plateau_count += 1
    
    pct_plateau = (plateau_count / adata.n_obs) * 100
    logger.info(f"Samples reaching plateau: {plateau_count}/{adata.n_obs} ({pct_plateau:.1f}%)")
    
    if pct_plateau < 50:
        logger.warning(
            f"⚠️ Only {pct_plateau:.1f}% of samples show plateau. "
            "Consider deeper sequencing for comprehensive diversity assessment."
        )
    else:
        logger.info(f"✓ {pct_plateau:.1f}% of samples show plateau - adequate sampling depth.")
    
    return all_curves_df
