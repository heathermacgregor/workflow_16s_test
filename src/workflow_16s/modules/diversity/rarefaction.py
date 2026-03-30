"""
Rarefaction Curves for Sequencing Depth Assessment

Shows whether sequencing depth was adequate to capture community diversity.
Curves should plateau if sequencing was sufficient.
"""

import numpy as np
import pandas as pd
import anndata as ad
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import logging

from workflow_16s.utils.logger import get_logger


def calculate_rarefaction_curve(counts: np.ndarray, 
                                step_size: Optional[int] = None,
                                n_steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate rarefaction curve for a single sample.
    
    Parameters
    ----------
    counts : array-like
        Feature counts for one sample
    step_size : int, optional
        Step size for rarefaction depths
    n_steps : int
        Number of rarefaction depths to sample
    
    Returns
    -------
    tuple
        (depths, richness) arrays for plotting
    """
    logger = get_logger("workflow_16s")
    total_reads = int(np.sum(counts))
    
    if total_reads == 0:
        return np.array([0]), np.array([0])
    
    # Rarefaction depths to sample
    if step_size is None:
        step_size = max(1, total_reads // n_steps)
    
    depths = np.arange(0, total_reads + 1, step_size)
    depths = np.append(depths, total_reads)  # Ensure we include final depth
    depths = np.unique(depths)  # Remove duplicates
    
    richness = np.zeros(len(depths))
    
    for i, depth in enumerate(depths):
        if depth == 0:
            richness[i] = 0
            continue
        
        # Rarefaction using multinomial sampling (100 iterations for stability)
        observed_richness = []
        for _ in range(100):
            # Subsample without replacement
            rarefied = np.random.multinomial(depth, counts / total_reads)
            observed_richness.append(np.sum(rarefied > 0))
        
        richness[i] = np.mean(observed_richness)
    
    return depths, richness


def rarefaction_curves_for_dataset(adata: ad.AnnData,
                                   sample_n: Optional[int] = None,
                                   n_steps: int = 10) -> pd.DataFrame:
    """
    Calculate rarefaction curves for all samples in AnnData.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with counts
    sample_n : int, optional
        Randomly sample N samples (for large datasets)
    n_steps : int
        Number of rarefaction depths per sample
    
    Returns
    -------
    DataFrame
        Long-format dataframe with columns: sample_id, depth, richness
    """
    logger = get_logger("workflow_16s")
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    n_samples = X.shape[0]
    
    # Sample if dataset is large
    if sample_n and n_samples > sample_n:
        logger.info(f'Sampling {sample_n} of {n_samples} samples for rarefaction')
        idx = np.random.choice(n_samples, sample_n, replace=False)
        X = X[idx, :]
        sample_names = adata.obs_names[idx]
    else:
        sample_names = adata.obs_names
    
    results = []
    
    for i, sample_name in enumerate(sample_names):
        if i % 100 == 0:
            logger.info(f'Processing sample {i+1}/{len(sample_names)}...')
        
        depths, richness = calculate_rarefaction_curve(X[i, :], n_steps=n_steps)
        
        for d, r in zip(depths, richness):
            results.append({
                'sample_id': sample_name,
                'depth': d,
                'richness': r
            })
    
    return pd.DataFrame(results)


def plot_rarefaction_curves(rarefaction_df: pd.DataFrame,
                            color_by: Optional[str] = None,
                            metadata: Optional[pd.DataFrame] = None,
                            sample_n: int = 50,
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot rarefaction curves with optional grouping.
    
    Parameters
    ----------
    rarefaction_df : DataFrame
        Output from rarefaction_curves_for_dataset()
    color_by : str, optional
        Column to color curves by (requires metadata)
    metadata : DataFrame, optional
        Sample metadata with color_by column
    sample_n : int
        Maximum number of samples to plot individually
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    
    Returns
    -------
    Figure
        Matplotlib figure object
    """
    logger = get_logger("workflow_16s")
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Individual curves
    unique_samples = rarefaction_df['sample_id'].unique()
    
    if len(unique_samples) > sample_n:
        # Too many samples, sample randomly
        sampled = np.random.choice(unique_samples, sample_n, replace=False)
        plot_df = rarefaction_df[rarefaction_df['sample_id'].isin(sampled)]
    else:
        plot_df = rarefaction_df
    
    # Plot individual curves
    for sample in plot_df['sample_id'].unique():
        sample_data = plot_df[plot_df['sample_id'] == sample]
        axes[0].plot(sample_data['depth'], sample_data['richness'], 
                    alpha=0.3, linewidth=0.5, color='gray')
    
    axes[0].set_xlabel('Sequencing Depth (reads)')
    axes[0].set_ylabel('Observed Richness (# features)')
    axes[0].set_title(f'Rarefaction Curves (n={len(plot_df["sample_id"].unique())})')
    axes[0].grid(alpha=0.3)
    
    # Right plot: Mean curve with CI
    depth_bins = np.linspace(0, rarefaction_df['depth'].max(), 20)
    mean_richness = []
    ci_lower = []
    ci_upper = []
    
    for i in range(len(depth_bins) - 1):
        bin_data = rarefaction_df[
            (rarefaction_df['depth'] >= depth_bins[i]) &
            (rarefaction_df['depth'] < depth_bins[i+1])
        ]
        
        if len(bin_data) > 0:
            mean_richness.append(bin_data['richness'].mean())
            ci_lower.append(bin_data['richness'].quantile(0.025))
            ci_upper.append(bin_data['richness'].quantile(0.975))
        else:
            mean_richness.append(np.nan)
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)
    
    bin_centers = (depth_bins[:-1] + depth_bins[1:]) / 2
    
    axes[1].plot(bin_centers, mean_richness, color='blue', linewidth=2, 
                label='Mean')
    axes[1].fill_between(bin_centers, ci_lower, ci_upper, 
                         alpha=0.3, color='blue', label='95% CI')
    
    axes[1].set_xlabel('Sequencing Depth (reads)')
    axes[1].set_ylabel('Observed Richness (# features)')
    axes[1].set_title('Mean Rarefaction Curve')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f'Rarefaction curve plot saved to {save_path}')
    
    return fig


def assess_sequencing_adequacy(rarefaction_df: pd.DataFrame,
                               plateau_threshold: float = 0.95) -> dict:
    """
    Assess whether sequencing depth was adequate.
    
    Curve is "adequate" if it reaches a plateau (final richness is close to asymptote).
    
    Parameters
    ----------
    rarefaction_df : DataFrame
        Output from rarefaction_curves_for_dataset()
    plateau_threshold : float
        Fraction of asymptote required to consider adequate (default 0.95)
    
    Returns
    -------
    dict
        Statistics on sequencing adequacy
    """
    logger = get_logger("workflow_16s")
    results = {}
    
    for sample in rarefaction_df['sample_id'].unique():
        sample_data = rarefaction_df[rarefaction_df['sample_id'] == sample]
        sample_data = sample_data.sort_values('depth')
        
        # Estimate asymptote using last 20% of curve
        tail_start = int(len(sample_data) * 0.8)
        asymptote = sample_data.iloc[tail_start:]['richness'].mean()
        
        # Final richness
        final_richness = sample_data.iloc[-1]['richness']
        
        # Plateau ratio
        plateau_ratio = final_richness / (asymptote + 1e-10)
        
        results[sample] = {
            'final_depth': sample_data.iloc[-1]['depth'],
            'final_richness': final_richness,
            'estimated_asymptote': asymptote,
            'plateau_ratio': plateau_ratio,
            'is_adequate': plateau_ratio >= plateau_threshold
        }
    
    results_df = pd.DataFrame(results).T
    
    n_adequate = results_df['is_adequate'].sum()
    n_total = len(results_df)
    
    summary = {
        'n_samples': n_total,
        'n_adequate': int(n_adequate),
        'pct_adequate': float(n_adequate / n_total * 100),
        'mean_plateau_ratio': float(results_df['plateau_ratio'].mean()),
        'median_final_depth': float(results_df['final_depth'].median()),
        'median_final_richness': float(results_df['final_richness'].median()),
    }
    
    logger.info(f"Sequencing adequacy: {summary['pct_adequate']:.1f}% of samples "
               f"reached {plateau_threshold*100}% of asymptote")
    
    return {
        'summary': summary,
        'per_sample': results_df
    }


def suggest_rarefaction_depth(adata: ad.AnnData, 
                              quantile: float = 0.1) -> int:
    """
    Suggest rarefaction depth based on read count distribution.
    
    Common practice: Use 10th percentile of read counts (excludes low-quality samples).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    quantile : float
        Quantile of read counts to use (default 0.1 for 10th percentile)
    
    Returns
    -------
    int
        Suggested rarefaction depth
    """
    logger = get_logger("workflow_16s")
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    read_counts = X.sum(axis=1)
    
    suggested_depth = int(np.quantile(read_counts, quantile))
    
    n_below = np.sum(read_counts < suggested_depth)
    pct_below = n_below / len(read_counts) * 100
    
    logger.info(f"Suggested rarefaction depth: {suggested_depth:,} reads")
    logger.info(f"This would exclude {n_below} samples ({pct_below:.1f}%)")
    logger.info(f"Read count range: {int(read_counts.min()):,} - {int(read_counts.max()):,}")
    
    return suggested_depth
