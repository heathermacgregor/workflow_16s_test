"""
Volcano Plots for Differential Abundance Visualization

Shows fold-change vs statistical significance.
Helps identify features that are both statistically significant AND biologically meaningful.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


def create_volcano_plot(results_df: pd.DataFrame,
                       fc_col: str = 'log2_fold_change',
                       pval_col: str = 'p_value',
                       fdr_col: Optional[str] = 'fdr',
                       feature_col: str = 'feature',
                       fc_threshold: float = 1.0,
                       pval_threshold: float = 0.05,
                       fdr_threshold: float = 0.05,
                       use_fdr: bool = True,
                       top_n_labels: int = 10,
                       figsize: Tuple[int, int] = (10, 8),
                       save_path: Optional[str] = None,
                       title: Optional[str] = None) -> plt.Figure:
    """
    Create volcano plot for differential abundance results.
    
    Parameters
    ----------
    results_df : DataFrame
        Results with columns for fold-change and p-values
    fc_col : str
        Column name for log2 fold-change
    pval_col : str
        Column name for p-values
    fdr_col : str, optional
        Column name for FDR-corrected p-values
    feature_col : str
        Column name for feature identifiers
    fc_threshold : float
        Absolute log2FC threshold for significance (default 1.0 = 2-fold)
    pval_threshold : float
        P-value threshold (if not using FDR)
    fdr_threshold : float
        FDR threshold (if using FDR)
    use_fdr : bool
        Whether to use FDR instead of raw p-values
    top_n_labels : int
        Number of top features to label
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    title : str, optional
        Plot title
    
    Returns
    -------
    Figure
        Matplotlib figure object
    """
    df = results_df.copy()
    
    # Determine significance column
    if use_fdr and fdr_col and fdr_col in df.columns:
        sig_col = fdr_col
        sig_threshold = fdr_threshold
        sig_label = 'FDR'
    else:
        sig_col = pval_col
        sig_threshold = pval_threshold
        sig_label = 'p-value'
    
    # Calculate -log10(p-value)
    df['-log10_p'] = -np.log10(df[sig_col] + 1e-300)  # Add small value to avoid log(0)
    
    # Classify features
    df['significance'] = 'Not significant'
    
    # Upregulated and significant
    df.loc[(df[fc_col] > fc_threshold) & (df[sig_col] < sig_threshold), 
           'significance'] = f'Upregulated ({sig_label} < {sig_threshold})'
    
    # Downregulated and significant
    df.loc[(df[fc_col] < -fc_threshold) & (df[sig_col] < sig_threshold), 
           'significance'] = f'Downregulated ({sig_label} < {sig_threshold})'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette
    colors = {
        'Not significant': 'gray',
        f'Upregulated ({sig_label} < {sig_threshold})': '#d62728',  # Red
        f'Downregulated ({sig_label} < {sig_threshold})': '#1f77b4'  # Blue
    }
    
    # Plot points by category
    for category in ['Not significant', 
                    f'Downregulated ({sig_label} < {sig_threshold})',
                    f'Upregulated ({sig_label} < {sig_threshold})']:
        subset = df[df['significance'] == category]
        if len(subset) > 0:
            ax.scatter(subset[fc_col], subset['-log10_p'], 
                      c=colors[category], label=category, 
                      alpha=0.6, s=20, edgecolors='none')
    
    # Add threshold lines
    ax.axhline(-np.log10(sig_threshold), color='black', linestyle='--', 
              linewidth=1, alpha=0.5, label=f'{sig_label} = {sig_threshold}')
    ax.axvline(fc_threshold, color='black', linestyle='--', 
              linewidth=1, alpha=0.5)
    ax.axvline(-fc_threshold, color='black', linestyle='--', 
              linewidth=1, alpha=0.5, label=f'FC = ±{2**fc_threshold:.1f}×')
    
    # Label top features
    if top_n_labels > 0:
        # Select top features by combined score (|FC| * -log10(p))
        df['score'] = abs(df[fc_col]) * df['-log10_p']
        top_features = df.nlargest(top_n_labels, 'score')
        
        for _, row in top_features.iterrows():
            ax.annotate(row[feature_col], 
                       xy=(row[fc_col], row['-log10_p']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7,
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='yellow', alpha=0.3))
    
    # Labels and legend
    ax.set_xlabel(f'Log2 Fold Change', fontsize=12)
    ax.set_ylabel(f'-Log10 {sig_label}', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Volcano Plot: Differential Abundance', fontsize=14)
    
    ax.legend(loc='upper left', frameon=True, fontsize=9)
    ax.grid(alpha=0.3, linestyle=':')
    
    # Add summary statistics
    n_up = np.sum(df['significance'].str.contains('Upregulated'))
    n_down = np.sum(df['significance'].str.contains('Downregulated'))
    n_total = len(df)
    
    summary_text = (f'Total features: {n_total}\n'
                   f'Upregulated: {n_up} ({n_up/n_total*100:.1f}%)\n'
                   f'Downregulated: {n_down} ({n_down/n_total*100:.1f}%)')
    
    ax.text(0.98, 0.02, summary_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f'Volcano plot saved to {save_path}')
    
    return fig


def create_ma_plot(results_df: pd.DataFrame,
                  mean_col: str = 'mean_abundance',
                  fc_col: str = 'log2_fold_change',
                  fdr_col: str = 'fdr',
                  feature_col: str = 'feature',
                  fdr_threshold: float = 0.05,
                  fc_threshold: float = 1.0,
                  figsize: Tuple[int, int] = (10, 8),
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Create MA plot (Mean vs Amplitude) - alternative to volcano plot.
    
    Shows log2FC vs mean abundance. Helps identify if differential abundance
    is driven by low-abundance features (often less reliable).
    
    Parameters
    ----------
    results_df : DataFrame
        Results with mean abundance and fold-change
    mean_col : str
        Column for mean abundance
    fc_col : str
        Column for log2 fold-change
    fdr_col : str
        Column for FDR values
    feature_col : str
        Column for feature names
    fdr_threshold : float
        FDR cutoff for significance
    fc_threshold : float
        Log2FC cutoff
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    Figure
        Matplotlib figure object
    """
    df = results_df.copy()
    
    # Log10 transform mean abundance
    df['log10_mean'] = np.log10(df[mean_col] + 1)
    
    # Classify features
    df['significance'] = 'Not significant'
    df.loc[(abs(df[fc_col]) > fc_threshold) & (df[fdr_col] < fdr_threshold), 
           'significance'] = f'Significant (FDR < {fdr_threshold})'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot non-significant
    nonsig = df[df['significance'] == 'Not significant']
    ax.scatter(nonsig['log10_mean'], nonsig[fc_col], 
              c='gray', alpha=0.3, s=10, label='Not significant')
    
    # Plot significant
    sig = df[df['significance'] != 'Not significant']
    ax.scatter(sig['log10_mean'], sig[fc_col], 
              c='red', alpha=0.6, s=20, label=f'Significant (FDR < {fdr_threshold})')
    
    # Add threshold lines
    ax.axhline(fc_threshold, color='blue', linestyle='--', 
              linewidth=1, alpha=0.5)
    ax.axhline(-fc_threshold, color='blue', linestyle='--', 
              linewidth=1, alpha=0.5, label=f'FC = ±{2**fc_threshold:.1f}×')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Log10 Mean Abundance', fontsize=12)
    ax.set_ylabel('Log2 Fold Change', fontsize=12)
    ax.set_title('MA Plot: Mean vs Amplitude', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f'MA plot saved to {save_path}')
    
    return fig


def effect_size_volcano(results_df: pd.DataFrame,
                       effect_size_col: str = 'cliffs_delta',
                       pval_col: str = 'p_value',
                       fdr_col: str = 'fdr',
                       feature_col: str = 'feature',
                       effect_threshold: float = 0.33,
                       fdr_threshold: float = 0.05,
                       top_n_labels: int = 10,
                       figsize: Tuple[int, int] = (10, 8),
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Volcano plot using effect size instead of fold-change.
    
    Useful for non-parametric effect sizes like Cliff's delta.
    
    Parameters
    ----------
    results_df : DataFrame
        Results with effect sizes and p-values
    effect_size_col : str
        Column for effect size
    pval_col : str
        Column for p-values
    fdr_col : str
        Column for FDR
    feature_col : str
        Column for feature names
    effect_threshold : float
        Effect size threshold (0.33 for Cliff's delta = medium effect)
    fdr_threshold : float
        FDR threshold
    top_n_labels : int
        Number of features to label
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    Figure
        Matplotlib figure object
    """
    df = results_df.copy()
    df['-log10_fdr'] = -np.log10(df[fdr_col] + 1e-300)
    
    # Classify
    df['category'] = 'Not significant'
    df.loc[(df[effect_size_col] > effect_threshold) & (df[fdr_col] < fdr_threshold),
           'category'] = 'Large effect (positive)'
    df.loc[(df[effect_size_col] < -effect_threshold) & (df[fdr_col] < fdr_threshold),
           'category'] = 'Large effect (negative)'
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {
        'Not significant': 'gray',
        'Large effect (positive)': '#d62728',
        'Large effect (negative)': '#1f77b4'
    }
    
    for cat in colors:
        subset = df[df['category'] == cat]
        if len(subset) > 0:
            ax.scatter(subset[effect_size_col], subset['-log10_fdr'],
                      c=colors[cat], label=cat, alpha=0.6, s=20)
    
    # Thresholds
    ax.axhline(-np.log10(fdr_threshold), color='black', 
              linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(effect_threshold, color='black', 
              linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(-effect_threshold, color='black', 
              linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel(f'{effect_size_col.replace("_", " ").title()}', fontsize=12)
    ax.set_ylabel('-Log10 FDR', fontsize=12)
    ax.set_title('Effect Size Volcano Plot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
