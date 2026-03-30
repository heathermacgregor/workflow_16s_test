import logging
from pathlib import Path
from typing import Optional, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import anndata as ad
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import scanpy as sc

logger = logging.getLogger("workflow_16s")

def visualize_batch_effects(
    adata: ad.AnnData, 
    batch_key: str = 'batch',
    color_by: Optional[str] = None,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a static diagnostic summary of batch effects (Matplotlib).
    Generates a side-by-side plot: 
    1. PCA colored by Batch
    2. PCA colored by Biology (or Variance Explained if no biology provided)
    
    Best for static reports (PDF/PNG).
    """
    # Compute PCA if not present
    if 'X_pca' not in adata.obsm:
        sc.tl.pca(adata, n_comps=20)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Batch Effect
    sc.pl.pca(
        adata, 
        color=batch_key, 
        ax=axes[0], 
        show=False, 
        title=f'PCA colored by Batch ({batch_key})',
        size=100
    )
    
    # Plot 2: Biology or Variance
    if color_by and color_by in adata.obs.columns:
        sc.pl.pca(
            adata, 
            color=color_by, 
            ax=axes[1], 
            show=False,
            title=f'PCA colored by Biology ({color_by})',
            size=100
        )
    else:
        # Show variance explained
        var_explained = adata.uns.get('pca', {}).get('variance_ratio', [])[:10]
        if len(var_explained) > 0:
            axes[1].bar(range(1, len(var_explained) + 1), var_explained)
            axes[1].set_xlabel('Principal Component')
            axes[1].set_ylabel('Variance Explained')
            axes[1].set_title('PCA Variance Explained')
            axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved static batch visualization to {output_path}")
    
    return fig

def plot_batch_pca_interactive(
    adata: ad.AnnData,
    batch_col: str = 'batch',
    biology_col: Optional[str] = None,
    output_path: Optional[Path] = None,
    title: str = "PCA: Batch Effect Visualization"
):
    """Create interactive PCA plot colored by batch (Plotly)."""
    # Ensure dense array
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    pca = PCA(n_components=3)
    pca_coords = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_
    
    plot_df = pd.DataFrame({
        'PC1': pca_coords[:, 0],
        'PC2': pca_coords[:, 1],
        'Batch': adata.obs[batch_col].astype(str),
        'Sample': adata.obs_names
    })
    
    symbol_col = None
    if biology_col and biology_col in adata.obs.columns:
        plot_df['Biology'] = adata.obs[biology_col].astype(str)
        symbol_col = 'Biology'
        
    fig = px.scatter(
        plot_df, x='PC1', y='PC2', color='Batch', symbol=symbol_col,
        hover_data=['Sample'], title=title,
        labels={'PC1': f'PC1 ({var_exp[0]:.1%})', 'PC2': f'PC2 ({var_exp[1]:.1%})'},
        template="plotly_white"
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    
    if output_path:
        fig.write_html(str(output_path))
        
    return fig

def plot_silhouette_analysis(
    adata: ad.AnnData,
    batch_col: str = 'batch',
    output_path: Optional[Path] = None
):
    """Create silhouette plot showing batch separation quality."""
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    batch_labels = pd.Categorical(adata.obs[batch_col])
    cluster_labels = batch_labels.codes
    
    if len(batch_labels.categories) < 2:
        return None
        
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    y_lower = 10
    
    for i, batch_name in enumerate(batch_labels.categories):
        ith_vals = sample_silhouette_values[cluster_labels == i]
        ith_vals.sort()
        size_cluster_i = ith_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / len(batch_labels.categories))
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals,
                         facecolor=color, edgecolor=color, alpha=0.7, label=batch_name)
        y_lower = y_upper + 10
        
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", label=f'Avg: {silhouette_avg:.2f}')
    ax.set_title(f'Silhouette Plot (Higher = Stronger Batch Effect)')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Batch')
    ax.legend(loc='upper right')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
    return fig

def plot_batch_heatmap(
    adata: ad.AnnData,
    batch_col: str = 'batch',
    biology_col: Optional[str] = None,
    n_features: int = 50,
    output_path: Optional[Path] = None
):
    """Hierarchical clustering heatmap colored by batch."""
    # Preprocessing for visualization only
    adata_sub = adata.copy()
    if adata_sub.n_vars > n_features:
        sc.pp.highly_variable_genes(adata_sub, n_top_genes=n_features)
        adata_sub = adata_sub[:, adata_sub.var.highly_variable]
        
    # Create color map for columns
    colors = pd.DataFrame(index=adata_sub.obs_names)
    
    # Batch colors
    unique_batches = adata_sub.obs[batch_col].unique()
    batch_pal = sns.color_palette("Set2", len(unique_batches))
    colors['Batch'] = adata_sub.obs[batch_col].map(dict(zip(unique_batches, batch_pal)))
    
    # Biology colors
    if biology_col and biology_col in adata_sub.obs.columns:
        unique_bio = adata_sub.obs[biology_col].unique()
        bio_pal = sns.color_palette("Set1", len(unique_bio))
        colors['Biology'] = adata_sub.obs[biology_col].map(dict(zip(unique_bio, bio_pal)))
    
    # Get dense matrix
    X = adata_sub.X.toarray().T if hasattr(adata_sub.X, 'toarray') else adata_sub.X.T
    
    g = sns.clustermap(
        X,
        col_colors=colors, 
        cmap='RdBu_r', 
        center=0, 
        yticklabels=False,
        xticklabels=False,
        figsize=(12, 10),
        cbar_kws={'label': 'Expression'}
    )
    
    g.fig.suptitle(f"Clustering by Top {n_features} Features", y=1.02)
    
    if output_path:
        g.savefig(output_path, dpi=300, bbox_inches='tight')
        
    return g