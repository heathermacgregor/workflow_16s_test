# ==================================================================================== #
#                       downstream/diversity/clustering.py
# ==================================================================================== #

import logging
import anndata as ad
import numpy as np
import pandas as pd
import plotly.express as px
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
from sklearn_extra.cluster import KMedoids
from scipy.sparse import issparse

# Import from the correct steps location
from workflow_16s.downstream.steps.preprocessing import AnalysisUtils
from workflow_16s.downstream.plotting import PlottingUtils, DEFAULT_HEIGHT
from workflow_16s.utils.logger import get_logger
from workflow_16s.utils.progress import get_progress_bar

logger = get_logger("workflow_16s")
plot_utils = PlottingUtils(logger)

def _local_clr_transform(adata: ad.AnnData, pseudocount: float = 1.0) -> np.ndarray:
    """Helper to perform CLR transform locally without relying on external utils."""
    try:
        # Convert sparse to dense if needed
        X = adata.X.toarray() if issparse(adata.X) else adata.X.copy()
        
        # Ensure float type to handle NaNs/Infs
        X = X.astype(float)
        
        # Add pseudocount
        X = X + pseudocount
        
        # Log transform
        X_log = np.log(X)
        
        # Geometric mean subtraction (CLR)
        # axis=1 is per-sample
        gm = np.mean(X_log, axis=1, keepdims=True)
        X_clr = X_log - gm
        
        # --- ROBUSTNESS FIX ---
        # Replace NaNs and Infs with 0 to prevent clustering crashes
        X_clr = np.nan_to_num(X_clr, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X_clr
        
    except Exception as e:
        logger.error(f"Local CLR transformation failed: {e}")
        # Fallback: return raw data (not ideal but prevents crash)
        return adata.X.toarray() if issparse(adata.X) else adata.X

def run_community_state_typing(adata: ad.AnnData, plot_dir_beta: Path, level: str = 'Genus', max_k: int = 10, 
                               max_samples_for_silhouette: int = 10000, subsample_fraction: float = 0.5) -> Optional[str]:
    """
    Performs CST using K-Medoids clustering.
    ** MODIFIED: HARD SKIP ENABLED **
    """
    # --- KILL SWITCH: FORCE SKIP ---
    logger.info(f"--- CST DISABLED (Skipping {level} Clustering) ---")
    return None
    # -------------------------------

    logger.info(f"--- Starting CST (Level: {level}) ---")
    cst_dir = plot_dir_beta / "CST"; cst_dir.mkdir(exist_ok=True, parents=True)
    
    # Aggregate data
    adata_agg = AnalysisUtils.get_analysis_adata(adata, level=level)
    if adata_agg is None or adata_agg.n_obs < max_k or adata_agg.n_vars < 2: return None
    
    # Perform CLR transform locally
    clr_data = _local_clr_transform(adata_agg, pseudocount=1)
    
    # Validate data before clustering
    if np.isnan(clr_data).any() or np.isinf(clr_data).any():
        logger.warning("NaNs/Infs detected after CLR. Forcing cleanup.")
        clr_data = np.nan_to_num(clr_data, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples = adata_agg.n_obs
    use_subsampling = n_samples > max_samples_for_silhouette
    
    if use_subsampling:
        subsample_size = int(n_samples * subsample_fraction)
        subsample_size = max(subsample_size, max_k * 100)  # Ensure enough samples per cluster
        subsample_size = min(subsample_size, max_samples_for_silhouette)
        logger.info(f"Large dataset ({n_samples:,} samples) - using subsampling strategy:")
        logger.info(f"  • Clustering full dataset with KMedoids")
        logger.info(f"  • Silhouette scores on {subsample_size:,} samples ({subsample_fraction:.0%})")
        np.random.seed(42)
        subsample_idx = np.random.choice(n_samples, subsample_size, replace=False)
        clr_subsample = clr_data[subsample_idx]
    else:
        logger.info(f"Dataset size: {n_samples:,} samples")
        clr_subsample = clr_data
    
    silhouette_scores = {}
    logger.info("Calculating silhouette coefficients...")
    max_possible_k = min(max_k, clr_data.shape[0] - 1)
    
    with get_progress_bar() as progress:
        task = progress.add_task(f"Clustering {level} (k=2 to {max_possible_k})", total=max_possible_k - 1)
        for k in range(2, max_k + 1):
            if k > clr_data.shape[0]: break
            try:
                # Fit on full data
                clusterer = KMedoids(n_clusters=k, metric='euclidean', method='pam', 
                                   init='k-medoids++', max_iter=300, random_state=42)
                labels_full = clusterer.fit_predict(clr_data)
                
                # Calculate silhouette on subsample
                if use_subsampling:
                    labels_subsample = labels_full[subsample_idx]
                    silhouette_scores[k] = silhouette_score(clr_subsample, labels_subsample, metric='euclidean')
                else:
                    silhouette_scores[k] = silhouette_score(clr_data, labels_full, metric='euclidean')
            except Exception as e:
                logger.debug(f"Clustering failed for k={k}: {e}")
            progress.update(task, advance=1)
            
    if not silhouette_scores: 
        logger.warning("No valid silhouette scores computed")
        return None
    
    score_df = pd.DataFrame.from_dict(silhouette_scores, orient='index', columns=['Silhouette Score']).reset_index().rename(columns={'index': 'k'})
    best_k = score_df.loc[score_df['Silhouette Score'].idxmax(), 'k']
    
    plot_title = f'Silhouette Score by k ({level})'
    if use_subsampling:
        plot_title += f' [Subsampled: {subsample_size:,}/{n_samples:,}]'
    
    fig = px.line(score_df, x='k', y='Silhouette Score', title=plot_title, markers=True)
    fig.add_vline(x=best_k, line_dash="dash", line_color="red", annotation_text=f"Best k={best_k}")
    plot_utils.save_plotly_fig(fig, cst_dir / f"cst_silhouette_score_{level}", batch=False)
    
    logger.info(f"Optimal k={best_k} (silhouette={silhouette_scores[best_k]:.3f})")
    final_labels = KMedoids(n_clusters=best_k, metric='euclidean', method='pam', init='k-medoids++', max_iter=300, random_state=42).fit_predict(clr_data)  # type: ignore
    cst_col = f"{level}_CST"; adata.obs[cst_col] = pd.Series(final_labels, index=adata_agg.obs_names).astype(str).astype('category')
    
    # Log cluster sizes
    cluster_sizes = pd.Series(final_labels).value_counts().sort_index()
    logger.info(f"Cluster sizes: {dict(cluster_sizes)}")
    
    pcoa_key = 'X_pcoa_braycurtis'
    if pcoa_key in adata_agg.obsm:
        adata_agg.obs[cst_col] = adata.obs[cst_col]
        try:
            sc.pl.embedding(adata_agg, basis=pcoa_key, color=cst_col, title=f'CST (k={best_k})', show=False)
            plt.savefig(cst_dir / f"cst_pcoa_{level}_k{best_k}.png", dpi=150, bbox_inches='tight'); plt.close()
        except: pass
        
    return cst_col

def audit_batch_bias(adata, cst_col, plot_dir):
    """Checks if CST clusters are biased by batch_original."""
    # 1. Calculate ARI (1.0 = perfect overlap with batch, 0.0 = random)
    if 'batch_original' not in adata.obs.columns:
        return 0.0
        
    ari_score = adjusted_rand_score(adata.obs[cst_col], adata.obs['batch_original'])
    logger.info(f"Batch Bias Audit (ARI): {ari_score:.4f} (Closer to 0 is better)")

    # 2. Per-sample Silhouette Analysis
    # Use local CLR transform instead of missing utility
    clr_data = _local_clr_transform(adata)
    
    try:
        adata.obs['silhouette'] = silhouette_samples(clr_data, adata.obs[cst_col])
        
        # 3. Plotting the Audit
        fig = px.box(adata.obs, x=cst_col, y='silhouette', color='batch_original',
                     title=f"CST Stability vs Batch ID (ARI: {ari_score:.4f})")
        fig.update_layout(showlegend=False) # Hide legend for 300+ batches
        fig.write_html(plot_dir / "cst_batch_bias_audit.html")
    except Exception as e:
        logger.warning(f"Skipping silhouette plot for batch audit: {e}")
        
    return ari_score