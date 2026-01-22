"""
Appropriate Batch Correction Methods for Microbiome Data

WARNING: ComBat and limma are NOT appropriate for microbiome data!
They assume continuous, normally-distributed data (gene expression).

Microbiome data is:
- Compositional (relative abundances sum to 1)
- Sparse (many zeros)
- Count-based (not continuous)
- Zero-inflated

Appropriate methods:
1. ConQuR - Batch effect removal for microbiome (R package)
2. Batch as covariate - Include in statistical models
3. Percentile normalization - Non-parametric batch correction
4. MMUPHin - Meta-analysis batch correction
"""

import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats
from typing import Optional, List
import logging

logger = logging.getLogger('workflow_16s')


def detect_batch_effects(adata: ad.AnnData, batch_key: str = 'batch',
                         n_features: int = 100) -> dict:
    """
    Detect presence of batch effects using PCA variance explained.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    batch_key : str
        Column in .obs containing batch information
    n_features : int
        Number of top variable features to use
    
    Returns
    -------
    dict
        Batch effect statistics and recommendation
    """
    if batch_key not in adata.obs.columns:
        return {'error': f'Batch key "{batch_key}" not found in metadata'}
    
    import scanpy as sc
    from sklearn.decomposition import PCA
    
    # Use highly variable features
    if adata.n_vars > n_features:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_features, subset=False)
        X_hvg = adata[:, adata.var['highly_variable']].X
    else:
        X_hvg = adata.X
    
    # Convert to dense if sparse
    if hasattr(X_hvg, 'toarray'):
        X_hvg = X_hvg.toarray()
    
    # PCA
    pca = PCA(n_components=min(10, X_hvg.shape[0] - 1))
    X_pca = pca.fit_transform(X_hvg)
    
    # Test if batch explains PCA variance (ANOVA on PC1)
    batches = adata.obs[batch_key].values
    unique_batches = np.unique(batches)
    
    if len(unique_batches) < 2:
        return {'warning': 'Only one batch detected, no correction needed'}
    
    # ANOVA: Does batch explain PC1 variance?
    groups = [X_pca[batches == b, 0] for b in unique_batches]
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Calculate R² (variance explained by batch)
    ss_total = np.sum((X_pca[:, 0] - np.mean(X_pca[:, 0])) ** 2)
    ss_batch = sum([len(g) * (np.mean(g) - np.mean(X_pca[:, 0])) ** 2 
                    for g in groups])
    r_squared = ss_batch / ss_total
    
    recommendation = 'none'
    if p_value < 0.01 and r_squared > 0.1:
        recommendation = 'correction_needed'
    elif p_value < 0.05:
        recommendation = 'consider_correction'
    
    return {
        'n_batches': len(unique_batches),
        'batch_sizes': {str(b): int(np.sum(batches == b)) for b in unique_batches},
        'pc1_variance_by_batch': float(r_squared),
        'anova_p_value': float(p_value),
        'anova_f_statistic': float(f_stat),
        'recommendation': recommendation
    }


def percentile_normalization(adata: ad.AnnData, batch_key: str = 'batch',
                             inplace: bool = True) -> Optional[ad.AnnData]:
    """
    Percentile normalization for batch correction.
    
    Non-parametric method: For each feature, transforms values to percentiles
    within each batch, then rescales to match global distribution.
    
    Appropriate for sparse, non-normal microbiome data.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    batch_key : str
        Column in .obs containing batch information
    inplace : bool
        Modify in place or return copy
    
    Returns
    -------
    AnnData or None
        Corrected data (if inplace=False)
    """
    if not inplace:
        adata = adata.copy()
    
    if batch_key not in adata.obs.columns:
        logger.error(f'Batch key "{batch_key}" not found')
        return adata if not inplace else None
    
    batches = adata.obs[batch_key].values
    unique_batches = np.unique(batches)
    
    if len(unique_batches) < 2:
        logger.info('Only one batch, skipping correction')
        return adata if not inplace else None
    
    logger.info(f'Applying percentile normalization across {len(unique_batches)} batches')
    
    # Convert to dense for easier manipulation
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X.copy()
    X_corrected = X.copy()
    
    # For each feature
    for j in range(X.shape[1]):
        feature_values = X[:, j]
        
        # Skip if all zeros
        if np.all(feature_values == 0):
            continue
        
        # Global percentiles (target distribution)
        global_percentiles = stats.rankdata(feature_values) / len(feature_values)
        
        # For each batch, map to global distribution
        for batch in unique_batches:
            batch_mask = batches == batch
            batch_values = feature_values[batch_mask]
            
            if len(batch_values) < 2:
                continue
            
            # Percentile ranks within batch
            batch_percentiles = stats.rankdata(batch_values) / len(batch_values)
            
            # Map to global distribution using quantile matching
            global_quantiles = np.quantile(feature_values, batch_percentiles)
            
            X_corrected[batch_mask, j] = global_quantiles
    
    adata.X = X_corrected
    logger.info('Percentile normalization complete')
    
    return None if inplace else adata


def add_batch_as_covariate(adata: ad.AnnData, batch_key: str = 'batch',
                           dummy_coding: bool = True) -> ad.AnnData:
    """
    Prepare batch information as covariates for statistical models.
    
    Most conservative approach: Include batch as predictor in models
    rather than trying to "remove" effects.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    batch_key : str
        Column in .obs containing batch information
    dummy_coding : bool
        Create dummy variables (one-hot encoding)
    
    Returns
    -------
    AnnData
        Data with batch covariates added to .obs
    """
    if batch_key not in adata.obs.columns:
        logger.error(f'Batch key "{batch_key}" not found')
        return adata
    
    batches = adata.obs[batch_key].values
    unique_batches = sorted(np.unique(batches))
    
    if dummy_coding:
        # Create dummy variables (drop first to avoid multicollinearity)
        for i, batch in enumerate(unique_batches[1:], start=1):
            col_name = f'{batch_key}_batch{i}'
            adata.obs[col_name] = (batches == batch).astype(int)
        
        logger.info(f'Added {len(unique_batches)-1} batch dummy variables')
    else:
        # Simple integer encoding
        batch_int = np.zeros(len(batches), dtype=int)
        for i, batch in enumerate(unique_batches):
            batch_int[batches == batch] = i
        
        adata.obs[f'{batch_key}_encoded'] = batch_int
        logger.info(f'Added batch encoding (0-{len(unique_batches)-1})')
    
    return adata


def visualize_batch_effects(adata: ad.AnnData, batch_key: str = 'batch',
                            color_by: Optional[str] = None,
                            save_path: Optional[str] = None):
    """
    Create diagnostic plots for batch effects (PCA, UMAP colored by batch).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    batch_key : str
        Column in .obs containing batch information
    color_by : str, optional
        Additional variable to color by
    save_path : str, optional
        Path to save figure
    """
    import scanpy as sc
    import matplotlib.pyplot as plt
    
    # Compute PCA if not present
    if 'X_pca' not in adata.obsm:
        sc.tl.pca(adata, n_comps=50)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PCA colored by batch
    sc.pl.pca(adata, color=batch_key, ax=axes[0], show=False, 
              title='PCA colored by batch')
    
    # PCA colored by other variable (if provided)
    if color_by and color_by in adata.obs.columns:
        sc.pl.pca(adata, color=color_by, ax=axes[1], show=False,
                  title=f'PCA colored by {color_by}')
    else:
        # Show variance explained
        var_explained = adata.uns['pca']['variance_ratio'][:10]
        axes[1].bar(range(1, 11), var_explained)
        axes[1].set_xlabel('Principal Component')
        axes[1].set_ylabel('Variance Explained')
        axes[1].set_title('PCA Variance Explained')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f'Batch effect diagnostic plot saved to {save_path}')
    
    return fig


# Note: ConQuR integration would require rpy2 and R package installation
def conqur_batch_correction(adata: ad.AnnData, batch_key: str = 'batch',
                            covariate_keys: Optional[List[str]] = None) -> ad.AnnData:
    """
    Apply ConQuR batch correction (requires R and ConQuR package).
    
    ConQuR is specifically designed for microbiome data batch correction.
    
    WARNING: Requires R installation with ConQuR package:
    > install.packages("doParallel")
    > devtools::install_github("wdl2459/ConQuR")
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    batch_key : str
        Column in .obs containing batch information
    covariate_keys : list of str, optional
        Biological covariates to preserve
    
    Returns
    -------
    AnnData
        Batch-corrected data
    """
    try:
        from rpy2.robjects import r, pandas2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
        
        # Load ConQuR
        conqur = importr('ConQuR')
        
        logger.info('Running ConQuR batch correction (may take several minutes)...')
        
        # Prepare data for R
        counts = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        batch = adata.obs[batch_key].values
        
        # Covariates
        if covariate_keys:
            covariates = adata.obs[covariate_keys].values
        else:
            covariates = None
        
        # Run ConQuR
        corrected = conqur.ConQuR(
            tax_tab=counts.T,  # ConQuR expects features × samples
            batchid=batch,
            covariates=covariates,
            batch_ref=batch[0]  # Use first batch as reference
        )
        
        # Update adata
        adata_corrected = adata.copy()
        adata_corrected.X = corrected.T
        
        logger.info('ConQuR batch correction complete')
        return adata_corrected
        
    except ImportError:
        logger.error('ConQuR requires R and rpy2. Install with: pip install rpy2')
        logger.error('Then install ConQuR in R: devtools::install_github("wdl2459/ConQuR")')
        return adata
    except Exception as e:
        logger.error(f'ConQuR failed: {e}')
        return adata
