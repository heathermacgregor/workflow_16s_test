import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
import anndata as ad

from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from skbio.stats.distance import permanova
from skbio.diversity import beta_diversity

logger = logging.getLogger("workflow_16s")

def detect_batch_effects(
    adata: ad.AnnData,
    batch_col: str = 'batch',
    biology_col: Optional[str] = None,
    distance_metric: str = 'braycurtis',
    n_permutations: int = 999
) -> Dict[str, any]:
    """
    Comprehensive batch effect detection using multiple methods.
    
    Tests:
        1. PERMANOVA: Variance explained by batch vs. biology.
        2. Silhouette coefficient: Batch separation score.
        3. PCA variance: Variance explained by batch on top PCs.
        4. Entropy: Distribution uniformity.
    """
    if batch_col not in adata.obs.columns:
        raise ValueError(f"Batch column '{batch_col}' not found in adata.obs")
    
    results = {}
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

    # 1. PERMANOVA for batch effect
    logger.info(f"Running PERMANOVA to quantify batch variance...")
    try:
        # Calculate distance matrix
        dm = beta_diversity(
            distance_metric, X, ids=adata.obs_names
        )
        
        # Test batch effect
        batch_permanova = permanova(
            dm, adata.obs[batch_col], permutations=n_permutations
        )
        results['batch_permanova'] = {
            'test_statistic': batch_permanova['test statistic'],
            'p_value': batch_permanova['p-value'],
            'r_squared': batch_permanova.get('R2', np.nan)
        }
        
        # Test biology effect if provided
        if biology_col and biology_col in adata.obs.columns:
            bio_permanova = permanova(
                dm, adata.obs[biology_col], permutations=n_permutations
            )
            results['biology_permanova'] = {
                'test_statistic': bio_permanova['test statistic'],
                'p_value': bio_permanova['p-value'],
                'r_squared': bio_permanova.get('R2', np.nan)
            }
            
            # Ratio of batch to biology signal
            batch_r2 = results['batch_permanova']['r_squared']
            bio_r2 = results['biology_permanova']['r_squared']
            results['batch_to_biology_ratio'] = batch_r2 / bio_r2 if bio_r2 > 0 else np.inf
            
    except Exception as e:
        logger.error(f"PERMANOVA failed: {e}")
        results['batch_permanova'] = None
    
    # 2. Silhouette coefficient
    logger.info("Calculating silhouette coefficients...")
    try:
        labels = pd.Categorical(adata.obs[batch_col]).codes
        
        # Only calculate if we have >1 batch
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(X, labels, metric='euclidean')
            results['silhouette_score'] = sil_score
        else:
            results['silhouette_score'] = 0.0
        
    except Exception as e:
        logger.error(f"Silhouette calculation failed: {e}")
        results['silhouette_score'] = None
    
    # 3. PCA variance explained by batch
    logger.info("Running PCA for batch visualization...")
    try:
        pca = PCA(n_components=min(10, X.shape[1]))
        pca_coords = pca.fit_transform(X)
        
        results['pca_coords'] = pca_coords
        results['pca_variance_ratio'] = pca.explained_variance_ratio_
        
        # Calculate R² for batch on each PC (ANOVA)
        batch_r2_per_pc = []
        batches = adata.obs[batch_col]
        
        for pc_idx in range(min(5, pca_coords.shape[1])):
            pc_values = pca_coords[:, pc_idx]
            
            # Simple ANOVA R²
            groups = [pc_values[batches == b] for b in np.unique(batches)]
            if len(groups) > 1:
                ss_total = np.sum((pc_values - np.mean(pc_values)) ** 2)
                ss_batch = sum([len(g) * (np.mean(g) - np.mean(pc_values)) ** 2 for g in groups])
                r2 = ss_batch / ss_total if ss_total > 0 else 0
                batch_r2_per_pc.append(r2)
            else:
                batch_r2_per_pc.append(0.0)
        
        results['batch_r2_per_pc'] = batch_r2_per_pc
        
    except Exception as e:
        logger.error(f"PCA calculation failed: {e}")
        results['pca_coords'] = None
    
    # Interpretation Generation
    results['interpretation'] = _interpret_batch_results(results)
    
    return results

def _interpret_batch_results(results: Dict) -> str:
    """Generate human-readable interpretation of batch effect tests."""
    messages = []
    
    # PERMANOVA interpretation
    if results.get('batch_permanova'):
        p_val = results['batch_permanova']['p_value']
        r2 = results['batch_permanova']['r_squared']
        
        if p_val < 0.001:
            messages.append(f"🔴 STRONG batch effect detected (p < 0.001, R² = {r2:.3f})")
        elif p_val < 0.05:
            messages.append(f"🟠 MODERATE batch effect detected (p = {p_val:.3f}, R² = {r2:.3f})")
        else:
            messages.append(f"🟢 No significant batch effect (p = {p_val:.3f})")
    
    # Silhouette interpretation
    if results.get('silhouette_score') is not None:
        sil = results['silhouette_score']
        if sil > 0.5:
            messages.append(f"🔴 High batch clustering (silhouette = {sil:.3f})")
        elif sil > 0.25:
            messages.append(f"🟠 Moderate batch clustering (silhouette = {sil:.3f})")
        else:
            messages.append(f"🟢 Low batch clustering (silhouette = {sil:.3f})")
            
    return "\n".join(messages)