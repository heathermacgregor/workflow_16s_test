import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats

from workflow_16s.downstream.utils import safe_write_h5ad
from workflow_16s.utils.logger import get_logger


def apply_conqur_correction(
    adata: ad.AnnData,
    batch_col: str = 'batch',
    covariate_cols: Optional[List[str]] = None,
    output_dir: Optional[Path] = None
) -> ad.AnnData:
    """
    Apply ConQuR batch effect correction (via R).
    Preserves compositional structure. Requires R and 'ConQuR' package.
    """
    logger = get_logger("workflow_16s")
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri, conversion
        from rpy2.robjects.packages import importr
    except ImportError:
        logger.error("ConQuR requires rpy2. Install with: pip install rpy2")
        return adata
    
    logger.info("Applying ConQuR batch correction...")
    
    try:
        # Import R packages
        importr('base')
        conqur = importr('ConQuR')
        
        # Prepare data
        counts = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        feature_table = pd.DataFrame(counts.T, index=adata.var_names, columns=adata.obs_names)
        
        # Prepare metadata
        meta_cols = [batch_col] + (covariate_cols if covariate_cols else [])
        meta = adata.obs[meta_cols].copy()
        
        with conversion.localconverter(ro.default_converter + pandas2ri.converter):
            r_counts = pandas2ri.py2rpy(feature_table)
            r_meta = pandas2ri.py2rpy(meta)
            r_batch = ro.StrVector(meta[batch_col].values.astype(str))
            
            if covariate_cols:
                r_covariates = pandas2ri.py2rpy(meta[covariate_cols])
            else:
                r_covariates = ro.NULL
            
            # Run ConQuR
            corrected = conqur.ConQuR(
                tax_tab=r_counts,
                batchid=r_batch,
                covariates=r_covariates,
                batch_ref=ro.r('NULL')
            )
            corrected_df = pandas2ri.rpy2py(corrected)
        
        # Create new AnnData
        adata_corrected = ad.AnnData(
            X=corrected_df.T.values,
            obs=adata.obs.copy(),
            var=adata.var.copy()
        )
        
        # Metadata updates
        adata_corrected.uns['batch_corrected'] = True
        adata_corrected.uns['batch_correction_method'] = 'ConQuR'
        
        if output_dir:
            output_file = Path(output_dir) / "adata_conqur_corrected.h5ad"
            safe_write_h5ad(adata_corrected, output_file)
            
        return adata_corrected
        
    except Exception as e:
        logger.error(f"ConQuR correction failed: {e}")
        return adata

def apply_combat_correction(
    adata: ad.AnnData,
    batch_col: str = 'batch',
    covariate_cols: Optional[List[str]] = None
) -> ad.AnnData:
    """
    Apply ComBat correction. Requires 'combat' python package.
    Note: Assumes data is already log-transformed or Gaussian-like.
    """
    logger = get_logger("workflow_16s")
    try:
        from combat.pycombat import pycombat
    except ImportError:
        logger.error("ComBat requires pycombat: pip install combat")
        return adata
    logger = get_logger("workflow_16s")
    logger.info("Applying ComBat correction...")
    
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    data_df = pd.DataFrame(X.T, index=adata.var_names, columns=adata.obs_names)
    
    batch = adata.obs[batch_col].values
    mod = adata.obs[covariate_cols] if covariate_cols else None
    
    try:
        corrected_df = pycombat(data_df, batch, mod=mod)
        
        adata_corrected = ad.AnnData(
            X=corrected_df.T.values,
            obs=adata.obs.copy(),
            var=adata.var.copy()
        )
        adata_corrected.uns['batch_corrected'] = True
        adata_corrected.uns['batch_correction_method'] = 'ComBat'
        
        return adata_corrected
    except Exception as e:
        logger.error(f"ComBat failed: {e}")
        return adata

def percentile_normalization(
    adata: ad.AnnData, 
    batch_key: str = 'batch',
    inplace: bool = True
) -> Optional[ad.AnnData]:
    """
    Non-parametric batch correction via percentile normalization.
    Good for sparse microbiome data where assumptions of ComBat fail.
    """
    if not inplace:
        adata = adata.copy()
    
    batches = adata.obs[batch_key].values
    unique_batches = np.unique(batches)
    
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X.copy()
    X_corrected = X.copy()
    
    for j in range(X.shape[1]):
        feature_values = X[:, j]
        if np.all(feature_values == 0): continue
        
        # Calculate global percentiles (target distribution)
        # Using rankdata to get percentile ranks
        
        for batch in unique_batches:
            batch_mask = batches == batch
            batch_values = feature_values[batch_mask]
            if len(batch_values) < 2: continue
            
            # Map batch values to their percentiles within the batch
            batch_percentiles = stats.rankdata(batch_values) / len(batch_values)
            
            # Map those percentiles back to the GLOBAL value distribution
            global_quantiles = np.quantile(feature_values, batch_percentiles)
            X_corrected[batch_mask, j] = global_quantiles
            
    adata.X = X_corrected
    adata.uns['batch_corrected'] = True
    adata.uns['batch_correction_method'] = 'PercentileNorm'
    
    return None if inplace else adata

def add_batch_as_covariate(
    adata: ad.AnnData, 
    batch_key: str = 'batch',
    dummy_coding: bool = True
) -> ad.AnnData:
    """Adds batch information as explicit covariate columns in .obs."""
    batches = adata.obs[batch_key].values
    unique_batches = sorted(np.unique(batches))
    
    if dummy_coding:
        for i, batch in enumerate(unique_batches[1:], start=1):
            col_name = f'{batch_key}_batch{i}'
            adata.obs[col_name] = (batches == batch).astype(int)
    else:
        batch_map = {b: i for i, b in enumerate(unique_batches)}
        adata.obs[f'{batch_key}_encoded'] = [batch_map[b] for b in batches]
        
    return adata