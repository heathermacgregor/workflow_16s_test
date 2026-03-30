# workflow_16s/modules/statistics/batch_correction/conqur.py

from typing import List, Optional
import anndata as ad

from workflow_16s.utils.logger import with_logger


@with_logger
def conqur_batch_correction(
    adata: ad.AnnData, batch_key: str = 'batch', covariate_keys: Optional[List[str]] = None
) -> ad.AnnData:
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