# workflow_16s/utils/andata/misc.py

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from scipy import sparse
from scipy.sparse import csc_matrix, csr_matrix, issparse

from workflow_16s.config import AppConfig
from workflow_16s.utils.logger import get_logger

def get_adata_level(adata_in: ad.AnnData, level: str) -> Union[ad.AnnData, None]:
    """
    Gets an AnnData object for a specific analysis level.
    - If level is ASV/OTU/FEATURE, returns original data (no aggregation).
    - If level is taxonomic, calls aggregate_adata_by_taxonomy.
    - If level is functional (in .obsm), creates a new AnnData object.
    """
    logger = get_logger("workflow_16s")
    
    # ASV/OTU/FEATURE - return original data, no aggregation needed
    if level.upper() in ['ASV', 'FEATURE', 'OTU']:
        logger.info(f"Getting analysis AnnData for native ASV level (no aggregation)")
        adata_asv = adata_in.copy()
        if 'raw_counts' not in adata_asv.layers:
            adata_asv.layers['raw_counts'] = adata_asv.X.copy()
        adata_asv.var.index.name = 'ASV'
        return adata_asv
        
    # First check if level is a known taxonomic rank, then check .obsm for functional data
    from workflow_16s.utils.analysis import AnalysisUtils
    if level in AnalysisUtils.TAX_LEVELS_ALL: 
        logger.info(f"Getting analysis AnnData for taxonomic level: {level}")
        return AnalysisUtils.aggregate_adata_by_taxonomy(adata_in, tax_level=level)
        
    elif level in adata_in.obsm:
        logger.info(f"Getting analysis AnnData for functional level: {level}")
        try:
            func_data = adata_in.obsm[level]
            if func_data is None: 
                logger.error(f"Data for {level} in .obsm is None")
                return None
            data_for_anndata: Union[pd.DataFrame, csc_matrix, csr_matrix, np.ndarray]
                
            # Assume data is an array if not a DataFrame, and try to convert it
            if not isinstance(func_data, pd.DataFrame):
                logger.debug(f"Converting array data from .obsm['{level}']")
                    
                if f"{level}_ids" in adata_in.uns: 
                    feature_names = adata_in.uns[f"{level}_ids"]
                elif hasattr(func_data, 'shape') and func_data.shape is not None and func_data.shape[0] > 0: 
                    # If shape[1] matches n_obs, likely (features x samples)
                    if func_data.shape[1] == adata_in.n_obs:
                        feature_names = [f"{level}_{i}" for i in range(func_data.shape[0])]
                    else:
                        feature_names = [f"{level}_{i}" for i in range(func_data.shape[1])]
                else: 
                    logger.error(f"Cannot determine feature names for {level}")
                    return None
                    
                # Determine orientation and transpose if necessary, ensuring we end up with (samples x features)
                if hasattr(func_data, 'shape') and func_data.shape is not None:
                    if func_data.shape[1] == adata_in.n_obs:
                        # Convert sparse matrix to dense array if needed
                        if issparse(func_data):
                            func_data_dense = func_data.toarray() # type: ignore
                        else:
                            func_data_dense = np.asarray(func_data)
                        func_df = pd.DataFrame(func_data_dense, index=feature_names, columns=adata_in.obs_names)
                        data_for_anndata = func_df.T
                    elif func_data.shape[0] == adata_in.n_obs:
                        # Convert scipy sparse arrays to matrices for compatibility
                        if hasattr(func_data, 'toarray'):
                            data_for_anndata = func_data if issparse(func_data) else np.asarray(func_data) # type: ignore
                        else:
                            data_for_anndata = np.asarray(func_data)
                    else:
                        logger.error(f"Shape mismatch for {level}: {func_data.shape} vs adata.n_obs {adata_in.n_obs}")
                        return None
                else:
                    return None
                
            else:
                logger.debug(f"Using existing DataFrame from .obsm['{level}'].")
                func_df = func_data.copy()
                 
                # Check orientation: if rows don't match n_obs, try transposing
                if func_df.shape[0] != adata_in.n_obs:
                    logger.warning(
                        f"DataFrame in .obsm['{level}'] has {func_df.shape[0]} rows, but adata has {adata_in.n_obs} obs.\n"
                        f"Attempting to transpose, assuming (features x samples)."
                    )
                    if func_df.shape[1] != adata_in.n_obs:
                        logger.error(f"Shape mismatch: DataFrame is {func_df.shape}, adata is {adata_in.n_obs} obs. Cannot orient.")
                        return None
                    data_for_anndata = func_df.T
                else:
                    data_for_anndata = func_df

            # Create new AnnData object 
            adata_func = ad.AnnData(data_for_anndata)
            adata_func.obs = adata_in.obs.loc[adata_func.obs_names].copy()
            adata_func.var.index.name = level
                
            # Create the 'raw_counts' layer for downstream functions
            if isinstance(data_for_anndata, pd.DataFrame):
                counts_values = data_for_anndata.values
            elif issparse(data_for_anndata):
                counts_values = data_for_anndata.toarray() # type: ignore
            else:
                counts_values = np.asarray(data_for_anndata)

            if (counts_values < 0).any():
                logger.warning(f"Negative values found in {level}. Shifting to non-negative for 'raw_counts'.")
                min_val = counts_values.min()
                adata_func.layers['raw_counts'] = counts_values - min_val
            else:
                adata_func.layers['raw_counts'] = counts_values.copy()
                    
            # Set .X to be the same as 'raw_counts' for functions that read .X
            adata_func.X = adata_func.layers['raw_counts'].copy()
             
            logger.info(f"Created functional AnnData: {adata_func.shape}")
            return adata_func
          
        except Exception as e: 
            logger.error(f"Failed to create AnnData from obsm key '{level}': {e}", exc_info=True)
            return None
    else: 
        logger.warning(f"Analysis level '{level}' not found in taxonomy or .obsm. Skipping.")
        return None

def aggregate_adata_by_taxonomy(adata_in: ad.AnnData, tax_level: str = 'Genus') -> Union[ad.AnnData, None]:
    """
    Aggregates an AnnData object using efficient sparse matrix multiplication.
    Includes critical fixes for duplicates, whitespaces, and unassigned taxa.
    """
    logger = get_logger("workflow_16s")
    logger.info(f"--- Aggregating AnnData by {tax_level} ---")
        
    if not adata_in.obs_names.is_unique:
        logger.warning("⚠️ Duplicate sample IDs found in input! Making unique (appending -1, -2)...")
        adata_in.obs_names_make_unique()

    adata_copy = adata_in.copy()
        
    if tax_level not in adata_copy.var.columns: 
        logger.error(f"Tax level '{tax_level}' not in .var.")
        return None
            
    tax_series = adata_copy.var[tax_level].astype(str).str.strip()
        
    # Unify all forms of "empty"
    tax_series = tax_series.replace(
        ['nan', 'NaN', 'None', '', '<NA>', 'NoneType'], 
        'Unassigned'
    )
    tax_series = tax_series.fillna('Unassigned')
        
    # Update the dataframe used for mapping
    adata_copy.var[tax_level] = tax_series

    # Get Counts Matrix
    if 'raw_counts' in adata_copy.layers: 
        counts_mtx = adata_copy.layers['raw_counts']
    else: 
        counts_mtx = adata_copy.X
            
    if sparse.issparse(counts_mtx):
        counts_mtx = counts_mtx.tocsc() # type: ignore
    else:
        # If not sparse, ensure it's a numpy array
        if hasattr(counts_mtx, 'toarray'):
            counts_mtx = counts_mtx.toarray() # type: ignore
        elif not isinstance(counts_mtx, np.ndarray):
            counts_mtx = np.array(counts_mtx)
            
    asv_to_tax_map = adata_copy.var[tax_level]
        
    # Get unique taxa and their indices
    unique_taxa, group_indices = np.unique(asv_to_tax_map, return_inverse=True)

    # Create sparse grouper matrix
    n_features = adata_copy.n_vars
    n_groups = len(unique_taxa)
        
    if counts_mtx is not None and hasattr(counts_mtx, 'dtype'):
        grouper_dtype = np.dtype(counts_mtx.dtype)
    else:
        grouper_dtype = np.float64
            
    M_grouper = csc_matrix(
        (np.ones(n_features, dtype=grouper_dtype), (group_indices, np.arange(n_features))), 
        shape=(n_groups, n_features)
    )

    # Perform the aggregation
    if not isinstance(counts_mtx, csr_matrix):
        if issparse(counts_mtx):
            counts_mtx = counts_mtx.tocsr() # type: ignore
        else:
            counts_mtx = csr_matrix(counts_mtx)
                
    agg_mtx = counts_mtx @ M_grouper.T 

    # Create the new AnnData
    new_var = pd.DataFrame(index=unique_taxa)
      
    if not isinstance(agg_mtx, csr_matrix):
        agg_mtx = csr_matrix(agg_mtx)

    adata_new = ad.AnnData(
        agg_mtx, 
        obs=adata_copy.obs.copy(), 
        var=new_var, 
        dtype=agg_mtx.dtype
    )
        
    # Explicitly set the index to the taxonomy strings
    adata_new.var_names = unique_taxa.astype(str).tolist()
    adata_new.var.index.name = tax_level
        
    # Save the name as a column so downstream tools can find it
    adata_new.var[tax_level] = adata_new.var.index.values
    adata_new.layers['raw_counts'] = csr_matrix(adata_new.X)
        
    # Filter 'Unassigned' 
    if 'Unassigned' in adata_new.var_names:
        if len(adata_new.var_names) > 1:
            logger.info("Filtering 'Unassigned' taxa.")
            adata_new = adata_new[:, adata_new.var_names != 'Unassigned'].copy()
        else:
            logger.warning("⚠️ All features mapped to 'Unassigned'! Keeping it.")
    try:
        from workflow_16s.utils.analysis import AnalysisUtils
        logger.info(f"Applying rCLR transform to {tax_level}-aggregated data...")
        transformed = AnalysisUtils.rclr_transform(adata_new)

        if transformed is not None:
            # rclr_transform returns sparse matrix to avoid memory explosion
            adata_new.X = transformed
            logger.info("✅ rCLR successful.")
        else:
            logger.error("❌ rCLR failed.")

    except Exception as e:
        logger.error(f"❌ Failed to apply rCLR during aggregation: {e}")

    # Ensure obs indices are string
    adata_new.obs_names = adata_new.obs_names.astype(str).tolist()
        
    logger.info(f"Aggregation complete. New shape: {adata_new.shape}")
    return adata_new
