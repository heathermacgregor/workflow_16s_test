from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from workflow_16s.utils.logger import get_logger

def filter_samples_and_features(
    adata: ad.AnnData, 
    min_counts_per_sample: Any = 100, 
    min_counts_per_feature: int = 2,
    min_cells_per_feature: int = 1,
    *args, **kwargs
) -> ad.AnnData:
    """Filters low-quality samples and rare features (ASVs)."""
    logger = get_logger("workflow_16s")
    if adata is None:
        return None

    target_min_sample = 100 
    
    if isinstance(min_counts_per_sample, (int, float)):
        target_min_sample = int(min_counts_per_sample)
    elif hasattr(min_counts_per_sample, 'preprocessing'): 
        try:
            target_min_sample = int(min_counts_per_sample.preprocessing.filter.min_sequencing_depth)
            min_counts_per_feature = int(min_counts_per_sample.preprocessing.filter.min_counts_feature)
            logger.info(f"Extracted min_sequencing_depth={target_min_sample} from passed Config object.")
        except Exception as e:
            logger.warning(f"Could not extract filtering param from config: {e}. Using default {target_min_sample}.")
    else:
        logger.warning(f"Received invalid type {type(min_counts_per_sample)} for min_counts. Using default {target_min_sample}.")

    logger.info(f"Filtering: min_counts_sample={target_min_sample}, min_counts_feature={min_counts_per_feature}, min_cells_feature={min_cells_per_feature}")
    
    try:
        sc.pp.filter_cells(adata, min_counts=target_min_sample)
        sc.pp.filter_genes(adata, min_counts=min_counts_per_feature)
        sc.pp.filter_genes(adata, min_cells=min_cells_per_feature)
        
        logger.info(f"Filtered data shape: {adata.shape}")
    except Exception as e:
        logger.error(f"Filtering failed: {e}")
        
    return adata

def clean_metadata(adata: ad.AnnData, *args, **kwargs) -> ad.AnnData:
    """Standardizes metadata in .obs (bytes->str, strip whitespace, unify NaNs)."""
    logger = get_logger("workflow_16s")
    logger.info("Cleaning metadata...")
    adata.obs.index = adata.obs.index.astype(str)
    for col in adata.obs.columns:
        if adata.obs[col].dtype == 'object' or isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
            if len(adata.obs) > 0 and isinstance(adata.obs[col].iloc[0], bytes):
                adata.obs[col] = adata.obs[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x))
            series = adata.obs[col].astype(str).str.strip()
            adata.obs[col] = series.replace(['nan', 'NaN', 'None', '<NA>', '', 'NoneType'], np.nan)
    return adata

def parse_taxonomy(adata: ad.AnnData, taxonomy_col: str = 'Taxon') -> ad.AnnData:
    """Parses a taxonomy string column into separate rank columns."""
    logger = get_logger("workflow_16s")
    if taxonomy_col not in adata.var.columns:
        for c in adata.var.columns:
            if c.lower() == taxonomy_col.lower():
                taxonomy_col = c
                break
        else:
            logger.warning(f"Taxonomy column '{taxonomy_col}' not found. Skipping parse.")
            return adata

    logger.info(f"Parsing taxonomy from column '{taxonomy_col}'...")
    ranks = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    try:
        tax_series = adata.var[taxonomy_col].astype(str)
        tax_df = tax_series.str.split(';', expand=True)
        tax_df = tax_df.apply(lambda x: x.str.strip())
        for i, rank in enumerate(ranks):
            if i < tax_df.shape[1]:
                adata.var[rank] = tax_df[i].replace(['', 'None', 'nan', '<NA>', 'NoneType'], 'Unassigned')
    except Exception as e:
        logger.error(f"Failed to parse taxonomy: {e}")
    return adata

def validate_metadata(adata: ad.AnnData, config=None) -> ad.AnnData:
    """Wrapper to clean metadata and ensure priority columns exist."""
    adata = clean_metadata(adata)
    required = ['latitude', 'longitude', 'facility_match']
    for col in required:
        if col not in adata.obs.columns:
            adata.obs[col] = False if col == 'facility_match' else np.nan
    return adata