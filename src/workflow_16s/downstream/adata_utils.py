# ==================================================================================== #
#                         downstream/adata_utils.py
# ==================================================================================== #

"""
AnnData Utilities for Enhanced Statistics

Helper functions for ensuring AnnData objects are compatible with h5py writing
and other operations.
"""

import os
import logging
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger('workflow_16s')


def get_resident_memory_gb() -> float:
    """
    Returns the resident set size (RES) in GB for the current process using /proc/self/statm.
    Linux only. Returns -1.0 on failure.
    """
    try:
        with open("/proc/self/statm", "r") as f:
            fields = f.readline().split()
            rss_pages = int(fields[1])
            # getconf PAGESIZE usually returns 4096, but we check sysconf
            page_size = os.sysconf("SC_PAGE_SIZE")
            rss_bytes = rss_pages * page_size
            return rss_bytes / 1e9  # Convert to GB
    except Exception:
        return -1.0


def fix_adata_dtypes(adata: ad.AnnData, inplace: bool = True) -> Optional[ad.AnnData]:
    """
    Fix dtype issues in AnnData that cause h5py write errors.
    
    Common issues:
    - Numeric columns stored as object dtype
    - Mixed type columns
    - String columns with non-string values
    - Reserved column names (e.g., _index)
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object to fix
    inplace : bool, default=True
        If True, modifies adata in place. If False, returns a copy.
    
    Returns
    -------
    ad.AnnData or None
        Fixed AnnData object if inplace=False, otherwise None
    """
    if not inplace:
        adata = adata.copy()

    # Remove reserved column names before any dtype fixing
    reserved_names = ['_index']
    for reserved in reserved_names:
        if reserved in adata.obs.columns:
            adata.obs = adata.obs.drop(columns=[reserved])
            logger.debug(f"Dropped reserved column '{reserved}' from obs")
        if reserved in adata.var.columns:
            adata.var = adata.var.drop(columns=[reserved])
            logger.debug(f"Dropped reserved column '{reserved}' from var")
    
    # Define patterns for columns that should be numeric
    numeric_patterns = [
        '_km', '_mg_per_', '_ug_per_', '_umol_per_', '_g_per_kg', '_us_per_cm',
        'SoilGrids_', 'Meteostat_', 'OpenMeteo_', 'EnvironmentalHealth_',
        'NOAA_Tides_distance', '_percent', '_mv', 'latitude', 'longitude',
        'elevation', 'depth', 'temperature', 'ph', 'USGS_Earthquake_count',
        'iNaturalist_count', '_prcp', '_pres', '_tavg', '_tmax', '_tmin', '_wspd',
        '_snow', '_tsun', '_wpgt', '_ozone', '_pm10', '_pm2_5', 'precipitation',
        'humidity', 'calcium', 'chlorine', 'carbon', 'iron', 'magnesium',
        'potassium', 'sulfate', 'chloride', 'manganese', 'zinc', 'uranium',
        'aluminium', 'alumium', 'barium', 'cadmium', 'chromium', 'copper',
        'lead', 'molybdenum', 'redox', 'chemical_oxygen_demand'
    ]
    
    # Define patterns for date columns
    date_patterns = ['_date', 'collection_date', '_time', 'temporal_coverage']
    
    # Fix adata.obs dtypes
    for col in adata.obs.columns:
        series = adata.obs[col]
        
        # Always treat 'lat' and 'lon' as float columns (CRITICAL for spatial mapping)
        if col in ('lat', 'lon', 'latitude', 'longitude'):
            try:
                numeric = pd.to_numeric(series, errors='coerce')
                adata.obs[col] = numeric.astype('float64')
                logger.debug(f"Forced '{col}' to float64 (geo coordinate)")
            except Exception as e:
                logger.debug(f"Failed to convert '{col}' to float64: {e}")
            continue

        # Skip if already correct
        if pd.api.types.is_numeric_dtype(series):
            continue
        if pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_categorical_dtype(series):
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        # Try to convert object dtype
        if series.dtype == object:
            # Check if column should be date
            is_date_col = any(pattern in col for pattern in date_patterns)
            if is_date_col:
                try:
                    dt_series = pd.to_datetime(series, errors='coerce')
                    adata.obs[col] = dt_series.dt.strftime('%Y-%m-%d').fillna('').astype(str)
                    logger.debug(f"Converted '{col}' to datetime string (ISO format)")
                    continue
                except Exception:
                    pass
            
            # Check if column should be numeric
            is_numeric_col = any(pattern in col for pattern in numeric_patterns)
            try:
                numeric = pd.to_numeric(series, errors='coerce')
                original_valid = series.notna().sum()
                converted_valid = numeric.notna().sum()
                
                # Heuristic: convert if >90% data is valid numeric, or if it matches a known numeric pattern
                if converted_valid >= original_valid * 0.9 or (is_numeric_col and converted_valid > 0):
                    # Check if it can be an Integer (no decimals)
                    if (numeric.dropna() % 1 == 0).all() and len(numeric.dropna()) > 0:
                        adata.obs[col] = numeric.astype('Int64')
                        logger.debug(f"Converted '{col}' to Int64")
                    else:
                        adata.obs[col] = numeric.astype('float64')
                        logger.debug(f"Converted '{col}' to float64")
                else:
                    # Fallback to string for h5py safety
                    adata.obs[col] = series.astype(str)
                    logger.debug(f"Converted '{col}' to string")
            except Exception as e:
                logger.debug(f"Converted '{col}' to string (conversion failed: {e})")
                adata.obs[col] = series.astype(str)
    
    # Fix adata.var dtypes (taxonomy columns, stats)
    for col in adata.var.columns:
        series = adata.var[col]
        
        # Skip if already numeric or categorical
        if pd.api.types.is_numeric_dtype(series):
            continue
        if pd.api.types.is_categorical_dtype(series):
            continue
        
        if series.dtype == object:
            try:
                numeric = pd.to_numeric(series, errors='coerce')
                original_valid = series.notna().sum()
                converted_valid = numeric.notna().sum()
                
                # Allow up to 10% loss for sparse data
                if converted_valid >= original_valid * 0.9:
                    if (numeric.dropna() % 1 == 0).all() and len(numeric.dropna()) > 0:
                        adata.var[col] = numeric.astype('Int64')
                        logger.debug(f"Converted var '{col}' to Int64")
                    else:
                        adata.var[col] = numeric.astype('float64')
                        logger.debug(f"Converted var '{col}' to float64")
                else:
                    adata.var[col] = series.astype(str)
                    logger.debug(f"Converted var '{col}' to string")
            except Exception:
                adata.var[col] = series.astype(str)
                logger.debug(f"Converted var '{col}' to string (failed conversion)")
    
    logger.info("Fixed AnnData dtypes for h5py compatibility")
    
    if not inplace:
        return adata


def safe_write_h5ad(adata: ad.AnnData, filename: str, fix_dtypes: bool = True, compression: Optional[str] = None):
    """
    Safely write AnnData to h5ad file with automatic dtype fixing.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object to save
    filename : str
        Output filename
    fix_dtypes : bool, default=True
        Whether to fix dtypes before writing
    compression : str, optional
        Compression to use ('gzip', 'lzf', None)
    """
    if fix_dtypes:
        # Make a copy to avoid modifying original if inplace fixing is not desired globally here
        # But fix_adata_dtypes default is inplace=True. 
        # To be safe for a "save" operation, we operate on a copy if we don't want to touch the runtime object.
        adata_copy = adata.copy()
        fix_adata_dtypes(adata_copy, inplace=True)
        adata_copy.write_h5ad(filename, compression=compression)
        logger.info(f"Saved AnnData to {filename} (with dtype fixes)")
    else:
        adata.write_h5ad(filename, compression=compression)
        logger.info(f"Saved AnnData to {filename}")


def inspect_adata_dtypes(adata: ad.AnnData) -> pd.DataFrame:
    """
    Inspect dtypes in AnnData object for debugging.
    
    Returns
    -------
    pd.DataFrame
        Summary of column dtypes in obs and var
    """
    results = []
    
    # Inspect obs
    for col in adata.obs.columns:
        results.append({
            'location': 'obs',
            'column': col,
            'dtype': str(adata.obs[col].dtype),
            'n_unique': adata.obs[col].nunique(),
            'has_na': adata.obs[col].isna().any()
        })
    
    # Inspect var
    for col in adata.var.columns:
        results.append({
            'location': 'var',
            'column': col,
            'dtype': str(adata.var[col].dtype),
            'n_unique': adata.var[col].nunique(),
            'has_na': adata.var[col].isna().any()
        })
    
    return pd.DataFrame(results)