# ==================================================================================== #
#                         downstream/utils/adata_utils.py
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
        if pd.api.types.is_numeric_dtype(series): continue
        if pd.api.types.is_bool_dtype(series): continue
        if isinstance(series.dtype, pd.CategoricalDtype): continue
        if pd.api.types.is_datetime64_any_dtype(series): continue

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
                except Exception: pass
            
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
        if isinstance(series.dtype, pd.CategoricalDtype): continue
        
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
        valid_compressions = ('gzip', 'lzf', None)
        compression_arg = compression if compression in valid_compressions else None
        adata_copy.write_h5ad(filename, compression=compression_arg)
        logger.info(f"Saved AnnData to {filename} (with dtype fixes)")
    else:
        valid_compressions = ('gzip', 'lzf', None)
        compression_arg = compression if compression in valid_compressions else None
        adata.write_h5ad(filename, compression=compression_arg)
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


import math
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from multiprocessing import cpu_count

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import scipy.sparse
from scipy.sparse import csc_matrix, csr_matrix, issparse
import joblib
from joblib import Parallel, delayed

from workflow_16s.config_schema import AppConfig
from workflow_16s.utils.logger import get_logger
from workflow_16s.utils.progress import get_progress_bar

logger = get_logger("workflow_16s")

# --- Constants ---
TAX_LEVELS = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
EXPECTED_VAR_DTYPES = {
    'Taxon': 'string', 
    'Confidence': 'Float64', 
    'sequence': 'string',
    **{level: 'string' for level in TAX_LEVELS}
}
TARGET_GENE_NORMALIZATION = {
    '16S': ['16S', '16S rRNA', '16S rRNA gene', '16s', '16s rrna'],
    '18S': ['18S', '18S rRNA', '18s'],
    'ITS': ['ITS', 'ITS1', 'ITS2', 'its'],
}

    

def _validate_one_file(f: Path) -> Tuple[str, Union[Path, str]]:
    try:
        adata_individual = sc.read_h5ad(f, backed='r')
        if adata_individual.n_vars == 0: return (f.name, "Zero features.")
        return (f.stem, f)
    except Exception as e: 
        return (f.name, f"Failed read. Error: {e}")

# --- Core Processing Steps ---
def _clean_numeric_series(series: pd.Series, col_name: str) -> pd.Series:
    """
    Safely attempts to convert a column to numeric.
    PROTECTIONS:
    - Skips columns that look like IDs (e.g. contain 'accession', 'id', 'alias').
    - Skips date columns (handled separately).
    - Requires >80% valid conversion rate to accept changes.
    """
    col_lower = col_name.lower()
    
    # 1. SKIP IDENTIFIERS & DATES explicitly
    # These often contain numbers but should REMAIN strings/objects
    protected_terms = [
        'accession', 'alias', 'id', 'name', 'sra', 'project', 'study', 'experiment', 'run', 
        'sample', 'submission', 'ftp', 'url', 'link', 'md5', 'date', 'created', 'updated', 
        'time', 'tax_lineage', 'refs', 'publication', 'citation', 'description'
    ]
    if any(term in col_lower for term in protected_terms):
        return series

    # 2. Standardize Missing Values
    missing_indicators = ["nan", "NAN", "NaN", "Null", "null", "None", "none", "", " ", "Missing", "missing", "na", "NA", "unknown"]
    clean = series.copy().astype(str).str.strip()
    is_missing = clean.isin(missing_indicators) | clean.isna() | (clean.str.lower() == 'nan')
    
    # 3. Try Simple Coercion (e.g., "10.5", "-5")
    numeric_simple = pd.to_numeric(clean, errors='coerce')
    
    non_missing_count = (~is_missing).sum()
    if non_missing_count == 0:
        return series # Return original if empty

    valid_simple = (~numeric_simple.isna()).sum()
    ratio_simple = valid_simple / non_missing_count

    if ratio_simple > 0.90:
        return numeric_simple

    # 4. Aggressive Cleaning (Units)
    # Only try this if it's NOT a protected ID column
    # Regex: Extract first float/int (e.g., "10.5 cm" -> 10.5)
    numeric_extracted = clean.str.extract(r'^(-?\d+\.?\d*)')[0]
    numeric_aggressive = pd.to_numeric(numeric_extracted, errors='coerce')
    
    valid_aggressive = (~numeric_aggressive.isna()).sum()
    ratio_aggressive = valid_aggressive / non_missing_count

    # Higher threshold for aggressive cleaning to avoid accidents
    if ratio_aggressive > 0.85:
        # LOGGING: Only log if we actually changed non-numeric text to numbers
        salvaged_mask = numeric_simple.isna() & ~numeric_aggressive.isna() & ~is_missing
        if salvaged_mask.sum() > 0:
            examples = series[salvaged_mask].head(3).to_dict()
            logger.info(f"    🔧 Column '{col_name}': detected units/text mixed with numbers. Converting to numeric.")
            logger.info(f"       Salvaged {salvaged_mask.sum()} values. Examples: {examples} -> {[numeric_aggressive[i] for i in examples]}")
            
        return numeric_aggressive

    return series

def clean_metadata(adata, config=None):
    """
    Standardizes metadata: handles missing values, unifies date formats, 
    and enforces numeric types ONLY for measurement columns.
    """
    # 1. Standardize Missing Values (Global)
    missing_indicators = ["nan", "Null", "null", "None", "none", "", " ", "Unknown", "unknown", "Missing"]
    adata.obs = adata.obs.replace(missing_indicators, np.nan)
    
    # 2. Iterate Columns for Type Inference
    for col in adata.obs.columns:
        # Only process object/categorical columns that are NOT already numeric
        if not pd.api.types.is_numeric_dtype(adata.obs[col]):
            cleaned_series = _clean_numeric_series(adata.obs[col], col)
            
            # Update only if conversion happened
            if pd.api.types.is_numeric_dtype(cleaned_series):
                adata.obs[col] = cleaned_series

    # 3. Standardize Dates (Vectorized)
    # Look for 'date' or 'time' in name, but ignore numeric years (e.g. 2020) if possible
    date_cols = [c for c in adata.obs.columns if any(x in c.lower() for x in ['date', 'time', 'created', 'updated'])]
    
    for col in date_cols:
        # Skip if already numeric (like 'year' = 2020) unless it's a full timestamp
        if pd.api.types.is_numeric_dtype(adata.obs[col]):
             continue
             
        try:
            # Force to datetime -> ISO format
            adata.obs[col] = pd.to_datetime(adata.obs[col], errors='coerce').dt.strftime('%Y-%m-%d')
        except Exception:
            continue

    return adata

def parse_taxonomy(adata):
    """
    Parses taxonomy strings into ranks (Kingdom..Species).
    Handles whitespace, prefixes, and 'unclassified' inheritance logic.
    """
    ranks = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    
    # Identify taxonomy column
    tax_col = next((c for c in adata.var.columns if c.lower() in ['taxon', 'taxonomy', 'lineage']), None)
    if not tax_col:
        for rank in ranks: adata.var[rank] = np.nan
        return adata

    try:
        # 1. Split Taxonomy String
        tax_df = adata.var[tax_col].astype(str).str.split(';', expand=True)
        
        if tax_df.shape[1] < len(ranks):
            for i in range(tax_df.shape[1], len(ranks)):
                tax_df[i] = np.nan
        
        tax_df = tax_df.iloc[:, :len(ranks)]
        tax_df.columns = ranks

        # 2. Vectorized Cleaning
        for rank in ranks:
            # Remove prefixes (d__, p__) and strip whitespace
            tax_df[rank] = tax_df[rank].str.replace(r'^[kpcofgsd]__', '', regex=True).str.strip()

        # 3. Handle 'Unclassified' / Missing Logic
        bad_values = ['unclassified', 'uncultured', 'ambiguous_taxa', '', 'nan', 'None']
        mask = tax_df.isin(bad_values) | tax_df.isna()
        clean_df = tax_df.where(~mask, np.nan)
        
        # Forward fill last valid rank
        filled_df = clean_df.ffill(axis=1)
        filled_df = filled_df.fillna("Unclassified")
        
        # Construct final "Unclassified Rank" strings
        final_df = clean_df.copy()
        for col in ranks:
            fallback = "Unclassified " + filled_df[col]
            final_df[col] = clean_df[col].combine_first(fallback)

        adata.var[ranks] = final_df[ranks]

    except Exception as e:
        logger.debug(f"Taxonomy parsing warning: {e}")
        for rank in ranks:
            if rank not in adata.var.columns: adata.var[rank] = np.nan

    return adata

def filter_samples_and_features(adata, config=None):
    """
    Removes Eukaryota, Mitochondria, Chloroplasts, and empty samples.
    """
    if adata.n_obs == 0: return adata
    
    to_drop = np.zeros(adata.n_vars, dtype=bool)

    # 1. Check for Contaminants
    if 'Kingdom' in adata.var.columns:
        is_euk = adata.var['Kingdom'].astype(str).str.contains('Eukaryota|Eukarya', case=False, na=False)
        to_drop = to_drop | is_euk

    if 'Family' in adata.var.columns:
        is_mito = adata.var['Family'].astype(str).str.contains('mitochondria', case=False, na=False)
        to_drop = to_drop | is_mito
        
    if 'Order' in adata.var.columns:
        is_chloro = adata.var['Order'].astype(str).str.contains('chloroplast', case=False, na=False)
        to_drop = to_drop | is_chloro

    if to_drop.sum() > 0:
        logger.debug(f"Dropping {to_drop.sum()} features (Eukaryota/Mito/Chloro).")
        adata = adata[:, ~to_drop].copy()

    # 2. Filter Empty Samples
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None, log1p=False)
    n_pre_samples = adata.n_obs
    sc.pp.filter_cells(adata, min_counts=1)
    
    if n_pre_samples - adata.n_obs > 0:
        logger.debug(f"Dropped {n_pre_samples - adata.n_obs} empty samples.")

    return adata

def get_cfg_value(cfg_obj, key, default=None):
    """Helper to safely get config values from dict or object."""
    if isinstance(cfg_obj, dict): return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)

def filter_low_depth_and_prevalence(adata: ad.AnnData, config: Union[AppConfig, dict]) -> Union[ad.AnnData, None]:
    """Filters by depth/prevalence."""
    if isinstance(config, dict):
        filter_config = config.get('preprocessing', {}).get('filter', {})
    else:
        filter_config = getattr(config.preprocessing, 'filter', None)

    if not get_cfg_value(filter_config, 'enabled', False): return adata
    
    min_depth = get_cfg_value(filter_config, 'min_sequencing_depth', 5000)
    min_prev = get_cfg_value(filter_config, 'min_sample_prevalence', 2)
    
    sc.settings.verbosity = 0
    sc.pp.filter_cells(adata, min_counts=min_depth)
    if adata.n_obs > 0:
        actual_min = min(min_prev, adata.n_obs)
        if actual_min > 1:
            if issparse(adata.X): adata.X = csc_matrix(adata.X)
            sc.pp.filter_genes(adata, min_cells=actual_min)
            if issparse(adata.X): adata.X = csr_matrix(adata.X)
    sc.pp.filter_cells(adata, min_counts=1)
    
    return adata