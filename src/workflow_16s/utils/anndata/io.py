# workflow_16s/utils/io/anndata.py

import hashlib
import io
import os
import pickle
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

import anndata as ad
import biom
import numpy as np
import pandas as pd
import skbio
import scanpy as sc
from pandas.api.types import is_extension_array_dtype
from scipy.sparse import issparse, csr_matrix
from skbio import TreeNode

from workflow_16s.config import AppConfig
#from workflow_16s.downstream.utils.adata_biology import (
#    filter_samples_and_features, parse_taxonomy
#)
from workflow_16s.utils.logger import get_logger
# workflow_16s/downstream/utils/adata_biology.py

from typing import Union
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.sparse import csc_matrix, csr_matrix, issparse

from workflow_16s.config import AppConfig
from workflow_16s.utils.logger import get_logger

def get_cfg_value(cfg_obj, key, default=None):
    if isinstance(cfg_obj, dict): return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)

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
        'time', 'tax_lineage', 'refs', 'publication', 'citation', 'description',
        'first_public', 'location', 'target_gene', 'pcr_primer', 'primer',
        'mapping_file', 'lcms_position', 'store_cond',  'earthquake', 'ena'
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
    # Ensure this exists in Step 4 of the function!
    clean_for_regex = clean.str.replace(',', '', regex=False)
    numeric_extracted = clean_for_regex.str.extract(r'^(-?\d+\.?\d*)')[0]
    numeric_aggressive = pd.to_numeric(numeric_extracted, errors='coerce')
    
    valid_aggressive = (~numeric_aggressive.isna()).sum()
    ratio_aggressive = valid_aggressive / non_missing_count

    # Higher threshold for aggressive cleaning to avoid accidents
    if ratio_aggressive > 0.85:
        # LOGGING: Only log if we actually changed non-numeric text to numbers
        salvaged_mask = numeric_simple.isna() & ~numeric_aggressive.isna() & ~is_missing
        if salvaged_mask.sum() > 0:
            examples_series = series[salvaged_mask].head(3)
            example_originals = examples_series.tolist()
            example_conversions = numeric_aggressive[salvaged_mask].head(3).tolist()
            logger = get_logger("workflow_16s")
            logger.info(f"    🔧 Column '{col_name}': detected units/text mixed with numbers. Converting to numeric.\n"
                        f"       Salvaged {salvaged_mask.sum()} values. Examples: {example_originals} -> {example_conversions}")
            
        return numeric_aggressive

    return series

def clean_metadata(adata, config=None):
    """
    Standardizes metadata: handles missing values, unifies date formats, 
    and enforces numeric types ONLY for measurement columns.
    """
    # 1. Standardize Missing Values (Global)
    missing_indicators = ["nan", "Null", "null", "None", "none", "", " ", "Unknown", "unknown", "Missing"]
    # Only apply global replacement to non-categorical columns
    non_cat_cols = adata.obs.select_dtypes(exclude=['category']).columns
    adata.obs[non_cat_cols] = adata.obs[non_cat_cols].replace(missing_indicators, np.nan)
    
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

def parse_taxonomy(adata: ad.AnnData) -> ad.AnnData:
    logger = get_logger("workflow_16s")
    ranks = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    tax_col = next((c for c in adata.var.columns if c.lower() in ['taxon', 'taxonomy', 'lineage']), None)
    
    if not tax_col:
        for rank in ranks: adata.var[rank] = np.nan
        return adata

    try:
        tax_df = adata.var[tax_col].astype(str).str.split(';', expand=True)
        if tax_df.shape[1] < len(ranks):
            for i in range(tax_df.shape[1], len(ranks)): tax_df[i] = np.nan
        
        tax_df = tax_df.iloc[:, :len(ranks)]
        tax_df.columns = ranks

        for rank in ranks:
            tax_df[rank] = tax_df[rank].str.replace(r'^[kpcofgsd]__', '', regex=True).str.strip()

        bad_values = ['unclassified', 'uncultured', 'ambiguous_taxa', '', 'nan', 'None']
        mask = tax_df.isin(bad_values) | tax_df.isna()
        clean_df = tax_df.where(~mask, np.nan)
        
        filled_df = clean_df.ffill(axis=1).fillna("Unclassified")
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

def filter_samples_and_features(adata: ad.AnnData, config=None) -> ad.AnnData:
    logger = get_logger("workflow_16s")
    if adata.n_obs == 0: return adata
    
    to_drop = np.zeros(adata.n_vars, dtype=bool)

    if 'Kingdom' in adata.var.columns:
        to_drop |= adata.var['Kingdom'].astype(str).str.contains('Eukaryota|Eukarya', case=False, na=False)
    if 'Family' in adata.var.columns:
        to_drop |= adata.var['Family'].astype(str).str.contains('mitochondria', case=False, na=False)
    if 'Order' in adata.var.columns:
        to_drop |= adata.var['Order'].astype(str).str.contains('chloroplast', case=False, na=False)

    if to_drop.sum() > 0:
        logger.debug(f"Dropping {to_drop.sum()} features (Eukaryota/Mito/Chloro).")
        adata = adata[:, ~to_drop].copy()

    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None, log1p=False)
    n_pre_samples = adata.n_obs
    sc.pp.filter_cells(adata, min_counts=1)
    
    if n_pre_samples - adata.n_obs > 0:
        logger.debug(f"Dropped {n_pre_samples - adata.n_obs} empty samples.")

    return adata

def filter_low_depth_and_prevalence(adata: ad.AnnData, config: Union[AppConfig, dict]) -> ad.AnnData:
    filter_config = config.get('preprocessing', {}).get('filter', {}) if isinstance(config, dict) else getattr(config.preprocessing, 'filter', None)
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

def _get_file_hash(filepath: Path) -> str:
    stats = filepath.stat()
    return hashlib.md5(f"{stats.st_size}_{stats.st_mtime}".encode()).hexdigest()[:8]

def _validate_cached_adata(adata) -> Tuple[bool, str]:
    """Quick sanity check on cached adata object."""
    if adata is None: return False, "None object"
    if not isinstance(adata, ad.AnnData): return False, "Not AnnData"
    if adata.n_obs == 0: return False, "Empty observations"
    return True, ""

def _sanitize_adata(adata: ad.AnnData) -> ad.AnnData:
    if adata.obs_names.name is None: adata.obs_names.name = 'sample_id'
    if adata.var_names.name is None: adata.var_names.name = 'feature_id'
    if not adata.obs.index.is_unique: adata.obs_names_make_unique()
    
    for idx_name in [adata.obs_names.name, adata.var_names.name]:
        if idx_name in adata.obs.columns:
            try: del adata.obs[idx_name]
            except: pass
            
    for coord in ['lat', 'lon', 'latitude', 'longitude']:
        if coord in adata.obs.columns:
            adata.obs[coord] = pd.to_numeric(adata.obs[coord], errors='coerce')

    problem_cols = ['facility_capacity', 'facility_match', 'study_accession']
    for col in adata.obs.columns:
        if col in problem_cols or adata.obs[col].dtype == 'object':
            adata.obs[col] = adata.obs[col].astype(str).replace(['nan', 'None', 'NaN'], '')
            
    try: adata.var_names = adata.var_names.str.strip().tolist()
    except: pass
    try: adata.obs_names = adata.obs_names.str.strip().tolist()
    except: pass
    
    if not issparse(adata.X): adata.X = csr_matrix(adata.X)
    return adata

def _sanitize_obs(adata_obj):
    for col in adata_obj.obs.columns:
        dtype_name = adata_obj.obs[col].dtype.name
        if dtype_name in ('object', 'category'):
            cleaned = adata_obj.obs[col].astype(str).replace(['nan', 'None', 'NaN', 'NoneType', '<NA>', '<nan>'], '')
            adata_obj.obs[col] = cleaned.astype('category') if dtype_name == 'category' else cleaned
        elif is_extension_array_dtype(adata_obj.obs[col].dtype):
            try: adata_obj.obs[col] = adata_obj.obs[col].to_numpy(dtype=float, na_value=np.nan)
            except Exception: adata_obj.obs[col] = adata_obj.obs[col].astype(str).replace(['nan', 'None', 'NaN', 'NoneType', '<NA>'], '')
    return adata_obj

def fix_adata_dtypes(adata: ad.AnnData, inplace: bool = True) -> Optional[ad.AnnData]:
    """Fix dtype issues in AnnData that cause h5py write errors."""
    logger = get_logger("workflow_16s")
    if not inplace: adata = adata.copy()

    stats = {"dropped": 0, "obs_numeric": 0, "obs_date": 0, "obs_str": 0, "var_numeric": 0, "var_str": 0}

    # 1. Remove reserved column names
    for reserved in ['_index']:
        if reserved in adata.obs.columns:
            adata.obs = adata.obs.drop(columns=[reserved])
            stats["dropped"] += 1
        if reserved in adata.var.columns:
            adata.var = adata.var.drop(columns=[reserved])
            stats["dropped"] += 1
    
    numeric_patterns = [
        '_km', '_mg_per_', '_ug_per_', '_umol_per_', '_g_per_kg', '_us_per_cm',
        'SoilGrids_', 'Meteostat_', 'OpenMeteo_', 'EnvironmentalHealth_',
        'NOAA_Tides_distance', '_percent', '_mv', 'latitude', 'longitude',
        'elevation', 'depth', 'temperature', 'ph', 'uranium'
    ]
    date_patterns = ['_date', 'collection_date', '_time', 'temporal_coverage']
    
    # 2. Fix obs dtypes
    for col in adata.obs.columns:
        series = adata.obs[col]
        if col in ('lat', 'lon', 'latitude', 'longitude'):
            try:
                adata.obs[col] = pd.to_numeric(series, errors='coerce').astype('float64')
                stats["obs_numeric"] += 1
            except Exception: pass
            continue

        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series) or \
            isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_datetime64_any_dtype(series):
            continue

        if series.dtype == object:
            if any(pattern in col for pattern in date_patterns):
                try:
                    adata.obs[col] = pd.to_datetime(series, errors='coerce').dt.strftime('%Y-%m-%d').fillna('').astype(str)
                    stats["obs_date"] += 1
                    continue
                except Exception: pass
            
            if any(pattern in col for pattern in numeric_patterns):
                try:
                    numeric = pd.to_numeric(series, errors='coerce')
                    valid_count, num_valid = series.notna().sum(), numeric.notna().sum()
                    if num_valid >= valid_count * 0.9 or num_valid > 0:
                        adata.obs[col] = numeric.astype('Int64' if (numeric.dropna() % 1 == 0).all() else 'float64')
                        stats["obs_numeric"] += 1
                    else:
                        adata.obs[col] = series.astype(str)
                        stats["obs_str"] += 1
                except Exception:
                    adata.obs[col] = series.astype(str)
                    stats["obs_str"] += 1
            else:
                adata.obs[col] = series.astype(str)
                stats["obs_str"] += 1
    
    # 3. Fix var dtypes
    for col in adata.var.columns:
        series = adata.var[col]
        if pd.api.types.is_numeric_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype): continue
        if series.dtype == object:
            try:
                numeric = pd.to_numeric(series, errors='coerce')
                if numeric.notna().sum() >= series.notna().sum() * 0.9:
                    adata.var[col] = numeric.astype('Int64' if (numeric.dropna() % 1 == 0).all() else 'float64')
                    stats["var_numeric"] += 1
                else:
                    adata.var[col] = series.astype(str)
                    stats["var_str"] += 1
            except Exception:
                adata.var[col] = series.astype(str)
                stats["var_str"] += 1

    logger.info(f"AnnData Dtypes Fixed | Obs: {stats['obs_numeric']} num, {stats['obs_date']} date, {stats['obs_str']} str | Var: {stats['var_numeric']} num, {stats['var_str']} str | Dropped: {stats['dropped']} reserved cols")
    if not inplace: return adata
    
def fix_adata_dtypes(adata: ad.AnnData, inplace: bool = True) -> Optional[ad.AnnData]:
    """
    Fix dtype issues in AnnData that cause h5py write errors and summarize changes.
    """
    logger = get_logger("workflow_16s")
    if not inplace:
        adata = adata.copy()

    # Summary counters
    stats = {
        "dropped": 0,
        "obs_numeric": 0,
        "obs_date": 0,
        "obs_str": 0,
        "var_numeric": 0,
        "var_str": 0
    }

    # 1. Remove reserved column names
    reserved_names = ['_index']
    for reserved in reserved_names:
        if reserved in adata.obs.columns:
            adata.obs = adata.obs.drop(columns=[reserved])
            stats["dropped"] += 1
        if reserved in adata.var.columns:
            adata.var = adata.var.drop(columns=[reserved])
            stats["dropped"] += 1
    
    numeric_patterns = [
        '_km', '_mg_per_', '_ug_per_', '_umol_per_', '_g_per_kg', '_us_per_cm',
        'SoilGrids_', 'Meteostat_', 'OpenMeteo_', 'EnvironmentalHealth_',
        'NOAA_Tides_distance', '_percent', '_mv', 'latitude', 'longitude',
        'elevation', 'depth', 'temperature', 'ph', 'uranium' # ... rest of patterns
    ]
    date_patterns = ['_date', 'collection_date', '_time', 'temporal_coverage']
    
    # 2. Fix adata.obs dtypes
    for col in adata.obs.columns:
        series = adata.obs[col]
        
        # Geo coordinates forced to float
        if col in ('lat', 'lon', 'latitude', 'longitude'):
            try:
                adata.obs[col] = pd.to_numeric(series, errors='coerce').astype('float64')
                stats["obs_numeric"] += 1
            except Exception: pass
            continue

        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series) or \
            isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_datetime64_any_dtype(series):
            continue

        if series.dtype == object:
            is_date_col = any(pattern in col for pattern in date_patterns)
            if is_date_col:
                try:
                    adata.obs[col] = pd.to_datetime(series, errors='coerce').dt.strftime('%Y-%m-%d').fillna('').astype(str)
                    stats["obs_date"] += 1
                    continue
                except Exception: pass
            
            is_numeric_col = any(pattern in col for pattern in numeric_patterns)
            try:
                numeric = pd.to_numeric(series, errors='coerce')
                valid_count = series.notna().sum()
                num_valid = numeric.notna().sum()
                
                if num_valid >= valid_count * 0.9 or (is_numeric_col and num_valid > 0):
                    if (numeric.dropna() % 1 == 0).all() and len(numeric.dropna()) > 0:
                        adata.obs[col] = numeric.astype('Int64')
                    else:
                        adata.obs[col] = numeric.astype('float64')
                    stats["obs_numeric"] += 1
                else:
                    adata.obs[col] = series.astype(str)
                    stats["obs_str"] += 1
            except Exception:
                adata.obs[col] = series.astype(str)
                stats["obs_str"] += 1
    
    # 3. Fix adata.var dtypes
    for col in adata.var.columns:
        series = adata.var[col]
        if pd.api.types.is_numeric_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            continue
        
        if series.dtype == object:
            try:
                numeric = pd.to_numeric(series, errors='coerce')
                if numeric.notna().sum() >= series.notna().sum() * 0.9:
                    adata.var[col] = numeric.astype('Int64' if (numeric.dropna() % 1 == 0).all() else 'float64')
                    stats["var_numeric"] += 1
                else:
                    adata.var[col] = series.astype(str)
                    stats["var_str"] += 1
            except Exception:
                adata.var[col] = series.astype(str)
                stats["var_str"] += 1

    # Single succinct summary logger statement
    summary = (
        f"AnnData Dtypes Fixed | "
        f"Obs: {stats['obs_numeric']} num, {stats['obs_date']} date, {stats['obs_str']} str | "
        f"Var: {stats['var_numeric']} num, {stats['var_str']} str | "
        f"Dropped: {stats['dropped']} reserved cols"
    )
    logger.info(summary)
    
    if not inplace:
        return adata
    
def _process_single_file(f: Path, config, cache_dir: Optional[Path] = None):
    """
    Worker function: Loads, cleans, and sanitizes a single .h5ad file.
    Handles caching internally.
    """
    logger = get_logger("workflow_16s")
    cache_file = None
    
    # 1. Check Cache
    if cache_dir:
        cache_file = cache_dir / f"{f.stem}_{_get_file_hash(f)}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as cf:
                    cached_adata = pickle.load(cf)
                is_valid, _ = _validate_cached_adata(cached_adata)
                if is_valid: 
                    return cached_adata
                else: 
                    cache_file.unlink() # Corrupt cache
            except Exception:
                if cache_file.exists(): cache_file.unlink()

    # 2. Load & Process
    try:
        adata = sc.read_h5ad(f)
        # A. Sanitize (Minimal fixes)
        adata = _sanitize_adata(adata)
        # B. Clean Metadata (Type Inference)
        adata = clean_metadata(adata, config)
        if adata is None or adata.n_obs == 0: return None
        # C. Parse & Filter
        adata = parse_taxonomy(adata)
        adata = filter_samples_and_features(adata, config)
        if adata is None or adata.n_obs == 0: return None
        adata = _sanitize_obs(adata)
        # D. Type Fixing (Final check for HDF5 compatibility)
        fix_adata_dtypes(adata)
        
        # 3. Save to Cache
        if cache_file:
            try:
                with open(cache_file, 'wb') as cf: 
                    pickle.dump(adata, cf, protocol=4)
            except Exception as e:
                logger.warning(f"Failed to cache {f.name}: {e}")
        return adata
        
    except Exception as e:
        logger.warning(f"Failed to load {f.name}: {e}")
        return None

def format_bytes(size: int) -> str:
    """Converts bytes to a human-readable string (KB, MB, GB)."""
    if size < 1024:
        return f"{size} bytes"
    elif size < 1024**2:
        return f"{size/1024:.2f} KB"
    elif size < 1024**3:
        return f"{size/1024**2:.2f} MB"
    else:
        return f"{size/1024**3:.2f} GB"
    
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
    logger = get_logger("workflow_16s")
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
        
def safe_write_h5ad(adata: ad.AnnData, filename: str, fix_dtypes: bool = True, compression: Optional[str] = None):
    """Safely write AnnData to h5ad file."""
    logger = get_logger("workflow_16s")
    if fix_dtypes:
        adata_copy = adata.copy()
        fix_adata_dtypes(adata_copy, inplace=True)
        adata_copy.write_h5ad(filename, compression=compression if compression in ('gzip', 'lzf', None) else None)
        logger.info(f"Saved AnnData to {filename} (with dtype fixes)")
    else:
        adata.write_h5ad(filename, compression=compression if compression in ('gzip', 'lzf', None) else None)
        logger.info(f"Saved AnnData to {filename}")

def safe_outer_merge(adatas: List[ad.AnnData]) -> ad.AnnData:
    """Safely merges AnnData objects while preserving Taxonomy."""
    merged = ad.concat(adatas, join="outer", merge="unique", uns_merge="unique", fill_value=0)
    all_vars = [a.var for a in adatas]
    combined_var = pd.concat(all_vars)
    combined_var = combined_var[~combined_var.index.duplicated(keep='first')]
    merged.var = combined_var.reindex(merged.var_names)
    return merged

def hierarchical_merge(adatas: List[ad.AnnData]) -> Optional[ad.AnnData]:
    """Merges a list of AnnData objects using concatenation."""
    if not adatas: return None
    try:
        # Outer join preserves features present in ANY dataset
        return safe_outer_merge(adatas)
    except Exception as e:
        logger = get_logger("workflow_16s")
        logger.error(f"Merge failed: {e}")
        return None
    
def create_anndata_from_qiime_artifacts(
    feature_table_biom_path: Path,
    taxonomy_tsv_path: Path,
    rep_seqs_fasta_path: Path,
    rooted_tree_nwk_path: Path,
    metadata_path: Path
) -> ad.AnnData:
    """
    Loads exported QIIME 2 files and creates a comprehensive AnnData object.
    The phylogenetic tree is serialized to a Newick string for file compatibility.
    """
    
    logger = get_logger("workflow_16s")
    logger.info("Starting AnnData object creation process from QIIME 2 artifacts.")

    # 1. Load BIOM Table (Counts)
    logger.info(f"--> Step 1: Loading BIOM feature table from: {feature_table_biom_path}")
    biom_table = biom.load_table(str(feature_table_biom_path))
    # matrix_data is already a scipy sparse matrix
    # We transpose because AnnData expects Samples (obs) x Features (var)
    sparse_matrix = biom_table.matrix_data.T.tocsr().astype('float32')
    #table_df = biom_table.to_dataframe(dense=True)
    logger.info(f"       ...Loaded table with {sparse_matrix.shape[0]} features and {sparse_matrix.shape[1]} samples.")

    # 2. Load Taxonomy
    logger.info(f"--> Step 2: Loading taxonomy data from: {taxonomy_tsv_path}")
    #tax_df = pd.read_csv(taxonomy_tsv_path, sep='\t', index_col=0)
    tax_df = pd.read_csv(taxonomy_tsv_path, sep='\t', index_col=0)
    tax_df.index.name = "feature-id"
    # If GG2 exports it as 'lineage', rename it back to 'Taxon' for your suite
    if 'lineage' in tax_df.columns and 'Taxon' not in tax_df.columns:
        tax_df.rename(columns={'lineage': 'Taxon'}, inplace=True)
    elif 'Taxon' not in tax_df.columns:
        # Handle cases where the column might be named 'taxonomy'
        possible_cols = ['taxonomy', 'lineage', 'Taxonomy']
        for col in possible_cols:
            if col in tax_df.columns:
                tax_df.rename(columns={col: 'Taxon'}, inplace=True)
                break
    
    logger.info(f"       ...Loaded taxonomy for {len(tax_df)} features.")

    # 3. Create initial AnnData object (observations x variables)
    logger.info("--> Step 3: Transposing table and creating initial AnnData object.")
    adata = ad.AnnData(
        X=sparse_matrix,
        obs_names=biom_table.ids(axis='sample'),
        var_names=biom_table.ids(axis='observation')
    )
    logger.info(f"       ...Created AnnData object with shape: {adata.n_obs} samples (obs) x {adata.n_vars} features (vars).")

    # 4. Add Sample Metadata to .obs
    logger.info(f"--> Step 4: Loading and attaching sample metadata to .obs from: {metadata_path}")
    if metadata_path.exists():
        original_adata_ids = adata.obs_names.copy()
        # Ensure index_col=0 to use the first column (sample IDs) as index
        sample_metadata = pd.read_csv(metadata_path, sep='\t', index_col=0, dtype=str)
        sample_metadata.dropna(how='all', axis=1, inplace=True) # Drop fully empty columns
        logger.info(f"       ...Loaded metadata for {sample_metadata.shape[0]} samples and {sample_metadata.shape[1]} columns.")

        original_obs_count = adata.n_obs
        # Reindex metadata to match AnnData's observation names (sample IDs)
        adata.obs = sample_metadata.reindex(adata.obs_names)
        # Check how many samples in AnnData found a match in the metadata
        matches = adata.obs.notna().any(axis=1).sum()
        logger.info(f"       ...Aligned metadata. {matches} of {original_obs_count} samples had matching metadata.")

        if matches == 0 and original_obs_count > 0:
            logger.error("CRITICAL: Sample ID mismatch detected. No samples in the feature table could be matched with the metadata file.")
            logger.error(f"       First 5 sample IDs from feature table (biom): {original_adata_ids[:5].tolist()}")
            logger.error(f"       First 5 sample IDs from metadata file (.tsv): {sample_metadata.index[:5].tolist()}")
            raise ValueError("Could not align sample metadata. Check sample IDs in BIOM table and metadata file.")

        logger.info("       ...Ensuring all sample metadata columns are string-formatted for compatibility.")
        # Fill NaNs with empty string and convert object columns to string
        for col in adata.obs.select_dtypes(include=['object']).columns:
            adata.obs[col] = adata.obs[col].fillna('').astype(str)
        # Also convert any remaining non-numeric, non-string columns if necessary
        for col in adata.obs.columns:
            if adata.obs[col].dtype not in ['float64', 'int64', 'bool', 'str']:
                adata.obs[col] = adata.obs[col].astype(str)

    # 5. Add Taxonomy to .var
    logger.info("--> Step 5: Attaching feature taxonomy data to .var")
    original_var_count = adata.n_vars
    # Join taxonomy data, aligning by feature ID (index)
    adata.var = adata.var.join(tax_df.reindex(adata.var_names))
    # Check how many features got taxonomy info
    tax_col = 'Taxon' # Default QIIME taxonomy column name
    if tax_col in adata.var.columns:
        matches = adata.var[tax_col].notna().sum()
        logger.info(f"       ...Aligned taxonomy. {matches} of {original_var_count} features had matching taxonomy.")
    else:
        logger.warning(f"       ...Taxonomy column '{tax_col}' not found after join.")

    # 6. Add Representative Sequences to .var
    logger.info(f"--> Step 6: Loading and attaching representative sequences to .var from: {rep_seqs_fasta_path}")
    try:
        seqs = {seq.metadata['id']: str(seq) for seq in skbio.read(str(rep_seqs_fasta_path), format='fasta')}
        logger.info(f"       ...Parsed {len(seqs)} sequences from FASTA file.")
        # Create a pandas Series, reindex to match AnnData's variable names (feature IDs)
        seq_series = pd.Series(seqs, name="sequence").reindex(adata.var_names)
        adata.var['sequence'] = seq_series
        matches = adata.var['sequence'].notna().sum()
        logger.info(f"       ...Attached sequences. {matches} of {original_var_count} features had a matching sequence.")
    except Exception as e:
        logger.error(f"       ...Error reading or processing representative sequences FASTA file: {e}")
        adata.var['sequence'] = pd.NA # Add column but indicate failure

    # 7. Add Phylogenetic Tree to .uns
    logger.info(f"--> Step 7: Loading and serializing phylogenetic tree to .uns from: {rooted_tree_nwk_path}")
    if not rooted_tree_nwk_path.exists():
        logger.warning(
            f"       ...Phylogenetic tree file not found at {rooted_tree_nwk_path}. This may indicate tree building failed in QIIME 2."
            f"       ...Downstream phylogenetic diversity analysis will not be available for this dataset."
        )
        adata.uns['phylogenetic_tree'] = None
    else:
        try:
            phylogenetic_tree = TreeNode.read(str(rooted_tree_nwk_path), format='newick')

            # Serialize tree to Newick string format
            with io.StringIO() as fh:
                phylogenetic_tree.write(fh, format='newick')
                newick_string = fh.getvalue()
            adata.uns['phylogenetic_tree'] = newick_string
            logger.info("       ...Successfully stored phylogenetic tree as a Newick string in adata.uns['phylogenetic_tree'].")
        except Exception as e:
            logger.error(f"       ...Error reading or processing phylogenetic tree file: {e}")
            adata.uns['phylogenetic_tree'] = None # Store None to indicate failure

    logger.info("--> Final Step: Ensuring all feature metadata columns are string-formatted for compatibility.")
    # Fill NaNs and convert object columns to string in .var
    for col in adata.var.select_dtypes(include=['object']).columns:
        adata.var[col] = adata.var[col].fillna('').astype(str)
    # Also convert any remaining non-numeric, non-string columns if necessary
    for col in adata.var.columns:
        if adata.var[col].dtype not in ['float64', 'int64', 'bool', 'str']:
            adata.var[col] = adata.var[col].astype(str)

    logger.info("✅ AnnData object creation complete.")
    return adata

def quick_taxonomy_check(adata: ad.AnnData) -> bool:
    """Quick check to see if taxonomy columns are present and look valid."""
    logger = get_logger("workflow_16s")
    logger.info("Performing quick taxonomy check...")
    logger.info(f"Variable columns in .var: {adata.var.columns.tolist()}")
    if 'Genus' in adata.var.columns:
        logger.info("Found 'Genus' column in .var.")
        if adata.var['Genus'].dropna().astype(str).str.strip().ne('').any():
            logger.info("'Genus' column has non-empty values.")
            logger.info(f"Unique 'Genus' values (sample of 20): {adata.var['Genus'].unique()[:20]}")
            return True
        else:
            logger.warning("'Genus' column appears to be empty or all null.")
            return False
    tax_cols = [col for col in adata.var.columns if 'genus' in col.lower()]
    if not tax_cols:
        logger.warning("No taxonomy columns found in .var. Expected a column with 'genus' in the name.")
        return False
    
    # Check if the identified taxonomy column has non-empty values
    for col in tax_cols:
        if adata.var[col].dropna().astype(str).str.strip().ne('').any():
            logger.info(f"Found taxonomy column '{col}' with non-empty values.")
            return True
    
    logger.warning(f"Taxonomy columns found ({tax_cols}) but they appear to be empty or all null.")
    return False

def validate_anndata_file(anndata_path: Path, subset_id: str):
    """
    Performs quality control checks on a newly created AnnData file.
    Checks for essential components like obs/var indices, taxonomy, sequences, and tree.
    """
    
    logger = get_logger("workflow_16s")
    logger.info(f"Starting AnnData validation for subset '{subset_id}'...")
    errors = []
    try:
        adata = ad.read_h5ad(anndata_path)

        # Basic structure checks
        if adata.n_obs == 0 or adata.n_vars == 0:
            errors.append("AnnData object is empty (n_obs=0 or n_vars=0).")
        # Check obs (samples)
        if not adata.obs.index.is_unique:
            errors.append("Sample IDs in '.obs.index' are not unique.")
        # Check if 'run_accession' exists, crucial for linking back
        if 'run_accession' not in adata.obs.columns:
            # If not present, check if the index itself looks like run accessions
            # This is a fallback check, ideally 'run_accession' column should exist
            if not all(re.match(r'[ESD]RR\d+', str(idx)) for idx in adata.obs.index):
                errors.append("'.obs' is missing the 'run_accession' column, and index does not appear to be run accessions.")

        # Check var (features)
        if not adata.var.index.is_unique:
            errors.append("Feature IDs in '.var.index' are not unique.")
        if 'Taxon' not in adata.var.columns: # Assuming 'Taxon' is the standard QIIME output column name
            errors.append("'.var' is missing the 'Taxon' column for taxonomy.")
        if 'sequence' not in adata.var.columns:
            errors.append("'.var' is missing the 'sequence' column for representative sequences.")
        elif adata.var['sequence'].isnull().any():
            errors.append("The 'sequence' column in '.var' contains null/missing values.")
        """
        # Check uns (unstructured data - tree)
        if 'phylogenetic_tree' not in adata.uns:
            errors.append("'.uns' is missing the 'phylogenetic_tree'.")
        else:
            tree_data = adata.uns['phylogenetic_tree']
            if not isinstance(tree_data, str) or not tree_data: # Check if it's a non-empty string
                errors.append(f"'.uns['phylogenetic_tree']' should be a non-empty string, but found type {type(tree_data)}.")
            else:
                # Validate if the string is parseable as Newick
                try:
                    TreeNode.read(io.StringIO(tree_data))
                except Exception as e:
                    errors.append(f"'.uns['phylogenetic_tree']' is not a valid Newick string. Parser error: {e}")
        """
        if 'phylogenetic_tree' in adata.uns:
            if adata.uns['phylogenetic_tree'] is not None:
                logger.info("  - Validation: Found valid 'phylogenetic_tree' in .uns.")
                # You could add more checks here, e.g., type(adata.uns['phylogenetic_tree'])
            else:
                logger.info("  - Validation: 'phylogenetic_tree' is present but None (accepted).")
        else:
            # This is the case you are hitting. It's now accepted.
            logger.info("  - Validation: No 'phylogenetic_tree' found in .uns (accepted).")
            
    except Exception as e:
        errors.append(f"Failed to read or perform basic validation on the H5AD file: {e}")

    # Report errors or success
    if errors:
        error_summary = "\n - ".join(errors)
        raise ValueError(f"AnnData validation failed for '{subset_id}' ({anndata_path}):\n - {error_summary}")
    else:
        logger.info(f"✅ AnnData file for '{subset_id}' passed all validation checks.")

def export_fasta(
    adata: ad.AnnData, 
    config: Union[AppConfig, dict], 
    output_dir: Union[str, Path]
) -> None:
    """Exports sequences to FASTA."""
    logger = get_logger("workflow_16s")
    if 'sequence' not in adata.var.columns: return
    fasta_path = Path(output_dir) / "all_features.fasta"
    try:
        with open(fasta_path, "w") as f:
            for feat_id, seq in adata.var['sequence'].dropna().items():
                f.write(f">{feat_id}\n{seq}\n")
        logger.info(f"FASTA exported to {fasta_path}")
    except Exception as e: logger.error(f"FASTA export failed: {e}")
    
import os
import platform
import datetime
import json
import zipfile
import yaml
import pandas as pd
from typing import Any

# ==========================================
# 🛠️ YAML LOADER FOR QIIME 2 DAG
# ==========================================
class SafeQiimeLoader(yaml.SafeLoader):
    pass

def construct_unknown(loader, tag_suffix, node):
    if isinstance(node, yaml.ScalarNode): return loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode): return loader.construct_sequence(node)
    elif isinstance(node, yaml.MappingNode): return loader.construct_mapping(node)

SafeQiimeLoader.add_multi_constructor('!', construct_unknown)

def extract_qza_provenance_dict(qza_path: str) -> dict:
    """Extracts the QIIME 2 DAG as a nested Python dictionary."""
    full_provenance = {}
    if not qza_path or not os.path.exists(qza_path):
        return {"error": "QZA file not found or not provided."}
        
    try:
        with zipfile.ZipFile(qza_path, 'r') as z:
            action_files = [f for f in z.namelist() if f.endswith('action.yaml') and 'provenance' in f]
            if not action_files: return {}
            
            for action_file in action_files:
                parts = action_file.split('/')
                node_type = "ancestor_step" if 'artifacts' in parts else "final_step"
                uuid = parts[parts.index('artifacts') + 1] if 'artifacts' in parts else parts[0]
                
                with z.open(action_file) as f:
                    action_data = yaml.load(f, Loader=SafeQiimeLoader)
                    
                    # 🛠️ THE FIX: Look in the 'action' block, not 'execution'
                    action_block = action_data.get('action', {})
                    
                    plugin_name = action_block.get('plugin', 'unknown_plugin')
                    action_name = action_block.get('action', 'unknown_action')
                    
                    full_provenance[uuid] = {
                        "step_type": node_type,
                        "plugin": f"{plugin_name}:{action_name}",
                        "parameters": action_block.get('parameters', {})
                    }
            return full_provenance
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 🧬 MAIN PROVENANCE INJECTION
# ==========================================
workflow_version = '1.2.0'

def embed_provenance(
    adata: Any, 
    subset_id: str,
    config: Any,
    start_time: datetime.datetime, # Ensure this is passed in if it's not a class method
    qza_source_path: str = None,   # Pass the target table.qza path here
    qiime2_env: str = 'qiime2-amplicon-2025.7',
):
    """Inserts processing timeline, system info, and QIIME DAG into h5ad .uns slot."""
    
    # 1. Build a unified, nested dictionary
    provenance_dict = {
        'workflow_metadata': {
            'subset_id': subset_id,
            'processing_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'duration_seconds': (datetime.datetime.now() - start_time).total_seconds(),
            'system_info': {
                'os': platform.system(),
                'node': platform.node(),
                'cpu_count': os.cpu_count()
            },
            'software_versions': {
                'workflow_version': workflow_version,
                'qiime2_env': qiime2_env,
                'pandas': pd.__version__
            },
            'parameters': {
                'threads_used': getattr(config.sequences.validate_16s, 'n_threads', 'unknown'),
                'trim_enabled': getattr(config.qiime2.per_dataset.trim, 'enabled', 'unknown')
            }
        },
        'qiime2_dag': extract_qza_provenance_dict(qza_source_path) if qza_source_path else {"status": "No QZA provided."}
    }
    
    # 2. Serialize the ENTIRE dictionary to a JSON string to prevent HDF5 crashes
    # Using default=str ensures datetime objects and weird types don't break the dump
    adata.uns['provenance'] = json.dumps(provenance_dict, indent=2, default=str)
    
    # get_logger("workflow_16s").info(f" 📜 Embedded standardized provenance JSON in {subset_id}.h5ad")