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
    """Parses taxonomy strings from 'Tax' or 'taxonomy' column."""
    logger = get_logger("workflow_16s")
    tax_col = 'Tax' if 'Tax' in adata.var.columns else 'taxonomy' if 'taxonomy' in adata.var.columns else None
    if not tax_col:
        logger.warning(f"No taxonomy column found. Available .var columns: {list(adata.var.columns)}")
        return adata

    def _get_genus(t):
        if not isinstance(t, str) or t == 'Unassigned': return 'Unassigned'
        parts = [p.strip() for p in t.split(';') if p.strip()]
        return parts[5] if len(parts) >= 6 else (f'Unclassified_{parts[-1] if parts else "Unknown"}')

    logger.info(f"Parsing Genus from '{tax_col}' column in adata.var (n_features={adata.n_vars})")
    adata.var['Genus'] = adata.var[tax_col].apply(_get_genus)
    
    # Count results
    unique_genera = set(adata.var['Genus'].dropna())
    unassigned_count = (adata.var['Genus'] == 'Unassigned').sum()
    logger.info(f"Taxonomy parsing complete: {len(unique_genera)} unique genera, {unassigned_count} unassigned features")
    
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