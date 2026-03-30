# workflow_16s/utils/pandas/misc.py
import pandas as pd
from typing import Tuple

def parse_lat_lon(lat_lon_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Parses various lat/lon string formats into numeric Series."""
    if lat_lon_series is None or lat_lon_series.isnull().all(): 
        return pd.Series(dtype='float64'), pd.Series(dtype='float64')
            
    regex = r'([\d\.-]+)\s*([NS])?[\s,]+([\d\.-]+)\s*([EW])?'
    parsed = lat_lon_series.astype(str).str.extract(regex)
        
    if parsed.empty or parsed.isnull().all().all(): 
         return (pd.Series(dtype='float64', index=lat_lon_series.index), 
                pd.Series(dtype='float64', index=lat_lon_series.index))
                    
    lat = pd.to_numeric(parsed[0], errors='coerce')
    lon = pd.to_numeric(parsed[2], errors='coerce')
        
    if 1 in parsed.columns: 
        lat[parsed[1].fillna('').str.upper() == 'S'] *= -1
    if 3 in parsed.columns: 
        lon[parsed[3].fillna('').str.upper() == 'W'] *= -1
            
    lat.index = lat_lon_series.index
    lon.index = lat_lon_series.index
    return lat, lon

# workflow_16s/src/workflow_16s/downstream/utils/metadata.py
"""Utility functions for metadata processing and standardization."""
from typing import Optional, Tuple
import pandas as pd
from workflow_16s.utils.logger import get_logger


TARGET_GENE_NORMALIZATION = {
    '16S': ['16S', '16S rRNA', '16S rRNA gene', '16s', '16s rrna'],
    '18S': ['18S', '18S rRNA', '18s'],
    'ITS': ['ITS', 'ITS1', 'ITS2', 'its'],
}

def normalize_target_gene(value: str) -> Optional[str]:
    if pd.isna(value) or value in ['', 'nan', 'None', 'NA']:
        return None
    value_str = str(value).strip()
    for canonical, synonyms in TARGET_GENE_NORMALIZATION.items():
        if value_str in synonyms:
            return canonical
    return value_str

def standardize_dates(obs_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    date_patterns = ['date', 'time', 'temporal', 'year', 'month', 'day']
    date_columns = [col for col in obs_df.columns if any(pattern in col.lower() for pattern in date_patterns)]
    standardized_count = 0
    for col in date_columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(obs_df[col]):
                dt_series = pd.to_datetime(obs_df[col], errors='coerce')
                obs_df[col] = dt_series.dt.strftime('%Y-%m-%d').replace('NaT', pd.NA)
                standardized_count += 1
                continue
            if pd.api.types.is_numeric_dtype(obs_df[col]): continue
            if obs_df[col].dtype == 'object' or pd.api.types.is_string_dtype(obs_df[col]):
                dt_series = pd.to_datetime(obs_df[col], errors='coerce')
                original_valid = obs_df[col].notna().sum()
                parsed_valid = dt_series.notna().sum()
                if parsed_valid >= original_valid * 0.1 and parsed_valid > 0:
                    obs_df[col] = dt_series.dt.strftime('%Y-%m-%d').replace('NaT', pd.NA)
                    standardized_count += 1
        except Exception: continue
    return obs_df, standardized_count

def coalesce_columns(
    df: pd.DataFrame, 
    base_col: str, 
    suffixes: list[str] | None = None
) -> Optional[pd.Series]:
    """
    Merges columns like 'elevation', 'elevation_sample', 'elevation_study' into one.
    Prioritizes the base column, then fills NAs from the suffixes in order.
    """
    if suffixes is None:
        suffixes = ['_sample', '_study', '_ena', '_facility']
    
    combined = None
    
    # Start with the base column (or empty if it doesn't exist)
    if base_col in df.columns:
        combined = df[base_col].copy()
    else:
        # If base doesn't exist, try to find the first suffix that does
        for suf in suffixes:
            col_name = f"{base_col}{suf}"
            if col_name in df.columns:
                combined = df[col_name].copy()
                break
    
    if combined is None: return None

    # Fill missing values from the suffix columns
    for suf in suffixes:
        col_name = f"{base_col}{suf}"
        if col_name in df.columns and col_name != combined.name:
            # Report how much we are salvaging
            missing_before = combined.isna().sum()
            combined = combined.fillna(df[col_name])
            missing_after = combined.isna().sum()
            
            if missing_before > missing_after:
                get_logger("workflow_16s").info(f"    🔗 Merged '{col_name}' into '{base_col}': Salvaged {missing_before - missing_after} values.")
    
    return combined

def filter_by_prevalence(
    X_df: pd.DataFrame, 
    metadata: pd.DataFrame, 
    min_sample_prevalence: float = 0.10, 
    min_project_prevalence: float = 0.05,
    batch_col: str = 'study_accession'
) -> pd.DataFrame:
    """Filters taxa based on their distribution across samples and projects."""
    logger = get_logger("workflow_16s")
    n_samples = len(X_df)
    n_projects = metadata[batch_col].nunique()
    
    # 1. Sample Prevalence: Must appear in at least X% of total samples
    # We check where abundance is > 0 (pre-CLR) or non-NaN
    sample_counts = (X_df > 0).sum(axis=0)
    keep_samples = sample_counts >= (min_sample_prevalence * n_samples)
    
    # 2. Project Prevalence: Must appear in at least X% of unique studies
    # This prevents the model from picking up Genera specific to just one lab
    project_presence = (X_df > 0).groupby(metadata[batch_col]).any().sum(axis=0)
    keep_projects = project_presence >= (min_project_prevalence * n_projects)
    
    # Combine filters (ensure both are Series with matching index)
    taxa_to_keep = keep_samples & keep_projects
    filtered_df = X_df.loc[:, taxa_to_keep].copy()
    
    # Ensure we always return a DataFrame (even if single column remains)
    if isinstance(filtered_df, pd.Series):
        filtered_df = filtered_df.to_frame()
    
    logger.info(f" ✨ Prevalence Filtering: {X_df.shape[1]} taxa -> {filtered_df.shape[1]} taxa "
                f"(Thresholds: {min_sample_prevalence:.0%} samples, {min_project_prevalence:.0%} projects)")
    
    return filtered_df

def clean_metadata_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main cleaning function to coalesce redundant columns and standardize format.
    """
    logger = get_logger("workflow_16s")
    logger.info("--- Starting Metadata Cleaning & Coalescing ---")
    df_clean = df.copy()

    # 1. IDENTIFY COLUMN GROUPS TO MERGE
    # We look for base names that often have _sample or _study versions
    common_bases = [
        'scientific_name', 'sample_alias', 'sample_title', 'description', 
        'elevation', 'depth', 'location', 'country', 'collection_date',
        'latitude', 'longitude', 'lat', 'lon', 'host_tax_id', 'isolation_source',
        'geo_loc_name', 'env_biome', 'env_material', 'env_feature'
    ]

    for base in common_bases:
        merged_series = coalesce_columns(df_clean, base)
        if merged_series is not None:
            df_clean[base] = merged_series
            # Optional: Drop the suffix columns after merging to clean up
            suffixes = ['_sample', '_study', '_ena', '_facility']
            cols_to_drop = [f"{base}{s}" for s in suffixes if f"{base}{s}" in df_clean.columns]
            if cols_to_drop:
                df_clean.drop(columns=cols_to_drop, inplace=True)

    # 2. SPECIAL MERGES (Geography)
    # Merge 'lat' and 'latitude' (and 'lon'/'longitude')
    logger.info(" 🌍 Merging Latitude/Longitude variants...")
    
    if 'lat' in df_clean.columns and 'latitude' in df_clean.columns:
        df_clean['latitude'] = df_clean['latitude'].fillna(df_clean['lat'])
    
    if 'lon' in df_clean.columns and 'longitude' in df_clean.columns:
        df_clean['longitude'] = df_clean['longitude'].fillna(df_clean['lon'])

    # 3. DROP TRULY EMPTY COLUMNS
    # Only drop columns that are effectively 100% empty (or just contain "not applicable")
    # We keep sparse chemical data (0.1% full) just in case you want to filter for it later.
    min_completeness = 0.001 # Keep if at least 0.1% full
    threshold = len(df_clean) * min_completeness
    
    initial_cols = df_clean.shape[1]
    df_clean = df_clean.dropna(thresh=threshold, axis=1)
    dropped_count = initial_cols - df_clean.shape[1]
    
    if dropped_count > 0:
        logger.info(f" 🗑️ Dropped {dropped_count} columns that were >99.9% empty.")

    logger.info(f" ✅ Metadata cleaning complete. Final shape: {df_clean.shape}")
    return df_clean