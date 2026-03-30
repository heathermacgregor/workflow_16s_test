# ==================================================================================== #
#                         downstream/utils/adata_utils.py
# ==================================================================================== #

"""
AnnData Utilities for Enhanced Statistics

Helper functions for ensuring AnnData objects are compatible with h5py writing
and other operations.
"""

import gc
import os
import hashlib
import math
import pickle
import psutil
import re
import requests
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from joblib import Parallel, delayed
from pandas.api.types import is_extension_array_dtype
from scipy.sparse import issparse, csr_matrix, csc_matrix

from workflow_16s.utils.logger import get_logger
from workflow_16s.utils.progress import get_progress_bar

from workflow_16s.utils.logger import get_logger


class GeoContextEnricher:
    """
    Derives forensic environmental context from Lat/Lon/Date using 
    mathematical models and high-speed APIs.
    """
    
    def __init__(self, adata):
        self.adata = adata
        self.obs = adata.obs
        
    def run_all(self):
        """Orchestrates the full enrichment pipeline."""
        logger = get_logger("workflow_16s")
        logger.info("--- 🌍 Starting Geo-Context Enrichment ---")
        
        # 1. Astronomical (Instant, Math-based)
        self._add_astronomical_features()
        
        # 2. Elevation (Batch API)
        self._add_elevation_batch()
        
        # 3. Historical Weather (API)
        self._add_historical_weather()
        
        return self.adata

    def _add_astronomical_features(self):
        """
        Calculates day length and season. 
        Forensic relevance: Photosynthetic activity (Cyanobacteria) and seasonal blooms.
        """
        logger = get_logger("workflow_16s")
        logger.info("   ☀️  Calculating solar/seasonal features...")
        
        def get_day_length(row):
            try:
                lat = float(row.get('lat'))
                date_obj = pd.to_datetime(row.get('collection_date'))
                if pd.isna(lat) or pd.isna(date_obj): return np.nan
                
                # Day of year (1-365)
                doy = date_obj.timetuple().tm_yday
                
                # Calculation
                p = math.pi / 180
                m = 1 - math.tan(lat * p) * math.tan(23.44 * p * math.cos((doy - 172) * 2 * math.pi / 365))
                m = max(0, min(m, 2))
                day_len = 24 * (1 - math.acos(1 - m) / math.pi)
                return day_len
            except:
                return np.nan

        def get_season(row):
            # 0=Spring, 1=Summer, 2=Autumn, 3=Winter (Simplified)
            try:
                date_obj = pd.to_datetime(row.get('collection_date'))
                lat = float(row.get('lat'))
                if pd.isna(date_obj): return "Unknown"
                
                doy = date_obj.timetuple().tm_yday
                # Hemisphere adjustment
                if lat < 0: doy = (doy + 182.5) % 365
                
                if 80 <= doy < 172: return "Spring"
                elif 172 <= doy < 264: return "Summer"
                elif 264 <= doy < 355: return "Autumn"
                else: return "Winter"
            except:
                return "Unknown"

        self.obs['calc_day_length_hours'] = self.obs.apply(get_day_length, axis=1)
        self.obs['calc_season'] = self.obs.apply(get_season, axis=1)

    def _add_elevation_batch(self, chunk_size=100):
        """
        Fetches elevation from Open-Elevation API in batches.
        Forensic relevance: Oxygen levels, UV exposure, atmospheric pressure.
        """
        logger = get_logger("workflow_16s")
        # Identify missing samples
        if 'elevation_m' not in self.obs.columns:
            self.obs['elevation_m'] = np.nan
            
        mask = (self.obs['lat'].notnull()) & \
               (self.obs['lon'].notnull()) & \
               (self.obs['elevation_m'].isnull())
        
        targets = self.obs[mask]
        if targets.empty:
            logger.info("   🏔️  Elevation data is complete. Skipping lookup.")
            return

        logger.info(f"   🏔️  Fetching elevation for {len(targets)} samples...")
        
        coords = []
        indices = []
        
        # Prepare batches
        for idx, row in targets.iterrows():
            coords.append({"latitude": row['lat'], "longitude": row['lon']})
            indices.append(idx)
            
            if len(coords) >= chunk_size:
                self._query_elevation(coords, indices)
                coords, indices = [], []
                time.sleep(0.5) # Be polite
        
        if coords: self._query_elevation(coords, indices)

    def _query_elevation(self, coords_list, indices_list):
        try:
            url = "https://api.open-elevation.com/api/v1/lookup"
            resp = requests.post(url, json={"locations": coords_list}, timeout=10)
            if resp.status_code == 200:
                results = resp.json()['results']
                for idx, res in zip(indices_list, results):
                    self.obs.at[idx, 'elevation_m'] = res['elevation']
        except Exception as e:
            logger = get_logger("workflow_16s")
            logger.warning(f"Elevation batch failed: {e}")

    def _add_historical_weather(self):
        """
        Smart-fetches weather with Deduplication, Caching, and Retry Logic.
        """
        logger = get_logger("workflow_16s")
        logger.info("   🌦️  Fetching historical weather (Smart Batching)...")
        # ✅ FIX: Initialize columns if they don't exist
        for col in ['weather_temp_avg', 'weather_precip_sum']:
            if col not in self.obs.columns:
                self.obs[col] = np.nan
        # 1. Setup Cache File
        # TODO: Make cache path configurable and more robust (e.g., include hash of unique keys to avoid collisions)
        cache_dir = Path("data/cache/weather") # Adjust path as needed
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "weather_lookup_cache.json"
        
        # Load existing cache
        weather_cache = {}
        if cache_file.exists():
            import json
            try:
                with open(cache_file, 'r') as f:
                    weather_cache = json.load(f)
                logger.info(f"       Loaded {len(weather_cache)} cached weather records.")
            except:
                logger.warning("       Cache file corrupted, starting fresh.")

        # 2. Identify Unique Contexts (Reduce API calls)
        # Create a temporary ID: "lat_rounded|lon_rounded|date"
        # 2 decimals = ~1.1km resolution. Sufficient for regional weather.
        self.obs['weather_key'] = (
            self.obs['lat'].round(2).astype(str) + "|" + 
            self.obs['lon'].round(2).astype(str) + "|" + 
            self.obs['collection_date'].astype(str)
        )
        
        # Filter: Needs Lat/Lon/Date, but missing Temp
        mask = (self.obs['lat'].notnull()) & \
               (self.obs['collection_date'].notnull()) & \
               (self.obs['weather_temp_avg'].isnull())
        
        # Only query keys we haven't seen before
        unique_keys = self.obs.loc[mask, 'weather_key'].unique()
        unique_keys = [k for k in unique_keys if k not in weather_cache]
        
        logger.info(f"       Found {len(unique_keys)} unique location/date combinations to fetch.")

        # 3. Fetch Loop with Rate Limiting
        import json
        
        for i, key in enumerate(unique_keys):
            try:
                lat_str, lon_str, date_str = key.split('|')
                
                # Check for "Unknown" or bad dates
                if 'nan' in key.lower() or len(date_str) < 10: 
                    weather_cache[key] = None # Mark as un-fetchable
                    continue

                # Prepare Date Range (3-day window)
                date_obj = pd.to_datetime(date_str)
                start = (date_obj - timedelta(days=1)).strftime('%Y-%m-%d')
                end = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')

                url = "https://archive-api.open-meteo.com/v1/archive"
                params = {
                    "latitude": lat_str,
                    "longitude": lon_str,
                    "start_date": start,
                    "end_date": end,
                    "daily": "temperature_2m_mean,precipitation_sum",
                    "timezone": "auto"
                }

                # Retry Logic
                max_retries = 3
                for attempt in range(max_retries):
                    r = requests.get(url, params=params, timeout=10)
                    
                    if r.status_code == 200:
                        data = r.json()
                        if 'daily' in data:
                            t_mean = np.mean(data['daily']['temperature_2m_mean'])
                            p_sum = np.sum(data['daily']['precipitation_sum'])
                            weather_cache[key] = {'t': t_mean, 'p': p_sum}
                        break # Success
                    
                    elif r.status_code == 429: # Rate Limit Hit
                        wait_time = 60 * (attempt + 1)
                        logger.warning(f"       ⚠️ API Rate Limit (429). Sleeping {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        break # Other error, skip

                # Progressive Save (Every 20 calls)
                if i % 20 == 0:
                    with open(cache_file, 'w') as f:
                        json.dump(weather_cache, f)
                    time.sleep(0.5) # Gentle rate limiting

            except Exception as e:
                logger.debug(f"Failed fetching {key}: {e}")
                continue

        # Final Save
        with open(cache_file, 'w') as f:
            json.dump(weather_cache, f)

        # 4. Map Results Back to DataFrame (Vectorized)
        logger.info("       Mapping weather data to samples...")
        
        # Convert cache to DataFrame for merge
        cache_df = pd.DataFrame.from_dict(weather_cache, orient='index')
        if not cache_df.empty:
            cache_df.index.name = 'weather_key'
            cache_df.columns = ['temp_lookup', 'precip_lookup']
            
            # Reset index on obs to ensure alignment (preserving original index)
            self.obs = self.obs.merge(cache_df, on='weather_key', how='left')
            
            # Fill main columns
            self.obs['weather_temp_avg'] = self.obs['weather_temp_avg'].fillna(self.obs['temp_lookup'])
            self.obs['weather_precip_sum'] = self.obs['weather_precip_sum'].fillna(self.obs['precip_lookup'])
            
            # Cleanup
            self.obs.drop(columns=['weather_key', 'temp_lookup', 'precip_lookup'], inplace=True, errors='ignore')
            self.adata.obs = self.obs # Ensure Anndata is updated

def run_enrichment(adata):
    """Wrapper function to be called from analysis pipeline."""
    enricher = GeoContextEnricher(adata)
    return enricher.run_all()

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

def sanitize_obs(adata_obj):
    logger = get_logger("workflow_16s")
    for col in adata_obj.obs.columns:
        dtype_name = adata_obj.obs[col].dtype.name
                        
        # 1. Fix Object AND Category columns
        if dtype_name == 'object' or dtype_name == 'category':
            # Force to pure string series first to safely use .replace()
            cleaned_series = adata_obj.obs[col].astype(str).replace(
                ['nan', 'None', 'NaN', 'NoneType', '<NA>', '<nan>'], ''
            )
            
            # CRITICAL: Restore the 'category' dtype if it originally had it
            if dtype_name == 'category':
                adata_obj.obs[col] = cleaned_series.astype('category')
            else:
                adata_obj.obs[col] = cleaned_series
                        
        # 2. Fix Nullable Float/Int (The Meteostat Fix)
        elif is_extension_array_dtype(adata_obj.obs[col].dtype):
            try:
                adata_obj.obs[col] = adata_obj.obs[col].to_numpy(dtype=float, na_value=np.nan)
            except Exception: # Good practice to catch explicit Exception
                cleaned_series = adata_obj.obs[col].astype(str).replace(
                    ['nan', 'None', 'NaN', 'NoneType', '<NA>'], ''
                )
                adata_obj.obs[col] = cleaned_series
                
    return adata_obj

def _sanitize_adata(adata: ad.AnnData) -> ad.AnnData:
    """
    Minimal sanitization to ensure merging works. 
    Does NOT force types aggressively to allow clean_metadata to do its job.
    """
    # 1. Fix Index Name Conflicts
    if adata.obs_names.name is None: adata.obs_names.name = 'sample_id'
    if adata.var_names.name is None: adata.var_names.name = 'feature_id'
    # If index name is duplicated as a column, remove the column to avoid confusion
    if not adata.obs.index.is_unique: adata.obs_names_make_unique()
    # Remove columns that duplicate the index name 
    for idx_name in [adata.obs_names.name, adata.var_names.name]:
        if idx_name in adata.obs.columns:
            try: del adata.obs[idx_name]
            except: pass
            
    # 2. Force coordinates to numeric (if present) and handle known problematic columns
    for coord in ['lat', 'lon', 'latitude', 'longitude']:
        if coord in adata.obs.columns:
            adata.obs[coord] = pd.to_numeric(adata.obs[coord], errors='coerce')

    problem_cols = ['facility_capacity', 'facility_match', 'study_accession']
    for col in adata.obs.columns:
        # If the column is the known culprit or an 'object' type, force it to string
        if col in problem_cols or adata.obs[col].dtype == 'object':
            # Convert to string and handle NaNs consistently
            adata.obs[col] = adata.obs[col].astype(str).replace(['nan', 'None', 'NaN'], '')
            
    # 3. Strip whitespace from index and variable names
    try: adata.var_names = adata.var_names.str.strip().tolist()
    except: pass
    try: adata.obs_names = adata.obs_names.str.strip().tolist()
    except: pass
    
    # 4. Ensure X is sparse (h5py doesn't like dense matrices with object dtypes)
    if not issparse(adata.X): adata.X = csr_matrix(adata.X)

    return adata

def safe_outer_merge(adatas: List[ad.AnnData]) -> ad.AnnData:
    """
    Safely merges AnnData objects using outer join while PRESERVING .var (Taxonomy).
    Standard ad.concat drops .var columns if they aren't shared; this fixes that.
    """
    # 1. Perform the standard merge (Gets X and Obs correct)
    merged = ad.concat(
        adatas, 
        join="outer", 
        merge="unique", 
        uns_merge="unique", 
        fill_value=0
    )
    
    # 2. Manually recover the .var (Feature Metadata)
    # Concatenate all .var dataframes from the inputs
    all_vars = [a.var for a in adatas]
    combined_var = pd.concat(all_vars)
    
    # Deduplicate: If an ASV appears in multiple datasets, just keep the first instance's taxonomy
    # (Taxonomy shouldn't change for the same ASV ID)
    combined_var = combined_var[~combined_var.index.duplicated(keep='first')]
    
    # 3. Re-align to the merged object's feature order
    # This fills missing taxa with NaN (which we'll handle later)
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

def _validate_cached_adata(adata) -> Tuple[bool, str]:
    """Quick sanity check on cached adata object."""
    if adata is None: return False, "None object"
    if not isinstance(adata, ad.AnnData): return False, "Not AnnData"
    if adata.n_obs == 0: return False, "Empty observations"
    return True, ""

def _get_file_hash(filepath: Path) -> str:
    """Fast hash of file stats to detect changes without reading content."""
    stats = filepath.stat()
    return hashlib.md5(f"{stats.st_size}_{stats.st_mtime}".encode()).hexdigest()[:8]

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
        
        # Call our future-proofed sanitizer instead of the bad loop!
        adata = sanitize_obs(adata)
        
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


def inspect_adata_dtypes(adata: ad.AnnData) -> pd.DataFrame:
    """
    Inspect dtypes in AnnData object for debugging.
    
    Returns
    -------
    pd.DataFrame
        Summary of column dtypes in obs and var
    """
    logger = get_logger("workflow_16s")
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

from workflow_16s.config import AppConfig
from workflow_16s.utils.logger import get_logger
from workflow_16s.utils.progress import get_progress_bar


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
        logger = get_logger("workflow_16s")
        logger.debug(f"Taxonomy parsing warning: {e}")
        for rank in ranks:
            if rank not in adata.var.columns: adata.var[rank] = np.nan

    return adata

def filter_samples_and_features(adata, config=None):
    """
    Removes Eukaryota, Mitochondria, Chloroplasts, and empty samples.
    """
    logger = get_logger("workflow_16s")
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