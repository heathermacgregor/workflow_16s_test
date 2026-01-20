# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
import os
import requests
import time
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from functools import lru_cache

# Third-Party Imports
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.metrics import confusion_matrix, classification_report
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local Imports
from workflow_16s import constants
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")
_session = requests.Session() # Create a single requests session for reuse

# ==================================== FUNCTIONS ===================================== #  

@lru_cache(maxsize=None)
def _geocode_query(query: str, user_agent: str) -> (float, float):
    """Get coordinates from Nominatim API with caching"""
    url = "https://nominatim.openstreetmap.org/search"
    params = {'q': query, 'format': 'json', 'limit': 1}
    headers = {'User-Agent': user_agent}
    try:
        response = _session.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except Exception as e:
        logger.error(f"Geocoding failed for '{query}': {e}")
    return None, None


def process_and_geocode_db(
    database: str = "GEM",
    file_path: str = constants.DEFAULT_GEM_PATH,
    user_agent: str = constants.DEFAULT_USER_AGENT
):
    """
    Process a data file (Excel or TSV) and add latitude/longitude coordinates
    """
    # Select parameters
    if database == "GEM":
        skip_rows, skip_first_col, column_names = 0, False, constants.DEFAULT_GEM_COLUMNS
    elif database == "NFCIS":
        skip_rows, skip_first_col, column_names = 8, True, constants.DEFAULT_NFCIS_COLUMNS
    else:
        raise ValueError(f"Unknown database: {database}")

    # Detect and load file (only needed columns)
    ext = os.path.splitext(file_path)[1].lower()
    usecols = None
    try:
        if ext in ['.xlsx', '.xls']:
            df_raw = pd.read_excel(
                file_path, header=None, skiprows=skip_rows, usecols=None
            )
        else:
            df_raw = pd.read_csv(
                file_path, sep='\t', header=None, skiprows=skip_rows, usecols=None, 
                encoding_errors='replace'
            )
    except Exception as e:
        df_raw = pd.read_csv(
            file_path, sep='\t', header=None, skiprows=skip_rows, encoding_errors='replace'
        )

    # Drop first column if needed
    df = df_raw.iloc[:, 1:] if skip_first_col else df_raw.copy()

    # Set header and reset
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    logger.info(f"Loaded '{database}' data with {df.shape[0]} NFC facilities")

    # Filter and rename
    df = df[list(column_names.values())]
    df = df.rename(columns={v: k for k, v in column_names.items()})
    df = df[list(column_names.keys())]

    # Prepare geocoding
    df['__query__'] = df['facility'].fillna('') + ', ' + df['country'].fillna('')
    unique_queries = df['__query__'].unique()

    # Geocode unique queries with progress
    coords = {}
    with get_progress_bar() as progress:
        task = progress.add_task(
            _format_task_desc("Geocoding unique locations"), 
            total=len(unique_queries)
        )
        for q in unique_queries:
            coords[q] = _geocode_query(q, user_agent)
            time.sleep(1) 
            progress.update(task, advance=1)

    # Map coords back to DataFrame
    df['latitude_deg']  = df['__query__'].map(lambda q: coords[q][0])
    df['longitude_deg'] = df['__query__'].map(lambda q: coords[q][1])
    df.drop(columns='__query__', inplace=True)

    return df


def sph2cart(latitudes, longitudes, R=6371):
    """Convert spherical lat/lon to Cartesian coordinates"""
    φ = np.radians(latitudes.astype(float))
    λ = np.radians(longitudes.astype(float))
    x = R * np.cos(φ) * np.cos(λ)
    y = R * np.cos(φ) * np.sin(λ)
    z = R * np.sin(φ)
    return np.column_stack((x, y, z))


def match_facilities_to_locations(
    facilities: pd.DataFrame,
    samples: pd.DataFrame,
    max_distance_km: float = 50
) -> pd.DataFrame:
    """Match locations to nearby facilities within a specified distance threshold.
    Handles missing coordinates by preserving original rows.
    """
    # Copy samples to preserve order and index
    samples = samples.reset_index(drop=True).copy()

    # Identify valid coordinates
    valid_mask = samples[['latitude_deg', 'longitude_deg']].notnull().all(axis=1)
    valid_samples = samples[valid_mask]
    invalid_samples = samples[~valid_mask]
    logger.info(len(valid_samples))
    logger.info(len(invalid_samples))

    # If no valid samples, return all unmatched
    if valid_samples.empty:
        matches = pd.DataFrame([
            {
                **{col: np.nan for col in facilities.columns}, 
                'facility_distance_km': np.nan, 
                'facility_match': False
            }
            for _ in range(len(samples))
        ])
        # Rename facility coordinate columns before returning
        matches = matches.rename(columns={
            'latitude_deg': 'facility_latitude_deg',
            'longitude_deg': 'facility_longitude_deg'
        })
        return pd.concat([samples, matches], axis=1)

    # Prepare facility KD-tree
    valid_fac = facilities.dropna(subset=['latitude_deg', 'longitude_deg']).reset_index(drop=True)
    fac_xyz = sph2cart(valid_fac['latitude_deg'], valid_fac['longitude_deg'])
    tree = cKDTree(fac_xyz)

    # Build sample coordinates
    samp_xyz = sph2cart(valid_samples['latitude_deg'], valid_samples['longitude_deg'])
    dists, idxs = tree.query(samp_xyz, distance_upper_bound=max_distance_km)

    # Build result records
    records = []
    # First handle valid_samples
    for dist, idx in zip(dists, idxs):
        if np.isfinite(dist):
            rec = valid_fac.iloc[idx].to_dict()
            rec.update({'facility_distance_km': dist, 'facility_match': True})
        else:
            rec = {col: np.nan for col in facilities.columns}
            rec.update({'facility_distance_km': np.nan, 'facility_match': False})
        records.append(rec)
    logger.info(len(records))
    # Then handle invalid_samples: no match
    for _ in range(len(invalid_samples)):
        rec = {col: np.nan for col in facilities.columns}
        rec.update({'facility_distance_km': np.nan, 'facility_match': False})
        records.append(rec)
    logger.info(len(records))
    # Combine matches in original order
    matches_df = pd.DataFrame(records)
    
    # Rename facility coordinate columns to avoid conflicts
    matches_df = matches_df.rename(columns={
        'latitude_deg': 'facility_latitude_deg',
        'longitude_deg': 'facility_longitude_deg',
        'country': 'facility_country'
    })
    for col in matches_df:
        samples[col] = col
    return samples

def load_nfc_facilities(cfg: Dict, output_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    databases = cfg.get("nfc_facilities", {}).get("databases", [{'name': "NFCIS"}, {'name': "GEM"}])
    use_local = cfg.get("nfc_facilities", {}).get('use_local', False)
    if output_dir:
        tsv_path = Path(output_dir) / 'nfc_facilities.csv'
    if use_local and tsv_path.exists():
        facilities_df = pd.read_csv(tsv_path, sep='\t')
    else:
        dfs = []
        for db in databases:
            path = constants.DEFAULT_NFCIS_PATH if db['name']=="NFCIS" else constants.DEFAULT_GEM_PATH
            dfs.append(process_and_geocode_db(database=db['name'], file_path=path))
        facilities_df = pd.concat(dfs, ignore_index=True).dropna(subset=['latitude_deg', 'longitude_deg'])
        facilities_df.to_csv(tsv_path, sep='\t', index=True)
    logger.info(f"Merged facilities: {facilities_df.shape}")
    return facilities_df
    
def match_facilities_to_samples(
    cfg: Dict,
    meta: pd.DataFrame,
    facilities_df: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    max_dist = cfg.get("nfc_facilities", {}).get("max_distance_km", 50)
    # Pass full metadata to ensure coordinate columns are available
    matched_df = match_facilities_to_locations(facilities_df, meta, max_distance_km=max_dist)
    
    # Define required metadata columns to keep
    required_meta_cols = [
        'nuclear_contamination_status', 
        'dataset_name', 
        'country', 
        'latitude_deg', 
        'longitude_deg'
    ]
    
    # Get new facility columns (including renamed coordinates)
    new_cols = [col for col in matched_df.columns if col not in meta.columns]
    
    # Combine required metadata and new facility columns
    result_cols = [col for col in required_meta_cols if col in matched_df] + new_cols
    
    # Log warning if any required columns are missing
    missing_cols = set(required_meta_cols) - set(result_cols)
    if missing_cols:
        logger.warning(f"Missing required columns in output: {', '.join(missing_cols)}")
    # Save full matched results
    matched_df[result_cols].to_csv(f"/usr2/people/macgregor/amplicon/test/facility_matches_{max_dist}km.tsv",
                      sep='\t', index=False)
    return matched_df

    
def find_nearby_nfc_facilities(
    cfg: Dict,
    meta: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load facility databases, geocode, merge, and match to sample metadata.
    Returns DataFrame with specific metadata columns + facility details.
    """
    databases = cfg.get("nfc_facilities", {}).get("databases", [{'name': "NFCIS"}, {'name': "GEM"}])
    use_local = cfg.get("nfc_facilities", {}).get('use_local', False)
    if output_dir:
        tsv_path = Path(output_dir) / 'nfc_facilities.csv'
    if use_local and tsv_path.exists():
        facilities_df = pd.read_csv(tsv_path, sep='\t')
    else:
        dfs = []
        for db in databases:
            path = constants.DEFAULT_NFCIS_PATH if db['name']=="NFCIS" else constants.DEFAULT_GEM_PATH
            dfs.append(process_and_geocode_db(database=db['name'], file_path=path))
        facilities_df = pd.concat(dfs, ignore_index=True).dropna(subset=['latitude_deg', 'longitude_deg'])
        facilities_df.to_csv(tsv_path, sep='\t', index=True)
    logger.info(f"Merged facilities: {facilities_df.shape}")

    max_dist = cfg.get("nfc_facilities", {}).get("max_distance_km", 50)
    # Pass full metadata to ensure coordinate columns are available
    matched_df = match_facilities_to_locations(facilities_df, meta, max_distance_km=max_dist)
    
    # Define required metadata columns to keep
    required_meta_cols = [
        'nuclear_contamination_status', 
        'dataset_name', 
        'country', 
        'latitude_deg', 
        'longitude_deg'
    ]
    
    # Get new facility columns (including renamed coordinates)
    new_cols = [col for col in matched_df.columns if col not in meta.columns]
    
    # Combine required metadata and new facility columns
    result_cols = [col for col in required_meta_cols if col in matched_df] + new_cols
    
    # Log warning if any required columns are missing
    missing_cols = set(required_meta_cols) - set(result_cols)
    if missing_cols:
        logger.warning(f"Missing required columns in output: {', '.join(missing_cols)}")
    # Save full matched results
    matched_df[result_cols].to_csv(f"/usr2/people/macgregor/amplicon/test/facility_matches_{max_dist}km.tsv",
                      sep='\t', index=False)
    return matched_df, facilities_df, matched_df[result_cols]


def analyze_contamination_correlation(
    df: pd.DataFrame,
    threshold: float = 0.5
) -> dict:
    """
    Analyzes correlation between facility proximity and contamination status.
    """
    required = ['facility_match', 'nuclear_contamination_status']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    analysis_df = df.set_index('#sampleid')[required].dropna().copy()
    # Contamination boolean
    if pd.api.types.is_numeric_dtype(analysis_df['nuclear_contamination_status']):
        analysis_df['contaminated'] = analysis_df['nuclear_contamination_status'] > threshold
    else:
        analysis_df['contaminated'] = (
            analysis_df['nuclear_contamination_status'].str.lower()
            .isin(['contaminated','positive','high','yes','true'])
        )

    facility_nearby = analysis_df['facility_match'].astype(bool)
    is_contaminated = analysis_df['contaminated']

    tn, fp, fn, tp = confusion_matrix(is_contaminated, facility_nearby, labels=[False, True]).ravel()

    total = len(analysis_df)
    summary = {
        'total_locations': total,
        'contamination_rate': is_contaminated.mean(),
        'facility_presence_rate': facility_nearby.mean(),
        'true_positive_rate': tp / (tp + fn) if tp+fn>0 else 0,
        'false_positive_rate': fp / (fp + tn) if fp+tn>0 else 0,
        'precision': tp / (tp + fp) if tp+fp>0 else 0,
        'relative_risk': (tp/(tp+fp)) / (fn/(fn+tn)) if fn+tn>0 else float('nan')
    }

    return {
        'summary_metrics': summary,
        'confusion_matrix': {'true_positive': tp, 'false_positive': fp, 'true_negative': tn, 'false_negative': fn},
        'contingency_table': pd.crosstab(
            facility_nearby, is_contaminated,
            rownames=['Facility Nearby'], colnames=['Contaminated'], margins=True
        ).to_dict(),
        'classification_report': classification_report(
            is_contaminated, facility_nearby,
            target_names=['Not Contaminated','Contaminated'], output_dict=True
        )
    }
