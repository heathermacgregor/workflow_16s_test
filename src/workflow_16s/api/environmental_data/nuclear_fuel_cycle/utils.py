# workflow_16s/api/environmental_data/nuclear_fuel_cycle/utils.py

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from workflow_16s.utils.logger import get_logger

# 1. Geometry Helpers
def sph2cart(lats: np.ndarray, lons: np.ndarray, R: float = 6371.0) -> np.ndarray:
    """Converts spherical (lat/lon) coordinates to 3D Cartesian coordinates."""
    φ = np.radians(lats.astype(float))
    λ = np.radians(lons.astype(float))
    x = R * np.cos(φ) * np.cos(λ)
    y = R * np.cos(φ) * np.sin(λ)
    z = R * np.sin(φ)
    return np.column_stack((x, y, z))

# 2. Data Cleaning & Standardization
def clean_fetched_ena_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes coordinates and removes invalid GPS entries."""
    if df.empty: return df
    cleaned_df = df.copy()

    # Priority check for standardized sample ID
    if '#sampleid' in cleaned_df.columns:
        cleaned_df.drop_duplicates(subset=['#sampleid'], keep='first', inplace=True)

    # Ensure numeric types for coordinate math
    cleaned_df['lat'] = pd.to_numeric(cleaned_df.get('lat'), errors='coerce')
    cleaned_df['lon'] = pd.to_numeric(cleaned_df.get('lon'), errors='coerce')

    # Standardize coordinate columns by combining alternatives
    for alt in [c for c in cleaned_df.columns if 'lat' in c.lower() and c != 'lat']:
        cleaned_df['lat'] = cleaned_df['lat'].combine_first(pd.to_numeric(cleaned_df[alt], errors='coerce'))
    for alt in [c for c in cleaned_df.columns if 'lon' in c.lower() and c != 'lon']:
        cleaned_df['lon'] = cleaned_df['lon'].combine_first(pd.to_numeric(cleaned_df[alt], errors='coerce'))

    cleaned_df.dropna(subset=['lat', 'lon'], inplace=True)
    
    # Validation: Filter for realistic global bounds and remove 0,0 trap
    mask = (cleaned_df['lat'].between(-90, 90)) & (cleaned_df['lon'].between(-180, 180))
    cleaned_df = cleaned_df[mask & ~((cleaned_df['lat'] == 0) & (cleaned_df['lon'] == 0))]
    
    return cleaned_df.reset_index(drop=True)

def consolidate_and_merge_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Merges redundant prefixed columns (run_, biosample_, etc.) into base names."""
    df_copy = df.copy()
    priority_prefixes = ['run_', 'experiment_', 'biosample_', 'study_']
    
    # Identify unique base names
    base_names = {col.split('_', 1)[1] for col in df_copy.columns 
                  if any(col.startswith(p) for p in priority_prefixes)}

    for base_name in sorted(list(base_names)):
        if base_name == 'accession': continue
        
        # Build priority list of existing columns for this base name
        existing_cols = [f"{p}{base_name}" for p in priority_prefixes if f"{p}{base_name}" in df_copy.columns]
        if len(existing_cols) < 2: continue

        # Combine first across the priority list
        merged_series = df_copy[existing_cols[0]]
        for i in range(1, len(existing_cols)):
            merged_series = merged_series.combine_first(df_copy[existing_cols[i]])

        df_copy[base_name] = merged_series
        df_copy.drop(columns=existing_cols, inplace=True)

    return df_copy

# 3. Biological Context Inferences
def resolve_facility_type(name: str, raw_type: str, source: str, definitions: Dict, defaults: Dict) -> str:
    """Infers a standardized facility type using regex and source-based fallbacks."""
    f_name = str(name).lower()
    f_type = str(raw_type).lower()
    f_source = str(source).upper()

    # A. Check for Keyword Matches in Name or Type
    for std_name, patterns in definitions.items():
        for pat in patterns:
            if re.search(pat, f_type) or re.search(pat, f_name):
                return std_name

    # B. Fallback to Source Defaults
    if f_source in defaults:
        return defaults[f_source]

    # C. Preservation: return original if available, else unknown
    return raw_type if raw_type and str(raw_type).lower() != 'nan' else 'Unknown'