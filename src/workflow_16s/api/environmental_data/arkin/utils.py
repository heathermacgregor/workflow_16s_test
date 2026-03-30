# workflow_16s/api/environmental_data/arkin/utils.py

# Add env_agents to path
# This is often better handled by project structure (e.g., pip install -e .)
# but we will keep it for now.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from env_agents.adapters import CANONICAL_SERVICES # type: ignore
from env_agents.core.models import RequestSpec, Geometry # type: ignore

import hashlib
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from workflow_16s.utils.logger import with_logger
from .constants import SERVICE_CONFIG

@with_logger
def standardize_column_name(df: pd.DataFrame, target_name: str, alternatives: list) -> pd.DataFrame:
    """Finds a column from alternatives and renames it to a target name."""
    df_columns_lower = {col.lower(): col for col in df.columns}

    if target_name in df.columns:
        return df

    for alt in alternatives:
        if alt.lower() in df_columns_lower:
            original_col_name = df_columns_lower[alt.lower()]
            logger.info(f"Standardizing: Renaming '{original_col_name}' to '{target_name}'.")
            df.rename(columns={original_col_name: target_name}, inplace=True)
            return df

    raise KeyError(f"Required column '{target_name}' not found. Searched: {alternatives}")

def bbox_around_point(lat: float, lon: float, radius_km: float) -> Geometry:
    """Creates a bounding box Geometry around a point for Earth Engine."""
    lat_deg_per_km = 1 / 110.574
    lon_deg_per_km = 1 / (111.320 * math.cos(math.radians(lat)) + 1e-9)
    buffer_lat = radius_km * lat_deg_per_km
    buffer_lon = radius_km * lon_deg_per_km
    return Geometry(
        type='bbox', 
        coordinates=[lon - buffer_lon, lat - buffer_lat, lon + buffer_lon, lat + buffer_lat]
    )

@with_logger
def validate_services() -> Set[str]:
    """Checks prerequisites for Arkin/Google services."""
    available_services = set(SERVICE_CONFIG.keys())

    # --- Check for Google Earth Engine ---
    try:
        import ee
        # Use high-volume endpoint for server efficiency
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    except Exception as e:
        logger.warning(f"Disabling EARTH_ENGINE: GEE initialization failed: {e}.")
        available_services.discard("EARTH_ENGINE")

    return available_services

def fetch_service_data(
    service_name: str, geometry: Any, time_range: Tuple[str, str], cache_manager: Any,
    asset_info: Optional[Tuple[str, str]] = None,
    max_retries: int = 3
) -> List[Dict]:
    """
    Fetches data for a single Arkin/Google service with retry logic.
    Uses the unified SQLite cache to prevent redundant GEE queries.
    """
    #from env_agents.adapters import CANONICAL_SERVICES
    #from env_agents.core.models import RequestSpec

    cache_params = {
        "service": service_name,
        "geometry": vars(geometry) if hasattr(geometry, '__dict__') else geometry,
        "time_range": time_range,
        "asset_id": asset_info[0] if asset_info else None
    }
    
    # Generate cache key from parameters (MD5 hash of serialized params)
    param_str = str(sorted(cache_params.items()))
    cache_key = hashlib.md5(param_str.encode()).hexdigest()

    # 1. SQLite Check
    if (cached_result := cache_manager.get(cache_key)) is not None:
        return cached_result

    # 2. Network Fetch (with Retry Logic)
    for attempt in range(max_retries):
        try:
            adapter_class = CANONICAL_SERVICES[service_name]
            adapter = adapter_class(asset_id=asset_info[0]) if asset_info else adapter_class()
            
            spec = RequestSpec(geometry=geometry, time_range=time_range)
            result = adapter._fetch_rows(spec)

            if isinstance(result, pd.DataFrame):
                result = result.to_dict(orient='records')
            
            if result is not None:
                cache_manager.set(cache_key, result)
                return result
                
        except Exception as e:
            if attempt == max_retries - 1:
                return []
            time.sleep(2 ** attempt)
    
    return []