"""
Earth MRI Handler (Mineral Resources from Remote Imagery)

Provides access to Earth MRI geochemical dataset for US samples.
Earth MRI (2025) provides measured geochemical data from USGS including:
- Soil/sediment geochemical compositions
- Trace elements
- Rare earth elements
- Available nation-wide (2025 release)

This is an improved alternative to USGS NURE (which often returns HTTP 400)
with better coverage, newer data, and documented API.

Reference: https://www.usgs.gov/centers/vbi/earth-mri
"""

import logging
from typing import Dict, Any, Tuple, List, Optional
import requests
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .base import BaseEnvironmentalAPI
from workflow_16s.utils.logger import get_logger

logger = get_logger(__name__)


class EarthMRIAPI(BaseEnvironmentalAPI):
    """
    Query Earth MRI geochemical data for US sample locations.
    
    Features:
    - US-wide coverage
    - Measured soil/sediment compositions
    - Trace elements and rare earths
    - 2025 release data
    - Spatial search (nearest neighbor)
    
    Returns:
    - Nearest sample distance
    - Available trace and rare earth elements
    - Sample metadata (location, date)
    """
    
    API_NAME = "Earth_MRI"
    
    # Earth MRI API endpoints
    API_BASE = "https://mrdata.usgs.gov/api/rest"
    
    # Key elements in Earth MRI
    ELEMENTS = ['Al', 'Ca', 'Fe', 'K', 'Mg', 'Mn', 'Na', 'P', 'Si', 'Ti',  # Major
                'As', 'Cd', 'Cr', 'Cu', 'Hg', 'Mo', 'Ni', 'Pb', 'Sb', 'Zn',  # Trace
                'La', 'Ce', 'Nd', 'Yb']  # Selected rare earths
    
    def __init__(self, verbose: bool = False, search_radius_km: float = 25.0,
                 cache_dir: Optional[Path] = None):
        """
        Initialize Earth MRI API client.
        
        Args:
            verbose: Enable verbose logging
            search_radius_km: Search radius for spatial queries (default 25 km)
            cache_dir: Optional cache directory for API responses
        """
        super().__init__(verbose=verbose)
        self.search_radius_km = search_radius_km
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.timeout = 10
        self.us_only = True  # Earth MRI is US-specific
    
    def check_requirements(self) -> Tuple[bool, str]:
        """
        Check if Earth MRI API is accessible.
        
        Returns:
            Tuple of (is_available, message)
        """
        try:
            # Test API connectivity with a known sample
            # Point in standard US test area
            response = requests.get(f"{self.API_BASE}/sample/",
                                   params={'lat': 40.0, 'lng': -105.0, 'limit': 1},
                                   timeout=self.timeout)
            if response.status_code == 200:
                return True, "Earth MRI API accessible"
            else:
                # 400 might mean no data at location, not API error
                if response.status_code == 400:
                    return True, "Earth MRI API accessible (no data at test location)"
                return False, f"Earth MRI API returned {response.status_code}"
        except Exception as e:
            return False, f"Earth MRI API error: {str(e)}"
    
    def _is_us_location(self, lat: float, lon: float) -> bool:
        """Check if location is in continental US."""
        # Continental US bounds (rough)
        return 24 <= lat <= 50 and -125 <= lon <= -65
    
    def _get_cache_key(self, lat: float, lon: float) -> str:
        """Generate cache key for coordinates."""
        return f"earthmri_{lat:.4f}_{lon:.4f}_{self.search_radius_km}km"
    
    def _load_cache(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Try to load cached result."""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{self._get_cache_key(lat, lon)}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    cached_time = datetime.fromisoformat(data.get('cached_at', ''))
                    if datetime.now() - cached_time < timedelta(days=30):
                        return data['result']
            except Exception as e:
                self.logger.debug(f"Cache load error: {e}")
        return None
    
    def _save_cache(self, lat: float, lon: float, result: Dict[str, Any]):
        """Save result to cache."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{self._get_cache_key(lat, lon)}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'cached_at': datetime.now().isoformat(),
                    'result': result
                }, f)
        except Exception as e:
            self.logger.debug(f"Cache save error: {e}")
    
    def fetch_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch Earth MRI geochemical data for a US location.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Dictionary with trace element concentrations and metadata
        """
        try:
            if not self._is_us_location(lat, lon):
                return {
                    'available': False,
                    'error': 'Earth MRI is US-only (continental bounds)',
                    'data': {elem: None for elem in self.ELEMENTS}
                }
            
            # Check cache first
            cached = self._load_cache(lat, lon)
            if cached:
                logger.debug(f"Earth MRI cache hit at ({lat}, {lon})")
                return cached
            
            # Convert radius to degrees (rough: 1 deg ≈ 111 km at equator)
            radius_deg = self.search_radius_km / 111.0
            
            params = {
                'lat': lat,
                'lng': lon,
                'radius_km': self.search_radius_km,
                'limit': 50,
                'format': 'json'
            }
            
            response = requests.get(f"{self.API_BASE}/sample/nearby",
                                   params=params,
                                   timeout=self.timeout)
            response.raise_for_status()
            
            results = response.json()
            
            # Process results
            result_dict = {
                'available': True,
                'sample_count': 0,
                'nearest_distance_km': None,
                'data': {elem: None for elem in self.ELEMENTS}
            }
            
            if not results or 'samples' not in results:
                self._save_cache(lat, lon, result_dict)
                return result_dict
            
            samples = results['samples']
            result_dict['sample_count'] = len(samples)
            
            if len(samples) > 0:
                # Get nearest sample
                nearest = samples[0]
                
                # Distance calculation
                if 'latitude' in nearest and 'longitude' in nearest:
                    dlat = (nearest['latitude'] - lat) * 111.0
                    dlon = (nearest['longitude'] - lon) * 111.0 * abs(np.cos(np.radians(lat)))
                    distance = (dlat**2 + dlon**2)**0.5
                    result_dict['nearest_distance_km'] = distance
                
                # Extract compositions/elements
                compositions = nearest.get('composition', {})
                if isinstance(compositions, list):
                    # Handle list of element entries
                    for elem_entry in compositions:
                        elem = elem_entry.get('element', '')
                        value = elem_entry.get('value')
                        if elem in self.ELEMENTS and value:
                            try:
                                result_dict['data'][elem] = float(value)
                            except (ValueError, TypeError):
                                pass
                elif isinstance(compositions, dict):
                    # Handle dict format
                    for elem in self.ELEMENTS:
                        val = compositions.get(elem) or compositions.get(elem.lower())
                        if val:
                            try:
                                result_dict['data'][elem] = float(val)
                            except (ValueError, TypeError):
                                pass
            
            self._save_cache(lat, lon, result_dict)
            return result_dict
        
        except Exception as e:
            self.logger.warning(f"Earth MRI fetch error at ({lat}, {lon}): {str(e)}")
            return {
                'available': False,
                'error': str(e),
                'data': {elem: None for elem in self.ELEMENTS}
            }
    
    def get_data(self, lat: float, lon: float, **kwargs) -> Dict[str, Any]:
        """Get geochemical data (interface method)."""
        return self.fetch_data(lat, lon)
    
    def fetch_and_enrich(self, df, lat_col: str, lon_col: str, 
                         sample_id_col: str = None):
        """
        Enrich dataframe with Earth MRI trace element data.
        
        Adds columns: em_As, em_Cd, em_Cu, em_Hg, em_Pb, em_Zn, etc.
        Also adds: em_sample_count, em_nearest_distance_km
        
        US-only. Non-US coordinates will receive NaN values.
        
        Args:
            df: Input dataframe with coordinates
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            sample_id_col: Optional sample ID column for logging
        
        Returns:
            Dataframe with added trace element columns
        """
        if not self.check_requirements()[0]:
            self.logger.warning("Earth MRI API not available")
            return df
        
        results = []
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            try:
                lat = row[lat_col]
                lon = row[lon_col]
                
                result_row = {f'em_{elem}': None for elem in self.ELEMENTS}
                result_row['em_sample_count'] = None
                result_row['em_nearest_distance_km'] = None
                
                if pd.isna(lat) or pd.isna(lon):
                    results.append(result_row)
                    continue
                
                if not self._is_us_location(lat, lon):
                    # Mark as non-US and return None values
                    result_row['em_is_us'] = False
                    results.append(result_row)
                    continue
                
                result_row['em_is_us'] = True
                data = self.fetch_data(lat, lon)
                
                if data.get('available'):
                    for elem in self.ELEMENTS:
                        result_row[f'em_{elem}'] = data['data'].get(elem)
                    result_row['em_sample_count'] = data.get('sample_count')
                    result_row['em_nearest_distance_km'] = data.get('nearest_distance_km')
                
                results.append(result_row)
                
                if self.verbose and (idx + 1) % 100 == 0:
                    self.logger.info(f"Earth MRI: Processed {idx + 1}/{total_rows} rows")
            
            except Exception as e:
                self.logger.error(f"Error processing row {idx}: {str(e)}")
                result_row = {f'em_{elem}': None for elem in self.ELEMENTS}
                result_row['em_sample_count'] = None
                result_row['em_nearest_distance_km'] = None
                result_row['em_is_us'] = False
                results.append(result_row)
        
        result_df = pd.DataFrame(results)
        return pd.concat([df, result_df], axis=1)
