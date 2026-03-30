"""
EarthChem API Handler

Provides access to the EarthChem Library's geochemical sample database.
Contains global geochemical analyses including:
- Soil/sediment metal concentrations (Cu, Pb, Zn, Ni, Cr, As, Hg, etc.)
- Rock geochemistry
- Ore minerals from deposits
- Multi-element analytical data

Free and open access via EarthChem Portal REST API.
Reference: https://earthchem.org/portal

This replaces USGS MRDS (which was externally blocked) with a more comprehensive
global geochemical database including direct metal measurements.
"""

import logging
from typing import Dict, Any, Tuple, List, Optional
import requests
from pathlib import Path
import json
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EarthChemAPI:
    """
    Query EarthChem Library for geochemical data near sample locations.
    
    Features:
    - Global coverage of measured metal concentrations
    - Spatial search (radius) around coordinates
    - Direct measurements (not modeled/predicted)
    - Multi-element capability
    
    Returns:
    - Nearest sample distance
    - Available metal concentrations (Cu, Pb, Zn, Ni, Cr, As, Hg, Fe, Mn)
    - Sample count within radius
    """
    
    API_NAME = "EarthChem"
    
    # EarthChem Portal API endpoints
    PORTAL_URL = "https://earthchem.org/portal/api"
    
    # Common metals to query
    METALS = ['Cu', 'Pb', 'Zn', 'Ni', 'Cr', 'As', 'Hg', 'Fe', 'Mn', 'Cd', 'Co']
    
    def __init__(self, verbose: bool = False, search_radius_km: float = 50.0,
                 cache_dir: Optional[Path] = None):
        """
        Initialize EarthChem API client.
        
        Args:
            verbose: Enable verbose logging
            search_radius_km: Search radius for spatial queries (default 50 km)
            cache_dir: Optional cache directory for API responses
        """
        self.verbose = verbose
        self.search_radius_km = search_radius_km
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.timeout = 10
        self.max_retries = 2
    
    def check_requirements(self) -> Tuple[bool, str]:
        """
        Check if EarthChem API is accessible.
        
        Returns:
            Tuple of (is_available, message)
        """
        try:
            # Test API connectivity
            response = requests.get(f"{self.PORTAL_URL}/search", 
                                   params={'limit': 1},
                                   timeout=self.timeout)
            if response.status_code == 200:
                return True, "EarthChem API accessible"
            else:
                return False, f"EarthChem API returned {response.status_code}"
        except Exception as e:
            return False, f"EarthChem API error: {str(e)}"
    
    def _get_cache_key(self, lat: float, lon: float) -> str:
        """Generate cache key for coordinates."""
        return f"earthchem_{lat:.4f}_{lon:.4f}_{self.search_radius_km}km"
    
    def _load_cache(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Try to load cached result."""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{self._get_cache_key(lat, lon)}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    # Check if cache is fresh (< 30 days)
                    cached_time = datetime.fromisoformat(data.get('cached_at', ''))
                    if datetime.now() - cached_time < timedelta(days=30):
                        return data['result']
            except Exception as e:
                logger.debug(f"Cache load error: {e}")
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
            logger.debug(f"Cache save error: {e}")
    
    def fetch_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch EarthChem geochemical data near a location.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Dictionary with metal concentrations and metadata
        """
        try:
            # Check cache first
            cached = self._load_cache(lat, lon)
            if cached:
                logger.debug(f"EarthChem cache hit at ({lat}, {lon})")
                return cached
            
            # Build bounding box (rough conversion: 1 degree ≈ 111 km)
            radius_deg = self.search_radius_km / 111.0
            bbox = {
                'north': lat + radius_deg,
                'south': lat - radius_deg,
                'east': lon + radius_deg,
                'west': lon - radius_deg,
            }
            
            params = {
                'lattop': bbox['north'],
                'latbottom': bbox['south'],
                'lngright': bbox['east'],
                'lngleft': bbox['west'],
                'limit': 100,  # Get up to 100 nearest samples
                'format': 'json'
            }
            
            response = requests.get(f"{self.PORTAL_URL}/search",
                                   params=params,
                                   timeout=self.timeout)
            response.raise_for_status()
            
            results = response.json()
            
            # Process results - find metals and compute stats
            result_dict = {
                'available': True,
                'sample_count': 0,
                'nearest_distance_km': None,
                'data': {}
            }
            
            if not results or 'rows' not in results:
                result_dict['data'] = {metal: None for metal in self.METALS}
                self._save_cache(lat, lon, result_dict)
                return result_dict
            
            rows = results['rows']
            result_dict['sample_count'] = len(rows)
            
            if len(rows) > 0:
                # Extract metal concentrations from first (nearest) sample
                first_sample = rows[0]
                
                # Estimate distance to first sample (rough approximation)
                if 'lat' in first_sample and 'lng' in first_sample:
                    dlat = (first_sample['lat'] - lat) * 111.0
                    dlon = (first_sample['lng'] - lon) * 111.0 * abs(np.cos(np.radians(lat)))
                    distance = (dlat**2 + dlon**2)**0.5
                    result_dict['nearest_distance_km'] = distance
                
                # Try to extract metal data
                # EarthChem typically stores in 'elements' or individual fields
                for metal in self.METALS:
                    val = None
                    for key in [metal, f'{metal.lower()}', f'{metal.lower()}_ppm',
                               'elements', 'composition']:
                        if key in first_sample:
                            try:
                                if isinstance(first_sample[key], dict):
                                    # Nested element data
                                    val = first_sample[key].get('value') or first_sample[key].get('concentration')
                                else:
                                    val = float(first_sample[key])
                                if val is not None:
                                    break
                            except (ValueError, TypeError, KeyError):
                                pass
                    
                    result_dict['data'][metal] = val
            else:
                # No data in bounding box
                result_dict['data'] = {metal: None for metal in self.METALS}
            
            self._save_cache(lat, lon, result_dict)
            return result_dict
        
        except Exception as e:
            logger.warning(f"EarthChem fetch error at ({lat}, {lon}): {str(e)}")
            return {
                'available': False,
                'error': str(e),
                'data': {metal: None for metal in self.METALS}
            }
    
    def get_data(self, lat: float, lon: float, **kwargs) -> Dict[str, Any]:
        """Get geochemical data (interface method)."""
        return self.fetch_data(lat, lon)
    
    def fetch_and_enrich(self, df, lat_col: str, lon_col: str, 
                         sample_id_col: str = None):
        """
        Enrich dataframe with EarthChem metal concentrations.
        
        Adds columns: ec_Cu, ec_Pb, ec_Zn, ec_Ni, ec_Cr, ec_As, ec_Hg, ec_Fe, ec_Mn
        Also adds: ec_sample_count, ec_nearest_distance_km
        
        Args:
            df: Input dataframe with coordinates
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            sample_id_col: Optional sample ID column for logging
        
        Returns:
            Dataframe with added metal concentration columns
        """
        if not self.check_requirements()[0]:
            logger.warning("EarthChem API not available")
            return df
        
        results = []
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            try:
                lat = row[lat_col]
                lon = row[lon_col]
                
                result_row = {metal: None for metal in self.METALS}
                result_row['ec_sample_count'] = None
                result_row['ec_nearest_distance_km'] = None
                
                if pd.isna(lat) or pd.isna(lon):
                    results.append(result_row)
                    continue
                
                data = self.fetch_data(lat, lon)
                
                if data.get('available'):
                    for metal in self.METALS:
                        result_row[f'ec_{metal}'] = data['data'].get(metal)
                    result_row['ec_sample_count'] = data.get('sample_count')
                    result_row['ec_nearest_distance_km'] = data.get('nearest_distance_km')
                
                results.append(result_row)
                
                if self.verbose and (idx + 1) % 100 == 0:
                    logger.info(f"EarthChem: Processed {idx + 1}/{total_rows} rows")
            
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                result_row = {metal: None for metal in self.METALS}
                result_row['ec_sample_count'] = None
                result_row['ec_nearest_distance_km'] = None
                results.append(result_row)
        
        result_df = pd.DataFrame(results)
        return pd.concat([df, result_df], axis=1)


# Rename columns for output convention
def rename_earthchem_cols(df):
    """Standardize EarthChem column names."""
    mapping = {'ec_{}'.format(m): 'earthchem_{}'.format(m.lower()) 
              for m in EarthChemAPI.METALS}
    mapping['ec_sample_count'] = 'earthchem_sample_count'
    mapping['ec_nearest_distance_km'] = 'earthchem_distance_km'
    return df.rename(columns=mapping)
