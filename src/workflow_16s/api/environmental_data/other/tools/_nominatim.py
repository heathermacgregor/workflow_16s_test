# workflow_16s/api/environmental_data/other/tools/_nominatim.py
"""
Nominatim API Handler - OpenStreetMap Reverse Geocoding

Provides geographic and administrative information at coordinates:
- Country, state, county, city, village
- Land use classification
- Address details
- OSM metadata

No API key required. Data is CC0 licensed (public domain).
Supports ~1 request/second rate limit.
"""

import requests
import pandas as pd
from typing import Dict, Any, List, Tuple
import time
from .base import BaseEnvironmentalAPI

class NominatimAPI(BaseEnvironmentalAPI):
    """Query OSM Nominatim for geographic information."""
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.base_url = "https://nominatim.openstreetmap.org/reverse"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        }
        self.rate_limit_delay = 1.1  # seconds between requests (OSM policy: 1 req/sec)
    
    def check_requirements(self) -> Tuple[bool, str]:
        """Check if Nominatim is accessible."""
        try:
            params = {'lat': 0, 'lon': 0, 'format': 'json'}
            response = requests.get(self.base_url, params=params, timeout=10, 
                                   headers=self.headers)
            if response.status_code == 200:
                return (True, "OK")
            else:
                return (False, f"HTTP {response.status_code}")
        except Exception as e:
            return (False, str(e)[:50])
    
    def get_data(self, lat: float, lon: float, **kwargs) -> Dict[str, Any]:
        """
        Retrieve geographic and administrative data for given coordinates.
        
        Implements the abstract method from BaseEnvironmentalAPI.
        Calls _fetch_data() to perform reverse geocoding via Nominatim API.
        
        Args:
            lat (float): Latitude coordinate
            lon (float): Longitude coordinate
            **kwargs: Additional keyword arguments (passed to _fetch_data)
        
        Returns:
            Dict[str, Any]: Dictionary containing location information (country, state, 
                          county, city, village, landuse, etc.) or error message
        """
        return self._fetch_data(lat, lon, **kwargs)
    
    def _fetch_data(self, lat: float, lon: float, **kwargs) -> Dict[str, Any]:
        """Reverse geocode coordinates to get location info."""
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json',
            'zoom': 18
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10,
                                   headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                address = data.get('address', {})
                
                return {
                    'country': address.get('country', 'Unknown'),
                    'state': address.get('state', ''),
                    'county': address.get('county', ''),
                    'city': address.get('city', 'address.get("town", ""), city/town...'),
                    'village': address.get('village', address.get('town', '')),
                    'landuse': address.get('landuse', ''),
                    'leisure': address.get('leisure', ''),
                    'natural': address.get('natural', ''),
                    'postcode': address.get('postcode', ''),
                    'osm_type': data.get('osm_type', ''),
                    'importance': data.get('importance', 0)
                }
            else:
                return {'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'error': str(e)[:100]}
    
    def fetch_and_enrich(self, adata, metadata_cols: List[str] = None) -> pd.DataFrame:
        """Reverse geocode all samples."""
        if metadata_cols is None:
            metadata_cols = ['latitude', 'longitude']
        
        results = []
        
        for idx, row in adata.iterrows():
            try:
                lat = float(row.get('latitude', row.get('lat', None)))
                lon = float(row.get('longitude', row.get('lon', None)))
                
                if pd.isna(lat) or pd.isna(lon):
                    results.append({})
                    continue
                
                data = self._fetch_data(lat, lon)
                
                # Convert to flat columns
                row_data = {
                    f'nominatim_{k}': v for k, v in data.items()
                }
                results.append(row_data)
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error processing {idx}: {e}")
                results.append({})
        
        return pd.DataFrame(results)
