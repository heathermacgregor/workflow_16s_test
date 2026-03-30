# workflow_16s/api/environmental_data/other/tools/_overpass.py
"""
Overpass API Handler - OpenStreetMap Data

Provides access to detailed geographic features from OpenStreetMap:
- Water bodies (lakes, rivers, streams)
- Forest coverage and land use
- Building data
- Infrastructure (roads, power lines, etc.)

No API key required. Data is CC0 licensed (public domain).
"""

import requests
import pandas as pd
from typing import Dict, Any, List, Tuple
import time
from .base import BaseEnvironmentalAPI

class OverpassAPI(BaseEnvironmentalAPI):
    """Query OpenStreetMap Overpass API for geographic features."""
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.base_url = "https://overpass-api.de/api/interpreter"
        self.rate_limit_delay = 2.0  # seconds between requests
    
    def check_requirements(self) -> Tuple[bool, str]:
        """Check if Overpass API is accessible."""
        try:
            # Test with a simple, tiny query
            test_query = "[bbox:0,0,0.001,0.001];(node;);out;timeout:5;"
            response = requests.post(self.base_url, data=test_query, timeout=5)
            if response.status_code == 200:
                return (True, "OK")
            else:
                return (False, f"HTTP {response.status_code}")
        except Exception as e:
            return (False, str(e)[:50])
    
    def get_data(self, lat: float, lon: float, **kwargs) -> Dict[str, Any]:
        """Get Overpass API data for a single location.
        
        Implements the abstract method from BaseEnvironmentalAPI.
        Fetches OpenStreetMap features (water bodies, forest cover) near coordinates.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            **kwargs: Additional arguments (radius_km, timeout, etc.)
            
        Returns:
            Dict with feature data or error information
        """
        return self._fetch_data(lat, lon, **kwargs)
    
    def _fetch_data(self, lat: float, lon: float, **kwargs) -> Dict[str, Any]:
        """Fetch OSM features near coordinates."""
        radius_km = kwargs.get('radius_km', 5)
        delta = radius_km / 111.0  # Convert km to degrees
        
        features_data = {}
        
        # Query 1: Water bodies
        try:
            time.sleep(self.rate_limit_delay)
            query = f"""[bbox:{lat-delta},{lon-delta},{lat+delta},{lon+delta}];
(node["water"];way["water"];relation["water"];);
out geom;timeout:10;"""
            
            response = requests.post(self.base_url, data=query, timeout=15)
            if response.status_code == 200:
                features_data['water_bodies'] = {
                    'present': True,
                    'features_count': response.text.count('<way') + response.text.count('<node')
                }
        except Exception as e:
            features_data['water_bodies'] = {'error': str(e)[:50]}
        
        # Query 2: Forest
        try:
            time.sleep(self.rate_limit_delay)
            query = f"""[bbox:{lat-delta},{lon-delta},{lat+delta},{lon+delta}];
(way["landuse"="forest"];relation["landuse"="forest"];);
out geom;timeout:10;"""
            
            response = requests.post(self.base_url, data=query, timeout=15)
            if response.status_code == 200:
                features_data['forest_cover'] = {
                    'present': True,
                    'features_count': response.text.count('<way') + response.text.count('<relation')
                }
        except Exception as e:
            features_data['forest_cover'] = {'error': str(e)[:50]}
        
        return features_data
    
    def fetch_and_enrich(self, adata, metadata_cols: List[str] = None) -> pd.DataFrame:
        """Fetch OSM features for all samples."""
        if metadata_cols is None:
            metadata_cols = ['latitude', 'longitude']
        
        results = []
        
        for idx, row in adata.iterrows():
            try:
                lat = float(row.get('latitude', row.get('lat', None)))
                lon = float(row.get('longitude', row.get('lon', None)))
                
                if pd.isna(lat) or pd.isna(lon):
                    continue
                
                data = self._fetch_data(lat, lon, radius_km=5)
                
                # Flatten into columnar format
                row_data = {
                    'osm_water_features': data.get('water_bodies', {}).get('features_count', 0),
                    'osm_forest_features': data.get('forest_cover', {}).get('features_count', 0),
                }
                results.append(row_data)
            except Exception as e:
                if self.verbose:
                    print(f"Error processing {idx}: {e}")
                results.append({})
        
        return pd.DataFrame(results)
