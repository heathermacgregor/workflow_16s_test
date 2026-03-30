"""
USGS Mineral Resources Data System (MRDS) API Handler

Provides mineral deposit locations, commodities, deposit types, and production history.

Data includes:
- Mine/prospect/occurrence locations (lat/lon)
- Commodities present (Au, Ag, Cu, Pb, Zn, U, etc.)
- Deposit type (porphyry copper, sediment-hosted Zn-Pb, etc.)
- Production history and grade
- Quality ranking (A–E, with A being most comprehensive)

API: https://mrdata.usgs.gov/catalog/api.php
"""

import logging
from typing import Dict, Any, Tuple
import requests
from .base import BaseEnvironmentalAPI

logger = logging.getLogger(__name__)


class USGSMRDSMinesAPI(BaseEnvironmentalAPI):
    """
    Query USGS Mineral Resources Data System for mineral deposits near sample locations.
    
    Returns information about mines, prospects, and mineral occurrences including:
    - Distance to nearest deposit
    - Number of deposits in search radius
    - Dominant commodities (Au, Ag, Cu, Pb, Zn, U, etc.)
    - Deposit types
    """
    
    API_NAME = "USGS_MRDS_Mines"
    BASE_URL = "https://mrdata.usgs.gov/mrds/search-bbox.php"
    
    def __init__(self, verbose: bool = False):
        """
        Initialize USGS MRDS API handler.
        
        Args:
            verbose: Enable verbose logging
        """
        super().__init__(verbose=verbose)
        self.timeout = 15
    
    def check_requirements(self) -> Tuple[bool, str]:
        """
        Check if USGS MRDS API is accessible.
        
        Returns:
            Tuple of (is_available, message)
        """
        try:
            # Test endpoint with a small query (global bbox)
            params = {
                'west': -180,
                'east': 180,
                'south': -90,
                'north': 90,
                'format': 'json'
            }
            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=self.timeout
            )
            if response.status_code == 200:
                if self.verbose:
                    logger.debug("USGS MRDS API is accessible")
                return True, "USGS MRDS API available"
            else:
                return False, f"USGS MRDS API returned status {response.status_code}"
        except Exception as e:
            return False, f"USGS MRDS API error: {str(e)}"
    
    def _fetch_data(self, lat: float, lon: float, radius_km: float = 20) -> Dict[str, Any]:
        """
        Fetch mineral deposit data for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in kilometers (default 20 km)
        
        Returns:
            Dictionary with mineral deposit information:
            - deposits_count: Number of deposits in area
            - nearest_deposit_distance_km: Distance to nearest deposit
            - commodities: List of commodities found
            - deposit_types: List of deposit types
            - deposits_list: Detailed list of deposits
        """
        try:
            # Convert km to degrees (rough: 1° ≈ 111 km)
            delta = radius_km / 111.0
            
            params = {
                'west': lon - delta,
                'east': lon + delta,
                'south': lat - delta,
                'north': lat + delta,
                'format': 'json'
            }
            
            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            if not data or 'features' not in data:
                return {
                    'deposits_count': 0,
                    'nearest_deposit_distance_km': None,
                    'commodities': [],
                    'deposit_types': [],
                    'deposits_list': []
                }
            
            deposits = data.get('features', [])
            
            # Extract commodities and deposit types
            commodities = set()
            deposit_types = set()
            distances = []
            
            for deposit in deposits:
                props = deposit.get('properties', {})
                
                # Collect commodities
                if 'commodity' in props and props['commodity']:
                    commodities.add(props['commodity'])
                
                # Collect deposit types
                if 'dep_type' in props and props['dep_type']:
                    deposit_types.add(props['dep_type'])
                
                # Calculate distance
                if 'geometry' in deposit and deposit['geometry'].get('coordinates'):
                    coords = deposit['geometry']['coordinates']
                    if len(coords) >= 2:
                        dep_lon, dep_lat = coords[0], coords[1]
                        # Simple distance approximation (haversine would be more accurate)
                        dist = ((dep_lat - lat)**2 + (dep_lon - lon)**2)**0.5 * 111
                        distances.append(dist)
            
            result = {
                'deposits_count': len(deposits),
                'nearest_deposit_distance_km': min(distances) if distances else None,
                'commodities': sorted(list(commodities)),
                'deposit_types': sorted(list(deposit_types)),
                'deposits_list': [
                    {
                        'name': d.get('properties', {}).get('site_name', 'Unknown'),
                        'commodity': d.get('properties', {}).get('commodity', ''),
                        'deposit_type': d.get('properties', {}).get('dep_type', ''),
                        'quality_rank': d.get('properties', {}).get('quality_flag', '')
                    }
                    for d in deposits[:10]  # Top 10 deposits
                ]
            }
            
            if self.verbose:
                logger.debug(f"MRDS: Found {result['deposits_count']} deposits, "
                            f"nearest at {result['nearest_deposit_distance_km']:.1f} km")
            
            return result
        
        except Exception as e:
            logger.error(f"Error fetching USGS MRDS data for ({lat}, {lon}): {str(e)}")
            return {
                'deposits_count': 0,
                'nearest_deposit_distance_km': None,
                'commodities': [],
                'deposit_types': [],
                'deposits_list': []
            }
    
    def get_data(self, lat: float, lon: float, **kwargs) -> Dict[str, Any]:
        """
        Get mineral deposit data (interface method for BaseEnvironmentalAPI).
        """
        return self._fetch_data(lat, lon, **kwargs)
    
    def fetch_and_enrich(self, df, lat_col: str, lon_col: str, 
                         sample_id_col: str = None, radius_km: float = 20):
        """
        Enrich dataframe with USGS MRDS mineral deposit data.
        
        Adds columns:
        - usgs_mrds_deposits_count
        - usgs_mrds_nearest_distance_km
        - usgs_mrds_commodities
        - usgs_mrds_deposit_types
        """
        import pandas as pd
        
        if not self.check_requirements()[0]:
            logger.warning("USGS MRDS API not available, skipping enrichment")
            return df
        
        results = []
        
        for idx, row in df.iterrows():
            try:
                lat = row[lat_col]
                lon = row[lon_col]
                
                if pd.isna(lat) or pd.isna(lon):
                    results.append({
                        'usgs_mrds_deposits_count': None,
                        'usgs_mrds_nearest_distance_km': None,
                        'usgs_mrds_commodities': None,
                        'usgs_mrds_deposit_types': None
                    })
                    continue
                
                data = self._fetch_data(lat, lon, radius_km)
                
                results.append({
                    'usgs_mrds_deposits_count': data['deposits_count'],
                    'usgs_mrds_nearest_distance_km': data['nearest_deposit_distance_km'],
                    'usgs_mrds_commodities': ','.join(data['commodities']) if data['commodities'] else None,
                    'usgs_mrds_deposit_types': ','.join(data['deposit_types']) if data['deposit_types'] else None
                })
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                results.append({
                    'usgs_mrds_deposits_count': None,
                    'usgs_mrds_nearest_distance_km': None,
                    'usgs_mrds_commodities': None,
                    'usgs_mrds_deposit_types': None
                })
        
        result_df = pd.DataFrame(results)
        return pd.concat([df, result_df], axis=1)
