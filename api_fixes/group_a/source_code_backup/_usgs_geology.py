"""
USGS Geologic Map Unit Data Handler

Provides geologic bedrock/parent material information at point locations.
Essential for understanding soil formation and background metal levels.

API: https://mrdata.usgs.gov/geology/state/point-unit.php
"""

import json
import logging
from typing import Dict, Any, Tuple
import requests
from .base import BaseEnvironmentalAPI

logger = logging.getLogger(__name__)


class USGSGeologicUnitsAPI(BaseEnvironmentalAPI):
    """
    Query USGS Geologic Map Unit data for bedrock and parent material information.
    
    Returns:
    - Rock type (igneous, sedimentary, metamorphic)
    - Geologic age
    - Parent material for soil formation
    - Potential indicator of background metal levels
    """
    
    API_NAME = "USGS_Geologic_Units"
    BASE_URL = "https://mrdata.usgs.gov/geology/state/point-unit.php"
    
    def __init__(self, verbose: bool = False):
        """
        Initialize USGS Geologic Units API handler.
        
        Args:
            verbose: Enable verbose logging
        """
        super().__init__(verbose=verbose)
        self.timeout = 15
    
    def check_requirements(self) -> Tuple[bool, str]:
        """
        Check if USGS Geologic Units API is accessible.
        
        Returns:
            Tuple of (is_available, message)
        """
        try:
            # Test with a US location
            params = {
                'latitude': 40.0,
                'longitude': -100.0,
                'format': 'json'
            }
            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=self.timeout
            )
            if response.status_code == 200:
                if self.verbose:
                    logger.debug("USGS Geologic Units API is accessible")
                return True, "USGS Geologic Units API available"
            else:
                return False, f"USGS Geologic Units API returned status {response.status_code}"
        except Exception as e:
            return False, f"USGS Geologic Units API error: {str(e)}"
    
    def _fetch_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch geologic unit data for a point location.
        
        Returns empty dict if location is outside US or data unavailable.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Dictionary with geologic information
        """
        try:
            params = {
                'latitude': lat,
                'longitude': lon,
                'format': 'json'
            }
            
            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Check if response body is empty before parsing JSON
            if not response.text or len(response.text.strip()) == 0:
                if self.verbose:
                    logger.debug(f"USGS Geology API returned empty response for ({lat}, {lon})")
                return {
                    'unit_found': False,
                    'unit_name': None,
                    'unit_age': None,
                    'unit_type': None,
                    'description': None
                }
            
            # Parse JSON with explicit error handling
            try:
                data = response.json()
            except (json.JSONDecodeError, ValueError) as je:
                # Response.text is not valid JSON
                logger.debug(f"USGS Geology API returned invalid JSON for ({lat}, {lon}): {str(je)}")
                if self.verbose:
                    logger.debug(f"Response text: {response.text[:100]}")  # Log first 100 chars
                return {
                    'unit_found': False,
                    'unit_name': None,
                    'unit_age': None,
                    'unit_type': None,
                    'description': None
                }
            
            if not data or 'type' not in data:
                return {
                    'unit_found': False,
                    'unit_name': None,
                    'unit_age': None,
                    'unit_type': None,
                    'description': None
                }
            
            features = data.get('features', [])
            
            if not features:
                return {
                    'unit_found': False,
                    'unit_name': None,
                    'unit_age': None,
                    'unit_type': None,
                    'description': None
                }
            
            # Get first (closest) feature
            unit = features[0]
            props = unit.get('properties', {})
            
            # Extract unit name and properties
            unit_name = props.get('unit_name', '')
            unit_age = props.get('age', '')
            unit_type = props.get('type', '')
            description = props.get('description', '')
            
            # Classify rock type
            rock_types = []
            unit_lower = unit_name.lower() + ' ' + description.lower()
            
            if any(x in unit_lower for x in ['granite', 'igneous', 'basalt', 'rhyolite']):
                rock_types.append('igneous')
            if any(x in unit_lower for x in ['sandstone', 'shale', 'limestone', 'sediment']):
                rock_types.append('sedimentary')
            if any(x in unit_lower for x in ['schist', 'gneiss', 'metamorph']):
                rock_types.append('metamorphic')
            
            result = {
                'unit_found': True,
                'unit_name': unit_name,
                'unit_age': unit_age,
                'unit_type': unit_type,
                'description': description,
                'rock_types': rock_types,
                'rock_type_primary': rock_types[0] if rock_types else None
            }
            
            if self.verbose:
                logger.debug(f"Geologic: Found unit '{unit_name}' ({unit_age})")
            
            return result
        
        except Exception as e:
            logger.warning(f"Error fetching geologic data for ({lat}, {lon}): {str(e)}")
            return {
                'unit_found': False,
                'unit_name': None,
                'unit_age': None,
                'unit_type': None,
                'description': None
            }
    
    def get_data(self, lat: float, lon: float, **kwargs) -> Dict[str, Any]:
        """
        Get geologic unit data (interface method for BaseEnvironmentalAPI).
        """
        return self._fetch_data(lat, lon, **kwargs)
    
    def fetch_and_enrich(self, df, lat_col: str, lon_col: str, 
                         sample_id_col: str = None):
        """
        Enrich dataframe with USGS Geologic Units data.
        
        Adds columns:
        - usgs_geologic_unit_name
        - usgs_geologic_age
        - usgs_geologic_type
        - usgs_rock_type_primary
        """
        import pandas as pd
        
        if not self.check_requirements()[0]:
            logger.warning("USGS Geologic Units API not available (US-only)")
            return df
        
        results = []
        
        for idx, row in df.iterrows():
            try:
                lat = row[lat_col]
                lon = row[lon_col]
                
                if pd.isna(lat) or pd.isna(lon):
                    results.append({
                        'usgs_geologic_unit_name': None,
                        'usgs_geologic_age': None,
                        'usgs_geologic_type': None,
                        'usgs_rock_type_primary': None
                    })
                    continue
                
                data = self._fetch_data(lat, lon)
                
                results.append({
                    'usgs_geologic_unit_name': data.get('unit_name'),
                    'usgs_geologic_age': data.get('unit_age'),
                    'usgs_geologic_type': data.get('unit_type'),
                    'usgs_rock_type_primary': data.get('rock_type_primary')
                })
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                results.append({
                    'usgs_geologic_unit_name': None,
                    'usgs_geologic_age': None,
                    'usgs_geologic_type': None,
                    'usgs_rock_type_primary': None
                })
        
        result_df = pd.DataFrame(results)
        return pd.concat([df, result_df], axis=1)
