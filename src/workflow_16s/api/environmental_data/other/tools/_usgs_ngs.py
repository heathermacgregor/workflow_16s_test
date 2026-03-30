"""
USGS National Geochemical Database (NGS) API Handler

Provides national-scale geochemical survey data for rock, sediment, and soil samples.

API: https://mrdata.usgs.gov/geochem/search-bbox.php
"""

import logging
from typing import Dict, Any, Tuple
import requests
from .base import BaseEnvironmentalAPI

logger = logging.getLogger(__name__)


class USGSNationalGeochemicalAPI(BaseEnvironmentalAPI):
    """
    Query USGS National Geochemical Database for soil/sediment geochemistry.
    
    Returns:
    - Number of samples in area
    - Elements detected (Cu, Pb, Zn, As, U, Th, etc.)
    - Mean concentrations for major elements
    """
    
    API_NAME = "USGS_NGS_Geochemical"
    BASE_URL = "https://mrdata.usgs.gov/geochem/search-bbox.php"
    
    # Key elements to track
    KEY_ELEMENTS = ['Cu', 'Pb', 'Zn', 'As', 'U', 'Th', 'Au', 'Ag', 'Fe', 'Mn']
    
    def __init__(self, verbose: bool = False):
        """
        Initialize USGS NGS API handler.
        
        Args:
            verbose: Enable verbose logging
        """
        super().__init__(verbose=verbose)
        self.timeout = 15
    
    def check_requirements(self) -> Tuple[bool, str]:
        """
        Check if USGS NGS API is accessible.
        
        Returns:
            Tuple of (is_available, message)
        """
        try:
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
                    logger.debug("USGS NGS API is accessible")
                return True, "USGS NGS API available"
            else:
                return False, f"USGS NGS API returned status {response.status_code}"
        except Exception as e:
            return False, f"USGS NGS API error: {str(e)}"
    
    def _fetch_data(self, lat: float, lon: float, radius_km: float = 20) -> Dict[str, Any]:
        """
        Fetch NGS geochemistry data for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in kilometers (default 20 km)
        
        Returns:
            Dictionary with geochemistry information
        """
        try:
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
            
            if not data or 'type' not in data:
                return {
                    'samples_count': 0,
                    'elements_detected': [],
                    'mean_concentrations': {}
                }
            
            samples = data.get('features', [])
            
            # Collect elements and concentrations
            all_elements = set()
            concentrations = {elem: [] for elem in self.KEY_ELEMENTS}
            
            for sample in samples:
                props = sample.get('properties', {})
                
                # Detect all available elements
                for key in props.keys():
                    if len(key) <= 3 and key.isupper():  # Simple heuristic for element symbols
                        all_elements.add(key)
                        if key in self.KEY_ELEMENTS and props[key] is not None:
                            try:
                                concentrations[key].append(float(props[key]))
                            except (ValueError, TypeError):
                                pass
            
            # Calculate statistics
            mean_conc = {}
            for elem, values in concentrations.items():
                if values:
                    mean_conc[elem] = sum(values) / len(values)
            
            result = {
                'samples_count': len(samples),
                'elements_detected': sorted(list(all_elements)),
                'mean_concentrations': mean_conc,
                'key_elements_found': list(mean_conc.keys())
            }
            
            if self.verbose:
                logger.debug(f"NGS: Found {len(samples)} samples, "
                            f"{len(all_elements)} elements detected, "
                            f"{len(mean_conc)} key elements with data")
            
            return result
        
        except Exception as e:
            logger.error(f"Error fetching USGS NGS data: {str(e)}")
            return {
                'samples_count': 0,
                'elements_detected': [],
                'mean_concentrations': {}
            }
    
    def get_data(self, lat: float, lon: float, **kwargs) -> Dict[str, Any]:
        """
        Get NGS geochemistry data (interface method for BaseEnvironmentalAPI).
        """
        return self._fetch_data(lat, lon, **kwargs)
    
    def fetch_and_enrich(self, df, lat_col: str, lon_col: str, 
                         sample_id_col: str = None, radius_km: float = 20):
        """
        Enrich dataframe with USGS NGS geochemistry data.
        
        Adds columns:
        - usgs_ngs_samples_count
        - usgs_ngs_elements_detected
        - usgs_ngs_mean_Cu, _mean_Pb, etc.
        """
        import pandas as pd
        
        if not self.check_requirements()[0]:
            logger.warning("USGS NGS API not available")
            return df
        
        results = []
        
        for idx, row in df.iterrows():
            try:
                lat = row[lat_col]
                lon = row[lon_col]
                
                if pd.isna(lat) or pd.isna(lon):
                    result_row = {
                        'usgs_ngs_samples_count': None,
                        'usgs_ngs_elements_detected': None
                    }
                    for elem in self.KEY_ELEMENTS:
                        result_row[f'usgs_ngs_mean_{elem}'] = None
                    results.append(result_row)
                    continue
                
                data = self._fetch_data(lat, lon, radius_km)
                
                result_row = {
                    'usgs_ngs_samples_count': data['samples_count'],
                    'usgs_ngs_elements_detected': len(data['elements_detected'])
                }
                
                for elem in self.KEY_ELEMENTS:
                    result_row[f'usgs_ngs_mean_{elem}'] = data['mean_concentrations'].get(elem)
                
                results.append(result_row)
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                result_row = {
                    'usgs_ngs_samples_count': None,
                    'usgs_ngs_elements_detected': None
                }
                for elem in self.KEY_ELEMENTS:
                    result_row[f'usgs_ngs_mean_{elem}'] = None
                results.append(result_row)
        
        result_df = pd.DataFrame(results)
        return pd.concat([df, result_df], axis=1)
