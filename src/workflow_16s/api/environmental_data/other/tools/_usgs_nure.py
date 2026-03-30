"""
USGS NURE (National Uranium Resource Evaluation) Geochemistry API Handler

Provides stream sediment and water geochemistry data with 50+ elements:
- U, Th, Ag, Au, Cu, Pb, Zn, As, Cd, Cr, Ni, Fe, Mn, Co, etc.

APIs:
- Sediment: https://mrdata.usgs.gov/nure/sediment/search-bbox.php
- Water: https://mrdata.usgs.gov/nure/water/search-bbox.php
"""

import logging
from typing import Dict, Any, Tuple, List
import requests
from .base import BaseEnvironmentalAPI

logger = logging.getLogger(__name__)


class USGSNUREGeochemistryAPI(BaseEnvironmentalAPI):
    """
    Query USGS NURE geochemistry data for stream sediments and water.
    
    Variables (50+ elements):
    - U, Th, Ag, Au, Cu, Pb, Zn, As, Cd, Cr, Ni, Fe, Mn, Co, and many others
    
    Returns:
    - Number of samples in area
    - Mean, min, max concentrations for key metals (U, Pb, Zn, Cu, As)
    - Anomalous samples (>90th percentile)
    """
    
    API_NAME = "USGS_NURE_Geochemistry"
    SEDIMENT_URL = "https://mrdata.usgs.gov/nure/sediment/search-bbox.php"
    WATER_URL = "https://mrdata.usgs.gov/nure/water/search-bbox.php"
    
    # Key elements to track
    KEY_ELEMENTS = ['U', 'Pb', 'Zn', 'Cu', 'As', 'Ag', 'Cd', 'Cr', 'Ni']
    
    def __init__(self, verbose: bool = False, sample_type: str = 'sediment'):
        """
        Initialize USGS NURE API handler.
        
        Args:
            verbose: Enable verbose logging
            sample_type: 'sediment' (default) or 'water'
        """
        super().__init__(verbose=verbose)
        self.timeout = 15
        self.sample_type = sample_type
        self.url = self.SEDIMENT_URL if sample_type == 'sediment' else self.WATER_URL
    
    def check_requirements(self) -> Tuple[bool, str]:
        """
        Check if USGS NURE API is accessible.
        
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
                self.url,
                params=params,
                timeout=self.timeout
            )
            if response.status_code == 200:
                if self.verbose:
                    logger.debug(f"USGS NURE {self.sample_type} API is accessible")
                return True, f"USGS NURE {self.sample_type} API available"
            else:
                return False, f"USGS NURE API returned status {response.status_code}"
        except Exception as e:
            return False, f"USGS NURE API error: {str(e)}"
    
    def _fetch_data(self, lat: float, lon: float, radius_km: float = 20) -> Dict[str, Any]:
        """
        Fetch NURE geochemistry data for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in kilometers (default 20 km)
        
        Returns:
            Dictionary with geochemistry statistics
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
                self.url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            if not data or 'type' not in data:
                return {
                    'samples_count': 0,
                    'mean_concentrations': {},
                    'max_concentrations': {},
                    'anomalous_samples': 0
                }
            
            samples = data.get('features', [])
            
            # Initialize concentration dictionaries
            concentrations = {elem: [] for elem in self.KEY_ELEMENTS}
            
            # Extract concentrations
            for sample in samples:
                props = sample.get('properties', {})
                
                for elem in self.KEY_ELEMENTS:
                    # Try multiple naming conventions
                    for key in [elem, elem.lower(), f'{elem}_ppm', f'{elem}_ppb']:
                        if key in props and props[key] is not None:
                            try:
                                concentrations[elem].append(float(props[key]))
                            except (ValueError, TypeError):
                                pass
                            break
            
            # Calculate statistics
            mean_conc = {}
            max_conc = {}
            
            for elem, values in concentrations.items():
                if values:
                    mean_conc[elem] = sum(values) / len(values)
                    max_conc[elem] = max(values)
            
            # Count anomalous samples (simplified: samples with >2 anomalous elements)
            anomalous = 0
            for sample in samples:
                props = sample.get('properties', {})
                anomalous_count = 0
                for elem in self.KEY_ELEMENTS:
                    for key in [elem, elem.lower(), f'{elem}_ppm']:
                        if key in props and props[key] is not None:
                            try:
                                val = float(props[key])
                                if elem in max_conc and val > max_conc[elem] * 0.75:
                                    anomalous_count += 1
                            except (ValueError, TypeError):
                                pass
                            break
                if anomalous_count >= 2:
                    anomalous += 1
            
            result = {
                'samples_count': len(samples),
                'mean_concentrations': mean_conc,
                'max_concentrations': max_conc,
                'anomalous_samples': anomalous,
                'elements_detected': list(mean_conc.keys())
            }
            
            if self.verbose:
                logger.debug(f"NURE {self.sample_type}: Found {len(samples)} samples, "
                            f"{anomalous} anomalous, {len(mean_conc)} elements with data")
            
            return result
        
        except Exception as e:
            logger.error(f"Error fetching USGS NURE data: {str(e)}")
            return {
                'samples_count': 0,
                'mean_concentrations': {},
                'max_concentrations': {},
                'anomalous_samples': 0,
                'elements_detected': []
            }
    
    def get_data(self, lat: float, lon: float, **kwargs) -> Dict[str, Any]:
        """
        Get NURE geochemistry data (interface method for BaseEnvironmentalAPI).
        """
        return self._fetch_data(lat, lon, **kwargs)
    
    def fetch_and_enrich(self, df, lat_col: str, lon_col: str, 
                         sample_id_col: str = None, radius_km: float = 20):
        """
        Enrich dataframe with USGS NURE geochemistry data.
        
        Adds columns:
        - usgs_nure_sediment_samples (or _water_samples)
        - usgs_nure_mean_U, _mean_Pb, _mean_Zn, etc.
        - usgs_nure_anomalous_samples
        """
        import pandas as pd
        
        if not self.check_requirements()[0]:
            logger.warning(f"USGS NURE {self.sample_type} API not available")
            return df
        
        results = []
        
        for idx, row in df.iterrows():
            try:
                lat = row[lat_col]
                lon = row[lon_col]
                
                if pd.isna(lat) or pd.isna(lon):
                    result_row = {'usgs_nure_samples_count': None, 'usgs_nure_anomalous': None}
                    for elem in self.KEY_ELEMENTS:
                        result_row[f'usgs_nure_mean_{elem}'] = None
                    results.append(result_row)
                    continue
                
                data = self._fetch_data(lat, lon, radius_km)
                
                result_row = {
                    'usgs_nure_samples_count': data['samples_count'],
                    'usgs_nure_anomalous': data['anomalous_samples']
                }
                
                for elem in self.KEY_ELEMENTS:
                    result_row[f'usgs_nure_mean_{elem}'] = data['mean_concentrations'].get(elem)
                
                results.append(result_row)
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                result_row = {'usgs_nure_samples_count': None, 'usgs_nure_anomalous': None}
                for elem in self.KEY_ELEMENTS:
                    result_row[f'usgs_nure_mean_{elem}'] = None
                results.append(result_row)
        
        result_df = pd.DataFrame(results)
        return pd.concat([df, result_df], axis=1)
