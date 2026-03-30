# workflow_16s/api/environmental_data/other/tools/_gbif.py
"""
GBIF API Handler - Global Biodiversity Information Facility

Provides species occurrence data and biodiversity information:
- Species observations near coordinates
- Taxonomic information
- Occurrence density
- Data source attribution
- Observation photography/media

No API key required. Data licensed under CC BY 4.0 or specific licenses.
Supports ~100 requests/minute rate limit.
"""

import requests
import pandas as pd
from typing import Dict, Any, List, Tuple
from collections import Counter
from .base import BaseEnvironmentalAPI

class GBIFAPI(BaseEnvironmentalAPI):
    """Query GBIF for species observations and biodiversity data."""
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.base_url = "https://api.gbif.org/v1/occurrence/search"
        self.taxon_url = "https://api.gbif.org/v1/species/search"
    
    def check_requirements(self) -> Tuple[bool, str]:
        """Check if GBIF API is accessible."""
        try:
            response = requests.get(self.base_url, params={'limit': 1}, timeout=10)
            if response.status_code == 200:
                return (True, "OK")
            else:
                return (False, f"HTTP {response.status_code}")
        except Exception as e:
            return (False, str(e)[:50])
    
    def get_data(self, lat: float, lon: float, **kwargs) -> Dict[str, Any]:
        """
        Fetch GBIF biodiversity data at the given coordinates.
        
        Implements the abstract method from BaseEnvironmentalAPI.
        Retrieves species observations, biodiversity metrics, and taxonomic 
        information for the specified location.
        
        Args:
            lat (float): Latitude coordinate.
            lon (float): Longitude coordinate.
            **kwargs: Additional arguments passed to _fetch_data (e.g., radius_km).
        
        Returns:
            Dict[str, Any]: Dictionary containing biodiversity metrics including
                - total_observations: Total matching observations in GBIF
                - observations_in_response: Number of observations retrieved
                - unique_species: Count of unique species
                - dominant_species: Top 3 most common species
                - kingdoms_present: Kingdoms represented in observations
                - dominant_kingdom: Most common kingdom
                - phyla_present: Phyla represented in observations
                - dominant_phylum: Most common phylum
                Returns error dict if request fails.
        """
        return self._fetch_data(lat, lon, **kwargs)
    
    def _fetch_data(self, lat: float, lon: float, **kwargs) -> Dict[str, Any]:
        """Fetch species observations near coordinates."""
        radius_km = kwargs.get('radius_km', 10)
        
        params = {
            'decimalLatitude': lat,
            'decimalLongitude': lon,
            'limit': 100,
            'offset': 0
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                # Extract biodiversity metrics
                species_list = [r.get('species', '') for r in results if 'species' in r]
                unique_species = len(set(species_list))
                
                kingdom_counts = Counter([r.get('kingdom', '') for r in results])
                phylum_counts = Counter([r.get('phylum', '') for r in results])
                
                # Get dominant species
                species_counts = Counter(species_list)
                dominant_species = [sp for sp, _ in species_counts.most_common(3)]
                
                return {
                    'total_observations': data.get('count', 0),
                    'observations_in_response': len(results),
                    'unique_species': unique_species,
                    'dominant_species': ', '.join(dominant_species),
                    'kingdoms_present': ', '.join(kingdom_counts.keys()) if kingdom_counts else 'N/A',
                    'dominant_kingdom': kingdom_counts.most_common(1)[0][0] if kingdom_counts else 'N/A',
                    'phyla_present': ', '.join(phylum_counts.keys()) if phylum_counts else 'N/A',
                    'dominant_phylum': phylum_counts.most_common(1)[0][0] if phylum_counts else 'N/A',
                }
            else:
                return {'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'error': str(e)[:100]}
    
    def fetch_and_enrich(self, adata, metadata_cols: List[str] = None) -> pd.DataFrame:
        """Fetch biodiversity data for all samples."""
        if metadata_cols is None:
            metadata_cols = ['latitude', 'longitude']
        
        results = []
        
        for idx, row in adata.iterrows():
            try:
                # Safe extraction of lat/lon with proper error handling
                lat_val = row.get('latitude') or row.get('lat')
                lon_val = row.get('longitude') or row.get('lon')
                
                if lat_val is None or lon_val is None:
                    results.append({})
                    continue
                
                lat = float(lat_val)
                lon = float(lon_val)
                
                if pd.isna(lat) or pd.isna(lon):
                    results.append({})
                    continue
                
                data = self._fetch_data(lat, lon)
                
                # Convert to flat columns with prefix
                row_data = {
                    f'gbif_{k}': v for k, v in data.items()
                }
                results.append(row_data)
                
            except (TypeError, ValueError) as e:
                if self.verbose:
                    self.logger.warning(f"Error processing row {idx}: Invalid lat/lon - {e}")
                results.append({})
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Error processing row {idx}: {e}")
                results.append({})
        
        return pd.DataFrame(results)
