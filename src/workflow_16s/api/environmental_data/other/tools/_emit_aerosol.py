# workflow_16s/api/environmental_data/other/tools/_emit_aerosol.py
"""
EMIT Aerosol Mineralogy Model

Earth Surface Mineral Dust Source Investigation (EMIT) global aerosol mineralogy.
Grid-based mineral concentrations from hyperspectral remote sensing.

Reference: https://earth.jpl.nasa.gov/emit/
"""

import os
import requests
import numpy as np
from typing import Optional, Dict, Any
from .base import BaseEnvironmentalAPI
from .cache import cache_api_call

class EMIT_AerosolAPI(BaseEnvironmentalAPI):
    """
    EMIT hyperspectral mineral aerosol data.
    Grid-based lookup for mineral dust composition.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.api_name = "EMIT_Aerosol"
        self.base_url = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMIT_L3_MIN_001"  # Placeholder
        self.timeout = 15
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Key mineral-forming elements
        self.elements = ['Al', 'Ca', 'Fe', 'Si', 'Ti', 'Mg', 'Mn', 'K']
    
    def check_requirements(self) -> tuple[bool, str]:
        """Verify EMIT aerosol data availability."""
        # EMIT (Earth Surface Mineral Dust Source Investigation) requires NASA EarthData authentication
        # Data available at: https://lpdaac.usgs.gov/products/emitl2aer/
        # Requires NASA Earthdata login and local download
        return False, "EMIT aerosol requires NASA EarthData authentication"
    
    @cache_api_call
    def get_data(self, lat: float, lon: float, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Query EMIT mineral composition at grid cell.
        
        Returns dict with:
        - emit_{element}_column_density_g_m2: Hyperspectral-derived column density
        - emit_grid_cell_center_lat/lon
        """
        try:
            # EMIT provides global gridded data (~60m resolution)
            # Query granule containing coordinates
            api_key = os.environ.get('NASA_API_KEY')
            if not api_key:
                logger.warning("NASA_API_KEY not set; EMIT API will be unavailable")
                return None
            params = {
                "latitude": lat,
                "longitude": lon,
                "format": "json"
            }
            
            response = self.session.get(
                "https://api.nasa.gov/planetary/earth/imagery",
                params={**params, "api_key": api_key},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            if not data or 'results' not in data:
                return None
            
            results_list = data.get('results', [])
            result = results_list[0] if results_list else {}
            
            # Extract mineral optical depth estimates
            mineral_data = {
                f"emit_{elem}_optical_depth": float(result.get(f'{elem}_OD', np.nan))
                for elem in self.elements
            }
            
            mineral_data.update({
                "emit_quality_flag": result.get('quality_flag', 0),
                "emit_retrieval_date": result.get('retrieval_date', '')
            })
            
            return mineral_data if any(np.isfinite(v) for v in mineral_data.values() if isinstance(v, float)) else None
            
        except Exception as e:
            self.logger.debug(f"EMIT API error: {e}")
            return None
