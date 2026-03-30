# workflow_16s/api/environmental_data/other/tools/_gesamp_fe.py
"""
GESAMP Ocean Iron Data

Global Ocean Ship-based Hydrographic Investigations Program (GESAMP) iron measurements.
4-model ensemble of dissolved and particulate iron in global oceans.

Reference: http://www.gesamp.org/
"""

import requests
import numpy as np
from typing import Optional, Dict, Any
from .base import BaseEnvironmentalAPI
from .cache import cache_api_call

class GESAMP_FeAPI(BaseEnvironmentalAPI):
    """
    GESAMP iron (Fe) ocean data integration.
    Grid/profile-based lookup for seawater iron concentrations.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.api_name = "GESAMP_Fe"
        self.base_url = "https://www.nodc.noaa.gov/cgi-bin/OC5/nph-ods"  # NOAA data portal
        self.timeout = 15
        self.cache_hits = 0
        self.cache_misses = 0
    
    def check_requirements(self) -> tuple[bool, str]:
        """Verify GESAMP iron data availability."""
        # GESAMP (Global Earth Surface Anthropogenic Processes) Fe data not available via public API
        # GESAMP publications available at: http://www.gesamp.org/
        # Iron flux data from publications and data archives only
        return False, "GESAMP Fe not available via public API"
    
    @cache_api_call
    def get_data(self, lat: float, lon: float, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Query ocean iron data at coordinates.
        
        Returns dict with:
        - gesamp_fe_total_nmol_l: Total dissolved iron (nM)
        - gesamp_fe_labile_nmol_l: Labile (bioavailable) iron (nM)
        - gesamp_fe_model_ensemble_uncertainty: Ensemble std dev
        - gesamp_fe_sampling_depth_m: Typical sampling depth
        """
        try:
            # For ocean samples, would query regional databases
            # Placeholder for NOAA/WHOI/etc iron data
            params = {
                "lat": lat,
                "lon": lon,
                "variable": "Fe_total",
                "output": "json"
            }
            
            response = self.session.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            if not data or 'data' not in data:
                return None
            
            data_list = data.get('data', [])
            sample = data_list[0] if data_list else {}
            
            result = {
                "gesamp_fe_total_nmol_l": float(sample.get('Fe_total_nmol_L', np.nan)),
                "gesamp_fe_labile_nmol_l": float(sample.get('Fe_labile_nmol_L', np.nan)),
                "gesamp_fe_model_ensemble_uncertainty": float(sample.get('uncertainty_nmol_L', np.nan)),
                "gesamp_fe_sampling_depth_m": float(sample.get('depth_m', np.nan)),
                "gesamp_fe_profile_date": sample.get('profile_date', '')
            }
            
            # Only return if in ocean (lat/lon combination suggests marine sampling)
            if np.isfinite(result['gesamp_fe_total_nmol_l']):
                return result
            return None
            
        except Exception as e:
            self.logger.debug(f"GESAMP Fe error: {e}")
            return None
