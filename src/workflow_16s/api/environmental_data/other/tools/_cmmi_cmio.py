# workflow_16s/api/environmental_data/other/tools/_cmmi_cmio.py
"""
CMMI Critical Minerals in Ore Deposits (CMiO) Database

Global cadastre of ore deposits with commodity concentrations.
20,000+ mining sites with multi-element geochemistry.

Reference: https://www.icmm.com/en-au/minerals/
"""

import requests
import numpy as np
from typing import Optional, Dict, Any
from .base import BaseEnvironmentalAPI
from .cache import cache_api_call

class CMMI_CMiO_API(BaseEnvironmentalAPI):
    """
    Critical Minerals Institute ore deposit database.
    Nearest mineralized site lookup for metal concentrations.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.api_name = "CMMI_CMiO"
        self.base_url = "https://cmio.cmm.org/api/deposits"  # Placeholder
        self.timeout = 15
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Critical minerals tracked
        self.commodities = ['Co', 'Ga', 'REEs', 'Se', 'Te', 'Zn', 'Cu', 'Ni', 'Mo', 'W', 'Rare_Earths']
    
    def check_requirements(self) -> tuple[bool, str]:
        """Verify CMMI/CMiO database availability."""
        # CMMI (Communities, Mining and Indigenous Issues) CMiO database is not publicly available via API
        # Data available through: https://www.icmm.com/ or direct contact
        return False, "CMMI CMiO database not available (restricted access)"
    
    @cache_api_call
    def get_data(self, lat: float, lon: float, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Query nearest ore deposit for commodity data.
        
        Returns dict with:
        - cmmi_cmio_{commodity}_grade_%: Grade percentage
        - cmmi_cmio_deposit_distance_km: Distance to deposit
        - cmmi_cmio_deposit_type: Deposit classification
        """
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "radius_km": 200,
                "limit": 1
            }
            
            response = self.session.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            if not data or 'deposits' not in data or not data['deposits']:
                return None
            
            deposit = data['deposits'][0]
            
            result = {
                f"cmmi_cmio_{commodity}_grade_pct": float(deposit.get(f'{commodity}_grade_pct', np.nan))
                for commodity in self.commodities
            }
            
            result.update({
                "cmmi_cmio_deposit_distance_km": float(deposit.get('distance_km', np.nan)),
                "cmmi_cmio_deposit_type": deposit.get('deposit_type', ''),
                "cmmi_cmio_deposit_country": deposit.get('country', '')
            })
            
            return result if any(np.isfinite(v) for v in result.values()) else None
            
        except Exception as e:
            self.logger.debug(f"CMiO API error: {e}")
            return None
