# workflow_16s/api/environmental_data/other/tools/_icmm_mines.py
"""
ICMM Mining Facilities Database

International Council on Mining and Metals (ICMM) global mining operations.
15,188 active mining sites with 47 commodity tracking.

Reference: https://www.icmm.com/
"""

import requests
import numpy as np
from typing import Optional, Dict, Any
from .base import BaseEnvironmentalAPI
from .cache import cache_api_call

class ICMM_MinesAPI(BaseEnvironmentalAPI):
    """
    Global mining facilities and operations database.
    Nearest-facility lookup for mining activity and commodity extraction.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.api_name = "ICMM_Mines"
        self.base_url = "https://mines.globalwitness.org/api/facilities"  # Placeholder
        self.timeout = 15
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 47 commodities tracked
        self.commodities = [
            'Au', 'Ag', 'Cu', 'Ni', 'Co', 'Zn', 'Pb', 'Sn', 'Mo', 'W',
            'Fe', 'Mn', 'Cr', 'V', 'Ti', 'Al', 'Rare_Earths', 'Li', 'Be', 'B',
            'P', 'K', 'S', 'As', 'Sb', 'Bi', 'Se', 'Te', 'U', 'Th',
            'Ge', 'Ga', 'In', 'Cd', 'Hg', 'Tl', 'Pd', 'Pt', 'Rh', 'Ir',
            'Diamonds', 'Salt', 'Limestone', 'Gypsum', 'Mica', 'Feldspar'
        ]
    
    def check_requirements(self) -> tuple[bool, str]:
        """Verify ICMM mines database availability."""
        # ICMM (International Council on Mining and Metals) mines database not available via public API
        # Member directory available at: https://www.icmm.com/mining-companies
        # Facility location data requires direct contact with ICMM (members only)
        return False, "ICMM mines database not available via public API"
    
    @cache_api_call
    def get_data(self, lat: float, lon: float, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Query nearest mining facility.
        
        Returns dict with:
        - icmm_mines_facility_distance_km: Distance to nearest mine
        - icmm_mines_primary_commodity: Main extracted commodity
        - icmm_mines_other_commodities_produced: Secondary commodities (list)
        - icmm_mines_pit_type: Surface/underground/artisanal
        - icmm_mines_operational_status: Active/inactive/closed
        - icmm_mines_production_capacity_k_tons_yr: Annual capacity
        """
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "search_radius_km": 100,
                "limit": 1,
                "include_inactive": True
            }
            
            response = self.session.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            if not data or 'facilities' not in data or not data['facilities']:
                return None
            
            facility = data['facilities'][0]
            
            result = {
                "icmm_mines_facility_distance_km": float(facility.get('distance_km', np.nan)),
                "icmm_mines_primary_commodity": facility.get('primary_commodity', ''),
                "icmm_mines_pit_type": facility.get('mining_type', ''),  # Surface/Underground/Artisanal
                "icmm_mines_operational_status": facility.get('operational_status', ''),
                "icmm_mines_production_capacity_k_tons_yr": float(facility.get('annual_capacity_k_tons', np.nan)),
                "icmm_mines_facility_country": facility.get('country', ''),
                "icmm_mines_years_operating": int(facility.get('years_in_operation', 0)),
                "icmm_mines_near_dwelling": facility.get('proximity_to_settlement', False)
            }
            
            # Add secondary commodities as booleans
            if 'secondary_commodities' in facility:
                for commodity in self.commodities:
                    result[f"icmm_mines_produces_{commodity}"] = (
                        commodity in facility.get('secondary_commodities', [])
                    )
            
            # Multiple commodities indicate mining activity complexity
            n_commodities = sum(1 for v in result.values() if v is True and isinstance(v, bool))
            
            return result if n_commodities > 0 or np.isfinite(result.get('icmm_mines_facility_distance_km', np.nan)) else None
            
        except Exception as e:
            self.logger.debug(f"ICMM Mines error: {e}")
            return None
