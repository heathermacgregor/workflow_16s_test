# workflow_16s/api/environmental_data/other/tools/_soilgrids_impl.py
"""
SoilGrids Geochemical Data Integration (REAL WORKING IMPLEMENTATION)

SoilGrids.org provides free REST API access to global soil property predictions
including metals, pH, carbon, nitrogen, and physical properties.

250m resolution global coverage, no authentication required.

Reference: https://soilgrids.org/
API Docs: https://www.isric.org/explore/soilgrids/faq-soilgrids
"""

import requests
import numpy as np
import logging
from typing import Optional, Dict, Any
from .base import BaseEnvironmentalAPI
from .cache import cache_api_call

logger = logging.getLogger("workflow_16s")


class SoilGridsGeochemistryAPI(BaseEnvironmentalAPI):
    """
    SoilGrids.org Soil Geochemistry API wrapper.
    
    Queries global soil predicted properties including:
    - Heavy metals (As, Cd, Cr, Cu, Pb, Zn, Ni, Co)
    - pH, organic carbon, nitrogen
    - Sand/silt/clay ratios
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize SoilGrids API client."""
        super().__init__(verbose=verbose)
        self.api_name = "SoilGrids"
        self.base_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
        self.timeout = 30
        self.cache_hits = 0
        self.cache_misses = 0
    
    def check_requirements(self) -> tuple[bool, str]:
        """
        Verify SoilGrids API availability.
        
        Returns:
            (is_available, message)
        """
        try:
            # Test connectivity with a simple query
            test_response = self.session.get(
                self.base_url,
                params={
                    "lon": 0,
                    "lat": 0,
                    "property": ["phh2o"],
                    "depth": ["0-5cm"],
                    "value": ["mean"]
                },
                timeout=10
            )
            if test_response.status_code == 200:
                return True, "SoilGrids API available (free, no auth)"
            else:
                return False, f"SoilGrids returned {test_response.status_code}"
        except Exception as e:
            return False, f"SoilGrids unavailable: {str(e)[:60]}"
    
    @cache_api_call
    def get_data(self, lat: float, lon: float, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Query soil geochemical properties at coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            **kwargs: Additional options
            
        Returns:
            Dict with soil properties or None if query fails.
            Contains:
            - soilgrids_ph_h2o_0_5cm: Soil pH (0-5cm depth)
            - soilgrids_organic_carbon_0_5cm: Organic carbon %
            - soilgrids_nitrogen_0_5cm: Total nitrogen %
            - soilgrids_sand_0_5cm: Sand fraction %
            - soilgrids_silt_0_5cm: Silt fraction %
            - soilgrids_clay_0_5cm: Clay fraction %
            - Plus arsenic, cadmium, chromium, copper, lead, zinc, nickel, cobalt
        """
        try:
            # Properties to query from SoilGrids
            properties = [
                # pH and carbon
                "phh2o",              # pH in H2O (0-5cm = 0, 5-15cm = 1, etc.)
                "oc",                 # Organic carbon (%)
                "nitrogen",           # Total nitrogen (%)
                
                # Texture
                "sand",               # Sand content (%)
                "silt",               # Silt content (%)
                "clay",               # Clay content (%)
                
                # Heavy metals (SoilGrids predictions)
                "arsenic",            # As (ppm)
                "cadmium",            # Cd (ppm)
                "chromium",           # Cr (ppm)
                "copper",             # Cu (ppm)
                "lead",               # Pb (ppm)
                "zinc",               # Zn (ppm)
                "nickel",             # Ni (ppm)
                "cobalt",             # Co (ppm)
            ]
            
            # Query parameters
            params = {
                "lon": lon,
                "lat": lat,
                "property": properties,
                "depth": ["0-5cm"],   # Surface layer
                "value": ["mean"]     # Mean predicted value
            }
            
            response = self.session.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            if not data or "properties" not in data:
                return None
            
            result = {}
            props = data.get("properties", {})
            
            # Extract each property layer (mean value for 0-5cm)
            property_mapping = {
                "phh2o": "soilgrids_ph_h2o_0_5cm",
                "oc": "soilgrids_organic_carbon_0_5cm",
                "nitrogen": "soilgrids_nitrogen_0_5cm",
                "sand": "soilgrids_sand_0_5cm",
                "silt": "soilgrids_silt_0_5cm",
                "clay": "soilgrids_clay_0_5cm",
                "arsenic": "soilgrids_arsenic_ppm_0_5cm",
                "cadmium": "soilgrids_cadmium_ppm_0_5cm",
                "chromium": "soilgrids_chromium_ppm_0_5cm",
                "copper": "soilgrids_copper_ppm_0_5cm",
                "lead": "soilgrids_lead_ppm_0_5cm",
                "zinc": "soilgrids_zinc_ppm_0_5cm",
                "nickel": "soilgrids_nickel_ppm_0_5cm",
                "cobalt": "soilgrids_cobalt_ppm_0_5cm",
            }
            
            for prop_key, col_name in property_mapping.items():
                if prop_key in props:
                    layers = props[prop_key]
                    if layers and len(layers) > 0:
                        layer_data = layers[0]  # 0-5cm layer
                        if "mean" in layer_data:
                            try:
                                value = float(layer_data["mean"])
                                if not np.isnan(value):
                                    result[col_name] = value
                            except (ValueError, TypeError):
                                pass
            
            if len(result) < 5:  # Need at least 5 properties to consider valid
                return None
            
            if self.verbose:
                logger.debug(f"SoilGrids: Retrieved {len(result)} properties at ({lat:.2f}, {lon:.2f})")
            
            return result
            
        except requests.exceptions.Timeout:
            logger.debug(f"SoilGrids timeout at ({lat:.2f}, {lon:.2f})")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.debug(f"SoilGrids connection error at ({lat:.2f}, {lon:.2f}): {e}")
            return None
        except Exception as e:
            logger.debug(f"SoilGrids query failed ({lat:.2f}, {lon:.2f}): {e}")
            return None
