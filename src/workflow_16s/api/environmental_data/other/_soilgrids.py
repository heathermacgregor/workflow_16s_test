# ==================================================================================== #

# Standard Library Imports
import logging
import requests
import json
import time  # Used for polite API requests
from typing import Optional, Dict, Any, List

# Local Imports
from workflow_16s.api.environmental_data.other import (
    BaseEnvironmentalAPI, REQUEST_TIMEOUT, cache_api_call
)
from workflow_16s.utils.logger import get_logger

# ==================================================================================== #

logger = get_logger()
REQUEST_TIMEOUT = 45 # Increased timeout for potentially large requests

# ==================================================================================== #

class SoilGridsAPI(BaseEnvironmentalAPI):
    """Fetches soil property data from the ISRIC SoilGrids database."""
    URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    DEFAULT_PROPERTIES = ["phh2o", "clay", "sand", "silt", "soc", "cec", "bdod", "nitrogen"]
    ALL_PROPERTIES = ["bdod", "cec", "cfvo", "clay", "sand", "silt", "nitrogen", "ocd", "ocs", "phh2o", "soc", "wrb"]
    DEFAULT_DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        super().__init__(verbose=self.verbose)
        self.base_url = self.URL

    def get_data( # type: ignore
        self, lat: float, lon: float, properties: Optional[List[str]] = None, 
        depths: Optional[List[str]] = None, fetch_all: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves soil properties. Acts as an orchestrator for single or multiple API calls.
        """
        depths_to_fetch = depths or self.DEFAULT_DEPTHS
        
        # If fetch_all is True, loop and make multiple smaller, cached requests
        if fetch_all:
            all_soil_properties = {}
            if self.verbose: logger.info(f"Fetching ALL {len(self.ALL_PROPERTIES)} properties via multiple requests...")
            for prop in self.ALL_PROPERTIES:
                # Fetch one property at a time using the cached internal method
                data = self._fetch_properties(lat, lon, properties=[prop], depths=depths_to_fetch)
                if data:
                    all_soil_properties.update(data)
                # Increased delay to prevent hitting rate limits
                time.sleep(1)
            return all_soil_properties

        # Standard logic for default or custom property lists (single request)
        properties_to_fetch = properties or self.DEFAULT_PROPERTIES
        return self._fetch_properties(lat, lon, properties=properties_to_fetch, depths=depths_to_fetch)

    @cache_api_call
    def _fetch_properties(
        self, lat: float, lon: float, properties: List[str], depths: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Internal method that performs a single, cached API request for a list of properties.
        """
        params = {"lon": lon, "lat": lat, "property": properties, "depth": depths, "value": ["mean"]}
        
        try:
            if self.verbose: logger.info(f"Fetching {len(properties)} soil properties for ({lat}, {lon})...")
            response = self.session.get(self.base_url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            if "properties" not in data or "layers" not in data["properties"]:
                logger.warning("Soil data not found in API response (likely no data for this specific point).")
                return None
            
            soil_properties = {}
            for layer in data["properties"]["layers"]:
                prop_name, unit_measure = layer["name"], layer["unit_measure"]
                unit = unit_measure.get("d_class_label", "value")
                divisor = float(unit_measure.get("conversion_factor", 1.0))
                
                for depth_interval in layer["depths"]:
                    depth_label = depth_interval["label"]
                    raw_value = depth_interval["values"]["mean"]
                    actual_value = round(raw_value / divisor, 2) if raw_value is not None else None
                    key = f"{prop_name}_{depth_label}"
                    soil_properties[key] = {"value": actual_value, "unit": unit}
            
            return soil_properties

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.error(f"SoilGrids API request failed: {e}. This is a rate-limiting error. The script includes delays, but the server is still busy. Please wait a moment and try again.")
            else:
                logger.error(f"SoilGrids API request failed with an HTTP error: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"SoilGrids API request failed with a network error: {e}")
            return None

# ==================================================================================== #

if __name__ == "__main__":
    
    print("--- Running SoilGrids API Example Test ---")

    # A rural area in California's Central Valley, likely to have soil data.
    FRESNO_COUNTY_LAT = 36.74
    FRESNO_COUNTY_LON = -119.77

    api_client = SoilGridsAPI(verbose=True)

    # --- Test 1: Fetching default (common) soil properties ---
    print("\n" + "="*50)
    print("## Test 1: Fetching default (common) soil properties...")
    print(f"(Default list: {api_client.DEFAULT_PROPERTIES})")
    print("="*50)
    default_data = api_client.get_data(lat=FRESNO_COUNTY_LAT, lon=FRESNO_COUNTY_LON)
    if default_data:
        print("✅ Success! Found default soil properties:")
        print(json.dumps(default_data, indent=2))
    else:
        print("❌ Failed to retrieve data.")

    # --- Test 2: Fetching ALL available properties ---
    print("\n" + "="*50)
    print("## Test 2: Fetching ALL available soil properties (fetch_all=True)...")
    print("="*50)
    all_data = api_client.get_data(lat=FRESNO_COUNTY_LAT, lon=FRESNO_COUNTY_LON, fetch_all=True)
    if all_data:
        print(f"✅ Success! Found {len(all_data)} data points for all properties:")
        # Print a few examples instead of the whole list
        summary = {k: v for i, (k, v) in enumerate(all_data.items()) if i < 5}
        print(json.dumps(summary, indent=2))
        if len(all_data) > 5:
            print(f"  ... and {len(all_data) - 5} more.")
    else:
        print("❌ Failed to retrieve data.")
        
    # --- Test 3: Fetching a specific, custom list of properties ---
    print("\n" + "="*50)
    print("## Test 3: Fetching custom properties (overriding default)...")
    print("(Bulk Density and Cation Exchange Capacity at deeper intervals)")
    print("="*50)
    custom_properties = ["bdod", "cec"]
    custom_depths = ["15-30cm", "30-60cm", "60-100cm"]
    custom_data = api_client.get_data(
        lat=FRESNO_COUNTY_LAT,
        lon=FRESNO_COUNTY_LON,
        properties=custom_properties,
        depths=custom_depths
    )
    if custom_data:
        print("✅ Success! Found custom soil properties:")
        print(json.dumps(custom_data, indent=2))
    else:
        print("❌ Failed to retrieve custom data.")