# ==================================================================================== #

# Standard Library Imports
import io
import json
import math
import requests
import statistics as stats_module
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Third-Party Imports
import pandas as pd

# Local Imports
from workflow_16s.api.environmental_data.other import (
    BaseEnvironmentalAPI, REQUEST_TIMEOUT, cache_api_call
)
from workflow_16s.utils.logger import get_logger

# ==================================================================================== #

logger = get_logger()
REQUEST_TIMEOUT = 30

# ==================================================================================== #

class USGS_Earthquake_API(BaseEnvironmentalAPI):
    """Fetches recent earthquake data from the USGS real-time catalog."""
    URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        super().__init__(verbose=self.verbose)
        self.base_url = self.URL
    
    @cache_api_call
    def get_data( # type: ignore
        self, lat: float, lon: float, radius_km: int = 150, min_magnitude: float = 2.5,
        fetch_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieves earthquakes near a location for a specific date or the last 30 days."""
        params = {
            "format": "geojson", "latitude": lat, "longitude": lon,
            "maxradiuskm": radius_km, "minmagnitude": min_magnitude,
            "orderby": "time"
        }
        
        if fetch_date:
            try:
                date_obj = datetime.strptime(fetch_date, '%Y-%m-%d')
                params["starttime"] = date_obj.strftime('%Y-%m-%d')
                params["endtime"] = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
            except ValueError:
                logger.error("Invalid date format for fetch_date. Use 'YYYY-MM-DD'.")
                return None
        
        try:
            if self.verbose: logger.info(f"Querying USGS for earthquakes near ({lat}, {lon})...")
            response = self.session.get(self.base_url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            earthquakes = []
            for feature in data.get("features", []):
                prop = feature.get("properties", {})
                dt_obj = datetime.fromtimestamp(prop.get("time", 0) / 1000)
                earthquakes.append({
                    "magnitude": prop.get("mag"),
                    "location": prop.get("place"),
                    "time_utc": dt_obj.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    "details_url": prop.get("url")
                })
            
            return {"earthquakes": earthquakes, "count": len(earthquakes)}
        except requests.exceptions.RequestException as e:
            logger.error(f"USGS Earthquake API request failed: {e}")
            return None


class USGS_Water_Services_API(BaseEnvironmentalAPI):
    """Fetches real-time or historical stream gauge data from the USGS Water Services API."""
    URL = "https://waterservices.usgs.gov/nwis"
    def __init__(self, verbose=False):
        self.verbose = verbose
        super().__init__(verbose=self.verbose)
        self.base_url = self.URL

    @cache_api_call
    def get_data( # type: ignore
        self, lat: float, lon: float, radius_miles: int = 10,
        fetch_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieves data from the nearest active USGS monitoring sites."""
        try:
            # Step 1: Find nearby active stream monitoring sites using the robust 'rdb' format
            lat_delta = radius_miles / 69.0
            lon_delta = radius_miles / (math.cos(math.radians(lat)) * 69.0)
            bBox = f"{lon - lon_delta:.6f},{lat - lat_delta:.6f},{lon + lon_delta:.6f},{lat + lat_delta:.6f}"
            
            # **FIX: Changed format to 'rdb' for the site service request**
            sites_params = {"format": "rdb", "bBox": bBox, "siteStatus": "active", "siteType": "ST", "hasDataTypeCd": "iv"}
            
            if self.verbose: logger.info(f"Finding USGS stream gauges for ({lat}, {lon})...")
            response = self.session.get(f"{self.base_url}/site/", params=sites_params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            # **FIX: Parse the tab-separated response instead of JSON**
            site_df = pd.read_csv(io.StringIO(response.text), sep='\t', comment='#')
            
            if 'site_no' not in site_df.columns or site_df.empty or len(site_df) < 2:
                logger.warning("No active monitoring sites with real-time data found in this area.")
                return {"sites": [], "count": 0}

            # The first data row is a definition row, so skip it
            site_df = site_df.iloc[1:]
            site_codes = site_df['site_no'].astype(str).tolist()[:3] # Get up to 3 site codes

            if not site_codes:
                logger.warning("No site codes could be extracted from the site service response.")
                return {"sites": [], "count": 0}

            # Step 2: Fetch data for those sites (this part correctly uses JSON)
            iv_params = {"format": "json", "sites": ",".join(site_codes), "parameterCd": "00060,00065"} # 00060=Streamflow, 00065=Gage height
            
            if fetch_date:
                date_obj = datetime.strptime(fetch_date, '%Y-%m-%d')
                iv_params["startDT"] = date_obj.strftime('%Y-%m-%d')
                iv_params["endDT"] = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
            
            if self.verbose: logger.info(f"Fetching data for sites: {', '.join(site_codes)}...")
            response = self.session.get(f"{self.base_url}/iv/", params=iv_params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            measurements_data = response.json()

            # Step 3: Process the results
            sites_dict = {}
            for ts in measurements_data.get("value", {}).get("timeSeries", []):
                site_name = ts["sourceInfo"]["siteName"]
                param_desc = ts["variable"]["variableDescription"]
                unit = ts["variable"]["unit"]["unitCode"]
                
                if site_name not in sites_dict:
                    sites_dict[site_name] = {"name": site_name, "measurements": {}}
                
                valid_values = [v for v in ts["values"][0]["value"] if v.get('value') is not None]
                if not valid_values: continue

                if fetch_date: # Calculate daily average
                    numeric_values = [float(v['value']) for v in valid_values]
                    avg_value = round(stats_module.mean(numeric_values), 2)
                    sites_dict[site_name]["measurements"][param_desc] = f"{avg_value} {unit} (daily avg)"
                else: # Get latest value
                    latest_reading = valid_values[0]
                    value = latest_reading['value']
                    timestamp = latest_reading['dateTime']
                    sites_dict[site_name]["measurements"][param_desc] = f"{value} {unit} at {timestamp}"

            results_list = list(sites_dict.values())
            return {"sites": results_list, "count": len(results_list)}

        except requests.exceptions.RequestException as e:
            logger.error(f"USGS Water Services API request failed: {e}")
            return None
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Failed to parse USGS Water Services response: {e}")
            return None

# ==================================================================================== #

if __name__ == "__main__":
    print("--- Running USGS APIs Example Test ---")

    # Berkeley, CA coordinates 🌉
    BERKELEY_LAT = 37.8715
    BERKELEY_LON = -122.2730
    TEST_DATE = "2024-09-30" # A recent date for historical queries

    # --- 1. Earthquake API ---
    print("\n" + "="*50)
    print("## Test 1: USGS Earthquake API")
    print("="*50)
    quake_client = USGS_Earthquake_API(verbose=True)
    
    print("\n---\n▶️ Test 1a: Fetching recent earthquakes (last 30 days)...\n---")
    recent_quakes = quake_client.get_data(BERKELEY_LAT, BERKELEY_LON)
    print("✅ Result:", json.dumps(recent_quakes, indent=2))

    print(f"\n---\n▶️ Test 1b: Fetching earthquakes for a single day ({TEST_DATE})...\n---")
    date_quakes = quake_client.get_data(BERKELEY_LAT, BERKELEY_LON, fetch_date=TEST_DATE)
    print("✅ Result:", json.dumps(date_quakes, indent=2))

    # --- 2. Water Services API ---
    print("\n" + "="*50)
    print("## Test 2: USGS Water Services API (FIXED)")
    print("="*50)
    water_client = USGS_Water_Services_API(verbose=True)
    
    print("\n---\n▶️ Test 2a: Fetching LATEST stream gauge data...\n---")
    latest_water_data = water_client.get_data(BERKELEY_LAT, BERKELEY_LON, radius_miles=20)
    print("✅ Result:", json.dumps(latest_water_data, indent=2))

    print(f"\n---\n▶️ Test 2b: Fetching DAILY AVERAGE stream gauge data for {TEST_DATE}...\n---")
    historical_water_data = water_client.get_data(BERKELEY_LAT, BERKELEY_LON, radius_miles=20, fetch_date=TEST_DATE)
    print("✅ Result:", json.dumps(historical_water_data, indent=2))