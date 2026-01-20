# ==================================================================================== #

# Standard Imports
import json
import logging
import requests
from typing import Any, Dict, List, Optional

# Third-Party Imports
import pandas as pd

# Local Imports
from workflow_16s.api.environmental_data.other import BaseEnvironmentalAPI, REQUEST_TIMEOUT, cache_api_call
from workflow_16s.utils.logger import get_logger

# ==================================================================================== #

logger = get_logger()

# ==================================================================================== #

class RadNetAPI(BaseEnvironmentalAPI):
    """
    Fetches radiological monitoring data from the EPA's RadNet system.
    Finds the nearest monitor and returns its latest readings.
    """
    URL = "https://iaspub.epa.gov/enviro/efservice/"
    def __init__(self, verbose: bool = False):
        """
        Initializes the RadNetAPI client.
        The base URL for monitor locations is pre-set.
        """
        super().__init__(verbose=verbose)
        self.base_url = self.URL

    @cache_api_call
    def get_data(self, lat: float, lon: float, **kwargs) -> Dict[str, Any]:
        """
        Finds the nearest RadNet monitor and retrieves its data.

        Args:
            lat: Latitude of the target location.
            lon: Longitude of the target location.

        Returns:
            A dictionary containing data from the nearest RadNet monitor.
        """
        # 1. Find the nearest monitoring station
        # The RadNet API can search for monitors within a specified box
        box_size = 0.5  # Search within a ~55km square box
        monitor_endpoint = f"RADNET_MONITOR/BOX/{lon-box_size}/{lat-box_size}/{lon+box_size}/{lat+box_size}/JSON"
        
        monitors_data = self.get_json(monitor_endpoint)
        if not monitors_data:
            return {"error": "Could not retrieve RadNet monitor list."}

        # Convert to DataFrame to easily find the closest monitor
        df = pd.DataFrame(monitors_data)
        if df.empty:
            return {"radnet_data": "No RadNet monitors found within search radius."}
            
        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'])
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'])

        # Calculate distance to find the nearest monitor
        df['distance'] = ((df['LATITUDE'] - lat)**2 + (df['LONGITUDE'] - lon)**2)**0.5
        nearest_monitor = df.loc[df['distance'].idxmin()]
        monitor_id = nearest_monitor['MONITOR_ID']
        
        # 2. Get the latest measurements from that station
        measurements_endpoint = f"RADNET_MEASUREMENTS/MONITOR_ID/{monitor_id}/JSON"
        measurements_data = self.get_json(measurements_endpoint)
        
        if not measurements_data:
            return {"error": f"Failed to retrieve measurements for Monitor ID {monitor_id}."}

        # 3. Format the results
        # Let's return the most recent measurement and station info
        latest_measurement = measurements_data[0] # API returns data sorted by date descending
        
        result = {
            "nearest_monitor_info": {
                "monitor_id": monitor_id,
                "city": nearest_monitor.get('CITY'),
                "state": nearest_monitor.get('STATE'),
                "distance_degrees": round(nearest_monitor['distance'], 4),
                "location": {
                    "type": "Point",
                    "coordinates": [nearest_monitor['LONGITUDE'], nearest_monitor['LATITUDE']]
                }
            },
            "latest_measurement": {
                "collection_date": latest_measurement.get('SAMPLE_DATE'),
                "analysis": latest_measurement.get('ANALYTE_NAME'),
                "result": f"{latest_measurement.get('RESULT_NUM')} {latest_measurement.get('RESULT_UNIT_CODE')}",
                "result_type": latest_measurement.get('RESULT_TYPE_NAME')
            }
        }
        
        return result
    
    def get_json(self, endpoint: str, params: Optional[Dict] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Constructs a URL, sends a GET request, and returns the parsed JSON response.

        Args:
            endpoint: The specific API endpoint to append to the base URL.
            params: A dictionary of query parameters for the request.

        Returns:
            A list of dictionaries parsed from the JSON response, or None if an error occurs.
        """
        url = f"{self.base_url}{endpoint}"
        
        if self.verbose:
            logger.info(f"⏳ Requesting data from: {url}")
            
        try:
            response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            # This will raise an HTTPError for bad responses (4xx or 5xx)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"🟥 HTTP error occurred for {url}: {http_err}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"🟥 Request failed for {url}: {req_err}")
        except ValueError as json_err:
            logger.error(f"🟥 Failed to parse JSON from {url}: {json_err}")
            
        return None

# ==================================================================================== #

if __name__ == "__main__":
    
    print("--- Running RadNet API Example Test ---")
    
    # Berkeley, CA coordinates ☢️
    berkeley_lat = 37.8715
    berkeley_lon = -122.2730
    
    # 1. Instantiate the API client with verbose logging
    api_client = RadNetAPI(verbose=True)
    
    # 2. Fetch data for the location
    print("\n" + "="*50)
    print("## Test 1: Fetching RadNet data for Berkeley, CA...")
    print("="*50)
    radnet_data = api_client.get_data(lat=berkeley_lat, lon=berkeley_lon)
    
    if radnet_data and "error" not in radnet_data:
        print("✅ Success! Found RadNet data:")
        print(json.dumps(radnet_data, indent=2))
    elif radnet_data:
        print(f"❌ Failed to retrieve data: {radnet_data.get('error')}")
    else:
        print("❌ Failed to retrieve data.")