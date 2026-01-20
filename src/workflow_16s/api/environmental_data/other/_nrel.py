# ==================================================================================== #

# Standard Library Imports
import io
import json
import os
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Third-Party Imports
import pandas as pd

# Local Imports
from workflow_16s.api.environmental_data.other import (
    BaseEnvironmentalAPI, REQUEST_TIMEOUT, cache_api_call
)
from workflow_16s.utils.logger import get_logger

# ==================================================================================== #

logger = get_logger()

# ==================================================================================== #

class NREL_Solar_API(BaseEnvironmentalAPI):
    """
    Fetches historical solar irradiance data from the NREL NSRDB.
    
    Documentation: https://developer.nrel.gov/docs/solar/nsrdb/
    
    Attributes:
        verbose (bool): If True, enables verbose logging.
        api_key (str): NREL API key from environment variable NREL_API_KEY.
        email (str): Contact email from environment variable EMAIL.
    """
    URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-5min-download.csv"
    BATCH_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-batch-download.json"
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        super().__init__(verbose=self.verbose)
        self.single_location_url = self.URL
        self.batch_location_url = self.BATCH_URL
        self.api_key = os.getenv("NREL_API_KEY")
        self.email = os.getenv("EMAIL")

    def check_requirements(self) -> Tuple[bool, Optional[str]]:
        """Checks for NREL API key and email."""
        if not self.api_key or not self.email:
            return False, "NREL_API_KEY and EMAIL environment variables must be set."
        return True, None

    @cache_api_call
    def get_data( # type: ignore
        self, lat: float, lon: float, year: int = 2022, fetch_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves solar data for a single location.
        
        If 'fetch_date' is provided, it calculates the average for that day.
        Otherwise, it calculates the annual average for the specified 'year'.

        Args:
            lat: Latitude of the location.
            lon: Longitude of the location.
            year: Year of data. Used if 'fetch_date' is None. Default is 2022.
            fetch_date: Optional specific date 'YYYY-MM-DD'. Overrides 'year'.
            
        Returns:
            A dictionary with solar data or None if the request fails.
        """
        try:
            year_to_fetch = int(fetch_date[:4]) if fetch_date else year
            if not (2018 <= year_to_fetch <= 2022):
                return {"error": f"NREL solar data is not available for {year_to_fetch}. Valid years are 2018-2022."}
        except (ValueError, TypeError):
            return {"error": "Invalid fetch_date format provided."}
        
        # The API uses 'names' to specify the year for this endpoint
        params = {
            "api_key": self.api_key,
            "email": self.email,
            "wkt": f"POINT({lon} {lat})",
            "names": str(year_to_fetch), # Use the validated year variable
            "utc": "false",
            "leap_day": "false",
            "attributes": "ghi,dhi,dni,solar_zenith_angle",
        }

        try:
            if self.verbose: logger.info(f"⏳ Fetching NREL solar data for ({lat}, {lon}) for the year {year_to_fetch}...")
            response = self.session.get(
                self.single_location_url, params=params, timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()

            df = pd.read_csv(io.StringIO(response.text), skiprows=2)

            if fetch_date:
                # Filter for the specific day and calculate daily averages
                date_obj = datetime.strptime(fetch_date, '%Y-%m-%d')
                day_df = df[(df['Month'] == date_obj.month) & (df['Day'] == date_obj.day)]
                if day_df.empty:
                    result = {"error": f"No data found for date {fetch_date}"}
                else:
                    result = {
                        f"daily_avg_{c.lower()}_w_m2": round(day_df[c].mean(), 2) 
                        for c in ["GHI", "DHI", "DNI"]
                    }
                    # Convert date string to a numeric representation (YYYYMMDD)
                    result["date"] = float(fetch_date.replace('-', ''))
            else:
                # Calculate annual averages (original behavior)
                result = {
                    f"annual_avg_{c.lower()}_w_m2": round(df[c].mean(), 2) 
                    for c in ["GHI", "DHI", "DNI"]
                }
                result["year"] = year_to_fetch

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"🟥 NREL Solar API request failed: {e}")
            return None
        except (pd.errors.ParserError, KeyError) as e:
            logger.error(f"🟥 Failed to parse NREL data: {e}")
            return None

    def get_batch_data(
        self, locations: List[Tuple[float, float]], year: int = 2022
    ) -> Dict[Tuple[float, float], Optional[Dict[str, Any]]]:
        """
        Retrieves solar data for multiple locations in a single batch request.
        (This method remains unchanged)
        """
        if not locations: return {}
        if not self.api_key or not self.email: return {loc: None for loc in locations}
        
        chunk_size = 100
        location_chunks = [
            locations[i:i + chunk_size] 
            for i in range(0, len(locations), chunk_size)
        ]
        
        all_results = {}
        for i, chunk in enumerate(location_chunks):
            if self.verbose: logger.info(f"⏳ Submitting NREL batch request {i+1} of {len(location_chunks)} for {len(chunk)} locations...")

            payload = {
                "api_key": self.api_key, "email": self.email,
                "wkt": ",".join([f"POINT({lon} {lat})" for lat, lon in chunk]),
                "names": year, "utc": "false", "attributes": "ghi,dhi,dni"
            }

            try:
                response = self.session.post(
                    self.batch_location_url, json=payload, 
                    timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()
                response_data = response.json()
                
                if "errors" in response_data and response_data["errors"]:
                    logger.error(f"🟥 NREL batch API returned errors: {response_data['errors']}")
                    for loc in chunk: all_results[loc] = {"error": response_data["errors"]}
                    continue

                location_results = response_data.get("outputs", [])
                for j, loc_data in enumerate(location_results):
                    loc_tuple = chunk[j]
                    solar_averages = {
                        "avg_ghi_w_m2": loc_data.get("avg_ghi"),
                        "avg_dhi_w_m2": loc_data.get("avg_dhi"),
                        "avg_dni_w_m2": loc_data.get("avg_dni")
                    }
                    all_results[loc_tuple] = solar_averages
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"🟥 NREL batch API request failed: {e}")
                for loc in chunk: all_results[loc] = {"error": str(e)}

        return all_results

# ==================================================================================== #

if __name__ == "__main__":
    
    print("--- Running NREL Solar API Example Test ---")

    # 1. Instantiate the API client with verbose logging
    api_client = NREL_Solar_API(verbose=True)

    # 2. Check for required API keys before proceeding
    ok, reason = api_client.check_requirements()
    if not ok:
        print("\n" + "="*50)
        print(f"‼️ ERROR: {reason}")
        print("You can get a key from: https://developer.nrel.gov/signup/")
        print("="*50)
    else:
        # Berkeley, CA coordinates ☀️
        berkeley_lat = 37.8715
        berkeley_lon = -122.2730

        # Example 1: Get ANNUAL average solar data for 2022
        print("\n" + "="*50)
        print("## Test 1: Fetching ANNUAL average solar data for 2022...")
        print("="*50)
        annual_data = api_client.get_data(
            lat=berkeley_lat,
            lon=berkeley_lon,
            year=2022
        )
        if annual_data:
            print("✅ Success! Found annual data:")
            print(json.dumps(annual_data, indent=2))
        else:
            print("❌ Failed to retrieve data.")

        # Example 2: Get DAILY average solar data for a specific date
        print("\n" + "="*50)
        print("## Test 2: Fetching DAILY average solar data for 2022-09-30...")
        print("="*50)
        daily_data = api_client.get_data(
            lat=berkeley_lat,
            lon=berkeley_lon,
            fetch_date="2022-09-30" # Using the new date filter
        )
        if daily_data:
            print("✅ Success! Found daily data:")
            print(json.dumps(daily_data, indent=2))
        else:
            print("❌ Failed to retrieve data for the specified date.")