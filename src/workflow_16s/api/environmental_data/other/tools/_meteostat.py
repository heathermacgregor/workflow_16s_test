# workflow_16s/api/environmental_data/other/tools/_meteostat.py

import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from .base import BaseEnvironmentalAPI, REQUEST_TIMEOUT
from .cache import cache_api_call
from workflow_16s.utils.logger import get_logger, with_logger

@with_logger
class MeteostatAPI(BaseEnvironmentalAPI):
    """
    Fetches historical daily weather data from the Meteostat API.
    
    NOTE: Meteostat is a sparse weather station network.
    Coverage is good in developed countries but sparse in:
    - Remote areas
    - Tropical regions
    - Polar regions  
    - Ocean areas
    
    Documentation: https://dev.meteostat.net/python/daily.html
    
    Attributes:
        verbose (bool): If True, enables verbose logging.
    """
    def __init__(self, verbose: bool = False, fallback_to_openmeteo: bool = False, **kwargs):
        self.verbose = verbose
        self.fallback_to_openmeteo = fallback_to_openmeteo
        super().__init__(verbose=self.verbose)
        self.logger = kwargs.get('logger') or get_logger("workflow_16s")

    @cache_api_call
    def get_data( 
        self, lat: float, lon: float, days: int = 30, 
        fetch_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetches weather conditions for a location.
        
        NOTE: Meteostat is a sparse weather station network.
        Coverage is good in developed countries but sparse in:
        - Remote areas
        - Tropical regions
        - Polar regions
        - Ocean areas
        
        If no data found, returns None. Consider OpenMeteo as alternative
        for global gridded data coverage.
        
        Fetches data for a specific day if 'fetch_date' is provided, otherwise
        calculates the average over a historical period defined by 'days'.
        
        Args:
            lat (float): Latitude of the location.
            lon (float): Longitude of the location.
            days (int): Number of past days to average over. Used if 'fetch_date' is None.
            fetch_date (Optional[str]): A specific date 'YYYY-MM-DD' to fetch data for. Overrides 'days'.

        Returns:
            Optional[Dict[str, Any]]: A dictionary of weather data or None on failure.
        """
        try:
            from meteostat import Point, Daily
            Daily.max_age = 0

            # Prioritize fetching for a specific date if provided
            if fetch_date:
                # Early check for invalid/missing dates (silent, no logging)
                import pandas as pd
                if fetch_date is None or pd.isna(fetch_date) or str(fetch_date).lower() in ['nan', 'nat', '']:
                    return {}

                try:
                    target_date = datetime.strptime(fetch_date, '%Y-%m-%d')
                    start = end = target_date
                    log_msg = f"⏳ Fetching Meteostat data for ({lat}, {lon}) on {fetch_date}..."
                except ValueError:
                    # Invalid date format - return silently, no error log
                    return {}
            else:
                # Fallback to original behavior (averaging over a period)
                end = datetime.now()
                start = end - timedelta(days=days)
                log_msg = f"⏳ Fetching Meteostat data for ({lat}, {lon}) over the last {days} days..."
            
            # Defensive: Convert coordinates to float (handle string inputs)
            try:
                lat = float(lat)
                lon = float(lon)
            except (ValueError, TypeError):
                self.logger.debug(f"⚠️ Invalid coordinates: lat={lat}, lon={lon}. Skipping.")
                return None
            
            # Validate coordinates are reasonable
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                self.logger.debug(f"⚠️ Coordinates out of bounds: lat={lat}, lon={lon}. Skipping.")
                return None
            
            location = Point(lat, lon)
            if self.verbose: self.logger.info(log_msg)
            try:
                data = Daily(location, start, end).fetch()
            except Exception as inner_e:
                if "attribute '_types'" in str(inner_e):
                    # Fallback for older versions if the first call fails
                    #self.logger.warning("⚠️ Meteostat internal error detected. Attempting direct API fallback...")
                    return None
                raise inner_e
            
            if data.empty:
                self.logger.info(
                    f"Meteostat: No weather station data at ({lat:.4f}, {lon:.4f}) - "
                    f"sparse station coverage"
                )
                self.logger.debug(
                    "Note: Remote/ocean areas may lack nearby weather stations. "
                    "Consider OpenMeteo for gridded global data."
                )
                return None
                
            # .mean() works for both a single row (single day) and multiple rows (date range)
            mean_data = data.mean().to_dict()
            return {f"meteostat_{k}": v for k, v in mean_data.items()}
            
        except ImportError:
            self.logger.error("🟥 Meteostat package not installed. Install with: pip install meteostat")
            return None
        except Exception as e:
            if not '_daily' in str(e).lower():  # Avoid logging expected "No data" exceptions as errors
                self.logger.error(f"🟥 Meteostat API request failed: {e}")
            return None


if __name__ == "__main__":
    
    # --- Example Usage ---
    print("--- Running Meteostat API Example Test ---")

    # Berkeley, CA coordinates ☀️
    berkeley_lat = 37.8715
    berkeley_lon = -122.2730

    # 1. Instantiate the API client with verbose logging
    api_client = MeteostatAPI(verbose=True)

    # 2. Example 1: Get average weather over the last 7 days (default behavior)
    print("\n" + "="*50)
    print("## Test 1: Fetching average weather for the last 7 days...")
    print("="*50)
    average_data = api_client.get_data(
        lat=berkeley_lat,
        lon=berkeley_lon,
        days=7
    )
    if average_data:
        print("✅ Success! Found average weather data:")
        print(json.dumps(average_data, indent=2))
    else:
        print("❌ Failed to retrieve data.")

    # 3. Example 2: Get weather for a specific historical date
    print("\n" + "="*50)
    print("## Test 2: Fetching weather for a specific date (2025-09-29)...")
    print("="*50)
    specific_date_data = api_client.get_data(
        lat=berkeley_lat,
        lon=berkeley_lon,
        fetch_date="2025-09-29" # Using the new date filter
    )
    if specific_date_data:
        print("✅ Success! Found weather data for the specified date:")
        print(json.dumps(specific_date_data, indent=2))
    else:
        print("❌ Failed to retrieve data for the specified date.")