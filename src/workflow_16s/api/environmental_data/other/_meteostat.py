# ==================================================================================== #

# Standard Imports
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

# Local Imports
from workflow_16s.api.environmental_data.other import (
    BaseEnvironmentalAPI, cache_api_call
)
from workflow_16s.utils.logger import get_logger

# ==================================================================================== #

logger = get_logger()

# ==================================================================================== #

class MeteostatAPI(BaseEnvironmentalAPI):
    """
    Fetches historical daily weather data from the Meteostat API.
    
    Documentation: https://dev.meteostat.net/python/daily.html
    
    Attributes:
        verbose (bool): If True, enables verbose logging.
    """
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        super().__init__(verbose=self.verbose)
    
    @cache_api_call
    def get_data( # type: ignore
        self, lat: float, lon: float, days: int = 30, fetch_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetches weather conditions for a location.
        
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
            
            # Prioritize fetching for a specific date if provided
            if fetch_date:
                try:
                    target_date = datetime.strptime(fetch_date, '%Y-%m-%d')
                    start = end = target_date
                    log_msg = f"⏳ Fetching Meteostat data for ({lat}, {lon}) on {fetch_date}..."
                except ValueError:
                    logger.error(f"🟥 Invalid date format for fetch_date. Please use 'YYYY-MM-DD'.")
                    return None
            else:
                # Fallback to original behavior (averaging over a period)
                end = datetime.now()
                start = end - timedelta(days=days)
                log_msg = f"⏳ Fetching Meteostat data for ({lat}, {lon}) over the last {days} days..."
            
            location = Point(lat, lon)
            if self.verbose: logger.info(log_msg)
            data = Daily(location, start, end).fetch()
            
            if data.empty:
                logger.warning("🟨 No Meteostat data returned for the specified location/date.")
                return None
                
            # .mean() works for both a single row (single day) and multiple rows (date range)
            mean_data = data.mean().to_dict()
            return {f"meteostat_{k}": v for k, v in mean_data.items()}
            
        except ImportError:
            logger.error("🟥 Meteostat package not installed. Install with: pip install meteostat")
            return None
        except Exception as e:
            logger.error(f"🟥 Meteostat API request failed: {e}")
            return None

# ==================================================================================== #

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