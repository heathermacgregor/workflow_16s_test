# ==================================================================================== #

# Standard Imports
import json
import requests
import statistics as stats_module
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

# Local Imports
from workflow_16s.api.environmental_data.other import (
    BaseEnvironmentalAPI, REQUEST_TIMEOUT, cache_api_call
)
from workflow_16s.utils.logger import get_logger

# ==================================================================================== #

logger = get_logger()

# ==================================================================================== #

class EnvironmentalHealthAPI(BaseEnvironmentalAPI):
    """Fetches air quality and pollen data from Open-Meteo APIs."""
    AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.air_quality_url = self.AIR_QUALITY_URL
        self.forecast_url = self.FORECAST_URL
    
    def _get_current_value(self, data: dict) -> Dict[str, str]:
        """Helper to find the current hour's value from an API response."""
        if "hourly" not in data or "time" not in data["hourly"]: return {}
        
        now = datetime.now()
        current_hour_str = now.strftime('%Y-%m-%dT%H:00')
        
        try: current_index = data["hourly"]["time"].index(current_hour_str)
        except ValueError: current_index = 0
        
        units = data.get("hourly_units", {})
        current_values = {}
        for key, values in data["hourly"].items():
            if key != "time" and isinstance(values, list) and current_index < len(values):
                current_values[key] = f"{values[current_index]} {units.get(key, '')}".strip()
        
        return current_values

    def _get_daily_average(self, data: dict, date: str) -> Dict[str, str]:
        """Helper to average all hourly values for a given day."""
        if "hourly" not in data: return {}

        units = data.get("hourly_units", {})
        daily_averages = {}
        for key, values in data["hourly"].items():
            if key != "time" and isinstance(values, list) and values:
                # Filter out None values before calculating mean
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    avg = round(stats_module.mean(valid_values), 2)
                    daily_averages[f"daily_avg_{key}"] = f"{avg} {units.get(key, '')}".strip()
        
        if daily_averages: daily_averages["date"] = date
        return daily_averages

    @cache_api_call
    def get_data( # type: ignore
        self, lat: float, lon: float, fetch_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieves current or historical data for a single location."""
        combined_forecast = {}
        
        # Define common parameters
        base_params = {"latitude": lat, "longitude": lon, "timezone": "auto"}
        if fetch_date:
            base_params["start_date"] = fetch_date
            base_params["end_date"] = fetch_date

        # Get Pollutants
        try:
            pollutant_params = {**base_params, "hourly": "pm10,pm2_5,ozone"}
            response = self.session.get(self.air_quality_url, params=pollutant_params, 
                                        timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if fetch_date:
                combined_forecast.update(self._get_daily_average(data, fetch_date))
            else:
                combined_forecast.update(self._get_current_value(data))
        except requests.exceptions.RequestException as e:
            logger.error(f"🟥 Air quality API request failed: {e}")
        
        # Get Pollen
        try:
            pollen_params = {
                **base_params, "hourly": "tree_pollen,grass_pollen,weed_pollen"
            }
            pollen_url = self.ARCHIVE_URL if fetch_date else self.forecast_url
            response = self.session.get(pollen_url, params=pollen_params,
                                        timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if fetch_date:
                combined_forecast.update(self._get_daily_average(data, fetch_date))
            else:
                combined_forecast.update(self._get_current_value(data))
        except requests.exceptions.RequestException as e:
            logger.error(f"🟥 Pollen API request failed: {e}")
            
        return combined_forecast


class SoilStateAPI(BaseEnvironmentalAPI):
    """Fetches current or historical soil temperature and moisture data."""
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    @cache_api_call
    def get_data( # type: ignore
        self, lat: float, lon: float, fetch_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieves current or daily average soil data for a location."""
        params = {"latitude": lat, "longitude": lon, "timezone": "auto"}
        
        soil_vars = (
            "soil_temperature_0cm,soil_temperature_6cm,soil_temperature_18cm,"
            "soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,soil_moisture_3_to_9cm"
        )

        if fetch_date:
            params["daily"] = soil_vars
            params["start_date"] = fetch_date
            params["end_date"] = fetch_date
            data_key, units_key = "daily", "daily_units"
        else:
            params["current"] = soil_vars
            data_key, units_key = "current", "current_units"

        try:
            response = self.session.get(self.base_url, params=params, 
                                        timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            if data_key not in data or units_key not in data: return None
            
            source_data, units = data[data_key], data[units_key]
            results = {}
            for key, value in source_data.items():
                if key not in ["time", "interval"]:
                    # For daily data, the value is a list with one item
                    val = value[0] if isinstance(value, list) else value
                    friendly_name = key.replace("_", " ").replace(" cm", "cm").capitalize()
                    results[friendly_name] = f"{val} {units.get(key, '')}".strip()
            return results
        except requests.exceptions.RequestException as e:
            logger.error(f"🟥 Soil state API request failed: {e}")
            return None


class OpenMeteoAPI(BaseEnvironmentalAPI):
    """Fetches historical daily weather data from the Open-Meteo Archive API."""
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    @cache_api_call
    def get_data( # type: ignore
        self, lat: float, lon: float, days: int = 30, fetch_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieves historical weather data for a date or period."""
        if fetch_date:
            start_date = end_date = fetch_date
        else:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        params = {
            "latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date, 
            "daily": ("temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum"), 
            "timezone": "auto"
        }
        try:
            response = self.session.get(self.base_url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            return data.get('daily')
        except requests.exceptions.RequestException as e:
            logger.error(f"🟥 OpenMeteo API request failed: {e}")
            return None

# ==================================================================================== #

if __name__ == "__main__":
    
    print("--- Running Open-Meteo APIs Example Test ---")
    
    # Berkeley, CA coordinates
    berkeley_lat = 37.8715
    berkeley_lon = -122.2730
    test_date = "2025-09-29"

    # --- 1. Environmental Health API ---
    print("\n" + "="*50)
    print("## Test 1: Environmental Health API")
    print("="*50)
    health_client = EnvironmentalHealthAPI(verbose=True)
    
    print("\n---\n▶️ Test 1a: Fetching CURRENT air quality and pollen...\n---")
    current_health_data = health_client.get_data(berkeley_lat, berkeley_lon)
    print("✅ Result:", json.dumps(current_health_data, indent=2))

    print(f"\n---\n▶️ Test 1b: Fetching DAILY AVERAGE air quality and pollen for {test_date}...\n---")
    historical_health_data = health_client.get_data(berkeley_lat, berkeley_lon, fetch_date=test_date)
    print("✅ Result:", json.dumps(historical_health_data, indent=2))
    
    # --- 2. Soil State API ---
    print("\n" + "="*50)
    print("## Test 2: Soil State API")
    print("="*50)
    soil_client = SoilStateAPI(verbose=True)

    print("\n---\n▶️ Test 2a: Fetching CURRENT soil state...\n---")
    current_soil_data = soil_client.get_data(berkeley_lat, berkeley_lon)
    print("✅ Result:", json.dumps(current_soil_data, indent=2))

    print(f"\n---\n▶️ Test 2b: Fetching DAILY AVERAGE soil state for {test_date}...\n---")
    historical_soil_data = soil_client.get_data(berkeley_lat, berkeley_lon, fetch_date=test_date)
    print("✅ Result:", json.dumps(historical_soil_data, indent=2))
    
    # --- 3. OpenMeteo Historical API ---
    print("\n" + "="*50)
    print("## Test 3: OpenMeteo Historical API")
    print("="*50)
    om_client = OpenMeteoAPI(verbose=True)

    print("\n---\n▶️ Test 3a: Fetching weather for the LAST 7 DAYS...\n---")
    recent_weather_data = om_client.get_data(berkeley_lat, berkeley_lon, days=7)
    print("✅ Result:", json.dumps(recent_weather_data, indent=2))

    print(f"\n---\n▶️ Test 3b: Fetching weather for a SINGLE DAY ({test_date})...\n---")
    single_day_weather_data = om_client.get_data(berkeley_lat, berkeley_lon, fetch_date=test_date)
    print("✅ Result:", json.dumps(single_day_weather_data, indent=2))