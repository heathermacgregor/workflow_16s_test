# workflow_16s/api/environmental_data/other/tools/_openmeteo.py

import json
import requests
import statistics as stats_module
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from .base import BaseEnvironmentalAPI, REQUEST_TIMEOUT
from .cache import cache_api_call
from workflow_16s.utils.logger import get_logger, with_logger

# Module-level logger for use in all class methods
logger = get_logger(__name__)

# OpenMeteo air quality API date range (approximately)
AIR_QUALITY_START_DATE = datetime(2022, 1, 1)  # Air quality data starts ~2022

class EnvironmentalHealthAPI(BaseEnvironmentalAPI):
    """Fetches air quality and pollen data from Open-Meteo APIs.
    
    Air Quality (PM10, PM2.5, Ozone):
    - Available: Current + historical from ~2010
    - Source: Copernicus Atmosphere Data Store
    - Coverage: Global
    
    Pollen (Tree, Grass, Weed):
    - Available: Current + historical from 2020-01-01 onwards
    - Source: European Mold Forecast
    - Coverage: Mainly Europe, limited global coverage
    - NOTE: For dates before 2020, pollen data unavailable (expected)
    
    Both endpoints may return no data for:
    - Very recent dates (< 2 days old, still processing)
    - Dates outside their respective availability ranges
    - Geographic regions with sparse coverage
    """
    AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.air_quality_url = self.AIR_QUALITY_URL
        self.forecast_url = self.FORECAST_URL
        # logger is available via module-level definition
    
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
        """Retrieves current or historical data for a single location.
        
        Note: Air quality historical data only available from ~2022-01-01 onwards.
        Older dates will skip air quality but may still fetch current conditions or pollen.
        """
        combined_forecast = {}
        
        # Define common parameters
        base_params = {"latitude": lat, "longitude": lon, "timezone": "auto"}
        if fetch_date:
            base_params["start_date"] = fetch_date
            base_params["end_date"] = fetch_date

        # Get Pollutants - with date range validation
        air_quality_available = True
        if fetch_date:
            # Check if fetch_date is within API's supported range
            try:
                fetch_datetime = datetime.strptime(fetch_date, '%Y-%m-%d')
                if fetch_datetime < AIR_QUALITY_START_DATE:
                    logger.debug(f"Air quality data not available for {fetch_date} (before {AIR_QUALITY_START_DATE.strftime('%Y-%m-%d')}). Will skip air quality request.")
                    air_quality_available = False
            except ValueError:
                logger.debug(f"Invalid date format for air quality check: {fetch_date}")
                air_quality_available = False
        
        if air_quality_available:
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
            except requests.exceptions.HTTPError as e:
                # Even with date validation, some dates near boundary may fail
                if e.response.status_code == 400 and fetch_date:
                    logger.debug(f"Air quality API returned 400 for {fetch_date} (date outside supported range). Skipping air quality.")
                else:
                    logger.error(f"🟥 Air quality API request failed: {e}")
            except requests.exceptions.RequestException as e:
                logger.error(f"🟥 Air quality API request failed: {e}")
        else:
            logger.debug(f"Air quality data skipped for {fetch_date} (outside supported range)")
        
        # Get Pollen - with comprehensive date range validation
        try:
            pollen_params = {
                **base_params, "hourly": "tree_pollen,grass_pollen,weed_pollen"
            }
            pollen_url = self.ARCHIVE_URL if fetch_date else self.forecast_url
            
            # Validate date is within pollen data availability
            if fetch_date and pollen_url == self.ARCHIVE_URL:
                try:
                    fetch_datetime = datetime.strptime(fetch_date, '%Y-%m-%d')
                    
                    # OpenMeteo pollen archive constraints:
                    # - Available from: ~2020-01-01 onwards
                    # - Data lag: Recent data (< 2 days) may not be available
                    min_pollen_date = datetime(2020, 1, 1)
                    days_ago = (datetime.now() - fetch_datetime).days
                    
                    # Check if date is too old
                    if fetch_datetime < min_pollen_date:
                        logger.info(
                            f"Pollen: Data unavailable for {fetch_date} - "
                            f"before pollen archive start date (2020-01-01). "
                            f"Pollen records are only available from 2020 onwards."
                        )
                        # Skip pollen for this date (don't query API)
                        # Return air quality data if available
                        return combined_forecast
                    
                    # Check if date is too recent (not yet processed)
                    if days_ago < 2:
                        logger.debug(
                            f"Pollen: {fetch_date} too recent "
                            f"(only {days_ago} days old, needs 2+ days processing)"
                        )
                        # Skip pollen for very recent dates
                        return combined_forecast
                except ValueError:
                    logger.debug(f"Invalid date format: {fetch_date}")
                    return combined_forecast
            
            # Attempt pollen query
            response = self.session.get(
                pollen_url,
                params=pollen_params,
                timeout=REQUEST_TIMEOUT
            )
            
            # Evaluate response
            if response.status_code == 400:
                # 400 usually means date is outside available range
                logger.info(
                    f"Pollen: 400 Bad Request for {fetch_date} - "
                    f"date may be outside available range (2020-01-01 onwards)"
                )
                # Return without pollen data
                return combined_forecast
            
            response.raise_for_status()
            data = response.json()
            
            if fetch_date:
                combined_forecast.update(self._get_daily_average(data, fetch_date))
            else:
                combined_forecast.update(self._get_current_value(data))
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400 and fetch_date:
                logger.info(
                    f"Pollen: Date {fetch_date} outside available range. "
                    f"OpenMeteo pollen archive starts 2020-01-01."
                )
            else:
                logger.warning(f"Pollen API request failed: {e}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Pollen API request failed: {e}")
            
        return combined_forecast

@with_logger
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
            params["hourly"] = soil_vars
            params["start_date"] = fetch_date
            params["end_date"] = fetch_date
            data_key, units_key = "hourly", "hourly_units"
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
                    if data_key == "hourly" and isinstance(value, list):
                        valid_values = [v for v in value if v is not None]
                        if not valid_values:
                            continue
                        val = round(stats_module.mean(valid_values), 2)
                    else:
                        # Current data comes back as a single float/int
                        val = value
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