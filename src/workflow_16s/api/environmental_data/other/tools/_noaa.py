# workflow_16s/api/environmental_data/other/tools/_noaa.py

import json
import requests
import statistics as stats_module
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from .base import BaseEnvironmentalAPI, REQUEST_TIMEOUT
from .cache import cache_api_call
from .constants import CACHE_DIR, is_us_location
from workflow_16s.utils.logger import get_logger, with_logger

# Module-level logger for use in all class methods
logger = get_logger(__name__)

@with_logger
class NOAA_Tides_API(BaseEnvironmentalAPI):
    """
    Fetches oceanographic data from the closest NOAA station.
    
    Documentation: https://api.tidesandcurrents.noaa.gov/api/prod/
    
    Attributes:
        verbose (bool): If True, enables verbose logging.
        base_url (str): Base URL for the NOAA Tides and Currents API.
        stations_file (Path): Path to the cached stations file.
        stations (list): List of station data loaded from the stations file.
    """
    URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    STATIONS_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        super().__init__(verbose=self.verbose)
        self.base_url = self.URL
        # Use the central cache directory for the stations file
        self.stations_file = CACHE_DIR / "noaa_stations.json"
        self.stations = None
    
    def _haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculates the distance between two lat/lon points in kilometers."""
        import math
        R = 6371
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        lat1, lat2 = math.radians(lat1), math.radians(lat2)
        a = math.sin(dLat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    
    def _load_stations(self) -> None:
        """Loads station data from a local file or downloads it if it doesn't exist."""
        if self.stations is not None: return
            
        if self.stations_file.exists():
            try:
                with open(self.stations_file, 'r') as f:
                    data = json.load(f)
                    self.stations = data.get("stations", [])
                if self.verbose: logger.info(f"🟩 Loaded {len(self.stations)} NOAA stations from cache")
                return
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"🟨 Failed to load NOAA station cache: {e}")
        
        if self.verbose: logger.info("⏳ Downloading NOAA station list...")
        try:
            response = self.session.get(self.STATIONS_URL, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            with open(self.stations_file, 'w') as f: json.dump(data, f)
            self.stations = data.get("stations", [])
            if self.verbose: logger.info(f"🟩 Downloaded {len(self.stations)} NOAA stations")
        except requests.exceptions.RequestException as e:
            logger.error(f"🟥 Failed to download NOAA stations: {e}")
            self.stations = []
    
    def find_closest_station(self, lat: float, lon: float) -> Tuple[Optional[Dict], float]:
        """Finds the closest station to a given lat/lon from the master list."""
        self._load_stations()
        if not self.stations: return None, float('inf')
        
        closest_station, min_distance = None, float('inf')
        for station in self.stations:
            try:
                station_lat, station_lon = float(station['lat']), float(station['lng'])
                distance = self._haversine_km(lat, lon, station_lat, station_lon)
                if distance < min_distance:
                    min_distance, closest_station = distance, station
            except (ValueError, KeyError): continue
        return closest_station, min_distance
    
    @cache_api_call
    def get_data( # type: ignore
        self, lat: float, lon: float, max_distance_km: int = 25,
        fetch_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Gets data for the closest station if it's within the distance threshold.

        Args:
            lat:             Latitude of the location.
            lon:             Longitude of the location.
            max_distance_km: Maximum distance in km to consider a station "close".
            fetch_date:      Optional date 'YYYY-MM-DD' to fetch historical data. If None, gets latest data.

        Returns:
            A dictionary with station data or an error message.
        """
        # US-only dataset: validate location is within US service area (includes coastal monitoring)
        if not is_us_location(lat, lon):
            logger.debug(f"NOAA Tides: Location ({lat:.4f}, {lon:.4f}) outside US service area")
            return None

        closest_station, distance = self.find_closest_station(lat, lon)
        if not closest_station: return {"error": "Could not find any NOAA stations."}
        
        if distance > max_distance_km:
            return {"error": (f"Closest station '{closest_station['name']}' is {distance:.2f} km away, beyond the {max_distance_km} km threshold.")}
        
        station_id = closest_station['id']
        products = ["water_level", "water_temperature", "wind", 
                    "air_temperature", "air_pressure"]
        results = {
            "station_name": closest_station['name'], "station_id": station_id,
            "distance_km": round(distance, 2), "measurements": {}
        }
        
        logger.info(f"⏳ Fetching NOAA data for station '{closest_station['name']}'...")
        
        for product in products:
            params = {
                "application": "environmental_data_script", "format": "json", 
                "station": station_id, "product": product, "time_zone": "lst_ldt", 
                "units": "english"
            }
            if product == 'water_level': params['datum'] = 'MLLW'

            if fetch_date:
                try:
                    # Format date from YYYY-MM-DD to YYYYMMDD and specify a 24-hour range
                    date_obj = datetime.strptime(fetch_date, '%Y-%m-%d')
                    params["begin_date"] = date_obj.strftime('%Y%m%d')
                    params["end_date"] = date_obj.strftime('%Y%m%d')
                except ValueError:
                    logger.error(f"🟥 Invalid date format for fetch_date. Please use 'YYYY-MM-DD'.")
                    continue
            else:
                params["date"] = "latest"

            try:
                response = self.session.get(self.base_url, params=params, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                
                if "error" in data or not data.get("data"): continue
                
                key = product.replace("_", " ").title()
                unit = {"water_level": "ft", "water_temperature": "°F", 
                        "air_temperature": "°F", "air_pressure": "mb"}.get(product, "")

                if fetch_date: # Handle historical data (average values)
                    if product == "wind": # For wind, just take the first reading of the day
                        measurement = data["data"][0]
                        results["measurements"][key] = f"{measurement.get('s')} knots from {measurement.get('d')}°"
                    else: # For other products, average the numeric values
                        values = [float(d['v']) for d in data['data'] if 'v' in d and d['v']]
                        if not values: continue
                        avg_value = round(stats_module.mean(values), 2)
                        results["measurements"][key] = f"{avg_value} {unit}".strip()
                    results["observation_date"] = fetch_date
                else: # Handle "latest" data
                    measurement = data["data"][0]
                    value = measurement.get("v")
                    if product == "wind":
                        results["measurements"][key] = f"{measurement.get('s')} knots from {measurement.get('d')}°"
                    else:
                        results["measurements"][key] = f"{value} {unit}".strip()
                    results["last_updated"] = measurement.get("t")

            except (requests.exceptions.RequestException, KeyError, IndexError, json.JSONDecodeError):
                continue
        
        return results


if __name__ == "__main__":
    
    # --- Example Usage ---
    print("--- Running NOAA Tides API Example Test ---")

    # Berkeley Marina coordinates 🌊
    berkeley_lat = 37.866
    berkeley_lon = -122.316

    # 1. Instantiate the API client with verbose logging
    api_client = NOAA_Tides_API(verbose=True)

    # 2. Example 1: Get the LATEST available data
    print("\n" + "="*50)
    print("## Test 1: Fetching LATEST oceanographic data...")
    print("="*50)
    latest_data = api_client.get_data(lat=berkeley_lat, lon=berkeley_lon)
    if latest_data:
        print("✅ Success! Found latest data:")
        print(json.dumps(latest_data, indent=2))
    else:
        print("❌ Failed to retrieve data.")

    # 3. Example 2: Get historical data for a specific date
    print("\n" + "="*50)
    print("## Test 2: Fetching historical data for 2025-09-29...")
    print("="*50)
    historical_data = api_client.get_data(
        lat=berkeley_lat,
        lon=berkeley_lon,
        fetch_date="2025-09-29" # Using the new date filter
    )
    if historical_data:
        print("✅ Success! Found historical data:")
        print(json.dumps(historical_data, indent=2))
    else:
        print("❌ Failed to retrieve data for the specified date.")