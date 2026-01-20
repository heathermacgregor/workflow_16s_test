# NOT WORKING !!

import os
import requests
import json
from datetime import datetime

class BaseEnvironmentalAPI:
    """A base class for environmental API wrappers."""
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.session = requests.Session()
        if self.verbose:
            print(f"INFO: {self.__class__.__name__} initialized.")
import os

# This script helps you debug if your environment variable is set correctly.
api_key = os.getenv("OPENAQ_API_KEY")

if api_key:
    print("✅ SUCCESS: The API key was found by the script.")
    # The line below prints the first and last 4 chars to confirm it's the right key
    print(f"   Key snippet: {api_key[:4]}...{api_key[-4:]}")
else:
    print("❌ FAILURE: The script could not find the OPENAQ_API_KEY environment variable.")
class OpenAQAPI(BaseEnvironmentalAPI):
    """
    Fetches the latest air quality data from the nearest OpenAQ monitoring station using the v3 API.
    """
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        # CHANGED: Updated base URL to v3
        self.base_url = "https://api.openaq.org/v3/latest"
        self.api_key = os.getenv("OPENAQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("API key not found. Please set the OPENAQ_API_KEY environment variable.")
            
        self.session.headers.update({"X-API-Key": self.api_key})

    def get_data(self, lat: float, lon: float, radius_km: int = 10):
        """
        Finds the latest measurements from the single closest station within a radius.
        """
        params = {
            "coordinates": f"{lat},{lon}",
            "radius": radius_km * 1000, # API expects meters
            "limit": 1 # We only want the single closest station
        }
        
        print(f"INFO: Fetching latest air quality data near ({lat}, {lon}) using v3...")
        
        response = self.session.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if not data.get("results"):
            print(f"WARNING: No OpenAQ stations found within {radius_km} km.")
            return None
        
        closest_station = data["results"][0]
        
        measurements = []
        # CHANGED: The key is now 'parameters' instead of 'measurements'
        for reading in closest_station.get("parameters", []):
            measurements.append({
                "pollutant": reading["parameter"],
                "value": reading["value"],
                "unit": reading["unit"],
                "last_updated": reading["lastUpdated"]
            })

        return {
            "station_name": closest_station["name"],
            "distance_km": round(closest_station["distance"] / 1000, 2),
            "coordinates": closest_station["coordinates"],
            "measurements": measurements
        }

# --- Example Usage ---
if __name__ == "__main__":
    BERKELEY_COORDS = (37.8716, -122.2727)

    try:
        api = OpenAQAPI(verbose=True)
        
        air_quality_data = api.get_data(lat=BERKELEY_COORDS[0], lon=BERKELEY_COORDS[1])
        
        if air_quality_data:
            print("\n--- Latest Air Quality Data (OpenAQ v3) ---")
            print(f"Closest Station: {air_quality_data['station_name']} ({air_quality_data['distance_km']} km away)")
            print("\nMeasurements:")
            for m in air_quality_data["measurements"]:
                updated_time = datetime.fromisoformat(m['last_updated']).strftime('%Y-%m-%d %I:%M %p')
                print(f"  - {m['pollutant'].upper():<6}: {m['value']} {m['unit']} (at {updated_time})")
            print("-----------------------------------------")
            
    except ValueError as e:
        print(f"\nERROR: {e}")
    except requests.exceptions.RequestException as e:
        print(f"\nERROR: An API request failed: {e}")