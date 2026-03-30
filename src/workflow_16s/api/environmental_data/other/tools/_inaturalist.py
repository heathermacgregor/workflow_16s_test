# workflow_16s/api/environmental_data/other/tools/_inaturalist.py

import requests
import json 
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from .base import BaseEnvironmentalAPI, REQUEST_TIMEOUT
from .cache import cache_api_call
from workflow_16s.utils.logger import get_logger, with_logger

load_dotenv()

# Module-level logger for use in all class methods
logger = get_logger(__name__)

@with_logger
class iNaturalistAPI(BaseEnvironmentalAPI):
    """
    Fetches recent biodiversity observations from the iNaturalist API.
    
    Documentation: https://api.inaturalist.org/v1/docs/
    
    Attributes:
        base_url (str): Base URL for the iNaturalist API.
        verbose (bool): If True, enables verbose logging.
    """
    URL = "https://api.inaturalist.org/v1/observations"
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.base_url = self.URL
        self.verbose = verbose
    
    @cache_api_call
    def get_data( 
        self, lat: float, lon: float, radius_km: int = 10, 
        taxon_id: Optional[int] = None, fetch_date: Optional[str] = None,
        limit: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves iNaturalist observations near a location, with an option to filter by date.
        
        Args:
            lat:         Latitude of the location.
            lon:         Longitude of the location. 
            radius_km:   Search radius in kilometers. Default is 10 km.
            taxon_id:    Optional iNaturalist taxon ID to filter observations.
            fetch_date:  Optional date to filter observations (e.g., 'YYYY-MM-DD').
            limit:       Maximum number of observations to retrieve. Default is 5.
        
        Returns:
            Optional[Dict[str, Any]]: A dictionary containing observation data or 
            None if the request fails.
        """
        params = {
            "lat": lat, "lng": lon, "radius": radius_km, "order": "desc",
            "order_by": "observed_on", "per_page": limit
        }
        
        if taxon_id: 
            params["taxon_id"] = taxon_id
        if fetch_date: 
            params["observed_on"] = fetch_date
        
        try:
            if self.verbose: logger.info(f"⏳ Querying iNaturalist for observations near ({lat}, {lon})...")
            response = self.session.get(
                self.base_url, params=params, timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
            observations = []
            for obs in data.get("results", [])[:limit]:
                # Safely extract species and photo
                species = obs.get("species_guess", "Unknown")
                photos = obs.get("photos", [])
                # Handle both missing photos list and empty photos list
                photo = photos[0] if photos and len(photos) > 0 else {}
                
                observations.append({
                    "species": species,
                    "observed_on": obs.get("observed_on_string", "Unknown"),
                    "user": obs.get("user", {}).get("login", "anonymous"),
                    "photo_url": photo.get("url", ""),
                    "inaturalist_url": obs.get("uri", "")
                })
                time.sleep(1)
            
            return {"observations": observations, "count": len(observations)}
        except requests.exceptions.RequestException as e:
            logger.error(f"🟥 iNaturalist API request failed: {e}")
            return None


if __name__ == "__main__":
    
    # --- Example Usage ---
    print("--- Running iNaturalist API Example Test ---")

    # Berkeley, CA coordinates 🐝
    berkeley_lat = 37.8715
    berkeley_lon = -122.2730

    # 1. Instantiate the API client with verbose logging
    api_client = iNaturalistAPI(verbose=True)

    # 2. Example 1: Get the 2 most recent observations without a date filter
    print("\n" + "="*50)
    print("## Test 1: Fetching recent observations (no date filter)...")
    print("="*50)
    recent_data = api_client.get_data(
        lat=berkeley_lat,
        lon=berkeley_lon,
        radius_km=5,
        limit=2
    )
    if recent_data:
        print("✅ Success! Found observations:")
        # Pretty-print the JSON response
        print(json.dumps(recent_data, indent=2))
    else:
        print("❌ Failed to retrieve data.")

    # 3. Example 2: Get observations from a specific date
    print("\n" + "="*50)
    print("## Test 2: Fetching observations from 2025-09-29...")
    print("="*50)
    specific_date_data = api_client.get_data(
        lat=berkeley_lat,
        lon=berkeley_lon,
        radius_km=5,
        fetch_date="2025-09-29", # Filtering by a specific date
        limit=2
    )
    if specific_date_data:
        print("✅ Success! Found observations for the specified date:")
        print(json.dumps(specific_date_data, indent=2))
    else:
        print("❌ Failed to retrieve data for the specified date.")