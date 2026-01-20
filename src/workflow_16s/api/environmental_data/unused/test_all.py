#!/usr/bin/env python3
"""
Environmental Data Aggregation Script

This script integrates multiple environmental APIs to collect comprehensive
environmental data for a given location. It's designed to be production-ready
with proper error handling, logging, and configuration management.
"""

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_EXPIRY_HOURS = 24
REQUEST_TIMEOUT = 30
MAX_WORKERS = 5

from workflow_16s.api.environmental_data import BaseEnvironmentalAPI
from workflow_16s.src.workflow_16s.api.environmental_data._inaturalist import iNaturalistAPI
from workflow_16s.src.workflow_16s.api.environmental_data._meteostat import MeteostatAPI
from workflow_16s.src.workflow_16s.api.environmental_data._nws import NWS_API, SevereWeatherAPI
from workflow_16s.src.workflow_16s.api.environmental_data._noaa import NOAA_Tides_API
from workflow_16s.src.workflow_16s.api.environmental_data._nrel import NREL_Solar_API
from workflow_16s.src.workflow_16s.api.environmental_data._openmeteo import EnvironmentalHealthAPI, OpenMeteoAPI, SoilStateAPI, GeocodingAPI
from workflow_16s.api.environmental_data.soilgrids import SoilGridsAPI
from workflow_16s.src.workflow_16s.api.environmental_data._usgs import USGS_Earthquake_API


def get_all_environmental_data(lat: float, lon: float, 
                              location_name: str = "") -> Dict[str, Any]:
    """
    Fetches data from all environmental APIs for a given location.
    
    Args:
        lat: Latitude of the location
        lon: Longitude of the location
        location_name: Name of the location (for display purposes)
    
    Returns:
        Dictionary containing all collected environmental data
    """
    results = {
        "location": {
            "name": location_name,
            "latitude": lat,
            "longitude": lon,
            "timestamp": datetime.now().isoformat()
        },
        "apis": {}
    }
    
    # Initialize API instances
    apis = [
        ("iNaturalist", iNaturalistAPI(verbose=True)),
        ("Meteostat", MeteostatAPI(verbose=True)),
        ("NWS", NWS_API(verbose=True)),
        ("NOAA_Tides", NOAA_Tides_API(verbose=True)),
        ("NREL_Solar", NREL_Solar_API(verbose=True)),
        ("EnvironmentalHealth", EnvironmentalHealthAPI(verbose=True)),
        ("SoilState", SoilStateAPI(verbose=True)),
        ("OpenMeteo", OpenMeteoAPI(verbose=True)),
        ("SoilGrids", SoilGridsAPI(verbose=True)),
        ("SevereWeather", SevereWeatherAPI(verbose=True)),
        ("USGS_Earthquake", USGS_Earthquake_API(verbose=True)),
    ]
    
    # Execute API calls in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_api = {
            executor.submit(api.get_data, lat, lon): name 
            for name, api in apis
        }
        
        for future in as_completed(future_to_api):
            api_name = future_to_api[future]
            try:
                data = future.result()
                results["apis"][api_name] = data
                logger.info(f"Successfully retrieved data from {api_name}")
            except Exception as e:
                logger.error(f"Error retrieving data from {api_name}: {e}")
                results["apis"][api_name] = {"error": str(e)}
    
    return results


def main():
    """Main function to run the environmental data collection script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect environmental data for a location")
    parser.add_argument("location", help="Location name or 'lat,lon' coordinates")
    parser.add_argument("--output", "-o", default="/usr2/people/macgregor/amplicon/workflow_16s/src/workflow_16s/api/environmental_data/environmental_data.json", 
                       help="Output file path (default: environmental_data.json)")
    parser.add_argument("--no-cache", action="store_true", 
                       help="Disable caching of API responses")
    
    args = parser.parse_args()
    
    # Disable caching if requested
    if args.no_cache:
        global CACHE_EXPIRY_HOURS
        CACHE_EXPIRY_HOURS = 0
        logger.info("Caching disabled")
    
    # Parse location input
    lat, lon, location_name = None, None, args.location
    
    # Check if input is coordinates
    if "," in args.location:
        try:
            parts = args.location.split(",")
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
            location_name = f"{lat},{lon}"
        except ValueError:
            logger.error("Invalid coordinate format. Use 'lat,lon' or a location name.")
            return
    
    # If not coordinates, use geocoding to find coordinates
    if lat is None or lon is None:
        geocoder = GeocodingAPI(verbose=True)
        locations = geocoder.get_coords(args.location, count=1)
        
        if not locations:
            logger.error(f"Could not find coordinates for location: {args.location}")
            return
            
        location = locations[0]
        lat = location["latitude"]
        lon = location["longitude"]
        location_name = location["name"]
        logger.info(f"Found coordinates for {location_name}: {lat}, {lon}")
    
    # Get all environmental data
    logger.info(f"Collecting environmental data for {location_name}...")
    environmental_data = get_all_environmental_data(lat, lon, location_name)
    
    # Save results to file
    with open(args.output, 'w') as f:
        json.dump(environmental_data, f, indent=2)
    
    logger.info(f"Environmental data saved to {args.output}")
    
    # Print summary
    print(f"\n=== ENVIRONMENTAL DATA SUMMARY FOR {location_name.upper()} ===")
    for api_name, data in environmental_data["apis"].items():
        status = "✓" if data and not data.get("error") else "✗"
        print(f"{status} {api_name}: {'Success' if status == '✓' else 'Failed'}")
    
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()