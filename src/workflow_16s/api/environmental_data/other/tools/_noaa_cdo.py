# workflow_16s/api/environmental_data/other/tools/_noaa_cdo.py

import requests
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from .base import BaseEnvironmentalAPI, REQUEST_TIMEOUT
from .cache import cache_api_call
from workflow_16s.utils.logger import get_logger, with_logger


@with_logger
class NOAA_CDO_API(BaseEnvironmentalAPI):
    """
    Fetches comprehensive climate data from NOAA Climate Data Online (CDO).
    
    Documentation: https://www.ncdc.noaa.gov/cdo-web/
    
    Attributes:
        verbose (bool): If True, enables verbose logging.
        api_key (str): NOAA CDO API key
    """
    URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
    
    def __init__(self, api_key: Optional[str] = None, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.base_url = self.URL
        self.api_key = api_key
        self.verbose = verbose

    def check_requirements(self) -> Tuple[bool, Optional[str]]:
        """Checks for NOAA CDO API key."""
        if not self.api_key:
            return False, "NOAA_CDO_API_KEY environment variable must be set."
        return True, None

    @cache_api_call
    def get_data(self, lat: float, lon: float, fetch_date: Optional[str] = None) -> Optional[Dict[str, Any]]:  # type: ignore
        """
        Retrieves climate data for a location from NOAA CDO.
        
        Args:
            lat: Latitude of the location
            lon: Longitude of the location
            fetch_date: Optional date in 'YYYY-MM-DD' format for historical data
            
        Returns:
            Dictionary with climate variables or None on failure
        """
        try:
            # NOAA CDO requires fetching station data first
            # For simplicity, we'll use the general endpoint
            headers = {"token": self.api_key}
            
            # Get data for closest available station
            # This is a simplified implementation - full NOAA CDO requires more complex queries
            params = {
                "datasetid": "GHCND",  # Global Historical Climatology Network - Daily
                "startdate": (datetime.strptime(fetch_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d') if fetch_date else (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                "enddate": fetch_date if fetch_date else datetime.now().strftime('%Y-%m-%d'),
                "limit": 1000,
                "latitude": lat,
                "longitude": lon,
                "radius": 25  # 25 km radius
            }
            
            response = self.session.get(
                f"{self.base_url}/data",
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and data['results']:
                    # Extract key climate variables
                    results = data['results']
                    climate_data = {
                        'noaa_cdo_temperature_c': None,
                        'noaa_cdo_precipitation_mm': None,
                        'noaa_cdo_snow_depth_cm': None,
                        'noaa_cdo_wind_speed_ms': None,
                    }
                    
                    # Parse results and compute averages
                    temps = []
                    precips = []
                    winds = []
                    
                    for record in results[:100]:  # Limit to first 100 records
                        if 'datatype' in record:
                            if record['datatype'] == 'TAVG' and 'value' in record:
                                temps.append(record['value'] / 10.0)  # NOAA stores in 0.1°C
                            elif record['datatype'] == 'PRCP' and 'value' in record:
                                precips.append(record['value'])  # in mm
                            elif record['datatype'] == 'WSPD' and 'value' in record:
                                winds.append(record['value'] / 10.0)  # in 0.1 m/s
                    
                    if temps:
                        climate_data['noaa_cdo_temperature_c'] = sum(temps) / len(temps)
                    if precips:
                        climate_data['noaa_cdo_precipitation_mm'] = sum(precips)
                    if winds:
                        climate_data['noaa_cdo_wind_speed_ms'] = sum(winds) / len(winds)
                    
                    return climate_data
            
            return None
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"NOAA CDO API error: {e}")
            return None
