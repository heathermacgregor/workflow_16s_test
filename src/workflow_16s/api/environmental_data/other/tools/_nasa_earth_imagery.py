# workflow_16s/api/environmental_data/other/tools/_nasa_earth_imagery.py

import requests
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from .base import BaseEnvironmentalAPI, REQUEST_TIMEOUT
from .cache import cache_api_call
from workflow_16s.utils.logger import get_logger, with_logger


@with_logger
class NASA_Earth_Imagery_API(BaseEnvironmentalAPI):
    """
    Fetches satellite imagery metadata and links from NASA Earth Imagery API.
    
    Uses: Landsat 8, Sentinel-2, MODIS data
    
    Documentation: https://api.nasa.gov/#earth
    
    Attributes:
        verbose (bool): If True, enables verbose logging.
        api_key (str): NASA API key
    """
    URL = "https://api.nasa.gov/planetary/earth"
    
    def __init__(self, api_key: Optional[str] = None, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.base_url = self.URL
        self.api_key = api_key
        self.verbose = verbose

    def check_requirements(self) -> Tuple[bool, Optional[str]]:
        """Checks for NASA API key."""
        if not self.api_key:
            return False, "NASA_EARTH_IMAGERY_API_KEY environment variable must be set."
        return True, None

    @cache_api_call
    def get_data(self, lat: float, lon: float, fetch_date: Optional[str] = None) -> Optional[Dict[str, Any]]:  # type: ignore
        """
        Retrieves satellite imagery information for a location.
        
        Args:
            lat: Latitude of the location
            lon: Longitude of the location
            fetch_date: Optional date in 'YYYY-MM-DD' format for imagery date
            
        Returns:
            Dictionary with imagery metadata or None on failure
        """
        try:
            # Get imagery assets (available satellite images)
            params = {
                "lon": lon,
                "lat": lat,
                "dim": 1,  # 1 deg x 1 deg
                "api_key": self.api_key
            }
            
            if fetch_date:
                params["begin"] = (datetime.strptime(fetch_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
                params["end"] = fetch_date
            else:
                params["end"] = datetime.now().strftime('%Y-%m-%d')
                params["begin"] = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # Get imagery assets
            response = self.session.get(
                f"{self.base_url}/assets",
                params=params,
                timeout=REQUEST_TIMEOUT
            )
            
            imagery_data = {
                'nasa_imagery_available': False,
                'nasa_imagery_assets_count': 0,
                'nasa_imagery_satellites': 'None',
                'nasa_imagery_most_recent': 'Unknown'
            }
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data:
                    results = data['results']
                    imagery_data['nasa_imagery_assets_count'] = len(results)
                    
                    if len(results) > 0:
                        imagery_data['nasa_imagery_available'] = True
                        
                        # Count by satellite
                        satellites = {}
                        most_recent_date = None
                        
                        for asset in results:
                            if 'instrument' in asset:
                                sat = asset['instrument']
                                satellites[sat] = satellites.get(sat, 0) + 1
                            
                            if 'acquired' in asset:
                                asset_date = datetime.fromisoformat(asset['acquired'].replace('Z', '+00:00'))
                                if most_recent_date is None or asset_date > most_recent_date:
                                    most_recent_date = asset_date
                        
                        imagery_data['nasa_imagery_satellites'] = ','.join([f"{k}({v})" for k, v in satellites.items()])
                        
                        if most_recent_date:
                            imagery_data['nasa_imagery_most_recent'] = most_recent_date.strftime('%Y-%m-%d')
            
            return imagery_data
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"NASA Earth Imagery API error: {e}")
            return None
