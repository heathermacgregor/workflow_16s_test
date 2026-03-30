# workflow_16s/api/environmental_data/other/tools/_nasa_earth_observatory.py

import requests
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from .base import BaseEnvironmentalAPI, REQUEST_TIMEOUT
from .cache import cache_api_call
from workflow_16s.utils.logger import get_logger, with_logger


@with_logger
class NASA_Earth_Observatory_API(BaseEnvironmentalAPI):
    """
    Fetches satellite imagery and observations from NASA Earth Observatory.
    
    Includes: MODIS, Landsat, Sentinel-2 imagery and derived products
    
    Documentation: https://earthdata.nasa.gov/
    
    Attributes:
        verbose (bool): If True, enables verbose logging.
        api_key (str): NASA Earth Observatory API key (JWT token)
    """
    URL = "https://api.earthdata.nasa.gov"
    
    def __init__(self, api_key: Optional[str] = None, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.base_url = self.URL
        self.api_key = api_key
        self.verbose = verbose

    def check_requirements(self) -> Tuple[bool, Optional[str]]:
        """Checks for NASA Earth Observatory API key."""
        if not self.api_key:
            return False, "NASA_EARTH_OBSERVATORY_API_KEY environment variable must be set."
        return True, None

    @cache_api_call
    def get_data(self, lat: float, lon: float, fetch_date: Optional[str] = None) -> Optional[Dict[str, Any]]:  # type: ignore
        """
        Retrieves satellite imagery observations for a location.
        
        Args:
            lat: Latitude of the location
            lon: Longitude of the location
            fetch_date: Optional date in 'YYYY-MM-DD' format for imagery date
            
        Returns:
            Dictionary with observatory metadata or None on failure
        """
        try:
            # Query available imagery from CMR (Common Metadata Repository)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            if fetch_date:
                date = datetime.strptime(fetch_date, '%Y-%m-%d')
                start_date = (date - timedelta(days=60)).strftime('%Y-%m-%dT00:00:00Z')
                end_date = date.strftime('%Y-%m-%dT23:59:59Z')
            else:
                end_date = datetime.now().strftime('%Y-%m-%dT23:59:59Z')
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%dT00:00:00Z')
            
            # Query CMR Search API for granules
            params = {
                "search_type": "granule",
                "short_name": ["MOD09GA", "MOD11A2", "MYD09GA"],  # MODIS products
                "point": f"{lon},{lat}",
                "temporal": f"{start_date},{end_date}",
                "pagesize": 100,
                "sortby": "start_date",
                "sortorder": "desc"
            }
            
            response = self.session.get(
                f"{self.base_url}/search/granules.json",
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )
            
            obs_data = {
                'nasa_eobs_imagery_count': 0,
                'nasa_eobs_products': 'None',
                'nasa_eobs_most_recent_date': 'Unknown',
                'nasa_eobs_data_quality': 'Unknown'
            }
            
            if response.status_code == 200:
                data = response.json()
                
                if 'feed' in data and 'entry' in data['feed']:
                    entries = data['feed']['entry']
                    obs_data['nasa_eobs_imagery_count'] = len(entries) if isinstance(entries, list) else 1
                    
                    if obs_data['nasa_eobs_imagery_count'] > 0:
                        products = {}
                        most_recent = None
                        
                        if not isinstance(entries, list):
                            entries = [entries]
                        
                        for entry in entries:
                            # Extract product info
                            if 'title' in entry:
                                title = entry['title']
                                # Extract product name
                                if 'MOD' in title:
                                    prod = title.split('MOD')[1][:5]
                                    products[f"MOD{prod}"] = products.get(f"MOD{prod}", 0) + 1
                            
                            # Extract date
                            if 'published' in entry:
                                try:
                                    entry_date = datetime.fromisoformat(entry['published'].replace('Z', '+00:00'))
                                    if most_recent is None or entry_date > most_recent:
                                        most_recent = entry_date
                                except:
                                    pass
                        
                        obs_data['nasa_eobs_products'] = ','.join([f"{k}({v})" for k, v in products.items()])
                        
                        if most_recent:
                            obs_data['nasa_eobs_most_recent_date'] = most_recent.strftime('%Y-%m-%d')
                        
                        # Estimate data quality
                        if obs_data['nasa_eobs_imagery_count'] > 10:
                            obs_data['nasa_eobs_data_quality'] = 'High Coverage'
                        elif obs_data['nasa_eobs_imagery_count'] > 3:
                            obs_data['nasa_eobs_data_quality'] = 'Good Coverage'
                        else:
                            obs_data['nasa_eobs_data_quality'] = 'Limited Coverage'
            
            return obs_data
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"NASA Earth Observatory API error: {e}")
            return None
