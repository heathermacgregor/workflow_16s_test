# workflow_16s/api/environmental_data/other/tools/_nasa_eonet.py

import requests
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from .base import BaseEnvironmentalAPI, REQUEST_TIMEOUT
from .cache import cache_api_call
from workflow_16s.utils.logger import get_logger, with_logger


@with_logger
class NASA_EONET_API(BaseEnvironmentalAPI):
    """
    Fetches environmental events from NASA Earth Observatory Network (EONET).
    
    Events include: Wildfires, Earthquakes, Volcanic Activity, Floods, Droughts, Hurricanes, etc.
    
    Documentation: https://eonet.gsfc.nasa.gov/api/v3/
    
    Attributes:
        verbose (bool): If True, enables verbose logging.
        api_key (str): NASA EONET API key
    """
    URL = "https://eonet.gsfc.nasa.gov/api/v3"
    
    def __init__(self, api_key: Optional[str] = None, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.base_url = self.URL
        self.api_key = api_key
        self.verbose = verbose

    def check_requirements(self) -> Tuple[bool, Optional[str]]:
        """Checks for NASA EONET API key."""
        if not self.api_key:
            return False, "NASA_EONET_API_KEY environment variable must be set."
        return True, None

    @cache_api_call
    def get_data(self, lat: float, lon: float, fetch_date: Optional[str] = None) -> Optional[Dict[str, Any]]:  # type: ignore
        """
        Retrieves environmental events near a location.
        
        Args:
            lat: Latitude of the location
            lon: Longitude of the location
            fetch_date: Optional date in 'YYYY-MM-DD' format for filtering events
            
        Returns:
            Dictionary with event counts and recent events or None on failure
        """
        try:
            # Get all events within a date range
            days_back = 365
            if fetch_date:
                event_date = datetime.strptime(fetch_date, '%Y-%m-%d')
                start_date = (event_date - timedelta(days=days_back)).strftime('%Y-%m-%d')
                end_date = event_date.strftime('%Y-%m-%d')
            else:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            params = {
                "api_key": self.api_key,
                "begin": start_date,
                "end": end_date,
                "limit": 100
            }
            
            response = self.session.get(
                f"{self.base_url}/events",
                params=params,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'events' in data:
                    events = data['events']
                    
                    # Count events and find nearby ones
                    event_count = 0
                    nearby_events = []
                    event_types = {}
                    
                    for event in events:
                        if 'geometries' in event:
                            # Calculate distance to location
                            for geom in event['geometries']:
                                if 'coordinates' in geom:
                                    coords = geom['coordinates']
                                    event_lon, event_lat = coords[0], coords[1]
                                    
                                    # Simple distance calculation (not exact, but good enough)
                                    dist = ((event_lat - lat)**2 + (event_lon - lon)**2)**0.5
                                    
                                    if dist < 5:  # Within ~5 degrees (~500 km)
                                        event_count += 1
                                        event_type = event.get('categories', [{}])[0].get('title', 'Unknown')
                                        event_types[event_type] = event_types.get(event_type, 0) + 1
                                        
                                        if len(nearby_events) < 5:  # Keep top 5
                                            nearby_events.append({
                                                'title': event.get('title', 'Unknown'),
                                                'type': event_type,
                                                'date': event.get('geometries', [{}])[0].get('date', 'Unknown'),
                                                'distance_degrees': dist
                                            })
                    
                    return {
                        'nasa_eonet_event_count': event_count,
                        'nasa_eonet_event_types': ','.join([f"{k}({v})" for k, v in event_types.items()]),
                        'nasa_eonet_nearby_events': str(nearby_events[:3]),  # Top 3 nearby
                    }
            
            return {
                'nasa_eonet_event_count': 0,
                'nasa_eonet_event_types': 'None',
                'nasa_eonet_nearby_events': '',
            }
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"NASA EONET API error: {e}")
            return None
