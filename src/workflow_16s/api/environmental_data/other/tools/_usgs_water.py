# workflow_16s/api/environmental_data/other/tools/_usgs_water.py

import requests
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from .base import BaseEnvironmentalAPI, REQUEST_TIMEOUT
from .cache import cache_api_call
from workflow_16s.utils.logger import get_logger, with_logger


@with_logger
class USGS_Water_API(BaseEnvironmentalAPI):
    """
    Fetches hydrological and water quality data from USGS National Water Information System (NWIS).
    
    Documentation: https://waterservices.usgs.gov/
    
    Attributes:
        verbose (bool): If True, enables verbose logging.
        api_key (str): USGS API key (optional for public endpoints)
    """
    URL = "https://waterservices.usgs.gov/nwis"
    
    def __init__(self, api_key: Optional[str] = None, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.base_url = self.URL
        self.api_key = api_key
        self.verbose = verbose

    def check_requirements(self) -> Tuple[bool, Optional[str]]:
        """USGS Water API has public endpoints, so check is minimal."""
        return True, None

    @cache_api_call
    def get_data(self, lat: float, lon: float, fetch_date: Optional[str] = None) -> Optional[Dict[str, Any]]:  # type: ignore
        """
        Retrieves water quality and hydrological data for a location.
        
        Args:
            lat: Latitude of the location
            lon: Longitude of the location
            fetch_date: Optional date in 'YYYY-MM-DD' format for recent data
            
        Returns:
            Dictionary with water quality metrics or None on failure
        """
        try:
            # Query nearby USGS gauge stations
            params = {
                "format": "json",
                "latitude": lat,
                "longitude": lon,
                "siteType": "ST,ST-DCH,ST-TS,ST-TS-Fallback",  # Stream gauge stations
                "outputFormat": "json"
            }
            
            # Get inventory of nearby sites
            response = self.session.get(
                f"{self.base_url}/site",
                params=params,
                timeout=REQUEST_TIMEOUT
            )
            
            water_data = {
                'usgs_water_station_count': 0,
                'usgs_water_discharge_cfs': None,
                'usgs_water_gage_height_ft': None,
                'usgs_water_quality_indicator': 'Unknown'
            }
            
            if response.status_code == 200:
                data = response.json()
                
                if 'value' in data and 'sitesCollection' in data['value']:
                    sites = data['value']['sitesCollection'][0]['monitoringSites']
                    water_data['usgs_water_station_count'] = len(sites)
                    
                    if len(sites) > 0:
                        # Get data from closest station
                        site_code = sites[0]['siteCode'][0]['value']
                        
                        # Fetch recent discharge and stage data
                        end_date = datetime.strptime(fetch_date, '%Y-%m-%d') if fetch_date else datetime.now()
                        start_date = end_date - timedelta(days=30)
                        
                        data_params = {
                            "sites": site_code,
                            "startDT": start_date.strftime('%Y-%m-%d'),
                            "endDT": end_date.strftime('%Y-%m-%d'),
                            "parameterCd": "00060,00065",  # Discharge and Gage Height
                            "format": "json"
                        }
                        
                        data_response = self.session.get(
                            f"{self.base_url}/iv",
                            params=data_params,
                            timeout=REQUEST_TIMEOUT
                        )
                        
                        if data_response.status_code == 200:
                            ts_data = data_response.json()
                            
                            if 'value' in ts_data and 'timeSeries' in ts_data['value']:
                                for ts in ts_data['value']['timeSeries']:
                                    param_code = ts['sourceInfo']['variable']['variableCode'][0]['value']
                                    
                                    if 'values' in ts and len(ts['values']) > 0:
                                        values = ts['values'][0]['value']
                                        if values:
                                            # Get most recent value
                                            latest = values[-1]
                                            
                                            if param_code == '00060':  # Discharge
                                                try:
                                                    water_data['usgs_water_discharge_cfs'] = float(latest['value'])
                                                except:
                                                    pass
                                            elif param_code == '00065':  # Gage Height
                                                try:
                                                    water_data['usgs_water_gage_height_ft'] = float(latest['value'])
                                                except:
                                                    pass
                        
                        # Estimate water quality
                        if water_data['usgs_water_discharge_cfs'] is not None:
                            if water_data['usgs_water_discharge_cfs'] > 1000:
                                water_data['usgs_water_quality_indicator'] = 'High Discharge'
                            elif water_data['usgs_water_discharge_cfs'] < 10:
                                water_data['usgs_water_quality_indicator'] = 'Low Discharge'
                            else:
                                water_data['usgs_water_quality_indicator'] = 'Moderate Discharge'
            
            return water_data
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"USGS Water API error: {e}")
            return None
