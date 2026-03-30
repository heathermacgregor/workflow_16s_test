"""
TerraClimate Environmental Data Handler

Provides access to TerraClimate dataset through Google Earth Engine.
TerraClimate is a global 4km resolution monthly climate dataset combining
monthly climate data from diverse sources into a unified global dataset.

Variables:
- Temperature (min, max, mean)
- Precipitation
- Runoff
- Soil Moisture (monthly)
- Vapor Pressure
- Evapotranspiration
- Shortwave Radiation
- Wind Speed

TerraClimate complements ERA5 with higher spatial resolution (4km vs 31km)
and monthly aggregation tailored for biological/ecological studies.

Coverage: Global
Resolution: 4 km (~0.04°)
Time Period: 1958-present (monthly)
Data Source: Multiple (GDCD, NASA, NOAA, USGS, etc.)

Reference: https://www.climatologylab.org/terraclimate.html
GEE Asset: IDAHO_EPSCOR/TERRACLIMATE/v13
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

try:
    import ee
except ImportError:
    ee = None

from .base import BaseEnvironmentalAPI, REQUEST_TIMEOUT
from .cache import cache_api_call
from workflow_16s.utils.logger import get_logger, with_logger


@with_logger
class TerraClimateAPI(BaseEnvironmentalAPI):
    """
    Query TerraClimate climate data through Google Earth Engine.
    
    Features:
    - 4km global resolution
    - Monthly climate data
    - Multiple climate variables
    - Higher resolution than ERA5 (comparable to local studies)
    - Requires GEE authentication
    
    Returns:
    Dictionary with aggregated climate statistics:
    - terraclimate_temperature_min_c: Minimum temperature (°C)
    - terraclimate_temperature_max_c: Maximum temperature (°C)
    - terraclimate_temperature_mean_c: Mean temperature (°C)
    - terraclimate_precipitation_mm: Total precipitation (mm)
    - terraclimate_precipitation_min_mm: Minimum monthly precipitation
    - terraclimate_precipitation_max_mm: Maximum monthly precipitation
    - terraclimate_runoff_mm: Runoff (mm)
    - terraclimate_soil_moisture: Soil moisture (mm)
    - terraclimate_vapor_pressure_kpa: Vapor pressure (kPa)
    - terraclimate_evapotranspiration_mm: Evapotranspiration (mm)
    - terraclimate_shortwave_radiation_w_m2: Shortwave radiation (W/m²)
    - terraclimate_wind_speed_m_s: Wind speed (m/s)
    - aggregation_months: Number of months in aggregation period
    - data_start_date: Start date of data period
    - data_end_date: End date of data period
    
    Example:
        api = TerraClimateAPI()
        is_available, msg = api.check_requirements()
        if is_available:
            climate_data = api.get_data(lat=0.0, lon=25.0, date="2020-06")
            if climate_data:
                print(f"Mean temp: {climate_data.get('terraclimate_temperature_mean_c')}°C")
    """
    
    API_NAME = "TerraClimate"
    GEE_ASSET_ID = "IDAHO_EPSCOR/TERRACLIMATE/v13"
    
    # TerraClimate bands available
    CLIMATE_VARIABLES = {
        'tmmx': 'terraclimate_temperature_max_c',
        'tmmn': 'terraclimate_temperature_min_c',
        'pr': 'terraclimate_precipitation_mm',
        'ro': 'terraclimate_runoff_mm',
        'vs': 'terraclimate_wind_speed_m_s',
        'sph': 'terraclimate_vapor_pressure_kpa',
        'aet': 'terraclimate_evapotranspiration_mm',
        'soil': 'terraclimate_soil_moisture',
        'srad': 'terraclimate_shortwave_radiation_w_m2',
    }
    
    def __init__(self, verbose: bool = False, authenticated: bool = True):
        """
        Initialize TerraClimate API client.
        
        Args:
            verbose: Enable verbose logging
            authenticated: Whether to use GEE authentication (default True).
                          Set False for manual EE initialization.
        """
        super().__init__(verbose=verbose)
        self.authenticated = authenticated and ee is not None
        self.logger = get_logger(__name__)
        
        if self.authenticated and ee is not None:
            try:
                # Initialize GEE (idempotent - safe to call multiple times)
                ee.Initialize()
                self.logger.debug("Google Earth Engine initialized for TerraClimate API")
            except ee.EEException as e:
                # GEE already initialized or authentication issue
                self.logger.debug(f"GEE initialization note: {e}")
            except Exception as e:
                # Other error during initialization
                self.logger.debug(f"GEE initialization warning: {e}")

    def check_requirements(self) -> Tuple[bool, Optional[str]]:
        """
        Check if TerraClimate is available.
        
        Returns:
            Tuple of (is_available, error_message)
            is_available: True if GEE is initialized and authenticated
            error_message: None if available, error description otherwise
            
        NOTE:
        The TerraClimate asset 'IDAHO_EPSCOR/TERRACLIMATE/v13' may not be accessible
        if the service account doesn't have permissions or if the asset path has
        changed in the GEE catalog. This is gracefully handled by returning False
        and allowing other data sources (ERA5, etc.) to be used.
        """
        if ee is None:
            error_msg = "Google Earth Engine Python package not installed. Install with: pip install earthengine-api"
            self.logger.warning(error_msg)
            return False, error_msg
        
        if not self.authenticated:
            error_msg = "Google Earth Engine not authenticated. Run: earthengine authenticate"
            self.logger.warning(error_msg)
            return False, error_msg
        
        try:
            # Test asset access
            asset = ee.ImageCollection(self.GEE_ASSET_ID)
            info = asset.first().getInfo()
            if info:
                self.logger.info("TerraClimate GEE asset accessible")
                return True, None
        except Exception as e:
            error_msg = f"TerraClimate GEE asset not accessible: {str(e)}"
            self.logger.debug(error_msg)
            # Provide informative message about asset availability
            if "not found" in str(e).lower():
                self.logger.debug(
                    f"Asset '{self.GEE_ASSET_ID}' not found or not accessible. "
                    f"Possible causes: (1) Asset path changed, (2) Service account lacks permissions, "
                    f"(3) Asset was deprecated. TerraClimate data will not be available. "
                    f"Falling back to ERA5 or other climate sources if available."
                )
            return False, error_msg
        
        return True, None

    @cache_api_call
    def get_data(
        self,
        lat: float,
        lon: float,
        date: Optional[str] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve TerraClimate data for a location.
        
        Aggregates monthly climate data over a period specified by the date parameter.
        If no date specified, defaults to past year of available data.
        
        Args:
            lat: Latitude of query location (-90 to 90)
            lon: Longitude of query location (-180 to 180)
            date: Optional date specification. Supports:
                  - "YYYY": Entire year e.g., "2020" (Jan-Dec)
                  - "YYYY-MM": Specific month e.g., "2020-06" (June only)
                  - None: Uses most recent available year
            **kwargs: Additional keyword arguments (e.g., logger from decorator)
            
        Returns:
            Dictionary with aggregated climate statistics or None if unavailable
            
        Raises:
            No exceptions raised; errors are logged and None returned
        """
        try:
            # Validate coordinates
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                self.logger.debug(f"Invalid coordinates: lat={lat}, lon={lon}")
                return None
            
            if not self.authenticated or ee is None:
                self.logger.debug("TerraClimate requires GEE authentication")
                return None
            
            logger = kwargs.get('logger', self.logger)
            
            # Query TerraClimate data
            result = self._query_terraclimate(lat, lon, date, logger=logger)
            
            if result is None:
                logger.debug(f"No TerraClimate data found at ({lat:.4f}, {lon:.4f})")
                return None
            
            return result
            
        except Exception as e:
            self.logger.error(f"TerraClimate query failed at ({lat:.4f}, {lon:.4f}): {str(e)}")
            return None

    def _query_terraclimate(
        self,
        lat: float,
        lon: float,
        date: Optional[str] = None,
        logger: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Query TerraClimate data from GEE.
        
        Args:
            lat: Latitude
            lon: Longitude
            date: Date specification (YYYY or YYYY-MM)
            logger: Logger instance
            
        Returns:
            Dictionary with climate data or None if unavailable
        """
        if logger is None:
            logger = self.logger
        
        try:
            # Create point geometry
            point = ee.Geometry.Point([lon, lat])
            
            # Parse date specification
            start_date, end_date = self._parse_date_range(date, logger)
            
            # Load TerraClimate collection
            collection = ee.ImageCollection(self.GEE_ASSET_ID)
            
            # Filter by date and location
            filtered = collection.filterBounds(point).filterDate(start_date, end_date)
            
            # Get number of months available
            n_images = filtered.size().getInfo()
            if n_images == 0:
                logger.debug(f"No TerraClimate data available for period {start_date} to {end_date}")
                return None
            
            logger.debug(f"Found {n_images} TerraClimate months at ({lat:.4f}, {lon:.4f})")
            
            # Calculate mean over time period
            mean_image = filtered.mean()
            
            # Select relevant bands
            bands_to_select = list(self.CLIMATE_VARIABLES.keys())
            selected = mean_image.select(bands_to_select)
            
            # Sample at point
            sample = selected.sample(point, scale=4000)  # 4km resolution
            data = sample.first().getInfo()
            
            if not data or 'properties' not in data:
                logger.debug("Empty response from TerraClimate sampling")
                return None
            
            # Rename properties to output names
            properties = data['properties']
            result = {}
            
            for gee_band, output_name in self.CLIMATE_VARIABLES.items():
                if gee_band in properties:
                    value = properties[gee_band]
                    # Filter out 0 or null values (common in TerraClimate)
                    if value is not None and (gee_band == 'pr' or value != 0):
                        result[output_name] = float(value)
            
            # Add metadata
            result['aggregation_months'] = n_images
            result['data_start_date'] = start_date
            result['data_end_date'] = end_date
            
            logger.debug(f"Retrieved {len(result)} TerraClimate variables")
            
            return result if result else None
            
        except Exception as e:
            logger.debug(f"TerraClimate GEE query error: {str(e)}")
            return None

    def _parse_date_range(
        self,
        date_spec: Optional[str],
        logger: Optional[Any] = None
    ) -> Tuple[str, str]:
        """
        Parse date specification into start and end dates.
        
        Args:
            date_spec: Date specification (YYYY, YYYY-MM, or None)
            logger: Logger instance
            
        Returns:
            Tuple of (start_date, end_date) in ISO format
        """
        if logger is None:
            logger = self.logger
        
        try:
            if date_spec is None:
                # Default: past year of available data (TerraClimate updates with lag)
                # Use 2 years ago to ensure data is available
                end_date = datetime.now() - timedelta(days=365)
                start_date = end_date - timedelta(days=365)
            elif len(date_spec) == 4:
                # YYYY format: entire year
                year = int(date_spec)
                start_date = datetime(year, 1, 1)
                end_date = datetime(year, 12, 31)
            elif len(date_spec) == 7 and date_spec[4] == '-':
                # YYYY-MM format: specific month
                year = int(date_spec[:4])
                month = int(date_spec[5:7])
                start_date = datetime(year, month, 1)
                # Get last day of month
                if month == 12:
                    end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
                else:
                    end_date = datetime(year, month + 1, 1) - timedelta(days=1)
            else:
                logger.warning(f"Invalid date format: {date_spec}. Using default.")
                end_date = datetime.now() - timedelta(days=365)
                start_date = end_date - timedelta(days=365)
            
            # Convert to ISO format
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            logger.debug(f"Date range: {start_str} to {end_str}")
            
            return start_str, end_str
            
        except (ValueError, AttributeError) as e:
            logger.warning(f"Date parsing error: {e}. Using default.")
            end_date = datetime.now() - timedelta(days=365)
            start_date = end_date - timedelta(days=365)
            return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
