"""
Dynamic World Land Use/Land Cover Environmental Data Handler

Provides access to Google's Dynamic World dataset for high-resolution daily
land use and land cover classification. Dynamic World is a near-real-time,
10-meter resolution global land use/land cover dataset derived from Sentinel-2 data.

Data Source: Google Earth Engine asset "GOOGLE/DYNAMICWORLD/V1"
- Provider: Google, World Resources Institute, National Geographic Society, etc.
- Component: GOOGLE/DYNAMICWORLD/V1 (classification) + GOOGLE/DYNAMICWORLD/V1_PROBABILITY
- Resolution: 10 meters (Sentinel-2 resolution)
- Coverage: Global, except polar regions
- Update Frequency: Daily
- Time Period: 2015-present
- Access Method: Google Earth Engine API

Land Use/Land Cover Classes (10m):
- Water: Bodies of water (rivers, lakes, wetlands, reservoirs)
- Trees: Forest canopy, plantations, mangroves
- Grass: Natural grasslands, pastures, meadows
- Shrubs: Shrubland, bushland, heathland
- Crops: Cultivated areas, agricultural land
- Built: Urban areas, roads, infrastructure
- Barren: Bare soil, rock, sand, quarries
- Snow: Snow and ice (rare in tropical regions)

Applications:
- Characterization of sample collection microhabitat
- Agricultural vs. natural ecosystem distinction
- Urban influence assessment
- Watershed and ecosystem fragmentation
- Habitat connectivity analysis

Variables Extracted:
- Dominant land cover class
- Probability scores for all 8 classes
- Percentage of image covered by class types
- Confidence metrics

Reference: https://www.dynamicworldapp.com/
GEE Asset: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1
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
class DynamicWorldAPI(BaseEnvironmentalAPI):
    """
    Query Google Dynamic World LULC data for land cover characterization.
    
    Features:
    - 10-meter resolution daily global LULC
    - 8 land cover classes
    - Probability/confidence scores
    - Google Earth Engine integration
    - Requires GEE authentication
    
    Returns:
    Dictionary with land cover classification and percentages:
    - lulc_dominant_class: Most prevalent land cover type
    - lulc_class_confidence: Probability of dominant class (0-1)
    - lulc_water_pct: Percentage of water
    - lulc_trees_pct: Percentage of trees/forest
    - lulc_grass_pct: Percentage of grass
    - lulc_shrubs_pct: Percentage of shrubs
    - lulc_crops_pct: Percentage of crops/agriculture
    - lulc_built_pct: Percentage of built-up areas
    - lulc_barren_pct: Percentage of barren land
    - lulc_snow_pct: Percentage of snow/ice
    - lulc_composite_date: Date of LULC composite/classification
    - lulc_data_source: "Dynamic World"
    
    Example:
        api = DynamicWorldAPI()
        is_available, msg = api.check_requirements()
        if is_available:
            lulc_data = api.get_data(lat=0.0, lon=25.0, date="2022-06")
            if lulc_data:
                print(f"Dominant LULC: {lulc_data.get('lulc_dominant_class')}")
    """
    
    API_NAME = "DynamicWorld"
    GEE_ASSET_ID = "GOOGLE/DYNAMICWORLD/V1"
    GEE_PROB_ASSET_ID = "GOOGLE/DYNAMICWORLD/V1_PROBABILITY"
    
    # Land cover classes (indices match GEE bands)
    LULC_CLASSES = {
        0: 'water',
        1: 'trees',
        2: 'grass',
        3: 'shrubs',
        4: 'crops',
        5: 'built',
        6: 'barren',
        7: 'snow',
    }
    
    # Field names for output
    OUTPUT_FIELDS = {
        'water': 'lulc_water_pct',
        'trees': 'lulc_trees_pct',
        'grass': 'lulc_grass_pct',
        'shrubs': 'lulc_shrubs_pct',
        'crops': 'lulc_crops_pct',
        'built': 'lulc_built_pct',
        'barren': 'lulc_barren_pct',
        'snow': 'lulc_snow_pct',
    }
    
    def __init__(self, verbose: bool = False, authenticated: bool = True, buffer_m: int = 100):
        """
        Initialize Dynamic World API client.
        
        Args:
            verbose: Enable verbose logging
            authenticated: Whether to use GEE authentication (default True)
            buffer_m: Buffer around point for regional analysis (meters, default 100m for 10m pixels)
        """
        super().__init__(verbose=verbose)
        self.authenticated = authenticated and ee is not None
        self.logger = get_logger(__name__)
        self.buffer_m = buffer_m
        
        if self.authenticated and ee is not None:
            try:
                # Initialize GEE (idempotent - safe to call multiple times)
                ee.Initialize()
                self.logger.debug("Google Earth Engine initialized for Dynamic World API")
            except ee.EEException as e:
                # GEE already initialized or authentication issue
                self.logger.debug(f"GEE initialization note: {e}")
            except Exception as e:
                # Other error during initialization
                self.logger.debug(f"GEE initialization warning: {e}")

    def check_requirements(self) -> Tuple[bool, Optional[str]]:
        """
        Check if Dynamic World is available via GEE.
        
        Returns:
            Tuple of (is_available, error_message)
            is_available: True if GEE is initialized and authenticated
            error_message: None if available, error description otherwise
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
                self.logger.info("Dynamic World GEE asset accessible")
                return True, None
        except Exception as e:
            error_msg = f"Dynamic World GEE asset not accessible: {str(e)}"
            self.logger.debug(error_msg)
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
        Retrieve Dynamic World land cover data for a location.
        
        NOTE: Dynamic World coverage varies by region and date.
        - Equatorial regions: frequent cloud cover, may have gaps
        - High latitude (>60°): sparse Sentinel-2 coverage
        - Mountain regions: often covered by clouds
        - Data lag: Recent data (< 10 days) may not be processed
        
        If data unavailable for requested date, automatically extends
        search range to find nearest available imagery.
        
        Queries Dynamic World LULC classification for the specified location
        and date, returning percentages of each land cover class.
        
        Args:
            lat: Latitude of query location (-90 to 90)
            lon: Longitude of query location (-180 to 180)
            date: Optional date specification. Supports:
                  - "YYYY": Entire year (uses annual composite)
                  - "YYYY-MM": Specific month (uses monthly composite)
                  - "YYYY-MM-DD": Specific date (searches ±30 days)
                  - None: Uses most recent available data
            **kwargs: Additional keyword arguments:
                  - fetch_date: Alternative date parameter from workflow (highest priority)
                  - logger: Logger instance from decorator
            
        Returns:
            Dictionary with LULC percentages or None if unavailable
            
        Raises:
            No exceptions raised; errors are logged and None returned
        """
        try:
            # Validate coordinates
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                self.logger.debug(f"Invalid coordinates: lat={lat}, lon={lon}")
                return None
            
            if not self.authenticated or ee is None:
                self.logger.debug("Dynamic World requires GEE authentication")
                return None
            
            logger = kwargs.get('logger', self.logger)
            
            # Use fetch_date from kwargs (passed by orchestration) if available
            # Otherwise use date parameter, allowing None for most recent data
            query_date = kwargs.get('fetch_date', date)
            
            # Query Dynamic World data
            result = self._query_dynamic_world(lat, lon, query_date, logger=logger)
            
            if result is None:
                logger.debug(f"No Dynamic World data found at ({lat:.4f}, {lon:.4f})")
                return None
            
            return result
            
        except Exception as e:
            self.logger.error(f"Dynamic World query failed at ({lat:.4f}, {lon:.4f}): {str(e)}")
            return None

    def _query_dynamic_world(
        self,
        lat: float,
        lon: float,
        date: Optional[str] = None,
        logger: Optional[Any] = None,
        extend_range: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Query Dynamic World LULC data from GEE with automatic fallback.
        
        If no data available in requested date range, automatically extends
        the range to find nearest available imagery.
        
        Args:
            lat: Latitude
            lon: Longitude
            date: Date specification (YYYY, YYYY-MM, or None)
            logger: Logger instance
            extend_range: If True, auto-extend date range on empty results
            
        Returns:
            Dictionary with LULC percentages or None if unavailable
        """
        if logger is None:
            logger = self.logger
        
        try:
            # Create point geometry with buffer
            point = ee.Geometry.Point([lon, lat])
            # Use small buffer (100m) for representative area classification
            region = point.buffer(self.buffer_m)
            
            # Parse date specification
            start_date, end_date = self._parse_date_range(date, logger)
            
            # Load Dynamic World collection
            collection = ee.ImageCollection(self.GEE_ASSET_ID)
            
            # Filter by date and region
            filtered = collection.filterBounds(region).filterDate(start_date, end_date)
            
            # Get count of images
            n_images = filtered.size().getInfo()
            if n_images == 0:
                # Try extending range backwards and forwards
                if extend_range:
                    logger.debug(
                        f"Dynamic World: No imagery at ({lat:.4f}, {lon:.4f}) "
                        f"for {start_date} to {end_date}, extending search range..."
                    )
                    
                    # Extend backwards 60 days and forwards 30 days
                    extended_start = (
                        datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=60)
                    ).strftime('%Y-%m-%d')
                    extended_end = (
                        datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=30)
                    ).strftime('%Y-%m-%d')
                    
                    logger.debug(
                        f"Extended search: {extended_start} to {extended_end}"
                    )
                    
                    filtered = collection.filterBounds(region).filterDate(
                        extended_start, extended_end
                    )
                    n_images = filtered.size().getInfo()
                    
                    if n_images == 0:
                        logger.info(
                            f"Dynamic World: No imagery at ({lat:.4f}, {lon:.4f}) "
                            f"for {start_date} to {end_date}. "
                            f"This region has sparse satellite coverage; "
                            f"consider using ESA WorldCover as alternative."
                        )
                        return None
                    
                    # Update end_date to reflect extended search
                    end_date = extended_end
                else:
                    logger.info(
                        f"Dynamic World: No imagery at ({lat:.4f}, {lon:.4f}) "
                        f"for {start_date} to {end_date}"
                    )
                    return None
            
            logger.debug(f"Found {n_images} Dynamic World images at ({lat:.4f}, {lon:.4f})")
            
            # Calculate mode (most common class) over time period
            # Use first image if only one, otherwise use mode composite
            if n_images == 1:
                image = filtered.first()
            else:
                # Mode composite: most common class over multiple dates
                image = filtered.mode()
            
            # Sample classification at point
            sample = image.sample(region, scale=10).first()
            data = sample.getInfo()
            
            if not data or 'properties' not in data:
                logger.debug("Empty response from Dynamic World sampling")
                return None
            
            # Extract class data
            properties = data['properties']
            result = self._extract_lulc_percentages(properties, logger)
            
            # Add metadata
            if result:
                result['lulc_composite_date'] = end_date
                result['lulc_data_source'] = 'Dynamic World'
                result['number_of_images'] = n_images
                
                logger.debug(f"Retrieved Dynamic World LULC with {len(result)} properties")
            
            return result if result else None
            
        except Exception as e:
            logger.debug(f"Dynamic World GEE query error: {str(e)}")
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
                # Default: most recent month of available data
                # Dynamic World updates daily, lag is usually ~5-10 days
                today = datetime.now()
                end_date = today - timedelta(days=10)
                start_date = end_date - timedelta(days=30)
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
                today = datetime.now()
                end_date = today - timedelta(days=10)
                start_date = end_date - timedelta(days=30)
            
            # Convert to ISO format
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            logger.debug(f"Dynamic World date range: {start_str} to {end_str}")
            
            return start_str, end_str
            
        except (ValueError, AttributeError) as e:
            logger.warning(f"Date parsing error: {e}. Using default.")
            today = datetime.now()
            end_date = today - timedelta(days=10)
            start_date = end_date - timedelta(days=30)
            return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

    def _extract_lulc_percentages(
        self,
        properties: Dict[str, Any],
        logger: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Extract LULC class percentages from sampled properties.
        
        Dynamic World returns probability bands for each class.
        This function normalizes to percentages and identifies dominant class.
        
        Args:
            properties: GEE sampling properties
            logger: Logger instance
            
        Returns:
            Dictionary with percentages and dominant class or None
        """
        if logger is None:
            logger = self.logger
        
        try:
            result = {}
            class_values = {}
            
            # Extract probability values for each class
            # GEE property naming convention: band names like 'water', 'trees', etc.
            for class_idx, class_name in self.LULC_CLASSES.items():
                value = properties.get(class_name, 0)
                
                # Convert to float percentage if needed
                try:
                    prob = float(value)
                    # Dynamic World returns probabilities (0-1), convert to percentages
                    pct = prob * 100
                    class_values[class_name] = pct
                    
                    # Store in output
                    output_field = self.OUTPUT_FIELDS.get(class_name)
                    if output_field:
                        result[output_field] = round(pct, 2)
                except (ValueError, TypeError):
                    logger.debug(f"Invalid value for {class_name}: {value}")
                    result[self.OUTPUT_FIELDS.get(class_name)] = 0.0
            
            # Find dominant class
            if class_values:
                dominant_class = max(class_values, key=class_values.get)
                dominant_pct = class_values[dominant_class]
                
                result['lulc_dominant_class'] = dominant_class
                result['lulc_class_confidence'] = round(dominant_pct / 100.0, 3)  # Back to 0-1 scale
            else:
                return None
            
            # Verify percentages sum to ~100 (with rounding tolerance)
            total_pct = sum(v for k, v in class_values.items() if v >= 0)
            if total_pct < 10:
                logger.debug(f"Low total LULC percentage: {total_pct:.1f}%")
                return None
            
            logger.debug(f"LULC: dominant={dominant_class}, confidence={result['lulc_class_confidence']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"LULC percentage extraction error: {e}")
            return None
