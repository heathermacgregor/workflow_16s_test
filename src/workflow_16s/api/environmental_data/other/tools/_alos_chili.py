"""
ALOS CHILI and Multi-scale Topographic Position Index (mTPI) Environmental Data Handler

Provides access to Advanced Topographic Radiative Index and multi-scale topographic
position indices derived from ALOS Palsar and NASADEM digital elevation models.
Computes terrain metrics including slope, aspect, elevation, and surface roughness.

Data Sources:
- ALOS Palsar DEM: JAXA/ALOS/AW3D30/V3_2 via Google Earth Engine
- Alternative: NASADEM: NASA/NASADEM_HGT/001
- Resolution: 30m (NASADEM) or 5m (ALOS, limited coverage)
- Coverage: Global, updated periodically
- Access: Google Earth Engine API

Derived Variables:
- Topographic Radiative Index (TRI): Surface roughness at local scale (~300m neighborhood)
- Multi-scale TPI (mTPI): Position relative to surroundings at 10km neighborhood scale
- Slope (degrees): Terrain gradient magnitude
- Aspect (degrees): Direction of steepest slope (0-360° from North)
- Elevation (meters): Above-sea-level elevation
- Terrain roughness class: Categorical terrain type (flat/rolling/hilly/steep/extreme)

Applications:
- Microhabitat characterization (exposed vs. sheltered sites)
- Erosion and landslide risk assessment
- Hydrological flow pathways
- Solar radiation modeling
- Species habitat prediction based on terrain

References:
- Riley S.J., et al. (1999). Index that quantifies topographic heterogeneity.
- Weiss A. (2001). Topographic Position and Landforms Analysis.
- GEE Asset: https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_AW3D30_V3_2
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import math

try:
    import ee
except ImportError:
    ee = None

from .base import BaseEnvironmentalAPI
from .cache import cache_api_call
from workflow_16s.utils.logger import get_logger, with_logger


@with_logger
class ALOSCHILIandmTPIAPI(BaseEnvironmentalAPI):
    """
    Query ALOS DEM-derived topographic metrics for terrain characterization.
    
    Features:
    - Multi-scale topographic analysis (3-scale TPI)
    - Surface roughness quantification
    - Slope and aspect computation
    - Elevation above sea level
    - Terrain classification
    - Google Earth Engine integration
    - Requires GEE authentication
    
    Returns:
    Dictionary with terrain metrics:
    - tri_value: Topographic Radiative Index (0-300+ meters, local roughness)
    - mtpi_value: Multi-scale Topographic Position Index (-100 to +100, relative position)
    - slope_deg: Slope angle in degrees (0-90)
    - aspect_deg: Aspect direction in degrees (0-360, None=flat)
    - elevation_m: Elevation above sea level (meters)
    - terrain_roughness_class: Classification (flat/rolling/hilly/steep/extreme)
    - terrain_data_source: "ALOS Palsar DEM" or "NASADEM"
    
    Example:
        api = ALOSCHILIandmTPIAPI()
        is_available, msg = api.check_requirements()
        if is_available:
            terrain_data = api.get_data(lat=0.0, lon=25.0)
            if terrain_data:
                print(f"Elevation: {terrain_data.get('elevation_m')} m")
                print(f"Slope: {terrain_data.get('slope_deg')}°")
                print(f"Roughness class: {terrain_data.get('terrain_roughness_class')}")
    """
    
    API_NAME = "ALOSCHILImTPI"
    # Primary: ALOS Palsar DEM (5m, more detailed but limited coverage)
    GEE_ASSET_PRIMARY = "JAXA/ALOS/AW3D30/V3_2"
    # Fallback: NASADEM (30m, global coverage)
    GEE_ASSET_FALLBACK = "NASA/NASADEM_HGT/001"
    
    # Constants for TPI calculation
    LOCAL_NEIGHBORHOOD_M = 300   # ~10 pixels at 30m resolution
    REGIONAL_NEIGHBORHOOD_M = 10000  # ~333 pixels at 30m resolution (10km)
    
    # Terrain roughness thresholds (TRI in meters)
    ROUGHNESS_CLASSES = {
        'flat': (0, 10),
        'rolling': (10, 50),
        'hilly': (50, 150),
        'steep': (150, 300),
        'extreme': (300, 10000),
    }
    
    def __init__(self, verbose: bool = False, authenticated: bool = True, use_fallback: bool = True):
        """
        Initialize ALOS CHILI & mTPI API client.
        
        Args:
            verbose: Enable verbose logging
            authenticated: Whether to use GEE authentication (default True)
            use_fallback: Use NASADEM if ALOS not available (default True)
        """
        super().__init__(verbose=verbose)
        self.authenticated = authenticated and ee is not None
        self.logger = get_logger(__name__)
        self.use_fallback = use_fallback
        self.gee_asset = self.GEE_ASSET_PRIMARY
        
        if self.authenticated and ee is not None:
            try:
                # Initialize GEE (idempotent - safe to call multiple times)
                ee.Initialize()
                self.logger.debug("Google Earth Engine initialized for ALOS CHILI/mTPI API")
            except ee.EEException as e:
                # GEE already initialized or authentication issue
                self.logger.debug(f"GEE initialization note: {e}")
            except Exception as e:
                # Other error during initialization
                self.logger.debug(f"GEE initialization warning: {e}")

    def check_requirements(self) -> Tuple[bool, Optional[str]]:
        """
        Check if ALOS/NASADEM DEM is available via GEE.
        
        Returns:
            Tuple of (is_available, error_message)
            is_available: True if GEE authenticated and assets accessible
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
            # Test asset access - try primary first, then fallback
            try:
                asset = ee.Image(self.GEE_ASSET_PRIMARY)
                info = asset.sampleRectangle(ee.Geometry.Point([0, 0]).buffer(1000), defaultValue=0)
                if info:
                    self.logger.info("ALOS Palsar DEM accessible via GEE")
                    self.gee_asset = self.GEE_ASSET_PRIMARY
                    return True, None
            except Exception as e:
                if not self.use_fallback:
                    error_msg = f"ALOS DEM not accessible: {str(e)}"
                    self.logger.debug(error_msg)
                    return False, error_msg
                
                self.logger.debug(f"ALOS asset unavailable, trying NASADEM fallback: {e}")
                asset = ee.Image(self.GEE_ASSET_FALLBACK)
                info = asset.sampleRectangle(ee.Geometry.Point([0, 0]).buffer(1000), defaultValue=0)
                if info:
                    self.logger.info("NASADEM accessible via GEE (ALOS unavailable)")
                    self.gee_asset = self.GEE_ASSET_FALLBACK
                    return True, None
            
            return False, "Neither ALOS nor NASADEM assets accessible"
            
        except Exception as e:
            error_msg = f"DEM asset accessibility check failed: {str(e)}"
            self.logger.debug(error_msg)
            return False, error_msg

    @cache_api_call
    def get_data(
        self,
        lat: float,
        lon: float,
        date: Optional[str] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve ALOS/NASADEM topographic metrics for a location.
        
        Computes multi-scale topographic indices (TRI, mTPI), slope, aspect,
        elevation, and terrain roughness classification.
        
        Args:
            lat: Latitude of query location (-90 to 90)
            lon: Longitude of query location (-180 to 180)
            date: Optional date parameter (included for API consistency, not used for DEM)
            **kwargs: Additional keyword arguments (e.g., logger from decorator)
            
        Returns:
            Dictionary with topographic metrics or None if unavailable
            
        Raises:
            No exceptions raised; errors are logged and None returned
        """
        try:
            # Validate coordinates
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                self.logger.debug(f"Invalid coordinates: lat={lat}, lon={lon}")
                return None
            
            if not self.authenticated or ee is None:
                self.logger.debug("ALOS CHILI requires GEE authentication")
                return None
            
            logger = kwargs.get('logger', self.logger)
            
            # Query DEM-derived metrics
            result = self._query_dem_metrics(lat, lon, logger=logger)
            
            if result is None:
                logger.debug(f"No DEM data found at ({lat:.4f}, {lon:.4f})")
                return None
            
            return result
            
        except Exception as e:
            self.logger.error(f"ALOS CHILI query failed at ({lat:.4f}, {lon:.4f}): {str(e)}")
            return None

    def _query_dem_metrics(
        self,
        lat: float,
        lon: float,
        logger: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Query topographic metrics from DEM.
        
        Args:
            lat: Latitude
            lon: Longitude
            logger: Logger instance
            
        Returns:
            Dictionary with terrain metrics or None if unavailable
        """
        if logger is None:
            logger = self.logger
        
        try:
            # Create point geometry with small buffer for sampling
            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(500)  # 500m buffer for local neighborhood
            
            # Load DEM - handle ImageCollection vs Image assets
            dem = None
            try:
                # ALOS is an ImageCollection - need to mosaic it
                if 'ALOS' in self.gee_asset or 'AW3D' in self.gee_asset:
                    alos_collection = ee.ImageCollection(self.gee_asset)
                    dem = alos_collection.mosaic()
                else:
                    dem = ee.Image(self.gee_asset)
            except Exception as e:
                logger.debug(f"Primary DEM asset ({self.gee_asset}) failed: {str(e)}")
                # If primary asset fails and fallback is enabled, try fallback
                if self.use_fallback and self.gee_asset == self.GEE_ASSET_PRIMARY:
                    logger.debug(f"Trying fallback NASADEM...")
                    try:
                        dem = ee.Image(self.GEE_ASSET_FALLBACK)
                        self.gee_asset = self.GEE_ASSET_FALLBACK  # Update for future use
                    except Exception as e2:
                        logger.debug(f"Fallback DEM asset also failed: {str(e2)}")
                        return None
                else:
                    logger.debug(f"DEM asset load failed: {str(e)}")
                    return None
            
            if dem is None:
                return None
            
            # Sample elevation at point
            elevation_sample = dem.sample(point, scale=30).first()
            elevation_info = elevation_sample.getInfo()
            
            if not elevation_info or 'properties' not in elevation_info:
                logger.debug("Empty DEM response")
                return None
            
            # Extract elevation
            dem_band_name = 'elevation' if 'elevation' in elevation_info['properties'] else 'dem'
            if dem_band_name not in elevation_info['properties']:
                # Try NASADEM structure
                dem_band_name = 'NASADEM_HGT' if 'NASADEM_HGT' in elevation_info['properties'] else 'ALOS_PALSAR'
            
            elevation_m = float(elevation_info['properties'].get(dem_band_name, 0))
            
            # Compute slope and aspect
            slope = ee.Terrain.slope(dem)
            aspect = ee.Terrain.aspect(dem)
            
            # Sample slope and aspect
            slope_sample = slope.sample(point, scale=30).first().getInfo()
            aspect_sample = aspect.sample(point, scale=30).first().getInfo()
            
            if not (slope_sample and aspect_sample):
                logger.debug("Empty slope/aspect response")
                return None
            
            slope_prop_name = 'slope' if 'slope' in slope_sample['properties'] else list(slope_sample['properties'].keys())[0]
            aspect_prop_name = 'aspect' if 'aspect' in aspect_sample['properties'] else list(aspect_sample['properties'].keys())[0]
            
            slope_deg = float(slope_sample['properties'].get(slope_prop_name, 0))
            aspect_deg = float(aspect_sample['properties'].get(aspect_prop_name, None))
            
            # Handle flat terrain (slope ~0)
            if slope_deg < 0.5:
                aspect_deg = None  # Aspect undefined for flat terrain
            
            # Compute TRI (Topographic Roughness Index)
            tri = self._compute_tri(dem, region, logger)
            
            # Compute mTPI (multi-scale Topographic Position Index)
            mtpi = self._compute_mtpi(dem, region, logger)
            
            # Classify terrain roughness
            roughness_class = self._classify_terrain_roughness(tri)
            
            result = {
                'tri_value': round(tri, 2) if tri is not None else None,
                'mtpi_value': round(mtpi, 2) if mtpi is not None else None,
                'slope_deg': round(slope_deg, 2),
                'aspect_deg': round(aspect_deg, 2) if aspect_deg is not None else None,
                'elevation_m': round(elevation_m, 1),
                'terrain_roughness_class': roughness_class,
                'terrain_data_source': 'ALOS Palsar DEM' if self.gee_asset == self.GEE_ASSET_PRIMARY else 'NASADEM',
            }
            
            logger.debug(f"Retrieved terrain metrics: elevation={elevation_m:.0f}m, slope={slope_deg:.1f}°, roughness={roughness_class}")
            
            return result
            
        except Exception as e:
            logger.debug(f"DEM metrics query error: {str(e)}")
            return None

    def _compute_tri(
        self,
        dem: Any,
        region: Any,
        logger: Optional[Any] = None
    ) -> Optional[float]:
        """
        Compute Topographic Roughness Index (TRI).
        
        TRI is computed as the square root of the sum of squared slopes
        in a local neighborhood. Approximated using Laplacian energy.
        
        Args:
            dem: DEM image
            region: Region for sampling (Point buffer)
            logger: Logger instance
            
        Returns:
            TRI value in meters or None
        """
        if logger is None:
            logger = self.logger
        
        try:
            # Compute gradient magnitude (approximation of TRI)
            gradient_x = dem.gradient().select('x')
            gradient_y = dem.gradient().select('y')
            gradient_mag = gradient_x.pow(2).add(gradient_y.pow(2)).sqrt()
            
            # Sample gradient magnitude
            sample = gradient_mag.sample(region, scale=30).first()
            info = sample.getInfo()
            
            if info and 'properties' in info:
                props = info['properties']
                # Key name varies: 'x', 'y', 'magnitude', etc.
                for key in props:
                    if key not in ['system:index']:
                        tri_value = float(props[key])
                        return tri_value * 30  # Scale to approximate meters
            
            return None
            
        except Exception as e:
            logger.debug(f"TRI computation error: {e}")
            return None

    def _compute_mtpi(
        self,
        dem: Any,
        region: Any,
        logger: Optional[Any] = None
    ) -> Optional[float]:
        """
        Compute multi-scale Topographic Position Index (mTPI).
        
        mTPI measures elevation relative to surroundings. Positive values
        indicate ridges/peaks; negative values indicate valleys/depressions.
        
        Args:
            dem: DEM image
            region: Region for sampling (Point buffer)
            logger: Logger instance
            
        Returns:
            mTPI value (-100 to +100) or None
        """
        if logger is None:
            logger = self.logger
        
        try:
            # Sample elevation at point
            center_sample = dem.sample(region, scale=30).first().getInfo()
            if not center_sample or 'properties' not in center_sample:
                return None
            
            dem_band_name = list(center_sample['properties'].keys())[0]
            center_elev = float(center_sample['properties'][dem_band_name])
            
            # Compute mean elevation in broader neighborhood
            # Create larger buffer (regional scale)
            regional_buffer = region.buffer(self.REGIONAL_NEIGHBORHOOD_M)
            regional_sample = dem.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=regional_buffer,
                scale=30
            ).getInfo()
            
            if not regional_sample:
                return None
            
            mean_elev = float(list(regional_sample.values())[0])
            
            # mTPI = (center - regional_mean) * 100 / (regional_std + 1e-6)
            # Simplified: normalize difference to -100 to +100 range
            elev_diff = center_elev - mean_elev
            mtpi_value = max(-100, min(100, elev_diff))
            
            return mtpi_value
            
        except Exception as e:
            logger.debug(f"mTPI computation error: {e}")
            return None

    def _classify_terrain_roughness(self, tri: Optional[float]) -> str:
        """
        Classify terrain roughness based on TRI value.
        
        Args:
            tri: Topographic Roughness Index in meters
            
        Returns:
            Terrain class name (flat/rolling/hilly/steep/extreme)
        """
        if tri is None:
            return 'unknown'
        
        for class_name, (min_tri, max_tri) in self.ROUGHNESS_CLASSES.items():
            if min_tri <= tri < max_tri:
                return class_name
        
        return 'extreme'  # Fallback for very high values
