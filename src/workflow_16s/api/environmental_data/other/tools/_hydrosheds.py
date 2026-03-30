"""
HydroSHEDS / HydroATLAS Environmental Data Handler

Provides access to hydrological features through Google Earth Engine.
Integrates HydroSHEDS (Hydrological data and maps based on SHuttle Elevation
Derivatives at multiple scales) and HydroATLAS (a comprehensive global
geospatial database of river basins, catchments, and sub-watersheds).

HydroSHEDS Datasets:
- Flow direction (D8 direction to next cell)
- Flow accumulation (catchment area upstream of cell)
- Watershed boundaries at multiple scales

HydroATLAS Datasets:
- Delineated basins at 15-second resolution
- Contributing upstream area
- River reach attributes
- Catchment properties

Variables:
- Upstream drainage area (km²)
- Flow direction (categorical: N, NE, E, SE, S, SW, W, NW)
- Basin ID (HydroATLAS basin identifier)
- Upstream flow accumulation (number of upstream cells)

Hydrological data is essential for understanding:
- Sample collection context (river networks, drainage patterns)
- Watershed-scale processes and properties
- Connectivity and flow pathways
- Microbial dispersal patterns

Coverage: Global
Resolution: HydroSHEDS (15s ≈ 500m), HydroATLAS (basin-level)
Data Sources: SRTM, ASTER, local DEM data
GEE Assets:
- HydroSHEDS: WWF/HydroSHEDS/03VFLS (flow lines)
- HydroATLAS: WWF/HydroATLAS/v1/Basins_15s

Reference: https://www.hydrosheds.org/
          https://www.hydrosheds.org/pages/hydroatlas
"""

import logging
from typing import Dict, Any, Optional, Tuple
import json

try:
    import ee
except ImportError:
    ee = None

from .base import BaseEnvironmentalAPI, REQUEST_TIMEOUT
from .cache import cache_api_call
from workflow_16s.utils.logger import get_logger, with_logger


@with_logger
class HydroSHEDSAPI(BaseEnvironmentalAPI):
    """
    Query HydroSHEDS hydrological features through Google Earth Engine.
    
    Features:
    - Global hydrological data at high resolution
    - Flow direction and accumulation
    - Watershed/basin delineation
    - Multiple spatial scales available
    - Requires GEE authentication
    
    Returns:
    Dictionary with hydrological features:
    - hydrosheds_upstream_area_km2: Contributing upstream area (km²)
    - hydrosheds_flow_direction: Primary flow direction (N, NE, E, SE, S, SW, W, NW)
    - hydrosheds_flow_accumulation: Number of upstream cells
    - hydrosheds_basin_id: HydroATLAS basin identifier
    - hydrosheds_is_riverine: Whether location is on river network (boolean)
    - hydrosheds_watershed_scale: Scale of watershed delineation (15s native)
    - data_source: Dataset source (HydroSHEDS or HydroATLAS)
    
    Example:
        api = HydroSHEDSAPI()
        is_available, msg = api.check_requirements()
        if is_available:
            hydro_data = api.get_data(lat=0.0, lon=25.0)
            if hydro_data:
                upstream_km2 = hydro_data.get('hydrosheds_upstream_area_km2')
                flow_dir = hydro_data.get('hydrosheds_flow_direction')
    """
    
    API_NAME = "HydroSHEDS"
    
    # GEE Asset IDs
    HYDROSHEDS_FLOW_LINES = "WWF/HydroSHEDS/03VFLS"  # Flow lines globally
    HYDROATLAS_BASINS = "WWF/HydroATLAS/v1/Basins_15s"  # Basin boundaries
    HYDROSHEDS_DEM = "WWF/HydroSHEDS/DEM30"  # DEM-derived products
    
    # Flow direction bit encoding (D8 convention)
    FLOW_DIRECTIONS = {
        1: 'E',      # East
        2: 'SE',     # Southeast
        4: 'S',      # South
        8: 'SW',     # Southwest
        16: 'W',     # West
        32: 'NW',    # Northwest
        64: 'N',     # North
        128: 'NE',   # Northeast
    }
    
    def __init__(self, verbose: bool = False, authenticated: bool = True):
        """
        Initialize HydroSHEDS API client.
        
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
                self.logger.debug("Google Earth Engine initialized for HydroSHEDS API")
            except ee.EEException as e:
                # GEE already initialized or authentication issue
                self.logger.debug(f"GEE initialization note: {e}")
            except Exception as e:
                # Other error during initialization
                self.logger.debug(f"GEE initialization warning: {e}")

    def check_requirements(self) -> Tuple[bool, Optional[str]]:
        """
        Check if HydroSHEDS is available.
        
        Returns:
            Tuple of (is_available, error_message)
            is_available: True if GEE is initialized and authenticated
            error_message: None if available, error description otherwise
            
        NOTE:
        The HydroATLAS asset 'WWF/HydroATLAS/v1/Basins_15s' may not be accessible
        if the service account doesn't have permissions or if the asset path has
        changed in the GEE catalog. This is gracefully handled by returning False
        and allowing other data sources to be used.
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
            # Test asset access to HydroATLAS basins
            # Note: This asset may not be available or accessible to all service accounts
            asset = ee.FeatureCollection(self.HYDROATLAS_BASINS)
            info = asset.first().getInfo()
            if info:
                self.logger.info("HydroSHEDS GEE assets accessible")
                return True, None
        except Exception as e:
            error_msg = f"HydroSHEDS GEE assets not accessible: {str(e)}"
            self.logger.debug(error_msg)
            # Provide informative message about asset availability
            if "not found" in str(e).lower():
                self.logger.debug(
                    f"Asset '{self.HYDROATLAS_BASINS}' not found or not accessible. "
                    f"Possible causes: (1) Asset path changed, (2) Service account lacks permissions, "
                    f"(3) Asset was deprecated. HydroSHEDS data will not be available."
                )
            return False, error_msg
        
        return True, None

    @cache_api_call
    def get_data(
        self,
        lat: float,
        lon: float,
        fetch_date: Optional[str] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve HydroSHEDS hydrological features for a location.
        
        Extracts flow direction, upstream area, and basin information
        from HydroSHEDS and HydroATLAS datasets.
        
        Args:
            lat: Latitude of query location (-90 to 90)
            lon: Longitude of query location (-180 to 180)
            fetch_date: Optional date parameter (not used; included for API consistency)
            **kwargs: Additional keyword arguments (e.g., logger from decorator)
            
        Returns:
            Dictionary with hydrological features or None if unavailable
            
        Raises:
            No exceptions raised; errors are logged and None returned
        """
        try:
            # Validate coordinates
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                self.logger.debug(f"Invalid coordinates: lat={lat}, lon={lon}")
                return None
            
            if not self.authenticated or ee is None:
                self.logger.debug("HydroSHEDS requires GEE authentication")
                return None
            
            logger = kwargs.get('logger', self.logger)
            
            # Query HydroSHEDS data
            result = self._query_hydrosheds(lat, lon, logger=logger)
            
            if result is None:
                logger.debug(f"No HydroSHEDS data found at ({lat:.4f}, {lon:.4f})")
                return None
            
            return result
            
        except Exception as e:
            self.logger.error(f"HydroSHEDS query failed at ({lat:.4f}, {lon:.4f}): {str(e)}")
            return None

    def _query_hydrosheds(
        self,
        lat: float,
        lon: float,
        logger: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Query HydroSHEDS data from GEE.
        
        Combines HydroSHEDS flow data and HydroATLAS basin information.
        
        Args:
            lat: Latitude
            lon: Longitude
            logger: Logger instance
            
        Returns:
            Dictionary with hydrological data or None if unavailable
        """
        if logger is None:
            logger = self.logger
        
        try:
            # Create point geometry
            point = ee.Geometry.Point([lon, lat])
            
            # Query HydroATLAS basins (vector data)
            basin_data = self._query_hydroatlas_basins(point, logger)
            
            # Query HydroSHEDS flow products (raster data)
            flow_data = self._query_hydrosheds_flow(point, logger)
            
            # Combine results
            result = {}
            if basin_data:
                result.update(basin_data)
            if flow_data:
                result.update(flow_data)
            
            if not result:
                logger.debug("No HydroSHEDS or HydroATLAS data found")
                return None
            
            logger.debug(f"Retrieved HydroSHEDS data with {len(result)} properties")
            return result
            
        except Exception as e:
            logger.debug(f"HydroSHEDS GEE query error: {str(e)}")
            return None

    def _query_hydroatlas_basins(
        self,
        point: Any,
        logger: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Query HydroATLAS basin features.
        
        Finds the basin containing the point and extracts properties.
        
        Args:
            point: ee.Geometry.Point object
            logger: Logger instance
            
        Returns:
            Dictionary with basin properties or None
        """
        if logger is None:
            logger = self.logger
        
        try:
            # Load HydroATLAS basins
            basins = ee.FeatureCollection(self.HYDROATLAS_BASINS)
            
            # Filter basins that contain the point
            basin_at_point = basins.filterBounds(point).first()
            
            if basin_at_point is None:
                logger.debug("No HydroATLAS basin found at location")
                return None
            
            # Extract basin properties
            properties = basin_at_point.getInfo()
            
            if not properties or 'properties' not in properties:
                return None
            
            props = properties['properties']
            result = {}
            
            # Extract relevant properties
            if 'BASIN_ID' in props:
                result['hydrosheds_basin_id'] = props['BASIN_ID']
            if 'UP_AREA' in props:
                # UP_AREA is in km²
                result['hydrosheds_upstream_area_km2'] = float(props['UP_AREA'])
            elif 'UP_ELEV_M' in props:
                # Some datasets may have different naming
                result['hydrosheds_upstream_area_km2'] = float(props.get('UP_ELEV_M', 0))
            
            result['hydrosheds_watershed_scale'] = '15_seconds'
            result['data_source'] = 'HydroATLAS'
            
            logger.debug(f"Retrieved HydroATLAS basin: {result.get('hydrosheds_basin_id')}")
            
            return result if result else None
            
        except Exception as e:
            logger.debug(f"HydroATLAS query error: {str(e)}")
            return None

    def _query_hydrosheds_flow(
        self,
        point: Any,
        logger: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Query HydroSHEDS flow direction and accumulation.
        
        Uses raster data products for flow analysis.
        
        Args:
            point: ee.Geometry.Point object
            logger: Logger instance
            
        Returns:
            Dictionary with flow properties or None
        """
        if logger is None:
            logger = self.logger
        
        try:
            # Load HydroSHEDS flow lines (using as proxy for river network)
            flow_lines = ee.FeatureCollection(self.HYDROSHEDS_FLOW_LINES)
            
            # Check if point is near a flow line (river network)
            buffer = point.buffer(500)  # 500m buffer
            intersecting = flow_lines.filterBounds(buffer)
            
            is_riverine = intersecting.size().getInfo() > 0
            
            result = {}
            result['hydrosheds_is_riverine'] = is_riverine
            result['data_source'] = 'HydroSHEDS'
            
            # Try to get flow direction and accumulation from raster products
            # Note: This requires DEM-derived products which may have different naming
            try:
                dem = ee.Image(self.HYDROSHEDS_DEM)
                
                if dem is not None:
                    # Calculate flow direction (D8)
                    flow_dir_image = ee.Terrain.flowDirection(dem)
                    
                    # Sample at point
                    sample = flow_dir_image.sample(point, scale=500)
                    data = sample.first().getInfo()
                    
                    if data and 'properties' in data:
                        flow_val = data['properties'].get('flow_direction')
                        if flow_val:
                            direction = self._decode_flow_direction(int(flow_val))
                            result['hydrosheds_flow_direction'] = direction
                            result['hydrosheds_flow_accumulation'] = int(flow_val)
            except Exception as e:
                logger.debug(f"Could not retrieve raster flow products: {e}")
            
            logger.debug(f"HydroSHEDS flow query: riverine={is_riverine}")
            
            return result if len(result) > 1 else None
            
        except Exception as e:
            logger.debug(f"HydroSHEDS flow query error: {str(e)}")
            return None

    def _decode_flow_direction(self, flow_code: int) -> str:
        """
        Decode D8 flow direction encoding.
        
        D8 uses 8 directions encoded as powers of 2:
        - 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
        
        Args:
            flow_code: D8 encoded direction value
            
        Returns:
            Direction string (N, NE, E, SE, S, SW, W, NW)
        """
        try:
            # Find which bit is set (primary flow direction)
            for bit_val, direction in sorted(self.FLOW_DIRECTIONS.items()):
                if flow_code & bit_val:
                    return direction
            # If no exact match, return the closest
            closest = min(self.FLOW_DIRECTIONS.items(), 
                         key=lambda x: abs(x[0] - flow_code))
            return closest[1]
        except Exception:
            return 'unknown'
