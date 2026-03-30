"""
Google Earth Engine Integration for Geospatial Metal Analysis

Leverages GEE datasets for:
- Geologic maps (USGS Geology)
- Soil properties (SoilGrids on GEE - higher resolution than REST API)
- Lithology classification
- Mining proximity analysis

Uses service account credentials from config.
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from dataclasses import dataclass
import json
from datetime import datetime

try:
    import ee
    HAS_EARTHENGINE = True
except ImportError:
    HAS_EARTHENGINE = False
    logger_module = logging.getLogger(__name__)
    logger_module.debug("google-earth-engine not installed; GEE features will be unavailable")

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GEEGeologyResult:
    """Result from GEE geologic query"""
    latitude: float
    longitude: float
    lithology: Optional[str]
    rock_type: Optional[str]
    formation: Optional[str]
    metal_bearing_confidence: float  # 0-1
    data_source: str
    timestamp: str


@dataclass
class GEESoilElement:
    """Soil element composition from GEE"""
    latitude: float
    longitude: float
    clay: Optional[float]  # %
    silt: Optional[float]  # %
    sand: Optional[float]  # %
    organic_carbon: Optional[float]  # dg/kg
    cation_exchange_capacity: Optional[float]  # cmol+/kg
    ph_water: Optional[float]
    bulk_density: Optional[float]  # cg/cm³
    timestamp: str


class EarthEngineGeologyClient:
    """
    Google Earth Engine client for geologic and soil data.
    
    Accesses:
    - USGS 3DEP Lithology (1km resolution)
    - SoilGrids v2.0 (250m resolution)
    - Mining sites and mineral deposits
    
    PERFORMANCE NOTE: Asset availability is checked once at initialization
    and cached. Individual sample queries skip unavailable assets instantly
    without retrying failed API calls.
    """
    
    # GEE Dataset IDs
    DATASETS = {
        'lithology': 'USGS/GMTED2010/SRTM30',  # Primary: DEM (more stable than 3DEP lithology)
        'lithology_fallback': 'USGS/3DEP/Lithology',  # Fallback: 3DEP lithology if primary unavailable
        'soilgrids_clay': 'ISRIC/SOILGRIDS/V2/clay',
        'soilgrids_silt': 'ISRIC/SOILGRIDS/V2/silt',
        'soilgrids_sand': 'ISRIC/SOILGRIDS/V2/sand',
        'soilgrids_soc': 'ISRIC/SOILGRIDS/V2/soc',
        'soilgrids_cec': 'ISRIC/SOILGRIDS/V2/cec',
        'soilgrids_phh2o': 'ISRIC/SOILGRIDS/V2/phh2o',
        'soilgrids_bdod': 'ISRIC/SOILGRIDS/V2/bdod',
        'min_depth': 0,  # meters
        'max_depth': 200  # meters
    }
    
    # Lithology classes to metal associations
    LITHOLOGY_METAL_MAP = {
        'granitic': {'uranium': 0.65, 'thorium': 0.70},
        'mafic': {'iron': 0.90, 'chromium': 0.60, 'nickel': 0.65},
        'ultramafic': {'nickel': 0.95, 'cobalt': 0.85, 'chromium': 0.80},
        'sulfide': {'copper': 0.80, 'zinc': 0.80, 'arsenic': 0.75, 'lead': 0.70},
        'iron_formation': {'iron': 1.0, 'vanadium': 0.60},
        'sedimentary': {'arsenic': 0.50, 'uranium': 0.40},
        'metamorphic': {'arsenic': 0.65, 'gold': 0.50},
    }
    
    def __init__(self, credentials_path: Optional[str] = None, project_id: Optional[str] = None):
        """
        Initialize GEE client.
        
        Args:
            credentials_path: Path to GEE service account JSON
                             If None, uses default authentication
            project_id: Google Cloud project ID for GEE
                       If None, GEE will use default project
        """
        self.credentials_path = credentials_path
        self.project_id = project_id
        self._authenticated = False
        
        # PERFORMANCE: Cache asset availability to avoid repeated failed queries
        self._asset_available: Dict[str, bool] = {}
        
        try:
            self._authenticate(credentials_path, project_id)
            # Validate assets once at startup
            self._validate_assets()
        except Exception as e:
            logger.warning(f"GEE authentication/validation failed: {e}. Will retry on operations.")
    
    def _validate_assets(self) -> None:
        """
        Check availability of all GEE assets upfront.
        
        Stores results in _asset_available dict to avoid repeated
        failed API calls during sample-level queries.
        """
        if not self._authenticated:
            return
        
        logger.info("Validating GEE asset availability...")
        
        # Check each asset once
        for asset_name, asset_id in self.DATASETS.items():
            if isinstance(asset_id, (int, float)):  # Skip numeric values like min_depth
                continue
            
            self._asset_available[asset_id] = self._check_asset_exists(asset_id)
            
            if self._asset_available[asset_id]:
                logger.debug(f"  ✅ Asset '{asset_id}' available")
            else:
                logger.warning(f"  ⚠️  Asset '{asset_id}' unavailable (will skip in sample queries)")
    
    def _check_asset_exists(self, asset_id: str) -> bool:
        """
        Check if a GEE asset exists and is accessible.
        
        PERFORMANCE NOTE: Call this sparingly (only at initialization).
        Results are cached in _asset_available for query-time checks.
        
        Args:
            asset_id: Full GEE asset ID (e.g., 'USGS/3DEP/Lithology')
        
        Returns:
            True if asset exists and is accessible, False otherwise
        """
        if not self._authenticated or not HAS_EARTHENGINE:
            return False
        
        try:
            # Try to load the asset without making a server call if possible
            # For ImageCollections, first() triggers the check
            if 'ImageCollection' in asset_id or '/' in asset_id:
                # Could be ImageCollection or Image
                try:
                    asset = ee.ImageCollection(asset_id).first()
                    _ = asset.getInfo()  # Force evaluation
                    return True
                except Exception:
                    # Not an ImageCollection, try as Image
                    try:
                        asset = ee.Image(asset_id)
                        _ = asset.getInfo()  # Force evaluation
                        return True
                    except Exception:
                        return False
            else:
                # Try as Image by default
                asset = ee.Image(asset_id)
                _ = asset.getInfo()
                return True
        
        except Exception as e:
            error_msg = str(e).lower()
            if any(phrase in error_msg for phrase in ['not found', 'does not exist', '404', 'does_not_exist']):
                logger.debug(f"GEE asset '{asset_id}' not found: {e}")
            else:
                logger.debug(f"GEE asset check failed for '{asset_id}': {e}")
            return False
    
    def is_asset_available(self, asset_id: str) -> bool:
        """
        Quick check if an asset is available (uses cached result).
        
        PERFORMANCE: O(1) lookup, no API call.
        
        Args:
            asset_id: Full GEE asset ID
        
        Returns:
            True if previously validated as available, False otherwise
        """
        return self._asset_available.get(asset_id, False)
    
    def _authenticate(self, credentials_path: Optional[str] = None, project_id: Optional[str] = None) -> None:
        """Authenticate with Google Earth Engine"""
        if not HAS_EARTHENGINE:
            logger.warning("google-earth-engine not installed. Install with: pip install earthengine-api")
            self._authenticated = False
            return
        
        try:
            if credentials_path and Path(credentials_path).exists():
                # Service account authentication
                ee.Authenticate(credentials_path)
            else:
                # Default authentication
                ee.Authenticate()
            
            # Initialize with project if provided
            init_kwargs = {}
            if project_id:
                init_kwargs['project'] = project_id
                logger.debug(f"Initializing GEE with project: {project_id}")
            
            ee.Initialize(**init_kwargs)
            self._authenticated = True
            msg = "Google Earth Engine authenticated successfully"
            if project_id:
                msg += f" (project: {project_id})"
            logger.info(msg)
        except Exception as e:
            logger.debug(f"GEE auth error: {e}")
            self._authenticated = False
    
    def query_geology_by_point(
        self,
        latitude: float,
        longitude: float,
        buffer_m: int = 1000,
        scale: int = 1000
    ) -> Optional[GEEGeologyResult]:
        """
        Query geologic information at a point with automatic fallback.
        
        Tries USGS/GMTED2010 first (more stable), falls back to 3DEP if needed.
        
        Args:
            latitude: Sample latitude (WGS84)
            longitude: Sample longitude (WGS84)
            buffer_m: Buffer radius around point (meters)
            scale: GEE sampling scale (meters)
        
        Returns:
            GEEGeologyResult or None
        
        Note:
            Returns None if GEE is not authenticated or all GEE assets
            are unavailable. Uses fallback logic to maximize coverage.
            
        PERFORMANCE: Asset availability is checked once at initialization.
            If unavailable, falls back instantly (no retry).
        """
        # Try primary asset first, then fallback
        asset_keys = ['lithology', 'lithology_fallback']
        
        for asset_key in asset_keys:
            asset_id = self.DATASETS.get(asset_key)
            if not asset_id:
                continue
                
            if not self.is_asset_available(asset_id):
                logger.debug(f"Asset {asset_key} ({asset_id}) not available, trying next fallback...")
                continue
            
            if not self._authenticated:
                try:
                    self._authenticate(self.credentials_path)
                except:
                    logger.debug("GEE not authenticated; geology queries disabled")
                    return None
            
            try:
                point = ee.Geometry.Point([longitude, latitude])
                buffer_region = point.buffer(buffer_m)
                
                # Try to query the asset
                try:
                    asset_data = ee.ImageCollection(asset_id).first()
                    if not asset_data:
                        asset_data = ee.Image(asset_id)
                except:
                    asset_data = ee.Image(asset_id)
                
                if asset_data:
                    # For elevation assets (GMTED, SRTM), use as proxy
                    if 'GMTED' in asset_id or 'SRTM' in asset_id:
                        try:
                            elevation_sample = asset_data.sample(buffer_region, scale).first()
                            if elevation_sample:
                                elev_info = elevation_sample.getInfo()
                                if elev_info:
                                    return GEEGeologyResult(
                                        latitude=latitude,
                                        longitude=longitude,
                                        lithology='Elevation Proxy',
                                        rock_type='Terrain',
                                        formation=None,
                                        metal_bearing_confidence=0.3,  # Low confidence for proxy
                                        data_source=f'{asset_id} via GEE',
                                        timestamp=datetime.now().isoformat()
                                    )
                        except Exception as e:
                            logger.debug(f"Elevation proxy extraction failed: {e}")
                    
                    # For lithology assets, extract full data
                    else:
                        try:
                            lithology_value = asset_data.sample(buffer_region, scale).first()
                            if lithology_value:
                                lithology_id = lithology_value.get('lithology').getInfo()
                                lithology_name = self._lithology_id_to_name(lithology_id)
                                rock_type = self._lithology_to_rock_type(lithology_name)
                                metal_confidence = self._get_metal_confidence(rock_type)
                                
                                return GEEGeologyResult(
                                    latitude=latitude,
                                    longitude=longitude,
                                    lithology=lithology_name,
                                    rock_type=rock_type,
                                    formation=None,
                                    metal_bearing_confidence=metal_confidence,
                                    data_source=f'{asset_id} via GEE',
                                    timestamp=datetime.now().isoformat()
                                )
                        except Exception as e:
                            logger.debug(f"Lithology extraction failed: {e}")
            
            except Exception as e:
                logger.debug(f"GEE query failed for {asset_key}: {e}")
                continue
        
        logger.debug(f"GEE geology query failed for all fallback assets at ({latitude}, {longitude})")
        return None
    
    def query_soil_elements_by_point(
        self,
        latitude: float,
        longitude: float,
        buffer_m: int = 500,
        scale: int = 250,
        depth_cm: int = 5
    ) -> Optional[GEESoilElement]:
        """
        Query soil element composition at a point.
        
        Args:
            latitude: Sample latitude (WGS84)
            longitude: Sample longitude (WGS84)
            buffer_m: Buffer radius around point (meters)
            scale: GEE sampling scale (meters, default 250m for SoilGrids)
            depth_cm: Soil depth (5, 15, 30, 60, 100, 200)
        
        Returns:
            GEESoilElement with composition data
        """
        if not self._authenticated:
            try:
                self._authenticate(self.credentials_path)
            except:
                return None
        
        try:
            point = ee.Geometry.Point([longitude, latitude])
            buffer_region = point.buffer(buffer_m)
            
            # Map layer names to depth bands
            depth_band = self._get_soilgrids_band(depth_cm)
            
            # Query each soil property
            soil_data = {}
            for property_name, dataset_id in [
                ('clay', self.DATASETS['soilgrids_clay']),
                ('silt', self.DATASETS['soilgrids_silt']),
                ('sand', self.DATASETS['soilgrids_sand']),
                ('soc', self.DATASETS['soilgrids_soc']),
                ('cec', self.DATASETS['soilgrids_cec']),
                ('phh2o', self.DATASETS['soilgrids_phh2o']),
                ('bdod', self.DATASETS['soilgrids_bdod']),
            ]:
                try:
                    collection = ee.ImageCollection(dataset_id)
                    image = collection.first()
                    
                    if image:
                        sample = image.sample(buffer_region, scale).first()
                        if sample:
                            # SoilGrids uses band naming: mean_0_5cm, etc.
                            band_name = f'mean_{self._depth_to_string(depth_cm)}'
                            value = sample.get(band_name).getInfo()
                            
                            if value is not None:
                                soil_data[property_name] = float(value)
                except Exception as e:
                    logger.debug(f"Could not fetch {property_name}: {e}")
            
            if soil_data:
                return GEESoilElement(
                    latitude=latitude,
                    longitude=longitude,
                    clay=soil_data.get('clay'),
                    silt=soil_data.get('silt'),
                    sand=soil_data.get('sand'),
                    organic_carbon=soil_data.get('soc'),
                    cation_exchange_capacity=soil_data.get('cec'),
                    ph_water=soil_data.get('phh2o'),
                    bulk_density=soil_data.get('bdod'),
                    timestamp=datetime.now().isoformat()
                )
            
            return None
        
        except Exception as e:
            logger.debug(f"GEE soil query failed: {e}")
            return None
    
    def query_mining_proximity(
        self,
        latitude: float,
        longitude: float,
        radius_km: int = 50
    ) -> Dict[str, Any]:
        """
        Query proximity to known mining sites.
        
        Returns:
            Dict with mine types and distances
        """
        # This would require mining sites dataset
        # Placeholder implementation
        return {
            'radius_km': radius_km,
            'metal_mines': [],
            'distance_to_nearest_mine_km': None,
            'note': 'Requires mining sites dataset registration'
        }
    
    def _lithology_id_to_name(self, lithology_id: Optional[int]) -> Optional[str]:
        """Convert USGS lithology ID to name"""
        lithology_map = {
            # Simplified mapping (full USGS database has 200+ classes)
            10: 'sedimentary',
            20: 'granitic',
            30: 'mafic',
            40: 'metamorphic',
            50: 'ultramafic',
        }
        return lithology_map.get(lithology_id)
    
    def _lithology_to_rock_type(self, lithology: Optional[str]) -> Optional[str]:
        """Convert lithology to rock type"""
        if not lithology:
            return None
        lithology_lower = lithology.lower()
        for rock_type in self.LITHOLOGY_METAL_MAP.keys():
            if rock_type in lithology_lower:
                return rock_type
        return None
    
    def _get_metal_confidence(self, rock_type: Optional[str]) -> float:
        """Get baseline metal enrichment confidence for rock type"""
        if not rock_type:
            return 0.5
        
        metal_scores = self.LITHOLOGY_METAL_MAP.get(rock_type, {})
        if metal_scores:
            # Return average of metal scores (how metal-rich is this rock type)
            return np.mean(list(metal_scores.values()))
        return 0.5
    
    @staticmethod
    def _get_soilgrids_band(depth_cm: int) -> str:
        """Get SoilGrids band name for depth"""
        depth_ranges = {
            5: '0_5',
            15: '5_15',
            30: '15_30',
            60: '30_60',
            100: '60_100',
            200: '100_200'
        }
        return depth_ranges.get(depth_cm, '0_5')
    
    @staticmethod
    def _depth_to_string(depth_cm: int) -> str:
        """Convert depth to SoilGrids string format"""
        return EarthEngineGeologyClient._get_soilgrids_band(depth_cm)


def get_gee_client(
    config: Optional[Dict[str, Any]] = None
) -> EarthEngineGeologyClient:
    """
    Factory function for creating configured GEE client.
    
    Args:
        config: Config dict with optional 'gee_credentials' and 'credentials' keys
                Looks for:
                  - config['paths']['gee_credentials'] (credentials path)
                  - config['credentials']['google_earth_engine_project'] (project ID)
                  Can be dict or Pydantic model
    
    Returns:
        Configured EarthEngineGeologyClient with project from config
    """
    credentials_path = None
    project_id = None
    
    # Extract credentials path
    if config and 'gee_credentials' in config:
        credentials_path = config['gee_credentials']
    elif config and 'paths' in config and 'gee_credentials' in config['paths']:
        credentials_path = config['paths']['gee_credentials']
    
    # Extract project ID from credentials config (handle both dict and Pydantic model)
    if config and 'credentials' in config:
        creds_config = config['credentials']
        
        # Try as dict first
        if isinstance(creds_config, dict):
            project_id = creds_config.get('google_earth_engine_project')
        else:
            # Try as Pydantic model or object with attributes
            try:
                project_id = getattr(creds_config, 'google_earth_engine_project', None)
            except (AttributeError, TypeError):
                pass
    
    if project_id:
        logger.info(f"Using GEE project from config: {project_id}")
    
    return EarthEngineGeologyClient(credentials_path=credentials_path, project_id=project_id)
