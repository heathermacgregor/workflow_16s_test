"""
Google Earth Engine Integration for Metal Selection Pressure Analysis

Leverages GEE's vast geospatial datasets:
- USGS Geological Maps (lithology, formation)
- Mineral composition mapping
- Mining proximity and activity detection
- Soil property databases at fine resolution
- Elevation and topography (relates to weathering)

Provides both point queries (sample location) and spatial aggregation (buffers).
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Earth Engine import with graceful handling
try:
    import ee
    EARTH_ENGINE_AVAILABLE = True
except ImportError:
    EARTH_ENGINE_AVAILABLE = False
    logger.warning("Earth Engine API not installed. Install with: pip install earthengine-api")


@dataclass
class EEGeologyResult:
    """Result from Earth Engine geology query"""
    latitude: float
    longitude: float
    lithology: Optional[str]
    formation: Optional[str]
    rock_type: Optional[str]
    elevation_m: Optional[float]
    slope_percent: Optional[float]
    mining_proximity_km: Optional[float]  # Distance to nearest mining activity
    metal_mineral_index: Optional[float]  # 0-1 score based on mineral spectroscopy
    query_timestamp: str


@dataclass
class EEElementResult:
    """Result from Earth Engine element/mineral composition query"""
    latitude: float
    longitude: float
    clay_percent: Optional[float]
    silt_percent: Optional[float]
    sand_percent: Optional[float]
    organic_carbon_percent: Optional[float]
    iron_oxide_percent: Optional[float]  # Proxy for iron-bearing minerals
    aluminum_oxide_percent: Optional[float]
    source_dataset: str
    query_timestamp: str


class GoogleEarthEngineClient:
    """
    Client for accessing geospatial data via Google Earth Engine.
    
    Provides:
    - Point-based queries at sample locations
    - Spatial aggregation (circular buffers around samples)
    - Geologic classification (USGS data)
    - Mineral composition inference (spectral indices)
    - Mining proximity analysis
    
    Auth:
    - Reads service account credentials from config
    - Authenticates once per session
    """
    
    def __init__(
        self,
        service_account_path: Optional[str] = None,
        project_id: Optional[str] = None,
        buffer_km: float = 5.0  # Buffer radius for spatial aggregation
    ):
        """
        Initialize Earth Engine client.
        
        Args:
            service_account_path: Path to GEE service account JSON
            project_id: GCP project ID
            buffer_km: Buffer radius (km) for spatial aggregation
        """
        self.service_account_path = service_account_path
        self.project_id = project_id
        self.buffer_km = buffer_km
        self.authenticated = False
        
        if EARTH_ENGINE_AVAILABLE:
            self._authenticate()
        else:
            logger.warning("Earth Engine not available; GEE features disabled")
    
    def _authenticate(self) -> None:
        """Authenticate with Google Earth Engine"""
        if not EARTH_ENGINE_AVAILABLE:
            return
        
        try:
            if self.service_account_path:
                # Authenticate with service account
                credentials = ee.ServiceAccountCredentials(
                    email=None,
                    key_file=self.service_account_path
                )
                ee.Initialize(credentials, project=self.project_id)
            else:
                # Try default credential chain
                ee.Initialize(project=self.project_id)
            
            self.authenticated = True
            logger.info("Earth Engine authenticated successfully")
        
        except Exception as e:
            logger.error(f"Earth Engine authentication failed: {e}")
            logger.info("Falling back to non-GEE data sources")
    
    def query_geology_at_point(
        self,
        latitude: float,
        longitude: float
    ) -> Optional[EEGeologyResult]:
        """
        Query lithology and geology at a single point (sample location).
        
        Uses:
        - USGS Geological Maps (when available in GEE)
        - GEBCO elevation data
        - Mining proximity (inferred from WorldBank data)
        
        Args:
            latitude: Sample latitude (WGS84)
            longitude: Sample longitude (WGS84)
        
        Returns:
            EEGeologyResult or None if query fails
        """
        if not self.authenticated:
            return None
        
        try:
            point = ee.Geometry.Point([longitude, latitude])
            
            # Query elevation (GEBCO digital elevation model)
            dem = ee.Image('GEBCO/2023')
            elevation = dem.sample(point, scale=30).first().get('elevation')
            elevation_val = elevation.getInfo()
            
            # Compute slope from elevation
            slope = ee.Terrain.slope(dem)
            slope_val = slope.sample(point, scale=30).first().get('slope').getInfo()
            
            # Query lithology (note: actual USGS geology may not be directly in GEE)
            # Fallback: use rock type proxies from elevation + slope + spectral data
            lithology = self._infer_lithology_from_dem(elevation_val, slope_val)
            
            # Query mining proximity (simplified - would need specific mining dataset)
            mining_dist = self._estimate_mining_proximity(latitude, longitude)
            
            # Query mineral index (from Landsat/Sentinel spectral data)
            mineral_index = self._compute_mineral_index_at_point(point)
            
            return EEGeologyResult(
                latitude=latitude,
                longitude=longitude,
                lithology=lithology,
                formation=None,  # Not direct from GEE
                rock_type=lithology,
                elevation_m=elevation_val,
                slope_percent=slope_val,
                mining_proximity_km=mining_dist,
                metal_mineral_index=mineral_index,
                query_timestamp=datetime.now().isoformat()
            )
        
        except Exception as e:
            logger.debug(f"GEE geology query failed: {e}")
            return None
    
    def query_elements_at_point(
        self,
        latitude: float,
        longitude: float
    ) -> Optional[EEElementResult]:
        """
        Query element/soil composition at a point.
        
        Combines:
        - SoilGrids (available in GEE)
        - Landsat-derived soil indices
        - Sentinel-2 spectral minerals
        
        Args:
            latitude: Sample latitude
            longitude: Sample longitude
        
        Returns:
            EEElementResult or None if query fails
        """
        if not self.authenticated:
            return None
        
        try:
            point = ee.Geometry.Point([longitude, latitude])
            
            # Query SoilGrids in GEE
            soilgrids = ee.Image('projects/soilgrids-isric/clay_mean')
            clay = soilgrids.sample(point, scale=250).first().get('clay_0-5cm_mean')
            clay_val = clay.getInfo() / 10.0  # Convert to percent
            
            # Query organic carbon from SoilGrids
            soc_img = ee.Image('projects/soilgrids-isric/soc_mean')
            soc = soc_img.sample(point, scale=250).first().get('soc_0-5cm_mean')
            soc_val = soc.getInfo() / 10.0 if soc else None
            
            # Query spectral mineral indices from Landsat 8
            sentinel = self._get_sentinel2_spectral_indices(point)
            
            return EEElementResult(
                latitude=latitude,
                longitude=longitude,
                clay_percent=clay_val,
                silt_percent=None,
                sand_percent=None,
                organic_carbon_percent=soc_val,
                iron_oxide_percent=sentinel.get('iron_oxide_idx'),
                aluminum_oxide_percent=sentinel.get('aluminum_idx'),
                source_dataset='SoilGrids + Sentinel-2 spectral',
                query_timestamp=datetime.now().isoformat()
            )
        
        except Exception as e:
            logger.debug(f"GEE element query failed: {e}")
            return None
    
    def _infer_lithology_from_dem(
        self,
        elevation_m: Optional[float],
        slope_percent: Optional[float]
    ) -> Optional[str]:
        """
        Infer rock type from DEM characteristics.
        
        Heuristic:
        - High elevation + steep slope → igneous/metamorphic
        - Low elevation + gentle slope → sedimentary
        - Moderate elevation → mixed
        """
        if elevation_m is None or slope_percent is None:
            return None
        
        if elevation_m > 2000 and slope_percent > 15:
            return "igneous"
        elif elevation_m < 500 and slope_percent < 5:
            return "sedimentary"
        elif slope_percent > 25:
            return "metamorphic"
        else:
            return "mixed"
    
    def _estimate_mining_proximity(
        self,
        latitude: float,
        longitude: float
    ) -> Optional[float]:
        """
        Estimate distance to nearest mining activity.
        
        Simplified: returns None for now.
        Full implementation would query mining permits database.
        
        Returns:
            Distance in km or None
        """
        # Would require mining database integration
        return None
    
    def _compute_mineral_index_at_point(self, point: Any) -> Optional[float]:
        """
        Compute mineral richness index from spectral data.
        
        Uses Sentinel-2 SWIR indices that correlate with mineral content.
        
        Returns:
            Score 0-1 indicating mineral richness
        """
        if not EARTH_ENGINE_AVAILABLE:
            return None
        
        try:
            # Sentinel-2 spectral indices related to minerals
            # Higher values = more iron oxides, clays (metal-bearing minerals)
            sentinel = (
                ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(point)
                .filterDate('2020-01-01', '2024-12-31')
                .median()
            )
            
            # Iron oxide index (Sentinel-2 bands)
            red = sentinel.select('B4')
            swir = sentinel.select('B11')
            iron_idx = red.divide(swir)
            
            # Sample and normalize
            value = iron_idx.sample(point, scale=20).first().get('B4').getInfo()
            
            # Normalize to 0-1 (heuristic bounds)
            normalized = max(0, min(1, value / 0.5))
            
            return normalized
        
        except Exception as e:
            logger.debug(f"Mineral index computation failed: {e}")
            return None
    
    def _get_sentinel2_spectral_indices(self, point: Any) -> Dict[str, Optional[float]]:
        """
        Compute multiple spectral indices from Sentinel-2 for mineral/element detection.
        
        Returns:
            Dict with indices
        """
        if not EARTH_ENGINE_AVAILABLE:
            return {'iron_oxide_idx': None, 'aluminum_idx': None}
        
        try:
            sentinel = (
                ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(point)
                .filterDate('2020-01-01', '2024-12-31')
                .median()
            )
            
            # Sample multiple spectral indices
            # These are simplified proxies for mineral composition
            red = sentinel.select('B4')
            swir1 = sentinel.select('B11')
            swir2 = sentinel.select('B12')
            
            iron_oxide = red.divide(swir1).sample(point, scale=20).first()
            aluminum_clay = swir1.divide(swir2).sample(point, scale=20).first()
            
            return {
                'iron_oxide_idx': iron_oxide.getInfo() if iron_oxide else None,
                'aluminum_idx': aluminum_clay.getInfo() if aluminum_clay else None
            }
        
        except Exception as e:
            logger.debug(f"Sentinel-2 index computation failed: {e}")
            return {'iron_oxide_idx': None, 'aluminum_idx': None}


class MetalProxyDataFuser:
    """
    Fuses multiple data sources (USGS, SoilGrids, GEE) into unified metal proxies.
    
    Priority order:
    1. Google Earth Engine (highest resolution, most current)
    2. Direct USGS APIs
    3. SoilGrids + element proxies
    4. Fallback geologic heuristics
    """
    
    def __init__(
        self,
        gee_client: Optional[GoogleEarthEngineClient] = None,
        soilgrids_client: Any = None  # Optional SoilGrids client
    ):
        """
        Initialize data fusion engine.
        
        Args:
            gee_client: Google Earth Engine client
            soilgrids_client: Optional SoilGrids client for fallback
        """
        self.gee_client = gee_client
        self.soilgrids_client = soilgrids_client
    
    def fuse_metal_proxies(
        self,
        latitude: float,
        longitude: float
    ) -> Dict[str, float]:
        """
        Fuse multiple data sources into unified metal enrichment scores.
        
        For each metal, combines:
        - Geologic indicators (GEE + USGS)
        - Element composition (SoilGrids + Sentinel spectral)
        - Mining proximity (GEE mining data)
        - Topographic weathering (GEE DEM)
        
        Returns:
            Dict mapping metal → enrichment score (0-1)
        """
        metal_scores = {}
        
        # Try GEE first
        if self.gee_client and self.gee_client.authenticated:
            gee_geology = self.gee_client.query_geology_at_point(latitude, longitude)
            gee_elements = self.gee_client.query_elements_at_point(latitude, longitude)
            
            if gee_geology and gee_elements:
                metal_scores = self._score_metals_from_gee(gee_geology, gee_elements)
        
        # Fallback if GEE unavailable or incomplete
        if not metal_scores:
            metal_scores = self._score_metals_from_fallback(latitude, longitude)
        
        return metal_scores
    
    def _score_metals_from_gee(
        self,
        geology: EEGeologyResult,
        elements: EEElementResult
    ) -> Dict[str, float]:
        """Score metals based on GEE geology and element data"""
        scores = {}
        
        # Uranium: acidic soils, clay-rich, granitic rocks
        if geology.rock_type == 'igneous' and elements.clay_percent:
            uranium = 0.5 + (elements.clay_percent / 100) * 0.3
            if geology.slope_percent and geology.slope_percent > 10:
                uranium += 0.1  # More weathering = more uranium exposure
            scores['uranium'] = min(1.0, uranium)
        
        # Arsenic: acidic soils, sulfide minerals (high iron oxide)
        if elements.iron_oxide_percent:
            arsenic = 0.3 + (elements.iron_oxide_percent * 0.5)
            scores['arsenic'] = min(1.0, arsenic)
        
        # Copper: mixed mineralogy, clay-rich
        if elements.clay_percent:
            copper = 0.4 + (elements.clay_percent / 100) * 0.3
            if elements.aluminum_oxide_percent:
                copper += elements.aluminum_oxide_percent * 0.2
            scores['copper'] = min(1.0, copper)
        
        # Lead: high organic carbon (sorbs to organics)
        if elements.organic_carbon_percent:
            lead = 0.5 + (elements.organic_carbon_percent / 20) * 0.3
            scores['lead'] = min(1.0, lead)
        
        # General score if mineral index available
        if geology.metal_mineral_index:
            for metal in ['zinc', 'nickel', 'cadmium']:
                if metal not in scores:
                    scores[metal] = geology.metal_mineral_index
        
        return scores
    
    def _score_metals_from_fallback(
        self,
        latitude: float,
        longitude: float
    ) -> Dict[str, float]:
        """Fallback scoring without GEE"""
        # Return neutral scores - geologic/element proxies will fill in
        return {
            'uranium': 0.5,
            'arsenic': 0.5,
            'copper': 0.5,
            'lead': 0.5,
            'zinc': 0.5,
            'nickel': 0.5,
            'cadmium': 0.5
        }


def get_gee_client(config: Optional[Dict[str, Any]] = None) -> Optional[GoogleEarthEngineClient]:
    """
    Factory function for creating GEE client from config.
    
    Args:
        config: Config dict with EARTH_ENGINE credentials
    
    Returns:
        GoogleEarthEngineClient or None if not available
    """
    if not EARTH_ENGINE_AVAILABLE:
        logger.warning("Earth Engine not installed")
        return None
    
    service_account_path = None
    project_id = None
    
    if config and 'EARTH_ENGINE' in config:
        ee_config = config['EARTH_ENGINE']
        service_account_path = ee_config.get('service_account_path')
        project_id = ee_config.get('project_id')
    
    return GoogleEarthEngineClient(
        service_account_path=service_account_path,
        project_id=project_id
    )
