"""
Google Earth Engine Priority Datasets for 16S Pipeline

Integrates high-value global datasets:
- Copernicus DEM (30m global elevation/relief)
- ERA5 (climate reanalysis: temperature, precipitation, humidity)
- WorldCover (ESA 10m global land cover)
- OpenLandMap (climate statistics, historical data)
- MODIS (vegetation indices, land cover)
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    import ee
except ImportError:
    ee = None

logger = logging.getLogger(__name__)


class CopernicusDEMAPI:
    """Access Copernicus 30m Digital Elevation Model globally."""
    
    ASSET_ID = 'COPERNICUS/DEM/GLO30'
    BANDS = ['DEM', 'EDM', 'FLM']  # DEM, Error, Void Filled Flag
    
    def __init__(self, authenticated: bool = False):
        self._authenticated = authenticated and ee is not None
    
    def query_by_point(
        self,
        latitude: float,
        longitude: float,
        scale_m: int = 30
    ) -> Optional[Dict[str, float]]:
        """
        Query elevation and relief at a point.
        
        Returns:
            Dict with elevation (m), slope (degrees), aspect (degrees)
        """
        if not self._authenticated or ee is None:
            return None
        
        try:
            point = ee.Geometry.Point([longitude, latitude])
            image = ee.Image(self.ASSET_ID)
            
            # Calculate slope and aspect
            terrain = ee.Terrain.products(image)
            
            sample = image.addBands(terrain).sample(point, scale_m)
            data = sample.first().getInfo()
            
            if data and 'properties' in data:
                props = data['properties']
                return {
                    'elevation_m': props.get('DEM'),
                    'slope_degrees': props.get('slope'),
                    'aspect_degrees': props.get('aspect'),
                    'relief_class': self._classify_relief(props.get('slope'))
                }
        except Exception as e:
            logger.debug(f"Copernicus DEM query failed: {e}")
        
        return None
    
    @staticmethod
    def _classify_relief(slope_degrees: Optional[float]) -> Optional[str]:
        """Classify slope into categorical relief types."""
        if slope_degrees is None:
            return None
        if slope_degrees < 5:
            return 'flat'
        elif slope_degrees < 15:
            return 'gently_sloping'
        elif slope_degrees < 30:
            return 'moderately_steep'
        else:
            return 'steep'


class ERA5ClimateAPI:
    """Access ERA5 climate reanalysis data (temperature, precipitation, etc.)."""
    
    # Monthly aggregates for faster access
    ASSET_ID = 'ECMWF/ERA5/MONTHLY'
    
    # Key bands for 16S analysis
    CLIMATE_BANDS = [
        'mean_2m_air_temperature',  # °C
        'maximum_2m_air_temperature',  # °C
        'minimum_2m_air_temperature',  # °C
        'total_precipitation',  # mm
        'mean_total_column_water_vapour',  # kg/m²
        'mean_sea_level_pressure',  # Pa
        'mean_surface_sensible_heat_flux',  # W/m²
        'mean_surface_latent_heat_flux',  # W/m²
    ]
    
    def __init__(self, authenticated: bool = False):
        self._authenticated = authenticated and ee is not None
    
    def query_by_point(
        self,
        latitude: float,
        longitude: float,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[Dict[str, float]]:
        """
        Query climate data for a point over time period.
        
        Args:
            latitude: Sample latitude
            longitude: Sample longitude
            start_date: ISO format (default: 1 year ago)
            end_date: ISO format (default: today)
            
        Returns:
            Dict with mean/min/max climate variables
        """
        if not self._authenticated or ee is None:
            return None
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).isoformat()
        if end_date is None:
            end_date = datetime.now().isoformat()
        
        try:
            point = ee.Geometry.Point([longitude, latitude])
            collection = ee.ImageCollection(self.ASSET_ID)\
                .filterBounds(point)\
                .filterDate(start_date, end_date)
            
            # Calculate mean/min/max
            mean_image = collection.mean()
            
            sample = mean_image.select(self.CLIMATE_BANDS).sample(point, 10000)
            data = sample.first().getInfo()
            
            if data and 'properties' in data:
                return data['properties']
        except Exception as e:
            logger.debug(f"ERA5 query failed: {e}")
        
        return None


class WorldCoverLandUseAPI:
    """ESA WorldCover 10m global land cover classification."""
    
    ASSET_LATEST = 'ESA/WorldCover/v200'  # v200 from 2021, most recent
    
    # Land cover classes
    CLASSES = {
        10: 'tree_cover',
        20: 'shrubland',
        30: 'herbaceous',
        40: 'herbaceous_wetland',
        50: 'moss_lichen',
        60: 'open_water',
        70: 'clouds_shadows',
        80: 'snow_ice',
        90: 'herbaceous_tundra',
        95: 'barren_sparse',
        100: 'built_up'
    }
    
    def __init__(self, authenticated: bool = False):
        self._authenticated = authenticated and ee is not None
    
    def query_by_point(
        self,
        latitude: float,
        longitude: float,
        buffer_km: float = 1.0
    ) -> Optional[Dict[str, float]]:
        """
        Query land cover around a point.
        
        Returns:
            Dict with percentage of each land cover class within buffer
        """
        if not self._authenticated or ee is None:
            return None
        
        try:
            point = ee.Geometry.Point([longitude, latitude])
            buffer = point.buffer(buffer_km * 1000)  # Convert to meters
            
            image = ee.Image(self.ASSET_LATEST)
            
            # Get histogram of land cover classes
            histogram = image.reduceRegion(
                reducer=ee.Reducer.histogram(),
                geometry=buffer,
                scale=10,
                maxPixels=1e6
            )
            
            return histogram.getInfo()
        except Exception as e:
            logger.debug(f"WorldCover query failed: {e}")
        
        return None


class OpenLandMapClimateAPI:
    """OpenLandMap climate statistics and historical data."""
    
    # Monthly precipitation normals from MODIS
    PRECIPITATION_ASSET = 'OpenLandMap/CLM/CLM_PRECIPITATION_SM2RAIN_M/v01'
    
    # Land surface temperature day/night
    LST_DAY_ASSET = 'OpenLandMap/CLM/CLM_LST_MOD11A2-DAY_M/v01'
    LST_NIGHT_ASSET = 'OpenLandMap/CLM/CLM_LST_MOD11A2-DAYNIGHT_M/v01'
    
    def __init__(self, authenticated: bool = False):
        self._authenticated = authenticated and ee is not None
    
    def query_annual_climate(
        self,
        latitude: float,
        longitude: float
    ) -> Optional[Dict[str, float]]:
        """
        Query annual climate statistics.
        
        Returns:
            Dict with monthly precipitation, temperature, etc.
        """
        if not self._authenticated or ee is None:
            return None
        
        try:
            point = ee.Geometry.Point([longitude, latitude])
            
            # Get monthly precipitation
            precip_image = ee.Image(self.PRECIPITATION_ASSET)
            precip_sample = precip_image.sample(point, 1000)
            precip_data = precip_sample.first().getInfo()
            
            if precip_data and 'properties' in precip_data:
                # Calculate annual total and mean monthly
                months = [k for k in precip_data['properties'].keys() 
                         if k in ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']]
                
                if months:
                    values = [precip_data['properties'].get(m, 0) for m in months]
                    return {
                        'annual_precipitation_mm': sum(values),
                        'mean_monthly_precipitation_mm': np.mean(values),
                        'precipitation_seasonality': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                    }
        except Exception as e:
            logger.debug(f"OpenLandMap query failed: {e}")
        
        return None


class JRCGlobalSurfaceWaterAPI:
    """Access JRC Global Surface Water dataset for water occurrence and seasonality."""
    
    ASSET_ID = 'JRC/GSW1_4/GlobalSurfaceWater'
    
    def __init__(self, authenticated: bool = False):
        self._authenticated = authenticated and ee is not None
    
    def query_by_point(
        self,
        latitude: float,
        longitude: float,
        scale_m: int = 30
    ) -> Optional[Dict[str, float]]:
        """
        Query water occurrence and seasonality at a point.
        
        Returns:
            Dict with:
            - occurrence_pct: 0-100% water occurrence across the year
            - seasonality_month: 1-12 month of peak water extent (if seasonal)
            - recurrence_pct: 0-100% recurrence of water detection
        """
        if not self._authenticated or ee is None:
            return None
        
        try:
            point = ee.Geometry.Point([longitude, latitude])
            image = ee.Image(self.ASSET_ID)
            
            sample = image.sample(point, scale_m)
            data = sample.first().getInfo()
            
            if data and 'properties' in data:
                props = data['properties']
                return {
                    'jrc_water_occurrence_pct': props.get('occurrence'),
                    'jrc_water_seasonality_month': props.get('seasonality'),
                    'jrc_water_recurrence_pct': props.get('recurrence')
                }
        except Exception as e:
            logger.debug(f"JRC Global Surface Water query failed: {e}")
        
        return None


class VIIRSNighttimeLightsAPI:
    """Access VIIRS and DMSP nighttime lights data."""
    
    VIIRS_ASSET_ID = 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG'
    DMSP_ASSET_ID = 'NOAA/DMSP-OLS/NIGHTTIME_LIGHTS'
    
    def __init__(self, authenticated: bool = False):
        self._authenticated = authenticated and ee is not None
    
    def query_by_point(
        self,
        latitude: float,
        longitude: float,
        year: int = 2020,
        scale_m: int = 463
    ) -> Optional[Dict[str, float]]:
        """
        Query nighttime lights radiance (VIIRS preferred, fallback to DMSP).
        
        Args:
            latitude: Sample latitude
            longitude: Sample longitude
            year: Year for which to extract data (default 2020)
            scale_m: Resolution in meters (VIIRS: 463m, DMSP: 1000m)
            
        Returns:
            Dict with:
            - radiance_nanoW_cm2_sr: Nighttime light radiance
            - source: 'VIIRS' or 'DMSP'
        """
        if not self._authenticated or ee is None:
            return None
        
        try:
            point = ee.Geometry.Point([longitude, latitude])
            
            # Try VIIRS first (post-2012, better quality, 463m resolution)
            try:
                viirs = ee.ImageCollection(self.VIIRS_ASSET_ID)
                viirs_filtered = viirs.filterDate(
                    f'{year}-01-01', f'{year}-12-31'
                ).median()
                
                sample = viirs_filtered.sample(point, scale_m)
                data = sample.first().getInfo()
                
                if data and 'properties' in data:
                    props = data['properties']
                    avg_rad = props.get('avg_rad')
                    if avg_rad is not None:
                        return {
                            'lights_radiance_nanoW_cm2_sr': avg_rad,
                            'lights_source': 'VIIRS'
                        }
            except Exception as e:
                logger.debug(f"VIIRS query failed, trying DMSP: {e}")
            
            # Fallback to DMSP (1992-2013, coarser 1000m resolution)
            dmsp = ee.ImageCollection(self.DMSP_ASSET_ID)
            dmsp_filtered = dmsp.filterDate(
                f'{year}-01-01', f'{year}-12-31'
            ).median()
            
            sample = dmsp_filtered.sample(point, 1000)
            data = sample.first().getInfo()
            
            if data and 'properties' in data:
                props = data['properties']
                stable_lights = props.get('stable_lights')
                if stable_lights is not None:
                    return {
                        'lights_radiance_nanoW_cm2_sr': stable_lights,
                        'lights_source': 'DMSP'
                    }
        except Exception as e:
            logger.debug(f"Nighttime lights query failed: {e}")
        
        return None


class HansenGlobalForestChangeAPI:
    """Access Hansen Global Forest Change dataset for tree cover loss/gain."""
    
    # Updated to latest available version (2023 v1_10)
    ASSET_ID = 'UMD/hansen/global_forest_change_2023_v1_10'
    
    def __init__(self, authenticated: bool = False):
        self._authenticated = authenticated and ee is not None
    
    def query_by_point(
        self,
        latitude: float,
        longitude: float,
        year: int = 2020,
        scale_m: int = 30
    ) -> Optional[Dict[str, float]]:
        """
        Query Hansen Global Forest Change metrics.
        
        Args:
            latitude: Sample latitude
            longitude: Sample longitude
            year: Reference year (used to calculate loss year)
            scale_m: Resolution in meters (30m native)
            
        Returns:
            Dict with:
            - tree_cover_2000_pct: Tree cover percentage in year 2000
            - forest_loss_binary: 1 if loss detected, 0 otherwise
            - forest_gain_pct: Forest gain percentage (2000-2012)
            - loss_year: Year of loss (if applicable)
        """
        if not self._authenticated or ee is None:
            return None
        
        try:
            point = ee.Geometry.Point([longitude, latitude])
            image = ee.Image(self.ASSET_ID)
            
            sample = image.sample(point, scale_m)
            data = sample.first().getInfo()
            
            if data and 'properties' in data:
                props = data['properties']
                
                result = {
                    'hansen_tree_cover_2000_pct': props.get('treecover2000'),
                    'hansen_forest_loss_binary': props.get('loss', 0),
                    'hansen_forest_gain_pct': props.get('gain', 0),
                    'hansen_loss_year_raw': props.get('lossyear')
                }
                
                # Convert loss year (since 2000) to calendar year
                if result['hansen_loss_year_raw'] and result['hansen_forest_loss_binary'] == 1:
                    result['hansen_loss_year_calendar'] = 2000 + result['hansen_loss_year_raw']
                else:
                    result['hansen_loss_year_calendar'] = None
                
                # Remove the raw value; we only need calendar year
                del result['hansen_loss_year_raw']
                
                return result
        except Exception as e:
            logger.debug(f"Hansen Global Forest Change query failed: {e}")
        
        return None


def get_global_gee_datasets() -> Dict[str, Dict]:
    """
    Return metadata for all integrated GEE datasets.
    
    Useful for documentation and discovery.
    """
    return {
        'copernicus_dem': {
            'title': 'Copernicus 30m DEM (GLO30)',
            'coverage': 'Global',
            'resolution': '30m',
            'variables': ['elevation', 'slope', 'aspect', 'relief'],
            'asset': 'COPERNICUS/DEM/GLO30'
        },
        'era5_climate': {
            'title': 'ERA5 Climate Reanalysis',
            'coverage': 'Global',
            'resolution': '31km',
            'variables': ['temperature', 'precipitation', 'humidity', 'pressure'],
            'asset': 'ECMWF/ERA5/MONTHLY'
        },
        'worldcover_lulc': {
            'title': 'ESA WorldCover 10m',
            'coverage': 'Global',
            'resolution': '10m',
            'variables': ['land_cover_class'],
            'asset': 'ESA/WorldCover/v200'
        },
        'openlandmap_climate': {
            'title': 'OpenLandMap Climate Statistics',
            'coverage': 'Global',
            'resolution': '1km',
            'variables': ['precipitation_monthly', 'temperature_monthly'],
            'asset': 'OpenLandMap/CLM/*'
        },
        'hansen_gfc': {
            'title': 'Hansen Global Forest Change 2023 v1.10',
            'coverage': 'Global',
            'resolution': '30m',
            'variables': ['tree_cover_2000', 'forest_loss', 'forest_gain', 'loss_year'],
            'asset': 'UMD/hansen/global_forest_change_2023_v1_10'
        },
        'jrc_water': {
            'title': 'JRC Global Surface Water',
            'coverage': 'Global',
            'resolution': '30m',
            'variables': ['water_occurrence', 'water_seasonality', 'water_recurrence'],
            'asset': 'JRC/GSW1_4/GlobalSurfaceWater'
        },
        'viirs_lights': {
            'title': 'VIIRS & DMSP Nighttime Lights',
            'coverage': 'Global',
            'resolution': '463m (VIIRS, 1000m DMSP fallback)',
            'variables': ['radiance_nanoW_cm2_sr', 'source'],
            'asset': 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG / NOAA/DMSP-OLS/NIGHTTIME_LIGHTS'
        }
    }
