"""
GEE-Based Environmental Enrichment Integration

Orchestrates fetching from all available GEE data sources with prioritization:
- HIGH PRIORITY (Global, Tested):
  * JRC Global Surface Water (30m, global water occurrence/seasonality)
  * VIIRS & DMSP Nighttime Lights (463m VIIRS, 1000m DMSP, global urban/development proxy)
- STANDARD PRIORITY (Global, Validated):
  * Copernicus DEM (30m global elevation/terrain/slope)
  * ERA5 (31km climate reanalysis: temperature, precipitation, humidity)
  * WorldCover (ESA 10m global land cover)
- REGIONAL PRIORITY:
  * ISDASOIL (African soil geochemistry - 21 datasets)
  * OpenLandMap (Climate statistics, historical data)
  * Hansen GFC (30m global forest cover, loss/gain)

This module is called during backfill to enrich metadata with these sources.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)


def enrich_with_gee_data(adata_obs: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """
    Enrich metadata with data from Google Earth Engine sources.
    
    Priority order:
    1. ISDASOIL (African soil geochemistry) - if samples in Africa
    2. Copernicus DEM (global elevation/terrain)
    3. ERA5 (global climate)
    4. WorldCover (global land cover)
    5. OpenLandMap (climate statistics)
    
    Args:
        adata_obs: AnnData observation metadata (pandas DataFrame)
        config: Optional config dict with GEE_AUTHENTICATED flag
        
    Returns:
        Updated obs DataFrame with new columns from GEE sources
    """
    obs = adata_obs.copy()
    
    # Find coordinate columns
    lat_col, lon_col = _find_coordinate_columns(obs)
    if lat_col is None or lon_col is None:
        logger.warning("Could not find coordinate columns for GEE enrichment")
        return obs
    
    # Check if GEE is authenticated
    is_authenticated = config and config.get('GEE_AUTHENTICATED', False) if config else False
    
    if not is_authenticated:
        logger.info("GEE authentication not configured. Skipping GEE enrichment.")
        logger.info("To enable: set GEE_AUTHENTICATED=true in config and authenticate with 'earthengine authenticate'")
        return obs
    
    logger.info(f"Starting GEE enrichment for {len(obs)} samples...")
    
    # Detect which regions have samples
    try:
        lats = pd.to_numeric(obs[lat_col], errors='coerce')
        lons = pd.to_numeric(obs[lon_col], errors='coerce')
        
        valid_mask = (lats.notna()) & (lons.notna())
        
        # Check coverage
        africa_mask = (lats >= -35) & (lats <= 37) & (lons >= -37) & (lons <= 55)
        africa_count = (valid_mask & africa_mask).sum()
        valid_count = valid_mask.sum()
        
        logger.info(f"  Coordinate coverage: {valid_count}/{len(obs)} samples")
        if africa_count > 0:
            logger.info(f"  African coverage: {africa_count} samples ({100*africa_count/valid_count:.1f}%)")
    except Exception as e:
        logger.warning(f"Could not assess sample coverage: {e}")
        return obs
    
    # Try ISDASOIL enrichment (African soil data)
    if africa_count > 0:
        logger.info("Attempting ISDASOIL enrichment (African soil geochemistry)...")
        try:
            obs = _enrich_with_isdasoil(obs, lat_col, lon_col)
        except Exception as e:
            logger.warning(f"ISDASOIL enrichment failed: {e}")
    
    # Try Copernicus DEM enrichment (global elevation)
    logger.info("Attempting Copernicus DEM enrichment (global elevation/terrain)...")
    try:
        obs = _enrich_with_dem(obs, lat_col, lon_col)
    except Exception as e:
        logger.warning(f"Copernicus DEM enrichment failed: {e}")
    
    # Try JRC Water enrichment (HIGH PRIORITY - global water coverage)
    logger.info("Attempting JRC Global Surface Water enrichment (HIGH PRIORITY)...")
    try:
        obs = _enrich_with_jrc_water(obs, lat_col, lon_col)
    except Exception as e:
        logger.warning(f"JRC Water enrichment failed: {e}")
    
    # Try VIIRS/DMSP Nighttime Lights enrichment (HIGH PRIORITY - global coverage)
    logger.info("Attempting VIIRS/DMSP Nighttime Lights enrichment (HIGH PRIORITY)...")
    try:
        obs = _enrich_with_viirs_lights(obs, lat_col, lon_col)
    except Exception as e:
        logger.warning(f"VIIRS/DMSP Lights enrichment failed: {e}")
    
    # Try Hansen Forest Change enrichment (global forest cover)
    logger.info("Attempting Hansen GFC enrichment (forest cover/loss/gain)...")
    try:
        obs = _enrich_with_hansen(obs, lat_col, lon_col)
    except Exception as e:
        logger.warning(f"Hansen GFC enrichment failed: {e}")
    
    # Try ERA5 climate enrichment (global)
    logger.info("Attempting ERA5 climate enrichment...")
    try:
        obs = _enrich_with_era5(obs, lat_col, lon_col)
    except Exception as e:
        logger.warning(f"ERA5 enrichment failed: {e}")
    
    # Try WorldCover land use enrichment (global)
    logger.info("Attempting WorldCover land use enrichment...")
    try:
        obs = _enrich_with_worldcover(obs, lat_col, lon_col)
    except Exception as e:
        logger.warning(f"WorldCover enrichment failed: {e}")
    
    logger.info(f"GEE enrichment complete. New columns: {[c for c in obs.columns if c not in adata_obs.columns]}")
    
    return obs


def _find_coordinate_columns(obs_df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Find latitude and longitude columns in metadata."""
    candidates = [
        ('lat', 'lon'),
        ('latitude', 'longitude'),
        ('LatitudeParsed', 'LongitudeParsed'),
        ('Latitude', 'Longitude'),
    ]
    
    for lat_col, lon_col in candidates:
        if lat_col in obs_df.columns and lon_col in obs_df.columns:
            return lat_col, lon_col
    
    return None, None


def _enrich_with_isdasoil(obs: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    """
    Enrich with ISDASOIL African soil geochemistry data.
    
    IMPORTANT: Filters to African region only before querying.
    ISDASOIL only covers Africa (-35 < lat < 37, -37 < lon < 55).
    Querying non-African samples wastes time and returns no data.
    
    Adds: Al, Fe, Zn, Ca, Mg, K, P, S, N (soil elements)
          pH, CEC, organic C, clay, sand, bedrock depth
    """
    try:
        from workflow_16s.api.environmental_data.google.isdasoil_geochemistry import ISDASoilGeochemistryAPI
    except ImportError:
        logger.warning("ISDASOIL module not available")
        return obs
    
    client = ISDASoilGeochemistryAPI(authenticated=True)
    if not client._authenticated:
        logger.warning("ISDASOIL client not authenticated with GEE")
        return obs
    
    lats = pd.to_numeric(obs[lat_col], errors='coerce')
    lons = pd.to_numeric(obs[lon_col], errors='coerce')
    
    # !!CRITICAL FIX!! Filter to African region BEFORE querying
    # ISDASOIL only covers Africa: -35 < lat < 37 and -37 < lon < 55
    # Without this, all non-African samples are wastefully queried and rejected
    africa_mask = (lats > -35) & (lats < 37) & (lons > -37) & (lons < 55)
    valid_idx = (lats.notna()) & (lons.notna()) & africa_mask
    
    africa_count = valid_idx.sum()
    if africa_count == 0:
        logger.debug("  No African samples found – skipping ISDASOIL")
        return obs
    
    logger.info(f"  Querying ISDASOIL for {africa_count} African samples...")
    
    # Select key properties for metal analysis
    properties = [
        'aluminium_extractable', 'iron_extractable', 'zinc_extractable',
        'calcium_extractable', 'magnesium_extractable', 'potassium_extractable',
        'phosphorus_extractable', 'sulphur_extractable', 'nitrogen_total',
        'clay_content', 'sand_content', 'ph', 'cation_exchange_capacity',
        'carbon_organic', 'bedrock_depth'
    ]
    
    # Extract valid African coordinates only
    coordinates = list(zip(lats[valid_idx], lons[valid_idx]))
    
    if not coordinates:
        logger.warning("No valid African coordinates for ISDASOIL")
        return obs
    
    # Query in batch
    result_df = client.query_batch_points(coordinates, properties)
    
    if result_df.empty:
        logger.warning("ISDASOIL query returned no results")
        return obs
    
    # Merge results back to obs
    valid_indices = obs.index[valid_idx]
    for idx, result_idx in enumerate(valid_indices):
        if idx < len(result_df):
            row = result_df.iloc[idx]
            for col in result_df.columns:
                if col not in ['lat', 'lon']:
                    obs.loc[result_idx, f'ISDASOIL_{col}'] = row[col]
    
    logger.info(f"  ISDASOIL: Added {len(result_df.columns) - 2} properties to {africa_count} African samples")
    
    return obs
    
    return obs


def _enrich_with_dem(obs: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    """
    Enrich with Copernicus 30m DEM elevation and terrain.
    
    Adds: elevation, slope, aspect, relief_class
    """
    try:
        from workflow_16s.api.environmental_data.google.global_gee_datasets import CopernicusDEMAPI
    except ImportError:
        logger.warning("Copernicus DEM module not available")
        return obs
    
    client = CopernicusDEMAPI(authenticated=True)
    if not client._authenticated:
        logger.warning("Copernicus DEM client not authenticated")
        return obs
    
    lats = pd.to_numeric(obs[lat_col], errors='coerce')
    lons = pd.to_numeric(obs[lon_col], errors='coerce')
    valid_idx = (lats.notna()) & (lons.notna())
    
    elevation = []
    slope = []
    aspect = []
    relief = []
    
    for i, (idx, (lat, lon)) in enumerate(zip(obs.index[valid_idx], zip(lats[valid_idx], lons[valid_idx]))):
        if i % 100 == 0:
            logger.debug(f"  Copernicus DEM: {i}/{valid_idx.sum()} samples")
        
        result = client.query_by_point(lat, lon)
        if result:
            elevation.append(result.get('elevation_m'))
            slope.append(result.get('slope_degrees'))
            aspect.append(result.get('aspect_degrees'))
            relief.append(result.get('relief_class'))
        else:
            elevation.append(np.nan)
            slope.append(np.nan)
            aspect.append(np.nan)
            relief.append(None)
    
    # Add to obs
    obs['DEM_elevation_m'] = np.nan
    obs['DEM_slope_degrees'] = np.nan
    obs['DEM_aspect_degrees'] = np.nan
    obs['DEM_relief_class'] = None
    
    obs.loc[obs.index[valid_idx], 'DEM_elevation_m'] = elevation
    obs.loc[obs.index[valid_idx], 'DEM_slope_degrees'] = slope
    obs.loc[obs.index[valid_idx], 'DEM_aspect_degrees'] = aspect
    obs.loc[obs.index[valid_idx], 'DEM_relief_class'] = relief
    
    logger.info(f"  Copernicus DEM: Added elevation, slope, aspect, relief_class")
    
    return obs


def _enrich_with_era5(obs: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    """Enrich with ERA5 climate data (temperature, precipitation, etc.)."""
    try:
        from workflow_16s.api.environmental_data.google.global_gee_datasets import ERA5ClimateAPI
    except ImportError:
        logger.warning("ERA5 module not available")
        return obs
    
    client = ERA5ClimateAPI(authenticated=True)
    if not client._authenticated:
        logger.warning("ERA5 client not authenticated")
        return obs
    
    lats = pd.to_numeric(obs[lat_col], errors='coerce')
    lons = pd.to_numeric(obs[lon_col], errors='coerce')
    valid_idx = (lats.notna()) & (lons.notna())
    
    logger.info(f"  ERA5: Querying {valid_idx.sum()} samples...")
    
    for i, (idx, (lat, lon)) in enumerate(zip(obs.index[valid_idx], zip(lats[valid_idx], lons[valid_idx]))):
        if i % 100 == 0:
            logger.debug(f"  ERA5: {i}/{valid_idx.sum()} samples")
        
        result = client.query_by_point(lat, lon)
        if result:
            for key, value in result.items():
                if key not in obs.columns:
                    obs[f'ERA5_{key}'] = np.nan
                obs.loc[idx, f'ERA5_{key}'] = value
    
    logger.info(f"  ERA5: Added climate variables")
    
    return obs


def _enrich_with_worldcover(obs: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    """Enrich with ESA WorldCover 10m land cover classification."""
    try:
        from workflow_16s.api.environmental_data.google.global_gee_datasets import WorldCoverLandUseAPI
    except ImportError:
        logger.warning("WorldCover module not available")
        return obs
    
    client = WorldCoverLandUseAPI(authenticated=True)
    if not client._authenticated:
        logger.warning("WorldCover client not authenticated")
        return obs
    
    lats = pd.to_numeric(obs[lat_col], errors='coerce')
    lons = pd.to_numeric(obs[lon_col], errors='coerce')
    valid_idx = (lats.notna()) & (lons.notna())
    
    obs['worldcover_primary_class'] = None
    
    for i, (idx, (lat, lon)) in enumerate(zip(obs.index[valid_idx], zip(lats[valid_idx], lons[valid_idx]))):
        if i % 100 == 0:
            logger.debug(f"  WorldCover: {i}/{valid_idx.sum()} samples")
        
        result = client.query_by_point(lat, lon)
        if result:
            obs.loc[idx, 'worldcover_primary_class'] = str(result)
    
    logger.info(f"  WorldCover: Added land cover classification")
    
    return obs


def _enrich_with_hansen(obs: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    """
    Enrich with Hansen Global Forest Change data (tree cover, forest loss/gain).
    
    Adds: tree_cover_2000_pct, forest_loss_binary, forest_gain_pct, loss_year
    """
    try:
        from workflow_16s.api.environmental_data.google.global_gee_datasets import HansenGlobalForestChangeAPI
    except ImportError:
        logger.warning("Hansen GFC module not available")
        return obs
    
    client = HansenGlobalForestChangeAPI(authenticated=True)
    if not client._authenticated:
        logger.warning("Hansen GFC client not authenticated with GEE")
        return obs
    
    lats = pd.to_numeric(obs[lat_col], errors='coerce')
    lons = pd.to_numeric(obs[lon_col], errors='coerce')
    valid_idx = (lats.notna()) & (lons.notna())
    
    logger.info(f"  Querying Hansen GFC for {valid_idx.sum()} samples...")
    
    # Initialize columns
    for col in ['hansen_tree_cover_2000_pct', 'hansen_forest_loss_binary', 
                'hansen_forest_gain_pct', 'hansen_loss_year_calendar']:
        obs[col] = np.nan
    
    for i, (idx, (lat, lon)) in enumerate(zip(obs.index[valid_idx], zip(lats[valid_idx], lons[valid_idx]))):
        if i % 100 == 0:
            logger.debug(f"  Hansen GFC: {i}/{valid_idx.sum()} samples")
        
        result = client.query_by_point(lat, lon)
        if result:
            for key, value in result.items():
                obs.loc[idx, key] = value
    
    logger.info(f"  Hansen GFC: Added forest cover, loss, gain, and loss year")
    
    return obs


def _enrich_with_jrc_water(obs: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    """
    Enrich with JRC Global Surface Water data (water occurrence, seasonality).
    
    Adds: water_occurrence_pct, water_seasonality_month, water_recurrence_pct
    """
    try:
        from workflow_16s.api.environmental_data.google.global_gee_datasets import JRCGlobalSurfaceWaterAPI
    except ImportError:
        logger.warning("JRC Global Surface Water module not available")
        return obs
    
    client = JRCGlobalSurfaceWaterAPI(authenticated=True)
    if not client._authenticated:
        logger.warning("JRC Water client not authenticated with GEE")
        return obs
    
    lats = pd.to_numeric(obs[lat_col], errors='coerce')
    lons = pd.to_numeric(obs[lon_col], errors='coerce')
    valid_idx = (lats.notna()) & (lons.notna())
    
    logger.info(f"  Querying JRC GSW for {valid_idx.sum()} samples...")
    
    # Initialize columns
    for col in ['jrc_water_occurrence_pct', 'jrc_water_seasonality_month', 'jrc_water_recurrence_pct']:
        obs[col] = np.nan
    
    for i, (idx, (lat, lon)) in enumerate(zip(obs.index[valid_idx], zip(lats[valid_idx], lons[valid_idx]))):
        if i % 100 == 0:
            logger.debug(f"  JRC GSW: {i}/{valid_idx.sum()} samples")
        
        result = client.query_by_point(lat, lon)
        if result:
            for key, value in result.items():
                obs.loc[idx, key] = value
    
    logger.info(f"  JRC Global Surface Water: Added water occurrence, seasonality, recurrence")
    
    return obs


def _enrich_with_viirs_lights(obs: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    """
    Enrich with VIIRS & DMSP nighttime lights (radiance, light source).
    
    Adds: lights_radiance_nanoW_cm2_sr, lights_source ('VIIRS' or 'DMSP')
    """
    try:
        from workflow_16s.api.environmental_data.google.global_gee_datasets import VIIRSNighttimeLightsAPI
    except ImportError:
        logger.warning("VIIRS Nighttime Lights module not available")
        return obs
    
    client = VIIRSNighttimeLightsAPI(authenticated=True)
    if not client._authenticated:
        logger.warning("VIIRS Lights client not authenticated with GEE")
        return obs
    
    lats = pd.to_numeric(obs[lat_col], errors='coerce')
    lons = pd.to_numeric(obs[lon_col], errors='coerce')
    valid_idx = (lats.notna()) & (lons.notna())
    
    logger.info(f"  Querying VIIRS/DMSP for {valid_idx.sum()} samples...")
    
    # Initialize columns
    obs['lights_radiance_nanoW_cm2_sr'] = np.nan
    obs['lights_source'] = None
    
    for i, (idx, (lat, lon)) in enumerate(zip(obs.index[valid_idx], zip(lats[valid_idx], lons[valid_idx]))):
        if i % 100 == 0:
            logger.debug(f"  VIIRS/DMSP: {i}/{valid_idx.sum()} samples")
        
        result = client.query_by_point(lat, lon)
        if result:
            obs.loc[idx, 'lights_radiance_nanoW_cm2_sr'] = result.get('lights_radiance_nanoW_cm2_sr')
            obs.loc[idx, 'lights_source'] = result.get('lights_source')
    
    logger.info(f"  VIIRS/DMSP Nighttime Lights: Added radiance and source")
    
    return obs
