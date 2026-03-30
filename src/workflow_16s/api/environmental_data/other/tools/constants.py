# workflow_16s/api/environmental_data/other/tools/constants.py

from pathlib import Path

from workflow_16s.utils.dir_utils import Project

# TODO: Remove hardcoded paths and values, and initialize these via the Project utility in the main workflow.
CACHE_DB_PATH = Path("project_01/cache/env_other.db")
CACHE_DIR = CACHE_DB_PATH.parent
CACHE_EXPIRY_HOURS = 720 
REQUEST_TIMEOUT = 30
MAX_WORKERS = 20  # Increased from 5 for better concurrent API throughput

# Location batching for efficient concurrent queries (9429 locations × 17 APIs = 160K calls)
LOCATION_BATCH_SIZE = 50  # Query 50 locations in parallel, reduces from 9429 sequential loops

# API-aware rate limiting tiers for optimized throughput
API_RATE_LIMIT_TIERS = {
    # Tier 1: Strict rate limits (≤1 req/sec)
    'tier_1_restrictive': {
        'apis': ['Nominatim_Geocoding', 'SoilGrids'],
        'max_concurrent': 1,
        'delay_seconds': 1.0,
    },
    # Tier 2: Moderate rate limits (10-50 req/sec)
    'tier_2_moderate': {
        'apis': ['USGS_Water', 'USGS_Geologic_Units', 'OpenTopography', 'Google_Earth_Engine'],
        'max_concurrent': 5,
        'delay_seconds': 0.2,
    },
    # Tier 3: High rate limits (100+ req/sec or no explicit limit)
    'tier_3_permissive': {
        'apis': ['OpenMeteo', 'GBIF_Biodiversity', 'NREL_Solar', 'Copernicus_DEM', 
                 'NOAA_CDO', 'Meteostat', 'GEOS_CF', 'Dynamic_World', 'WoSIS',
                 'EarthData_ModisNASA', 'TerraClimate', 'Copernicus_Atmosphere'],
        'max_concurrent': 20,
        'delay_seconds': 0.05,
    },
}

# APIs that support temporal/date parameters
# These will receive date information when available from sample metadata
APIS_WITH_DATE_SUPPORT = {
    'OpenMeteo',                # Monthly/daily data
    'Copernicus_CDS',           # Historical climate data
    'NOAA_Tides',               # Time-series observations
    'NOAA_CDO',                 # Historical weather
    'Meteostat',                # Daily weather data
    'NASA_EONET',               # Event timing important
    'EMIT_Aerosol',             # Date-specific observations
    'RadNet',                   # Radiation monitoring
    'Copernicus_Atmosphere',    # Time-dependent chemistry
    'TerraClimate',             # Monthly climate (Phase 1)
    'Dynamic_World',            # Daily LULC (Phase 2)
    'Google_Satellite_Embeddings', # Annual composites
    'CMEMS_Marine',             # Daily oceanographic (Phase 3)
}

# US-only datasets - geographic service area validation
US_ONLY_APIS = {
    'NREL_Solar_API',           # NREL solar data US only
    'USGS_Water_Services_API',  # USGS water monitoring US only
    'RadNetAPI',                # EPA RadNet monitoring US only
    'NOAA_Tides_API',           # NOAA tidal stations US coastal
    'SevereWeatherAPI',         # NOAA severe weather US only
}


def is_us_location(lat: float, lon: float) -> bool:
    """
    Check if coordinate is within US service areas.

    Covers: Continental US, Alaska, Hawaii, and Puerto Rico/USVI.
    Used for validation of US-only environmental datasets.

    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)

    Returns:
        True if location is within US service areas, False otherwise.
    """
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return False

    # Continental US (approximate bounds)
    if -125 <= lon <= -66 and 24 <= lat <= 50:
        return True

    # Alaska (approximate bounds)
    if -188 <= lon <= -130 and 50 <= lat <= 72:
        return True

    # Hawaii (approximate bounds)
    if -160 <= lon <= -154 and 18 <= lat <= 23:
        return True

    # Puerto Rico and USVI (approximate bounds)
    if -67.5 <= lon <= -64.5 and 17.5 <= lat <= 18.5:
        return True

    return False