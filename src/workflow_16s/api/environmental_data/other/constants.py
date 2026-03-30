# workflow_16s/api/environmental_data/other/constants.py

MAX_WORKERS = 20  # Increased from 10 for better concurrent API throughput

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

APIS_WITH_DATE_SUPPORT = [
    "Meteostat", "NOAA_Tides", "USGS_Earthquake", "NREL_Solar", 
    "EnvironmentalHealth", "SoilState", "OpenMeteo",
    "EMIT_Aerosol", "GESAMP_Fe",  # Support temporal water/atmospheric data
    # New Phase 1-3 APIs with date support
    "GEOS_CF",  # Daily air quality analysis (must use sample collection date, not today)
    "TerraClimate",  # Monthly climate data
    "Dynamic_World",  # Daily LULC updates
    "CMEMS_Marine",  # Daily ocean analysis/forecast
    "Google_Satellite_Embeddings",  # Annual embeddings (year parameter)
]