# workflow_16s/api/environmental_data/arkin/constants.py

# Define Earth Engine assets to be queried
EE_ASSETS = [
    ("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL", "Alpha_Earth_Embeddings"),
    ("MODIS/061/MOD13Q1", "MODIS_Vegetation_Indices"),
    ("MODIS/061/MOD11A1", "MODIS_Land_Surface_Temperature"),
    ("MODIS/061/MCD15A3H", "MODIS_Leaf_Area_Index"),
    ("LANDSAT/LC08/C02/T1_L2", "Landsat_8_Surface_Reflectance"),
    ("ECMWF/ERA5_LAND/HOURLY", "ERA5_Land_Hourly"),
    ("NASA/GLDAS/V021/NOAH/G025/T3H", "GLDAS_Noah_Land_Surface_Model"),
    ("USGS/SRTMGL1_003", "SRTM_Digital_Elevation"),
    ("Oxford/MAP/accessibility_to_cities_2015_v1_0", "Accessibility_to_Cities")
]

# Define default parameters for each service
SERVICE_CONFIG = {
    "EARTH_ENGINE": {"timeout": 600},
    "EPA_AQS": {"timeout": 300, "max_records": 15000},
    "GBIF": {"timeout": 300, "max_records": 10000},
    "NASA_POWER": {"timeout": 300},
    "OpenAQ": {"timeout": 300, "max_records": 20000},
    "OSM_Overpass": {"timeout": 300, "max_records": 20000},
    "SoilGrids": {"timeout": 600, "max_pixels": 10000, "statistics": ["mean"], "include_wrb": True},
    "USGS_NWIS": {"timeout": 300}, "SSURGO": {"timeout": 300}, "WQP": {"timeout": 300}
}
MAX_CONCURRENT_SAMPLES = 8