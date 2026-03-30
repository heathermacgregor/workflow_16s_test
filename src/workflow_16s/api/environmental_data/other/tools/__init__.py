# workflow_16s/api/environmental_data/other/tools/__init__.py

from .base import BaseEnvironmentalAPI, REQUEST_TIMEOUT
from .cache import cache_api_call
from ._inaturalist import iNaturalistAPI
from ._meteostat import MeteostatAPI
from ._noaa import NOAA_Tides_API
from ._nrel import NREL_Solar_API
from ._nws import NWS_API, SevereWeatherAPI
from ._openmeteo import EnvironmentalHealthAPI, SoilStateAPI, OpenMeteoAPI
from ._radnet import RadNetAPI
from ._soilgrids import SoilGridsAPI
from ._usgs import USGS_Earthquake_API, USGS_Water_Services_API
from ._csu_soil import CSUSoilAPI
from ._usgs_cmibs import USGS_CMIBS_API
from ._cmmi_cmio import CMMI_CMiO_API
from ._emit_aerosol import EMIT_AerosolAPI
from ._gesamp_fe import GESAMP_FeAPI
from ._icmm_mines import ICMM_MinesAPI
from ._noaa_cdo import NOAA_CDO_API
from ._nasa_eonet import NASA_EONET_API
from ._opentopography import OpenTopography_API
from ._usgs_water import USGS_Water_API
from ._nasa_earth_imagery import NASA_Earth_Imagery_API
from ._copernicus_cds import Copernicus_CDS_API
from ._nasa_earth_observatory import NASA_Earth_Observatory_API
from ._overpass import OverpassAPI
from ._nominatim import NominatimAPI
from ._gbif import GBIFAPI
from ._usgs_mrds import USGSMRDSMinesAPI
from ._usgs_nure import USGSNUREGeochemistryAPI
from ._usgs_ngs import USGSNationalGeochemicalAPI
from ._usgs_geology import USGSGeologicUnitsAPI
from ._mgnify import MGnifyAPI
from ._soilgrids_python import SoilGridsPythonAPI
from ._google_earth_engine import GoogleEarthEngineAPI
from ._gee_mega_image import (
    get_enabled_datasets,
    get_band_info_from_config,
    create_mega_image,
    MegaImageBuilder,
    GEE_DATASETS,
)
from ._wosis import WoSISAPI
from ._terraclimate import TerraClimateAPI
from ._hydrosheds import HydroSHEDSAPI
from ._geos_cf import GEOSCFAPI
from ._global_power_plants import GlobalPowerPlantsAPI
from ._dynamic_world import DynamicWorldAPI
from ._alos_chili import ALOSCHILIandmTPIAPI
from ._cmems import CMEMSMarineDataAPI
from ._google_satellite_embeddings import GoogleSatelliteEmbeddingsAPI

__all__ = [
    "BaseEnvironmentalAPI", "REQUEST_TIMEOUT", "cache_api_call",
    "iNaturalistAPI", "MeteostatAPI", "NOAA_Tides_API", 
    "NREL_Solar_API", "NWS_API", "SevereWeatherAPI",
    "EnvironmentalHealthAPI", "SoilStateAPI", "OpenMeteoAPI",
    "RadNetAPI", "SoilGridsAPI", "USGS_Earthquake_API", 
    "USGS_Water_Services_API",
    "CSUSoilAPI", "USGS_CMIBS_API", "CMMI_CMiO_API",
    "EMIT_AerosolAPI", "GESAMP_FeAPI", "ICMM_MinesAPI",
    "NOAA_CDO_API", "NASA_EONET_API", "OpenTopography_API",
    "USGS_Water_API", "NASA_Earth_Imagery_API", "Copernicus_CDS_API",
    "NASA_Earth_Observatory_API",
    # OSM & Biodiversity APIs
    "OverpassAPI", "NominatimAPI", "GBIFAPI",
    # USGS Geochemistry & Mineral Data
    "USGSMRDSMinesAPI", "USGSNUREGeochemistryAPI", "USGSNationalGeochemicalAPI",
    "USGSGeologicUnitsAPI",
    # Metagenomics & Soil Data
    "MGnifyAPI", "SoilGridsPythonAPI",
    # Multi-dataset Earth Engine
    "GoogleEarthEngineAPI",
    # GEE Mega-Image Creation & Tier Filtering
    "get_enabled_datasets", "get_band_info_from_config", 
    "create_mega_image", "MegaImageBuilder", "GEE_DATASETS",
    # New Environmental Data Sources (Phase 1-3)
    "WoSISAPI", "TerraClimateAPI", "HydroSHEDSAPI",
    "GEOSCFAPI", "GlobalPowerPlantsAPI", "DynamicWorldAPI",
    # Advanced Topographic & Marine Data
    "ALOSCHILIandmTPIAPI", "CMEMSMarineDataAPI",
    # Google Satellite Embeddings (AlphaEarth)
    "GoogleSatelliteEmbeddingsAPI",
]