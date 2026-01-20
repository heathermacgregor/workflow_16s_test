from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import pandas as pd
import geopandas as gpd
import biom
import ee
import geopy.distance
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import time
import os
from datetime import datetime, timedelta
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

# Third-party libraries used within API classes
from meteostat import Point, Daily
import pyaqsapi as aqs

# Local utility imports (assuming this structure)
# Ensure you have a utility that provides a rich.progress.Progress instance.
# For example, in a file named `..utils/progress.py`:
# from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
#
# def get_progress_bar():
#     return Progress(
#         SpinnerColumn(),
#         "[progress.description]{task.description}",
#         BarColumn(),
#         "[progress.percentage]{task.percentage:>3.0f}%",
#         TimeElapsedColumn(),
#     )
from ..utils.progress import get_progress_bar

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Suppress pandas SettingWithCopyWarning, as we are aware of the operations
pd.options.mode.chained_assignment = None

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++ API INTEGRATION CLASSES +++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class BaseEnvironmentalAPI:
    """
    A base class for environmental data REST APIs.
    This class provides a common structure for session management with automatic retries.
    """
    def __init__(self, api_key: Optional[str] = None, verbose: bool = False):
        self.api_key = api_key
        self.base_url = ""
        self.api_name = self.__class__.__name__
        self.verbose = verbose
        self.session = self._create_session_with_retries()

    def _create_session_with_retries(self) -> requests.Session:
        """Creates a requests.Session with a retry strategy."""
        session = requests.Session()
        retry = Retry(
            total=5, read=5, connect=5, backoff_factor=0.3,
            status_forcelist=(500, 502, 503, 504),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def get_data(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """
        Abstract method to be implemented by subclasses.
        Polls the API for data at a given latitude and longitude.
        Should return a dictionary of processed data or None on failure.
        """
        raise NotImplementedError


class OpenWeatherMapAPI(BaseEnvironmentalAPI):
    def __init__(self, api_key, verbose=False):
        super().__init__(api_key, verbose=verbose)
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

    def get_data(self, lat, lon):
        params = {"lat": lat, "lon": lon, "appid": self.api_key, "units": "metric"}
        response = self.session.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return {
            "owm_temp_c": data.get('main', {}).get('temp'),
            "owm_feels_like_c": data.get('main', {}).get('feels_like'),
            "owm_temp_min_c": data.get('main', {}).get('temp_min'),
            "owm_temp_max_c": data.get('main', {}).get('temp_max'),
            "owm_pressure_hpa": data.get('main', {}).get('pressure'),
            "owm_humidity_percent": data.get('main', {}).get('humidity'),
            "owm_visibility_m": data.get('visibility'),
            "owm_wind_speed_ms": data.get('wind', {}).get('speed'),
            "owm_wind_deg": data.get('wind', {}).get('deg'),
            "owm_wind_gust_ms": data.get('wind', {}).get('gust'),
            "owm_cloudiness_percent": data.get('clouds', {}).get('all'),
        }

class MeteostatAPI(BaseEnvironmentalAPI):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)

    def get_data(self, lat, lon):
        end = datetime.now()
        start = end - timedelta(days=30)
        location = Point(lat, lon)
        data = Daily(location, start, end).fetch()
        if data.empty:
            return None
        mean_data = data.mean().to_dict()
        return {f"meteostat_{k}_avg": v for k, v in mean_data.items()}

class EPAAQSAPI(BaseEnvironmentalAPI):
    def __init__(self, email, key, verbose=False):
        super().__init__(key, verbose=verbose)
        self.email = email

    def get_data(self, lat, lon):
        bbox_size = 0.1
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        
        ozone_data = aqs.bybox(
            email=self.email, key=self.api_key, param="44201",
            bdate=start_date.strftime("%Y%m%d"), edate=end_date.strftime("%Y%m%d"),
            minlat=lat - bbox_size, maxlat=lat + bbox_size,
            minlon=lon - bbox_size, maxlon=lon + bbox_size
        )
        if ozone_data is not None and not ozone_data.empty:
            return {"epaaqs_ozone_ppb_avg": ozone_data['arithmetic_mean'].mean()}
        return None


class AirNowAPI(BaseEnvironmentalAPI):
    def __init__(self, api_key, verbose=False):
        super().__init__(api_key, verbose=verbose)
        self.base_url = "https://www.airnowapi.org/aq/observation/latLong/current"

    def get_data(self, lat, lon):
        params = {"latitude": lat, "longitude": lon, "distance": 50, "API_KEY": self.api_key, "format": "application/json"}
        response = self.session.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if not data:
            return None
        
        aqi_values = [rec['AQI'] for rec in data if 'AQI' in rec]
        return {"airnow_aqi_avg": np.mean(aqi_values) if aqi_values else None}

class OpenAQAPI(BaseEnvironmentalAPI):
    def __init__(self, api_key=None, verbose=False):
        super().__init__(api_key, verbose=verbose)
        self.base_url = "https://api.openaq.org/v3/latest"

    def get_data(self, lat, lon):
        params = {
            "coordinates": f"{lat},{lon}",
            "radius": 10000,
            "parameter": ["pm25", "o3"]
        }
        headers = {"accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        
        response = self.session.get(self.base_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json().get('results', [])
        if not data:
            return None
        
        pm25_values = []
        o3_values = []
        for location in data:
            for measurement in location.get('measurements', []):
                value = measurement.get('value')
                if value is None or value < 0: continue
                if measurement.get('parameter') == 'pm25':
                    pm25_values.append(value)
                elif measurement.get('parameter') == 'o3':
                    o3_values.append(value)
        
        return {
            "openaq_pm25_ug_m3_avg": np.mean(pm25_values) if pm25_values else None,
            "openaq_o3_ug_m3_avg": np.mean(o3_values) if o3_values else None
        }

class USGS_Earthquake_API(BaseEnvironmentalAPI):
    def __init__(self, api_key=None, verbose=False):
        super().__init__(api_key, verbose=verbose)
        self.base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    def get_data(self, lat, lon):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        params = {
            "format": "geojson", "latitude": lat, "longitude": lon,
            "maxradiuskm": 50, "starttime": start_date.isoformat(),
            "endtime": end_date.isoformat(), "minmagnitude": 2
        }
        response = self.session.get(self.base_url, params=params)
        response.raise_for_status()
        features = response.json().get('features', [])
        if not features:
            return {"usgs_earthquake_count_50km_1yr": 0, "usgs_earthquake_max_mag": None}
        
        magnitudes = [f['properties']['mag'] for f in features if f['properties']['mag'] is not None]
        return {
            "usgs_earthquake_count_50km_1yr": len(features),
            "usgs_earthquake_max_mag": max(magnitudes) if magnitudes else None,
        }

class DOE_AlternativeFuel_API(BaseEnvironmentalAPI):
    def __init__(self, api_key, verbose=False):
        super().__init__(api_key, verbose=verbose)
        self.base_url = "https://developer.nrel.gov/api/alt-fuel-stations/v1/nearest.json"
    
    def get_data(self, lat, lon):
        params = {"api_key": self.api_key, "latitude": lat, "longitude": lon, "radius": 10, "fuel_type": "ELEC,HY,LPG,CNG,BD,E85,LNG", "limit": 200}
        response = self.session.get(self.base_url, params=params)
        response.raise_for_status()
        return {"doe_alt_fuel_stations_10mi_radius": response.json().get('total_results', 0)}

class USGSWaterAPI(BaseEnvironmentalAPI):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        self.base_url = "https://waterservices.usgs.gov/nwis/iv/"

    def get_data(self, lat, lon):
        lat_offset = 0.2
        lon_offset = 0.2 / np.cos(np.radians(lat))
        bbox = f"{lon - lon_offset},{lat - lat_offset},{lon + lon_offset},{lat + lat_offset}"
        params = {"format": "json", "bBox": bbox, "parameterCd": "00060", "siteStatus": "active"}
        response = self.session.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if not data['value']['timeSeries']:
            return None
        
        min_dist, closest_value = float('inf'), None
        for series in data['value']['timeSeries']:
            site_info = series['sourceInfo']
            site_lat = float(site_info['geoLocation']['geogLocation']['latitude'])
            site_lon = float(site_info['geoLocation']['geogLocation']['longitude'])
            dist = geopy.distance.geodesic((lat, lon), (site_lat, site_lon)).km
            
            if dist < min_dist:
                value_records = series.get('values', [{}])[0].get('value', [])
                if value_records:
                    min_dist, closest_value = dist, float(value_records[-1]['value'])

        return {"usgswater_streamflow_cfs": closest_value} if closest_value is not None else None


class SentinelHubAPI(BaseEnvironmentalAPI):
    def __init__(self, client_id, client_secret, verbose=False):
        super().__init__(verbose=verbose)
        self.client_id, self.client_secret = client_id, client_secret
        self.token_url = "https://services.sentinel-hub.com/oauth/token"
        self.process_url = "https://services.sentinel-hub.com/api/v1/statistics"
        self.access_token, self.token_expiry = None, 0
        self._get_access_token()

    def _get_access_token(self):
        if self.verbose: print("    SentinelHub: Authenticating...")
        payload = {'grant_type': 'client_credentials', 'client_id': self.client_id, 'client_secret': self.client_secret}
        try:
            response = self.session.post(self.token_url, data=payload)
            response.raise_for_status()
            token_info = response.json()
            self.access_token = token_info.get('access_token')
            self.token_expiry = time.time() + token_info.get('expires_in', 3600) - 60
        except requests.exceptions.RequestException as e:
            if self.verbose: print(f"    SentinelHub: Failed to get access token: {e}")
            self.access_token = None

    def get_data(self, lat, lon):
        if time.time() > self.token_expiry or self.access_token is None: self._get_access_token()
        if self.access_token is None: return None

        bbox_size_deg, end_date = 0.0045, datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        bbox = [lon - bbox_size_deg, lat - bbox_size_deg, lon + bbox_size_deg, lat + bbox_size_deg]
        evalscript = """//VERSION=3
        function setup(){return{input:["B04","B08","SCL"],output:{bands:1,sampleType:"FLOAT32"}}}
        function evaluatePixel(sample){if(sample.SCL==4||sample.SCL==5){let ndvi=(sample.B08-sample.B04)/(sample.B08+sample.B04);return[ndvi]}return[]}"""
        request_body = {"input": {"bounds": {"bbox": bbox},"data": [{"dataFilter": {"timeRange": {"from": start_date.isoformat()+"Z","to": end_date.isoformat()+"Z"}, "maxCloudCoverage": 20}, "type": "sentinel-2-l2a"}]},"aggregation": {"evalscript": evalscript,"aggregationInterval": {"of": "P30D"}}}
        headers = {'Authorization': f'Bearer {self.access_token}', 'Content-Type': 'application/json', 'Accept': 'application/json'}

        response = self.session.post(self.process_url, json=request_body, headers=headers)
        response.raise_for_status()
        stats = response.json()
        
        try:
            return {"sentinelhub_ndvi_mean": stats['data'][0]['outputs']['B0']['stats']['mean']}
        except (IndexError, KeyError, TypeError):
            return None

class RadNetAPI(BaseEnvironmentalAPI):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        self.all_monitors = self._load_monitor_list()

    def _load_monitor_list(self):
        """Loads monitor list from a local cache or downloads it from a stable CSV link."""
        cache_file = Path("radnet_monitors.pkl")
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < (30 * 86400):
            if self.verbose: print("    RadNet: Loading monitor list from cache.")
            return pd.read_pickle(cache_file)

        if self.verbose: print("    RadNet: Downloading fresh monitor list from new source...")
        url = "https://www.epa.gov/system/files/other-files/2023-01/radnet-monitoring-locations.csv"
        try:
            data = pd.read_csv(url)
            data.rename(columns={'Latitude': 'SITE_LATITUDE', 'Longitude': 'SITE_LONGITUDE', 'Location Name': 'LOCATION_NAME'}, inplace=True)
            for col in ['SITE_LATITUDE', 'SITE_LONGITUDE']: data[col] = pd.to_numeric(data[col], errors='coerce')
            data.dropna(subset=['SITE_LATITUDE', 'SITE_LONGITUDE'], inplace=True)
            data.to_pickle(cache_file)
            return data
        except Exception as e:
            if self.verbose: print(f"    RadNet: Failed to download monitor list: {e}")
            return pd.DataFrame()

    def get_data(self, lat, lon):
        if self.all_monitors.empty: return None
        dists = self.all_monitors.apply(lambda r: geopy.distance.geodesic((lat, lon), (r['SITE_LATITUDE'], r['SITE_LONGITUDE'])).km, axis=1)
        if dists.empty: return None
            
        closest_monitor = self.all_monitors.loc[dists.idxmin()]
        min_dist_km = dists.min()
        
        location_name, state_code = closest_monitor.get('LOCATION_NAME'), closest_monitor.get('State')
        if not location_name or not state_code: return None

        end_date, start_date = datetime.now(), datetime.now() - timedelta(days=90)
        query_url = f"https://iaspub.epa.gov/enviro/efservice/radnet_monitoring_data/STATE_CODE/{state_code}/LOCATION_NAME/{quote(location_name)}/ANALYTE_NAME/GROSS BETA/COLLECTION_DATE/>{start_date.strftime('%m/%d/%Y')}/rows/0:1/JSON"
        
        try:
            res = self.session.get(query_url, timeout=20).json()
            if res:
                measurement = pd.to_numeric(res[0].get('MEASUREMENT_VALUE'), errors='coerce')
                collection_date = pd.to_datetime(res[0].get('COLLECTION_DATE'), errors='coerce')
                if pd.notna(measurement) and pd.notna(collection_date):
                    return {
                        "radnet_gross_beta_pci_m3": measurement,
                        "radnet_dist_to_monitor_km": min_dist_km,
                        "radnet_days_since_measurement": (datetime.now() - collection_date).days
                    }
        except Exception: return {"radnet_dist_to_monitor_km": min_dist_km}
        return {"radnet_dist_to_monitor_km": min_dist_km}

class OpenMeteoAPI(BaseEnvironmentalAPI):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"

    def get_data(self, lat, lon):
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        params = {"latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date, "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,shortwave_radiation_sum,wind_speed_10m_max", "timezone": "auto"}
        response = self.session.get(self.base_url, params=params); response.raise_for_status()
        data = response.json()
        if 'daily' not in data: return None
        
        def safe_stat(values, operation='mean'):
            v = [x for x in values if x is not None]
            if not v: return None
            return np.mean(v) if operation == 'mean' else np.sum(v)
            
        return {
            "openmeteo_temp_c_avg": safe_stat(data['daily'].get('temperature_2m_mean', [])),
            "openmeteo_rh_percent_avg": safe_stat(data['daily'].get('relative_humidity_2m_mean', [])),
            "openmeteo_precip_mm_sum": safe_stat(data['daily'].get('precipitation_sum', []), 'sum'),
            "openmeteo_solar_rad_mj_m2_sum": safe_stat(data['daily'].get('shortwave_radiation_sum', []), 'sum'),
            "openmeteo_wind_kmh_avg": safe_stat(data['daily'].get('wind_speed_10m_max', []))
        }

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++ GOOGLE EARTH ENGINE CLASS +++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class GoogleEarthEngineAPI:
    """
    Handles fetching data from Google Earth Engine (GEE). 🌎🛰️

    This class uses the 'ee' client library and processes all samples in a single
    batch request for high efficiency.
    
    NOTE: Requires GEE authentication. Run 'earthengine authenticate' in your
    terminal and follow the instructions before using this class.
    """
    def __init__(self, ee_project: str, verbose: bool = False):
        self.api_name = self.__class__.__name__
        self.verbose = verbose
        try:
            ee.Initialize(project=ee_project)
            if self.verbose: print(f"✅ {self.api_name}: Successfully initialized.")
        except Exception as e:
            raise RuntimeError(f"Google Earth Engine initialization failed. Please ensure you have authenticated via 'earthengine authenticate' and that the project '{ee_project}' is valid. Original error: {e}")

    def poll_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose: print(f"  Querying {self.api_name}: Preparing batch request for {len(df)} samples...")

        features = [ee.Feature(ee.Geometry.Point(row['longitude_deg'], row['latitude_deg']), {'sample_id': index}) for index, row in df.iterrows()]
        feature_collection = ee.FeatureCollection(features)

        # Define date ranges for temporal averages (relative to Sept 12, 2025)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date_1yr = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        start_date_5yr = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')

        # Define base assets
        dem = ee.Image('USGS/SRTMGL1_003')
        gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
        #glim_fc = ee.FeatureCollection("projects/earthengine-community/assets/GLiM/v1")
        
        # Create a list of all image bands to sample
        images_to_sample = [
            dem.select('elevation').rename('gee_elevation_m'),
            ee.Terrain.slope(dem).rename('gee_slope_deg'),
            ee.Terrain.aspect(dem).rename('gee_aspect_deg'),
            #ee.Image('OREGONSTATE/PRISM/AN81m/normals/ppt').rename('gee_prism_ppt_mm'),
            #ee.Image('OREGONSTATE/PRISM/AN81m/normals/tmean').rename('gee_prism_tmean_c'),
            ee.Image('USGS/NLCD_RELEASES/2021_REL/NLCD/2021').select('landcover').rename('gee_nlcd_landcover_class'),
            ee.Image("CIESIN/GPWv411/GPW_Population_Density/gpw_v4_population_density_rev11_2020_30_sec").select('population_density').rename('gee_population_density'),
            ee.Image("projects/soilgrids-isric/phh2o_mean").select('phh2o_0-5cm_mean').rename('gee_soilgrids_ph'),
            ee.Image("projects/soilgrids-isric/soc_mean").select('soc_0-5cm_mean').rename('gee_soilgrids_soc_g_kg'),
            ee.Image("projects/soilgrids-isric/clay_mean").select('clay_0-5cm_mean').rename('gee_soilgrids_clay_g_kg'),
            ee.ImageCollection('MODIS/061/MOD13A2').filter(ee.Filter.date(start_date_5yr, end_date)).select('NDVI').mean().multiply(0.0001).rename('gee_modis_ndvi_5yr_avg'),
            ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2').filter(ee.Filter.date(start_date_1yr, end_date)).select('tropospheric_NO2_column_number_density').mean().rename('gee_tropomi_no2_mol_m2'),
            ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filter(ee.Filter.date(start_date_1yr, end_date)).select('soil').mean().multiply(0.1).rename('gee_terraclimate_soil_moisture_mm'),
            gsw.select('occurrence').gt(50).Not().fastDistanceTransform(2048, 'euclidean').sqrt().rename('gee_distance_to_water_m'),
            ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG").filter(ee.Filter.date(start_date_1yr, end_date)).select('avg_rad').mean().rename('gee_viirs_night_lights'),
            #ee.Image("CSP/HM/GlobalHumanModification").select('gHM').rename('gee_human_modification'),
            #ee.Image("RESOLVE/ECOREGIONS/2017").select('ECO_ID').rename('gee_ecoregion_id'),
            ee.ImageCollection('MODIS/061/MOD14A1').filter(ee.Filter.date(start_date_5yr, end_date)).select('MaxFRP').mean().rename('gee_modis_fire_frp_5yr_avg'),
            ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filter(ee.Filter.date(start_date_1yr, end_date)).select('vs').mean().multiply(0.1).rename('gee_terraclimate_wind_speed_ms'),
            #ee.ImageCollection("ECMWF/ERA5-LAND/MONTHLY_AGGR").filter(ee.Filter.date(start_date_1yr, end_date)).select('temperature_2m').mean().subtract(273.15).rename('gee_era5_temperature_c_1yr_avg'),
            #ee.ImageCollection("ECMWF/ERA5-LAND/MONTHLY_AGGR").filter(ee.Filter.date(start_date_1yr, end_date)).select('total_precipitation_sum').mean().multiply(1000).rename('gee_era5_precipitation_mm_1yr_avg'),
            ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019").select('discrete_classification').rename('gee_copernicus_landcover'),
            ee.ImageCollection('MODIS/061/MCD19A2_GRANULES').filter(ee.Filter.date(start_date_5yr, end_date)).select('Optical_Depth_055').mean().multiply(0.001).rename('gee_modis_aod_5yr_avg'),
            #ee.Image(0).uint8().paint(glim_fc, 'LITH_CODE').rename('gee_lithology_id'),
            ee.ImageCollection('MODIS/061/MCD15A3H').filter(ee.Filter.date(start_date_5yr, end_date)).select('Lai').mean().multiply(0.1).rename('gee_modis_lai_5yr_avg'),
        ]
        
        final_image = ee.Image.cat(images_to_sample)
        if self.verbose: print(f"  Querying {self.api_name}: Submitting batch request to Earth Engine...")
        reduced_data = final_image.reduceRegions(collection=feature_collection, reducer=ee.Reducer.first(), scale=90).getInfo()

        if self.verbose: print(f"  Querying {self.api_name}: Processing results...")
        results = [f['properties'] for f in reduced_data['features'] if any(k.startswith('gee_') for k in f['properties'])]
        if not results:
            if self.verbose: print(f"  [yellow]~[/yellow] No new data from {self.api_name}.")
            return df

        results_df = pd.DataFrame(results).set_index('sample_id')
        return df.join(results_df)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++ DATA DOWNLOADER CLASS +++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class EnvironmentalDataDownloader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.verbose = config.get('verbose', False)
        self.rest_api_classes = {
            "openaq": OpenAQAPI, "usgs_water": USGSWaterAPI, "openweathermap": OpenWeatherMapAPI,
            "meteostat": MeteostatAPI, "epa_aqs": EPAAQSAPI, "airnow": AirNowAPI,
            "usgs_earthquake": USGS_Earthquake_API, "doe_alt_fuel": DOE_AlternativeFuel_API,
            "sentinelhub": SentinelHubAPI, "radnet": RadNetAPI, "open_meteo": OpenMeteoAPI,
        }
        self.rest_apis = {}
        self.gee_client = None
        self._initialize_apis()

    def _initialize_apis(self):
        creds = self.config.get('credentials', {})
        if creds.get('EE_PROJECT'):
            self.gee_client = GoogleEarthEngineAPI(ee_project=creds['EE_PROJECT'], verbose=self.verbose)

        for name, api_cls in self.rest_api_classes.items():
            try:
                if name == 'openaq': self.rest_apis[name] = api_cls(api_key=creds.get('OPENAQ_API_KEY'), verbose=self.verbose)
                elif name in ['usgs_water', 'meteostat', 'radnet', 'open_meteo', 'usgs_earthquake']: self.rest_apis[name] = api_cls(verbose=self.verbose)
                elif name == 'openweathermap' and creds.get('OPENWEATHERMAP_API_KEY'): self.rest_apis[name] = api_cls(api_key=creds['OPENWEATHERMAP_API_KEY'], verbose=self.verbose)
                elif name == 'epa_aqs' and creds.get('EPA_AQS_EMAIL') and creds.get('EPA_AQS_API_KEY'): self.rest_apis[name] = api_cls(email=creds['EPA_AQS_EMAIL'], key=creds['EPA_AQS_API_KEY'], verbose=self.verbose)
                elif name == 'airnow' and creds.get('AIRNOW_API_KEY'): self.rest_apis[name] = api_cls(api_key=creds['AIRNOW_API_KEY'], verbose=self.verbose)
                elif name == 'doe_alt_fuel' and creds.get('DOE_ALT_FUEL_API_KEY'): self.rest_apis[name] = api_cls(api_key=creds['DOE_ALT_FUEL_API_KEY'], verbose=self.verbose)
                elif name == 'sentinelhub' and creds.get('SENTINELHUB_CLIENT_ID') and creds.get('SENTINELHUB_CLIENT_SECRET'): self.rest_apis[name] = api_cls(client_id=creds['SENTINELHUB_CLIENT_ID'], client_secret=creds['SENTINELHUB_CLIENT_SECRET'], verbose=self.verbose)
            except Exception as e:
                if self.verbose: print(f"Could not initialize {name}: {e}")
        
        if self.verbose: print(f"Initialized {len(self.rest_apis)} REST APIs and {1 if self.gee_client else 0} GEE client.")

    def _process_one_sample(self, sample_id: str, row: pd.Series, progress) -> Dict[str, Any]:
        lat, lon = row['latitude_deg'], row['longitude_deg']
        sample_results = {'#sampleid': sample_id}
        api_task_id = progress.add_task(f"  APIs for [bold]{sample_id}[/]", total=len(self.rest_apis))
        for api_name, api_client in self.rest_apis.items():
            try:
                data = api_client.get_data(lat, lon)
                if data: sample_results.update(data)
            except Exception: pass
            progress.update(api_task_id, advance=1)
        progress.remove_task(api_task_id)
        return sample_results

    def download_all(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.gee_client:
            try:
                df = self.gee_client.poll_dataframe(df)
                if self.verbose: print(f"  [green]✔[/green] Success from {self.gee_client.api_name}.")
            except Exception as e:
                if self.verbose: print(f"  [red]✖[/red] {self.gee_client.api_name} generated an exception: {e}")

        all_results = []
        progress = get_progress_bar()
        with progress:
            samples_task_id = progress.add_task("[cyan]Processing REST APIs for samples...", total=len(df))
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_sample = {executor.submit(self._process_one_sample, sid, r, progress): sid for sid, r in df.iterrows()}
                for future in as_completed(future_to_sample):
                    try:
                        res = future.result()
                        if len(res) > 1: all_results.append(res)
                    except Exception as e:
                        progress.console.print(f"[red]ERROR processing sample {future_to_sample[future]}: {e}")
                    progress.update(samples_task_id, advance=1)

        if all_results:
            results_df = pd.DataFrame(all_results).set_index('#sampleid')
            df = df.join(results_df)
        return df

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++ ANALYSIS & VISUALIZATION 📊 ++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def analyze_and_visualize(df: pd.DataFrame, genus_cols: list, verbose: bool = False):
    if verbose: print("Performing final analysis and visualization...")
    OUTPUT_DIR = Path('./')
    
    # Dynamically find all environmental columns from GEE and other APIs
    gee_cols = [col for col in df.columns if col.startswith('gee_')]
    api_cols = [col for col in df.columns if any(p in col for p in ['owm_', 'meteostat_', 'epaaqs_', 'airnow_', 'openaq_', 'usgs_', 'doe_', 'usgswater_', 'sentinelhub_', 'radnet_', 'openmeteo_'])]
    other_env_cols = ['facility_distance_km']
    all_env_cols = gee_cols + api_cols + other_env_cols

    valid_env_cols = [col for col in all_env_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any()]
    top_genera = df[genus_cols].sum().sort_values(ascending=False).head(20).index.tolist()
    
    analysis_df = df[valid_env_cols + top_genera].dropna()
    final_valid_env_cols = [col for col in valid_env_cols if col in analysis_df.columns]
    final_top_genera = [col for col in top_genera if col in analysis_df.columns]

    if analysis_df.shape[0] < 5 or len(final_valid_env_cols) < 2 or len(final_top_genera) < 2:
        if verbose: print("Not enough overlapping data for correlation analysis after dropping NAs.")
    else:
        if verbose: print("Generating correlation heatmap...")
        correlation_matrix = analysis_df.corr(method='spearman')
        plt.figure(figsize=(max(12, len(final_valid_env_cols)*0.4), max(10, len(final_top_genera)*0.4)))
        sns.heatmap(correlation_matrix.loc[final_top_genera, final_valid_env_cols], annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", annot_kws={"size": 8})
        plt.title('Spearman Correlation: Top Genera vs. Environmental Variables', fontsize=16)
        plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
        plt.savefig(OUTPUT_DIR / 'correlation_heatmap_extended.png', dpi=300, bbox_inches='tight'); plt.close()
        if verbose: print(f"✅ Saved correlation heatmap to {OUTPUT_DIR.resolve()}")

    microbe_df = df[genus_cols].dropna(axis=1, how='all').fillna(0)
    if microbe_df.shape[0] < 3:
        if verbose: print("Not enough samples for PCoA."); return

    if verbose: print("Performing PCoA on microbial community data...")
    bc_dm = beta_diversity('braycurtis', microbe_df.values, microbe_df.index)
    pcoa_results = pcoa(bc_dm)
    pcoa_df = pcoa_results.samples[['PC1', 'PC2']]
    
    cols_to_join = ['gee_human_modification', 'gee_elevation_m', 'sentinelhub_ndvi_mean', 'airnow_aqi_avg']
    existing_cols = [col for col in cols_to_join if col in df.columns]
    if existing_cols: pcoa_df = pcoa_df.join(df[existing_cols])
    pcoa_df.dropna(inplace=True)

    if pcoa_df.empty:
        if verbose: print("Not enough data for PCoA plot after dropping NAs."); return
    
    hue_var = next((col for col in cols_to_join if col in pcoa_df.columns and pcoa_df[col].nunique() > 1), None)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pcoa_df, x='PC1', y='PC2', hue=hue_var, palette='viridis_r', s=100, alpha=0.8)
    plt.xlabel(f"PC1 ({pcoa_results.proportion_explained.get('PC1', 0):.2%})")
    plt.ylabel(f"PC2 ({pcoa_results.proportion_explained.get('PC2', 0):.2%})")
    plt.title('PCoA of Microbial Communities (Bray-Curtis)'); plt.legend(title=hue_var)
    plt.axhline(0, color='grey', lw=0.5, linestyle='--'); plt.axvline(0, color='grey', lw=0.5, linestyle='--')
    plt.savefig(OUTPUT_DIR / 'pcoa_plot.png', dpi=300, bbox_inches='tight'); plt.close()
    if verbose: print(f"✅ Saved PCoA plot to {OUTPUT_DIR.resolve()}")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++ MAIN SCRIPT LOGIC +++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def load_data(table: biom.Table, metadata: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    if verbose: print("Loading and merging data...")
    sample_id_col = next((col for col in metadata.columns if col.lower() == '#sampleid'), None)
    if not sample_id_col: raise ValueError("Could not find '#sampleid' column in metadata.")
    
    metadata.set_index(sample_id_col, inplace=True)
    table_df = pd.DataFrame(table.to_dataframe().T)
    
    metadata.index, table_df.index = metadata.index.astype(str).str.lower(), table_df.index.astype(str).str.lower()
    metadata, table_df = metadata[~metadata.index.duplicated(keep='first')], table_df[~table_df.index.duplicated(keep='first')]
    merged_df = metadata.join(table_df, how='inner')
    if merged_df.empty: raise ValueError("No overlapping samples found between metadata and BIOM table.")
    
    lon_col = next((c for c in merged_df.columns if c.lower() == 'longitude_deg'), None)
    lat_col = next((c for c in merged_df.columns if c.lower() == 'latitude_deg'), None)
    if not lon_col or not lat_col: raise ValueError("Could not find longitude/latitude columns.")
    
    merged_df.rename(columns={lon_col: 'longitude_deg', lat_col: 'latitude_deg'}, inplace=True)
    merged_df.dropna(subset=['longitude_deg', 'latitude_deg'], inplace=True)
    if merged_df.empty: raise ValueError("No samples with valid coordinates remaining.")
    
    if verbose: print(f"✅ Successfully loaded and merged data for {len(merged_df)} samples.")
    return merged_df

def env(config: Dict, table: biom.Table, metadata: pd.DataFrame):
    enabled = config.get("environmental_data", {}).get("enabled", False) 
    if not enabled:
        return
    verbose = config.get('verbose', False)
    if verbose: print("--- Running Environmental Analysis Script ---")
    
    OUTPUT_FILE = config.get('output_file', 'enriched_environmental_data.csv')
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    df = load_data(table, metadata, verbose=verbose)
    downloader = EnvironmentalDataDownloader(config)
    df = downloader.download_all(df)
    
    genus_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('k__')]
    
    if verbose: print(f"Saving final data table to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE)
    
    analyze_and_visualize(df, genus_cols, verbose=verbose)
    
    print("\n--- Script Finished ---")
    print(f"📊 Final data table saved to: {OUTPUT_FILE}")
    print(f"📈 Visualizations saved in the current directory.")
    return df

if __name__ == '__main__':
    print("--- Running in Standalone Demo Mode ---")
    config = {
        'verbose': True, 'output_file': 'demo_enriched_data.csv',
        'credentials': {
            'EE_PROJECT': os.environ.get('EE_PROJECT', 'your-gcp-project-id'),
            'OPENWEATHERMAP_API_KEY': os.environ.get('OPENWEATHERMAP_API_KEY'),
            'AIRNOW_API_KEY': os.environ.get('AIRNOW_API_KEY', "3B2E33D0-0102-4A2A-ABE8-08211D3E1D25"),
            'SENTINELHUB_CLIENT_ID': os.environ.get('SENTINELHUB_CLIENT_ID'),
            'SENTINELHUB_CLIENT_SECRET': os.environ.get('SENTINELHUB_CLIENT_SECRET'),
            'EPA_AQS_EMAIL': os.environ.get('EPA_AQS_EMAIL', 'test@test.com'),
            'EPA_AQS_API_KEY': os.environ.get('EPA_AQS_API_KEY'),
            'DOE_ALT_FUEL_API_KEY': os.environ.get('DOE_ALT_FUEL_API_KEY', "DEMO_KEY"),
            'OPENAQ_API_KEY': os.environ.get('OPENAQ_API_KEY'),
        }
    }
    mock_metadata = pd.DataFrame({
        '#sampleid': ['sample1', 'sample2', 'sample3', 'sample4'],
        'latitude_deg': [34.05, 40.71, 37.77, 41.88],
        'longitude_deg': [-118.24, -74.00, -122.42, -87.63],
        'facility_distance_km': [10.5, 50.2, 5.1, 150.8]
    })
    mock_otu_data = np.array([[100,20,0,500], [50,800,10,0], [0,150,400,50], [300,0,20,250]])
    sample_ids = ['sample1', 'sample2', 'sample3', 'sample4']
    otu_ids = ['k__B;p__P;g__E','k__B;p__F;g__L','k__B;p__A;g__B','k__B;p__B;g__B']
    mock_table = biom.Table(mock_otu_data, otu_ids, sample_ids)
    
    env(config, mock_table, mock_metadata)