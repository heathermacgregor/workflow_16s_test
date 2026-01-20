# ==================================================================================== #

# Standard Imports
import io
import os.path
from functools import wraps
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import hashlib
import json
import logging
import re
import requests
from requests.exceptions import RequestException # Import RequestException for robust retries
import time
import warnings
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import ast

# Third-Party Imports
import ee
import pandas as pd
from ee.batch import Export
from ee.data import getAsset
from ee.ee_exception import EEException
from ee.feature import Feature
from ee.featurecollection import FeatureCollection
from ee.geometry import Geometry
from ee.image import Image
from ee.imagecollection import ImageCollection
from ee.reducer import Reducer
from ee.filter import Filter
from ee.join import Join

# NOTE: Assuming these are in a local package or file.
from workflow_16s.constants import GEE_CATALOG_FILE_PATH, GEE_CATALOG_FILE_ZENODO_URL
# from workflow_16s.utils.progress import get_progress_bar # Removed for stability

# ==================================================================================== #

# Promote DeprecationWarning to an Exception
warnings.filterwarnings('error', category=DeprecationWarning)
logger = logging.getLogger("workflow_16s")

# Define list of assets to skip at the top level
CORRUPTED_ASSETS = [
    "Oxford/MAP/EVI_5km_Monthly",
    "Oxford/MAP/IGBP_Fractional_Landcover_5km_Annual",
    "Oxford/MAP/LST_Day_5km_Monthly",
    "Oxford/MAP/LST_Night_5km_Monthly",
    "Oxford/MAP/TCB_5km_Monthly",
    "Oxford/MAP/TCW_5km_Monthly",
    "Oxford/MAP/accessibility_to_cities_2015_v1_0",
    "Oxford/MAP/accessibility_to_healthcare_2019",
    "Oxford/MAP/friction_surface_2015_v1_0",
    "Oxford/MAP/friction_surface_2019",
    "CSP/ERGo/1_0/US/CHILI"
]

# ==================================================================================== #

def gee_retry(max_retries=3, delay=5, backoff=2):
    """A decorator to retry GEE and network operations with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None  # Variable to store the last exception
            retries = 0
            current_delay = delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                # Catch GEE, Google API, and general network errors
                except (EEException, HttpError, RequestException) as e:
                    last_exc = e  # Store the exception
                    # Don't retry on client-side errors like bad band names
                    if 'did not match any bands' in str(e):
                        raise
                    retries += 1
                    logger.warning(f"Attempt {retries}/{max_retries} failed for {func.__name__}: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            logger.error(f"Function {func.__name__} failed after {max_retries} attempts.")
            if last_exc:
                raise last_exc  # Re-raise the stored exception
            # This fallback should not be reached in normal operation
            raise RuntimeError(f"Function {func.__name__} failed after retries, but no exception was captured.")
        return wrapper
    return decorator

# ==================================================================================== #

class CacheManager:
    """Handles file-based caching for GEE asset info and final results."""
    def __init__(self, cache_dir: Union[str, Path] = '.gee_cache', verbose: bool = True):
        self.cache_path = Path(cache_dir)
        self.asset_cache_path = self.cache_path / 'asset_types'
        self.results_cache_path = self.cache_path / 'run_results'
        self.verbose = verbose
        
        self.cache_path.mkdir(exist_ok=True)
        self.asset_cache_path.mkdir(exist_ok=True)
        self.results_cache_path.mkdir(exist_ok=True)
        
        if self.verbose:
            logger.info(f"Cache enabled. Location: {self.cache_path.resolve()}")

    def _sanitize_filename(self, name: str) -> str:
        return name.replace('/', '_').replace(':', '_')

    def get_asset_type(self, asset_id: str) -> Optional[str]:
        safe_filename = self._sanitize_filename(asset_id)
        cache_file = self.asset_cache_path / f"{safe_filename}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                logger.debug(f"🟩 Cache HIT for asset type: {asset_id}")
                return data.get('type')
        logger.debug(f"🟥 Cache MISS for asset type: {asset_id}")
        return None

    def set_asset_type(self, asset_id: str, asset_type: Optional[str]):
        safe_filename = self._sanitize_filename(asset_id)
        cache_file = self.asset_cache_path / f"{safe_filename}.json"
        with open(cache_file, 'w') as f:
            json.dump({'id': asset_id, 'type': asset_type}, f, indent=4)

    def get_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        cache_file = self.results_cache_path / f"{key}.parquet"
        if cache_file.exists():
            try:
                if self.verbose:
                    logger.info(f"🟩 Cache HIT for run result: {key}")
                return pd.read_parquet(cache_file)
            except Exception as e:
                logger.warning(f"🟨 Could not read cache file {cache_file}: {e}")
                return None
        if self.verbose:
            logger.info(f"🟥 Cache MISS for run result: {key}")
        return None

    def set_dataframe(self, key: str, df: pd.DataFrame):
        cache_file = self.results_cache_path / f"{key}.parquet"
        df.to_parquet(cache_file, index=False)
        if self.verbose:
            logger.info(f"🟩 Saved run result to cache: {key}")

# ==================================================================================== #

class GoogleDriveManager:
    """Handles Google Drive authentication, file search, and downloads."""
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    TOKEN_PATH = Path('.gee_cache/gdrive_token.json')
    CREDENTIALS_PATH = Path('credentials.json')

    def __init__(self):
        self.creds = self._get_credentials()
        self.service = build('drive', 'v3', credentials=self.creds)

    def _get_credentials(self):
        creds = None
        if self.TOKEN_PATH.exists():
            creds = Credentials.from_authorized_user_file(str(self.TOKEN_PATH), self.SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self.CREDENTIALS_PATH.exists():
                    raise FileNotFoundError(
                        "Google Drive 'credentials.json' not found. "
                        "Please download it from the Google Cloud Console."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.CREDENTIALS_PATH), self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            self.TOKEN_PATH.parent.mkdir(exist_ok=True)
            with open(self.TOKEN_PATH, 'w') as token:
                token.write(creds.to_json())
        return creds

    @gee_retry()
    def download_file(self, filename: str, folder: str = 'gee_exports') -> pd.DataFrame:
        try:
            folder_query = f"mimeType='application/vnd.google-apps.folder' and name='{folder}' and trashed=false"
            folder_results = self.service.files().list(q=folder_query, fields="files(id)").execute()
            folder_items = folder_results.get('files', [])
            if not folder_items:
                raise FileNotFoundError(f"Google Drive folder '{folder}' not found.")
            folder_id = folder_items[0]['id']

            file_query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
            file_results = self.service.files().list(q=file_query, fields="files(id)").execute()
            file_items = file_results.get('files', [])
            if not file_items:
                raise FileNotFoundError(f"File '{filename}' not found in Drive folder '{folder}'.")
            file_id = file_items[0]['id']

            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            fh.seek(0)
            return pd.read_csv(fh)

        except HttpError as error:
            logger.error(f"An error occurred with Google Drive API: {error}")
            raise
        except FileNotFoundError as error:
            logger.error(error)
            raise

# ==================================================================================== #

def is_bbox_global(bbox: List, tolerance: int = 10) -> bool:
    if not bbox or len(bbox) != 4:
        return False
    lon_min, lat_min, lon_max, lat_max = bbox
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    return (360 - tolerance) <= lon_range <= (360 + tolerance) and \
           (180 - tolerance) <= lat_range <= (180 + tolerance)

def find_global_datasets(catalog_file: Union[str, Path]) -> List[Dict[str, str]]:
    try:
        with open(catalog_file, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"🟥 Error reading catalog file '{catalog_file}': {e}")
        return []

    global_datasets = []
    global_keywords = ['global', 'world', 'worldwide']

    for asset in data:
        if asset.get("gee:status") == "deprecated":
            continue

        is_global = False
        title = asset.get('title', 'No Title')
        description = asset.get('description', '')
        asset_id = asset.get('id', 'No ID')

        if any(keyword in title.lower() for keyword in global_keywords) or \
           any(keyword in description.lower() for keyword in global_keywords):
            is_global = True

        try:
            bbox = asset['extent']['spatial']['bbox'][0]
            if is_bbox_global(bbox):
                is_global = True
        except (KeyError, IndexError, TypeError):
            pass

        if is_global:
            dataset_info = {'title': title, 'id': asset_id}
            if dataset_info not in global_datasets:
                global_datasets.append(dataset_info)

    return global_datasets

# ==================================================================================== #

def get_catalog_file(file_path: Union[str, Path], zenodo_url: str) -> None:
    file_path = Path(file_path)
    if file_path.is_file():
        return
    logger.info(f"Downloading GEE catalog file from {zenodo_url}...")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(zenodo_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info("Catalog file downloaded successfully.")
    except requests.exceptions.RequestException as e:
        logger.error(f"🟥 Error downloading catalog file: {e}")
        if file_path.exists(): 
            file_path.unlink()

# ==================================================================================== #

class GEEDataCollector:
    url_high_volume = 'https://earthengine-highvolume.googleapis.com'
    
    def __init__(
        self, 
        config: Dict,
        datasets: Optional[List] = None,
        use_cache: bool = True,
        max_workers: int = 5
    ):
        self.config = config
        self.verbose = self.config.get('verbose', True)
        self.max_workers = max_workers
        
        p = self.config.get('google_earth_engine', {}).get('catalog_file')
        self.catalog_file_path = Path(p) if p else GEE_CATALOG_FILE_PATH
        get_catalog_file(self.catalog_file_path, zenodo_url=GEE_CATALOG_FILE_ZENODO_URL)
        
        self.use_cache = use_cache
        if self.use_cache:
            self.cache = CacheManager()
            
        ee_project = self.config.get('credentials', {}).get('GEE_PROJECT')
        if not ee_project:
            raise ValueError("GEE_PROJECT must be specified in the configuration.")
        try:
            ee.Initialize(project=ee_project, opt_url=self.url_high_volume)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize GEE. Have you run 'earthengine authenticate'? Original error: {e}"
            )
        
        cfg_datasets = self.config.get('google_earth_engine', {}).get('datasets')
        all_datasets = datasets if datasets is not None else cfg_datasets
        if all_datasets is None:
            all_datasets = find_global_datasets(self.catalog_file_path)
        
        self.datasets = [d for d in all_datasets if d['id'] not in CORRUPTED_ASSETS]
        
        if not self.datasets:
            raise ValueError("No valid datasets found to process.")
        
        self.assets = {'raster': [], 'vector': []}
        for dataset in self.datasets:
            if 'deprecated' in dataset.get('title', '').lower():
                continue
            # --- FIX: Gracefully handle assets that don't exist or are inaccessible ---
            try:
                asset_type = self._get_asset_type(dataset['id'])
                if asset_type in ['IMAGE', 'IMAGE_COLLECTION']:
                    self.assets['raster'].append(dataset)
                elif asset_type in ['FEATURE', 'FEATURE_COLLECTION']:
                    self.assets['vector'].append(dataset)
            except Exception as e:
                logger.warning(f"⚠️ Could not determine asset type for '{dataset['id']}'. Skipping. Reason: {e}")
                continue
        
        self.gdrive = GoogleDriveManager()
    
    @gee_retry()
    def _get_asset_type(self, asset_id: str) -> Optional[str]:
        if self.use_cache:
            cached_type = self.cache.get_asset_type(asset_id)
            if cached_type is not None:
                return cached_type
        asset_type = None
        try:
            info = getAsset(asset_id)
            if info:
                asset_type = info['type']
        except EEException as e:
            # Re-raise the exception to be handled by the decorator or the calling function
            raise 
        except Exception as e:
            logger.error(f"🟥 Skipping asset '{asset_id}' due to unexpected error: {e}")
            # Re-raise for consistency
            raise
        
        if self.use_cache:
            self.cache.set_asset_type(asset_id, asset_type)
        return asset_type
    
    def _export_and_fetch_results(self, collection: FeatureCollection, description: str) -> pd.DataFrame:
        export_folder = 'gee_exports'
        filename = f"{description}.csv"
        
        if self.use_cache:
            cached_df = self.cache.get_dataframe(description)
            if cached_df is not None:
                return cached_df

        task = Export.table.toDrive(
            collection=collection,
            description=description.replace("_", "")[:100],
            folder=export_folder,
            fileNamePrefix=description,
            fileFormat='CSV'
        )
        task.start()

        logger.info(f"GEE export task started (ID: {task.id}). Waiting for completion...")
        start_time = time.time()
        while task.active():
            if time.time() - start_time > 1800:
                task.cancel()
                raise RuntimeError(f"GEE task {task.id} timed out after 30 minutes.")
            status = task.status()['state']
            if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
                break
            time.sleep(15)
        
        final_status = task.status()
        if final_status['state'] != 'COMPLETED':
            error_msg = final_status.get('error_message', 'No error message from GEE.')
            raise RuntimeError(f"🟥 GEE export task failed for {description}. Reason: {error_msg}")

        logger.info(f"🟩 GEE task '{task.id}' completed. Downloading from Google Drive...")
        df = self.gdrive.download_file(filename=filename, folder=export_folder)

        if self.use_cache:
            self.cache.set_dataframe(description, df)
        
        return df
        
    def run(self, df: pd.DataFrame, lat_col: str, lon_col: str, scale: int = 1000) -> pd.DataFrame:
        if lat_col not in df.columns or lon_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{lat_col}' and '{lon_col}' columns.")
        
        input_df = df.copy()
        input_df['unique_id'] = range(len(input_df))
        
        points_fc = self._df_to_gee(input_df, lat_col, lon_col)
        
        # Create a directory for individual TSV exports
        export_dir = Path('gee_tsv_exports')
        export_dir.mkdir(exist_ok=True)
        logger.info(f"Individual asset TSV files will be saved to: {export_dir.resolve()}")

        # This will be the final merged dataframe, to maintain original behavior for __main__
        final_df = input_df.copy()
        
        data_frames = []
        
        # --- SEQUENTIAL PROCESSING TO PREVENT SEGMENTATION FAULT ---
        total_assets = len(self.assets['raster']) + len(self.assets['vector'])
        processed_count = 0
        logger.info(f"Starting sequential processing of {total_assets} assets.")

        # Process rasters sequentially
        for asset in self.assets['raster']:
            logger.info(f"Processing raster asset ({processed_count + 1}/{total_assets}): {asset['id']}")
            try:
                result_df = self._process_raster_asset(points_fc, asset, scale)
                if result_df is not None and not result_df.empty:
                    data_frames.append(result_df)
            except Exception as exc:
                logger.error(f"❌ Asset {asset['id']} failed processing: {exc}")
                result_df = None
            processed_count += 1
            
            if result_df is not None and not result_df.empty:
                result_df['unique_id'] = range(len(result_df))
                logger.info(result_df)
                # --- 1. Export this single asset's data to a TSV file ---
                export_data = pd.merge(input_df[['unique_id', lat_col, lon_col]], result_df, on='unique_id', how='left')
                export_data['coordinates'] = export_data.apply(lambda row: f"({row[lat_col]}, {row[lon_col]})", axis=1)
                    
                data_collection_cols = [col for col in result_df.columns if col not in ['unique_id', '.geo']]
                final_cols = ['unique_id', 'coordinates'] + data_collection_cols
                    
                export_data_final = export_data[[col for col in final_cols if col in export_data.columns]]
                    
                safe_filename = re.sub(r'[^a-zA-Z0-9_]', '_', asset['id'])
                export_path = export_dir / f"{safe_filename}.tsv"
                export_data_final.to_csv(export_path, sep='\t', index=False)
                logger.info(f"✅ Successfully exported data for '{asset['id']}' to {export_path}")

                # --- 2. Merge results into the final comprehensive dataframe ---
                result_df_clean = result_df.drop(columns=[c for c in ['.geo'] if c in result_df.columns], errors='ignore')
                final_df = pd.merge(final_df, result_df_clean, on='unique_id', how='left')
            else:
                logger.warning(f"⚠️ No data returned for asset '{asset['id']}'. Skipping.")
            
        # Process vectors sequentially
        for asset in self.assets['vector']:
            logger.info(f"Processing vector asset ({processed_count + 1}/{total_assets}): {asset['id']}")
            try:
                result_df = self._process_vector_asset(points_fc, asset)
                if result_df is not None and not result_df.empty:
                    data_frames.append(result_df)
            except Exception as exc:
                logger.error(f"❌ Asset {asset['id']} failed processing: {exc}")
            processed_count += 1
        
        if not data_frames:
            logger.warning("No data was retrieved from GEE.")
            return input_df.drop(columns=['unique_id'])
            
        final_df = input_df
        for result_df in data_frames:
            result_df = result_df.drop(columns=[c for c in ['.geo', 'geo_lon', 'geo_lat'] if c in result_df.columns], errors='ignore')
            final_df = pd.merge(final_df, result_df, on='unique_id', how='left')

        return final_df.drop(columns=['unique_id'])
    
    def _sanitize_asset_id(self, asset_id: str) -> str:
        """Helper to create a safe, unique prefix from an asset ID."""
        return re.sub(r'[^a-zA-Z0-9_]', '_', asset_id)
       
    def _process_raster_asset(self, points_fc: FeatureCollection, asset: Dict, scale: int) -> Optional[pd.DataFrame]:
        asset_id = asset['id']
        image_to_sample = self._get_image_from_asset(asset_id)
        if image_to_sample is None: return None

        reducer = Reducer.first()
        sampled_fc = image_to_sample.reduceRegions(collection=points_fc, reducer=reducer, scale=scale)
        
        cache_key = self._generate_cache_key(asset_id, scale)
        df = self._export_and_fetch_results(sampled_fc, cache_key)
        
        # --- FIX: Rename the generic 'first' column to prevent merge conflicts ---
        safe_prefix = self._sanitize_asset_id(asset_id)
        band_names = image_to_sample.bandNames().getInfo()
        if 'first' in df.columns and band_names is not None and len(band_names) == 1:
            df = df.rename(columns={'first': safe_prefix})

        # --- FIX: Handle multi-band images where one band is named 'first' ---
        elif f"{safe_prefix}_first" in df.columns:
             df = df.rename(columns={f"{safe_prefix}_first": f"{safe_prefix}"})
        
        cols_to_drop = [c for c in ['system:index', '.geo'] if c in df.columns]
        return df.drop(columns=cols_to_drop)

    def _process_vector_asset(self, points_fc: FeatureCollection, asset: Dict) -> Optional[pd.DataFrame]:
        asset_id = asset['id']
        vector_fc = FeatureCollection(asset_id)
        safe_prefix = re.sub(r'[^a-zA-Z0-9_]', '_', asset_id)
        spatial_filter = Filter.intersects(leftField='.geo', rightField='.geo', maxError=10)
        joined_fc = Join.saveFirst(matchKey=safe_prefix).apply(points_fc, vector_fc, spatial_filter)

        cache_key = self._generate_cache_key(asset_id)
        df = self._export_and_fetch_results(joined_fc, cache_key)
        
        return self._post_process_vector_df(df, asset_id)

    def _post_process_vector_df(self, df: pd.DataFrame, asset_id: str) -> pd.DataFrame:
        col_name = re.sub(r'[^a-zA-Z0-9_]', '_', asset_id)
        if col_name not in df.columns:
            return df

        def extract_props(row_str):
            if pd.isna(row_str): return {}
            try:
                feature_dict = ast.literal_eval(row_str) if isinstance(row_str, str) else row_str
                return feature_dict.get('properties', {})
            except (ValueError, SyntaxError): return {}
        
        props_df = pd.json_normalize(df[col_name].apply(extract_props).tolist())
        props_df.columns = [f"{col_name}_{p}" for p in props_df.columns]
        
        processed_df = pd.concat([df[['unique_id']], props_df], axis=1)
        return processed_df

    def _generate_cache_key(self, asset_id: str, scale: Optional[int] = None) -> str:
        key = f"asset_{asset_id.replace('/', '_')}"
        if scale:
            key += f"_s{scale}"
        return key

    def _df_to_gee(self, df: pd.DataFrame, lat_col: str, lon_col: str) -> FeatureCollection:
        features = []
        for _, row in df.iterrows():
            try:
                lat, lon = float(row[lat_col]), float(row[lon_col])
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    continue
                geom = Geometry.Point([lon, lat])
                props = {'unique_id': row['unique_id']} 
                feature = Feature(geom, props)
                features.append(feature)
            except (ValueError, TypeError):
                continue
        return FeatureCollection(features)
        
    @gee_retry()
    def _get_image_from_asset(self, asset_id: str) -> Optional[Image]:
        """Gets a representative ee.Image from an asset_id, handling ImageCollections."""
        try:
            asset_info = getAsset(asset_id)
            asset_type = asset_info.get('type', 'Unknown').lower()
            safe_prefix = re.sub(r'[^a-zA-Z0-9_]', '_', asset_id)

            image = None
            if asset_type == 'image':
                image = Image(asset_id)
            elif asset_type == 'image_collection':
                collection = ImageCollection(asset_id)
                collection_size = collection.size().getInfo()
                
                if collection_size is not None and collection_size > 0:
                    first_image = Image(collection.first())
                    bands = first_image.bandNames()
                    image = collection.select(bands).mosaic()
            
            if image:
                current_bands = image.bandNames().getInfo()
                if not current_bands: return None
                renamed_bands = [f"{safe_prefix}_{b}" for b in current_bands]
                return image.rename(renamed_bands)

        except EEException as e:
            logger.error(f"❌ GEE error preparing image for {asset_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error preparing image for {asset_id}: {e}")
        
        return None

# ==================================================================================== #

if __name__ == "__main__":
    # --- Example Usage ---
    # Create a dummy config for demonstration
    # You MUST replace 'your-gee-project' with your actual GEE project ID
    # You MUST have a 'credentials.json' file for Google Drive access
    from workflow_16s.config import get_config # type: ignore

    config = get_config()
    """
    NFC_FACILITIES_PATH = '/usr2/people/macgregor/amplicon/test/data/nfc/facilities.tsv'
    def clean_df(df):
        df = df.dropna(subset=['latitude', 'longitude'])
        df = df[df['latitude'].between(-90, 90)]
        df = df[df['longitude'].between(-180, 180)]
        return df
    nfc_df = pd.read_csv(NFC_FACILITIES_PATH, sep='\t', low_memory=False)
    nfc_df = clean_df(nfc_df)
    """
    import pandas as pd
    SAMPLES_PATH = '/usr2/people/macgregor/amplicon/test/data/merged/metadata/final_metadata.tsv'
    # 1. Load the data
    samples_df = pd.read_csv(SAMPLES_PATH, sep='\t', low_memory=False)
    print(f"Initial shape: {samples_df.shape}")

    # 2. First, handle NaN-to-None conversion for all OTHER property columns.
    samples_df = samples_df.astype(object).where(pd.notna(samples_df), None)

    # 3. NOW, perform the strict cleaning on the coordinate columns as the FINAL step.
    samples_df['latitude_deg'] = pd.to_numeric(samples_df['latitude_deg'], errors='coerce')
    samples_df['longitude_deg'] = pd.to_numeric(samples_df['longitude_deg'], errors='coerce')
    samples_df.dropna(subset=['latitude_deg', 'longitude_deg'], inplace=True)
    samples_df = samples_df[samples_df['latitude_deg'].between(-90, 90)]
    samples_df = samples_df[samples_df['longitude_deg'].between(-180, 180)]
    # 1. Reset the index. This creates a new, sequential index.
    print("Resetting the index...")
    samples_df.reset_index(inplace=True)
    samples_df.drop(columns=['latitude_deg', 'longitude_deg'], inplace=True)
    samples_df.rename(
        columns={'latitude_deg': 'latitude', 'longitude_deg': 'longitude'},
        inplace=True
    )
    cols_to_keep = [x for x in ['latitude', 'longitude', '#sampleid', 'dataset_name', 'env_biome', 'env_feature', 'env_material', 'collection_date', 'nuclear_contamination_status', 'facility_match', 'facility_distance_km'] if x in samples_df.columns]
    samples_df = samples_df[cols_to_keep]
    # 2. Immediately verify the result.
    print("\n--- Verifying data AFTER resetting index ---")
    print(samples_df.info())
    nfc_df = samples_df.copy()
    logger.info(f"Using {len(nfc_df)} cleaned facilities.")
    time.sleep(20)

    try:
        collector = GEEDataCollector(config, use_cache=True) 
        # The run method now returns the final, merged dataframe
        final_output = collector.run(nfc_df, lat_col='latitude', lon_col='longitude', scale=1000)
        
        if not final_output.empty and len(final_output.columns) > len(nfc_df.columns):
            print("\n--- Successfully retrieved and processed data ---")
            print(final_output.head())
        else:
            print("\n--- Process failed or returned no new data ---")

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"\nAn error occurred: {e}")

