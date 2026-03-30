# workflow_16s/api/environmental_data/google/arkin_env_agents.py

# ==================================================================================== #

# Standard Imports
import hashlib
import json
import logging
import math
import sys
import sqlite3
import pickle
import time
import tempfile
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Third-Party Imports
import pandas as pd
from pydantic import ValidationError
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Local Imports
from workflow_16s.config import AppConfig
from workflow_16s.utils.dir_utils import Project
from workflow_16s.utils.logger import with_logger
from workflow_16s.utils.metadata_utils import export_tsv, process_metadata, standardize_lat_lon_columns
from workflow_16s.utils.progress import get_progress_bar
# Add env_agents to path
# This is often better handled by project structure (e.g., pip install -e .)
# but we will keep it for now.
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from env_agents.adapters import CANONICAL_SERVICES # type: ignore
from env_agents.core.models import RequestSpec, Geometry # type: ignore


# ==================================================================================== #

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

# =============================== CACHE MANAGER ================================== #

class CustomJSONEncoder(json.JSONEncoder):
    """A custom JSON encoder that handles special data types."""
    def default(self, o: Any) -> Any:
        if isinstance(o, (datetime, date, pd.Timestamp)): return o.isoformat()
        if isinstance(o, pd.DataFrame): return o.to_dict(orient='records')
        if hasattr(o, 'getInfo'): return o.getInfo()
        return super().default(o)

#@with_logger
class SQLiteCacheManager:
    """Handles SQLite-backed caching for Arkin Environmental Agents."""
    def __init__(self, cache_dir: Path, db_name: str = "arkin_env_agents.db"):
        self.db_path = cache_dir / db_name
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.stats = {'hits': 0, 'misses': 0, 'writes': 0, 'errors': 0}
        self.failed_services = {}
        from workflow_16s.utils.logger import get_logger
        self.logger = get_logger("workflow_16s")

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    data BLOB,
                    timestamp REAL
                )
            """)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

    def get_cache_key(self, params: Dict[str, Any]) -> str:
        serialized = json.dumps(params, sort_keys=True).encode('utf-8')
        return hashlib.sha256(serialized).hexdigest()

    def get(self, key: str) -> Optional[List[Dict]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT data FROM cache WHERE key = ?", (key,))
                row = cursor.fetchone()
                if row:
                    self.stats['hits'] += 1
                    return pickle.loads(row[0])
        except Exception as e:
            self.logger.warning(f"Cache read error for {key}: {e}")
            self.stats['errors'] += 1
        
        self.stats['misses'] += 1
        return None

    def set(self, key: str, data: List[Dict]):
        if not data: return
        try:
            # We use pickle here because it handles the complex Earth Engine 
            # and Pandas types more robustly than raw JSON in a DB blob
            blob = pickle.dumps(data)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, data, timestamp) VALUES (?, ?, ?)",
                    (key, blob, time.time())
                )
            self.stats['writes'] += 1
        except Exception as e:
            self.logger.warning(f"Cache write error for {key}: {e}")
            self.stats['errors'] += 1
    
    def track_failed_service(self, service_name: str):
        """Track services that fail consistently."""
        self.failed_services[service_name] = self.failed_services.get(service_name, 0) + 1
    
    def should_skip_service(self, service_name: str, max_failures: int = 5) -> bool:
        """Check if service should be skipped due to consistent failures."""
        return self.failed_services.get(service_name, 0) >= max_failures
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total * 100) if total > 0 else 0
        return {
            **self.stats,
            'total_requests': total,
            'hit_rate_pct': round(hit_rate, 1)
        }

# ============================= CORE FUNCTIONS =================================== #

@with_logger
def standardize_column_name(df: pd.DataFrame, target_name: str, alternatives: list) -> pd.DataFrame:
    """
    Finds a column from a list of alternatives (case-insensitive) and renames it to a target name.

    Args:
        df: The DataFrame to modify.
        target_name: The final, standardized column name (e.g., 'run_accession').
        alternatives: A list of possible column names to search for.

    Returns:
        The modified DataFrame with the column renamed.

    Raises:
        KeyError: If none of the alternative columns are found.
    """
    df_columns_lower = {col.lower(): col for col in df.columns}

    # Check if the target column already exists
    if target_name in df.columns:
        return df

    # Search for alternatives
    for alt in alternatives:
        if alt.lower() in df_columns_lower:
            original_col_name = df_columns_lower[alt.lower()]
            logger.info(f"Found alternative column '{original_col_name}'. Renaming to '{target_name}'.")
            df.rename(columns={original_col_name: target_name}, inplace=True)
            return df

    # If no column was found, raise an error
    raise KeyError(f"Required column '{target_name}' not found. Searched for alternatives: {alternatives}")


def bbox_around_point(lat: float, lon: float, radius_km: float) -> Geometry:
    """Creates a bounding box Geometry around a point."""
    lat_deg_per_km = 1 / 110.574
    lon_deg_per_km = 1 / (111.320 * math.cos(math.radians(lat)) + 1e-9)
    buffer_lat = radius_km * lat_deg_per_km
    buffer_lon = radius_km * lon_deg_per_km
    return Geometry(type='bbox', coordinates=[lon - buffer_lon, lat - buffer_lat, lon + buffer_lon, lat + buffer_lon])

@with_logger
def _validate_services() -> Set[str]:
    """Checks prerequisites for services and returns a set of available services."""
    available_services = set(SERVICE_CONFIG.keys())

    # --- Check for Google Earth Engine ---
    try:
        import ee # type: ignore
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    except Exception as e:
        logger.warning(f"Disabling EARTH_ENGINE: Failed to initialize Google Earth Engine. Error: {e}")
        available_services.discard("EARTH_ENGINE")

    # --- Add other checks here for services that need keys/setup ---
    # Example:
    # if not os.getenv("SOME_API_KEY"):
    #     logger.warning("Disabling SOME_SERVICE: API key is not set in environment.")
    #     available_services.discard("SOME_SERVICE")

    return available_services

@with_logger
def fetch_service_data(
    service_name: str, spec: RequestSpec, cache_manager: SQLiteCacheManager,
    asset_info: Optional[Tuple[str, str]] = None,
    max_retries: int = 3
) -> List[Dict]:
    """Fetches data for a single service with retry logic and better error handling."""
    
    # Skip if service has failed too many times
    if cache_manager.should_skip_service(service_name):
        logger.debug(f"Skipping {service_name} due to consistent failures")
        return []
    
    # 1. Generate Key
    cache_params = {
        "service": service_name,
        "geometry": vars(spec.geometry),
        "time_range": spec.time_range,
        "asset_id": asset_info[0] if asset_info else None
    }
    cache_key = cache_manager.get_cache_key(cache_params)

    # 2. SQLite Look-up (Instant, even with 100k entries)
    if (cached_result := cache_manager.get(cache_key)) is not None:
        return cached_result

    # Retry logic for transient failures
    for attempt in range(max_retries):
        try:
            adapter_class = CANONICAL_SERVICES[service_name]
            adapter = adapter_class(asset_id=asset_info[0]) if asset_info else adapter_class()
            
            # Set timeout from config or use default
            timeout = SERVICE_CONFIG.get(service_name, {}).get('timeout', 300)
            
            result = adapter._fetch_rows(spec)

            if isinstance(result, pd.DataFrame):
                result = result.to_dict(orient='records')
            
            if result:
                cache_manager.set(cache_key, result)
                return result
            else:
                # Empty result is valid, cache it
                cache_manager.set(cache_key, [])
                return []
                
        except KeyError:
            logger.error(f"Service {service_name} not found in CANONICAL_SERVICES")
            cache_manager.track_failed_service(service_name)
            return []
        except TimeoutError:
            logger.warning(f"Timeout for {service_name}" + (f" ({asset_info[1]})" if asset_info else "") + f" (attempt {attempt+1}/{max_retries})")
            if attempt == max_retries - 1:
                cache_manager.track_failed_service(service_name)
            else:
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            error_msg = str(e)[:150]
            logger.error(f"Service Error for {service_name}" + (f" ({asset_info[1]})" if asset_info else "") + f": {error_msg}... (attempt {attempt+1}/{max_retries})")
            
            # Don't retry for certain error types
            if any(x in str(e).lower() for x in ['authentication', 'permission', 'quota exceeded']):
                cache_manager.track_failed_service(service_name)
                return []
            
            if attempt == max_retries - 1:
                cache_manager.track_failed_service(service_name)
                return []
            else:
                import time
                time.sleep(2 ** attempt)
    
    return []

@with_logger
def get_environmental_data_for_group(
    lat: float, lon: float, collection_date: str,
    cache_manager: SQLiteCacheManager, available_services: Set[str]
) -> pd.DataFrame:
    """Orchestrates data fetching from all available services for a single location group."""
    logger.info(f"Fetching data for location ({lat:.4f}, {lon:.4f}) on {collection_date}...")

    # FIX: Pad the end date by 1 day to prevent Earth Engine's "Empty date range" error
    try:
        dt_obj = datetime.strptime(collection_date, "%Y-%m-%d")
        end_date = (dt_obj + timedelta(days=15)).strftime("%Y-%m-%d")
        time_range = (collection_date, end_date)
    except ValueError:
        # Graceful fallback if date string formatting is unexpected
        time_range = (collection_date, collection_date)
        
    geometry = bbox_around_point(lat, lon, radius_km=1.0)
    all_results = []

    for service_name in available_services:
        config = SERVICE_CONFIG.get(service_name, {})
        spec = RequestSpec(geometry=geometry, time_range=time_range, extra=config)

        if service_name == "EARTH_ENGINE":
            for asset_id, asset_name in EE_ASSETS:
                result = fetch_service_data(service_name, spec, cache_manager, asset_info=(asset_id, asset_name))
                if result:
                    for row in result: row['service'] = f"EARTH_ENGINE_{asset_name}"
                    all_results.extend(result)
        else:
            result = fetch_service_data(service_name, spec, cache_manager)
            if result:
                for row in result: row['service'] = service_name
                all_results.extend(result)

    if not all_results:
        return pd.DataFrame()

    return pd.DataFrame(all_results)

@with_logger
def process_location_group(
    group_key: Tuple, group_df: pd.DataFrame,
    project_dir: Project, cache_manager: SQLiteCacheManager, available_services: Set[str]
) -> Optional[pd.DataFrame]:
    """
    Processes a group of samples sharing the same location and date, saves the
    data to a file, and returns it as a DataFrame.
    """
    lat, lon, collection_date = group_key
    try:
        # Define the primary name and its alternatives
        accession_alternatives = ['run_accession', 'accession', 'sample_id', '#sampleid']
        group_df = standardize_column_name(group_df, 'run_accession', accession_alternatives)
    except KeyError as e:
        logger.error(e)
        return None

    sample_ids = group_df['run_accession'].tolist()

    fusion_df = get_environmental_data_for_group(lat, lon, collection_date, cache_manager, available_services)

    if fusion_df.empty:
        logger.warning(f"No environmental data found for location ({lat:.4f}, {lon:.4f}) on {collection_date}.")
        return None

    # Add sample metadata to the results
    fusion_df['associated_sample_ids'] = ", ".join(sample_ids)
    fusion_df['sample_collection_date'] = collection_date
    fusion_df['sample_lat'] = lat
    fusion_df['sample_lon'] = lon

    # Save the result once per group
    output_dir = project_dir.raw_data / "environmental_context"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a filename safe from special characters
    safe_filename_key = f"{lat}_{lon}_{collection_date}".replace('.', '_')
    output_path = output_dir / f"env_data_{safe_filename_key}.tsv"

    fusion_df.to_csv(output_path, sep="\t", index=False)
    logger.debug(f"Saved {len(fusion_df)} records for {len(sample_ids)} samples to {output_path}")
    
    return fusion_df

# =============================== MAIN EXECUTION ================================= #

@with_logger
def main(metadata_path: Union[str, Path], project_dir: Project, progress_obj: Any = None) -> Optional[pd.DataFrame]:
    """
    Main function to group samples, fetch environmental data concurrently,
    and return a consolidated DataFrame. Dashboard-safe.
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return None

    df = pd.read_csv(metadata_path, sep="\t", low_memory=False)
    df = standardize_lat_lon_columns(df)

    # Ensure required columns exist and have the right type
    for col in ['run_accession', 'collection_date', 'lat', 'lon']:
        if col not in df.columns:
            logger.error(f"Required column '{col}' not found in metadata. Aborting.")
            return None
            
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df['collection_date'] = pd.to_datetime(df['collection_date'], errors='coerce').dt.strftime('%Y-%m-%d')

    valid_rows = df.dropna(subset=["run_accession", "collection_date", "lat", "lon"])

    # Group by unique location and date
    grouped = valid_rows.groupby(['lat', 'lon', 'collection_date'])
    logger.info(f"Found {len(valid_rows)} valid samples, grouped into {len(grouped)} unique location-date pairs for processing.")

    # Initialize cache and check for service availability ONCE
    cache_manager = SQLiteCacheManager(project_dir.cache / "env_agents")
    available_services = _validate_services()

    if not available_services:
        logger.error("No environmental services are available. Aborting.")
        return None

    # Unified Progress Handling
    p = progress_obj
    standalone = False
    if p is None:
        p = get_progress_bar()
        p.start()
        standalone = True

    all_group_dfs = []
    task = p.add_task("[cyan]Arkin Agents: Fetching contextual data", total=len(grouped))

    try:
        # Use ThreadPoolExecutor to process groups concurrently
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SAMPLES) as executor:
            futures = [
                executor.submit(process_location_group, key, group, project_dir, cache_manager, available_services)
                for key, group in grouped
            ]

            for future in as_completed(futures):
                try:
                    result_df = future.result()
                    if result_df is not None and not result_df.empty:
                        all_group_dfs.append(result_df)
                except Exception as e:
                    logger.error(f"A location group failed during processing: {e}", exc_info=True)
                finally:
                    p.update(task, advance=1)
    finally:
        if standalone:
            p.stop()
        else:
            p.remove_task(task)
    
    if not all_group_dfs:
        logger.warning("Environmental data fetching complete, but no data was collected.")
        return None

    # Consolidate all results into a single DataFrame
    final_df = pd.concat(all_group_dfs, ignore_index=True)
    logger.info(f"Environmental data fetching complete. Collected {len(final_df)} total records.")
    logger.info(f"Final DataFrame shape: {final_df.shape}")
    
    # Log column details
    # Convert dtypes to string to avoid potential formatting issues in logs
    dtype_info = final_df.dtypes.apply(lambda x: x.name).to_dict()
    logger.info(f"Columns: {json.dumps(dtype_info, indent=2)}")

    return final_df

# =============================== CLASS WRAPPER ================================== #
@with_logger
class ArkinEnvAgents:
    """
    Orchestrator for the Arkin Lab Environmental Agents.
    Groups samples by location/date and fetches satellite/soil/weather context.
    """
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = logger
        self.available_services = _validate_services()
        
        # Determine Earth Engine project from credentials config
        self.ee_project = getattr(config.credentials, 'google_earth_engine_project', None)
        if "EARTH_ENGINE" in self.available_services and self.ee_project:
            try:
                import ee
                ee.Initialize(project=self.ee_project, opt_url='https://earthengine-highvolume.googleapis.com')
                self.logger.info(f"GEE initialized successfully with project: {self.ee_project}")
            except Exception as e:
                self.logger.error(f"GEE initialization failed even with project ID: {e}")

    async def process_dataset(self, dataset_id: str, ena_metadata: pd.DataFrame, progress_obj: Any = None) -> Optional[pd.DataFrame]:
        """
        Entry point for the pipeline orchestrator.
        Takes a BioProject's metadata and enriches it with environmental data.
        """
        if ena_metadata.empty or not self.available_services:
            return None

        self.logger.info(f"🌿 [Arkin Agents] Starting environmental enrichment for {dataset_id}...")
        
        # Use a project object for pathing
        project_dir = Project(self.config)
        cache_manager = SQLiteCacheManager(project_dir.cache / "env_agents")

        # 1. Standardize and Clean Metadata
        df = ena_metadata.copy()
        df = standardize_lat_lon_columns(df)
        
        # Ensure numeric coordinates
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        
        # Handle date formatting (fallback to standard if possible)
        if 'collection_date' in df.columns:
            df['collection_date'] = pd.to_datetime(df['collection_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        else:
            self.logger.warning(f"No 'collection_date' found for {dataset_id}. Environmental context will be limited.")
            return None

        # Drop rows that are missing the "Holy Trinity" of environmental context
        valid_rows = df.dropna(subset=["run_accession", "collection_date", "lat", "lon"])
        if valid_rows.empty:
            self.logger.warning(f"No samples in {dataset_id} have enough spatio-temporal data for environmental context.")
            return None

        # 2. Group by unique location and date to minimize API calls
        grouped = valid_rows.groupby(['lat', 'lon', 'collection_date'])
        
        # 3. Concurrent Execution
        all_group_dfs = []
        
        # Plumbing for progress reporting
        p = progress_obj or get_progress_bar()
        task = p.add_task(f"[cyan]Arkin Agents: {dataset_id}", total=len(grouped))

        # We run the blocking ThreadPool in an executor to keep the main loop async
        loop = asyncio.get_event_loop()
        
        def _execute_sync_batch():
            results = []
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SAMPLES) as executor:
                futures = [
                    executor.submit(process_location_group, key, group, project_dir, cache_manager, self.available_services)
                    for key, group in grouped
                ]
                for future in as_completed(futures):
                    res = future.result()
                    if res is not None: results.append(res)
                    if progress_obj: p.update(task, advance=1)
            return results

        try:
            all_group_dfs = await loop.run_in_executor(None, _execute_sync_batch)
        finally:
            if not progress_obj: p.stop()
            else: p.remove_task(task)

        if not all_group_dfs:
            return None

        # 4. Consolidate and Return
        final_df = pd.concat(all_group_dfs, ignore_index=True)
        return final_df