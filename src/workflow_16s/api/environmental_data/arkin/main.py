# workflow_16s/api/environmental_data/arkin/main.py

import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

from workflow_16s.utils.dir_utils import Project
from workflow_16s.utils.logger import get_logger, with_logger
from workflow_16s.utils.progress import get_progress_bar
from workflow_16s.utils.metadata_utils import standardize_lat_lon_columns

from ..other.tools.cache import CacheManager 
from .constants import EE_ASSETS, SERVICE_CONFIG, MAX_CONCURRENT_SAMPLES
from .utils import bbox_around_point, validate_services, fetch_service_data


@with_logger
class ArkinEnvAgents:
    def __init__(self, config, progress_obj=None, **kwargs):
        self.config = config
        self.logger = get_logger("workflow_16s")
        self.available_services = validate_services()
        self.project_dir = Project(self.config)
        cache_dir = self.project_dir.cache / "env" 
        self.cache = CacheManager(cache_dir)
        
        self.ee_project = getattr(config.credentials, 'google_earth_engine_project', None)
        self._init_gee()

    def _init_gee(self):
        """Initializes Earth Engine with the project specified in config."""
        if "EARTH_ENGINE" in self.available_services and self.ee_project:
            try:
                import ee
                ee.Initialize(project=self.ee_project, opt_url='https://earthengine-highvolume.googleapis.com')
                self.logger.info(f"GEE initialized: {self.ee_project}")
            except Exception as e:
                self.logger.error(f"GEE initialization failed: {e}")

    async def process_dataset(self, dataset_id: str, ena_metadata: pd.DataFrame, progress_obj: Any = None) -> Optional[pd.DataFrame]:
        """High-level entry point to enrich a BioProject with environmental data."""
        if ena_metadata.empty or not self.available_services:
            return None

        # 1. Spatial Rounding (2 decimals ~1.1km) to maximize cache hits
        df = ena_metadata.copy()
        df = standardize_lat_lon_columns(df)
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce").round(2)
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce").round(2)
        df['collection_date'] = pd.to_datetime(df['collection_date'], errors='coerce').dt.strftime('%Y-%m-%d')

        valid_rows = df.dropna(subset=["run_accession", "collection_date", "lat", "lon"])
        if valid_rows.empty:
            return None

        grouped = valid_rows.groupby(['lat', 'lon', 'collection_date'])
        
        # 2. Rich Progress Integration
        p = progress_obj or get_progress_bar()
        task = p.add_task(f"[cyan]Arkin Agents: {dataset_id}", total=len(grouped))
        
        def _execute_batch():
            results = []
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SAMPLES) as executor:
                futures = [executor.submit(self.process_location_group, key, group) for key, group in grouped]
                for future in as_completed(futures):
                    res = future.result()
                    if res is not None: results.append(res)
                    p.update(task, advance=1)
            return results

        try:
            loop = asyncio.get_event_loop()
            all_group_dfs = await loop.run_in_executor(None, _execute_batch)
        finally:
            if progress_obj: p.remove_task(task)

        return pd.concat(all_group_dfs, ignore_index=True) if all_group_dfs else None

    def process_location_group(self, group_key: Tuple, group_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Fetches and structures data for a specific location-date pair."""
        lat, lon, collection_date = group_key
        fusion_df = self.get_environmental_data_for_group(lat, lon, collection_date)

        if fusion_df.empty: return None

        fusion_df['associated_sample_ids'] = ", ".join(group_df['run_accession'].astype(str).tolist())
        fusion_df['sample_collection_date'] = collection_date
        fusion_df['sample_lat'], fusion_df['sample_lon'] = lat, lon
        return fusion_df

    def get_environmental_data_for_group(self, lat: float, lon: float, collection_date: str) -> pd.DataFrame:
        """Core fetching logic using unified SQLite cache."""
        dt_obj = datetime.strptime(collection_date, "%Y-%m-%d")
        time_range = (collection_date, (dt_obj + timedelta(days=15)).strftime("%Y-%m-%d"))
        geometry = bbox_around_point(lat, lon, radius_km=1.0)
        
        all_results = []
        for service in self.available_services:
            result = fetch_service_data(service, geometry, time_range, self.cache)
            if result:
                for row in result: row['service'] = service
                all_results.extend(result)
        return pd.DataFrame(all_results)

@with_logger
async def run_arkin_enrichment(metadata_path: Union[str, Path], project_dir: Any, config: Any = None, progress_obj: Any = None) -> Optional[pd.DataFrame]:
    """
    Primary entry point used by processor.py.
    Initializes the class and processes the metadata file.
    """
    logger = get_logger("workflow_16s")
    if not Path(metadata_path).exists():
        logger.error(f"Metadata not found: {metadata_path}")
        return None

    if config is None:
        logger.error("Configuration not provided to run_arkin_enrichment")
        return None

    # Load config and initialize agents
    agents = ArkinEnvAgents(config)
    
    # Load metadata
    df = pd.read_csv(metadata_path, sep="\t", low_memory=False)
    dataset_id = Path(metadata_path).stem
    
    # Run enrichment
    return await agents.process_dataset(dataset_id, df, progress_obj=progress_obj)