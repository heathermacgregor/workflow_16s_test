import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import asyncio
import concurrent.futures

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN

from workflow_16s.api.sequence.ena import get_n_samples_by_bioproject_async
from workflow_16s.config import AppConfig
from workflow_16s.utils.dir_utils import Project
from workflow_16s.utils.logger import get_logger, with_logger
from workflow_16s.utils.progress import get_progress_bar
from workflow_16s.utils.metadata_utils import standardize_lat_lon_columns

from ...geospatial.universal_finder import UniversalFacilityFetcher
from ..other.tools.cache import CacheManager 
from .constants import SOURCE_PRIORITY, TYPE_DEFINITIONS, SOURCE_DEFAULTS
from .tools import (
    Analogs, DNFSBFacilityDB, load_facilities, GNPT, GeocodingService, 
    NFCFDB, JRC, world_uranium_mines, NRC, Wikidata, WNA
)
from .utils import (
    clean_fetched_ena_samples, sph2cart, resolve_facility_type,
    consolidate_and_merge_columns
)

# ✅ Module-level logger
logger = get_logger("workflow_16s")

@with_logger
class NFCFacilitiesHandler:
    """
    Handler for Nuclear & Contamination Analog Facilities.
    Aggregates databases, annotates samples, and builds cached AnnData atlases.
    Uses a unified SQLite cache and supports Rich UI dashboarding.
    """

    def __init__(self, config: AppConfig, progress_obj: Any = None, fetcher: Any = None, **kwargs):
        self.config = config
        if not self.config.nfc_facilities.enabled:
            return

        self.logger = kwargs.get('logger') or get_logger("workflow_16s")
        self.verbose = self.config.verbose
        self.max_distance_km = self.config.nfc_facilities.max_distance_km
        self.fetcher = fetcher
        self.email = self.config.credentials.ena_email
        
        # UI Plumbing: Support for the shared Rich dashboard
        self.progress = progress_obj if progress_obj else get_progress_bar()
        self._standalone = progress_obj is None

        # Pathing & Project Context
        self.project_dir = Project(self.config)
        self.output_dir = self.project_dir.raw_data / "_nfc_facilities"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Unified Cache Setup
        # Points to the same SQLite DB as 'arkin' and 'other' env modules
        cache_path = self.project_dir.cache / "env"
        self.logger.info(f"Initializing SQLite cache at: {cache_path}")
        self.cache = CacheManager(cache_path)

        # File Paths
        self.facilities_path = self.output_dir / 'facilities.tsv'
        self.atlas_path = self.output_dir / "nfc_atlas.h5ad"

        # Services
        self.geocoder = GeocodingService(config, self.output_dir)
        self.nfc_facilities_df = pd.DataFrame()

    async def nfc_facilities(self) -> pd.DataFrame:
        """Main Pipeline: Fetch -> Geocode -> De-duplicate -> Enrich."""
        self.logger.info("Starting Unified Facility Pipeline...")
        
        # 1. SQLite Cache Check
        cache_key = "master_facility_list"
        # 🟢 Use to_thread to prevent startup hang
        cached_df = await asyncio.to_thread(self.cache.get, cache_key)
        
        if cached_df is not None and not cached_df.empty:
            self.logger.info(f" 💿 Loaded {len(cached_df)} facilities from SQLite cache.")
            self.nfc_facilities_df = cached_df
            # 🟢 Give the event loop a millisecond to breathe before starting the sweep
            await asyncio.sleep(0.1) 
            return self.nfc_facilities_df

        # 2. Aggregation Phase
        df = self._get_data_concurrently()
        
        # 3. Geocoding & Refinement
        df = await self._geocode_facilities(df)
        df = self._deduplicate_by_proximity(df)
        df = self._enrich_metadata_standardized(df)
        
        self.nfc_facilities_df = standardize_lat_lon_columns(df)
        
        # 4. Final Save & Cache
        self.nfc_facilities_df.to_csv(self.facilities_path, sep='\t', index=False)
        self.cache.set(cache_key, self.nfc_facilities_df)
        
        self.logger.info(f" ✅ Created master facility database ({len(self.nfc_facilities_df)} sites).")
        return self.nfc_facilities_df

    def _get_data_concurrently(self) -> pd.DataFrame:
        """Aggregates all nuclear and analog databases using a thread pool."""
        self.logger.info(" 📑 Aggregating global facility databases...")
        
        loaders = {
            "nrc":      lambda: NRC().load(),
            "wna":      lambda: WNA().load(),
            "jrc":      lambda: JRC().load(),
            "wikidata": lambda: Wikidata().load(),
            "dnfsb":    lambda: DNFSBFacilityDB(str(self.output_dir)).load(),
            "gem":      lambda: GNPT(self.output_dir).load(),
            "nfcis":    lambda: NFCFDB(self.output_dir).load(),
            "iaea":     lambda: NFCFDB(self.output_dir).load(),
            "mindat":   lambda: world_uranium_mines(self.config),
            "analogs":  lambda: Analogs().load()
        }

        target_dbs = [db.lower() for db in self.config.nfc_facilities.databases]
        database_dfs = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_name = {}

            for name, loader_func in loaders.items():
                if name in target_dbs:
                    future_to_name[executor.submit(loader_func)] = name

            osm_fetcher = UniversalFacilityFetcher()
            for db_name in target_dbs:
                if db_name.startswith("osm_"):
                    facility_type = db_name.replace("osm_", "")
                    if facility_type in osm_fetcher.FACILITY_TAGS:
                        future_to_name[executor.submit(
                            lambda f=facility_type: asyncio.run(osm_fetcher.fetch_locations(f))
                        )] = db_name

            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    df = future.result()
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        df['data_source'] = name.upper()
                        database_dfs.append(df)
                        self.logger.info(f" ✅ '{name}' loaded {len(df)} facilities.")
                except Exception as e:
                    self.logger.error(f" ❌ Source '{name}' failed to load: {e}")

        if not database_dfs:
            return pd.DataFrame()

        combined = pd.concat(database_dfs, ignore_index=True)
        return consolidate_and_merge_columns(combined)

    async def _geocode_facilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing coordinates using the Geocoding service."""
        if 'lat' not in df.columns: df['lat'] = np.nan
        if 'lon' not in df.columns: df['lon'] = np.nan
        if 'facility' not in df.columns: df['facility'] = "Unknown"
        if 'country' not in df.columns: df['country'] = "Unknown"

        missing_mask = (df['lat'].isna()) | (df['lat'] == 0) | (df['lon'].isna())
        missing = df[missing_mask].copy()
        
        if missing.empty: 
            return df

        queries = (missing['facility'].astype(str) + ", " + missing['country'].astype(str)).unique().tolist()
        self.logger.info(f" 🗺️ Geocoding {len(queries)} missing facility locations...")
        
        results = await self.geocoder.geocode_batch(queries)
        coord_map = {q: res for q, res in zip(queries, results) if res}

        for idx, row in missing.iterrows():
            q = f"{row['facility']}, {row['country']}"
            if q in coord_map:
                df.at[idx, 'lat'] = coord_map[q]['lat']
                df.at[idx, 'lon'] = coord_map[q]['lon']
        return df

    def _deduplicate_by_proximity(self, df: pd.DataFrame, eps_km: float = 2.0) -> pd.DataFrame:
        """Merges duplicate facilities across databases using DBSCAN."""
        df = df.dropna(subset=['lat', 'lon']).copy()
        if df.empty: return df
        
        coords = np.radians(df[['lat', 'lon']].values)
        db = DBSCAN(eps=eps_km/6371.0, min_samples=1, metric='haversine').fit(coords)
        df['cluster_id'] = db.labels_
        
        df['priority'] = df['data_source'].map(SOURCE_PRIORITY).fillna(99)
        return df.sort_values('priority').groupby('cluster_id').first().reset_index(drop=True)

    def _enrich_metadata_standardized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Infers facility types using standardized regex patterns."""
        self.logger.info("Enriching facility metadata...")
        
        def apply_logic(row):
            return resolve_facility_type(
                name=row.get('facility', ''),
                raw_type=row.get('facility_type', ''),
                source=row.get('data_source', ''),
                definitions=TYPE_DEFINITIONS,
                defaults=SOURCE_DEFAULTS
            )

        df['facility_type_standard'] = df.apply(apply_logic, axis=1)
        df['facility_category'] = np.where(df.get('is_nuclear', True), 'Nuclear Fuel Cycle', 'Contamination Analog')
        return df

    async def get_nearby_samples(self, fetcher=None, task_id=None) -> pd.DataFrame:
        """High-level search for samples near known facilities."""
        self.logger.info("📡 DEBUG: Entering get_nearby_samples...")
        
        if self.nfc_facilities_df.empty: 
            await self.nfc_facilities()
            
        fetcher_to_use = fetcher if fetcher else self.fetcher
        
        # 🟢 CRITICAL CHECK: If this is None, the loop below will hang/crash silently
        if fetcher_to_use is None:
            self.logger.error("❌ CRITICAL: No fetcher provided to NFCFacilitiesHandler!")
            return pd.DataFrame()

        self.logger.info(f"📡 DEBUG: Starting sweep with fetcher: {type(fetcher_to_use)}")
        results = await self._run_global_sweep(self.nfc_facilities_df, fetcher_to_use, task_id=task_id)
        return results

    async def _run_global_sweep(self, target_coords: pd.DataFrame, fetcher: Any, desc: str = "Global Sweep", task_id: Any = None) -> pd.DataFrame:
        self.logger.info(f"🚀 DEBUG: Inside _run_global_sweep. Targets: {len(target_coords)}")
        
        if target_coords.empty: 
            return pd.DataFrame()

        local_task = False
        if task_id is None:
            if self._standalone: self.progress.start()
            task_id = self.progress.add_task(f"[cyan]{desc}", total=len(target_coords))
            local_task = True
            
        all_found_samples = []

        for i, (idx, row) in enumerate(target_coords.iterrows()):
            lat, lon = round(row['lat'], 2), round(row['lon'], 2)
            
            # 🟢 DEBUG: Print every 10th coordinate to verify movement
            if i % 10 == 0:
                self.logger.info(f"🛰️ Processing facility {i}/{len(target_coords)}: ({lat}, {lon})")
            facility_name = row.get('facility', f"Point_{lat}_{lon}")
            resume_key = f"sweep_done_{lat}_{lon}_r5"

            # 🟢 NON-BLOCKING GET: Offload to a background thread
            cached_data = await asyncio.to_thread(self.cache.get, resume_key)
            
            if cached_data is not None:
                if isinstance(cached_data, list):
                    all_found_samples.extend(cached_data)
                self.progress.update(task_id, advance=1)
                continue

            try:
                radius = self.max_distance_km if self.max_distance_km else 5.0
                samples = await fetcher.find_nearby_samples(lat, lon, radius=radius)
                
                current_hits = [] 
                # 🟢 STANDARDIZATION LOGIC: Now safe because 'samples' is awaited
                if isinstance(samples, pd.DataFrame):
                    if not samples.empty: # This is where the 'coroutine' error happened
                        df_to_add = samples.copy()
                        df_to_add['query_lat'], df_to_add['query_lon'] = lat, lon
                        df_to_add['target_facility'] = facility_name
                        current_hits = df_to_add.to_dict('records')
                elif isinstance(samples, list) and len(samples) > 0:
                    for s in samples:
                        s['query_lat'], s['query_lon'] = lat, lon
                        s['target_facility'] = facility_name
                    current_hits = samples

                # 🟢 NON-BLOCKING SET: Prevent the UI from hanging during the write
                await asyncio.to_thread(self.cache.set, resume_key, current_hits if current_hits else "EMPTY")
                all_found_samples.extend(current_hits)
                    
            except Exception as e:
                self.logger.debug(f"Sweep error at {facility_name}: {e}")
            finally:
                self.progress.update(task_id, advance=1)

        if local_task:
            if self._standalone: self.progress.stop()
            else: self.progress.remove_task(task_id)

        if not all_found_samples:
            return pd.DataFrame()

        sweep_df = pd.DataFrame(all_found_samples)
        
        # 🟢 FIXED: Flexible deduplication to prevent KeyError
        id_col = 'sample_accession' if 'sample_accession' in sweep_df.columns else 'accession'
        return sweep_df.drop_duplicates(subset=[id_col]) if id_col in sweep_df.columns else sweep_df

    def annotate_samples(self, samples_df: pd.DataFrame) -> pd.DataFrame:
        """Calculates multi-view distances for every sample."""
        self.logger.info(f" 📝 Annotating {len(samples_df)} samples with NFC proximity features...")
        return self._calculate_proximities(self.nfc_facilities_df, samples_df)

    def _calculate_proximities(self, facilities: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
        """Vectorized distance calculations using BallTree."""
        s_coords = np.radians(samples[['lat', 'lon']].values)
        f_coords = np.radians(facilities[['lat', 'lon']].values)
        
        tree = BallTree(f_coords, metric='haversine')
        dists, idxs = tree.query(s_coords, k=1)
        
        samples['facility_distance_km'] = dists.flatten() * 6371.0
        samples['nearest_facility'] = facilities.iloc[idxs.flatten()]['facility'].values
        return samples
    
    def plot_global_facilities_map(self, output_filename: str = "global_facilities_map.html"):
        """Generates the interactive Plotly map of all aggregated facilities."""
        from workflow_16s.visualization.nuclear_fuel_cycle.main import plot_global_facilities_map as _plot_map
        
        if self.nfc_facilities_df.empty:
            self.logger.warning("No facilities to plot. Run nfc_facilities() first.")
            return

        self.logger.info(f" 🗺️ Generating global facilities map: {output_filename}")
        dest = self.output_dir / output_filename
        return _plot_map(self.nfc_facilities_df, self.output_dir, output_filename)