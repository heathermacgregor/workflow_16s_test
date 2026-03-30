# workflow_16s/src/workflow_16s/api/nuclear_fuel_cycle/nfc.py
# ===================================== IMPORTS ====================================== #

import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Standard Imports
import logging
import pickle
import re
import time
import concurrent.futures
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, Set, Tuple, Union

# Third Party Imports
import anndata as ad
import aiohttp
import asyncio
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN

# Local Imports
from workflow_16s.api.sequence.ena import (
    run_searches_from_dataframe_async,
    get_n_samples_by_bioproject_async,
    SQLiteCacheManager as EnaCacheManager
)
from workflow_16s.config import AppConfig
from workflow_16s.utils.dir_utils import Project
from workflow_16s.utils.metadata_utils import process_metadata, standardize_lat_lon_columns
from workflow_16s.utils.progress import get_progress_bar
from workflow_16s.utils.logger import get_logger
from workflow_16s.visualization.nfc import (
    plot_all_facilities_and_fetched_samples_geo, plot_all_facilities_and_samples_geo,
    plot_facility_with_samples_geo
)

# --- INTEGRATED SOURCES ---
from . import (
    _dnfsb, _gem, _iaea, _mindat, 
    _nrc, _jrc, _wna, _wikidata,
    _analogs, _geocode
)

# ==================================================================================== #

logger = logging.getLogger("workflow_16s")

# ==================================================================================== #
# [Caching Decorators]
# ==================================================================================== #

def cache_to_file(cache_filename: str) -> Callable:
    """Decorator to cache the entire result of a method using pickle."""
    logger = get_logger("workflow_16s")
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> pd.DataFrame:
            cache_file_path = self.cache_dir / cache_filename
            if self.use_local and cache_file_path.exists():
                logger.info(f"Loading cached data from: {cache_file_path}")
                try:
                    with open(cache_file_path, 'rb') as f: 
                        loaded_obj = pickle.load(f)
                    if isinstance(loaded_obj, pd.DataFrame): 
                        return loaded_obj
                except Exception as e:
                    logger.warning(f"Cache load failed: {e}")

            result_df = func(self, *args, **kwargs)
            if not result_df.empty:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                with open(cache_file_path, 'wb') as f: 
                    pickle.dump(result_df, f)
            return result_df
        return wrapper
    return decorator

def cache_rows_to_file(cache_filename: str, id_column: str) -> Callable:
    logger = get_logger("workflow_16s")
    """Decorator to cache DataFrame results with incremental updates using pickle."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, input_df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
            cache_file_path = self.cache_dir / cache_filename
            if input_df.empty: 
                return pd.DataFrame()

            cached_df = pd.DataFrame()
            cached_ids: Set[str] = set()

            if self.use_local and cache_file_path.exists():
                try:
                    with open(cache_file_path, 'rb') as f: 
                        cached_df = pickle.load(f)
                    if not cached_df.empty and id_column in cached_df.columns:
                        cached_ids = set(cached_df[id_column].astype(str).unique())
                except: 
                    pass

            input_ids = set(input_df[id_column].astype(str))
            unprocessed_ids = input_ids - cached_ids

            if not unprocessed_ids:
                return cached_df[cached_df[id_column].astype(str).isin(input_ids)].copy()

            unprocessed_df = input_df[input_df[id_column].astype(str).isin(unprocessed_ids)].copy()
            logger.info(f"Processing {len(unprocessed_df)} new rows for {cache_filename}...")

            new_results = await func(self, unprocessed_df, *args, **kwargs)
            
            if new_results.empty:
                return cached_df[cached_df[id_column].astype(str).isin(input_ids)].copy()

            combined = pd.concat([cached_df, new_results], ignore_index=True).drop_duplicates(subset=[id_column])
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file_path, 'wb') as f: 
                pickle.dump(combined, f)
            
            return combined[combined[id_column].astype(str).isin(input_ids)].copy()
        return wrapper
    return decorator

def sph2cart(lats, lons, R=6371) -> np.ndarray:
    φ = np.radians(lats.astype(float))
    λ = np.radians(lons.astype(float))
    x = R * np.cos(φ) * np.cos(λ)
    y = R * np.cos(φ) * np.sin(λ)
    z = R * np.sin(φ)
    return np.column_stack((x, y, z))


# [Data Cleaning]
def consolidate_and_merge_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    priority_prefixes = ['run_', 'experiment_', 'biosample_', 'study_']
    base_names = set()
    for col in df_copy.columns:
        for prefix in priority_prefixes:
            if col.startswith(prefix): 
                base_names.add(col[len(prefix):])
                break

    for base_name in sorted(list(base_names)):
        if base_name == 'accession': continue
        existing_cols = [
            f"{p}{base_name}" 
            for p in priority_prefixes 
            if f"{p}{base_name}" in df_copy.columns
        ]
        if len(existing_cols) < 2: continue

        merged_series = df_copy[existing_cols[0]]
        for i in range(1, len(existing_cols)):
            merged_series = merged_series.combine_first(df_copy[existing_cols[i]])

        df_copy[base_name] = merged_series
        df_copy = df_copy.drop(columns=existing_cols)

    return df_copy

def clean_fetched_ena_samples(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    cleaned_df = df.copy()

    if '#sampleid' in cleaned_df.columns:
        cleaned_df.drop_duplicates(subset=['#sampleid'], keep='first', inplace=True)

    lat_cols = [c for c in cleaned_df.columns if 'lat' in c.lower() and c != 'lat']
    lon_cols = [c for c in cleaned_df.columns if 'lon' in c.lower() and c != 'lon']
    
    if 'lat' not in cleaned_df.columns: cleaned_df['lat'] = np.nan
    if 'lon' not in cleaned_df.columns: cleaned_df['lon'] = np.nan

    # FIX: Explicitly convert existing primary columns to numeric first
    cleaned_df['lat'] = pd.to_numeric(cleaned_df['lat'], errors='coerce')
    cleaned_df['lon'] = pd.to_numeric(cleaned_df['lon'], errors='coerce')

    for col in lat_cols: 
        cleaned_df['lat'] = cleaned_df['lat'].combine_first(pd.to_numeric(cleaned_df[col], errors='coerce'))
    for col in lon_cols: 
        cleaned_df['lon'] = cleaned_df['lon'].combine_first(pd.to_numeric(cleaned_df[col], errors='coerce'))

    cleaned_df.dropna(subset=['#sampleid', 'lat', 'lon'], inplace=True)
    
    # These comparisons now succeed because 'lat' and 'lon' are guaranteed floats
    cleaned_df = cleaned_df[
        (cleaned_df['lat'] >= -90) & (cleaned_df['lat'] <= 90)
        & (cleaned_df['lon'] >= -180) & (cleaned_df['lon'] <= 180)
    ]
    cleaned_df = cleaned_df[~((cleaned_df['lat'] == 0) & (cleaned_df['lon'] == 0))]
    return cleaned_df.reset_index(drop=True)

async def clean_df(df, config) -> pd.DataFrame:
    df = consolidate_and_merge_columns(df)
    df['#sampleid'] = df.get('run_accession', df.index.astype(str))
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    df = clean_fetched_ena_samples(df)
    df = df.drop_duplicates(subset=['#sampleid'])
    df = await process_metadata(df=df, output_path="", config=config)
    return df


# [Main Handler Class]
class NFCFacilitiesHandler:
    """
    Handler for Nuclear & Contamination Analog Facilities.
    Aggregates databases, annotates samples, and builds cached AnnData atlases.
    """
    
    SOURCE_PRIORITY = {
        "NFCIS": 1, "NRC": 2, "JRC": 3, "WNA": 4, 
        "DNFSB": 5, "GEM": 6, "WIKIDATA": 7, "MINDAT": 8,
        "WIKIDATA_ANALOG": 9
    }

    mindat_columns_to_keep = [
        "facility", "country", "lat", "lon", "elements",
        "refs", "wikipedia", "data_source"
    ]

    def __init__(self, config: AppConfig, progress_obj: Any = None, fetcher: Any = None):
        self.config = config
        if not self.config.nfc_facilities.enabled: return

        self.verbose = self.config.verbose
        self.logger = get_logger("workflow_16s")
        self.user_agent = config.web.user_agent or "workflow_16s/1.0"
        self.use_local = self.config.nfc_facilities.use_local
        self.database_names = [db.lower() for db in self.config.nfc_facilities.databases]
        self.email = self.config.credentials.ena_email
        self.fetcher = fetcher
        # Plumbing for Unified UI
        self.progress = progress_obj if progress_obj else get_progress_bar()
        self._standalone = progress_obj is None

        self.project_dir = Project(self.config)
        if self.project_dir:
            self.output_dir = self.project_dir.raw_data / "_nfc_facilities"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir = self.project_dir.cache / "nfc_facilities"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Paths
            self.facilities_path = self.output_dir / 'facilities.tsv'
            self.facilities_geocoded_path = self.output_dir / 'facilities_geocoded.tsv'
            self.nearby_samples_path = self.output_dir / f"nearby_samples_{self.config.nfc_facilities.max_distance_km}km.tsv"
            self.matches_output_path = self.output_dir / f"matches_{self.config.nfc_facilities.distance_threshold_km}km.tsv"
            self.atlas_path = self.output_dir / "nfc_atlas.h5ad"

        self.geocoder = _geocode.GeocodingService(config, self.output_dir)
        self.nfc_facilities_df = pd.DataFrame()
        self.nearby_samples_df = pd.DataFrame()

    def log(self, msg):
        return (lambda msg: self.logger.debug(msg)) if self.verbose else (lambda *_: None)

    async def nfc_facilities(self) -> pd.DataFrame:
        """Main Pipeline: Fetch -> Geocode -> De-duplicate -> Enrich."""
        self.logger.info(f"Starting Unified Facility Pipeline...")
        if self.use_local and self.facilities_path.exists():
            self.logger.info(f" 💿 Loading cached facilities from {self.facilities_path}")
            try:
                df = pd.read_csv(self.facilities_path, sep='\t')
                self.nfc_facilities_df = standardize_lat_lon_columns(df)
                if hasattr(self.config.nfc_facilities, 'maps') and self.config.nfc_facilities.maps:
                    self.plot_global_facilities_map()
                return self.nfc_facilities_df
            except Exception as e:
                self.logger.warning(f" ⚠️ Failed to load cached facilities: {e}")
                pass
            
        df = self._get_data()
        df = await self._geocode(df)
        df = self._deduplicate_facilities(df)
        df = self._enrich_metadata(df)
        self.nfc_facilities_df = standardize_lat_lon_columns(df)
            
        if self.project_dir:
            self.nfc_facilities_df.to_csv(self.facilities_path, sep='\t', index=False)
            self.logger.info(f" 💾 Saved master facility database ({len(self.nfc_facilities_df)} sites) to {self.facilities_path}.")
            if hasattr(self.config.nfc_facilities, 'maps') and self.config.nfc_facilities.maps:
                self.plot_global_facilities_map()
        return self.nfc_facilities_df

    async def match_samples(self, adata, distance_threshold_km: Optional[float] = None):
        """
        Refined matching logic that prioritizes high-quality coordinate columns
        and uses Haversine math for high-precision forensic hits.
        """
        # Get distance threshold 
        if distance_threshold_km is None:
            distance_threshold_km = self.config.nfc_facilities.distance_threshold_km

        if self.nfc_facilities_df.empty:
            await self.nfc_facilities()
        
        facilities_df = self.nfc_facilities_df
        if facilities_df.empty:
            self.logger.warning(" ⚠️ No facilities found. Skipping matching.")
            return

        # 2. Coordinate Column Priority: Use the most complete data
        # Prioritize 'lat_sample' (backfilled) over 'latitude' (original)
        lat_col = next((c for c in adata.obs.columns if c.lower() in ['lat_sample', 'latitude', 'lat']), None)
        lon_col = next((c for c in adata.obs.columns if c.lower() in ['lon_sample', 'longitude', 'lon']), None)

        if not lat_col or not lon_col:
            self.logger.warning(" ⚠️ No coordinate columns found in adata.obs.")
            return

        # 3. Build Precision Tree (Haversine/Radians)
        valid_facilities = facilities_df.dropna(subset=['lat', 'lon']).copy()
        f_coords = np.radians(valid_facilities[['lat', 'lon']].values)
        tree = BallTree(f_coords, metric='haversine')

        # 4. Process Samples
        # Identify samples that actually have coordinates (ignore the 0,0 trap)
        valid_sample_mask = adata.obs[lat_col].notnull() & (adata.obs[lat_col] != 0)
        if not valid_sample_mask.any():
            self.logger.warning(" ⚠️ No samples have valid coordinates for matching.")
            return

        sample_coords = np.radians(adata.obs.loc[valid_sample_mask, [lat_col, lon_col]].values)
        
        # Query nearest facility
        dists, indices = tree.query(sample_coords, k=1)
        distances_km = dists.flatten() * 6371.0
        
        # 5. Assign Logical Flags
        is_industry = distances_km <= distance_threshold_km
        nearest_is_nuclear = valid_facilities.iloc[indices.flatten()]['is_nuclear'].values
        
        # Initialize all as False/NaN
        adata.obs['industry_match'] = False
        adata.obs['facility_match'] = False
        adata.obs['analog_match'] = False
        adata.obs['facility_distance_km'] = np.nan

        # Map results back to the valid samples
        adata.obs.loc[valid_sample_mask, 'industry_match'] = is_industry
        adata.obs.loc[valid_sample_mask, 'facility_match'] = is_industry & nearest_is_nuclear
        adata.obs.loc[valid_sample_mask, 'analog_match'] = is_industry & (~nearest_is_nuclear)
        adata.obs.loc[valid_sample_mask, 'facility_distance_km'] = distances_km

        # 6. Map Metadata (Name, Type, etc.)
        nearest_facs = valid_facilities.iloc[indices.flatten()]
        # Only update the samples that actually matched
        match_mask = valid_sample_mask.copy()
        match_mask[valid_sample_mask] = is_industry # Only samples near enough
        
        adata.obs.loc[match_mask, 'facility_name'] = nearest_facs.loc[is_industry, 'facility'].values
        adata.obs.loc[match_mask, 'facility_type'] = nearest_facs.loc[is_industry, 'facility_type_standard'].values
        
        self.logger.info(f" ✅ Matching complete. Found {adata.obs['facility_match'].sum()} Nuclear hits using {lat_col}.")
        
    def annotate_samples(self, samples_df: pd.DataFrame) -> pd.DataFrame:
        """Updates sample metadata with category-specific proximities."""
        if self.nfc_facilities_df.empty:
            self.logger.warning("Facilities not loaded. Run nfc_facilities() first.")
            return samples_df
        self.logger.info(f" 📝 Annotating {len(samples_df)} samples with Multi-View proximity features...")
        return self._multi_view_matching(self.nfc_facilities_df, samples_df)

    def build_atlas(self, samples_df: pd.DataFrame, data_dir: Union[str, Path]) -> Optional[ad.AnnData]:
        """
        Loads biological data (.h5ad) for matching samples, merges them, 
        injects enriched metadata, and SAVES the result to disk (Caching).
        """
        if self.use_local and self.atlas_path.exists():
            self.logger.info(f" 💿 Loading cached NFC Atlas from {self.atlas_path}")
            return ad.read_h5ad(self.atlas_path)

        data_dir = Path(data_dir)
        if not data_dir.exists():
            self.logger.error(f" 🚫 Data directory {data_dir} does not exist. Cannot build atlas.")
            return None

        annotated_meta = self.annotate_samples(samples_df)
        valid_ids = annotated_meta.index.astype(str).tolist()
        
        adatas = []
        self.logger.info(f" 🚧 Building Atlas: Scanning {data_dir} for {len(valid_ids)} samples...")
        
        found_count = 0
        
        # Unified Dashboard safe start/stop logic
        if self._standalone: self.progress.start()
        
        task = self.progress.add_task("[yellow]Building NFC Atlas", total=len(valid_ids))
        for sample_id in valid_ids:
            fpath = data_dir / f"{sample_id}.h5ad"
            if not fpath.exists(): 
                fpath = data_dir / f"{sample_id}_raw.h5ad"
            
            if fpath.exists():
                try:
                    a = ad.read_h5ad(fpath)
                    a.obs.index = a.obs.index.astype(str)
                    for col in annotated_meta.columns:
                        a.obs[col] = annotated_meta.loc[sample_id, col]
                    adatas.append(a)
                    found_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to load {fpath}: {e}")
            self.progress.update(task, advance=1)

        if self._standalone: self.progress.stop()
        else: self.progress.remove_task(task)

        if not adatas:
            self.logger.warning(" ⚠️ No matching .h5ad files found. Atlas not created.")
            return None

        self.logger.info(f" 🗂️ Concatenating {found_count} samples into master Atlas...")
        try:
            master_adata = ad.concat(
                adatas, 
                join="outer", 
                label="batch", 
                keys=[str(i) for i in range(len(adatas))]
            )
            master_adata.obs = master_adata.obs.join(
                annotated_meta, 
                rsuffix="_meta", 
                how="left"
            )
            
            master_adata.write(self.atlas_path)
            self.logger.info(f" 💾 Saved NFC Atlas to {self.atlas_path}.")
            return master_adata
            
        except Exception as e:
            self.logger.error(f" 🚫 Atlas concatenation failed: {e}")
            return None

    @cache_to_file(cache_filename="01_raw_facilities.pkl")
    def _get_data(self) -> pd.DataFrame:
        """Aggregates data from all loaders and enforces schema standards."""
        self.logger.info(" 📑 Aggregating data for all facilities (Nuclear + Nuclear Analogs)...")
        
        # Loaders dict with common aliases integrated
        loaders = {
            "dnfsb": lambda: _dnfsb.load_facilities(self.config),
            "mindat": lambda: _mindat.world_uranium_mines(self.config)[self.mindat_columns_to_keep],
            "gem": lambda: _gem.GNPT(self.output_dir).load(),
            "nfcis": lambda: _iaea.NFCFDB(self.output_dir).load(),
            "iaea": lambda: _iaea.NFCFDB(self.output_dir).load(), # Alias for IAEA
            "nrc": lambda: _nrc.NRC().load(),
            "wna": lambda: _wna.WNA().load(),
            "jrc": lambda: _jrc.JRC().load(),
            "wikidata": lambda: _wikidata.Wikidata().load(),
            "analogs": lambda: _analogs.Analogs().load()
        }
        
        target_dbs = self.config.nfc_facilities.databases or loaders.keys()
        database_dfs = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_name = {}
            for name, loader in loaders.items():
                if any(name.lower() == d.lower() for d in target_dbs) or name == "analogs":
                    future_to_name[executor.submit(loader)] = name
            
            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    df = future.result()
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        # Normalize Columns Internally
                        rename_map = {
                            'Country': 'country', 
                            'facility_country': 'country',
                            'Facility': 'facility',
                            'Facility_Type': 'facility_type',
                            'Latitude': 'lat', 
                            'Longitude': 'lon'
                        }
                        df.rename(columns=rename_map, inplace=True)
                        
                        if 'data_source' not in df.columns: 
                            df['data_source'] = name.upper()
                        else: 
                            df['data_source'] = df['data_source'].str.upper()
                        database_dfs.append(df)
                    else:
                        self.logger.warning(f" ⚠️ Loader '{name}' returned empty or invalid data.")
                except Exception as e: 
                    self.logger.error(f"Loader '{name}' failed: {e}")
        
        if not database_dfs: 
            self.logger.warning(" ⚠️ No facility data sources loaded successfully.")
            return pd.DataFrame()
            
        combined = pd.concat(database_dfs, axis=0, ignore_index=True)
        
        # Enforce Strict Schema
        required_cols = ['country', 'facility', 'lat', 'lon', 'facility_type']
        for col in required_cols:
            if col not in combined.columns:
                self.logger.warning(f" ⚠️ Column '{col}' missing from aggregated data. Creating it with NaNs.")
                combined[col] = np.nan
        
        if 'is_nuclear' not in combined.columns: 
            combined['is_nuclear'] = True
        combined['is_nuclear'] = combined['is_nuclear'].fillna(True)
        
        combined['facility_category'] = np.where(combined['is_nuclear'], 'Nuclear Fuel Cycle', 'Contamination Analog')
        
        # Clean string columns
        obj_cols = combined.select_dtypes(include="object").columns
        combined[obj_cols] = combined[obj_cols].apply(lambda x: x.astype(str).str.replace(r"\s+", " ", regex=True).str.strip())
        
        # Safe Country Replacement
        if 'country' in combined.columns:
            combined["country"] = combined["country"].replace({
                "USA": "United States of America", 
                "UK": "United Kingdom"
            })
        else:
            self.logger.warning(" ⚠️ 'country' column missing from facility data. Skipping standardization.")
            
        return combined

    def _deduplicate_facilities(self, df: pd.DataFrame, eps_km: float = 2.0) -> pd.DataFrame:
        self.logger.info(f" 🏷️ De-duplicating facilities within {eps_km} km...")
        df = df.copy()
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        valid_coords = df.dropna(subset=['lat', 'lon'])
        if valid_coords.empty: return df
        df['priority_score'] = df['data_source'].str.upper().map(self.SOURCE_PRIORITY).fillna(99)
        coords = np.radians(valid_coords[['lat', 'lon']].values)
        db = DBSCAN(
            eps=eps_km/6371.0088, 
            min_samples=1, 
            metric='haversine', 
            algorithm='ball_tree'
        ).fit(coords)
        df.loc[valid_coords.index, 'cluster_id'] = db.labels_
        merged_rows = []
        no_coords = df[df['cluster_id'].isna()]
        if not no_coords.empty: merged_rows.append(no_coords)
        def merge_cluster(group):
            group = group.sort_values('priority_score')
            best_row = group.iloc[0].copy()
            if len(group) > 1:
                for _, row in group.iloc[1:].iterrows():
                    best_row = best_row.fillna(row)
            best_row['merged_sources'] = ",".join(group['data_source'].unique())
            return best_row
        deduplicated = df.dropna(subset=['cluster_id']).groupby('cluster_id').apply(merge_cluster)
        if merged_rows: final_df = pd.concat([deduplicated, *merged_rows], ignore_index=True)
        else: final_df = deduplicated.reset_index(drop=True)
        return final_df

    def _enrich_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Enriching metadata: Standardizing types and inferring from Source/Name...")
        
        # 1. Regex Map: Patterns to look for in 'facility_type' OR 'facility' name
        #    Format: { 'Standardized Name': [list of regex patterns] }
        type_definitions = {
            'Nuclear Power Plant': [
                r'(?i)pressuri[sz]ed|PWR', r'(?i)boiling|BWR', r'(?i)candu|heavy|PHWR', 
                r'(?i)reactor', r'(?i)npp', r'(?i)power plant', r'(?i)generating station'
            ],
            'Uranium Mine': [
                r'(?i)uranium.*mine', r'(?i)open.*pit', r'(?i)in-situ', r'(?i)isr', 
                r'(?i)deposit', r'(?i)prospect', r'(?i)project' # Common in Mindat names
            ],
            'Uranium Mill': [r'(?i)mill'],
            'Salt Mine': [r'(?i)salt'],
            'Potash Mine': [r'(?i)potash'],
            'Fertilizer Plant': [r'(?i)fertilizer|nitrate|ammonia'],
            'Acid Plant': [r'(?i)acid', r'(?i)sulfuric'],
            'Coal Power Plant': [r'(?i)coal'],
            'Gold Mine': [r'(?i)gold'],
            'Copper Mine': [r'(?i)copper'],
            'Smelter': [r'(?i)aluminium|smelter'],
            'Desalination Plant': [r'(?i)desalination'],
            'Geothermal Plant': [r'(?i)geothermal'],
            'Waste Storage': [r'(?i)storage', r'(?i)repository', r'(?i)disposal'],
            'Superfund Site': [r'(?i)superfund']
        }

        # 2. Source Map: Default types based on the database origin
        #    This catches "Rossing" (from MINDAT) even if the name lacks "Mine"
        source_defaults = {
            'MINDAT': 'Uranium Mine',
            'WNA': 'Nuclear Power Plant', # Mostly NPPs
            'PRIS': 'Nuclear Power Plant'
        }

        def resolve_type(row):
            # A. Get relevant fields
            f_type = str(row['facility_type']) if pd.notna(row['facility_type']) else ""
            f_name = str(row['facility']) if pd.notna(row['facility']) else ""
            f_source = str(row['data_source']).upper() if pd.notna(row['data_source']) else ""

            # B. Check for Explicit Match in 'facility_type' column first
            for std_name, patterns in type_definitions.items():
                for pat in patterns:
                    if re.search(pat, f_type):
                        return std_name

            # C. Check 'facility' Name for keywords
            for std_name, patterns in type_definitions.items():
                for pat in patterns:
                    if re.search(pat, f_name):
                        return std_name
            
            # D. Fallback: Check Data Source
            if f_source in source_defaults:
                return source_defaults[f_source]

            # E. Keep original if not empty, else Unknown
            return f_type if f_type and f_type.lower() != 'nan' else 'Unknown'

        # Apply logic
        if 'facility_type' not in df.columns:
            df['facility_type'] = np.nan
            
        df['facility_type_standard'] = df.apply(resolve_type, axis=1)
        
        # Validation Log
        if self.verbose:
            counts = df['facility_type_standard'].value_counts()
            logger.info(f" ✅ Enrichment Complete. Top types inferred:\n{counts.head(5)}")

        return df

    @cache_rows_to_file(cache_filename="02_geocoded_facilities.pkl", id_column="facility")
    async def _geocode(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'lat' not in df.columns: df['lat'] = np.nan
        if 'lon' not in df.columns: df['lon'] = np.nan
        missing_mask = (df['lat'].isna()) | (df['lon'].isna()) | (df['lat'] == 0) | (df['lon'] == 0)
        missing = df[missing_mask].copy()
        
        if missing.empty:
            return df
            
        # Create search queries: "Facility Name, Country"
        unique_q = (missing['facility'].fillna('') + ", " + missing['country'].fillna('')).unique().tolist()
        
        logger.info(f" 🗺️ Geocoding {len(unique_q)} facilities...")
        results = await self.geocoder.geocode_batch(unique_q)
        
        # Create mapping dictionary
        coord_map = {q: res for q, res in zip(unique_q, results) if res}
        
        # Apply mapping
        for idx, row in missing.iterrows():
            query = f"{row['facility']}, {row['country']}"
            if query in coord_map:
                df.at[idx, 'lat'] = coord_map[query]['lat']
                df.at[idx, 'lon'] = coord_map[query]['lon']
                
        return df

    def _multi_view_matching(self, facilities: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
        EARTH_RADIUS_KM = 6371
        samples = samples.copy()
        
        # Ensure we can restore the index later
        if "original_index" not in samples.columns: 
            samples["original_index"] = samples.index
            
        # Normalize coords
        if 'lat' not in samples.columns: samples.rename(columns={'latitude_deg': 'lat'}, inplace=True)
        if 'lon' not in samples.columns: samples.rename(columns={'longitude_deg': 'lon'}, inplace=True)
        
        valid_mask = samples[['lat', 'lon']].notnull().all(axis=1)
        valid_samples = samples[valid_mask].copy()
        
        if valid_samples.empty: return samples
        
        s_coords = np.radians(valid_samples[['lat', 'lon']].values)

        # 1. Distance Views (Specific categories)
        views = {
            "dist_nuclear_km":   facilities['is_nuclear'] == True,
            "dist_analog_km":    facilities['is_nuclear'] == False,
            "dist_salts_km":     facilities['facility_type_standard'].str.contains(r'Salt|Potash', case=False, na=False),
            "dist_nitrates_km": facilities['facility_type_standard'].str.contains(r'Fertilizer|Acid', case=False, na=False),
            "dist_metals_km":    facilities['facility_type_standard'].str.contains(r'Gold|Copper|Smelter|Rare', case=False, na=False),
            "dist_uranium_km":   facilities['facility_type_standard'].str.contains(r'Uranium', case=False, na=False),
            "dist_thermal_km":   facilities['facility_type_standard'].str.contains(r'Coal|Geothermal|Desalination|Smelter', case=False, na=False),
            "dist_desalination_km": facilities['facility_type_standard'].str.contains(r'Desalination', case=False, na=False)
        }

        for col_name, mask in views.items():
            subset = facilities[mask].dropna(subset=['lat', 'lon'])
            if subset.empty:
                valid_samples[col_name] = np.nan
                continue
            f_coords = np.radians(subset[['lat', 'lon']].values)
            tree = BallTree(f_coords, metric='haversine')
            dists, _ = tree.query(s_coords, k=1)
            valid_samples[col_name] = dists.flatten() * EARTH_RADIUS_KM

        # 2. General Matching (Nearest Neighbor against ALL facilities)
        clean_facilities = facilities.dropna(subset=['lat', 'lon']).reset_index(drop=True)
        f_coords_all = np.radians(clean_facilities[['lat', 'lon']].values)
        tree_all = BallTree(f_coords_all, metric='haversine')
        dists_all, idxs_all = tree_all.query(s_coords, k=1)
        
        # Distances
        valid_samples['facility_distance_km'] = dists_all.flatten() * EARTH_RADIUS_KM
        
        # 3. Direct Metadata Injection (NO MERGE)
        nearest_facs = clean_facilities.iloc[idxs_all.flatten()]
        
        # Assign columns directly (Vectorized & Safe)
        valid_samples['facility'] = nearest_facs['facility'].values
        valid_samples['facility_type'] = nearest_facs['facility_type'].values
        valid_samples['facility_country'] = nearest_facs['country'].values
        
        # Logic for boolean flags
        nearest_is_nuclear = nearest_facs['is_nuclear'].values
        limit = self.config.nfc_facilities.max_distance_km
        
        is_industry = valid_samples['facility_distance_km'] <= limit
        valid_samples['industry_match'] = is_industry
        valid_samples['facility_match'] = is_industry & nearest_is_nuclear
        valid_samples['analog_match'] = is_industry & (~nearest_is_nuclear)

        # 4. Reassemble
        final_df = pd.concat([valid_samples, samples[~valid_mask]], axis=0)
        
        # Restore the original index 
        return final_df.set_index("original_index")

    def _match_facilities_with_locations(
        self, 
        facilities: pd.DataFrame, 
        samples: pd.DataFrame
    ) -> pd.DataFrame:
        return self._multi_view_matching(facilities, samples)

    async def get_nearby_samples(self, fetcher=None) -> pd.DataFrame:
        """Fetch samples near facilities, passing through the singleton fetcher."""
        if self.nfc_facilities_df.empty: 
            return pd.DataFrame()
            
        if self.use_local and self.nearby_samples_path.exists():
            try: 
                # Load from disk if we've already done this work
                return pd.read_csv(self.nearby_samples_path, sep='\t')
            except: 
                pass

        # PASS FETCHER HERE:
        self.nearby_samples_df = await self._fetch_and_process_nearby_samples(
            self.nfc_facilities_df, 
            fetcher=fetcher
        )
        
        if not self.nearby_samples_df.empty:
            self.nearby_samples_df.to_csv(self.nearby_samples_path, sep='\t', index=False)
            self.plot_fetched_maps(self.nearby_samples_df)
            
        return self.nearby_samples_df

    @cache_rows_to_file(cache_filename="03_nearby_samples.pkl", id_column="facility")
    async def _fetch_and_process_nearby_samples(self, facilities_df: pd.DataFrame, fetcher=None) -> pd.DataFrame:
        """
        Global search for samples near facilities using the Singleton fetcher.
        Ensures File Descriptor reuse across the entire geographic sweep.
        """
        if facilities_df.empty: 
            return pd.DataFrame()
        
        # --- 1. CONFIGURATION ---
        match_radius = self.config.nfc_facilities.max_distance_km
        search_radius = max(match_radius * 3.0, 50.0) 
        MIN_SAMPLES_PER_STUDY = getattr(self.config.nfc_facilities, 'min_samples_per_study', 15)
        
        self.logger.info(f" 🚀 [DATA-CENTRIC MODE] Global sweep for {len(facilities_df)} facilities...")

        # 1. Imports
        from workflow_16s.api.ena.metadata.fetcher import ENAFetcher as ENAFetcher
        from workflow_16s.api.ena.utils import optimize_dataframe_operations, apply_filters_vectorized
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def null_context(obj):
            yield obj

        # --- 2. CONTEXT MANAGEMENT (The FD Fix) ---
        if fetcher is None:
            # Standalone fallback: create a new session
            cache_manager = EnaCacheManager(cache_dir=self.project_dir.cache / "ena_finder")
            context_manager = ENAFetcher(email=self.email, max_concurrent=10, cache_manager=cache_manager)
        else:
            # Singleton path: reuse the Upstream session and DON'T close it
            context_manager = null_context(fetcher)

        # --- 3. PHASE 1: GLOBAL GEOGRAPHIC SWEEP ---
        all_samples_dict = {} 
        
        if self._standalone: self.progress.start()
        
        async with context_manager as active_fetcher:
            sweep_task = self.progress.add_task("[cyan]Global Sweep (Finding unique samples)...", total=len(facilities_df))
            valid_facs = facilities_df.dropna(subset=['lat', 'lon'])
            
            for _, row in valid_facs.iterrows():
                try:
                    # Use the 'active_fetcher' which points to our singleton
                    samples_list = await active_fetcher.find_nearby_samples(row['lat'], row['lon'], search_radius)
                    for s in samples_list:
                        acc = s.get('accession')
                        if acc and acc not in all_samples_dict:
                            all_samples_dict[acc] = s
                except Exception as e:
                    self.logger.debug(f"Error finding samples near {row['facility']}: {e}")
                self.progress.update(sweep_task, advance=1)
            
            unique_samples = list(all_samples_dict.keys())
            self.logger.info(f" 🌍 Global sweep complete. Found {len(unique_samples)} UNIQUE samples.")
            
            if not unique_samples: 
                if self._standalone: self.progress.stop()
                return pd.DataFrame()
            
            # --- 4. PHASE 2: MASTER METADATA FETCH (Using Singleton) ---
            self.logger.info(" 📦 Initiating Master Metadata Fetch...")
            
            # Fetch Experiments
            experiments = await active_fetcher.fetch_ena_data_in_batches(
                "read_experiment", "sample_accession", unique_samples, 
                chunk_size=100, progress_obj=self.progress, with_progress_bar=not self._standalone
            )
            
            experiment_accs = list(set([e.get('experiment_accession') for e in experiments if e.get('experiment_accession')]))
            
            # Fetch Runs
            runs = await active_fetcher.fetch_ena_data_in_batches(
                "read_run", "experiment_accession", experiment_accs, 
                chunk_size=100, progress_obj=self.progress, with_progress_bar=not self._standalone
            )
            
            # Fetch Studies
            study_accs = list(set([e.get('study_accession') for e in experiments if e.get('study_accession')]))
            studies = await active_fetcher.fetch_ena_data_in_batches(
                "study", "study_accession", study_accs, 
                chunk_size=100, progress_obj=self.progress, with_progress_bar=not self._standalone
            )
            
            # Fetch BioSamples
            biosamples = await active_fetcher.fetch_biosamples_in_batches(
                unique_samples, progress_obj=self.progress, with_progress_bar=not self._standalone
            )

        if self._standalone: self.progress.stop()

        # 4. PHASE 3: MASTER MERGE & FILTER
        self.logger.info(" ⚙️ Performing Master Merge and Vectorized Filtering (CPU Phase)...")
        raw_samples_list = list(all_samples_dict.values())
        
        merged_df = optimize_dataframe_operations(
            samples=raw_samples_list,
            experiments=experiments,
            runs=runs,
            studies=studies,
            biosamples_info=biosamples
        )
        
        filtered_df = apply_filters_vectorized(merged_df, amplicon=True)
        self.logger.info(f" 🦠 Pre-Taxonomy Survivors: {len(filtered_df)}")
        
        if filtered_df.empty: return pd.DataFrame()

        # 5. CLEAN & RESOLVE TAXONOMY (Massive speedup: only runs on survivors!)
        self.logger.info(" 🧬 Resolving Taxonomy and cleaning dataframe...")
        cleaned_df = await clean_df(filtered_df, self.config)

        # 6. GEOMETRIC MAPPING (Assign samples to nearest facility)
        self.logger.info(" 🗺️ Mapping survivors back to their respective geographic facilities...")
        matched_df = self._match_facilities_with_locations(facilities_df, cleaned_df)
        
        if matched_df.empty or 'study_accession' not in matched_df.columns:
            return matched_df

        # 7. STUDY FILTERING
        self.logger.info(f" 📄 Final Protocol: Mixed (True/False) AND >= {MIN_SAMPLES_PER_STUDY} samples per study...")
        
        def filter_studies(group):
            has_match = (group['facility_match'] == True).any()
            has_control = (group['facility_match'] == False).any()
            is_large_enough = len(group) >= MIN_SAMPLES_PER_STUDY
            return has_match and has_control and is_large_enough

        studies_to_keep = matched_df.groupby('study_accession').filter(filter_studies)
        
        n_orig = matched_df['study_accession'].nunique()
        n_kept = studies_to_keep['study_accession'].nunique()
        self.logger.info(f" 📋 Final Yield:\n"
                    f"    - Original Studies: {n_orig}\n"
                    f"    - Valid Studies:    {n_kept}\n"
                    f"    - Dropped:          {n_orig - n_kept}")
        
        return studies_to_keep
    
    def plot_fetched_maps(self, fetched_df):
        if fetched_df is None or fetched_df.empty: 
            return
        try:
            plot_all_facilities_and_fetched_samples_geo(
                fetched_df, 
                self.nfc_facilities_df, 
                "study_accession",
                self.output_dir / "fetched_samples_map", 
                False
            )
        except Exception as e: 
            self.logger.warning(f" ⚠️ Map plotting failed: {e}")

    async def get_nfc_project_accessions(self, fetcher=None, **kwargs) -> list[str]:
        """
        Retrieves unique study accessions near facilities. 
        Passes the singleton fetcher through to prevent FD exhaustion.
        """
        cache_path = self.cache_dir / "nfc_project_accessions.pkl"
        
        if self.use_local and cache_path.exists():
            try: 
                with open(cache_path, 'rb') as f: 
                    return pickle.load(f)
            except: 
                pass
            
        # PASS FETCHER HERE: This triggers the actual network sweep
        await self.get_nearby_samples(fetcher=fetcher)
        
        if self.nearby_samples_df.empty or 'study_accession' not in self.nearby_samples_df.columns: 
            return []
            
        accs = self.nearby_samples_df['study_accession'].dropna().unique().tolist()
        
        try: 
            with open(cache_path, 'wb') as f: 
                pickle.dump(accs, f)
        except: 
            pass
            
        return accs

    async def get_contaminated_sample_count_async(self) -> int:
        projs = await self.get_nfc_project_accessions()
        if not projs: return 0
        manager = EnaCacheManager(cache_dir=self.project_dir.cache / "ena_metadata")
        total = 0
        for pid in projs:
            try: 
                total += await get_n_samples_by_bioproject_async(
                    bioproject_accession=pid, 
                    email=self.email, 
                    cache_manager=manager,
                    progress_obj=self.progress # PASS PLUMBING
                )
            except: pass
        return total

    async def get_non_contaminated_project_accessions_async(self, fetcher=None) -> list[str]:
        """
        Grid-crawling engine for control site discovery.
        Passes the Singleton fetcher through to ensure session/socket reuse.
        """
        cache_path = self.cache_dir / "non_contaminated_project_accessions.pkl"
        
        if self.use_local and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f: 
                    return pickle.load(f)
            except:
                pass
                
        if self.nfc_facilities_df.empty: 
            await self.nfc_facilities()
            
        if self.nfc_facilities_df.empty: 
            return []
        
        self.logger.info(" 🌐 Creating global grid for control sites (buffer > 50km from facilities)...")
        
        # 1. Coordinate Math (BallTree for exclusion zone)
        f_coords = np.radians(self.nfc_facilities_df.dropna(subset=['lat','lon'])[['lat','lon']])
        tree = BallTree(f_coords, metric='haversine')
        
        lats = np.linspace(-90, 90, 37)
        lons = np.linspace(-180, 180, 73)
        g_lats, g_lons = np.meshgrid(lats, lons)
        grid = pd.DataFrame({'lat': g_lats.flatten(), 'lon': g_lons.flatten()})
        
        dists, _ = tree.query(np.radians(grid[['lat','lon']]), k=1)
        safe_grid = grid[dists.flatten() * 6371 > 50].copy() # Ensure > 50km away
        
        if safe_grid.empty: 
            return []
        
        # 2. Batch Logic
        BATCH_SIZE, SLEEP_TIME = 10, 2.0
        all_samples = []
        chunks = [safe_grid[i:i + BATCH_SIZE] for i in range(0, len(safe_grid), BATCH_SIZE)]
        
        # 3. Execution (The Singleton Connection Pool)
        if self._standalone: self.progress.start()
        
        task = self.progress.add_task("[cyan]Crawling grid for control sites", total=len(chunks))
        
        from workflow_16s.api.ena.finder import run_searches_from_dataframe_async
        
        for i, chunk in enumerate(chunks):
            # THE FIX: Pass the 'fetcher' singleton from Upstream into the finder
            chunk_samples = await run_searches_from_dataframe_async(
                input_df=chunk, 
                radius=75.0, 
                email=self.email, 
                amplicon=True, 
                fetcher=fetcher, # <--- CRITICAL PASSTHROUGH
                max_concurrent=5, 
                cache_dir=self.project_dir.cache / "ena_finder",
                progress_obj=self.progress 
            )
            
            if not chunk_samples.empty: 
                all_samples.append(chunk_samples)
                
            self.progress.update(task, advance=1)
            
            # Avoid hammering EBI even with shared sockets
            if i < len(chunks) - 1 and not chunk_samples.empty: 
                await asyncio.sleep(SLEEP_TIME)

        if self._standalone: self.progress.stop()
        else: self.progress.remove_task(task)

        # 4. Cleanup & Caching
        if not all_samples: 
            return []
            
        final_samples = pd.concat(all_samples, ignore_index=True)
        accs = final_samples['study_accession'].dropna().unique().tolist() if 'study_accession' in final_samples.columns else []
        
        with open(cache_path, 'wb') as f: 
            pickle.dump(accs, f)
            
        return accs
    
    def plot_global_facilities_map(self, output_filename: str = "global_facilities_map.html"):
        """
        Generates an interactive Plotly map of ALL aggregated facilities.
        Colors points by 'facility_category' (Nuclear vs Analog) and shapes by 'facility_type'.
        """
        from workflow_16s.visualization.nuclear_fuel_cycle.main import plot_global_facilities_map as _plot_global_facilities_map
        return _plot_global_facilities_map(self.nfc_facilities_df, self.output_dir, output_filename)