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
from typing import Callable, Optional, Set, Tuple, Union

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
from workflow_16s.api.ena.finder import run_searches_from_dataframe_async
from workflow_16s.api.ena.metadata_api import get_n_samples_by_bioproject_async
from workflow_16s.api.ena.metadata.cache import CacheManager as EnaCacheManager
from workflow_16s.config_schema import AppConfig
from workflow_16s.utils.dir_utils import Project
from workflow_16s.utils.metadata_utils import process_metadata, standardize_lat_lon_columns
from workflow_16s.utils.progress import get_progress_bar
from workflow_16s.visualization.nfc import (
    plot_all_facilities_and_fetched_samples_geo, plot_all_facilities_and_samples_geo,
    plot_facility_with_samples_geo
)

# --- INTEGRATED SOURCES ---
from workflow_16s.api.nuclear_fuel_cycle import (
    _dnfsb, _gem, _iaea, _mindat, 
    _nrc, _jrc, _wna, _wikidata,
    _analogs
)

# ==================================================================================== #

logger = logging.getLogger("workflow_16s")

# ==================================================================================== #
# [Caching Decorators]
# ==================================================================================== #

def cache_to_file(cache_filename: str) -> Callable:
    """Decorator to cache the entire result of a method using pickle."""
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

# ==================================================================================== #
# [Visualization Helpers]
# ==================================================================================== #

def plot_facility_match_pie_chart(df: pd.DataFrame, output_dir: Path,
                                  file_stem: str = "facility_match_distribution"):
    if 'facility_match' in df.columns:
        match_counts = df['facility_match'].astype(str).value_counts()
        fig = px.pie(names=match_counts.index, values=match_counts.values,
                     title='Distribution of Samples Matching a Nearby Facility',
                     hole=0.3,
                     color_discrete_map={'True': '#1f77b4', 'False': '#d62728'})
        fig.update_traces(textinfo='percent+label', pull=[0.05, 0])
        fig.update_layout(legend_title_text='Facility Match Status')
        output_path = output_dir / file_stem
        fig.write_html(f"{output_path}.html")
        fig.write_image(f"{output_path}.png", scale=2)

def sph2cart(lats, lons, R=6371) -> np.ndarray:
    φ = np.radians(lats.astype(float))
    λ = np.radians(lons.astype(float))
    x = R * np.cos(φ) * np.cos(λ)
    y = R * np.cos(φ) * np.sin(λ)
    z = R * np.sin(φ)
    return np.column_stack((x, y, z))

# ==================================================================================== #
# [Data Cleaning]
# ==================================================================================== #

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
        existing_cols = [f"{p}{base_name}" for p in priority_prefixes if f"{p}{base_name}" in df_copy.columns]
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

    for col in lat_cols: cleaned_df['lat'] = cleaned_df['lat'].combine_first(pd.to_numeric(cleaned_df[col], errors='coerce'))
    for col in lon_cols: cleaned_df['lon'] = cleaned_df['lon'].combine_first(pd.to_numeric(cleaned_df[col], errors='coerce'))

    cleaned_df.dropna(subset=['#sampleid', 'lat', 'lon'], inplace=True)
    cleaned_df = cleaned_df[
        (cleaned_df['lat'] >= -90) & (cleaned_df['lat'] <= 90) &
        (cleaned_df['lon'] >= -180) & (cleaned_df['lon'] <= 180)
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
    df['facility_match'] = True
    return df

# ==================================================================================== #
# [Main Handler Class]
# ==================================================================================== #

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

    def __init__(self, config: AppConfig):
        self.config = config
        if not self.config.nfc_facilities.enabled: return

        self.verbose = self.config.verbose
        self.user_agent = config.web.user_agent or "workflow_16s/1.0"
        self.use_local = self.config.nfc_facilities.use_cache
        self.database_names = [db.lower() for db in self.config.nfc_facilities.databases]
        self.email = self.config.credentials.ena_email

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

        self.nfc_facilities_df = pd.DataFrame()
        self.nearby_samples_df = pd.DataFrame()

    def log(self, msg):
        return (lambda msg: logger.debug(msg)) if self.verbose else (lambda *_: None)

    async def nfc_facilities(self) -> pd.DataFrame:
        """Main Pipeline: Fetch -> Geocode -> De-duplicate -> Enrich."""
        logger.info(f"Starting Unified Facility Pipeline...")
        df = self._get_data()
        df = await self._geocode(df)
        df = self._deduplicate_facilities(df)
        df = self._enrich_metadata(df)
        self.nfc_facilities_df = standardize_lat_lon_columns(df)
        
        if self.project_dir:
            self.nfc_facilities_df.to_csv(self.facilities_path, sep='\t', index=False)
            logger.info(f"Saved master facility database ({len(self.nfc_facilities_df)} sites) to {self.facilities_path}")

        return self.nfc_facilities_df

    def annotate_samples(self, samples_df: pd.DataFrame) -> pd.DataFrame:
        """Updates sample metadata with category-specific proximities."""
        if self.nfc_facilities_df.empty:
            logger.warning("Facilities not loaded. Run nfc_facilities() first.")
            return samples_df
        logger.info(f"Annotating {len(samples_df)} samples with Multi-View proximity features...")
        return self._multi_view_matching(self.nfc_facilities_df, samples_df)

    def build_atlas(self, samples_df: pd.DataFrame, data_dir: Union[str, Path]) -> Optional[ad.AnnData]:
        """
        Loads biological data (.h5ad) for matching samples, merges them, 
        injects enriched metadata, and SAVES the result to disk (Caching).
        """
        if self.use_local and self.atlas_path.exists():
            logger.info(f"Loading cached NFC Atlas from {self.atlas_path}")
            return ad.read_h5ad(self.atlas_path)

        data_dir = Path(data_dir)
        if not data_dir.exists():
            logger.error(f"Data directory {data_dir} does not exist. Cannot build atlas.")
            return None

        # Annotate samples first to ensure we have categories
        annotated_meta = self.annotate_samples(samples_df)
        valid_ids = annotated_meta.index.astype(str).tolist()
        
        adatas = []
        logger.info(f"Building Atlas: Scanning {data_dir} for {len(valid_ids)} samples...")
        
        found_count = 0
        with get_progress_bar() as p:
            task = p.add_task("Loading samples...", total=len(valid_ids))
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
                        logger.warning(f"Failed to load {fpath}: {e}")
                p.update(task, advance=1)

        if not adatas:
            logger.warning("No matching .h5ad files found. Atlas not created.")
            return None

        logger.info(f"Concatenating {found_count} samples into master Atlas...")
        try:
            master_adata = ad.concat(adatas, join="outer", label="batch", keys=[str(i) for i in range(len(adatas))])
            master_adata.obs = master_adata.obs.join(annotated_meta, rsuffix="_meta", how="left")
            
            master_adata.write(self.atlas_path)
            logger.info(f"✅ Success! NFC Atlas saved to {self.atlas_path}")
            return master_adata
            
        except Exception as e:
            logger.error(f"Atlas concatenation failed: {e}")
            return None

    @cache_to_file(cache_filename="01_raw_facilities.pkl")
    def _get_data(self) -> pd.DataFrame:
        logger.info("Aggregating ALL facilities (Nuclear + Analog)...")
        loaders = {
            "dnfsb": lambda: _dnfsb.load_facilities(self.config),
            "mindat": lambda: _mindat.world_uranium_mines(self.config)[self.mindat_columns_to_keep],
            "gem": lambda: _gem.GNPT(self.output_dir).load(),
            "nfcis": lambda: _iaea.NFCFDB(self.output_dir).load(),
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
                        if 'data_source' not in df.columns: 
                            df['data_source'] = name.upper()
                        else: 
                            df['data_source'] = df['data_source'].str.upper()
                        database_dfs.append(df)
                except: 
                    pass
        if not database_dfs: 
            return pd.DataFrame()
        combined = pd.concat(database_dfs, axis=0, ignore_index=True)
        if 'is_nuclear' not in combined.columns: 
            combined['is_nuclear'] = True
        combined['is_nuclear'] = combined['is_nuclear'].fillna(True)
        combined['facility_category'] = np.where(combined['is_nuclear'], 'Nuclear Fuel Cycle', 'Contamination Analog')
        obj_cols = combined.select_dtypes(include="object").columns
        combined[obj_cols] = combined[obj_cols].apply(lambda x: x.astype(str).str.replace(r"\s+", " ", regex=True).str.strip())
        combined["country"] = combined["country"].replace({"USA": "United States of America", "UK": "United Kingdom"})
        return combined

    def _deduplicate_facilities(self, df: pd.DataFrame, eps_km: float = 2.0) -> pd.DataFrame:
        logger.info(f"De-duplicating facilities within {eps_km} km...")
        df = df.copy()
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        valid_coords = df.dropna(subset=['lat', 'lon'])
        if valid_coords.empty: return df
        df['priority_score'] = df['data_source'].str.upper().map(self.SOURCE_PRIORITY).fillna(99)
        coords = np.radians(valid_coords[['lat', 'lon']].values)
        db = DBSCAN(eps=eps_km/6371.0088, min_samples=1, metric='haversine', algorithm='ball_tree').fit(coords)
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
        logger.info("Enriching metadata...")
        type_map = {
            r'(?i)pressuri[sz]ed': 'PWR', r'(?i)boiling': 'BWR', r'(?i)candu|heavy': 'PHWR',
            r'(?i)mine|open pit': 'Uranium Mine', r'(?i)mill': 'Uranium Mill',
            r'(?i)salt': 'Salt Mine', r'(?i)fertilizer|nitrate': 'Fertilizer Plant',
            r'(?i)potash': 'Potash Mine', r'(?i)superfund': 'Superfund Site', 
            r'(?i)acid': 'Acid Plant', r'(?i)coal': 'Coal Power Plant',
            r'(?i)gold': 'Gold Mine', r'(?i)copper': 'Copper Mine',
            r'(?i)desalination': 'Desalination Plant', r'(?i)geothermal': 'Geothermal Plant',
            r'(?i)aluminium|smelter': 'Smelter'
        }
        def standardize(val):
            for p, c in type_map.items():
                if re.search(p, str(val)): return c
            return str(val)
        if 'facility_type' in df.columns:
            df['facility_type_standard'] = df['facility_type'].apply(standardize)
        return df

    async def _fetch_coord(self, session, query: str) -> Tuple[str, Optional[Tuple[float, float]]]:
        url = "https://nominatim.openstreetmap.org/search"
        headers = {"User-Agent": self.user_agent}
        try:
            async with session.get(url, params={'q': query, 'format': 'json', 'limit': 1}, headers=headers) as r:
                if r.status == 200:
                    data = await r.json()
                    if data: 
                        return query, (float(data[0]['lat']), float(data[0]['lon']))
        except: 
            pass
        await asyncio.sleep(1.0)
        return query, (None, None)

    async def _geocode_async(self, queries: list) -> dict:
        coords = {}
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_coord(session, q) for q in queries]
            with get_progress_bar() as progress:
                tid = progress.add_task("Geocoding...", total=len(tasks))
                for f in asyncio.as_completed(tasks):
                    q, res = await f
                    if res != (None, None): 
                        coords[q] = res
                    progress.update(tid, advance=1)
        return coords

    @cache_rows_to_file(cache_filename="02_geocoded_facilities.pkl", id_column="facility")
    async def _geocode(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'lat' not in df.columns: df['lat'] = np.nan
        if 'lon' not in df.columns: df['lon'] = np.nan
        missing = df[df['lat'].isna() | df['lon'].isna()].copy()
        if missing.empty: return df
        unique_q = (missing['facility'].fillna('') + ", " + missing['country'].fillna('')).unique().tolist()
        logger.info(f"Geocoding {len(unique_q)} facilities...")
        coords = await self._geocode_async(unique_q)
        df.loc[df['lat'].isna(), 'lat'] = (df['facility'] + ", " + df['country']).map(lambda x: coords.get(x, (np.nan, np.nan))[0])
        df.loc[df['lon'].isna(), 'lon'] = (df['facility'] + ", " + df['country']).map(lambda x: coords.get(x, (np.nan, np.nan))[1])
        return df

    def _multi_view_matching(self, facilities: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
        EARTH_RADIUS_KM = 6371
        samples = samples.copy()
        if "original_index" not in samples.columns: samples["original_index"] = samples.index
        if 'lat' not in samples.columns: samples.rename(columns={'latitude_deg': 'lat'}, inplace=True)
        if 'lon' not in samples.columns: samples.rename(columns={'longitude_deg': 'lon'}, inplace=True)
        valid_mask = samples[['lat', 'lon']].notnull().all(axis=1)
        valid_samples = samples[valid_mask].copy()
        if valid_samples.empty: return samples
        s_coords = np.radians(valid_samples[['lat', 'lon']].values)

        views = {
            "dist_nuclear_km":  facilities['is_nuclear'] == True,
            "dist_analog_km":   facilities['is_nuclear'] == False,
            "dist_salts_km":    facilities['facility_type_standard'].str.contains(r'Salt|Potash', case=False, na=False),
            "dist_nitrates_km": facilities['facility_type_standard'].str.contains(r'Fertilizer|Acid', case=False, na=False),
            "dist_metals_km":   facilities['facility_type_standard'].str.contains(r'Gold|Copper|Smelter|Rare', case=False, na=False),
            "dist_uranium_km":  facilities['facility_type_standard'].str.contains(r'Uranium', case=False, na=False),
            "dist_thermal_km":  facilities['facility_type_standard'].str.contains(r'Coal|Geothermal|Desalination|Smelter', case=False, na=False),
            "dist_desalination_km": facilities['facility_type_standard'].str.contains(r'Desalination', case=False, na=False)
        }

        for col_name, mask in views.items():
            subset = facilities[mask]
            if subset.empty:
                valid_samples[col_name] = np.nan
                continue
            f_coords = np.radians(subset[['lat', 'lon']].values)
            tree = BallTree(f_coords, metric='haversine')
            dists, _ = tree.query(s_coords, k=1)
            valid_samples[col_name] = dists.flatten() * EARTH_RADIUS_KM

        f_coords_all = np.radians(facilities[['lat', 'lon']].values)
        tree_all = BallTree(f_coords_all, metric='haversine')
        dists_all, idxs_all = tree_all.query(s_coords, k=1)
        valid_samples['facility_distance_km'] = dists_all.flatten() * EARTH_RADIUS_KM
        valid_samples['facility'] = facilities.iloc[idxs_all.flatten()]['facility'].values
        valid_samples['facility_match'] = valid_samples['facility_distance_km'] <= self.config.nfc_facilities.max_distance_km
        
        merged = pd.merge(valid_samples, facilities.add_suffix('_facility'), left_on='facility', right_on='facility_facility', how='left')
        final_df = pd.concat([merged, samples[~valid_mask]], ignore_index=True)
        return final_df.set_index("original_index")

    def _match_facilities_with_locations(self, facilities: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
        return self._multi_view_matching(facilities, samples)

    async def get_nearby_samples(self) -> pd.DataFrame:
        if self.nfc_facilities_df.empty: 
            return pd.DataFrame()
        if self.use_local and self.nearby_samples_path.exists():
            try: 
                return pd.read_csv(self.nearby_samples_path, sep='\t')
            except: 
                pass
        self.nearby_samples_df = await self._fetch_and_process_nearby_samples(self.nfc_facilities_df)
        if not self.nearby_samples_df.empty:
            self.nearby_samples_df.to_csv(self.nearby_samples_path, sep='\t', index=False)
            self.plot_fetched_maps(None, self.nearby_samples_df)
        return self.nearby_samples_df

    @cache_rows_to_file(cache_filename="03_nearby_samples.pkl", id_column="facility")
    async def _fetch_and_process_nearby_samples(self, facilities_df: pd.DataFrame) -> pd.DataFrame:
        if facilities_df.empty: 
            return pd.DataFrame()
        logger.info(f"Searching ENA for samples near {len(facilities_df)} facilities...")
        samples = await run_searches_from_dataframe_async(
            input_df=facilities_df, radius=self.config.nfc_facilities.max_distance_km,
            email=self.email, amplicon=True, max_concurrent=5, cache_dir=self.project_dir.cache / "ena_finder"
        )
        if samples.empty: 
            return pd.DataFrame()
        samples = await clean_df(samples, self.config)
        return self._match_facilities_with_locations(facilities_df, samples)

    def plot_fetched_maps(self, adata, fetched_df):
        if fetched_df is None or fetched_df.empty: 
            return
        try:
            plot_all_facilities_and_fetched_samples_geo(
                fetched_df, self.nfc_facilities_df, "study_accession",
                self.output_dir / "fetched_samples_map", False
            )
        except Exception as e: 
            logger.warning(f"Map plotting failed: {e}")

    async def get_nfc_project_accessions(self) -> list[str]:
        cache_path = self.cache_dir / "nfc_project_accessions.pkl"
        if self.use_local and cache_path.exists():
            try: 
                with open(cache_path, 'rb') as f: 
                    return pickle.load(f)
            except: 
                pass
        await self.get_nearby_samples()
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
                total += await get_n_samples_by_bioproject_async(pid, self.email, manager)
            except: 
                pass
        return total

    async def get_non_contaminated_project_accessions_async(self) -> list[str]:
        cache_path = self.cache_dir / "non_contaminated_project_accessions.pkl"
        if self.use_local and cache_path.exists():
            with open(cache_path, 'rb') as f: 
                return pickle.load(f)
        if self.nfc_facilities_df.empty: 
            await self.nfc_facilities()
        if self.nfc_facilities_df.empty: 
            return []
        logger.info("Generating global grid for control sites...")
        f_coords = np.radians(self.nfc_facilities_df.dropna(subset=['lat','lon'])[['lat','lon']])
        tree = BallTree(f_coords, metric='haversine')
        lats = np.linspace(-90, 90, 37)
        lons = np.linspace(-180, 180, 73)
        g_lats, g_lons = np.meshgrid(lats, lons)
        grid = pd.DataFrame({'lat': g_lats.flatten(), 'lon': g_lons.flatten()})
        dists, _ = tree.query(np.radians(grid[['lat','lon']]), k=1)
        safe_grid = grid[dists.flatten() * 6371 > 50].copy()
        if safe_grid.empty: 
            return []
        BATCH_SIZE, SLEEP_TIME = 10, 2.0
        all_samples = []
        chunks = [safe_grid[i:i + BATCH_SIZE] for i in range(0, len(safe_grid), BATCH_SIZE)]
        with get_progress_bar() as p:
            task = p.add_task("Crawling safe grid...", total=len(chunks))
            for i, chunk in enumerate(chunks):
                chunk_samples = await run_searches_from_dataframe_async(
                    chunk, 200.0, self.email, True, 5, self.project_dir.cache / "ena_finder"
                )
                if not chunk_samples.empty: 
                    all_samples.append(chunk_samples)
                p.update(task, advance=1)
                if i < len(chunks) - 1: 
                    await asyncio.sleep(SLEEP_TIME)

        if not all_samples: 
            return []
        final_samples = pd.concat(all_samples, ignore_index=True)
        accs = final_samples['study_accession'].dropna().unique().tolist() if 'study_accession' in final_samples.columns else []
        with open(cache_path, 'wb') as f: 
            pickle.dump(accs, f)
        return accs