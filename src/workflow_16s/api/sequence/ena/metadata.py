"""
ENA Environmental Sample Data Retrieval - Consolidated & Improved.

Orchestrates asynchronous fetching, caching, and processing of ENA data
based on geographic locations or BioProject.
"""

import argparse
import asyncio
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from workflow_16s.utils.progress import get_progress_bar
from workflow_16s.utils.logger import get_logger, with_logger

from .cache import SQLiteCacheManager as CacheManager 

from .constants import DEFAULT_EMAIL, DEFAULT_CACHE_DIR
from .utils import (
    process_and_structure_data,
    apply_filters,
    save_results_by_location
)

# HARD CAP TO PREVENT API HANGS ON MASSIVE HOTSPOTS
MAX_SAMPLES_PER_SWEEP = 1000

logger = get_logger("workflow_16s")

@with_logger
async def get_samples_by_location_async(
    lat: float, 
    lon: float, 
    radius: Union[int, float], 
    email: str = DEFAULT_EMAIL,
    max_concurrent: int = 50, 
    cache_manager: Optional[CacheManager] = None,
    fetcher: Optional[Any] = None,
    progress_obj: Any = None 
) -> pd.DataFrame:
    """Fetches and structures all sample data within a geographic radius with location-level caching."""
    from .fetcher import ENAFetcher
    
    location_key = f"loc_search_{round(lat, 2)}_{round(lon, 2)}_r{radius}"
    
    if cache_manager:
        cached_df = await cache_manager.get(location_key)
        if cached_df is not None and not cached_df.empty:
            logger.info(f" 🚀 Teleporting: Loaded results for ({lat}, {lon})")
            return cached_df

    async def _get_data(fetcher_instance: ENAFetcher) -> pd.DataFrame:
        logger.info(f"Finding samples within {radius}km of ({lat}, {lon})...")
        
        from .finder import find_nearby_samples_async
        samples = await find_nearby_samples_async(
            fetcher_instance.session, lat, lon, radius, 
            progress_obj, cache_manager=cache_manager
        )
        
        if samples is None or (pd.DataFrame(samples).empty if isinstance(samples, list) else samples.empty):
            return pd.DataFrame()
            logger.info(f"No samples found within {radius}km of ({lat}, {lon}).")
            if cache_manager:
                await cache_manager.set(location_key, pd.DataFrame())
            return pd.DataFrame()

        if len(samples) > MAX_SAMPLES_PER_SWEEP:
            logger.warning(f"⚠️ API BOTTLENECK AVOIDED: Found {len(samples)} samples. Capping at {MAX_SAMPLES_PER_SWEEP}.")
            samples = samples[:MAX_SAMPLES_PER_SWEEP]

        sample_accessions = [s['accession'] for s in samples if 'accession' in s]
        logger.info(f"Found {len(sample_accessions)} samples. Pulling deeper metadata...")

        runs_task = fetcher_instance.fetch_ena_data_in_batches(
            "read_run", "sample_accession", sample_accessions,
            with_progress_bar=False, progress_obj=progress_obj
        )
        biosamples_task = fetcher_instance.fetch_biosamples_batch(
            sample_accessions,
            with_progress_bar=False, progress_obj=progress_obj
        )
        
        tax_ids = list(set([s.get('tax_id') for s in samples if s.get('tax_id')]))
        taxonomy_task = fetcher_instance.fetch_taxonomies(
            tax_ids,
            with_progress_bar=False, progress_obj=progress_obj
        )

        runs_result, biosamples_result, taxonomies_result = await asyncio.gather(
            runs_task, biosamples_task, taxonomy_task, return_exceptions=True
        )

        if isinstance(runs_result, Exception): 
            logger.error(f"Error fetching runs for location ({lat},{lon}): {runs_result}")
            runs = []
        else: runs = runs_result
            
        if isinstance(biosamples_result, Exception): 
            logger.error(f"Error fetching biosamples for location ({lat},{lon}): {biosamples_result}")
            biosamples_info = {}
        else: biosamples_info = biosamples_result
            
        structured_data = process_and_structure_data(
            samples, 
            runs if isinstance(runs, list) else [], 
            biosamples_info if isinstance(biosamples_info, dict) else {}
        )

        if not structured_data.empty and cache_manager:
            await cache_manager.set(location_key, structured_data)

        return structured_data

    if fetcher:
        return await _get_data(fetcher)
    else:
        async with ENAFetcher(email, max_concurrent, cache_manager=cache_manager) as new_fetcher:
            return await _get_data(new_fetcher)

def get_samples_by_location(*args, **kwargs) -> pd.DataFrame:
    """Synchronous wrapper for get_samples_by_location_async."""
    return asyncio.run(get_samples_by_location_async(*args, **kwargs))

async def get_counts_bulk_async(
    accessions: List[str], 
    email: str, 
    max_concurrent: int = 15,
    chunk_size: int = 50,
    cache_manager: Optional[CacheManager] = None,
    progress_obj: Any = None
) -> Dict[str, int]:
    """Fetches sample counts for BioProjects using the SQLite batch engine."""
    from .fetcher import ENAFetcher
    async with ENAFetcher(email, max_concurrent, cache_manager=cache_manager) as fetcher:
        all_samples = await fetcher.fetch_ena_batch(
            "sample", "study_accession", accessions
        )
        if not all_samples:
            return {acc: 0 for acc in accessions}
        
        df = pd.DataFrame(all_samples)
        counts = df['study_accession'].value_counts().to_dict()
        return {acc: counts.get(acc, 0) for acc in accessions}

class ENAClient:
    """Consolidated ENA Client Discovery Wrapper."""
    def __init__(self, config, fetcher: Optional[Any] = None):
        self.config = config
        self.email = config.credentials.ena_email
        self.fetcher = fetcher 
        self.logger = get_logger("workflow_16s")
        self.cache_manager = fetcher.cache_manager if fetcher else None

    async def get_project_metadata(self, dataset_id: str) -> pd.DataFrame:
        """Fetches metadata for a BioProject, leveraging SQLite cache."""
        return await get_samples_by_bioproject_async(
            dataset_id, 
            email=self.email, 
            fetcher=self.fetcher,
            cache_manager=self.cache_manager
        )

    async def search_projects(self, query: str, limit: int = 1000) -> List[str]:
        """Discovery Engine: Search for studies by keyword."""
        params = {
            "result": "read_run",
            "query": query,
            "fields": "study_accession", 
            "format": "json",
            "limit": limit if limit > 0 else 1000
        }
        url = "https://www.ebi.ac.uk/ena/portal/api/search"
        
        try:
            if self.fetcher and self.fetcher.session:
                async with self.fetcher.session.get(url, params=params) as response:
                    if response.status == 204: return []
                    response.raise_for_status()
                    data = await response.json()
            else:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, params=params)
                    if resp.status_code == 204: return []
                    resp.raise_for_status()
                    data = resp.json()
            return sorted(list(set([item['study_accession'] for item in data if 'study_accession' in item])))
        except Exception as e:
            self.logger.error(f"ENA Search Failed: {e}")
            return []

@with_logger
async def get_n_samples_by_bioproject_async( 
    bioproject_accession: str, 
    email: str = DEFAULT_EMAIL, 
    max_concurrent: int = 15,
    cache_manager: Optional[CacheManager] = None, 
    fetcher: Optional[Any] = None,
    progress_obj: Any = None
) -> int:
    """Fetches only the count of samples for a given BioProject accession."""
    from .fetcher import ENAFetcher
    async def _get_count(fetcher_instance: ENAFetcher) -> int:
        samples = await fetcher_instance.fetch_ena_data_in_batches(
            result_type="sample",
            query_key="study_accession",
            accessions=[bioproject_accession],
            fields="accession",
            with_progress_bar=False,
            progress_obj=progress_obj 
        )
        return len(samples) if samples else 0

    if fetcher: return await _get_count(fetcher)
    else:
        async with ENAFetcher(email, max_concurrent, cache_manager) as new_fetcher:
            return await _get_count(new_fetcher)

@with_logger
async def get_samples_by_bioproject_async(
    bioproject_accession: str, 
    email: str = DEFAULT_EMAIL, 
    max_concurrent: int = 15,
    cache_manager: Optional[CacheManager] = None, 
    fetcher: Optional[Any] = None,
    progress_obj: Any = None
) -> pd.DataFrame:
    """Fetches and structures all sample, run, biosample, and taxonomy data for a BioProject with caching."""
    from .fetcher import ENAFetcher
    cache_key = f"project_meta_{bioproject_accession}"
    if cache_manager:
        cached_df = await cache_manager.get(cache_key)
        if cached_df is not None:
            logger.info(f" 📦 Loaded {bioproject_accession} metadata from cache.")
            return cached_df

    async def _get_data(fetcher_instance: ENAFetcher) -> pd.DataFrame:
        samples = await fetcher_instance.fetch_ena_data_in_batches(
            "sample", "study_accession", [bioproject_accession],
            fields="accession,tax_id", with_progress_bar=False, progress_obj=progress_obj
        )
        if samples is None or (pd.DataFrame(samples).empty if isinstance(samples, list) else samples.empty):
            return pd.DataFrame()
        if len(samples) > MAX_SAMPLES_PER_SWEEP:
            samples = samples[:MAX_SAMPLES_PER_SWEEP]
        
        sample_accessions = [s['accession'] for s in samples if 'accession' in s]
        runs_task = fetcher_instance.fetch_ena_data_in_batches("read_run", "sample_accession", sample_accessions)
        biosamples_task = fetcher_instance.fetch_biosamples_batch(sample_accessions)
        tax_ids = [s.get('tax_id') for s in samples if s.get('tax_id')] 
        taxonomy_task = fetcher_instance.fetch_taxonomies(tax_ids)

        runs, biosamples, taxonomies = await asyncio.gather(runs_task, biosamples_task, taxonomy_task)
        structured_data = process_and_structure_data(samples, runs or [], biosamples or {})

        if isinstance(taxonomies, dict) and not structured_data.empty:
            structured_data['taxonomy_lineage'] = structured_data['tax_id'].astype(str).map(taxonomies).fillna("N/A")

        if cache_manager and not structured_data.empty:
            await cache_manager.set(cache_key, structured_data)
        return structured_data

    if fetcher: return await _get_data(fetcher)
    else:
        async with ENAFetcher(email, max_concurrent, cache_manager) as new_fetcher:
            return await _get_data(new_fetcher)

def get_samples_by_bioproject(*args, **kwargs) -> pd.DataFrame:
    return asyncio.run(get_samples_by_bioproject_async(*args, **kwargs))

async def process_single_location(
    fetcher: Any, lat: float, lon: float, radius: Union[int, float],
    amplicon: bool, no_host: bool, progress_obj: Any = None
) -> pd.DataFrame:
    df = await get_samples_by_location_async(
        lat=lat, lon=lon, radius=radius, email=fetcher.email, 
        max_concurrent=fetcher.max_concurrent, cache_manager=fetcher.cache_manager,
        fetcher=fetcher, progress_obj=progress_obj
    )
    filtered_df = apply_filters(df, amplicon, no_host)
    if not filtered_df.empty:
        filtered_df['query_lat'], filtered_df['query_lon'] = lat, lon
    return filtered_df

@with_logger
async def run_location_searches_from_file(args: argparse.Namespace, progress_obj: Any = None) -> pd.DataFrame:
    from .fetcher import ENAFetcher
    input_df = pd.read_csv(args.input_file)
    input_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True, errors='ignore')
    unique_coords = input_df[['lat', 'lon']].dropna().drop_duplicates()
    cache_manager = CacheManager(args.cache_dir) if not args.no_cache else None
    p = progress_obj if progress_obj else get_progress_bar()
    
    async with ENAFetcher(args.email, args.max_concurrent, cache_manager=cache_manager) as fetcher:
        semaphore = asyncio.Semaphore(args.max_concurrent_locations)
        async def worker(lat, lon):
            async with semaphore:
                return await process_single_location(fetcher, lat, lon, args.radius, args.amplicon, args.no_host, p)
        results = await asyncio.gather(*[worker(r.lat, r.lon) for r in unique_coords.itertuples()])

    all_dfs = [df for df in results if not df.empty]
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

@with_logger
def main():
    parser = argparse.ArgumentParser(description="Optimized ENA sample finder.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--input-file", type=Path)
    mode_group.add_argument("--bioproject", type=str)
    parser.add_argument("--radius", type=int, default=50)
    parser.add_argument("--email", type=str, default=DEFAULT_EMAIL)
    parser.add_argument("--no-host", action="store_true")
    parser.add_argument("--amplicon", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=15)
    parser.add_argument("--max-concurrent-locations", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default="ena_metadata_results")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

if __name__ == "__main__":
    main()