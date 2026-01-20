# workflow_16s/api/ena/finder.py

# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import argparse
import logging
import sys
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Third Party Imports
import aiohttp
import pandas as pd

# Local Imports
from workflow_16s.utils.progress import get_progress_bar
from workflow_16s.utils.logger import get_logger

# Import from new refactored modules
from .cache import CacheManager
from .constants import ENA_API_URL
from .fetcher import OptimizedENAFetcher
from .utils import (
    optimize_dataframe_operations, 
    apply_filters_vectorized, 
    process_and_save_by_location
)

# ==================================================================================== #

logger = get_logger()

# ==================================================================================== #

async def find_nearby_samples_async(
    session: aiohttp.ClientSession, latitude: float, longitude: float, 
    radius: Union[int, float], progress=None, task_id=None,
    cache_manager: Optional[CacheManager] = None
) -> Optional[List[Dict]]:
    """Async version of find_nearby_samples with caching and progress tracking."""
    cache_key = None
    if cache_manager:
        cache_key = cache_manager.get_cache_key("nearby", latitude=latitude, 
                                                longitude=longitude, radius=radius)
        cached_data = await cache_manager.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Cache HIT for samples near ({latitude}, {longitude})")
            if progress and task_id is not None:
                progress.update(task_id, description=f"Found {len(cached_data)} cached samples at ({latitude}, {longitude})")
            return cached_data

    fields = ["accession", "scientific_name", "collection_date", "location", "description", "host"]
    query = f"geo_circ({latitude},{longitude},{radius})"
    params = {
        "result": "sample", "query": query, "fields": ",".join(fields),
        "format": "json", "limit": 0
    }
    
    try:
        logger.debug(f"Querying ENA for samples within {radius}km of ({latitude}, {longitude})")
        async with session.get(ENA_API_URL, params=params) as response:
            if response.status == 204: result = []
            else:
                response.raise_for_status()
                result = await response.json()
            
            if progress and task_id is not None:
                progress.update(task_id, description=f"Found {len(result)} samples at ({latitude}, {longitude})")
            
            if cache_manager and result and cache_key is not None:
                await cache_manager.set(cache_key, result)

            return result
            
    except Exception as e:
        logger.error(f"Error finding nearby samples: {e}")
        return None

async def _process_location_data(
    fetcher, latitude, longitude, radius, amplicon, progress, location_task_id
):
    """Helper function to process location data with existing fetcher."""
    if fetcher.session is None:
        logger.error("Fetcher session is not initialized.")
        return pd.DataFrame()

    samples = await find_nearby_samples_async(
        fetcher.session, latitude, longitude, radius, progress, location_task_id, 
        fetcher.cache_manager
    )

    if samples is None or not samples:
        logger.debug("No samples found at the specified location.")
        return pd.DataFrame()

    sample_accessions = [s['accession'] for s in samples if 'accession' in s]
    logger.debug(f"Found {len(sample_accessions)} samples. Fetching related data...")

    try:
        experiments_task = fetcher.fetch_ena_batch(
            "read_experiment", "sample_accession", sample_accessions
        )
        biosamples_task = fetcher.fetch_biosamples_batch(sample_accessions)
        experiments, biosamples_info = await asyncio.gather(experiments_task, biosamples_task, 
                                                            return_exceptions=True)
        
        if isinstance(experiments, Exception):
            logger.error(f"Error fetching experiments: {experiments}"); experiments = []
        if isinstance(biosamples_info, Exception):
            logger.error(f"Error fetching biosamples: {biosamples_info}"); biosamples_info = {}

    except Exception as e:
        logger.error(f"Error in concurrent fetch operations: {e}")
        return pd.DataFrame()

    if not experiments: return pd.DataFrame()

    try:
        exp_accessions = [
            e['experiment_accession'] 
            for e in experiments if 'experiment_accession' in e
        ]
        study_accessions = list(set(e['study_accession'] 
                                    for e in experiments if 'study_accession' in e))

        runs_task = fetcher.fetch_ena_batch("read_run", "experiment_accession", exp_accessions)
        studies_task = fetcher.fetch_ena_batch("study", "study_accession", study_accessions)
        runs, studies = await asyncio.gather(runs_task, studies_task, return_exceptions=True)

        if isinstance(runs, Exception): logger.error(f"Error fetching runs: {runs}"); runs = []
        if isinstance(studies, Exception): logger.error(f"Error fetching studies: {studies}"); studies = []

    except Exception as e:
        logger.error(f"Error fetching runs and studies: {e}"); runs, studies = [], []

    logger.debug("Processing data with optimized operations...")
    df = optimize_dataframe_operations(samples, experiments, runs, studies, biosamples_info)
    if df.empty: return df

    df = apply_filters_vectorized(df, amplicon)

    if not df.empty:
        exp_cols = [c for c in df.columns if c.startswith('experiment_')]
        if exp_cols:
            df['completeness_score'] = df[exp_cols].notna().sum(axis=1)
            df = df.sort_values('completeness_score', ascending=False).drop(columns=['completeness_score'])

    return df

async def get_ena_data_by_location_async(
    latitude: float, longitude: float, radius: Union[int, float] = 50,
    email: str = "macgregor@berkeley.edu", amplicon: bool = True, verbose: bool = False,
    max_concurrent: int = 10, progress=None, location_task_id=None, fetcher=None,
    cache_manager: Optional[CacheManager] = None
) -> pd.DataFrame:
    """Async version of get_ena_data_by_location with shared progress tracking."""
    if fetcher is None:
        async with OptimizedENAFetcher(
            email, max_concurrent, progress=progress, cache_manager=cache_manager
        ) as new_fetcher:
            return await _process_location_data(
                new_fetcher, latitude, longitude, radius, amplicon, progress, 
                location_task_id
            )
    else:
        return await _process_location_data(
            fetcher, latitude, longitude, radius, amplicon, progress, 
            location_task_id
        )

async def run_searches_from_dataframe_async(
    input_df: pd.DataFrame, radius: Union[int, float], email: str,
    amplicon: bool, verbose: bool = True, max_concurrent: int = 5,
    cache_dir: Optional[Union[str, Path]] = ".finder_cache/"
) -> pd.DataFrame:
    """Async version with concurrent location processing and shared session management."""

    rename_map = {}
    if 'latitude' in input_df.columns and 'lat' not in input_df.columns:
        rename_map['latitude'] = 'lat'
    if 'longitude' in input_df.columns and 'lon' not in input_df.columns:
        rename_map['longitude'] = 'lon'
    if rename_map:
        logger.info(f"Standardizing location columns: {rename_map}")
        input_df = input_df.rename(columns=rename_map)

    if 'lat' not in input_df.columns or 'lon' not in input_df.columns:
        logger.error("Input DataFrame must contain 'lat' and 'lon' columns.")
        return pd.DataFrame()
    
    cache_manager = CacheManager(Path(cache_dir)) if cache_dir else None

    unique_coords = input_df[['lat', 'lon']].drop_duplicates().dropna()
    logger.debug(f"Processing {len(unique_coords)} unique coordinate pairs")

    with get_progress_bar() as progress:
        async with OptimizedENAFetcher(email, max_concurrent * 2, progress=progress, cache_manager=cache_manager) as fetcher:
            main_task = progress.add_task("Processing locations", total=len(unique_coords))
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_location(row):
                async with semaphore:
                    lat, lon = row['lat'], row['lon']
                    if verbose: progress.update(main_task, description=f"Processing ({lat:.3f}, {lon:.3f})")
                    
                    try:
                        result_df = await get_ena_data_by_location_async(
                            lat, lon, radius, email, amplicon, verbose, 10, progress,
                            main_task, fetcher
                        )
                        if not result_df.empty:
                            result_df['query_lat'] = lat
                            result_df['query_lon'] = lon
                            if verbose: progress.update(main_task, description=f"Completed ({lat:.3f}, {lon:.3f}) - {len(result_df)} results")
                            return result_df
                        else:
                            if verbose: progress.update(main_task, description=f"Completed ({lat:.3f}, {lon:.3f}) - no results")
                            return None
                    except Exception as e:
                        logger.error(f"Error processing location ({lat}, {lon}): {e}")
                        return None
                    finally: progress.update(main_task, advance=1)

            tasks = [process_location(row) for _, row in unique_coords.iterrows()]
            results = await asyncio.gather(*tasks, return_exceptions=True)

    all_results_dfs = []
    for result in results:
        if isinstance(result, pd.DataFrame) and not result.empty:
            all_results_dfs.append(result)
        elif isinstance(result, Exception):
            logger.error(f"Location processing failed: {result}")

    if not all_results_dfs: return pd.DataFrame()

    return pd.concat(all_results_dfs, ignore_index=True)

# ==================================================================================== #
#                          SYNCHRONOUS WRAPPERS FOR CLI
# ==================================================================================== #

def get_ena_data_by_location(*args, **kwargs):
    return asyncio.run(get_ena_data_by_location_async(*args, **kwargs))

def run_searches_from_dataframe(*args, **kwargs):
    return asyncio.run(run_searches_from_dataframe_async(*args, **kwargs))

# ==================================================================================== #
#                               COMMAND-LINE INTERFACE
# ==================================================================================== #

if __name__ == "__main__":
    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "workflow_16s_cache"

    parser = argparse.ArgumentParser(
        description="Optimized batch ENA sample finder with async processing and caching.",
        epilog="Example: python ena_finder_optimized.py locations.csv --radius 25 --email user@domain.com"
    )
    parser.add_argument("input_file", type=Path, help="Path to CSV file with 'latitude' and 'longitude' columns.")
    parser.add_argument("--radius", type=int, default=50, help="Search radius in kilometers.")
    parser.add_argument("--email", type=str, required=True, help="Email address for API identification.")
    parser.add_argument("--no-host", action="store_true", help="Filter out host-associated samples.")
    parser.add_argument("--amplicon", action="store_true", help="Filter for amplicon sequencing data.")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Maximum concurrent requests.")
    parser.add_argument("--output-dir", type=Path, default="results", help="Output directory for results.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help=f"Directory to store cache files. Defaults to: {DEFAULT_CACHE_DIR}")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching, ignoring the default and any specified --cache-dir.")
    parser.add_argument("--clean-cache", action="store_true", help="Clean all expired files from the cache and exit.")
    args = parser.parse_args()
    
    if args.clean_cache:
        if args.no_cache:
            print("Error: Cannot use --clean-cache and --no-cache together.", file=sys.stderr)
            sys.exit(1)
        print(f"Cleaning cache at: {args.cache_dir}")
        cache_manager = CacheManager(args.cache_dir)
        cache_manager.clear_expired()
        sys.exit(0)

    if not args.input_file.exists() or not args.input_file.is_file():
        print(f"Error: The file '{args.input_file}' was not found or is not a file.", file=sys.stderr)
        sys.exit(1)

    try:
        locations_df = pd.read_csv(args.input_file)
    except Exception as e:
        print(f"Error reading the input file '{args.input_file}': {e}", file=sys.stderr)
        sys.exit(1)

    cache_directory_to_use = None
    if not args.no_cache: cache_directory_to_use = args.cache_dir

    combined_df = asyncio.run(run_searches_from_dataframe_async(
        locations_df, args.radius, args.email, args.amplicon, True, 
        args.max_concurrent, cache_directory_to_use
    ))

    if combined_df.empty:
        print("\nNo data found for any of the provided locations.")
        sys.exit(0)

    process_and_save_by_location(combined_df, args.output_dir)
    print("\nOptimized batch processing complete.")