# workflow_16s/api/ena/finder.py

import argparse
import asyncio
import hashlib
import math
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiohttp
import pandas as pd

from workflow_16s.utils.progress import get_progress_bar
from workflow_16s.utils.logger import get_logger

from .cache import SQLiteCacheManager as CacheManager 
from .constants import ENA_API_URL
from .fetcher import ENAFetcher
from .utils import (
    optimize_dataframe_operations, 
    apply_filters_vectorized, 
    process_and_save_by_location
)
logger = get_logger("workflow_16s")
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in kilometers."""
    # Ensure all inputs are floats (may come from strings in data/SQLite)
    try:
        lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
    except (ValueError, TypeError) as e:
        raise ValueError(f"haversine_distance requires numeric lat/lon, got: {type(lat1).__name__}, {type(lon1).__name__}, {type(lat2).__name__}, {type(lon2).__name__}") from e
    
    R = 6371  # Earth radius
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2)**2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2)
    return R * 2 * math.asin(math.sqrt(a))

@asynccontextmanager
async def null_context(obj):
    """Context manager that does nothing, used to share a singleton fetcher."""
    yield obj

# workflow_16s/api/ena/finder.py

import asyncio
import hashlib
import aiohttp
import pandas as pd
from typing import Optional, Union, Dict, List

from workflow_16s.utils.logger import get_logger
from .constants import ENA_API_URL

async def find_nearby_samples_async(
    session: aiohttp.ClientSession, 
    latitude: float, 
    longitude: float, 
    radius: Union[int, float], 
    progress=None, 
    task_id=None,
    cache_manager=None
) -> pd.DataFrame:
    """Async version of find_nearby_samples with SQLite caching and DF output."""
    logger = get_logger("workflow_16s")
    
    # 1. Cache Key Generation
    cache_key = None
    if cache_manager:
        key_raw = f"nearby_{round(latitude,3)}_{round(longitude,3)}_{radius}"
        cache_key = hashlib.sha256(key_raw.encode()).hexdigest()
        
        # 🟢 Clean await, no to_thread needed!
        cached_data = await cache_manager.get(cache_key)
        if cached_data is not None:
            return pd.DataFrame(cached_data) if isinstance(cached_data, list) else cached_data

    # 2. API Query Parameters
    query = f"geo_circ({latitude},{longitude},{radius})"
    params = {
        "result": "sample", 
        "query": query, 
        "fields": "accession,scientific_name,collection_date,location,description,host",
        "format": "json", 
        "limit": 0
    }
    
    try:
        async with session.get(ENA_API_URL, params=params) as response:
            if response.status == 204: return pd.DataFrame() 
            response.raise_for_status()
            result = await response.json()
            
            # 3. Cache the raw result
            if cache_manager and result and cache_key is not None:
                await cache_manager.set(cache_key, result)
            
            return pd.DataFrame(result)
            
    except Exception as e:
        logger.error(f"Error finding nearby samples at ({latitude}, {longitude}): {e}")
        return pd.DataFrame()
    
async def _process_location_data(fetcher, latitude, longitude, radius, amplicon, progress, location_task_id):
    """Internal logic to fetch and merge all metadata for a specific coordinate."""    
    samples = await find_nearby_samples_async(
        fetcher.session, latitude, longitude, radius, progress, location_task_id, fetcher.cache_manager
    )

    if not samples:
        return pd.DataFrame()

    sample_accessions = [s['accession'] for s in samples if 'accession' in s]
    
    # Concurrent fetch of Experiments and BioSamples
    experiments = await fetcher.fetch_ena_batch("read_experiment", "sample_accession", sample_accessions)
    biosamples_info = await fetcher.fetch_biosamples_batch(sample_accessions)

    if not experiments: return pd.DataFrame()

    exp_accessions = [e['experiment_accession'] for e in experiments if 'experiment_accession' in e]
    study_accessions = list(set(e['study_accession'] for e in experiments if 'study_accession' in e))

    # Concurrent fetch of Runs and Studies
    runs = await fetcher.fetch_ena_batch("read_run", "experiment_accession", exp_accessions)
    studies = await fetcher.fetch_ena_batch("study", "study_accession", study_accessions)

    df = optimize_dataframe_operations(samples, experiments, runs, studies, biosamples_info)
    return apply_filters_vectorized(df, amplicon)

async def get_ena_data_by_location_async(
    latitude: float, longitude: float, radius: Union[int, float] = 50,
    email: str = "", amplicon: bool = True, verbose: bool = False,
    max_concurrent: int = 10, progress=None, location_task_id=None, 
    fetcher=None, cache_manager: Optional[CacheManager] = None
) -> pd.DataFrame:
    """Entry point for a single location search."""
    if fetcher is None:
        async with ENAFetcher(email, max_concurrent, progress=progress, cache_manager=cache_manager) as new_fetcher:
            return await _process_location_data(new_fetcher, latitude, longitude, radius, amplicon, progress, location_task_id)
    return await _process_location_data(fetcher, latitude, longitude, radius, amplicon, progress, location_task_id)

async def run_searches_from_dataframe_async(
    input_df: pd.DataFrame, radius: Union[int, float], email: str,
    amplicon: bool, verbose: bool = True, max_concurrent: int = 5,
    cache_dir: Optional[Union[str, Path]] = None,
    progress_obj: Any = None, fetcher: Optional[Any] = None 
) -> pd.DataFrame:
    """Async engine updated for SQLite context management."""
    logger = get_logger("workflow_16s")
    
    if 'latitude' in input_df.columns: input_df = input_df.rename(columns={'latitude': 'lat'})
    if 'longitude' in input_df.columns: input_df = input_df.rename(columns={'longitude': 'lon'})

    # NEW: Initialize SQLite Cache if dir provided
    cache_manager = CacheManager(Path(cache_dir)) if cache_dir else None
    # NEW: Fetch all existing 'nearby' search centers from SQLite 
    # to see what we've already covered in previous years/runs.
    existing_searches = []
    if cache_manager:
        with sqlite3.connect(cache_manager.db_path) as conn:
            # We look for keys starting with 'nearby_' which store (lat, lon, radius)
            cursor = conn.execute("SELECT key, timestamp FROM cache WHERE key LIKE 'nearby_%'")
            for key, ts in cursor.fetchall():
                # Extract metadata from our hashed key format or a secondary metadata table
                # For this logic to work perfectly, we'll store search metadata in a 
                # new SQLite table called 'spatial_index'
                pass
    unique_coords = input_df[['lat', 'lon']].drop_duplicates().dropna()
    progress = progress_obj if progress_obj else get_progress_bar()
    standalone = progress_obj is None
    if standalone: progress.start()

    # Reuse fetcher if available, otherwise spin up new one with SQLite cache
    if fetcher is None:
        context_manager = ENAFetcher(email, max_concurrent * 2, progress=progress, cache_manager=cache_manager)
    else:
        context_manager = null_context(fetcher)

    results = []
    try:
        async with context_manager as active_fetcher:
            main_task = progress.add_task("[cyan]Processing ENA locations", total=len(unique_coords))
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def worker(row):
                async with semaphore:
                    lat, lon = row['lat'], row['lon']
            
                    # --- SPATIAL DEDUPLICATION CHECK ---
                    is_covered = False
                    if cache_manager:
                        # Check if this point falls inside the radius of a PREVIOUS successful search
                        # stored in our new spatial index
                        with sqlite3.connect(cache_manager.db_path) as conn:
                            # Find any previous search where: distance(current, old) + buffer < old_radius
                            # This is a 'Point-in-Circle' test
                            cursor = conn.execute("SELECT lat, lon, radius FROM spatial_index")
                            for s_lat, s_lon, s_rad in cursor.fetchall():
                                dist = haversine_distance(lat, lon, s_lat, s_lon)
                                # If the new point is well within an old search area, skip it!
                                if dist < (s_rad * 0.9): # 90% threshold for safety
                                    is_covered = True
                                    break
                    
                    if is_covered:
                        self.logger.info(f"📍 Skipping {lat}, {lon}: Already covered by existing spatial cache.")
                        progress.update(main_task, advance=1)
                        return None
                    try:
                        # Ensure we pass the SQLite manager down the chain
                        res = await get_ena_data_by_location_async(
                            lat, lon, radius, email, amplicon, verbose, 3, 
                            progress, main_task, fetcher=active_fetcher, cache_manager=cache_manager
                        )
                        if res is not None and not res.empty:
                            res['query_lat'], res['query_lon'] = lat, lon
                            return res
                    except Exception as e:
                        logger.error(f"Worker failed for ({lat}, {lon}): {e}")
                    finally:
                        progress.update(main_task, advance=1)
                return None

            tasks = [worker(row) for _, row in unique_coords.iterrows()]
            results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if standalone: progress.stop()

    dfs = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def get_ena_data_by_location(*args, **kwargs):
    return asyncio.run(get_ena_data_by_location_async(*args, **kwargs))

def run_searches_from_dataframe(*args, **kwargs):
    return asyncio.run(run_searches_from_dataframe_async(*args, **kwargs))


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
        print(f"Cleaning cache at: {args.cache_dir}")
        cache_manager = CacheManager(args.cache_dir)
        # NEW: SQLite cleanup is much faster than file-based globbing
        asyncio.run(cache_manager.clear_expired())
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