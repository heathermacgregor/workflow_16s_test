# workflow_16s/api/ena/metadata_api.py

"""
ENA Environmental Sample Data Retrieval - Consolidated & Improved.

Orchestrates asynchronous fetching, caching, and processing of ENA data
based on geographic locations or BioProject.
"""
# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional, Union

# Third Party Imports
import pandas as pd

# Local Imports
from workflow_16s.utils.progress import get_progress_bar
from workflow_16s.utils.logger import get_logger

# Import from refactored modules
# Note the path change: from .metadata.cache etc.
from .metadata.cache import CacheManager
from .metadata.constants import DEFAULT_EMAIL, DEFAULT_CACHE_DIR
from .metadata.fetcher import ENAFetcher
from .metadata.utils import (
    process_and_structure_data,
    apply_filters,
    save_results_by_location
)

# ================================= CONFIGURATION ================================== #

logger = get_logger()

# ======================== HIGH-LEVEL WORKFLOW FUNCTIONS ======================= #

async def get_samples_by_location_async(
    lat: float, 
    lon: float, 
    radius: Union[int, float], 
    email: str = DEFAULT_EMAIL,
    max_concurrent: int = 15, 
    cache_manager: Optional[CacheManager] = None,
    fetcher: Optional[ENAFetcher] = None 
) -> pd.DataFrame:
    """Fetches and structures all sample data within a geographic radius."""
    async def _get_data(fetcher_instance: ENAFetcher) -> pd.DataFrame:
        logger.info(f"Finding samples within {radius}km of ({lat}, {lon})...")
        samples = await fetcher_instance.find_nearby_samples(lat, lon, radius)
        if not samples:
            logger.info(f"No samples found within {radius}km of ({lat}, {lon}).")
            return pd.DataFrame()

        sample_accessions = [s['accession'] for s in samples if 'accession' in s]
        logger.info(f"Found {len(sample_accessions)} samples. Fetching associated run, biosample, and taxonomy data...")

        # Concurrently fetch all related data
        runs_task = fetcher_instance.fetch_ena_data_in_batches(
            "read_run", 
            "sample_accession", 
            sample_accessions, 
            with_progress_bar=False
        )
        biosamples_task = fetcher_instance.fetch_biosamples_in_batches(
            sample_accessions 
        )
        tax_ids = [s.get('tax_id') for s in samples if s.get('tax_id')] 
        taxonomy_task = fetcher_instance.fetch_taxonomies(tax_ids)

        # Await all tasks
        runs_result, biosamples_result, taxonomies_result = await asyncio.gather(
            runs_task, 
            biosamples_task,
            taxonomy_task, 
            return_exceptions=True
        )

        # Handle potential errors from gather
        if isinstance(runs_result, Exception): 
            logger.error(f"Error fetching runs for location ({lat},{lon}): {runs_result}")
            runs = []
        else:
            runs = runs_result
        if isinstance(biosamples_result, Exception): 
            logger.error(f"Error fetching biosamples for location ({lat},{lon}): {biosamples_result}")
            biosamples_info = {}
        else:
            biosamples_info = biosamples_result
        if isinstance(taxonomies_result, Exception): 
            logger.error(f"Error fetching taxonomies for location ({lat},{lon}): {taxonomies_result}")
            taxonomies = {}
        else:
            taxonomies = taxonomies_result


        # Process data even if some parts failed
        structured_data = process_and_structure_data(samples, runs if isinstance(runs, list) else [], biosamples_info if isinstance(biosamples_info, dict) else {})

        # Add taxonomy lineage if fetched and data exists
        if isinstance(taxonomies, dict) and taxonomies and not structured_data.empty and 'tax_id' in structured_data.columns:
            numeric_tax_id = pd.to_numeric(structured_data['tax_id'], errors='coerce')
            valid_tax_ids_map = numeric_tax_id.dropna().astype(int) # Series of valid integer IDs
            lineage_map = valid_tax_ids_map.map(taxonomies).fillna("N/A")
            structured_data['taxonomy_lineage'] = lineage_map

        return structured_data

    # Manage fetcher creation if not provided
    if fetcher:
        return await _get_data(fetcher)
    else:
        # Create a new fetcher context if none was passed
        async with ENAFetcher(email, max_concurrent, cache_manager) as new_fetcher:
            return await _get_data(new_fetcher)

def get_samples_by_location(*args, **kwargs) -> pd.DataFrame:
    """Synchronous wrapper for get_samples_by_location_async."""
    return asyncio.run(get_samples_by_location_async(*args, **kwargs))


async def get_n_samples_by_bioproject_async( # Renamed from original duplicate name
    bioproject_accession: str, 
    email: str = DEFAULT_EMAIL, 
    max_concurrent: int = 15,
    cache_manager: Optional[CacheManager] = None, 
    fetcher: Optional[ENAFetcher] = None
) -> int:
    """ Fetches only the count of samples for a given BioProject accession."""
    async def _get_count(fetcher_instance: ENAFetcher) -> int:
        samples = await fetcher_instance.fetch_ena_data_in_batches(
            result_type="sample",
            query_key="study_accession",
            accessions=[bioproject_accession],
            with_progress_bar=False 
        )
        count = len(samples) if samples else 0
        logger.debug(f"Counted {count} samples for BioProject {bioproject_accession}.")
        return count

    # Manage fetcher instance
    if fetcher:
        return await _get_count(fetcher)
    else:
        async with ENAFetcher(email, max_concurrent, cache_manager) as new_fetcher:
            return await _get_count(new_fetcher)


async def get_samples_by_bioproject_async(
    bioproject_accession: str, 
    email: str = DEFAULT_EMAIL, 
    max_concurrent: int = 15,
    cache_manager: Optional[CacheManager] = None, 
    fetcher: Optional[ENAFetcher] = None
) -> pd.DataFrame:
    """Fetches and structures all sample, run, biosample, and taxonomy data for a BioProject."""
    async def _get_data(fetcher_instance: ENAFetcher) -> pd.DataFrame:
        logger.info(f"Processing BioProject: {bioproject_accession}")

        # Step 1: Find all samples for the BioProject
        samples = await fetcher_instance.fetch_ena_data_in_batches(
            "sample", 
            "study_accession", 
            [bioproject_accession],
            with_progress_bar=True
        )
        if not samples: 
            logger.info(f"No samples found for BioProject {bioproject_accession}.")
            return pd.DataFrame()
        
        sample_accessions = [s['accession'] for s in samples if 'accession' in s]
        logger.info(f"Found {len(sample_accessions)} samples. Fetching associated run, biosample, and taxonomy data...")

        runs_task = fetcher_instance.fetch_ena_data_in_batches(
            "read_run", 
            "sample_accession", 
            sample_accessions, 
            with_progress_bar=False
        )
        biosamples_task = fetcher_instance.fetch_biosamples_in_batches(sample_accessions)
        tax_ids = [s.get('tax_id') for s in samples if s.get('tax_id')] 
        taxonomy_task = fetcher_instance.fetch_taxonomies(tax_ids)

        # Await all tasks
        runs_result, biosamples_result, taxonomies_result = await asyncio.gather(
            runs_task,
            biosamples_task,
            taxonomy_task,
            return_exceptions=True
        )

        # Handle errors from gather results
        if isinstance(runs_result, Exception): 
            logger.error(f"Error fetching runs for {bioproject_accession}: {runs_result}")
            runs = []
        else:
            runs = runs_result
        if isinstance(biosamples_result, Exception): 
            logger.error(f"Error fetching biosamples for {bioproject_accession}: {biosamples_result}")
            biosamples_info = {}
        else:
            biosamples_info = biosamples_result
        if isinstance(taxonomies_result, Exception): 
            logger.error(f"Error fetching taxonomies for {bioproject_accession}: {taxonomies_result}")
            taxonomies = {}
        else:
            taxonomies = taxonomies_result

        # Step 3: Combine fetched data into a DataFrame
        structured_data = process_and_structure_data(
            samples, 
            runs if isinstance(runs, list) else [], 
            biosamples_info if isinstance(biosamples_info, dict) else {}
        )

        # Step 4: Add taxonomy lineage if available
        if isinstance(taxonomies, dict) and taxonomies and not structured_data.empty and 'tax_id' in structured_data.columns:
            numeric_tax_id = pd.to_numeric(structured_data['tax_id'], errors='coerce')
            valid_tax_ids_map = numeric_tax_id.dropna().astype(int)
            lineage_map = valid_tax_ids_map.map(taxonomies).fillna("N/A")
            structured_data['taxonomy_lineage'] = lineage_map

        return structured_data

    # Manage fetcher instance
    if fetcher:
        return await _get_data(fetcher)
    else:
        async with ENAFetcher(email, max_concurrent, cache_manager) as new_fetcher:
            return await _get_data(new_fetcher)


def get_samples_by_bioproject(*args, **kwargs) -> pd.DataFrame:
    """Synchronous wrapper for get_samples_by_bioproject_async."""
    return asyncio.run(get_samples_by_bioproject_async(*args, **kwargs))

# ======================== SCRIPT-SPECIFIC WORKFLOW FUNCTIONS ====================== #

async def process_single_location(
    fetcher: ENAFetcher, lat: float, lon: float, radius: Union[int, float],
    amplicon: bool, no_host: bool
) -> pd.DataFrame:
    """Orchestrates fetching, structuring, and filtering for one location."""
    # Pass the existing fetcher to the main data retrieval function
    df = await get_samples_by_location_async(
        lat, 
        lon, 
        radius, 
        fetcher.email, 
        fetcher.max_concurrent,
        fetcher.cache_manager, 
        fetcher=fetcher
    )

    # Apply filters after data retrieval
    filtered_df = apply_filters(df, amplicon, no_host)

    # Add query coordinates for grouping results later, if data remains
    if not filtered_df.empty:
        filtered_df['query_lat'] = lat
        filtered_df['query_lon'] = lon

    return filtered_df

async def run_location_searches_from_file(args: argparse.Namespace) -> pd.DataFrame:
    """Main async function to run searches for all locations in an input file."""
    try:
        input_df = pd.read_csv(args.input_file)
        # --- Robust Lat/Lon Column Handling ---
        lat_col, lon_col = None, None
        if 'lat' in input_df.columns: 
            lat_col = 'lat'
        elif 'latitude' in input_df.columns: 
            lat_col = 'latitude'
        if 'lon' in input_df.columns: 
            lon_col = 'lon'
        elif 'longitude' in input_df.columns: 
            lon_col = 'longitude'

        if not lat_col or not lon_col:
            logger.error("Input CSV must contain latitude ('lat' or 'latitude') and longitude ('lon' or 'longitude') columns.")
            return pd.DataFrame()

        # Rename to 'lat', 'lon' for consistency and convert to numeric
        input_df.rename(columns={lat_col: 'lat', lon_col: 'lon'}, inplace=True)
        input_df['lat'] = pd.to_numeric(input_df['lat'], errors='coerce')
        input_df['lon'] = pd.to_numeric(input_df['lon'], errors='coerce')
        # --- End Column Handling ---

    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input_file}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error reading input file {args.input_file}: {e}")
        return pd.DataFrame()

    cache_manager = CacheManager(args.cache_dir) if not args.no_cache else None
    # Ensure unique, valid coordinate pairs are processed
    unique_coords = input_df[['lat', 'lon']].dropna().drop_duplicates()
    if unique_coords.empty:
        logger.warning("No valid, unique coordinate pairs found in the input file after cleaning.")
        return pd.DataFrame()
    logger.info(f"Found {len(unique_coords)} unique valid coordinate pairs to process.")

    all_results_dfs = []
    # Use context manager for progress bar
    with get_progress_bar() as progress:
        main_task = progress.add_task("Processing locations...", total=len(unique_coords))

        # Create one fetcher context for all location tasks
        async with ENAFetcher(args.email, args.max_concurrent, cache_manager) as fetcher:
            # Limit how many locations run truly concurrently using a semaphore
            location_semaphore = asyncio.Semaphore(args.max_concurrent_locations)

            async def location_worker(lat, lon):
                # Acquire semaphore before processing a location
                async with location_semaphore:
                    try:
                        progress.update(main_task, description=f"Processing ({lat:.3f}, {lon:.3f})...")
                        # Pass the shared fetcher instance to the processing function
                        result_df = await process_single_location(
                            fetcher, 
                            lat, 
                            lon, 
                            args.radius,
                            args.amplicon, 
                            args.no_host
                        )
                        count = len(result_df)
                        progress.update(main_task, description=f"Done ({lat:.3f}, {lon:.3f}) - {count} results.", advance=1)
                        return result_df # Return DataFrame (might be empty)
                    except Exception as e:
                        # Log error but allow other tasks to continue
                        logger.error(f"Error processing location ({lat}, {lon}): {e}", exc_info=True)
                        progress.update(main_task, description=f"Error ({lat:.3f}, {lon:.3f})", advance=1)
                        return pd.DataFrame() # Return empty DataFrame on error

            # Create tasks for all unique coordinates
            tasks = [location_worker(row.lat, row.lon) for row in unique_coords.itertuples()]
            # Gather results from all tasks
            results = await asyncio.gather(*tasks, return_exceptions=False) # Don't let one failure stop others

    # Collect non-empty results after all tasks are done
    for res_df in results:
        if isinstance(res_df, pd.DataFrame) and not res_df.empty:
            all_results_dfs.append(res_df)

    if not all_results_dfs:
        logger.info("No results found for any location after processing.")
        return pd.DataFrame()

    # Concatenate all results into a single DataFrame
    logger.info(f"Concatenating results from {len(all_results_dfs)} locations.")
    return pd.concat(all_results_dfs, ignore_index=True)


# =================================== MAIN EXECUTION ================================= #

def main():
    """Parses command-line arguments and starts the asynchronous data retrieval."""
    parser = argparse.ArgumentParser(
        description="Optimized ENA sample finder by location or BioProject, with async processing and caching.",
        epilog="Examples:\n"
            "  python -m workflow_16s.api.ena.metadata --input-file locations.csv --radius 25\n"
            "  python -m workflow_16s.api.ena.metadata --bioproject PRJNA293382 --no-host"
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--input-file", type=Path, help="Path to CSV file with lat/lon columns.")
    mode_group.add_argument("--bioproject", type=str, help="A single BioProject accession (e.g., PRJNA12345).")

    parser.add_argument("--radius", type=int, default=50, help="Search radius in km (for location search).")
    parser.add_argument("--email", type=str, default=DEFAULT_EMAIL, help=f"Email for API ID (default: {DEFAULT_EMAIL}).")
    parser.add_argument("--no-host", action="store_true", help="Filter OUT host-associated samples (e.g., human).")
    parser.add_argument("--amplicon", action="store_true", default=False, help="Filter FOR amplicon sequencing data (default: False).")
    parser.add_argument("--max-concurrent", type=int, default=15, help="Max concurrent internal API requests PER location/project.")
    parser.add_argument("--max-concurrent-locations", type=int, default=5, help="Max number of locations/projects to process simultaneously.")

    parser.add_argument("--output-dir", type=Path, default="ena_metadata_results", help="Output directory for results.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help=f"Cache directory (default: {DEFAULT_CACHE_DIR}).")
    parser.add_argument("--no-cache", action="store_true", help="Disable all request caching.")

    args = parser.parse_args()

    # --- Workflow Execution ---
    if args.input_file:
        # Run the location-based workflow from a file
        logger.info(f"Starting location-based search from file: {args.input_file}")
        combined_df = asyncio.run(run_location_searches_from_file(args))
        if combined_df.empty:
            print("\nNo data found matching criteria for any provided locations.")
        else:
            save_results_by_location(combined_df, args.output_dir)
            print(f"\nLocation-based processing complete. Results saved in '{args.output_dir.resolve()}'.")

    elif args.bioproject:
        # Run the BioProject-based workflow
        logger.info(f"Starting BioProject search for: {args.bioproject}")
        cache_manager = CacheManager(args.cache_dir) if not args.no_cache else None
        # Use synchronous wrapper for single project processing
        raw_df = get_samples_by_bioproject(
            args.bioproject, args.email, args.max_concurrent, cache_manager
        )

        filtered_df = apply_filters(raw_df, args.amplicon, args.no_host)

        if filtered_df.empty:
            print(f"\nNo data matching filters found for BioProject {args.bioproject}.")
        else:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"bioproject_{args.bioproject}.csv"
            filepath = output_dir / filename
            logger.info(f"Saving {len(filtered_df)} filtered records for BioProject {args.bioproject} to '{filepath.resolve()}'")
            try:
                filtered_df.to_csv(filepath, index=False)
                print(f"\nBioProject processing complete. Results saved to '{filepath.resolve()}'.")
            except Exception as e:
                logger.error(f"Failed to save results for BioProject {args.bioproject} to {filepath}: {e}")


if __name__ == "__main__":
    # Setup basic logging to catch early argument parsing or config errors
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)-8s %(name)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    # Run the main function which handles detailed logging setup later if config loaded
    main()