"""
Example usage patterns for metadata fetcher and enrichment modules.

This file demonstrates practical integration patterns with the workflow_16s pipeline.
"""

import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Optional

# Import the new metadata modules
from workflow_16s.api.sequence.ena.metadata_fetcher import (
    ENAMetadataFetcher,
    SRAMetadataFetcher,
    MetadataMerger,
)
from workflow_16s.api.sequence.ena.metadata_enrichment import (
    enrich_metadata_with_location,
    enrich_metadata_with_dates,
    create_metadata_enrichment_pipeline,
)
from workflow_16s.api.sequence.ena.cache import SQLiteCacheManager
from workflow_16s.utils.logger import get_logger

logger = get_logger("workflow_16s")


# ============================================================================
# Pattern 1: Simple Metadata Fetching
# ============================================================================

async def fetch_sample_metadata_simple(accession: str) -> Optional[dict]:
    """
    Fetch metadata for a single sample accession.

    Args:
        accession: Sample accession (e.g., "ERS123456")

    Returns:
        Dictionary with sample metadata or None
    """
    fetcher = ENAMetadataFetcher("user@example.com")

    try:
        # Fetch sample metadata
        sample = await fetcher.fetch_sample_metadata(accession)

        if sample:
            logger.info(f"Fetched metadata for {accession}")
            return sample
        else:
            logger.warning(f"No metadata found for {accession}")
            return None

    finally:
        await fetcher.close()


# ============================================================================
# Pattern 2: Batch Metadata Fetching with Caching
# ============================================================================

async def fetch_batch_with_cache(
    accessions: List[str],
    cache_dir: Optional[Path] = None,
) -> dict:
    """
    Fetch metadata for multiple accessions with caching.

    Args:
        accessions: List of sample accessions
        cache_dir: Optional cache directory (uses default if None)

    Returns:
        Dictionary mapping accession -> metadata
    """
    # Initialize cache
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "ena_metadata"

    cache_manager = SQLiteCacheManager(cache_dir)

    # Initialize fetcher
    fetcher = ENAMetadataFetcher(
        email="user@example.com",
        cache_manager=cache_manager
    )

    try:
        # Fetch run metadata in batches
        metadata = await fetcher.fetch_run_metadata_batch(
            accessions,
            batch_size=50
        )

        logger.info(f"Fetched metadata for {len(metadata)}/{len(accessions)} accessions")
        return metadata

    finally:
        await fetcher.close()
        await cache_manager.close()


# ============================================================================
# Pattern 3: ENA + SRA Metadata Merging
# ============================================================================

async def fetch_and_merge_metadata(
    accessions: List[str],
    sra_api_key: Optional[str] = None,
) -> dict:
    """
    Fetch metadata from ENA and SRA, then merge intelligently.

    Args:
        accessions: List of sample accessions
        sra_api_key: Optional NCBI API key for higher rate limits

    Returns:
        Dictionary mapping accession -> merged metadata
    """
    # Initialize fetchers
    ena_fetcher = ENAMetadataFetcher("user@example.com")
    sra_fetcher = SRAMetadataFetcher(
        email="user@example.com",
        api_key=sra_api_key
    )

    try:
        # Fetch from both sources concurrently
        ena_data = await ena_fetcher.fetch_run_metadata_batch(accessions)

        sra_data = {}
        for acc in accessions:
            try:
                sra_meta = await sra_fetcher.fetch_sra_details(acc)
                if sra_meta:
                    sra_data[acc] = sra_meta
            except Exception as e:
                logger.warning(f"Failed to fetch SRA data for {acc}: {e}")

        # Merge with ENA priority
        merged = MetadataMerger.merge_batch(ena_data, sra_data)

        logger.info(f"Merged metadata from {len(merged)} accessions")
        logger.info(f"ENA sources: {sum(1 for m in merged.values() if 'ENA' in m.get('_metadata_sources', []))}")
        logger.info(f"SRA sources: {sum(1 for m in merged.values() if 'SRA' in m.get('_metadata_sources', []))}")

        return merged

    finally:
        await ena_fetcher.close()
        await sra_fetcher.close()


# ============================================================================
# Pattern 4: Complete Enrichment Pipeline
# ============================================================================

async def enrich_dataframe_complete(
    input_file: Path,
    output_file: Path,
    fetch_from_api: bool = False,
) -> pd.DataFrame:
    """
    Complete enrichment pipeline: load, fetch (optional), enrich, save.

    Args:
        input_file: Input CSV with raw metadata
        output_file: Output path for enriched data
        fetch_from_api: Whether to fetch missing metadata from API

    Returns:
        Enriched DataFrame
    """
    # Load input data
    logger.info(f"Loading metadata from {input_file}")
    df = pd.read_csv(input_file)

    # Optionally fetch missing metadata
    if fetch_from_api and "accession" in df.columns:
        logger.info("Fetching additional metadata from ENA API...")
        accessions = df["accession"].tolist()
        metadata = await fetch_batch_with_cache(accessions)

        # Merge into DataFrame
        metadata_df = pd.DataFrame.from_dict(metadata, orient="index")
        df = df.merge(metadata_df, left_on="accession", right_index=True, how="left")

    # Run enrichment pipeline
    logger.info("Running metadata enrichment pipeline...")
    df_enriched = create_metadata_enrichment_pipeline(
        df,
        extract_location=True,
        standardize_dates=True,
        collection_date_col="collection_date"
    )

    # Save enriched data
    logger.info(f"Saving enriched metadata to {output_file}")
    df_enriched.to_csv(output_file, index=False)

    return df_enriched


# ============================================================================
# Pattern 5: Location-Based Metadata Extraction
# ============================================================================

def extract_coordinates_from_batch(
    df: pd.DataFrame,
    lat_fields: Optional[List[str]] = None,
    lon_fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extract and standardize coordinates from a DataFrame.

    Handles various coordinate formats:
    - Explicit lat/lon columns
    - Embedded in location strings
    - DMS format
    - Named format (latitude=X, longitude=Y)

    Args:
        df: DataFrame with location data
        lat_fields: Fields to check for latitude
        lon_fields: Fields to check for longitude

    Returns:
        DataFrame with extracted 'latitude' and 'longitude' columns
    """
    logger.info(f"Extracting coordinates from {len(df)} records...")

    if lat_fields is None:
        lat_fields = ["lat", "latitude", "sample_lat"]

    if lon_fields is None:
        lon_fields = ["lon", "longitude", "sample_lon"]

    df_enriched = enrich_metadata_with_location(
        df,
        lat_fields=lat_fields,
        lon_fields=lon_fields,
        location_fields=["environment", "location", "isolation_source"]
    )

    # Report extraction success
    valid_coords = df_enriched["latitude"].notna().sum()
    logger.info(f"Successfully extracted {valid_coords}/{len(df_enriched)} coordinates")

    return df_enriched


# ============================================================================
# Pattern 6: Date Standardization
# ============================================================================

def standardize_collection_dates(
    df: pd.DataFrame,
    date_col: str = "collection_date",
) -> pd.DataFrame:
    """
    Standardize collection dates in a DataFrame.

    Supports multiple date formats and tracks precision:
    - day: YYYY-MM-DD
    - month: YYYY-MM
    - year: YYYY
    - unknown: Unparseable

    Args:
        df: DataFrame with collection dates
        date_col: Column name containing dates

    Returns:
        DataFrame with standardized dates and precision info
    """
    logger.info(f"Standardizing dates in '{date_col}' column...")

    df_enriched = enrich_metadata_with_dates(
        df,
        collection_date_col=date_col,
        target_col=f"{date_col}_standardized",
        precision_col=f"{date_col}_precision"
    )

    # Report precision distribution
    if f"{date_col}_precision" in df_enriched.columns:
        precision_counts = df_enriched[f"{date_col}_precision"].value_counts()
        for precision, count in precision_counts.items():
            logger.info(f"  {precision}: {count} records")

    return df_enriched


# ============================================================================
# Pattern 7: Error-Tolerant Batch Processing
# ============================================================================

async def fetch_with_error_tolerance(
    accessions: List[str],
    max_retries: int = 3,
    skip_errors: bool = True,
) -> dict:
    """
    Fetch metadata with error tolerance and retry logic.

    Args:
        accessions: List of sample accessions
        max_retries: Maximum retry attempts per accession
        skip_errors: If True, continue on error; if False, raise exception

    Returns:
        Dictionary of successfully fetched metadata
    """
    fetcher = ENAMetadataFetcher("user@example.com")
    results = {}
    failed = []

    try:
        for i, acc in enumerate(accessions):
            retry_count = 0

            while retry_count < max_retries:
                try:
                    metadata = await fetcher.fetch_sample_metadata(acc, use_cache=False)

                    if metadata:
                        results[acc] = metadata
                        logger.debug(f"[{i+1}/{len(accessions)}] Fetched {acc}")
                        break
                    else:
                        logger.warning(f"[{i+1}/{len(accessions)}] No data found for {acc}")
                        failed.append(acc)
                        break

                except Exception as e:
                    retry_count += 1
                    logger.warning(
                        f"[{i+1}/{len(accessions)}] Error fetching {acc} "
                        f"(attempt {retry_count}/{max_retries}): {e}"
                    )

                    if retry_count < max_retries:
                        # Exponential backoff
                        wait_time = 2 ** retry_count
                        logger.info(f"Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    elif not skip_errors:
                        raise

            if retry_count >= max_retries and acc not in results:
                failed.append(acc)

        logger.info(f"Successfully fetched {len(results)}/{len(accessions)} accessions")
        if failed:
            logger.warning(f"Failed to fetch {len(failed)} accessions: {failed}")

        return results

    finally:
        await fetcher.close()


# ============================================================================
# Example: End-to-End Integration
# ============================================================================

async def complete_workflow_example():
    """
    Complete example: load, fetch, merge, enrich, and save metadata.
    """
    # Setup
    input_file = Path("raw_samples.csv")
    output_file = Path("enriched_samples.csv")
    cache_dir = Path.home() / ".cache" / "ena_metadata"

    # Step 1: Load raw data
    logger.info("Step 1: Loading raw data...")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} samples")

    # Step 2: Fetch missing metadata from API
    logger.info("Step 2: Fetching metadata from ENA...")
    accessions = df["accession"].dropna().tolist()
    metadata = await fetch_batch_with_cache(accessions, cache_dir)

    # Step 3: Merge fetched metadata
    logger.info("Step 3: Merging metadata...")
    metadata_df = pd.DataFrame.from_dict(metadata, orient="index")
    df = df.merge(metadata_df, left_on="accession", right_index=True, how="left")

    # Step 4: Extract coordinates
    logger.info("Step 4: Extracting geographic coordinates...")
    df = extract_coordinates_from_batch(df)

    # Step 5: Standardize dates
    logger.info("Step 5: Standardizing collection dates...")
    df = standardize_collection_dates(df)

    # Step 6: Save enriched data
    logger.info("Step 6: Saving enriched data...")
    df.to_csv(output_file, index=False)

    logger.info(f"Enriched metadata saved to {output_file}")
    logger.info(f"Columns: {', '.join(df.columns)}")

    return df


if __name__ == "__main__":
    # Example: Fetch single sample
    print("\n=== Example 1: Simple Fetch ===")
    sample = asyncio.run(fetch_sample_metadata_simple("ERS012345"))
    if sample:
        print(f"Collection Date: {sample.get('collection_date')}")
        print(f"Latitude: {sample.get('lat')}")
        print(f"Longitude: {sample.get('lon')}")

    # Example: Batch fetch
    print("\n=== Example 2: Batch Fetch ===")
    metadata = asyncio.run(fetch_batch_with_cache(
        ["ERS001", "ERS002", "ERS003"]
    ))
    print(f"Fetched metadata for {len(metadata)} accessions")

    # Example: Merge ENA + SRA
    print("\n=== Example 3: ENA + SRA Merge ===")
    merged = asyncio.run(fetch_and_merge_metadata(
        ["ERS001", "ERS002"]
    ))
    print(f"Merged {len(merged)} accessions from ENA and SRA")
