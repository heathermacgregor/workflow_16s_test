"""
Coordinate-Based Fallback Search for Non-ENA Samples

This module implements a coordinate-based search to find nearby ENA samples
and infer/supplement metadata for non-ENA samples that lack direct accessions
but have geographic coordinates.

Key Features:
- Finds nearby ENA samples within specified radius (default 0.1 degrees ≈ 11 km)
- Intelligently merges metadata from nearby samples
- Handles conflicts and multiple matches
- Performance optimized for 100K+ samples
- Optional/configurable (can be disabled)
- Marks inferred metadata with provenance tags
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import hashlib

import anndata as ad
from workflow_16s.api.sequence.ena.finder import find_nearby_samples_async
from workflow_16s.api.sequence.ena.cache import SQLiteCacheManager as CacheManager


logger = logging.getLogger("workflow_16s")


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points in kilometers."""
    import math

    try:
        lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
    except (ValueError, TypeError):
        return float('inf')

    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2)
    return R * 2 * math.asin(math.sqrt(a))


def _identify_coordinate_columns(obs_df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Find latitude and longitude columns in metadata.

    Checks for multiple naming conventions:
    1. 'lat' / 'lon'
    2. 'latitude' / 'longitude'
    3. 'LatitudeParsed' / 'LongitudeParsed'

    Returns:
        Tuple of (lat_col, lon_col) or (None, None) if not found
    """
    candidates = [
        ('lat', 'lon'),
        ('latitude', 'longitude'),
        ('LatitudeParsed', 'LongitudeParsed'),
        ('Latitude', 'Longitude'),
        ('Lat', 'Lon'),
    ]

    for lat_col, lon_col in candidates:
        if lat_col in obs_df.columns and lon_col in obs_df.columns:
            return lat_col, lon_col

    return None, None


def _has_accession(row: pd.Series) -> bool:
    """Check if a sample has any SRA/ENA accession."""
    accession_cols = ['run_accession', 'sample_accession', 'experiment_accession',
                     'study_accession', 'secondary_sample_accession', 'sample_id']

    for col in accession_cols:
        if col in row.index and pd.notna(row[col]):
            val = str(row[col]).strip().upper()
            if val and val.startswith(('SRR', 'ERR', 'DRR', 'SRS', 'SAMN', 'SAME', 'SAMD', 'SRX', 'ERX', 'DRX')):
                return True
    return False


def _select_best_match(
    nearby_samples: pd.DataFrame,
    query_lat: float,
    query_lon: float
) -> Optional[Dict]:
    """
    Select the best matching sample from nearby results.

    Strategy:
    1. Prefer samples with complete metadata (collection_date + location)
    2. Prefer samples closest to query point
    3. Prefer environmental/soil samples over host-associated

    Args:
        nearby_samples: DataFrame from find_nearby_samples_async
        query_lat: Query latitude
        query_lon: Query longitude

    Returns:
        Best sample as dict, or None if no suitable match
    """
    if nearby_samples.empty:
        return None

    # Score candidates by metadata completeness
    nearby_samples_copy = nearby_samples.copy()

    # Calculate distance to query point
    if 'lat' in nearby_samples_copy.columns and 'lon' in nearby_samples_copy.columns:
        nearby_samples_copy['distance_km'] = nearby_samples_copy.apply(
            lambda row: haversine_distance(
                query_lat, query_lon,
                float(row['lat']) if pd.notna(row['lat']) else query_lat,
                float(row['lon']) if pd.notna(row['lon']) else query_lon
            ),
            axis=1
        )
    else:
        nearby_samples_copy['distance_km'] = 0

    # Score based on metadata completeness
    nearby_samples_copy['metadata_score'] = 0

    if 'collection_date' in nearby_samples_copy.columns:
        nearby_samples_copy['metadata_score'] += nearby_samples_copy['collection_date'].notna().astype(int) * 3

    if 'location' in nearby_samples_copy.columns:
        nearby_samples_copy['metadata_score'] += nearby_samples_copy['location'].notna().astype(int) * 2

    # Prefer samples with lower host association
    if 'host' in nearby_samples_copy.columns:
        nearby_samples_copy['host_score'] = nearby_samples_copy['host'].isna().astype(int)
    else:
        nearby_samples_copy['host_score'] = 1

    # Combined score: metadata_score (primary), distance (secondary), host (tertiary)
    nearby_samples_copy['combined_score'] = (
        nearby_samples_copy['metadata_score'] * 1000 -
        nearby_samples_copy['distance_km'] -
        (~nearby_samples_copy['host_score'].astype(bool)).astype(int) * 10
    )

    # Get best match
    best_idx = nearby_samples_copy['combined_score'].idxmax()
    best_match = nearby_samples_copy.loc[best_idx].to_dict()

    return best_match


def _merge_metadata_fields(
    target_row: pd.Series,
    source_dict: Dict,
    fields_to_copy: Optional[List[str]] = None,
    tag_inferred: bool = True
) -> pd.Series:
    """
    Merge metadata from nearby sample into target sample.

    Args:
        target_row: Target sample row (may be modified)
        source_dict: Source sample data dict
        fields_to_copy: List of fields to copy (None = auto-detect)
        tag_inferred: If True, add '_nearby_inferred' suffix to field names

    Returns:
        Modified target row with merged fields
    """
    if fields_to_copy is None:
        # Auto-detect fields: anything not an accession or distance metric
        exclude = {'accession', 'distance_km', 'combined_score', 'metadata_score', 'host_score'}
        fields_to_copy = [k for k in source_dict.keys() if k not in exclude]

    for field in fields_to_copy:
        if field not in source_dict or pd.isna(source_dict[field]):
            continue

        # Check if target already has this field filled
        if field in target_row.index and pd.notna(target_row[field]):
            continue

        # Add field with optional tag
        field_name = f"{field}_nearby_inferred" if tag_inferred else field
        target_row[field_name] = source_dict[field]

    return target_row


async def supplement_with_nearby_samples_async(
    adata: ad.AnnData,
    lat_col: str,
    lon_col: str,
    radius_degrees: float = 0.1,
    email: str = "default@example.com",
    max_concurrent: int = 5,
    cache_dir: Optional[Path] = None,
    logger_instance: Optional[logging.Logger] = None,
    tag_inferred: bool = True,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Asynchronously supplement non-ENA samples with nearby ENA metadata.

    This is the core async implementation that:
    1. Identifies samples without ENA accessions but WITH coordinates
    2. Searches for nearby ENA samples (within radius_degrees)
    3. Merges selective metadata from best nearby matches
    4. Tags inferred fields for traceability

    Args:
        adata: AnnData object with metadata
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        radius_degrees: Search radius in degrees (~0.1° ≈ 11 km)
        email: Email for ENA API requests
        max_concurrent: Maximum concurrent API requests
        cache_dir: Optional directory for caching nearby searches
        logger_instance: Optional logger
        tag_inferred: If True, suffix inferred fields with '_nearby_inferred'
        max_samples: Optional limit on samples to process (for testing)

    Returns:
        Modified adata.obs DataFrame
    """
    if logger_instance is None:
        logger_instance = logging.getLogger("workflow_16s")

    obs = adata.obs.copy()

    # Step 1: Identify candidates - non-ENA samples with coordinates
    logger_instance.debug("   🔍 Identifying non-ENA samples with coordinates...")

    # Convert coordinates to numeric
    lat_numeric = pd.to_numeric(obs[lat_col], errors='coerce')
    lon_numeric = pd.to_numeric(obs[lon_col], errors='coerce')

    # Find candidates: no accession + has coordinates
    candidates_mask = (
        (lat_numeric.notna()) &
        (lon_numeric.notna()) &
        ~obs.index.map(lambda idx: _has_accession(obs.loc[idx]))
    )

    candidates = obs[candidates_mask].copy()

    if candidates.empty:
        logger_instance.debug("   ℹ️  No non-ENA samples with coordinates found for fallback search")
        return obs

    logger_instance.info(f"   🎯 Found {len(candidates)}/{len(obs)} samples without accessions but WITH coordinates")

    # Apply max_samples limit if specified
    if max_samples and len(candidates) > max_samples:
        candidates = candidates.iloc[:max_samples]
        logger_instance.debug(f"   ⚙️  Limited to {max_samples} samples for processing (testing mode)")

    # Step 2: Initialize cache if provided
    cache_manager = CacheManager(cache_dir) if cache_dir else None

    # Step 3: Async search for nearby samples
    logger_instance.debug(f"   🚀 Starting async search for nearby samples (radius: {radius_degrees}°)...")

    # Convert degrees to km for logging (rough approximation)
    radius_km = radius_degrees * 111  # 1 degree ≈ 111 km
    logger_instance.debug(f"      Radius: {radius_degrees}° ≈ {radius_km:.0f} km")

    async def search_worker(sample_id: str, lat: float, lon: float) -> Tuple[str, Optional[Dict]]:
        """Search for nearby samples for a single coordinate."""
        try:
            import aiohttp
            from workflow_16s.api.sequence.ena.constants import ENA_API_URL

            # Use global session to avoid resource exhaustion
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    # Wrap with timeout to prevent hanging on single slow request
                    result_df = await asyncio.wait_for(
                        find_nearby_samples_async(
                            session=session,
                            latitude=lat,
                            longitude=lon,
                            radius=radius_km,
                            cache_manager=cache_manager
                        ),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger_instance.warning(f"      ⏱️  Timeout searching for {sample_id} (30s)")
                    return sample_id, None

                if result_df.empty:
                    return sample_id, None

                # Select best match from nearby samples
                best_match = _select_best_match(result_df, lat, lon)
                return sample_id, best_match

        except Exception as e:
            logger_instance.warning(f"      ⚠️  Search failed for {sample_id}: {e}")
            return sample_id, None

    # Run searches concurrently with semaphore
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_search(sample_id: str, lat: float, lon: float):
        async with semaphore:
            return await search_worker(sample_id, lat, lon)

    # Create tasks for all candidates
    tasks = []
    for sample_id, row in candidates.iterrows():
        lat = float(lat_numeric[sample_id])
        lon = float(lon_numeric[sample_id])
        tasks.append(bounded_search(sample_id, lat, lon))

    # Execute all searches
    logger_instance.debug(f"   📡 Running {len(tasks)} concurrent searches...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Step 4: Process results and merge metadata
    logger_instance.debug("   ✨ Processing search results...")

    merged_count = 0
    fields_merged_counts = {}

    for result in results:
        if isinstance(result, Exception):
            logger_instance.debug(f"      ⚠️  Exception in search: {result}")
            continue

        sample_id, best_match = result

        if best_match is None:
            continue

        # Merge metadata from best match
        try:
            # Define which fields to copy (prioritize collection_date, location, etc.)
            priority_fields = ['collection_date', 'location', 'host', 'scientific_name',
                             'description', 'accession', 'lat', 'lon']

            for field in priority_fields:
                if field not in best_match:
                    continue

                if pd.isna(best_match[field]):
                    continue

                # Check if target already has this field
                if field in obs.columns and pd.notna(obs.loc[sample_id, field]):
                    continue

                # Add field with inferred tag
                field_name = f"{field}_nearby_inferred" if tag_inferred else field
                obs.loc[sample_id, field_name] = best_match[field]

                # Track merged fields
                if field not in fields_merged_counts:
                    fields_merged_counts[field] = 0
                fields_merged_counts[field] += 1

            merged_count += 1

        except Exception as e:
            logger_instance.warning(f"      ⚠️  Failed to merge metadata for {sample_id}: {e}")

    # Step 5: Log results
    if merged_count > 0:
        logger_instance.info(f"   ✅ Supplement complete: {merged_count}/{len(candidates)} samples enriched with nearby metadata")

        # Log field-level statistics
        logger_instance.debug(f"   📋 Fields merged (count):")
        for field, count in sorted(fields_merged_counts.items(), key=lambda x: x[1], reverse=True):
            logger_instance.debug(f"      • {field:20s}: {count:4d} samples")
    else:
        logger_instance.debug(f"   ℹ️  No nearby samples found for any candidates")

    # Close cache if it was opened
    if cache_manager and hasattr(cache_manager, 'close'):
        await cache_manager.close()

    return obs


def supplement_with_nearby_samples(
    adata: ad.AnnData,
    config: Optional[any] = None,
    logger_instance: Optional[logging.Logger] = None,
    **kwargs
) -> None:
    """
    Synchronous wrapper for coordinate-based fallback search.

    Supplements non-ENA samples with nearby ENA metadata if:
    - Sample lacks ENA/SRA accessions
    - Sample has valid lat/lon coordinates
    - Nearby samples are found within search radius

    Configuration (from config.coordinate_fallback or kwargs):
    - enabled: bool (default True) - Enable/disable feature
    - radius_degrees: float (default 0.1) - Search radius (~11 km)
    - max_concurrent: int (default 5) - Max concurrent API calls
    - tag_inferred: bool (default True) - Tag inferred fields
    - cache_dir: Optional[Path] - Cache directory for searches

    Args:
        adata: AnnData object to modify in-place
        config: AppConfig object with optional coordinate_fallback settings
        logger_instance: Optional logger
        **kwargs: Override config settings (e.g., radius_degrees=0.2)
    """
    if logger_instance is None:
        logger_instance = logging.getLogger("workflow_16s")

    # Parse configuration
    enabled = kwargs.get('enabled', True)
    if not enabled:
        logger_instance.debug("   ℹ️  Coordinate fallback search disabled")
        return

    # Get config from AppConfig object if provided
    if config and hasattr(config, 'coordinate_fallback'):
        cfg = config.coordinate_fallback
        defaults = {
            'enabled': getattr(cfg, 'enabled', True),
            'radius_degrees': getattr(cfg, 'radius_degrees', 0.1),
            'max_concurrent': getattr(cfg, 'max_concurrent', 5),
            'tag_inferred': getattr(cfg, 'tag_inferred', True),
            'cache_dir': getattr(cfg, 'cache_dir', None),
        }
        defaults.update(kwargs)
    else:
        defaults = {
            'radius_degrees': kwargs.get('radius_degrees', 0.1),
            'max_concurrent': kwargs.get('max_concurrent', 5),
            'tag_inferred': kwargs.get('tag_inferred', True),
            'cache_dir': kwargs.get('cache_dir', None),
            'email': kwargs.get('email', 'default@example.com'),
        }

    # Find coordinate columns
    lat_col, lon_col = _identify_coordinate_columns(adata.obs)

    if lat_col is None or lon_col is None:
        logger_instance.debug("   ℹ️  No lat/lon coordinates found for coordinate fallback search")
        return

    # Run async search
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop running, create one
            loop = None

        if loop is not None:
            # Already in event loop, use run_until_complete (don't use asyncio.run)
            adata.obs = loop.run_until_complete(supplement_with_nearby_samples_async(
                adata=adata,
                lat_col=lat_col,
                lon_col=lon_col,
                radius_degrees=defaults.get('radius_degrees', 0.1),
                email=defaults.get('email', 'default@example.com'),
                max_concurrent=defaults.get('max_concurrent', 5),
                cache_dir=defaults.get('cache_dir'),
                logger_instance=logger_instance,
                tag_inferred=defaults.get('tag_inferred', True),
            ))
        else:
            # Not in event loop, can safely use asyncio.run
            adata.obs = asyncio.run(supplement_with_nearby_samples_async(
                adata=adata,
                lat_col=lat_col,
                lon_col=lon_col,
                radius_degrees=defaults.get('radius_degrees', 0.1),
                email=defaults.get('email', 'default@example.com'),
                max_concurrent=defaults.get('max_concurrent', 5),
                cache_dir=defaults.get('cache_dir'),
                logger_instance=logger_instance,
                tag_inferred=defaults.get('tag_inferred', True),
            ))

    except Exception as e:
        logger_instance.warning(f"   ⚠️  Coordinate fallback search failed: {e}")
        import traceback
        logger_instance.debug(f"   Traceback:\n{traceback.format_exc()}")
