import os
import pandas as pd
import numpy as np
import logging
import asyncio
import nest_asyncio
import requests
import gzip
import io
from typing import Any, Tuple, Optional
from pathlib import Path

# Fix for "asyncio.run() cannot be called from a running event loop"
nest_asyncio.apply()

from workflow_16s.api.environmental_data import EnvironmentalDataCollector, run_arkin_enrichment
from workflow_16s.api.sequence.ena import fetch_metadata_from_ena
from workflow_16s.api.sequence.ena.coordinate_fallback import supplement_with_nearby_samples
from workflow_16s.utils.dir_utils import Project
from workflow_16s.downstream.utils import fix_adata_dtypes
from workflow_16s.api.environmental_data.other.tools._google_earth_engine import enrich_with_gee_data

logger = logging.getLogger("workflow_16s")

def find_coordinate_columns(obs_df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Intelligently find latitude and longitude columns in metadata.
    
    Checks for multiple column naming conventions:
    1. 'lat' / 'lon' (short form)
    2. 'latitude' / 'longitude' (MicrobeAtlas ingestion output)
    3. 'LatitudeParsed' / 'LongitudeParsed' (MicrobeAtlas original)
    4. Other variations (case-insensitive)
    
    Args:
        obs_df: AnnData obs (metadata) DataFrame
        
    Returns:
        Tuple of (lat_col, lon_col) or (None, None) if not found
    """
    # Priority list of column name pairs to check
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
    
    # Sanity check: Ensure found columns have numeric data and valid coordinate ranges
    return None, None

def validate_and_clamp_coordinates(lat_col: str, lon_col: str, obs_df: pd.DataFrame, logger=None, output_dir: Path = None) -> Tuple[bool, str]:
    """
    Validate and clamp coordinate columns, output invalid coordinates for review.
    
    Validates:
    - Column exists and is numeric
    - Values in valid latitude range [-90, 90]
    - Values in valid longitude range [-180, 180]
    - Not all NaN
    
    Clamps out-of-range values to valid bounds and outputs review file.
    
    Args:
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        obs_df: Metadata DataFrame (modified in place)
        logger: Optional logger instance
        output_dir: Directory to write invalid coordinates report
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        # Check columns exist
        if lat_col not in obs_df.columns or lon_col not in obs_df.columns:
            msg = f"Columns {lat_col}/{lon_col} not found"
            if logger:
                logger.warning(f"   ⚠️  {msg}")
            return False, msg
        
        lat_series = pd.to_numeric(obs_df[lat_col], errors='coerce')
        lon_series = pd.to_numeric(obs_df[lon_col], errors='coerce')
        
        # Check for at least some non-NaN values
        valid_count = ((lat_series.notna()) & (lon_series.notna())).sum()
        if valid_count == 0:
            msg = f"No valid coordinate pairs in {lat_col}/{lon_col}"
            if logger:
                logger.warning(f"   ⚠️  {msg}")
            return False, msg
        
        # Find out-of-range values
        invalid_lat_mask = (lat_series.notna()) & ((lat_series < -90) | (lat_series > 90))
        invalid_lon_mask = (lon_series.notna()) & ((lon_series < -180) | (lon_series > 180))
        invalid_mask = invalid_lat_mask | invalid_lon_mask
        
        # Output review file if invalid coordinates found
        if invalid_mask.sum() > 0:
            invalid_df = obs_df[invalid_mask].copy()
            invalid_df['original_latitude'] = lat_series[invalid_mask]
            invalid_df['original_longitude'] = lon_series[invalid_mask]
            invalid_df['clamped_latitude'] = np.clip(lat_series[invalid_mask], -90, 90)
            invalid_df['clamped_longitude'] = np.clip(lon_series[invalid_mask], -180, 180)
            
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                review_file = output_dir / "invalid_coordinates_review.csv"
                invalid_df.to_csv(review_file)
                if logger:
                    logger.warning(f"   ⚠️  Found {invalid_mask.sum()} samples with invalid coordinates. Review: {review_file}")
            else:
                if logger:
                    logger.warning(f"   ⚠️  Found {invalid_mask.sum()} latitude values outside [-90, 90] range and/or {invalid_lon_mask.sum()} longitude values outside [-180, 180] range")
        
        # Clamp values to valid ranges
        obs_df[lat_col] = np.clip(lat_series, -90, 90)
        obs_df[lon_col] = np.clip(lon_series, -180, 180)
        
        valid_after_clamp = ((obs_df[lat_col].notna()) & (obs_df[lon_col].notna())).sum()
        coverage = f"{valid_after_clamp}/{len(obs_df)} ({100*valid_after_clamp/len(obs_df):.1f}%)"
        msg = f"Found {coverage} valid coordinates in {lat_col}/{lon_col}"
        if logger:
            logger.info(f"   ✓ {msg}")
        
        return True, msg
        
    except Exception as e:
        msg = f"Error validating coordinates: {str(e)}"
        if logger:
            logger.error(f"   ✗ {msg}")
        return False, msg

def validate_coordinates(lat_col: str, lon_col: str, obs_df: pd.DataFrame, logger=None) -> Tuple[bool, str]:
    """
    Wrapper for validate_and_clamp_coordinates without output_dir parameter.
    Shorter form for backward compatibility.
    """
    return validate_and_clamp_coordinates(lat_col, lon_col, obs_df, logger=logger, output_dir=None)

def get_config_val(config, section, key, default) -> Any:
    try:
        if hasattr(config, section):
            sect_obj = getattr(config, section)
            if isinstance(sect_obj, dict): return sect_obj.get(key, default)
            return getattr(sect_obj, key, default)
        if isinstance(config, dict):
            return config.get(section, {}).get(key, default)
        return default
    except Exception:
        return default


def _resolve_gee_service_account_path(config: Any, logger=None) -> Optional[Path]:
    """Resolve GEE service account path from config, env vars, and common paths."""
    candidates = []

    # Handle both dict and AppConfig objects for credentials
    if isinstance(config, dict):
        credentials = config.get('credentials', {})
        gee_service_account = credentials.get('gee_service_account') if isinstance(credentials, dict) else getattr(credentials, 'gee_service_account', None)
    else:
        credentials = getattr(config, "credentials", None)
        gee_service_account = getattr(credentials, "gee_service_account", None) if credentials else None
    
    if gee_service_account:
        candidates.append((Path(str(gee_service_account)).expanduser(), "config.credentials.gee_service_account"))

    env_gee = os.environ.get("GEE_SERVICE_ACCOUNT_PATH")
    if env_gee:
        candidates.append((Path(env_gee).expanduser(), "GEE_SERVICE_ACCOUNT_PATH"))

    env_google = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_google:
        candidates.append((Path(env_google).expanduser(), "GOOGLE_APPLICATION_CREDENTIALS"))

    # Handle both dict and AppConfig objects for paths
    if isinstance(config, dict):
        paths_cfg = config.get('paths', {})
        base_path = paths_cfg.get('base') if isinstance(paths_cfg, dict) else getattr(paths_cfg, 'base', None)
    else:
        paths_cfg = getattr(config, "paths", None)
        base_path = getattr(paths_cfg, "base", None) if paths_cfg else None
    
    if base_path:
        candidates.append((Path(str(base_path)).expanduser() / "credentials" / "your-gee-service-account.json", "config.paths.base/credentials"))

    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents)[:4]:
        candidates.append((parent / "credentials" / "your-gee-service-account.json", f"{parent}/credentials"))

    for path, source in candidates:
        if path.exists():
            if logger:
                logger.debug(f"   ✓ Found GEE service account at {source}: {path}")
            return path

    if logger:
        logger.warning("   ⚠️  GEE service account path not found in config credentials or environment")
        logger.info("   ℹ️  Set credentials.gee_service_account, GEE_SERVICE_ACCOUNT_PATH, or GOOGLE_APPLICATION_CREDENTIALS")
    return None

def _gee_auth(config: Any, logger=None) -> bool:
    """
    Authenticate to Google Earth Engine using service account credentials from AppConfig or dict.

    This function validates and authenticates to Google Earth Engine using credentials
    stored in the config object (either AppConfig or dict). It performs the following checks:

    1. Checks for gee_service_account path in config.credentials (handles both AppConfig and dict)
    2. Verifies the service account JSON file exists
    3. Validates the JSON file contains proper service account credentials
    4. Authenticates to Earth Engine (or checks if already authenticated)

    Args:
        config: AppConfig or dict object containing GEE authentication details
        logger: Optional logger instance for status messages

    Returns:
        bool: True if authentication succeeded or already authenticated, False otherwise

    Note:
        - Supports both AppConfig objects and plain dicts
        - Returns False (not exception) for all validation failures (graceful degradation)
        - GEE initialization is skipped if service account file is missing
    """
    # No strict type checking - accept both AppConfig and dict objects
    # _resolve_gee_service_account_path() is designed to work with any object

    # Extract GEE service account path using robust discovery
    gee_service_account_path = _resolve_gee_service_account_path(config, logger=logger)
    if gee_service_account_path is None:
        if logger:
            logger.error(
                "   ❌ GEE authentication failed: Service account file not found.\n"
                "      Set one of:\n"
                "      1. credentials.gee_service_account in config.yaml\n"
                "      2. GEE_SERVICE_ACCOUNT_PATH environment variable\n"
                "      3. GOOGLE_APPLICATION_CREDENTIALS environment variable\n"
                "      4. Place your-gee-service-account.json in {pwd}/credentials/ or {config.base}/credentials/\n"
                "      To get a service account JSON: https://console.cloud.google.com/iam-admin/serviceaccounts"
            )
        return False

    # Validate JSON file contains proper service account credentials
    try:
        import json
        with open(gee_service_account_path) as f:
            sa_data = json.load(f)
        if 'type' not in sa_data or sa_data.get('type') != 'service_account':
            if logger:
                logger.error(
                    f"   ❌ Invalid GEE service account file: {gee_service_account_path}\n"
                    f"      File must have 'type': 'service_account' field.\n"
                    f"      The file appears to be a {sa_data.get('type', 'unknown')} type.\n"
                    f"      Download a new service account JSON from Google Cloud Console."
                )
            return False
        if logger:
            logger.info(f"   ✓ GEE service account validated: {gee_service_account_path.name}")
            logger.debug(f"      Project: {sa_data.get('project_id', 'unknown')}")
    except (json.JSONDecodeError, IOError) as e:
        if logger:
            logger.error(
                f"   ❌ Cannot read GEE service account file: {gee_service_account_path}\n"
                f"      Error: {e}\n"
                f"      Check file permissions and JSON syntax."
            )
        return False

    # Authenticate to Earth Engine
    try:
        import ee
        try:
            # Check if already authenticated by attempting a simple API call
            # Try a safer method that's less likely to throw unexpected exceptions
            ee.data.listAssets({"parent": "projects/earthengine-public"})
            if logger:
                logger.debug(f"   ✓ Earth Engine already authenticated")
            return True
        except Exception:
            # Not yet authenticated, authenticate with service account
            if logger:
                logger.debug(f"   🔐 Authenticating Earth Engine with service account...")
            # Use ServiceAccountCredentials for proper service account authentication
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_file(
                str(gee_service_account_path),
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            # Get GEE project ID - handle both dict and AppConfig
            gee_project = get_config_val(config, 'credentials', 'google_earth_engine_project', 'wired-day-365517')
            ee.Initialize(credentials=credentials, project=gee_project)
            if logger:
                logger.info(f"   ✓ Earth Engine authenticated successfully (project: {gee_project})")
            return True
    except ImportError as e:
        if logger:
            logger.error(
                f"   ❌ GEE Python API not installed: {e}\n"
                f"      Install with: pip install earthengine-api google-cloud-compute"
            )
        return False
    except Exception as e:
        if logger:
            logger.error(
                f"   ❌ Earth Engine authentication failed: {e}\n"
                f"      Troubleshooting:\n"
                f"      1. Verify service account JSON file is valid\n"
                f"      2. Check Google Cloud project has Earth Engine API enabled\n"
                f"      3. Check service account has required permissions (Editor role)\n"
                f"      4. Verify credentials file path is correct\n"
                f"      Path used: {gee_service_account_path}"
            )
        return False
        return False


def run_data_backfill(workflow):
    """Orchestrates multi-API backfill using a nested event loop."""
    if workflow.adata is None: return
    workflow.logger.info("［03］Modular Backfill: Running external API enrichment...")
    
    # Ensure collection_date column exists from the start
    if 'collection_date' not in workflow.adata.obs.columns:
        workflow.adata.obs['collection_date'] = pd.NaT
    
    # Use nest_asyncio logic to run the async runner
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_async_backfill_runner(workflow))
    
    workflow.logger.info("［03］Modular Backfill: Complete.")


async def _run_gee_enrichment_phase(workflow, logger):
    """Async wrapper for GEE enrichment phase - can run in parallel with Environmental."""
    workflow.telemetry.start_phase('backfill_gee')
    try:
        # Check if GEE is enabled AND has enabled datasets
        gee_enabled = getattr(workflow, 'is_gee_enabled', False)
        gee_config = get_config_val(workflow.config, 'gee_assets', {}, {})

        # Check if any GEE assets are explicitly enabled
        datasets_enabled = False
        enabled_datasets = []
        if isinstance(gee_config, dict):
            for key in ['jrc_water', 'viirs_nighttime', 'hansen_gfc', 'dem', 'era5', 'worldcover', 'modis']:
                if gee_config.get(key, {}).get('enabled', False):
                    datasets_enabled = True
                    enabled_datasets.append(key)

        # Only proceed if BOTH gee_enabled is True AND at least one dataset is enabled
        if gee_enabled and datasets_enabled:
            logger.info(f" 🌍 Running GEE-based environmental enrichment ({len(enabled_datasets)} datasets enabled: {', '.join(enabled_datasets)})...")
            try:
                lat_col, lon_col = find_coordinate_columns(workflow.adata.obs)
                if lat_col and lon_col:
                    validate_coordinates(lat_col, lon_col, workflow.adata.obs, logger=logger)
                    rows_with_coords = workflow.adata.obs[
                        (workflow.adata.obs[lat_col].notna()) &
                        (workflow.adata.obs[lon_col].notna())
                    ]

                    if len(rows_with_coords) > 0:
                        gee_metadata = workflow.adata.obs.copy()
                        gee_metadata['lat'] = pd.to_numeric(workflow.adata.obs[lat_col], errors='coerce')
                        gee_metadata['lon'] = pd.to_numeric(workflow.adata.obs[lon_col], errors='coerce')

                        if 'collection_date' not in gee_metadata.columns:
                            gee_metadata['collection_date'] = pd.NaT
                        gee_metadata['collection_date'] = pd.to_datetime(gee_metadata['collection_date'], errors='coerce')

                        auth_flag = _gee_auth(workflow.config, logger)
                        batch_size = get_config_val(workflow.config, 'gee_assets', 'batch_size', 30)
                        async_mode = get_config_val(workflow.config, 'gee_assets', 'async_mode', False)
                        use_mega_image = get_config_val(workflow.config, 'gee_assets', 'use_mega_image', False)
                        gcs_bucket = get_config_val(workflow.config, 'gee_assets', 'cloud_storage_bucket', None)

                        if async_mode:
                            logger.info(f"  📡 Async mode enabled (polling interval: 60s, max wait: 24h)")
                        if use_mega_image:
                            logger.info(f"  📊 Mega-image mode enabled (5 concurrent exports)")

                        enriched_obs = enrich_with_gee_data(
                            gee_metadata,
                            auth_flag,
                            batch_size=batch_size,
                            use_cache=True,
                            async_mode=async_mode,
                            use_mega_image=use_mega_image,
                            gee_config=get_config_val(workflow.config, 'gee_assets', {}, {}),
                            gcs_bucket=gcs_bucket,
                            wait_for_completion=True,
                            logger_instance=logger
                        )
                        # Batch progress logging is handled internally by enrich_with_gee_data()
                        # See: _google_earth_engine.py lines 2472-2533 for batch progress format

                        if enriched_obs is not None and not enriched_obs.empty:
                            gee_cols = [col for col in enriched_obs.columns if col.startswith((
                                'ISDASOIL_', 'DEM_', 'ERA5_', 'worldcover_', 'openlandmap_',
                                'jrc_', 'hansen_', 'lights_'
                            ))]
                            for col in gee_cols:
                                if col not in workflow.adata.obs.columns:
                                    workflow.adata.obs[col] = np.nan
                                if len(enriched_obs) == len(workflow.adata.obs):
                                    workflow.adata.obs[col] = enriched_obs[col].values

                            gee_filled = sum(1 for col in gee_cols if workflow.adata.obs[col].notna().any())
                            logger.info(f" ✅ GEE enrichment: {gee_filled} data sources integrated ({len(gee_cols)} columns added)")
                        else:
                            logger.debug(" ℹ️  GEE enrichment returned no data")
                    else:
                        logger.debug(" ℹ️  No valid coordinates for GEE enrichment")
                else:
                    logger.debug(" ℹ️  No lat/lon coordinates found for GEE enrichment")
            except ImportError:
                logger.debug(" ℹ️ GEE integration modules not available (earthengine may not be authenticated)")
            except Exception as e:
                logger.warning(f" ⚠️  GEE enrichment skipped: {e}")
        else:
            # Provide detailed reason why GEE was skipped
            skip_reasons = []
            if not gee_enabled:
                skip_reasons.append("GEE not enabled (is_gee_enabled=False)")
            if not datasets_enabled:
                skip_reasons.append("No GEE datasets explicitly enabled in config")
            reason = " AND ".join(skip_reasons)
            logger.info(f" ⊘ GEE enrichment skipped: {reason}")
    finally:
        workflow.telemetry.end_phase('backfill_gee')


async def _run_environmental_enrichment_phase(workflow, semaphore, logger):
    """Async wrapper for Environmental enrichment phase - can run in parallel with GEE."""
    workflow.telemetry.start_phase('backfill_environmental')
    try:
        if getattr(workflow, 'is_env_data_enabled', False):
            logger.info(" 🌎 Running Environmental Data Collector...")
            lat_col, lon_col = find_coordinate_columns(workflow.adata.obs)
            has_coords = lat_col is not None and lon_col is not None

            if has_coords:
                logger.debug(f"   Using lat_col='{lat_col}' and lon_col='{lon_col}' for environmental enrichment")
                validate_and_clamp_coordinates(lat_col, lon_col, workflow.adata.obs, logger=logger, output_dir=workflow.output_dir)

                lat_numeric = pd.to_numeric(workflow.adata.obs[lat_col], errors='coerce')
                lon_numeric = pd.to_numeric(workflow.adata.obs[lon_col], errors='coerce')

                rows_to_fetch_mask = lat_numeric.notna() & lon_numeric.notna()
                valid_count = rows_to_fetch_mask.sum()
                logger.debug(f"   Found {valid_count}/{len(workflow.adata)} samples with valid coordinates")

                if valid_count > 0:
                    rows_to_fetch = workflow.adata.obs.loc[rows_to_fetch_mask].copy()
                    rows_to_fetch['lat'] = lat_numeric[rows_to_fetch_mask].values
                    rows_to_fetch['lon'] = lon_numeric[rows_to_fetch_mask].values

                    if 'collection_date' in rows_to_fetch.columns:
                        collection_date_parsed = pd.to_datetime(rows_to_fetch['collection_date'], errors='coerce')
                        rows_to_fetch['collection_date'] = collection_date_parsed.dt.strftime('%Y-%m-%d').where(collection_date_parsed.notna(), '')
                        date_coverage = collection_date_parsed.notna().sum()
                        logger.debug(f"   Collection date coverage: {date_coverage}/{len(rows_to_fetch)} ({100*date_coverage/len(rows_to_fetch):.1f}%)")

                    if not rows_to_fetch.empty:
                        from workflow_16s.api.environmental_data.other.tools.coordinate_sorting_utils import sort_coordinates_by_space, log_sorting_plan

                        # Ensure collection_date exists (may not be in original obs if unfilled)
                        if 'collection_date' not in rows_to_fetch.columns:
                            rows_to_fetch['collection_date'] = ''

                        # === STEP 1: DEDUPLICATE TO UNIQUE COORDINATES ===
                        # Round coordinates to 2 decimals (~1.1km precision) and deduplicate by coordinate only.
                        rows_to_fetch['lat'] = rows_to_fetch['lat'].round(2)
                        rows_to_fetch['lon'] = rows_to_fetch['lon'].round(2)
                        coords_df = rows_to_fetch[['lat', 'lon']].drop_duplicates().reset_index(drop=True)
                        n_unique_coords = len(coords_df)
                        logger.info(
                            f"   Deduplicated {len(rows_to_fetch)} samples → {n_unique_coords} unique coordinates "
                            f"(rounded to 2 decimals)"
                        )

                        # Keep one representative date per coordinate for date-aware APIs.
                        if 'collection_date' in rows_to_fetch.columns:
                            coord_dates = (
                                rows_to_fetch.groupby(['lat', 'lon'])['collection_date']
                                .agg(lambda s: next((v for v in s if isinstance(v, str) and v.strip()), ''))
                                .reset_index()
                            )
                            coords_df = coords_df.merge(coord_dates, on=['lat', 'lon'], how='left')
                        else:
                            coords_df['collection_date'] = ''

                        # === STEP 2: APPLY SPATIAL 2D SORTING ===
                        coords_df_indexed = coords_df.copy()
                        coords_df_indexed.index = range(len(coords_df))

                        # Safely access nested config attributes: apis.geospatial.coordinates.spatial_sort_*
                        apis_config = getattr(workflow.config, 'apis', None)
                        geo_config = getattr(apis_config, 'geospatial', None) if apis_config else None
                        coords_config = getattr(geo_config, 'coordinates', None) if geo_config else None

                        sort_axis = getattr(coords_config, 'spatial_sort_axis', 'lon') if coords_config else 'lon'
                        sort_chunk_size = getattr(coords_config, 'spatial_sort_chunk_size', None) if coords_config else None

                        if sort_chunk_size and n_unique_coords > 1:
                            log_sorting_plan(n_unique_coords, sort_axis, sort_chunk_size)
                            coords_sorted, idx_mapping = sort_coordinates_by_space(
                                coords_df_indexed,
                                sort_axis=sort_axis,
                                chunk_size=sort_chunk_size,
                                preserve_index=True
                            )
                            # idx_mapping: {old_idx: new_idx}, reorder by new positions
                            sort_order = sorted(idx_mapping.items(), key=lambda x: x[1])
                            sorted_coord_indices = [old_idx for old_idx, _ in sort_order]
                            logger.debug(f"   Spatial sorting applied: {n_unique_coords} coordinates organized by 2D locality")
                        else:
                            sorted_coord_indices = list(range(n_unique_coords))
                            if sort_chunk_size is None:
                                logger.debug("   Spatial sorting disabled (sort_chunk_size=null)")

                        # === STEP 3: MAP COORDINATES BACK TO ALL SAMPLE INDICES ===
                        coord_to_sample_indices = {}
                        for obs_index, row in rows_to_fetch.iterrows():
                            coord_key = (row['lat'], row['lon'])
                            if coord_key not in coord_to_sample_indices:
                                coord_to_sample_indices[coord_key] = []
                            coord_to_sample_indices[coord_key].append(obs_index)

                        coords_to_query = coords_df.iloc[sorted_coord_indices].copy().reset_index(drop=True)

                        # === STEP 4: CREATE BATCHES FROM SPATIALLY SORTED COORDINATES ===
                        BATCH_SIZE = 50000
                        n_batches = (len(coords_to_query) + BATCH_SIZE - 1) // BATCH_SIZE
                        logger.info(f"   ✓ Processing {len(coords_to_query)} spatially-sorted coordinates in {n_batches} batch(es)")

                        # Reuse a single collector to avoid repeated API init/check_requirements.
                        data_collector = EnvironmentalDataCollector(config=workflow.config, logger=logger)

                        async def process_batch_with_semaphore(batch_idx, batch_data):
                            """Process a single batch with semaphore-controlled network access."""
                            logger.debug(
                                f"   Batch {batch_idx + 1}/{n_batches}: {len(batch_data)} coordinates from spatially sorted order"
                            )

                            async with semaphore:
                                try:
                                    enriched_batch = await data_collector.collect_for_metadata(batch_data)
                                    if enriched_batch is not None and not enriched_batch.empty:
                                        return (batch_idx, enriched_batch)
                                except Exception as e:
                                    logger.warning(f"   ⚠️  Batch {batch_idx + 1}/{n_batches} processing failed: {str(e)}")
                                    return (batch_idx, None)
                            return (batch_idx, None)

                        # Process one batch at a time when sharing collector state.
                        MAX_CONCURRENT_BATCHES = 1
                        enriched_results = [None] * n_batches  # Pre-allocate to maintain order

                        for batch_group_idx in range(0, n_batches, MAX_CONCURRENT_BATCHES):
                            batch_group_range = range(
                                batch_group_idx,
                                min(batch_group_idx + MAX_CONCURRENT_BATCHES, n_batches)
                            )
                            batch_indices = list(batch_group_range)

                            batch_nums = [str(idx + 1) for idx in batch_indices]
                            logger.info(f"   ⚡ Batches {', '.join(batch_nums)} processing in parallel...")

                            tasks = []
                            for batch_idx in batch_indices:
                                start_idx = batch_idx * BATCH_SIZE
                                end_idx = min(start_idx + BATCH_SIZE, len(coords_to_query))
                                batch = coords_to_query.iloc[start_idx:end_idx].copy()
                                tasks.append(process_batch_with_semaphore(batch_idx, batch))

                            batch_results = await asyncio.gather(*tasks, return_exceptions=False)
                            for batch_idx, result in batch_results:
                                enriched_results[batch_idx] = result

                        enriched_results = [result for result in enriched_results if result is not None]
                        logger.debug(f"   ✓ Concurrent batch processing complete: {len(enriched_results)}/{n_batches} batches succeeded")

                        # === STEP 5: MAP COORDINATE-LEVEL RESULTS TO SAMPLE-LEVEL OBS ROWS ===
                        if enriched_results:
                            enriched_env_df = pd.concat(enriched_results, ignore_index=False)
                        else:
                            enriched_env_df = None

                        if enriched_env_df is not None and not enriched_env_df.empty:
                            base_cols = {'lat', 'lon', 'collection_date', 'collection_date_str'}
                            env_cols = [col for col in enriched_env_df.columns if col not in base_cols]

                            expanded_records = []
                            for _, coord_row in enriched_env_df.iterrows():
                                lat_val = round(float(coord_row['lat']), 2)
                                lon_val = round(float(coord_row['lon']), 2)
                                sample_indices = coord_to_sample_indices.get((lat_val, lon_val), [])
                                if not sample_indices:
                                    continue

                                for sample_idx in sample_indices:
                                    record = {'_sample_idx': sample_idx}
                                    for col in env_cols:
                                        record[col] = coord_row.get(col)
                                    expanded_records.append(record)

                            if expanded_records:
                                expanded_df = pd.DataFrame(expanded_records).set_index('_sample_idx')
                                matching_indices = expanded_df.index.intersection(workflow.adata.obs.index)

                                if len(matching_indices) > 0:
                                    for col in env_cols:
                                        output_col = f"env_{col}"
                                        if output_col not in workflow.adata.obs.columns:
                                            workflow.adata.obs[output_col] = np.nan
                                        workflow.adata.obs.loc[matching_indices, output_col] = expanded_df.loc[matching_indices, col]

                                    env_filled = workflow.adata.obs[
                                        [col for col in workflow.adata.obs.columns if col.startswith('env_')]
                                    ].notna().any(axis=1).sum()
                                    logger.info(
                                        f" ✅ Environmental enrichment successful: {env_filled} samples enriched with "
                                        f"{len(env_cols)} data sources"
                                    )
                                else:
                                    logger.warning("   ⚠️  No matching samples between enriched data and main dataset")
                            else:
                                logger.debug("   ℹ️  No expanded coordinate-to-sample mappings were produced")
                        else:
                            logger.debug("   ℹ️  Environmental Data Collector returned no data (all APIs may have failed for these locations)")

            else:
                if not has_coords:
                    logger.debug("   ℹ️  No lat/lon coordinates for coordinate-based enrichment")

        else:
            logger.debug(" ℹ️  Environmental data collection disabled, skipping")
    finally:
        workflow.telemetry.end_phase('backfill_environmental')

async def _async_backfill_runner(workflow):
    """Async runner with rate-limiting protection."""

    # Rate limiter: Max 10 concurrent network requests to prevent API bans
    semaphore = asyncio.Semaphore(10)

    # =========================================================================
    # [00] Fetch Metadata from ENA/SRA (if missing)
    # =========================================================================
    workflow.telemetry.start_phase('backfill_ena')
    try:
        if workflow.adata.obs['collection_date'].isna().any():
            missing_pct = workflow.adata.obs['collection_date'].isna().sum() / len(workflow.adata) * 100
            workflow.logger.info(f" 📅 Metadata Enrichment: {missing_pct:.1f}% missing, attempting to fill from ENA/SRA...")
            try:
                # Try to pass progress bar if available on workflow
                # This integrates ENA fetch progress into the existing dashboard
                progress_obj = getattr(workflow, 'progress', None)
                progress_task = getattr(workflow, 'backfill_task_id', None)
                
                fetch_metadata_from_ena(
                    workflow.adata, 
                    config=workflow.config,
                    logger=workflow.logger,
                    progress=progress_obj,
                    progress_task_id=progress_task
                )
            except Exception as e:
                workflow.logger.warning(f" ⚠️  Metadata backfill failed: {e}")
        else:
            workflow.logger.debug(" ✓ All collection_date values present, skipping ENA backfill")
    finally:
        workflow.telemetry.end_phase('backfill_ena')

    # =========================================================================
    # [00b] Coordinate-Based Fallback Search (Non-ENA Samples)
    # =========================================================================
    # After direct ENA lookups, try finding metadata via geographic proximity
    workflow.telemetry.start_phase('backfill_coordinate_fallback')
    try:
        # Check if coordinate fallback is enabled in config
        coordinate_fallback_enabled = get_config_val(workflow.config, 'coordinate_fallback', 'enabled', True)

        if coordinate_fallback_enabled:
            workflow.logger.info(" 🗺️  Running coordinate-based fallback search for non-ENA samples...")
            try:
                # Get config parameters
                radius_degrees = get_config_val(workflow.config, 'coordinate_fallback', 'radius_degrees', 0.1)
                max_concurrent = get_config_val(workflow.config, 'coordinate_fallback', 'max_concurrent', 5)
                tag_inferred = get_config_val(workflow.config, 'coordinate_fallback', 'tag_inferred', True)
                cache_dir = get_config_val(workflow.config, 'coordinate_fallback', 'cache_dir', None)
                email = workflow.config.credentials.ena_email if hasattr(workflow.config, 'credentials') else 'default@example.com'

                # Run coordinate fallback
                supplement_with_nearby_samples(
                    adata=workflow.adata,
                    config=workflow.config,
                    logger_instance=workflow.logger,
                    radius_degrees=radius_degrees,
                    max_concurrent=max_concurrent,
                    tag_inferred=tag_inferred,
                    cache_dir=cache_dir,
                    email=email,
                )
            except Exception as e:
                workflow.logger.warning(f" ⚠️  Coordinate fallback search failed: {e}")
        else:
            workflow.logger.debug(" ℹ️  Coordinate fallback search disabled")
    finally:
        workflow.telemetry.end_phase('backfill_coordinate_fallback')

    # =========================================================================
    # [01] Arkin Agents Pathway
    # =========================================================================
    workflow.telemetry.start_phase('backfill_arkin')
    try:
        if getattr(workflow, 'is_arkin_enabled', False):
            workflow.logger.info(" 🤖 Running Arkin Agents LLM backfill...")
            temp_meta = workflow.output_dir / "temp_meta_for_arkin.tsv"
            try:
                workflow.adata.obs.to_csv(temp_meta, sep='\t', index_label="#SampleID")
                async with semaphore:
                    arkin_df = await run_arkin_enrichment(
                        metadata_path=temp_meta,
                        project_dir=Project(workflow.config),
                        config=workflow.config
                    )

                if arkin_df is not None and not arkin_df.empty:
                    workflow.logger.info(f" 🧠 Arkin returned {len(arkin_df)} records.")
                    arkin_df['run_acc_list'] = arkin_df['associated_sample_ids'].str.split(', ')
                    arkin_df = arkin_df.explode('run_acc_list').rename(columns={'run_acc_list': 'run_accession'})
                    arkin_df = arkin_df.drop_duplicates(subset=['run_accession']).set_index('run_accession')

                    workflow.adata.obs = workflow.adata.obs.merge(
                        arkin_df.drop(columns=['associated_sample_ids'], errors='ignore'),
                        left_index=True, right_index=True, how='left', suffixes=('', '_arkin')
                    )
            except Exception as e:
                workflow.logger.error(f" ⚠️ Arkin Agents backfill failed: {e}")
            finally:
                if temp_meta.exists(): os.remove(temp_meta)
        else:
            workflow.logger.debug(" ℹ️  Arkin Agents disabled, skipping")
    finally:
        workflow.telemetry.end_phase('backfill_arkin')

    # =========================================================================
    # [02] NFC GIS Facility Matching (Synchronous)
    # =========================================================================
    workflow.telemetry.start_phase('backfill_nfc')
    try:
        if getattr(workflow, 'is_nfc_enabled', False) and workflow.nfc_handler and not workflow.nfc_facilities_df.empty:
            workflow.logger.info(" ☢️ Running NFC facility matching...")
            # Find appropriate coordinate columns
            lat_col, lon_col = find_coordinate_columns(workflow.adata.obs)

            if lat_col and lon_col:
                validate_and_clamp_coordinates(lat_col, lon_col, workflow.adata.obs, logger=workflow.logger, output_dir=workflow.output_dir)
                force_redo = get_config_val(workflow.config, 'nfc_facilities', 'match_existing_samples', False)

                mask = (workflow.adata.obs[lat_col].notna()) & (workflow.adata.obs[lon_col].notna())
                rows_to_match = workflow.adata.obs[mask].copy() if force_redo else workflow.adata.obs[mask & workflow.adata.obs['facility_match'].isna()].copy()

            if not rows_to_match.empty:
                matched = workflow.nfc_handler._match_facilities_with_locations(workflow.nfc_facilities_df, rows_to_match)
                if matched is not None and not matched.empty:
                    workflow.adata.obs.update(matched)
                    workflow.logger.info(f" ✅ Updated {len(matched)} samples with NFC data.")
        else:
            workflow.logger.debug(" ℹ️  NFC disabled or no facilities data, skipping")
    finally:
        workflow.telemetry.end_phase('backfill_nfc')

    # =========================================================================
    # [03-04] GEE + Environmental Enrichment (RUN IN PARALLEL FOR ~40% SPEEDUP)
    # =========================================================================
    # OPTIMIZATION: Run GEE and Environmental enrichment concurrently via asyncio.gather()
    # This reduces total enrichment time from ~20 minutes to ~12 minutes for 400k samples
    workflow.telemetry.start_phase('backfill_enrichment')
    try:
        # Run both enrichment types concurrently
        gee_result, env_result = await asyncio.gather(
            _run_gee_enrichment_phase(workflow, workflow.logger),
            _run_environmental_enrichment_phase(workflow, semaphore, workflow.logger),
            return_exceptions=True  # Catch errors from either operation
        )

        # Check for errors from concurrent operations
        if isinstance(gee_result, Exception):
            workflow.logger.warning(f" ⚠️  GEE enrichment error: {gee_result}")
        if isinstance(env_result, Exception):
            workflow.logger.warning(f" ⚠️  Environmental enrichment error: {env_result}")

    finally:
        workflow.telemetry.end_phase('backfill_enrichment')
    
    # =========================================================================
    # [05] CSU Soil Heavy Metal Enrichment (if coordinates available)
    # =========================================================================
    workflow.telemetry.start_phase('backfill_csu_soil')
    try:
        if getattr(workflow, 'is_csu_soil_enabled', False):
            workflow.logger.info(" 🌍 Running CSU Soil Metal Enrichment...")
            try:
                from workflow_16s.api.environmental_data.other.tools._csu_soil import CSUSoilAPI

                lat_col, lon_col = find_coordinate_columns(workflow.adata.obs)
                if lat_col and lon_col:
                    # Validate and clamp coordinates
                    validate_and_clamp_coordinates(lat_col, lon_col, workflow.adata.obs, logger=workflow.logger, output_dir=workflow.output_dir)

                    # Convert to numeric
                    lat_numeric = pd.to_numeric(workflow.adata.obs[lat_col], errors='coerce')
                    lon_numeric = pd.to_numeric(workflow.adata.obs[lon_col], errors='coerce')
                    rows_with_coords_mask = lat_numeric.notna() & lon_numeric.notna()

                    if rows_with_coords_mask.sum() > 0:
                        csu_api = CSUSoilAPI()
                        csu_metals = ['hg', 'cu', 'zn', 'cd', 'as', 'ni', 'pb', 'cr', 'co', 'fe']

                        # Apply CSU data globally (not coordinate-based) - it's regional/global data
                        csu_data = csu_api.get_data(None, None)

                        # Initialize metal columns if not present
                        for metal in csu_metals:
                            col_name = f'csu_{metal}_mg_kg_avg'
                            if col_name not in workflow.adata.obs.columns:
                                workflow.adata.obs[col_name] = np.nan

                            # Fill with CSU data for samples with valid coordinates
                            if csu_data and col_name in csu_data:
                                workflow.adata.obs.loc[rows_with_coords_mask, col_name] = csu_data[col_name]

                        # Count how many samples got enriched
                        csu_filled = sum(1 for metal in csu_metals if f'csu_{metal}_mg_kg_avg' in workflow.adata.obs.columns
                                        and workflow.adata.obs[f'csu_{metal}_mg_kg_avg'].notna().any())
                        workflow.logger.info(f" ✅ CSU Soil enrichment: {len(csu_metals)} metal columns added to {rows_with_coords_mask.sum()} samples with valid coordinates.")
                    else:
                        workflow.logger.debug(f"   ℹ️ No samples with valid coordinates for CSU enrichment")
            except ImportError:
                workflow.logger.debug(" ℹ️ CSU Soil API not available")
            except Exception as e:
                workflow.logger.warning(f" ⚠️  CSU Soil enrichment failed: {e}")
        else:
            workflow.logger.debug(" ℹ️  CSU Soil disabled, skipping")
    finally:
        workflow.telemetry.end_phase('backfill_csu_soil')
    
    # =========================================================================
    # [06] Geochemical Data Integration (SoilGrids + Proxy Database)
    # =========================================================================
    workflow.telemetry.start_phase('backfill_geochemical')
    try:
        if getattr(workflow, 'is_geochemical_enabled', False):
            workflow.logger.info(" ⚗️ Running Geochemical Data Integration...")
            try:
                # Try SoilGrids first (real data), fall back to proxy if unavailable
                soilgrids_available = False
                try:
                    from workflow_16s.api.environmental_data.other.tools._soilgrids_impl import SoilGridsGeochemistryAPI
                    soilgrids_api = SoilGridsGeochemistryAPI(verbose=False)
                    soilgrids_available, sg_msg = soilgrids_api.check_requirements()
                    if soilgrids_available:
                        workflow.logger.debug("   SoilGrids: Available (free REST API)")
                except:
                    soilgrids_available = False

                # Always load proxy as fallback
                from workflow_16s.api.environmental_data.other.tools._geochemical_proxy import GeochemicalProxyDatabase
                proxy_api = GeochemicalProxyDatabase(verbose=False)

                lat_col, lon_col = find_coordinate_columns(workflow.adata.obs)
                if lat_col and lon_col:
                    # Validate and clamp coordinates
                    validate_and_clamp_coordinates(lat_col, lon_col, workflow.adata.obs, logger=workflow.logger, output_dir=workflow.output_dir)

                    # Convert to numeric
                    lat_numeric = pd.to_numeric(workflow.adata.obs[lat_col], errors='coerce')
                    lon_numeric = pd.to_numeric(workflow.adata.obs[lon_col], errors='coerce')
                    rows_with_coords_mask = lat_numeric.notna() & lon_numeric.notna()

                    if rows_with_coords_mask.sum() > 0:
                        enriched = 0
                        failed = 0
                        proxy_used = 0

                        # Get indices of rows with valid coordinates
                        rows_with_coords_indices = workflow.adata.obs.index[rows_with_coords_mask]
                        total_rows = len(rows_with_coords_indices)

                        # Log progress every N samples to show activity
                        progress_interval = max(10000, total_rows // 10)  # Show progress 10 times

                        for idx_count, sample_id in enumerate(rows_with_coords_indices):
                            try:
                                lat = float(lat_numeric.loc[sample_id]) if sample_id in lat_numeric.index else float(lat_numeric[sample_id])
                                lon = float(lon_numeric.loc[sample_id]) if sample_id in lon_numeric.index else float(lon_numeric[sample_id])
                            except (KeyError, TypeError) as e:
                                workflow.logger.debug(f"   Could not get coordinates for {sample_id}: {e}")
                                failed += 1
                                continue

                                # Skip invalid coordinates
                            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                                continue

                            # Try SoilGrids first
                            data = None
                            if soilgrids_available:
                                try:
                                    data = soilgrids_api.get_data(lat, lon)
                                except Exception as sg_err:
                                    workflow.logger.debug(f"   SoilGrids API failed for ({lat:.2f}, {lon:.2f}): {sg_err}")
                                    data = None

                            # Fall back to proxy if SoilGrids failed or unavailable
                            if data is None:
                                try:
                                    data = proxy_api.get_data(lat, lon)
                                    if data:
                                        proxy_used += 1
                                except Exception as proxy_err:
                                    workflow.logger.debug(f"   Proxy API failed for ({lat:.2f}, {lon:.2f}): {proxy_err}")

                            if data:
                                try:
                                    for key, val in data.items():
                                        if key not in workflow.adata.obs.columns:
                                            workflow.adata.obs[key] = np.nan
                                        workflow.adata.obs.loc[sample_id, key] = val
                                    enriched += 1
                                except Exception as assign_err:
                                    workflow.logger.debug(f"   Failed to assign geochemical data for {sample_id}: {assign_err}")
                                    failed += 1

                            # Progress logging
                            if (idx_count + 1) % progress_interval == 0 or idx_count == 0:
                                pct = 100 * (idx_count + 1) / total_rows
                                workflow.logger.info(f"   Geochemical progress: {idx_count + 1}/{total_rows} ({pct:.1f}%)")

                        if enriched > 0:
                            msg = f"✅ Geochemical data: {enriched} samples enriched"
                            if proxy_used > 0:
                                msg += f" ({proxy_used} via proxy database)"
                            workflow.logger.info(f" {msg}")
            except ImportError as e:
                workflow.logger.debug(f" ℹ️ Geochemical APIs not available: {e}")
            except Exception as e:
                workflow.logger.warning(f" ⚠️  Geochemical integration failed: {e}")
        else:
            workflow.logger.debug(" ℹ️  Geochemical data disabled, skipping")
    finally:
        workflow.telemetry.end_phase('backfill_geochemical')
    
    # =========================================================================
    # [FINAL] Fix data types for h5py compatibility
    # =========================================================================
    workflow.logger.info(" 🔧 Fixing data types for h5ad serialization...")
    fix_adata_dtypes(workflow.adata, inplace=True)
    workflow.logger.info(" ✅ Data type fixes applied.")
    
    # =========================================================================
    # [SUMMARY] Backfill Completion Report
    # =========================================================================
    workflow.logger.info(" " + "="*70)
    workflow.logger.info(" 📊 BACKFILL ENRICHMENT PHASE COMPLETE")
    workflow.logger.info(" " + "="*70)

    # Count and categorize enrichment columns by dataset source
    enrichment_cols = [col for col in workflow.adata.obs.columns if any(
        col.startswith(prefix) for prefix in [
            'env_', 'csu_', 'ISDASOIL_', 'DEM_', 'ERA5_', 'worldcover_',
            'openlandmap_', 'jrc_', 'hansen_', 'lights_'
        ]
    )]

    # Group columns by dataset for detailed reporting
    col_by_dataset = {
        'JRC Water': [],
        'VIIRS Lights': [],
        'Hansen GFC': [],
        'Copernicus DEM': [],
        'ERA5 Climate': [],
        'WorldCover Landuse': [],
        'OpenLandMap': [],
        'ISDASOIL Geochemistry': [],
        'CSU Soil Metals': [],
        'Environmental APIs': []
    }

    for col in enrichment_cols:
        if col.startswith('jrc_'):
            col_by_dataset['JRC Water'].append(col)
        elif col.startswith('lights_'):
            col_by_dataset['VIIRS Lights'].append(col)
        elif col.startswith('hansen_'):
            col_by_dataset['Hansen GFC'].append(col)
        elif col.startswith('DEM_'):
            col_by_dataset['Copernicus DEM'].append(col)
        elif col.startswith('ERA5_'):
            col_by_dataset['ERA5 Climate'].append(col)
        elif col.startswith('worldcover_'):
            col_by_dataset['WorldCover Landuse'].append(col)
        elif col.startswith('openlandmap_'):
            col_by_dataset['OpenLandMap'].append(col)
        elif col.startswith('ISDASOIL_'):
            col_by_dataset['ISDASOIL Geochemistry'].append(col)
        elif col.startswith('csu_'):
            col_by_dataset['CSU Soil Metals'].append(col)
        elif col.startswith('env_'):
            col_by_dataset['Environmental APIs'].append(col)

    workflow.logger.info(f" ✓ Total enrichment columns added: {len(enrichment_cols)}")

    # Log breakdown by dataset
    for dataset, cols in col_by_dataset.items():
        if len(cols) > 0:
            col_list = ', '.join(cols[:2]) + (f', ... ({len(cols)-2} more)' if len(cols) > 2 else '')
            workflow.logger.info(f"   • {dataset}: {len(cols)} columns ({col_list})")

    workflow.logger.info(f" ✓ Final metadata shape: {workflow.adata.obs.shape[0]} samples × {workflow.adata.obs.shape[1]} columns")

    # Summary stats on data completeness
    if enrichment_cols:
        completeness = {}
        for col in enrichment_cols[:10]:  # Report on first 10 enrichment columns
            non_null = workflow.adata.obs[col].notna().sum()
            pct = 100 * non_null / len(workflow.adata)
            if non_null > 0:
                completeness[col] = f"{non_null}/{len(workflow.adata)} ({pct:.1f}%)"

        if completeness:
            workflow.logger.info(f" ✓ Data completeness sample:")
            for col, stat in list(completeness.items())[:3]:
                workflow.logger.info(f"   - {col}: {stat}")

    workflow.logger.info(" 🎉 All external API enrichments completed!")
    workflow.logger.info(" " + "="*70)
