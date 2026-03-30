"""
ENA Metadata Backfill Module

Handles fetching and merging missing metadata columns from ENA/SRA into AnnData objects.
Supports both cached and fresh API calls with proper async handling.
"""

import asyncio
import logging
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad

from workflow_16s.config import AppConfig
from workflow_16s.utils.progress import get_progress_bar
from .fetcher import ENAFetcher
from .cache import SQLiteCacheManager as CacheManager


def _get_record_type(record: Dict[str, Any]) -> str:
    """
    Classify a record by accession type to determine if it's a run, biosample, or experiment.
    
    Args:
        record: Dictionary with metadata record from ENA API
        
    Returns:
        str: One of 'run', 'biosample', or 'experiment'
    """
    accessions = {
        'run_accession': record.get('run_accession'),
        'sample_accession': record.get('sample_accession'),
        'secondary_sample_accession': record.get('secondary_sample_accession'),
        'experiment_accession': record.get('experiment_accession'),
        'primary_accession': record.get('primary_accession'),
    }
    
    # Classify by the strongest accession signal
    if accessions['run_accession']:
        run_acc = str(accessions['run_accession']).upper()
        if run_acc.startswith(('SRR', 'ERR', 'DRR')):
            return 'run'
    
    if accessions['experiment_accession']:
        exp_acc = str(accessions['experiment_accession']).upper()
        if exp_acc.startswith(('SRX', 'ERX', 'DRX')):
            return 'experiment'
    
    # If we have sample info, it's biosample-like
    if accessions['sample_accession'] or accessions['secondary_sample_accession']:
        return 'biosample'
    
    # Default to biosample
    return 'biosample'


def _separate_records_by_type(records: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
    """
    Separate records from mixed ENA API responses into type-specific dictionaries.
    
    Records from ENA API can contain biosamples, runs, and experiments mixed together.
    This function separates them while preserving ALL columns and using appropriate indices.
    
    Args:
        records: List of records from ENA API
        
    Returns:
        Tuple of (runs_dict, biosamples_dict, experiments_dict) keyed by appropriate accession
    """
    runs = {}
    biosamples = {}
    experiments = {}
    
    for record in records:
        if not record or not isinstance(record, dict):
            continue
        
        record_type = _get_record_type(record)
        
        if record_type == 'run':
            # Index by run_accession; preserve all columns
            run_acc = record.get('run_accession')
            if run_acc:
                runs[str(run_acc)] = record
        elif record_type == 'experiment':
            # Index by experiment_accession; preserve all columns
            exp_acc = record.get('experiment_accession')
            if exp_acc:
                experiments[str(exp_acc)] = record
        else:  # biosample
            # Index by primary sample accession (prefer secondary_sample_accession or sample_accession)
            primary_acc = record.get('sample_accession') or record.get('secondary_sample_accession')
            if not primary_acc:
                # Fallback for biosamples without explicit sample accession
                primary_acc = record.get('primary_accession') or record.get('accession')
            if primary_acc:
                biosamples[str(primary_acc)] = record
    
    return runs, biosamples, experiments


def _merge_with_logging(
    obs: pd.DataFrame,
    ena_records: Dict[str, Dict],
    accession_col: str,
    record_type: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Merge ENA records into obs DataFrame, preserving all columns and logging the merge process.

    OPTIMIZED: Uses indexed merge (O(n+m)) instead of unindexed merge (O(n×m))

    Args:
        obs: Original observation DataFrame (has sample row indices and may have accession column)
        ena_records: Dictionary of records keyed by their accession
        accession_col: Column name in obs to use for joining (e.g., 'run_accession', 'sample_id')
        record_type: Type of record being merged ('run', 'biosample', 'experiment')
        logger: Logger instance for debug output

    Returns:
        Updated obs DataFrame with new columns added from ENA records
    """
    if not ena_records or accession_col not in obs.columns:
        if not ena_records:
            logger.debug(f"   ℹ️  No {record_type} records to merge")
        else:
            logger.debug(f"   ⚠️  Column '{accession_col}' not found in obs (available: {list(obs.columns)[:5]}...)")
        return obs

    # Create DataFrame from ENA records, preserving the accession keys as a column
    ena_list = []
    for accession, record in ena_records.items():
        record_copy = record.copy()
        # Store the accession key that we'll use for merging
        record_copy['_merge_key'] = accession
        ena_list.append(record_copy)

    ena_df = pd.DataFrame(ena_list)

    if ena_df.empty:
        return obs

    # OPTIMIZATION: Use indexed merge instead of unindexed merge
    # Set the merge key as index for O(n+m) complexity instead of O(n×m)
    ena_df = ena_df.set_index('_merge_key')
    obs = obs.set_index(accession_col)

    try:
        # Merge using indices (much faster for large DataFrames)
        merged = obs.join(
            ena_df,
            how='left',
            lsuffix='',
            rsuffix=f'_{record_type}'
        )
    finally:
        # Reset index to restore original structure
        merged = merged.reset_index()
        merged.rename(columns={accession_col: accession_col}, inplace=True)

    
    # Remove the merge key column
    if '_merge_key' in merged.columns:
        merged.drop(columns=['_merge_key'], inplace=True)
    
    # Track columns before merge
    cols_before = set(obs.columns)
    
    # Find new columns
    cols_after = set(merged.columns)
    new_cols = sorted(cols_after - cols_before)
    
    if new_cols:
        non_null_counts = merged[new_cols].notna().sum()
        logger.debug(f"   📊 {record_type.upper()} Merge: Added {len(new_cols)} columns:")
        for col in new_cols:
            non_null = non_null_counts[col]
            total = len(merged)
            pct = (non_null / total * 100) if total > 0 else 0
            logger.debug(f"      • {col:30s} | {non_null:6d}/{total:6d} ({pct:5.1f}%)")
    
    return merged


def fetch_metadata_from_ena(
    adata: ad.AnnData,
    config: AppConfig,
    logger: Optional[logging.Logger] = None,
    progress: Optional[Any] = None,
    progress_task_id: Optional[int] = None
):
    """
    Backfill missing metadata columns in adata.obs from ENA/SRA.

    Uses ENAFetcher to:
    1. Detect SRA/BioSample accessions in metadata
    2. Fetch comprehensive metadata via ENA API
    3. Handle secondary sample accessions (SAMN/SRS/SAME/SAMD) via NCBI lookup
    4. Merge all retrieved fields (collection_date, coordinates, instrument, etc.)

    Properly caches results to avoid re-fetching on subsequent runs.

    Args:
        adata: AnnData object to enrich
        config: AppConfig configuration object
        logger: Optional logger instance
        progress: Optional Rich Progress object for tracking (integrates with existing dashboard)
        progress_task_id: Optional task ID if adding sub-tasks to existing progress
    """
    if logger is None:
        logger = logging.getLogger("workflow_16s")

    obs = adata.obs.copy()

    # Log collection_date status BEFORE enrichment
    initial_collection_date_count = obs["collection_date"].notna().sum() if "collection_date" in obs.columns else 0
    initial_collection_date_pct = (initial_collection_date_count / len(obs)) * 100 if len(obs) > 0 else 0
    logger.info(f"   📅 Initial collection_date coverage: {initial_collection_date_count}/{len(obs)} samples ({initial_collection_date_pct:.1f}%)")

    # Find all SRA accessions (run, sample, biosample, experiment)
    accession_candidates = {}
    accession_cols = set()

    # Priority: explicit run_accession column
    if "run_accession" in obs.columns:
        for val in obs["run_accession"].dropna().unique():
            val_str = str(val).strip().upper()
            if val_str and val_str.startswith(
                ("SRR", "ERR", "DRR", "SRS", "SAMN", "SAME", "SAMD", "SRX", "ERX", "DRX")
            ):
                accession_candidates[val_str] = "run_accession"
                accession_cols.add("run_accession")

    # Check for sample_id that contains accessions
    if "sample_id" in obs.columns:
        for val in obs["sample_id"].dropna().unique():
            val_str = str(val).strip().upper()
            if val_str and val_str.startswith(("SRR", "ERR", "DRR", "SRS", "SAMN", "SAME", "SAMD")):
                accession_candidates[val_str] = "sample_id"
                accession_cols.add("sample_id")

    # Scan other columns for accession-like values
    keywords = ["run", "sample", "accession", "sra", "biosample", "srr", "err", "drr", "srx"]
    for col in obs.columns:
        if col in accession_cols:
            continue
        col_lower = col.lower()
        if any(kw in col_lower for kw in keywords):
            for val in obs[col].dropna().unique():
                val_str = str(val).strip().upper()
                if val_str and val_str.startswith(
                    ("SRR", "ERR", "DRR", "SRS", "SAMN", "SAME", "SAMD", "SRX", "ERX", "DRX")
                ):
                    accession_candidates[val_str] = col
                    accession_cols.add(col)

    if not accession_candidates:
        logger.warning("   ⚠️  No SRA accessions found in metadata columns")
        adata.obs = obs
        return

    logger.info(f"   📅 ENA Backfill: Found {len(accession_candidates)} accessions, fetching metadata...")

    # Check if max_api_requests limit is set in config
    max_requests = getattr(config.ena_backfill, 'max_api_requests', None) if hasattr(config, 'ena_backfill') else None
    accession_list = list(accession_candidates.keys())
    
    if max_requests and len(accession_list) > max_requests:
        logger.warning(f"   ⚠️  Limiting ENA requests to {max_requests} accessions (configured max). Skipping {len(accession_list) - max_requests} additional accessions.")
        accession_list = accession_list[:max_requests]
        accession_candidates = {acc: accession_candidates[acc] for acc in accession_list}

    # Initialize variables for finally block
    cache_manager = None
    loop = None

    # Use ENAFetcher with proper async handling
    try:
        email = config.credentials.ena_email  # Default, may be overridden

        # Determine cache directory from config (in priority order)
        cache_dir = None

        # 1. Check if user specified a custom cache_dir in config.ena_backfill.cache_dir
        if hasattr(config.ena_backfill, 'cache_dir') and config.ena_backfill.cache_dir:
            cache_dir = Path(config.ena_backfill.cache_dir)
            logger.debug(f"   🗂️ Using config-specified cache directory: {cache_dir}")
        else:
            # 2. Fall back to project-based cache directory (respects project structure from config.yaml)
            cache_dir = config.paths.project / ".cache" / "ena"
            logger.debug(f"   🗂️ Using project cache directory: {cache_dir}")

        # Create cache directory with error handling
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            # Fallback to home directory if project cache fails
            logger.warning(
                f"   ⚠️  Could not create cache dir ({cache_dir}): {e}. "
                f"Falling back to home directory."
            )
            cache_dir = Path.home() / ".cache" / "ena_metadata"
            cache_dir.mkdir(parents=True, exist_ok=True)

        # Convert cache TTL from days to seconds (config.ena_backfill.cache_ttl_days)
        ttl_seconds = config.ena_backfill.cache_ttl_days * 86400
        cache_manager = CacheManager(cache_dir, ttl_seconds=ttl_seconds)
        
        # Get Phase 2 async config
        use_phases = getattr(config.ena_backfill, 'fetch_phases', True)
        phase2_async = getattr(config.ena_backfill, 'phase2_async', True)

        async def _fetch_phase2_async(missing_biosamples, cache_mgr, email_addr, fetcher):
            """Background async task to fetch Phase 2 extended fields."""
            try:
                logger.debug(f"   🔄 Phase 2: Fetching {len(missing_biosamples)} extended fields in background...")
                phase2_data = await fetcher.fetch_biosamples_batch(
                    missing_biosamples,
                    with_progress_bar=False,
                    chunk_size=50
                )
                if phase2_data:
                    logger.debug(f"   ✅ Phase 2 enrichment complete: {len(phase2_data)} records updated")
            except Exception as e:
                logger.debug(f"   ⚠️  Phase 2 background fetch failed: {e}")

        async def _fetch_ena_batch():
            # Create a progress task for ENA fetch if progress object is available
            batch_task_id = None
            if progress is not None:
                try:
                    accession_count = len(accession_candidates)
                    batch_task_id = progress.add_task(
                        f"[cyan]📥 ENA Fetch[/cyan]",
                        total=accession_count
                    )
                except Exception as e:
                    logger.debug(f"Could not create progress task: {e}")
            
            # 1. READ LOCAL: Check the cache for each individual accession first (using efficient bulk lookup)
            cached_runs = []
            missing_biosamples = []
            missing_runs = []
            missing_exps = []

            accession_list = list(accession_candidates.keys())

            # 🟢 OPTIMIZATION: Use bulk cache lookup instead of individual gets (273K+ lookups)
            cached_dict = await cache_manager.get_bulk(accession_list)
            cached_runs = list(cached_dict.values())

            # Determine which accessions are missing and sort by type
            for acc in accession_list:
                if acc not in cached_dict:
                    # Sort into missing buckets for the API
                    if acc.startswith(("SRS", "SAMN", "SAME", "SAMD")):
                        missing_biosamples.append(acc)
                    elif acc.startswith(("SRR", "ERR", "DRR")):
                        missing_runs.append(acc)
                    elif acc.startswith(("SRX", "ERX", "DRX")):
                        missing_exps.append(acc)

            logger.info(
                f"   🗄️ Cache hit for {len(cached_runs)} accessions. "
                f"Fetching {len(missing_biosamples) + len(missing_runs) + len(missing_exps)} from ENA..."
            )

            # Filter out None values from cached results
            cached_runs = [r for r in cached_runs if r is not None and isinstance(r, dict)]

            # If everything was cached, return immediately!
            if not (missing_biosamples or missing_runs or missing_exps):
                return cached_runs

            # 2. FETCH BULK: Only ask ENA for what is missing
            new_runs = []
            # NOTE: Pass progress bar to fetcher if available (integrates with existing dashboard)
            # If no progress bar is provided, ENAFetcher will use logger-based updates
            async with ENAFetcher(
                email, 
                cache_manager=cache_manager,
                progress=progress,  # Pass existing progress bar (may be None if not provided)
                progress_task_id=progress_task_id
            ) as fetcher:
                if missing_biosamples:
                    logger.debug(f"   📥 Phase 1: Fetching {len(missing_biosamples)} essential fields...")
                    # Phase 1: Fetch essential fields only (fast)
                    biosamples_data = await fetcher.fetch_biosamples_phase1(
                        missing_biosamples, 
                        with_progress_bar=False
                    )
                    if biosamples_data:
                        new_runs.extend(list(biosamples_data.values()))
                    logger.debug(f"   ✅ Phase 1 complete: {len(biosamples_data or {})} records fetched")
                    
                    # Queue Phase 2 async (non-blocking)
                    use_phases = getattr(config.ena_backfill, 'fetch_phases', True)
                    phase2_async = getattr(config.ena_backfill, 'phase2_async', True)
                    if use_phases and phase2_async and missing_biosamples:
                        asyncio.create_task(
                            _fetch_phase2_async(
                                missing_biosamples,
                                cache_manager,
                                email,
                                fetcher
                            )
                        )
                        logger.debug(f"   🔄 Phase 2 running in background")

                # HIERARCHICAL ENRICHMENT: Extract linked runs from biosamples
                # For each biosample, check if it has a linked run_accession
                linked_run_accessions = set()
                for record in new_runs:
                    if isinstance(record, dict) and 'run_accession' in record:
                        run_acc = record.get('run_accession')
                        if run_acc and str(run_acc).strip().upper().startswith(('SRR', 'ERR', 'DRR')):
                            linked_run_accessions.add(str(run_acc).upper())
                
                # Remove accessions we already fetched or have in cache
                runs_to_fetch = linked_run_accessions - set(cached_dict.keys()) - set(missing_runs)
                
                if runs_to_fetch:
                    logger.debug(f"   🔗 Hierarchical enrichment: Found {len(runs_to_fetch)} linked runs from biosamples, fetching...")
                    linked_run_data = await fetcher.fetch_ena_data_in_batches(
                        "read_run", "run_accession", list(runs_to_fetch), with_progress_bar=False
                    )
                    if linked_run_data:
                        new_runs.extend(linked_run_data)

                if missing_runs:
                    run_data = await fetcher.fetch_ena_data_in_batches(
                        "read_run", "run_accession", missing_runs, with_progress_bar=False
                    )
                    if run_data:
                        new_runs.extend(run_data)

                if missing_exps:
                    exp_data = await fetcher.fetch_ena_data_in_batches(
                        "experiment", "experiment_accession", missing_exps, with_progress_bar=False
                    )
                    if exp_data:
                        new_runs.extend(exp_data)

            # 3. WRITE LOCAL: Break the bulk response apart and cache per sample
            for record in new_runs:
                # Find the ID for this specific record (check common ENA keys)
                record_id = (
                    record.get("run_accession")
                    or record.get("sample_accession")
                    or record.get("secondary_sample_accession")
                    or record.get("primary_accession")
                )

                if record_id:
                    # Ensure record has the accession identifier for later retrieval
                    # This is critical for DataFrame creation since accession keys are removed in fetch
                    record_with_accession = record.copy()
                    
                    # Add the accession using the appropriate key based on what we have
                    if record_id.startswith(("SRR", "ERR", "DRR")):
                        record_with_accession["run_accession"] = record_id
                    elif record_id.startswith(("SRX", "ERX", "DRX")):
                        record_with_accession["experiment_accession"] = record_id
                    elif record_id.startswith(("SRS", "SAMN", "SAME", "SAMD")):
                        record_with_accession["sample_accession"] = record_id
                    else:
                        record_with_accession["accession"] = record_id
                    
                    await cache_manager.set(record_id, record_with_accession)

            logger.info(f" ✅ Added {len(new_runs)} new records to local sample cache.")

            # Filter out None/invalid values from both cached and new results
            all_runs = cached_runs + new_runs
            all_runs = [r for r in all_runs if r is not None and isinstance(r, dict) and len(r) > 0]

            # Return the combined list of cached and newly fetched data
            return all_runs

        # Run async fetch with error handling for date parsing
        loop = asyncio.get_event_loop()
        try:
            ena_results = loop.run_until_complete(_fetch_ena_batch())
        except RuntimeError:
            # Already in event loop
            try:
                ena_results = asyncio.run(_fetch_ena_batch())
            except (ValueError, TypeError) as date_error:
                # Handle date parsing errors gracefully
                logger.warning(f"   ⚠️  Date parsing error in ENA data: {date_error}")
                logger.debug(f"   Skipping ENA backfill due to date parsing issue")
                adata.obs = obs
                return

        if not ena_results:
            logger.warning("   ⚠️  ENA fetch returned no results for accessions")
            adata.obs = obs
            return

        logger.debug(f"   📥 Processing {len(ena_results)} ENA records...")

        # ===== SEPARATE RECORDS BY TYPE AND PREPARE FOR MERGE =====
        # ENA API returns mixed record types (runs, biosamples, experiments).
        # Separate them to preserve type-specific columns (e.g., library_strategy for runs).
        logger.debug(f"   📊 Separating {len(ena_results)} records by type...")
        runs, biosamples, experiments = _separate_records_by_type(ena_results)
        log_msg = "   🔀 Record type breakdown:"
        if runs:
            log_msg += f" {len(runs)} runs,"
        if biosamples:
            log_msg += f" {len(biosamples)} biosamples,"
        if experiments:
            log_msg += f" {len(experiments)} experiments"
        logger.debug(log_msg)

        # ===== PARSE LATITUDE/LONGITUDE FROM ENA DATA =====
        # ENA returns 'lat' and 'lon' as string or float
        # Ensure they are numeric and valid in all record types
        # OPTIMIZATION: Vectorized operations on all records at once (instead of nested loops)
        for record_dict in [runs, biosamples, experiments]:
            if not record_dict:
                continue

            # Convert dict values to list for DataFrame conversion
            records_list = list(record_dict.values())
            accessions = list(record_dict.keys())

            # Create temporary DataFrame for vectorized operations
            temp_df = pd.DataFrame(records_list, index=accessions)

            if 'lat' in temp_df.columns:
                # Vectorized numeric conversion
                temp_df['lat'] = pd.to_numeric(temp_df['lat'], errors='coerce')
                # Validate latitude range [-90, 90]
                temp_df.loc[(temp_df['lat'] < -90) | (temp_df['lat'] > 90), 'lat'] = None

            if 'lon' in temp_df.columns:
                # Vectorized numeric conversion
                temp_df['lon'] = pd.to_numeric(temp_df['lon'], errors='coerce')
                # Validate longitude range [-180, 180]
                temp_df.loc[(temp_df['lon'] < -180) | (temp_df['lon'] > 180), 'lon'] = None

            # OPTIMIZATION: Update original records using fast dict updates instead of O(n²) list search
            # Before: for idx_pos, accession in enumerate(accessions): record_dict[accession] = ...
            # This was O(n²) due to accessions.index() lookup ❌
            # After: Use dictionary instead of list lookups
            for idx_pos, accession in enumerate(accessions):
                if 'lat' in temp_df.columns:
                    record_dict[accession]['lat'] = temp_df.iloc[idx_pos]['lat']
                if 'lon' in temp_df.columns:
                    record_dict[accession]['lon'] = temp_df.iloc[idx_pos]['lon']
        
        # ===== SANITIZE DATETIME COLUMNS =====
        # Handle date ranges (e.g., "2020/2021") by taking first date
        # OPTIMIZATION: Vectorized operations on entire columns (instead of row-by-row loops)
        def vectorized_date_parse(series):
            """Vectorized date parsing for entire Series."""
            if series.empty:
                return pd.Series([], dtype='datetime64[ns]')

            # Convert to string, handle NaN/None/empty
            str_series = series.astype(str).str.strip()
            str_series = str_series.replace(['nan', 'None', ' '], '')

            # Handle date ranges: take first part if "/" exists
            str_series = str_series.str.split('/').str[0]

            # Vectorized datetime conversion
            result = pd.to_datetime(str_series, errors='coerce')
            return result

        datetime_keys = ['collection_date', 'collection_date_start', 'collection_date_end',
                        'first_public', 'last_updated', 'submission_date']

        for record_dict in [runs, biosamples, experiments]:
            if not record_dict:
                continue

            # Convert to DataFrame for vectorized operations
            records_list = list(record_dict.values())
            accessions = list(record_dict.keys())
            temp_df = pd.DataFrame(records_list, index=accessions)

            # Process each datetime key
            for key in datetime_keys:
                if key in temp_df.columns:
                    # Vectorized date parsing
                    temp_df[key] = vectorized_date_parse(temp_df[key])

            # Update original records with parsed dates
            for idx_pos, accession in enumerate(accessions):
                for key in datetime_keys:
                    if key in temp_df.columns:
                        record_dict[accession][key] = temp_df.iloc[idx_pos][key]

        # ===== MERGE EACH RECORD TYPE SEPARATELY =====
        # Process runs, biosamples, and experiments separately to preserve all columns
        # Note: Hierarchical enrichment is applied - if biosamples were fetched, their
        # linked runs are automatically extracted and fetched to provide run-level metadata
        # (e.g., library_strategy, instrument_model, spot_count) along with biosample data.
        logger.debug(f"   🔄 Merging records by type into observation metadata...")
        
        # Determine merge columns by analyzing accession_candidates
        # Group accessions by their source column and type
        run_accession_cols = set()
        sample_accession_cols = set()
        exp_accession_cols = set()
        
        for accession, source_col in accession_candidates.items():
            accession_upper = str(accession).upper()
            # Classify accession by type
            if accession_upper.startswith(("SRR", "ERR", "DRR")):
                run_accession_cols.add(source_col)
            elif accession_upper.startswith(("SRS", "SAMN", "SAME", "SAMD")):
                sample_accession_cols.add(source_col)
            elif accession_upper.startswith(("SRX", "ERX", "DRX")):
                exp_accession_cols.add(source_col)
        
        # For each type, prefer columns in this order
        run_cols_to_try = list(run_accession_cols) + ["run_accession"] if run_accession_cols else ["run_accession"]
        sample_cols_to_try = list(sample_accession_cols) + ["sample_accession", "secondary_sample_accession", "sample_id"] if sample_accession_cols else ["sample_accession", "secondary_sample_accession", "sample_id"]
        exp_cols_to_try = list(exp_accession_cols) + ["experiment_accession"] if exp_accession_cols else ["experiment_accession"]
        
        # Find actual run column to use (prefer discovered columns first)
        run_col_to_use = next((col for col in run_cols_to_try if col in obs.columns), None)
        
        # Find actual sample column to use (prefer discovered columns first)
        sample_col_to_use = next((col for col in sample_cols_to_try if col in obs.columns), None)
        
        # Find actual experiment column to use (prefer discovered columns first)
        exp_col_to_use = next((col for col in exp_cols_to_try if col in obs.columns), None)
        
        # Merge runs using detected column
        if runs and run_col_to_use:
            obs = _merge_with_logging(obs, runs, run_col_to_use, "run", logger)
            logger.debug(f"   ✅ Merged {len(runs)} run records using column '{run_col_to_use}'")
        elif runs:
            logger.debug(f"   ⚠️  {len(runs)} run records found but no compatible accession column in obs (tried: {run_cols_to_try})")
        
        # Merge biosamples using detected column
        if biosamples and sample_col_to_use:
            obs = _merge_with_logging(obs, biosamples, sample_col_to_use, "biosample", logger)
            logger.debug(f"   ✅ Merged {len(biosamples)} biosample records using column '{sample_col_to_use}'")
        elif biosamples:
            logger.debug(f"   ⚠️  {len(biosamples)} biosample records found but no compatible accession column in obs (tried: {sample_cols_to_try})")
        
        # Merge experiments using detected column
        if experiments and exp_col_to_use:
            obs = _merge_with_logging(obs, experiments, exp_col_to_use, "experiment", logger)
            logger.debug(f"   ✅ Merged {len(experiments)} experiment records using column '{exp_col_to_use}'")
        elif experiments:
            logger.debug(f"   ⚠️  {len(experiments)} experiment records found but no compatible accession column in obs (tried: {exp_cols_to_try})")

        # ===== CONSOLIDATE SUFFIXED COLUMNS =====
        # After merging runs, biosamples, and experiments, we have columns like:
        # collection_date_biosample, collection_date_run, collection_date_experiment
        # We need to consolidate these back into "collection_date" (or other base names)
        
        suffixed_types = ['_biosample', '_run', '_experiment']
        base_col_names = set()
        
        # Find all suffixed columns
        for col in obs.columns:
            for suffix in suffixed_types:
                if col.endswith(suffix):
                    base_name = col[:-len(suffix)]
                    base_col_names.add(base_name)
        
        # For each base column, consolidate the suffixed versions
        for base_col in base_col_names:
            suffixed_cols = [f"{base_col}{suffix}" for suffix in suffixed_types 
                           if f"{base_col}{suffix}" in obs.columns]
            
            if len(suffixed_cols) > 0:
                # Create base column if it doesn't exist
                if base_col not in obs.columns:
                    obs[base_col] = None
                
                # Fill NaN values in base_col with values from suffixed columns (in priority order)
                for suffixed_col in suffixed_cols:
                    mask = obs[base_col].isna() & obs[suffixed_col].notna()
                    obs.loc[mask, base_col] = obs.loc[mask, suffixed_col]
                
                # Remove the suffixed columns after consolidation
                obs.drop(columns=suffixed_cols, inplace=True)
                logger.debug(f"   🔄 Consolidated {len(suffixed_cols)} suffixed columns for '{base_col}'")

        # Ensure collection_date in final obs
        if "collection_date" not in obs.columns:
            obs["collection_date"] = pd.NaT

        # ===== LOG NEW ENA COLUMNS ADDED =====
        # Determine which columns were originally in obs vs newly added from ENA
        original_cols = set(adata.obs.columns)
        new_cols_from_ena = sorted([col for col in obs.columns if col not in original_cols])

        if new_cols_from_ena:
            logger.debug(f"   📋 ENA Added {len(new_cols_from_ena)} New Columns:")

            # Log each new column with sample values
            for col in new_cols_from_ena:
                col_data = obs[col]
                non_null_count = col_data.notna().sum()
                non_null_pct = (non_null_count / len(col_data)) * 100 if len(col_data) > 0 else 0
                dtype_name = str(col_data.dtype)

                # Get sample non-null value
                sample_val = None
                sample_rows = col_data[col_data.notna()].head(3).values
                if len(sample_rows) > 0:
                    sample_val = str(sample_rows[0])[:50]  # Truncate long strings

                logger.debug(f"     • {col:25s} | {non_null_count:6d}/{len(col_data):6d} ({non_null_pct:5.1f}%) | {dtype_name:12s} | sample: {sample_val}")

            # Special emphasis on coordinates if present
            if 'lat' in new_cols_from_ena or 'lon' in new_cols_from_ena:
                coords_found = 0
                if 'lat' in obs.columns and 'lon' in obs.columns:
                    coords_found = ((obs['lat'].notna()) & (obs['lon'].notna())).sum()
                logger.debug(f"   ✨ Coordinates available: {coords_found} samples have both lat and lon")

        # ===== LOG COLLECTION_DATE ENRICHMENT RESULTS =====
        final_collection_date_count = obs["collection_date"].notna().sum() if "collection_date" in obs.columns else 0
        final_collection_date_pct = (final_collection_date_count / len(obs)) * 100 if len(obs) > 0 else 0
        collection_date_gained = max(0, final_collection_date_count - initial_collection_date_count)

        total_records = len(runs) + len(biosamples) + len(experiments)
        total_new_cols = len(new_cols_from_ena) if new_cols_from_ena else 0

        logger.info(
            f"   ✅ ENA Enrichment Complete:"
            f" {total_records} records merged, {total_new_cols} new columns added"
        )
        logger.info(
            f"   📅 Collection Date Coverage: {initial_collection_date_count} → {final_collection_date_count} samples "
            f"({initial_collection_date_pct:.1f}% → {final_collection_date_pct:.1f}%) | "
            f"+{collection_date_gained} samples gained collection_date"
        )
        adata.obs = obs

    except Exception as e:
        import traceback

        logger.warning(f"   ⚠️  ENA backfill failed: {e}")
        logger.debug(f"   🔍 Traceback:\n{traceback.format_exc()}")
        adata.obs = obs
    finally:
        if cache_manager and hasattr(cache_manager, "close") and loop:
            try:
                loop.run_until_complete(cache_manager.close())
            except Exception:
                pass
