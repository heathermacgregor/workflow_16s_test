# workflow_16s/upstream/metadata/partition.py

# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import asyncio
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

# Third-Party Imports
import pandas as pd
from Bio import Entrez

# Local Imports
from workflow_16s.api.ena.metadata_api import get_samples_by_bioproject_async
from workflow_16s.api.ena.metadata.cache import CacheManager as EnaCacheManager
from workflow_16s.api.ena.sequences import SequenceFetcher
from workflow_16s.api.nuclear_fuel_cycle.nfc import NFCFacilitiesHandler
from workflow_16s.api.qiime import execute
from workflow_16s.config_schema import AppConfig
from workflow_16s.upstream.sequences.analysis import run_comprehensive_analysis
from workflow_16s.utils.dir_utils import Project, SubSet
from workflow_16s.utils.publication_fetcher import PublicationFetcher

# Import from our new refactored modules
from .constants import exclusion_keywords
from .utils import (
    PartitionCacheManager,
    format_bytes,
    is_host_associated,
    find_keyword_matches
)
from .processor import PartitionProcessor

# ==================================================================================== #

from workflow_16s.utils.logger import get_logger
logger = get_logger()

# ========================== CORE PROCESSING CLASS ========================== #

class DatasetPartition:
    """Orchestrates the fetching, analysis, partitioning, and processing of 16S datasets."""
    ENA_PATTERN = re.compile(r"^PRJ[EDN][A-Z]\d{4,}$", re.IGNORECASE)
    FWD_PRIMER_COL = "pcr_primer_fwd_seq"
    REV_PRIMER_COL = "pcr_primer_rev_seq"
    MIN_RUNS_THRESHOLD = 5

    def __init__(
        self, config: AppConfig, publication_fetcher: PublicationFetcher,
        region_to_pairs_map: Dict[str, list],
        nfc_handler: Optional[NFCFacilitiesHandler] = None,
        nfc_facilities_df: Optional[pd.DataFrame] = None
    ):
        self.config = config
        self.project_dir = Project(config)
        self.mode = self.config.sequences.pcr_primers.mode.lower()
        # Set Entrez email globally or ensure fetcher handles it
        if config.credentials.ena_email: Entrez.email = config.credentials.ena_email
        else: logger.warning("No email provided in config for Entrez (NCBI Taxonomy). Using default or potentially hitting rate limits."); Entrez.email = "default_user@example.com" # Provide a default

        self.publication_fetcher = publication_fetcher
        self.region_to_pairs_map = region_to_pairs_map

        # Pass NFC info to the processor
        self.nfc_handler = nfc_handler
        self.nfc_facilities_df = nfc_facilities_df if nfc_facilities_df is not None else pd.DataFrame()

        self.processed_h5ad_paths: List[Path] = []
        self.failed: List[Dict[str, str]] = []

        self.cache_db_path = self.project_dir.cache / "partition_cache.db"
        self.cache = PartitionCacheManager(self.cache_db_path)
        logger.info(f"Using SQLite partition cache at: {self.cache_db_path}")

    async def run(
        self, datasets: Dict[str, Dict[str, Any]], ena_cache_manager: EnaCacheManager
    ) -> Tuple[List[Path], List[Dict[str, str]]]:
        """Processes a batch of one or more datasets."""
        # Ensure ena_cache_manager is passed correctly
        if not isinstance(ena_cache_manager, EnaCacheManager): logger.warning("Invalid EnaCacheManager provided to partitioner run method. Caching might fail.")

        for dataset_id, info in datasets.items():
            logger.info(f"--- Starting processing for dataset: {dataset_id} ---")
            try:
                # Attempt to get full metadata DataFrame from cache
                metadata_cache_key = f"metadata_{dataset_id}" # Consistent key
                metadata_df = await ena_cache_manager.get(metadata_cache_key)

                # Convert potential list/dict from cache back to DataFrame if needed
                if metadata_df is not None and not isinstance(metadata_df, pd.DataFrame):
                    try:
                        metadata_df = pd.DataFrame(metadata_df)
                        logger.debug(f"Converted cached metadata for {dataset_id} back to DataFrame.")
                    except ValueError as e:
                        logger.warning(f"Could not convert cached metadata for {dataset_id} to DataFrame ({e}). Re-fetching.")
                        metadata_df = None # Force re-fetch

                # If cache miss or invalid cache item, fetch fresh metadata
                if metadata_df is None:
                    logger.info(f"Metadata DataFrame for {dataset_id} not in cache or invalid, fetching fresh metadata.")
                    # Fetch the full metadata DataFrame using the imported function
                    metadata_df = await get_samples_by_bioproject_async( # Now defined via import
                        bioproject_accession=dataset_id,
                        email=self.config.credentials.ena_email,
                        cache_manager=ena_cache_manager # Pass cache manager to potentially cache this result
                    )
                    # Cache the newly fetched DataFrame (if successful and not empty)
                    if metadata_df is not None and not metadata_df.empty:
                        # Convert DataFrame to list of dicts for JSON cache compatibility
                        await ena_cache_manager.set(metadata_cache_key, metadata_df.to_dict(orient='records'))
                    elif metadata_df is None: # Handle case where fetch itself failed
                        logger.error(f"Failed to fetch metadata for {dataset_id}. Skipping.")
                        self.failed.append({"dataset": dataset_id, "error": "Failed to fetch ENA metadata"})
                        continue # Skip to next dataset

                # Check if fetching resulted in an empty DataFrame
                if metadata_df.empty:
                    logger.warning(f"Skipping '{dataset_id}', no valid samples found or fetched.")
                    self.failed.append({"dataset": dataset_id, "error": "No valid samples"})
                    # Cache this failure? Optional.
                    self.cache.add_failed_dataset(dataset_id=dataset_id, reason="No valid samples")
                    continue # Skip to next dataset

                # Proceed with processing using the (potentially newly fetched) metadata_df
                await self.process(dataset_id, info, metadata_df)

            except Exception as e:
                logger.error(f"Partitioning setup failed unexpectedly for dataset {dataset_id}: {e}", exc_info=True)
                self.failed.append({"dataset": dataset_id, "error": f"Setup error: {str(e)}"})
                # Cache dataset failure if setup fails
                self.cache.add_failed_dataset(dataset_id=dataset_id, reason=f"Setup error: {str(e)}")

        return self.processed_h5ad_paths, self.failed

    def _log_prediction_summary(self, analysis_reports: Dict[str, Dict[str, Any]]):
        """Logs a summary of predicted regions from sequence analysis."""
        if not analysis_reports: logger.info("No analysis reports to summarize."); return
        # Group runs by prediction and reasoning
        summary_groups = defaultdict(list)
        for run_id, report in analysis_reports.items():
            prediction = report.get('prediction', 'Undetermined')
            reasoning = report.get('reasoning', 'No reasoning provided.')
            summary_groups[(prediction, reasoning)].append(run_id)
        # Sort groups by size (most common first)
        sorted_groups = sorted(summary_groups.items(), key=lambda item: len(item[1]), reverse=True)
        # Format and log the summary
        text = ["--- Run Prediction Summary ---"]
        for (prediction, reasoning), runs in sorted_groups:
            text.append(f" - {len(runs):>4} runs -> Prediction: {prediction:<12} | Reason: {reasoning}")
        logger.info('\n'.join(text))

    def _log_dataset_summary(self, dataset: str, info: Dict[str, Any], citations: List[str]):
        """Logs basic information about the dataset being processed."""
        platform = info.get('instrument_platform', 'N/A')
        model = info.get('instrument_model', 'N/A')
        layout = info.get('library_layout', 'N/A')
        logger.info(
            f"\n--- Dataset Info: {dataset.upper()} ---\n" # Clearer header
            f"    Description: {info.get('description', 'N/A')}\n"
            f"    Sequencing: {platform} ({model})\n"
            f"    Layout: {layout}"
        )
        if citations:
            logger.info("   Publications:")
            for cite in citations:
                logger.info(f"     - {cite}")

    async def auto(
        self, dataset: str, info: Dict[str, Any], metadata: pd.DataFrame,
        ena_runs: pd.DataFrame, citations: List[str]
    ) -> Dict[str, List[Path]]:
        """Handles partitioning for datasets in 'auto' primer mode."""
        if not self.config.sequences.validate_16s.enabled:
            logger.error("Auto mode requires 'validate_16s' to be enabled in config. Skipping dataset.")
            self.failed.append({"dataset": dataset, "error": "Auto mode disabled in config"})
            return {} # Return empty dict indicating no files processed

        est_dir = self.project_dir.raw_data / dataset # Directory for analysis outputs
        # Lists to store predictions and runs needing analysis
        run_predictions = [] # Stores dicts {'run_accession': ..., 'predicted_region': ...}
        runs_to_analyze_accs = set(metadata['run_accession']) # All runs initially need analysis

        # --- Check Cache for Existing Run Statuses ---
        cached_statuses = self.cache.get_dataset_run_statuses(dataset)
        if cached_statuses:
            logger.info(f"Found {len(cached_statuses)} cached run statuses for dataset {dataset}.")
            runs_found_in_cache = set()
            for run_acc, status_info in cached_statuses.items():
                if run_acc in runs_to_analyze_accs:
                    # If cached status is SUCCESS, add prediction to list
                    if status_info.get('status') == 'SUCCESS' and status_info.get('predicted_region'):
                        run_predictions.append({
                            'run_accession': run_acc,
                            'predicted_region': status_info['predicted_region']
                        })
                    # Mark run as processed via cache, regardless of status (SUCCESS, UNDETERMINED, FAILED_*)
                    runs_found_in_cache.add(run_acc)

            # Remove runs found in cache from the set needing analysis
            runs_to_analyze_accs -= runs_found_in_cache
            logger.info(f"{len(runs_found_in_cache)} runs found in cache. {len(runs_to_analyze_accs)} runs remaining for analysis.")

        # --- Analyze Runs Not Found in Cache ---
        run_file_paths_for_analysis: Dict[str, Path | List[Path]] = {} # Store paths only for runs analyzed now
        if not runs_to_analyze_accs:
            logger.info("All runs for this dataset were already cached. Proceeding to partitioning based on cached predictions.")
            if not run_predictions:  logger.warning(f"All runs cached for {dataset}, but no successful predictions found. Cannot partition."); return {} # Return empty paths dict

        else: # Runs need analysis
            logger.info(f"{len(runs_to_analyze_accs)} runs require sequence analysis.")
            metadata_to_process = metadata[metadata['run_accession'].isin(runs_to_analyze_accs)]
            if metadata_to_process.empty:
                logger.warning(f"Metadata filtering resulted in no runs left for analysis (Set was {runs_to_analyze_accs}). Check metadata consistency.")
                # If no runs to analyze but cached predictions exist, proceed. Otherwise fail.
                if not run_predictions: logger.error("No runs to analyze and no cached successful predictions. Cannot partition."); return {}
                # If cached predictions exist, continue to partition stage below
            else:
                # --- Download and Analyze Uncached Runs ---
                seq_raw_dir = self.project_dir.raw_data / dataset / "seqs" / "raw"
                fetcher = SequenceFetcher(fastq_dir=str(seq_raw_dir))
                # Select only the runs needing download based on ENA info
                ena_runs_to_download = ena_runs[ena_runs.index.isin(metadata_to_process["run_accession"])]

                if not ena_runs_to_download.empty:
                    logger.info(f"Downloading FASTQ files for {len(ena_runs_to_download)} runs...")
                    downloaded_paths = fetcher.download_run_fastq_concurrent(ena_runs_to_download)
                    # --- File Size Check ---
                    MAX_FILE_SIZE_GB = self.config.sequences.max_file_size_gb
                    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_GB * (1024**3)
                    oversized_file_found = False
                    for run_id, file_list in downloaded_paths.items():
                        if run_id not in runs_to_analyze_accs: continue # Skip if somehow downloaded but not needed
                        for file_path_str in file_list:
                            file_path = Path(file_path_str)
                            if file_path.is_file(): # Check if file exists after download attempt
                                try:
                                    file_size = file_path.stat().st_size
                                    if file_size > MAX_FILE_SIZE_BYTES:
                                        logger.warning(
                                            f"SKIPPING DATASET {dataset}: File '{file_path.name}' ({run_id}) is "
                                            f"{format_bytes(file_size)}, exceeding limit of "
                                            f"{MAX_FILE_SIZE_GB} GB. Likely metagenomic."
                                        )
                                        # Record failure and cache it
                                        self.failed.append({
                                            "dataset": dataset,
                                            "error": f"File size exceeded limit: {format_bytes(file_size)} ({file_path.name})"
                                        })
                                        self.cache.add_failed_dataset(
                                            dataset_id=dataset,
                                            reason=f"File size exceeded limit ({file_path.name})"
                                        )
                                        oversized_file_found = True
                                        break # Stop checking files for this run
                                except FileNotFoundError: logger.warning(f"File listed for download but not found after attempt: {file_path}")
                            else:
                                logger.warning(f"Downloaded path does not exist or is not a file: {file_path}")
                        if oversized_file_found: break # Stop checking runs for this dataset
                    if oversized_file_found:
                        # Clean up potentially downloaded files for the failed dataset
                        paths_to_clean = {k: [Path(f) for f in v] for k, v in downloaded_paths.items()}
                        self._cleanup_raw_files(dataset, paths_to_clean)
                        return {} # Stop processing this dataset entirely

                    # Store paths only for successfully downloaded runs needing analysis
                    # Ensure paths are valid Path objects and exist
                    run_file_paths_for_analysis.update({
                        k: [Path(f) for f in v if Path(f).is_file()] # Check is_file again
                        for k, v in downloaded_paths.items()
                        if k in runs_to_analyze_accs and v and any(Path(f).is_file() for f in v) # Ensure list not empty and contains file
                    })
                    # Log if some downloads seem to have failed silently or resulted in non-files
                    missing_paths = runs_to_analyze_accs - set(run_file_paths_for_analysis.keys())
                    if missing_paths:
                        logger.warning(f"Could not confirm valid downloaded files for runs: {missing_paths}. They will be excluded from analysis.")
                        # Update runs_to_analyze_accs to only include those with files
                        runs_to_analyze_accs = set(run_file_paths_for_analysis.keys())

                else: # No ENA info for runs needing analysis
                    logger.warning(f"No ENA run download information found for the {len(runs_to_analyze_accs)} runs needing analysis.")
                    # Check if local files exist as a fallback? Requires consistent naming.
                    logger.info(f"Attempting to find local files in {seq_raw_dir}...")
                    found_local_count = 0
                    runs_without_local_files = set()
                    for run_id in runs_to_analyze_accs: # Iterate original set
                        local_files = sorted(seq_raw_dir.glob(f"{run_id}*.fastq.gz"))
                        valid_local_files = [p for p in local_files if p.is_file()]
                        if valid_local_files:
                            run_file_paths_for_analysis[run_id] = valid_local_files
                            found_local_count += 1
                        else:
                            logger.warning(f"No local FASTQ files found for run {run_id}. Excluding from analysis.")
                            runs_without_local_files.add(run_id)

                    logger.info(f"Found local files for {found_local_count} runs.")
                    # Update the set needing analysis if local files were missing
                    runs_to_analyze_accs -= runs_without_local_files

                    if not run_file_paths_for_analysis: # If still no files after local check
                        # If no files found locally either, proceed only if cached predictions exist
                        if not run_predictions:
                            logger.error(f"Cannot download or find local files for runs needing analysis in {dataset}, and no cached predictions available. Cannot partition.")
                            # Cache these runs as failed?
                            for run_id in runs_to_analyze_accs: self.cache.add_run_status(run_id, dataset, 'FAILED_MISSING_FILES')
                            return {}
                        else: logger.warning("Proceeding to partitioning using only cached predictions as no analyzable files were found.")

                # --- Run Sequence Analysis ---
                if run_file_paths_for_analysis: # Only run if there are files to analyze
                    logger.info(f"Running sequence analysis for {len(run_file_paths_for_analysis)} runs...")
                    analysis_reports = await run_comprehensive_analysis(
                        run_file_paths=run_file_paths_for_analysis,
                        output_dir=est_dir, # Output analysis logs/temp files here
                        vsearch_db=self.config.paths.vsearch_db,
                        primer_db_path=self.config.paths.primer_db,
                        region_to_pairs_map=self.region_to_pairs_map,
                        threads=self.config.sequences.validate_16s.n_threads,
                        max_concurrency=self.config.sequences.validate_16s.concurrent_jobs
                    )
                    self._log_prediction_summary(analysis_reports)

                    # Add new predictions and cache them
                    analysis_success_count = 0
                    for run_id, report in analysis_reports.items():
                        prediction = report.get('prediction', 'Undetermined') # Default if missing
                        # Determine status based on prediction
                        status = 'SUCCESS' if prediction != "Undetermined" else 'UNDETERMINED'
                        # Handle potential analysis failures reported
                        if report.get('status') == 'FAILED':
                            status = 'FAILED_ANALYSIS'
                            logger.warning(f"Analysis failed for run {run_id}. Reason: {report.get('reasoning', 'Unknown')}")

                        # Dump report to YAML string for caching
                        try: report_yaml_str = yaml.dump(report, default_flow_style=False, sort_keys=False, indent=2)
                        except Exception as dump_e:
                            logger.error(f"Failed to dump analysis report to YAML for {run_id}: {dump_e}")
                            report_yaml_str = f"Error dumping report: {dump_e}"

                        # Add to predictions list *only* if analysis status is SUCCESS
                        if status == 'SUCCESS':
                            run_predictions.append({'run_accession': run_id, 'predicted_region': prediction})
                            analysis_success_count += 1
                        # Cache the status (SUCCESS, UNDETERMINED, FAILED_ANALYSIS)
                        self.cache.add_run_status(run_id, dataset, status, prediction, report_yaml_str)
                    logger.info(f"Sequence analysis complete. Successful predictions for {analysis_success_count} runs.")
                else: logger.info("Skipping sequence analysis step as no new FASTQ files were available/found.")

        # --- Partitioning Stage (using all successful predictions: cached + new) ---
        if not run_predictions:
            logger.error(f"No successful region predictions available (cached or new) for dataset {dataset}. Cannot partition.")
            self.failed.append({"dataset": dataset, "error": "No successful region predictions"})
            # Cache this dataset failure only if no runs were successful (prevents overwriting if only *some* failed)
            # Check if *any* status exists for the dataset's runs before declaring the *dataset* failed
            if not self.cache.get_dataset_run_statuses(dataset):
                self.cache.add_failed_dataset(dataset_id=dataset, reason="No successful region predictions")
            return {} # Stop processing

        predictions_df = pd.DataFrame(run_predictions)
        # Merge predictions with the *original* full metadata for partitioning
        # Use how='inner' to keep only runs that are in both metadata and have successful predictions
        successful_metadata = pd.merge(metadata, predictions_df, on='run_accession', how='inner')

        if successful_metadata.empty: logger.error(f"Merging successful predictions with metadata resulted in empty DataFrame for {dataset}. Check 'run_accession' consistency."); return {}

        # --- Gather File Paths for Successful Runs (Needed for QIIME) ---
        # We need paths for ALL successfully predicted runs (cached + new) if they exist locally
        fastq_dir = self.project_dir.raw_data / dataset / "seqs" / "raw"
        all_partition_run_paths: Dict[str, List[Path]] = {} 
        runs_missing_files = set() 

        for run_id in successful_metadata['run_accession']:
            # Check paths obtained from analysis step
            if run_id in run_file_paths_for_analysis:
                paths = run_file_paths_for_analysis[run_id]
                if isinstance(paths, Path): paths = [paths]
                
                valid_paths = [p for p in paths if p.is_file()]
                if len(valid_paths) == len(paths) and valid_paths: 
                    all_partition_run_paths[run_id] = valid_paths
                else:
                    logger.warning(f"Files previously listed for {run_id} are now missing. Excluding.")
                    runs_missing_files.add(run_id)
            else:
                # --- LOGIC FOR CACHED RUNS ---
                # Search for files if run prediction came from cache
                paths = sorted([p for p in fastq_dir.glob(f"{run_id}*.fastq.gz") if p.is_file()])
                
                if paths:
                    all_partition_run_paths[run_id] = paths
                else:
                    # --- NEW: ATTEMPT REDOWNLOAD LOGIC ---
                    logger.info(f"Cache hit for {run_id} but FASTQ is missing locally. Attempting redownload...")
                    
                    # Check if we have ENA info to perform a download
                    if run_id in ena_runs.index:
                        try:
                            # Initialize fetcher locally
                            fetcher = SequenceFetcher(fastq_dir=str(fastq_dir))
                            
                            # Get the specific row for this run
                            run_info_df = ena_runs.loc[[run_id]]
                            
                            # Attempt download
                            downloaded = fetcher.download_run_fastq_concurrent(run_info_df)
                            
                            # Verify the result
                            new_paths = [Path(p) for p in downloaded.get(run_id, []) if Path(p).is_file()]
                            
                            if new_paths:
                                logger.info(f"Redownload successful for {run_id}.")
                                all_partition_run_paths[run_id] = new_paths
                            else:
                                logger.warning(f"Redownload attempted for {run_id} but files still missing. Excluding.")
                                runs_missing_files.add(run_id)
                        except Exception as e:
                            logger.error(f"Error during redownload attempt for {run_id}: {e}")
                            runs_missing_files.add(run_id)
                    else:
                        logger.warning(f"FASTQ missing for {run_id} and no ENA info available to redownload. Excluding.")
                        runs_missing_files.add(run_id)
                        
        # Filter metadata again to remove runs with missing files before grouping
        if runs_missing_files:
            logger.info(f"Removing {len(runs_missing_files)} runs from partitioning due to missing files.")
            successful_metadata = successful_metadata[~successful_metadata['run_accession'].isin(runs_missing_files)]
            if successful_metadata.empty:
                logger.error(f"All successfully predicted runs for {dataset} are missing required FASTQ files. Cannot partition.")
                # Cache these runs as failed partitioning?
                for run_id in runs_missing_files: self.cache.add_run_status(run_id, dataset, 'FAILED_MISSING_FILES')
                return {}

        # --- Group by Partition Criteria and Process ---
        group_columns = ["predicted_region", "library_layout", "instrument_platform"]
        # Ensure grouping columns exist in the filtered metadata
        missing_group_cols = [col for col in group_columns if col not in successful_metadata.columns]
        if missing_group_cols:
            # Try to fill missing layout/platform from 'info' dict as fallback before failing
            for col in missing_group_cols:
                if col in info and pd.notna(info[col]):
                    logger.warning(f"Grouping column '{col}' missing in metadata, filling from dataset info.")
                    successful_metadata[col] = info[col]
                else:
                    logger.error(f"Cannot group for partitioning: Critical column '{col}' missing in metadata and dataset info for {dataset}.")
                    self.failed.append({"dataset": dataset, "error": f"Missing critical grouping column: {col}"})
                    return {}

        runs_in_final_partitions = set() # Track runs successfully included
        tasks = [] # Async tasks for processing each valid partition
        # Instantiate the processor once for this dataset
        processor = PartitionProcessor(self.config, self.project_dir, self.nfc_handler, self.nfc_facilities_df)

        logger.info(f"Grouping {len(successful_metadata)} runs with files by {group_columns}...")
        # Use dropna=False to potentially handle cases where platform/layout might be missing but region is known
        for group_keys, group_df in successful_metadata.groupby(group_columns, dropna=False):
            # Handle potential NaN grouping keys
            predicted_region, layout, platform = group_keys
            # Convert Nones/NaNs to 'NA' string for key generation and logging
            region_str = str(predicted_region) if pd.notna(predicted_region) else "NA"
            layout_str = str(layout) if pd.notna(layout) else "NA"
            platform_str = str(platform) if pd.notna(platform) else "NA"
            group_key_str = f"Region:{region_str}/Layout:{layout_str}/Platform:{platform_str}"

            # Apply minimum run threshold check
            if len(group_df) < self.MIN_RUNS_THRESHOLD:
                logger.warning(f"Skipping partition {group_key_str}: {len(group_df)} runs < {self.MIN_RUNS_THRESHOLD} required.")
                # Cache these runs as failed partitioning due to size
                for run_id in group_df['run_accession']: self.cache.add_run_status(run_id, dataset, 'FAILED_PARTITIONING_SIZE')
                continue # Skip this small partition

            # Define parameters for this partition
            params = self._process_partition(group_df, dataset, layout_str, platform_str, region_str, "N/A", "N/A")
            subset_id = self._generate_subset_id(params)
            anndata_path = self.project_dir.processed_data / f"{subset_id}.h5ad"

            logger.info(f"Queueing partition for processing: {subset_id} ({len(group_df)} runs)")
            # --- START FIX ---
            # Create a dictionary of paths ONLY for the runs in this partition
            partition_paths = {
                run: paths 
                for run, paths in all_partition_run_paths.items() 
                if run in group_df['run_accession'].values
            }
            
            # Update the call to use 'partition_paths' instead of 'all_partition_run_paths'
            tasks.append(processor.process_partition(
                group_df.copy(), 
                dataset, 
                params, 
                partition_paths,  # <--- CHANGED FROM all_partition_run_paths
                ena_runs, 
                subset_id, 
                anndata_path
            ))
            # --- END FIX ---
            # Track runs included in potentially successful partitions
            runs_in_final_partitions.update(group_df['run_accession'].tolist())

        # Execute partition processing tasks concurrently
        processed_paths_count = 0
        if tasks:
            logger.info(f"Executing processing for {len(tasks)} partitions concurrently...")
            results = await asyncio.gather(*tasks, return_exceptions=True) # Catch errors per task
            for i, result in enumerate(results):
                # Need to map result back to subset_id for better error reporting
                # For now, just collect successful paths and log errors generally
                if isinstance(result, Path):
                    self.processed_h5ad_paths.append(result)
                    processed_paths_count += 1
                elif isinstance(result, Exception):
                    # Log error from gather results
                    # Try to get subset_id if possible (assuming order is preserved and params accessible)
                    failed_subset_id = "unknown_partition" # Fallback
                    try:
                        # This relies on tasks preserving order and accessing params used to create them
                        # A more robust way would be to wrap tasks or return tuples (subset_id, result)
                        failed_params = tasks[i].__self__.params # Accessing internals - might break
                        failed_subset_id = self._generate_subset_id(failed_params)
                    except Exception: pass # Ignore errors trying to get subset_id
                    logger.error(f"Partition processing task failed (subset: {failed_subset_id}): {result}", exc_info=True) # Include traceback
                    self.failed.append({"dataset": failed_subset_id, "error": str(result)})

                # Else: Task returned None (likely internal failure logged by processor)

            logger.info(f"Finished processing partitions for {dataset}. Successful AnnData files created: {processed_paths_count}/{len(tasks)}.")
        else: logger.info(f"No valid partitions met criteria for processing for dataset {dataset}.")

        # Cache status for runs that were successfully predicted but didn't make it into a *final* partition
        # This includes runs missing files, or in groups below threshold
        all_successfully_predicted_runs = set(predictions_df['run_accession'])
        failed_partitioning_runs = all_successfully_predicted_runs - runs_in_final_partitions
        if failed_partitioning_runs:
            logger.info(f"Caching {len(failed_partitioning_runs)} runs for {dataset} as 'FAILED_PARTITIONING' (missing files or small group).")
            for run_id in failed_partitioning_runs:
                # Check existing status before overwriting; don't overwrite if already failed earlier
                current_status_info = self.cache.get_dataset_run_statuses(dataset).get(run_id)
                if not current_status_info or not current_status_info.get('status','').startswith('FAILED'):
                    # Determine specific reason if possible
                    reason_status = 'FAILED_MISSING_FILES' if run_id in runs_missing_files else 'FAILED_PARTITIONING_SIZE'
                    self.cache.add_run_status(run_id, dataset, reason_status)

        # Return the paths confirmed to exist for runs included in processed partitions
        # This dict is used for potential cleanup later
        return {run: paths for run, paths in all_partition_run_paths.items() if run in runs_in_final_partitions}

    async def manual(
        self, dataset: str, info: Dict[str, Any], metadata: pd.DataFrame,
        ena_runs: pd.DataFrame, citations: List[str]
    ) -> Dict[str, List[Path]]:
        """Handles partitioning for datasets in 'manual' primer mode."""
        self._log_dataset_summary(dataset, info, citations)
        group_columns = ["library_layout", "instrument_platform"] # Group by tech specs

        # Extract required primer/region info from the dataset info dict
        fwd_primer = info.get(self.FWD_PRIMER_COL)
        rev_primer = info.get(self.REV_PRIMER_COL)
        target_subfragment = info.get("target_subfragment") # Assumed key name
        if not all([fwd_primer, rev_primer, target_subfragment]):
            # Log specific missing info
            missing = [k for k,v in {self.FWD_PRIMER_COL: fwd_primer, self.REV_PRIMER_COL: rev_primer, "target_subfragment": target_subfragment}.items() if not v]
            error_msg = f"Manual mode dataset '{dataset}' requires '{self.FWD_PRIMER_COL}', '{self.REV_PRIMER_COL}', and 'target_subfragment' in dataset info. Missing: {missing}."
            logger.error(error_msg)
            self.failed.append({"dataset": dataset, "error": error_msg})
            # Cache failure
            self.cache.add_failed_dataset(dataset_id=dataset, reason=f"Missing manual params: {missing}")
            return {} # Stop processing this dataset

        # --- Download Files (if ENA info available) ---
        seq_raw_dir = self.project_dir.raw_data / dataset / "seqs" / "raw"
        fetcher = SequenceFetcher(fastq_dir=str(seq_raw_dir))
        # Select runs from ENA info that are actually present in the provided metadata
        runs_in_metadata = set(metadata["run_accession"])
        ena_runs_to_download = ena_runs[ena_runs.index.isin(runs_in_metadata)]

        run_file_paths: Dict[str, List[Path]] = {} # Store paths for all runs in metadata
        if not ena_runs_to_download.empty:
            logger.info(f"Downloading FASTQ files for {len(ena_runs_to_download)} runs in manual dataset {dataset}...")
            downloaded_paths = fetcher.download_run_fastq_concurrent(ena_runs_to_download)
            # Validate downloaded paths
            for k, v in downloaded_paths.items():
                # Ensure v is a list and files exist
                valid_paths = [Path(f) for f in v if isinstance(v, list) and Path(f).is_file()]
                if valid_paths: run_file_paths[k] = valid_paths
                else: logger.warning(f"Download reported for run {k} but no valid files found.")
        else: logger.warning(f"No ENA run download information found for runs in manual dataset {dataset}. Attempting to find local files.")

        # --- Attempt to Find Local Files for ALL runs in metadata (downloaded or not) ---
        logger.info(f"Verifying/finding local FASTQ files for all {len(runs_in_metadata)} runs in {seq_raw_dir}...")
        runs_missing_files_final = set()
        for run_id in runs_in_metadata:
            if run_id not in run_file_paths: # If not already populated by download
                local_files = sorted(seq_raw_dir.glob(f"{run_id}*.fastq.gz"))
                valid_local_files = [p for p in local_files if p.is_file()]
                if valid_local_files:
                    run_file_paths[run_id] = valid_local_files
                else:
                    logger.warning(f"No local FASTQ files found for run {run_id}. Excluding.")
                    runs_missing_files_final.add(run_id)
        logger.info(f"Found/verified local files for {len(run_file_paths)} runs.")

        # Filter metadata to only include runs with files found
        if runs_missing_files_final:
            metadata_with_files = metadata[~metadata['run_accession'].isin(runs_missing_files_final)].copy()
            if metadata_with_files.empty:
                logger.error(f"No runs with FASTQ files found for manual dataset {dataset}. Cannot partition.")
                # Cache failure?
                self.cache.add_failed_dataset(dataset_id=dataset, reason="No FASTQ files found for runs")
                return {}
        else: metadata_with_files = metadata.copy()

        # --- Group by Tech Specs and Process Partitions ---
        tasks = []
        processor = PartitionProcessor(
            self.config, self.project_dir, self.nfc_handler, self.nfc_facilities_df
        )
        runs_in_final_partitions = set() # Track runs included

        # Check if grouping columns exist
        missing_group_cols = [col for col in group_columns if col not in metadata_with_files.columns]
        if missing_group_cols:
            logger.error(f"Cannot group for manual partitioning: Missing columns {missing_group_cols} in metadata for {dataset}.")
            self.failed.append({"dataset": dataset, "error": f"Missing grouping columns: {missing_group_cols}"})
            return {}

        logger.info(f"Grouping {len(metadata_with_files)} runs by {group_columns}...")
        for group_keys, group_df in metadata_with_files.groupby(group_columns, dropna=False):
            layout, platform = group_keys
            # Handle potential None/NaN layout/platform if dropna=False is used
            layout_str = str(layout) if pd.notna(layout) else "NA"
            platform_str = str(platform) if pd.notna(platform) else "NA"
            group_key_str = f"{layout_str}/{platform_str}"

            if len(group_df) < self.MIN_RUNS_THRESHOLD:
                logger.warning(f"Skipping manual partition {group_key_str}: {len(group_df)} runs < {self.MIN_RUNS_THRESHOLD} required.")
                # Cache these runs as failed?
                for run_id in group_df['run_accession']: self.cache.add_run_status(run_id, dataset, 'FAILED_PARTITIONING_SIZE')
                continue

            # Define parameters for this partition (primers/region are fixed)
            params = self._process_partition(
                group_df, dataset, layout_str, platform_str, str(target_subfragment),
                str(fwd_primer), str(rev_primer)
            )
            subset_id = self._generate_subset_id(params)
            anndata_path = self.project_dir.processed_data / f"{subset_id}.h5ad"

            logger.info(f"Queueing manual partition for processing: {subset_id} ({len(group_df)} runs)")
            # --- START FIX ---
            partition_paths = {
                run: paths 
                for run, paths in run_file_paths.items() 
                if run in group_df['run_accession'].values
            }

            tasks.append(processor.process_partition(
                group_df.copy(), 
                dataset, 
                params, 
                partition_paths, # <--- CHANGED FROM run_file_paths
                ena_runs, 
                subset_id, 
                anndata_path
            ))
            # --- END FIX ---
            runs_in_final_partitions.update(group_df['run_accession'].tolist())

        # Execute tasks
        processed_paths_count = 0
        if tasks:
            logger.info(f"Executing processing for {len(tasks)} manual partitions concurrently...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Path):
                    self.processed_h5ad_paths.append(result)
                    processed_paths_count += 1
                elif isinstance(result, Exception):
                    logger.error(f"Manual partition processing task failed: {result}", exc_info=True)
                    # Add failure (subset ID might be tricky to get reliably here)
            logger.info(f"Finished processing manual partitions for {dataset}. Successful AnnData files: {processed_paths_count}/{len(tasks)}.")
        else: logger.info(f"No valid manual partitions met criteria for processing for dataset {dataset}.")

        # Cache status for runs that didn't make it into a final partition
        failed_partitioning_runs = runs_in_metadata - runs_in_final_partitions
        if failed_partitioning_runs:
            logger.info(f"Caching {len(failed_partitioning_runs)} runs for manual dataset {dataset} as 'FAILED_PARTITIONING'.")
            for run_id in failed_partitioning_runs:
                current_status_info = self.cache.get_dataset_run_statuses(dataset).get(run_id)
                # Only cache failure if not already failed for another reason
                if not current_status_info or not current_status_info.get('status','').startswith('FAILED'):
                    reason_status = 'FAILED_MISSING_FILES' if run_id in runs_missing_files_final else 'FAILED_PARTITIONING_SIZE'
                    self.cache.add_run_status(run_id, dataset, reason_status)

        # Return paths only for runs included in processed partitions (for potential cleanup)
        return {run: paths for run, paths in run_file_paths.items() if run in runs_in_final_partitions}

    async def process(self, dataset: str, info: Dict[str, Any], ena_metadata: pd.DataFrame) -> None:
        """Processes a single dataset: filters, updates info, dispatches to auto/manual."""
        files_to_cleanup: Dict[str, List[Path]] = {} # Track files downloaded for cleanup
        try:
            citations = []
            # --- Publication Fetching (only for ENA pattern IDs) ---
            if self.ENA_PATTERN.match(dataset):
                logger.info(f"Searching for publications linked to ENA project {dataset}...")
                try:
                    # --- FIX: Access config attribute directly ---
                    use_cache_flag = getattr(self.config, 'use_publication_cache', True) # Default True
                    # --- END FIX ---
                    publications = await asyncio.to_thread(
                        self.publication_fetcher.extract_bioproject_sequencing_info,
                        dataset, use_cache=use_cache_flag
                    )
                except AttributeError as e: logger.error(f"Configuration error during publication fetch setup for {dataset}: {e}. Check config schema."); publications = []
                except Exception as pub_e: logger.error(f"Publication fetch failed for {dataset}: {pub_e}", exc_info=True); publications = [] # Treat as no publications found

                # --- Host Filter (Publication) ---
                if publications: # Check if list is not empty
                    pub_info = publications[0] # Use the first publication found
                    search_text = f"{pub_info.get('study_title', '')} {pub_info.get('study_abstract', '')}".lower()

                    # Use imported 'exclusion_keywords' constant and 'is_host_associated' utility
                    found_keyword = is_host_associated(search_text, exclusion_keywords)
                    if found_keyword:
                        reason = f"Filtered (Publication): Host keyword '{found_keyword}'"
                        logger.warning(f"{reason} in dataset '{dataset}'. Skipping.")
                        self.failed.append({"dataset": dataset, "error": reason})
                        self.cache.add_failed_dataset(dataset_id=dataset, reason=reason)
                        return # Skip this dataset entirely

                    # Format citations if filter passed
                    citations = [
                        f"{p.get('publication_title', 'N/A')} (DOI: {p.get('doi', 'N/A') or p.get('pmid', 'N/A')})"
                        for p in publications
                    ]
                    # Log extracted keywords from primary publication if available
                    if oldest_pub := publications[0]:
                        if extracted := oldest_pub.get("extracted_info"):
                            logger.info(f"Primary publication keywords for {dataset}:")
                            for key, values in extracted.items(): 
                                logger.info(f"   - {key.replace('_', ' ').title()}: {', '.join(values)}")

            # --- Host Filter (Metadata) ---
            if not ena_metadata.empty:
                logger.info(f"Scanning metadata ({len(ena_metadata)} rows) for host keywords...")
                metadata_matches = find_keyword_matches(ena_metadata, exclusion_keywords)
                if metadata_matches:
                    first_match = metadata_matches[0]
                    reason = f"Filtered (Metadata): Host keyword '{first_match['keyword']}' found in column '{first_match['column']}'"
                    # Log details of first few matches
                    summary = [ f"'{m['keyword']}' in '{m['column']}' ({m['count']}x, e.g., '{str(m['example_context'])[:50]}...')" for m in metadata_matches[:3]]
                    logger.warning(f"{reason} for dataset '{dataset}'. First few matches: {'; '.join(summary)}. Skipping.")
                    self.failed.append({"dataset": dataset, "error": reason})
                    self.cache.add_failed_dataset(dataset_id=dataset, reason=reason)
                    return # Skip this dataset entirely
            else: logger.warning(f"Metadata DataFrame for {dataset} is empty, cannot perform metadata host keyword scan.")
            logger.info("Host-association checks passed (or metadata was empty).")

            # --- Technical Strategy Filters (WGS, RNA-Seq, Metatranscriptomic) ---
            if not ena_metadata.empty:
                # 1. Define exclusions
                # Strategies to exclude
                exclude_strategies = {'WGS', 'RNA-SEQ', 'MIRNA-SEQ', 'NCRNA-SEQ', 'SSRNA-SEQ'}
                # Sources to exclude
                exclude_sources = {'METATRANSCRIPTOMIC', 'TRANSCRIPTOMIC'}

                # 2. Identify rows to drop
                mask_to_drop = pd.Series(False, index=ena_metadata.index)

                # Check library_strategy
                if "library_strategy" in ena_metadata.columns:
                    mask_to_drop |= ena_metadata["library_strategy"].astype(str).str.upper().isin(exclude_strategies)

                # Check library_source
                if "library_source" in ena_metadata.columns:
                    mask_to_drop |= ena_metadata["library_source"].astype(str).str.upper().isin(exclude_sources)

                drop_count = mask_to_drop.sum()

                if drop_count > 0:
                    # 3. Apply Filter
                    ena_metadata = ena_metadata[~mask_to_drop]
                    logger.info(f"Filtered {drop_count} samples (WGS/RNA-Seq/Metatranscriptomic) from dataset {dataset}. {len(ena_metadata)} samples remaining.")

                    # 4. Check if dataset is now empty
                    if ena_metadata.empty:
                        reason = "Filtered (Technical): All samples were WGS, RNA-Seq, or Metatranscriptomic."
                        logger.warning(f"{reason} Skipping dataset {dataset}.")
                        self.failed.append({"dataset": dataset, "error": reason})
                        self.cache.add_failed_dataset(dataset_id=dataset, reason=reason)
                        return # Skip this dataset entirely
                    
            # Create a DataFrame indexed by run_accession if metadata isn't empty
            ena_runs = ena_metadata.set_index("run_accession", drop=False) if not ena_metadata.empty and "run_accession" in ena_metadata.columns else pd.DataFrame()

            if self.ENA_PATTERN.match(dataset):
                # Check if essential info fields are missing or clearly placeholder 'N/A'
                needs_update = False
                check_fields = ['description', 'instrument_platform', 'instrument_model', 'library_layout']
                for field in check_fields:
                    # Check if field missing, is NaN, or is exactly 'N/A' string
                    if field not in info or pd.isna(info.get(field)) or info.get(field) == 'N/A': needs_update = True; break

                if needs_update and not ena_metadata.empty:
                    logger.info(f"Attempting to update missing info for {dataset} from fetched ENA metadata...")
                    mode_values = ena_metadata.mode(dropna=True) # dropna=True ignores NaN values when finding mode
                    if not mode_values.empty:
                        mode_row = mode_values.iloc[0] # Get the first mode row (Series)
                        # Try 'project_name', then 'biosample_project_name', keep original if both fail
                        desc = mode_row.get('project_name')
                        if not desc or pd.isna(desc): desc = mode_row.get('biosample_project_name') # Try fallback column
                        # Only update if a valid name was found
                        if desc and pd.notna(desc): info['description'] = desc
                        # else: keep original info['description']
                        # Safely update other fields using .get() with fallback to original value in `info` dict
                        platform = mode_row.get('instrument_platform')
                        if platform and pd.notna(platform): info['instrument_platform'] = platform
                        model = mode_row.get('instrument_model')
                        if model and pd.notna(model): info['instrument_model'] = model
                        layout = mode_row.get('library_layout')
                        if layout and pd.notna(layout): info['library_layout'] = layout
                        logger.info("Dataset info updated using ENA metadata modes (if available).")
                    else: logger.warning(f"Could not determine mode values from ENA metadata for {dataset} to update info.")
                elif needs_update: logger.warning(f"Dataset info missing/NA for ENA project {dataset}, but no ENA metadata available to update it.")

            # --- Add Citations to Metadata ---
            if citations and not ena_metadata.empty:
                publication_string = "; ".join(citations)
                # Add/update 'publications' column; use .assign() to avoid SettingWithCopyWarning
                ena_metadata = ena_metadata.assign(publications=publication_string)
                logger.info(f"Added 'publications' column to metadata for {dataset}.")

            # Log summary before dispatching
            self._log_dataset_summary(dataset, info, citations)

            # --- Dispatch based on Mode ---
            current_mode = self.mode # Use instance mode
            # Removed defunct local_config check
            if current_mode == "manual": logger.info(f"Processing '{dataset}' in MANUAL mode."); files_to_cleanup = await self.manual(dataset, info, ena_metadata, ena_runs, citations)
            elif current_mode == "auto": logger.info(f"Processing '{dataset}' in AUTO mode."); files_to_cleanup = await self.auto(dataset, info, ena_metadata, ena_runs, citations)
            else: raise ValueError(f"Invalid primer mode '{current_mode}' determined for dataset {dataset}.")

        except Exception as e:
            logger.error(f"Processing dataset {dataset} failed unexpectedly: {str(e)}", exc_info=True)
            self.failed.append({"dataset": dataset, "error": f"Unexpected error: {str(e)}"})
            # Cache dataset-level failure if it happens here
            self.cache.add_failed_dataset(dataset_id=dataset, reason=f"Unexpected error: {str(e)}")
        finally:
            # Cleanup downloaded files for this dataset regardless of success/failure
            if files_to_cleanup: self._cleanup_raw_files(dataset, files_to_cleanup)
            else: logger.debug(f"No files tracked for cleanup for dataset {dataset}.")


    def _cleanup_raw_files(self, dataset: str, run_file_paths: Dict[str, List[Path]]):
        """Removes raw FASTQ files if cleanup is enabled in config."""
        cleanup_enabled = getattr(self.config.sequences, 'cleanup_raw_files', True) # Default to True if missing
        if cleanup_enabled and run_file_paths:
            num_files = sum(len(files) for files in run_file_paths.values())
            logger.info(f"Cleaning up {num_files} raw FASTQ files for dataset {dataset}...")
            cleaned_count = 0
            error_count = 0
            for run_id, files in run_file_paths.items():
                for file_path in files:
                    if file_path.is_file():
                        try: file_path.unlink(); cleaned_count += 1; logger.debug(f"Deleted {file_path}")
                        except OSError as e: logger.error(f"Error deleting file {file_path}: {e}"); error_count += 1
            logger.info(f"Cleanup for {dataset} finished. Removed {cleaned_count} files. Encountered {error_count} errors.")
        elif not cleanup_enabled: logger.info(f"Skipping raw file cleanup for dataset {dataset} as per configuration.")
        else: logger.debug(f"No files provided for cleanup for dataset {dataset}.")


    def _process_partition(
        self, partition_df: pd.DataFrame, dataset: str, layout: str, platform: str,
        target_subfragment: str, fwd_primer: Optional[str], rev_primer: Optional[str]
    ) -> Dict[str, Any]:
        """Creates a dictionary summarizing partition parameters for the processor."""
        # Ensure primers/region are strings or None
        fwd_primer_str = str(fwd_primer) if fwd_primer else None
        rev_primer_str = str(rev_primer) if rev_primer else None
        target_subfragment_str = str(target_subfragment) if target_subfragment else None

        return {
            "dataset": dataset,
            "metadata": partition_df, # Pass the actual DataFrame slice
            "n_runs": len(partition_df),
            "library_layout": str(layout), # Ensure string type
            "instrument_platform": str(platform), # Ensure string type
            "target_subfragment": target_subfragment_str,
            "pcr_primer_fwd_seq": fwd_primer_str,
            "pcr_primer_rev_seq": rev_primer_str
        }

    def _generate_subset_id(self, subset_info: Dict[str, Any]) -> str:
        """Generates a unique, sanitized ID string for a data subset based on its parameters."""
        # Function to sanitize strings for use in filenames/IDs
        # Allow alphanumeric, hyphen, period. Replace others with underscore. Handle None.
        sanitize = lambda s: re.sub(r"[^a-zA-Z0-9\-\.]", "_", str(s)) if s is not None else "NA"

        # Build the ID parts, ensuring all are strings and sanitized
        parts = [
            sanitize(subset_info.get("dataset")),
            sanitize(subset_info.get("instrument_platform")),
            sanitize(subset_info.get("library_layout")),
            sanitize(subset_info.get("target_subfragment")),
            f"FWD_{sanitize(subset_info.get('pcr_primer_fwd_seq'))}", 
            f"REV_{sanitize(subset_info.get('pcr_primer_rev_seq'))}"
        ]
        # Join non-empty, sanitized parts with a period, convert to upper case
        return ".".join(part for part in parts if part and part != 'NA').upper()

    def _run_metaanalysis(self):
        """Runs the final phylogenetic metaanalysis if enough subsets succeeded."""
        if len(self.processed_h5ad_paths) < 2:
            logger.warning(f"Skipping metaanalysis: only {len(self.processed_h5ad_paths)} subset(s) processed successfully. Need >= 2.")
            return

        logger.info(f"--- Starting Phylogenetic Meta-Analysis ({len(self.processed_h5ad_paths)} subsets) ---")
        try:
            # Ensure the target directory for metaanalysis outputs exists
            metaanalysis_output_dir = self.project_dir.processed_data / "meta-analysis"
            metaanalysis_output_dir.mkdir(parents=True, exist_ok=True) # Create if needed

            # Call the execution function, passing the directory containing h5ad files
            execute.phylogenetic_metaanalysis(
                app_config=self.config,
                # Pass the directory where individual .h5ad files are stored
                project_dir=self.project_dir.processed_data
            )
            logger.info("--- Phylogenetic Meta-Analysis Completed Successfully ---")
        except FileNotFoundError as e: logger.error(f"Meta-analysis failed: Input file or directory not found. Check QIIME 2 installation and paths. Error: {e}", exc_info=True)
        except Exception as e: logger.error(f"Phylogenetic meta-analysis failed unexpectedly: {e}", exc_info=True)


    def _report(self):
        """Logs a summary report of successful and failed subsets/datasets."""
        n_success = len(self.processed_h5ad_paths)
        
        # Get unique item IDs from the 'dataset' key in each dict
        unique_failed_items = set(item.get("dataset", "Unknown") for item in self.failed)
        n_failed = len(unique_failed_items)

        logger.info(f"\n{'='*25} Workflow Summary {'='*25}")
        logger.info(f"Successfully processed subsets (h5ad files): {n_success}")
        logger.info(f"Datasets/Subsets with failures reported: {n_failed}")

        if self.processed_h5ad_paths:
            logger.info("Successful subset files generated:")
            subset_files = sorted([p.name for p in self.processed_h5ad_paths])
            for subset_file in subset_files: 
                logger.info(f"  - {subset_file}")

        if self.failed:
            logger.info("Failure details:")
            # Group errors by item ID for clarity
            errors_by_item = defaultdict(list)
            # Adapt to the new List[Dict] structure
            for item_dict in self.failed:
                item_id = item_dict.get("dataset", "Unknown")
                error = item_dict.get("error", "No error message")
                # Store unique errors per item
                if error not in errors_by_item[item_id]: errors_by_item[item_id].append(error)

            for item_id in sorted(errors_by_item.keys()):
                logger.info(f"  - Item '{item_id}':")
                for error in errors_by_item[item_id]:
                    # Truncate long error messages
                    error_short = (str(error)[:200] + '...') if len(str(error)) > 200 else str(error)
                    logger.info(f"    - {error_short}")
        logger.info(f"{'='*68}\n")