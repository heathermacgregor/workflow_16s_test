"""
16S rRNA Analysis Pipeline
----------------------------------------------------------------------------------------
Comprehensive workflow for analysis of 16S rRNA amplicon sequencing data.
This version includes optional functionality to identify samples near
Nuclear Fuel Cycle (NFC) facilities and integrate them into the analysis.
"""
# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import copy
import logging
import yaml
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Third-Party Imports
import anndata as ad
import asyncio
import pandas as pd
from pydantic import ValidationError

# Local Imports
from workflow_16s.api.ena.metadata_api import get_n_samples_by_bioproject_async
from workflow_16s.api.ena.metadata.cache import CacheManager as EnaCacheManager
from workflow_16s.api.nuclear_fuel_cycle.nfc import NFCFacilitiesHandler
from workflow_16s.api.qiime import execute
from workflow_16s.config_schema import AppConfig
from workflow_16s.upstream.metadata.partition import DatasetPartition
from workflow_16s.upstream.sequences.analysis import PrimerFinder
from workflow_16s.upstream.sequences.probebase import import_and_save_database
from workflow_16s.utils.dir_utils import Project
from workflow_16s.utils.logger import get_logger, setup_logging
from workflow_16s.utils.progress import get_progress_bar
from workflow_16s.utils.publication_fetcher import PublicationFetcher

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = get_logger()

# ============================= HELPER FUNCTIONS =================================== #

def load_datasets_list(path: Union[str, Path]) -> List[str]:
    """Load dataset IDs from a text file, ignoring empty/whitespace lines."""
    try:
        path = Path(path)
        if not path.is_file(): raise FileNotFoundError(f"Dataset list file not found at: {path}")
        with open(path, "r") as f: return [line.strip() for line in f if line.strip()]
    except FileNotFoundError as e: logger.error(e); return []
    except Exception as e: logger.error(f"Error reading dataset list file {path}: {e}"); return []

def load_datasets_info(tsv_path: Union[str, Path]) -> pd.DataFrame:
    """Load dataset metadata from a TSV file."""
    try:
        tsv_path = Path(tsv_path)
        if not tsv_path.is_file(): raise FileNotFoundError(f"Dataset info file not found at: {tsv_path}")
        # Specify dtype for accession to prevent numeric conversion
        df = pd.read_csv(tsv_path, sep="\t", dtype={'ena_project_accession': str, 'dataset_id': str}) 
        # Remove unnamed columns resulting from Excel spreadsheet saves
        return df.loc[:, ~df.columns.str.startswith('Unnamed')] 
    except FileNotFoundError as e: logger.error(e); return pd.DataFrame()
    except Exception as e: logger.error(f"Error reading dataset info file {tsv_path}: {e}"); return pd.DataFrame()


# ================================= MAIN CLASS ===================================== #

class Upstream:
    def __init__(self, config: AppConfig):
        self.config = config
        self.project_dir = Project(config)
        # Load datasets and ensure minimal requirements are met
        self.datasets = load_datasets_list(config.paths.dataset_list)
        self.datasets_info = load_datasets_info(config.paths.dataset_info)
        if not self.datasets: raise ValueError(f"Cannot run workflow: Dataset list '{config.paths.dataset_list}' is missing, empty, or unreadable.")
        if self.datasets_info.empty: raise ValueError(f"Cannot run workflow: Dataset info file '{config.paths.dataset_info}' is missing, empty, unreadable, or contains no valid data.")
        # Step 1: Ensure the primer database exists before doing anything else.
        self._probebase_setup()
        # Step 2: Discover all primers ONCE at startup.
        logger.info("--- Initializing Primer Discovery ---")
        primer_finder = PrimerFinder(self.config.paths.primer_db)
        self.region_to_pairs_map = primer_finder.get_primer_pairs_for_regions()
        if not self.region_to_pairs_map: raise RuntimeError("Primer discovery failed or no primer pairs found in database. Cannot proceed.")
        # Initialize NFC handler and data if enabled
        if config.nfc_facilities.enabled:
            self.nfc_handler: Optional[NFCFacilitiesHandler] = NFCFacilitiesHandler(config)
            self.nfc_facilities_df = pd.DataFrame() # Will be populated later if needed
        else:
            self.nfc_handler = None
            self.nfc_facilities_df = pd.DataFrame()
        # Initialize lists to store results
        self.processed_subsets: List[str] = []
        self.failed_subsets: List[Tuple[str, str]] = []
        self.data_objects: List[ad.AnnData] = []
        # Initialize Publication Fetcher
        self.publication_fetcher = self._initialize_publication_fetcher()

    def _probebase_setup(self):
        """Checks for the primer SQLite database and creates it if it's missing."""
        primer_db_path = Path(self.config.paths.primer_db) 
        primer_db_path.parent.mkdir(parents=True, exist_ok=True)
        # Build the primer database if it doesn't exist or is invalid
        import_and_save_database(db_path=primer_db_path) 

    def _initialize_publication_fetcher(self) -> PublicationFetcher:
        """Initializes the advanced publication fetcher with caching."""
        publication_cache_path = self.project_dir.cache / "publications.db"
        fetcher = PublicationFetcher(config=self.config, cache_path=str(publication_cache_path))
        logger.info("Advanced publication fetcher initialized.")
        return fetcher

    async def execute(self):
        """Main execution workflow orchestrating all steps."""
        logger.info("Starting upstream workflow execution...")
        await self.nfc()
        await self.sort_datasets()
        await self.process_datasets()
        self._run_metaanalysis()
        logger.info("Upstream workflow execution finished.")
        return self.data_objects

    async def nfc(self):
        """
        Fetches NFC facility data, identifies contaminated and non-contaminated
        projects, reports on the sample balance, and adds all found projects
        to the processing list.
        """
        if not self.nfc_handler: logger.info("NFC facility processing is disabled."); return
        logger.info("NFC Facility Processing:")
        self.nfc_facilities_df = await self.nfc_handler.nfc_facilities()
        if not self.config.nfc_facilities.fetch_nearby_samples: logger.info("NFC sample fetching is disabled in config."); return
        if self.nfc_facilities_df.empty: logger.warning("NFC nearby sample fetching enabled, but failed to load facility data. Skipping."); return
        # --- 1. Get Contaminated Projects and Count ---
        logger.info("Fetching 'contaminated' (NFC-nearby) projects and samples...")
        try:
            contaminated_accessions = await self.nfc_handler.get_nfc_project_accessions()
            contaminated_sample_count = await self.nfc_handler.get_contaminated_sample_count_async()
            if not contaminated_accessions:
                logger.warning("No contaminated projects found.")
                contaminated_sample_count = 0
            else: logger.info(f"Contaminated pool: {len(contaminated_accessions)} projects with {contaminated_sample_count} total samples.")
            # Add new contaminated projects to the main dataset list
            current_set = set(self.datasets)
            new_contaminated_projects = [p for p in contaminated_accessions if p not in current_set]
            if new_contaminated_projects:
                logger.info(f"Adding {len(new_contaminated_projects)} new contaminated projects to the processing list.")
                self.datasets.extend(new_contaminated_projects)
            else: logger.info("All known contaminated projects are already in the dataset list.")
        except Exception as e:
            logger.error(f"Failed during contaminated project fetching: {e}", exc_info=True)
            contaminated_sample_count = 0 
        # --- 2. Get Non-Contaminated Projects and Count ---
        logger.info("Fetching 'non-contaminated' (far from NFC) projects to balance dataset...")
        non_contaminated_sample_count = 0
        non_contaminated_projects_to_add = []
        try:
            non_contaminated_accessions = await self.nfc_handler.get_non_contaminated_project_accessions_async()
            if not non_contaminated_accessions: logger.warning("Could not find any non-contaminated projects.")
            else:
                # We need to count the samples for this new list
                ena_cache_manager = EnaCacheManager(cache_dir=self.project_dir.cache / "ena_metadata")
                current_set = set(self.datasets) # Update set to include new contaminated projects
                with get_progress_bar() as progress:
                    task = progress.add_task("Counting non-contaminated samples...", total=len(non_contaminated_accessions))
                    for proj_id in non_contaminated_accessions:
                        progress.update(task, description=f"Counting {proj_id}...")
                        if proj_id in current_set: progress.update(task, advance=1); continue # Skip if already in the list
                        try:
                            n_samples = await get_n_samples_by_bioproject_async(bioproject_accession=proj_id, email=self.config.credentials.ena_email, cache_manager=ena_cache_manager)
                            if n_samples > 0:
                                non_contaminated_sample_count += n_samples
                                non_contaminated_projects_to_add.append(proj_id)
                        except Exception as e: logger.warning(f"Failed to count samples for non-contaminated project {proj_id}: {e}")
                        progress.update(task, advance=1)
                logger.info(f"Found {len(non_contaminated_projects_to_add)} new non-contaminated projects with {non_contaminated_sample_count} total samples.")
                # Add new non-contaminated projects to the main dataset list
                if non_contaminated_projects_to_add: self.datasets.extend(non_contaminated_projects_to_add)
        except Exception as e: logger.error(f"Failed during non-contaminated project fetching: {e}", exc_info=True)
        # --- 3. Report Balance and Finalize List ---
        if contaminated_sample_count > non_contaminated_sample_count: samples_needed = contaminated_sample_count - non_contaminated_sample_count; logger.info(f"Dataset is unbalanced. Need ~{samples_needed} more non-contaminated samples for a 50/50 split.")
        else: logger.info("Dataset is balanced or has sufficient non-contaminated samples.")
        # De-duplicate and sort the final list
        self.datasets = sorted(list(set(self.datasets)))
        logger.info(f"Total projects to process (original + NFC): {len(self.datasets)}")
        logger.info("--- NFC Facility Processing Finished ---")


    async def sort_datasets(self):
        """Sorts datasets by sample count and filters invalid/small ones."""
        logger.info("Validating and counting samples for each dataset...")
        if not self.datasets: logger.warning("No datasets to process after initial loading and NFC check."); return
        logger.info(f"Initial dataset count before validation: {len(self.datasets)}")
        # Filter invalid dataset IDs 
        MAX_ID_LENGTH = 100
        original_count = len(self.datasets)
        valid_datasets = []
        for ds in self.datasets:
            # Basic type/content validation (String, not empty, reasonable length, not JSON/list literal)
            if (isinstance(ds, str) and ds.strip() and len(ds) < MAX_ID_LENGTH and not ds.strip().startswith(('[', '{'))):
                valid_datasets.append(ds.strip()) # Clean whitespace
            else: logger.warning(f"Filtering invalid dataset ID format: {ds!r}")
        invalid_count = original_count - len(valid_datasets)
        if invalid_count > 0: logger.info(f"Filtered {invalid_count} invalidly formatted dataset IDs.")
        if not valid_datasets: logger.error("No valid dataset IDs remaining after format filtering."); self.datasets = []; return
        self.datasets = valid_datasets
        logger.info(f"Dataset count after format validation: {len(self.datasets)}")
        # Use the correct CacheManager (EnaCacheManager) from the metadata sub-package
        ena_cache_manager = EnaCacheManager(cache_dir=self.project_dir.cache / "ena_metadata")
        results = []
        # Use progress bar for counting
        with get_progress_bar() as progress:
            main_task = progress.add_task("Counting samples...", total=len(self.datasets))
            for dataset_id in self.datasets:
                progress.update(main_task, description=f"Counting {dataset_id}...")
                # Determine dataset type (ENA or Manual) from info file or default to ENA
                is_known_dataset = True
                try: dataset_info = self._find_best_match(dataset_id); dataset_type = dataset_info.get('dataset_type', 'ENA').upper()
                except ValueError: dataset_type = 'ENA'; is_known_dataset = False; logger.debug(f"Dataset '{dataset_id}' not in info file, assuming ENA type for counting.")
                n_samples = -1 # Use -1 to indicate count not yet determined
                if dataset_type == 'ENA':
                    cache_key = f"metadata_{dataset_id}"
                    cached_metadata_df = await ena_cache_manager.get(cache_key)
                    # Fetch count from the cached metadata if available
                    if cached_metadata_df is not None:
                        if isinstance(cached_metadata_df, pd.DataFrame): n_samples = len(cached_metadata_df); logger.debug(f"Cache hit for {dataset_id} metadata: {n_samples} samples.")
                        else: logger.warning(f"Unexpected data type in cache for {dataset_id} ({type(cached_metadata_df)}). Re-fetching count.")
                    # Fetch count directly
                    if n_samples == -1:
                        try: n_samples = await get_n_samples_by_bioproject_async(bioproject_accession=dataset_id, email=self.config.credentials.ena_email, cache_manager=ena_cache_manager)
                        except Exception as e: logger.error(f"Failed to count samples for ENA dataset {dataset_id}: {e}"); n_samples = 0 # Assume 0 if count fails
                elif dataset_type == 'MANUAL':
                    # We can't easily count samples for MANUAL here without loading files.
                    # Assign a placeholder count (e.g., a large number or 0) and rely on the partitioner's minimum run threshold later. Using 0 for now.
                    logger.info(f"Dataset '{dataset_id}' is MANUAL, skipping ENA sample count.")
                    n_samples = 0 # Placeholder, actual check happens during partitioning
                else:
                    logger.warning(f"Unknown dataset type '{dataset_type}' for {dataset_id}. Assigning sample count 0.")
                    n_samples = 0
                results.append({'dataset_id': dataset_id, 'n_samples': n_samples, 'is_known': is_known_dataset})
                progress.update(main_task, advance=1, description=f"Counted {dataset_id} ({n_samples if n_samples >= 0 else 'N/A'} samples)")
        # Filter datasets with fewer than min_samples_threshold (only for ENA datasets)
        min_samples_threshold = 5
        filtered_results = []
        skipped_count = 0
        for item in results:
            # Determine type again, defaulting to ENA if unknown during counting
            dataset_type = 'ENA'
            if item['is_known']:
                try: dataset_type = self._find_best_match(item['dataset_id']).get('dataset_type', 'ENA').upper()
                except ValueError: pass # Keep default ENA if lookup fails again
            # Apply threshold only if ENA type and count was successful
            if dataset_type == 'ENA' and item['n_samples'] < min_samples_threshold:
                logger.info(f"Filtering out dataset '{item['dataset_id']}' (Type: ENA, Samples: {item['n_samples']} < {min_samples_threshold}).")
                skipped_count += 1
            else: filtered_results.append(item) # Keep MANUAL datasets and ENA datasets meeting threshold or with failed count (-1)
        # Sort remaining datasets by sample count (ascending - smaller first)
        self.datasets = [d['dataset_id'] for d in sorted(filtered_results, key=lambda x: x['n_samples'])]
        if skipped_count > 0: logger.info(f"Removed {skipped_count} ENA datasets with fewer than {min_samples_threshold} samples.")
        logger.info(f"Final sorted dataset order ({len(self.datasets)} datasets): {self.datasets}")

    async def process_datasets(self):
        """Orchestrates the partitioning and processing of all valid datasets."""
        if not self.datasets:  logger.warning("No datasets left to process after sorting/filtering."); return []

        # Use the correct CacheManager alias
        ena_cache_manager = EnaCacheManager(cache_dir=self.project_dir.cache / "ena_metadata")
        datasets_to_skip = ['PRJNA589635', 'PRJNA169373']
        for i, dataset_id in enumerate(self.datasets):
            if dataset_id in datasets_to_skip: continue
            logger.info(f"\n{'='*10} Processing dataset {i+1}/{len(self.datasets)}: {dataset_id} {'='*10}")
            try:
                is_nfc_added_or_unknown = False
                try: dataset_info = self._find_best_match(dataset_id)
                except ValueError:
                    logger.warning(f"Dataset '{dataset_id}' not found in info file; treating as new ENA dataset for processing.")
                    dataset_info = pd.Series({
                        'dataset_type': 'ENA',
                        'ena_project_accession': dataset_id,
                        'dataset_id': dataset_id,
                        # Essential defaults expected by partitioner
                        'description': 'N/A',
                        'instrument_platform': 'N/A',
                        'instrument_model': 'N/A',
                        'library_layout': 'N/A',
                    })
                    is_nfc_added_or_unknown = True # Mark as newly added or unknown

                local_config = copy.deepcopy(self.config)
                # If newly added or unknown, force 'auto' mode for primer finding
                if is_nfc_added_or_unknown:
                    if local_config.sequences.pcr_primers.mode != 'auto':
                        logger.info(f"Forcing 'auto' primer mode for unknown/new dataset: {dataset_id}")
                        local_config.sequences.pcr_primers.mode = 'auto'

                # Instantiate the DatasetPartition class (handles internal processing)
                partitioner = DatasetPartition(
                    config=local_config,
                    publication_fetcher=self.publication_fetcher,
                    region_to_pairs_map=self.region_to_pairs_map,
                    nfc_handler=self.nfc_handler,
                    nfc_facilities_df=self.nfc_facilities_df
                )
                # Run the partitioner for the current dataset
                # The partitioner internally handles fetching ENA metadata if needed
                successful_h5ad_paths, failed_partitions = await partitioner.run(
                    {dataset_id: dataset_info.to_dict()}, # Pass dataset info as dict
                    ena_cache_manager=ena_cache_manager # Pass cache manager for ENA calls within partitioner
                )

                # --- Process Results ---
                # Load successfully created AnnData objects
                for h5ad_path in successful_h5ad_paths:
                    try:
                        adata = ad.read_h5ad(h5ad_path)
                        # Optional: Add basic checks on loaded adata? (e.g., non-empty)
                        if adata.n_obs > 0 and adata.n_vars > 0:
                            self.data_objects.append(adata)
                            subset_id = h5ad_path.stem # Get ID from filename
                            self.processed_subsets.append(subset_id)
                            logger.info(f"Successfully loaded AnnData object: {h5ad_path} ({adata.n_obs} obs x {adata.n_vars} vars)")
                        else: raise ValueError(f"Loaded AnnData object is empty ({h5ad_path})")

                    except Exception as e:
                        subset_id = h5ad_path.stem; error_msg = f"Failed to load or validate generated AnnData object {h5ad_path}: {e}"
                        logger.error(error_msg); self.failed_subsets.append((subset_id, error_msg))

                # Record failures reported by the partitioner (can include dataset-level failures)
                self.failed_subsets.extend([(f['dataset'], f['error']) for f in failed_partitions])

            except KeyError as e: error_msg = f"A required configuration or metadata column is missing: {e}. Check config and dataset info."; logger.error(f"Error processing dataset '{dataset_id}': {error_msg}"); self.failed_subsets.append((dataset_id, error_msg))
            except Exception as dataset_error: logger.error(f"Unexpected error processing dataset '{dataset_id}': {dataset_error}", exc_info=True); self.failed_subsets.append((dataset_id, str(dataset_error)))

        # Final report after processing all datasets
        self._report()
        return self.processed_subsets # Return list of successful subset IDs

    def _find_best_match(self, dataset_id: str) -> pd.Series:
        """Finds metadata record, prioritizing 'ENA' types if duplicates exist."""
        # Normalize dataset_id for comparison
        norm_id = dataset_id.strip().upper()
        # Ensure required columns exist in self.datasets_info
        required_cols = ['dataset_id', 'ena_project_accession', 'dataset_type']
        missing_cols = [col for col in required_cols if col not in self.datasets_info.columns]
        if missing_cols: raise ValueError(f"Dataset info file is missing required columns: {missing_cols}")
        # Create boolean masks for matching IDs (case-insensitive, handle NA safely)
        ena_acc_match = (self.datasets_info['ena_project_accession'].fillna('').str.upper() == norm_id)
        dataset_id_match = (self.datasets_info['dataset_id'].fillna('').str.upper() == norm_id)
        # Combine masks: match either ena_project_accession OR dataset_id
        combined_match_mask = ena_acc_match | dataset_id_match
        # Filter the DataFrame based on the combined mask
        potential_matches = self.datasets_info[combined_match_mask].copy() # Use copy to avoid warnings on modification
        if potential_matches.empty: raise ValueError(f"No metadata match found for dataset: '{dataset_id}' in the dataset info file.")
        # Prioritize ENA type if multiple matches are found
        potential_matches['dataset_type_upper'] = potential_matches['dataset_type'].fillna('ENA').str.upper() # Handle NA type
        ena_matches = potential_matches[potential_matches['dataset_type_upper'] == 'ENA']

        if not ena_matches.empty: return ena_matches.iloc[0] # Return the first ENA match
        else: return potential_matches.iloc[0] # If no ENA matches, return the first match of any other type (e.g., MANUAL)

    def _run_metaanalysis(self):
        """Runs the final phylogenetic metaanalysis if enough subsets succeeded."""
        if len(self.processed_subsets) < 2: logger.warning(f"Skipping metaanalysis: only {len(self.processed_subsets)} subset(s) processed successfully. Need >= 2."); return
        logger.info(f"--- Starting Phylogenetic Meta-Analysis ({len(self.processed_subsets)} subsets) ---")
        try: execute.phylogenetic_metaanalysis(app_config=self.config, project_dir=self.project_dir.processed_data); logger.info("--- Phylogenetic Meta-Analysis Completed Successfully ---")
        except Exception as e: logger.error(f"Phylogenetic meta-analysis failed: {e}", exc_info=True)

    def _report(self):
        """Logs a summary report of successful and failed subsets/datasets."""
        n_success = len(self.processed_subsets)
        # Count unique failed items (datasets or specific subsets)
        unique_failed_items = set(item[0] for item in self.failed_subsets); n_failed = len(unique_failed_items)
        logger.info(f"\n{'='*20} Workflow Summary {'='*20}\nSuccessfully processed subsets generated: {n_success}\nDatasets/Subsets with failures: {n_failed}")

        if self.processed_subsets:
            logger.info("Successful subset IDs generated:")
            # Sort for consistent reporting
            for subset in sorted(self.processed_subsets): 
                logger.info(f"  - {subset}")

        if self.failed_subsets:
            logger.info("Failure details:")
            # Group errors by item ID for clarity
            errors_by_item = {}
            for item_id, error in self.failed_subsets:
                if item_id not in errors_by_item: errors_by_item[item_id] = []
                errors_by_item[item_id].append(error)

            for item_id in sorted(errors_by_item.keys()):
                # Report each unique error message once per item
                unique_errors = set(errors_by_item[item_id]); logger.info(f"  - Item '{item_id}':")
                for error in unique_errors:
                    # Truncate long error messages
                    error_short = (str(error)[:200] + '...') if len(str(error)) > 200 else str(error); logger.info(f"    - {error_short}")
        logger.info(f"{'='*60}\n")


# ==================================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 16S rRNA upstream analysis pipeline.")
    parser.add_argument(
        "-c", "--config", type=Path,
        # Default relative path assumes execution from a specific directory structure
        default=Path("/usr2/people/macgregor/amplicon/workflow_16s/config/config.yaml"),
        help="Path to the YAML configuration file for the workflow."
    )
    args = parser.parse_args()

    # Basic logging setup initially to catch early errors
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s", # Added logger name
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Get logger instance after basicConfig
    logger = get_logger() # Use a specific name

    config_path = args.config
    try:
        # --- Configuration Loading and Validation ---
        logger.info(f"Loading configuration from: {config_path.resolve()}")
        if not config_path.is_file(): raise FileNotFoundError(f"Config file not found at '{config_path.resolve()}'")
        with open(config_path, 'r') as f: config_dict = yaml.safe_load(f)
        config = AppConfig(**config_dict)
        logger.info("Configuration loaded and validated successfully.")
        # --- Project Directory Setup ---
        project_dir = Project(config)
        logger.info(f"Project directory set to: {project_dir.main.resolve()}")
        # --- Reconfigure Logging based on Config ---
        logger_path = setup_logging(log_dir_path=project_dir.logs) # Use project_dir.logs
    # --- Error Handling for Setup ---
    except FileNotFoundError as e: logger.critical(str(e)); exit(1)
    except ValidationError as e: logger.critical(f"Configuration file '{config_path.resolve()}' is invalid:\n{e}"); exit(1)
    except Exception as e: logger.critical(f"An error occurred during initialization: {e}", exc_info=True); exit(1)
    # --- Workflow Execution ---
    try:
        logger.info("Instantiating Upstream workflow processor...")
        datasets_processor = Upstream(config)
        logger.info("Starting workflow execution...")
        # Run the main async execute method
        asyncio.run(datasets_processor.execute())
        logger.info("Workflow execution finished successfully.") # Log success
    except ValueError as e: logger.critical(f"Workflow setup failed: {e}"); exit(1)
    except RuntimeError as e: logger.critical(f"Workflow runtime error: {e}"); exit(1)
    except Exception as e: logger.critical(f"Workflow execution failed with an unexpected error: {e}", exc_info=True); exit(1)