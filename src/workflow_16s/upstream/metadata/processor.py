# workflow_16s/upstream/metadata/processor.py

# Standard Library Imports
import ast
import json
import logging
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

# Local Imports
from workflow_16s.api.environmental_data.other.execute import EnvironmentalDataCollector
from workflow_16s.api.environmental_data.google.arkin_env_agents import main as arkin_env_agents
from workflow_16s.api.ena.sequences import SequenceFetcher
from workflow_16s.api.nuclear_fuel_cycle.nfc import NFCFacilitiesHandler
from workflow_16s.api.qiime import execute
from workflow_16s.config_schema import AppConfig
from workflow_16s.metadata.manager import MetadataManager, process_metadata
from workflow_16s.utils.dir_utils import Project, QIIME, RawData, SubSet
from workflow_16s.utils.logger import get_logger
from workflow_16s.downstream.adata_utils import safe_write_h5ad
from .utils import (
    create_anndata_from_qiime_artifacts, format_bytes, validate_anndata_file
)

# ==================================================================================== #

logger = get_logger()

# ============================ PARTITION PROCESSOR CLASS ============================= #

class PartitionProcessor:
    """
    Handles the processing of a single, defined metadata partition.
    
    This class is responsible for:
        1.  Enriching the partition's metadata (NFC, Env, Arkin).
        2.  Verifying and fetching required FASTQ files.
        3.  Writing QIIME2 manifest and metadata files.
        4.  Executing the QIIME2 pipeline.
        5.  Creating and validating the final AnnData (.h5ad) file.
    """
    def __init__(
        self,
        config: AppConfig,
        project_dir: Project,
        nfc_handler: Optional[NFCFacilitiesHandler] = None,
        nfc_facilities_df: Optional[pd.DataFrame] = None
    ):
        self.config = config
        self.project_dir = project_dir
        self.nfc_handler = nfc_handler
        self.nfc_facilities_df = nfc_facilities_df if nfc_facilities_df is not None else pd.DataFrame()

    def _unpack_value(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if value.size == 0: return np.nan
            elif value.size == 1: value = value.item()
            else: return value.tolist()
        if isinstance(value, list):
            if not value: return np.nan
            if len(value) == 1: value = value[0]
            else: return value
        try:
            if pd.isna(value): return np.nan
        except (TypeError, ValueError): pass
        if isinstance(value, str):
            s = value.strip()
            if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')):
                try: return self._unpack_value(ast.literal_eval(s))
                except (ValueError, SyntaxError): return value
        if isinstance(value, dict) and 'value' in value and 'unit' in value: return value.get('value')
        return value

    def _clean_and_unpack_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("🧹 Cleaning and unpacking cell values...")
        cleaned_df = df.copy()
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            if col not in ['run_accession', '#sampleid']: cleaned_df[col] = cleaned_df[col].apply(self._unpack_value)
        logger.info("✅ Unpacking complete.")
        return cleaned_df

    def _aggregate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("🏗️ Aggregating multiple rows per sample ID...")
        if 'run_accession' not in df.columns or df.empty: return df
        
        # 1. Identify columns that should definitely be flattened (SoilGrids, Meteostat)
        # We want 'first' or 'mean' for these, not lists.
        env_prefixes = ['SoilGrids_', 'Meteostat_', 'NASA_', 'OpenMeteo_']
        
        def custom_list_aggregator(series: pd.Series) -> Union[List[Any], float]:
            unique_items, seen_reps = [], set()
            all_items = []
            for item in series.dropna():
                if isinstance(item, list): all_items.extend(item)
                else: all_items.append(item)
            for item in all_items:
                try: rep = json.dumps(item, sort_keys=True)
                except TypeError: rep = str(item)
                if rep not in seen_reps:
                    seen_reps.add(rep)
                    unique_items.append(item)
            return unique_items if unique_items else np.nan

        # 2. Define aggregation strategy
        agg_dict = {}
        for col in df.columns:
            if col == 'run_accession': continue
            
            # Specific columns to list-aggregate
            if col in ['gbif_id', 'species', 'scientific_name', 'iNaturalist_observations', 'attributes', 'USGS_Earthquake_earthquakes']:
                agg_dict[col] = custom_list_aggregator
            
            # Detect Numeric Environmental Columns -> Use 'mean' (to average duplicates) or 'first'
            elif any(col.startswith(prefix) for prefix in env_prefixes) and pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = 'mean' 
            
            # Default for everything else -> 'first'
            else:
                agg_dict[col] = 'first'

        aggregated_df = df.groupby('run_accession').agg(agg_dict).reset_index()
        logger.info(f"✅ Aggregation complete. Data shape changed from {df.shape} to {aggregated_df.shape}.")
        return aggregated_df
    
    async def _ensure_fastq_files_exist(
        self, run_accessions_needed: List[str], existing_run_paths: Dict[str, List[Path]],
        dataset_id: str, ena_runs_df: pd.DataFrame
    ) -> Dict[str, List[Path]]:
        """Verifies that FASTQ files exist for a list of runs, re-downloading if necessary."""
        runs_to_redownload = []
        for run_id in run_accessions_needed:
            paths = existing_run_paths.get(run_id, [])
            if not paths or not all(p.exists() for p in paths): runs_to_redownload.append(run_id)

        if runs_to_redownload:
            logger.warning(f"{len(runs_to_redownload)} runs are missing local FASTQ files. Attempting to re-download...")
            runs_to_fetch_df = ena_runs_df[ena_runs_df.index.isin(runs_to_redownload)]
            
            if not runs_to_fetch_df.empty:
                fetcher = SequenceFetcher(fastq_dir=str(self.project_dir.raw_data / dataset_id / "seqs" / "raw"))
                downloaded_paths = fetcher.download_run_fastq_concurrent(runs_to_fetch_df)
                
                for run_id, paths in downloaded_paths.items():
                    existing_run_paths[run_id] = [Path(p) for p in paths]
                logger.info(f"✅ Successfully re-downloaded files for {len(downloaded_paths)} runs.")
            else: logger.error(f"Could not find metadata in ENA to re-download {len(runs_to_redownload)} missing runs.")
        
        return existing_run_paths

    def _write_qiime_manifest(self, subset_id: str, partition_metadata: pd.DataFrame, run_file_paths: Dict[str, List[Path]]):
        rows = []
        
        try: layout = partition_metadata['library_layout'].iloc[0].lower()
        except (KeyError, IndexError): logger.error(f"Cannot determine library layout for partition {subset_id}."); return

        run_accession_col = 'run_accession' if 'run_accession' in partition_metadata.columns else 'accession'
        if run_accession_col in partition_metadata.columns: sample_ids = partition_metadata[run_accession_col].tolist()
        else: sample_ids = partition_metadata.index.tolist(); logger.warning(f"'{run_accession_col}' not found, using index as sample IDs for manifest.")

        # Track skip reasons for better diagnostics
        skip_reasons = {'no_paths': 0, 'missing_files': 0, 'incomplete_pairs': 0}
        sample_path_examples = []  # Store examples for debugging
        
        for sample_id in sample_ids:
            paths = run_file_paths.get(str(sample_id))
            if not paths: 
                logger.warning(f"No file paths found for run {sample_id} in {subset_id}, skipping from manifest.")
                skip_reasons['no_paths'] += 1
                continue
            
            # Store example for first few samples
            if len(sample_path_examples) < 3:
                sample_path_examples.append({
                    'sample_id': sample_id,
                    'provided_paths': [str(p) for p in paths],
                    'existing_paths': [str(p) for p in paths if p.exists()]
                })
            
            existing_paths = sorted([p.resolve() for p in paths if p.exists()])
            
            if layout == 'paired':
                if len(existing_paths) < 2: 
                    logger.warning(f"Paired-end sample {sample_id} is missing one or both FASTQ files. Found: {len(existing_paths)}. Skipping from manifest.")
                    skip_reasons['incomplete_pairs'] += 1
                    continue
                if len(existing_paths) > 2: logger.warning(f"Found >2 files for paired-end sample {sample_id}, using first two.")
                rows.append({'sample-id': sample_id, 'forward-absolute-filepath': str(existing_paths[0]), 'reverse-absolute-filepath': str(existing_paths[1])})
            
            elif layout == 'single':
                if len(existing_paths) < 1: 
                    logger.warning(f"Single-end sample {sample_id} is missing its FASTQ file. Skipping from manifest.")
                    skip_reasons['missing_files'] += 1
                    continue
                rows.append({'sample-id': sample_id, 'absolute-filepath': str(existing_paths[0])})

        if not rows:
            # Provide detailed diagnostic information
            total_samples = len(sample_ids)
            error_details = f"No valid FASTQ files were found for any of the {total_samples} samples in partition {subset_id}. "
            error_details += f"Skip reasons: {skip_reasons['no_paths']} samples with no file paths, "
            error_details += f"{skip_reasons['incomplete_pairs']} incomplete paired-end samples (expected 2 files, found 1 or 0), "
            error_details += f"{skip_reasons['missing_files']} missing single-end files. "
            error_details += f"Library layout: {layout}. "
            error_details += "This usually indicates a download failure or file path mapping issue. "
            
            # Add example paths for debugging
            if sample_path_examples:
                error_details += f"\n\nExample file paths checked (first {len(sample_path_examples)} samples):"
                for example in sample_path_examples:
                    error_details += f"\n  Sample {example['sample_id']}:"
                    error_details += f"\n    Provided: {example['provided_paths']}"
                    error_details += f"\n    Exist: {example['existing_paths']}"
            
            error_details += "\n\nCannot create a manifest."
            
            logger.error(error_details)
            raise RuntimeError(error_details)

        qiime_dir = self.project_dir.qiime / subset_id
        qiime_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = qiime_dir / "manifest.tsv"
        
        pd.DataFrame(rows).to_csv(manifest_path, sep='\t', index=False)
        logger.info(f"QIIME2 manifest for '{subset_id}' written to {manifest_path}")

    def _write_sample_metadata(
        self, subset_id: str, partition_metadata: pd.DataFrame
    ):
        qiime_dir = self.project_dir.qiime / subset_id
        qiime_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_to_save = partition_metadata.copy()
        run_accession_col = 'run_accession' if 'run_accession' in metadata_to_save.columns else 'accession' 
        metadata_to_save.rename(columns={run_accession_col: 'run_accession'}, inplace=True)
        
        # 1. Convert lists/dicts to strings first
        for col in metadata_to_save.select_dtypes(include=['object']).columns:
            if any(isinstance(val, (list, dict)) for val in metadata_to_save[col].dropna()):
                logger.debug(f"Converting column '{col}' containing lists/dicts to string for TSV export.")
                metadata_to_save[col] = metadata_to_save[col].astype(str)

        # 2. Remove newlines from all string columns to prevent TSV breakage
        # We replace \r and \n with a space so words don't get merged (e.g. "Hello\nWorld" -> "Hello World")
        str_cols = metadata_to_save.select_dtypes(include=['object']).columns
        metadata_to_save[str_cols] = metadata_to_save[str_cols].replace(r'[\r\n]+', ' ', regex=True)

        if '#SampleID' in metadata_to_save.columns: metadata_to_save.drop(columns=['#SampleID'], inplace=True)
        metadata_to_save.insert(0, '#SampleID', metadata_to_save['run_accession'])
        metadata_to_save.set_index('#SampleID', inplace=True)
        
        metadata_to_save.to_csv(qiime_dir / "metadata.tsv", sep='\t')
        logger.info(f"QIIME2 sample metadata for '{subset_id}' written to {qiime_dir / 'metadata.tsv'}")
        return qiime_dir / "metadata.tsv"

    async def process_partition(self, group: pd.DataFrame, dataset: str, params: Dict[str, Any], all_run_paths: Dict[str, List[Path]], ena_runs: pd.DataFrame, subset_id: str, anndata_path: Path) -> Optional[Path]:
        """
        Executes the full processing and enrichment pipeline for a single partition.
        """
        subset_dir = SubSet(self.project_dir, subset_id)
        
        all_run_paths = await self._ensure_fastq_files_exist(
            run_accessions_needed=group['run_accession'].tolist(),
            existing_run_paths=all_run_paths, 
            dataset_id=dataset,
            ena_runs_df=ena_runs
        )
        
        subset_metadata = group.copy()
        raw_data_dir = RawData(subset_dir)
        temp_metadata_path = raw_data_dir.metadata_tsv
        
        # Use the static method from MetadataManager
        MetadataManager.export_tsv(subset_metadata, temp_metadata_path)

        logger.info(f"Running advanced cleaning and enrichment on metadata for {subset_id}...")
        try: subset_metadata = await process_metadata(subset_metadata, temp_metadata_path, self.config)
        except Exception as e: logger.error(f"Metadata cleaning failed for {subset_id}: {e}", exc_info=True)

        # NFC Handler
        if self.nfc_handler and not self.nfc_facilities_df.empty:
            try:
                logger.info(f"Performing NFC facility matching for subset: {subset_id}")
                subset_metadata = self.nfc_handler._match_facilities_with_locations(self.nfc_facilities_df, subset_metadata)
            except KeyError: logger.warning(f"Skipping NFC matching for {subset_id}: Missing location columns.")

        # --- FIX: Smart Fallback for missing collection_date ---
        # 1. Ensure collection_date exists
        if 'collection_date' not in subset_metadata.columns:
            subset_metadata['collection_date'] = pd.NaT
        else:
            subset_metadata['collection_date'] = pd.to_datetime(subset_metadata['collection_date'], errors='coerce')

        # 2. Identify potential date columns
        candidate_cols = []
        for col in subset_metadata.columns:
            if col == 'collection_date': continue
            
            # Added 'end' to the keyword search list
            is_name_match = any(x in col.lower() for x in ['date', 'time', 'start', 'end', 'created', 'timestamp'])
            is_dtype_match = is_datetime64_any_dtype(subset_metadata[col])

            if is_name_match or is_dtype_match: candidate_cols.append(col)

        # 3. Prioritize specific columns (Start -> End -> Others)
        # We insert 'end' at index 0, then 'start' at index 0, resulting in [Start, End, Others...]
        if 'collection_date_end' in candidate_cols:
            candidate_cols.insert(0, candidate_cols.pop(candidate_cols.index('collection_date_end')))
        if 'collection_date_start' in candidate_cols:
            candidate_cols.insert(0, candidate_cols.pop(candidate_cols.index('collection_date_start')))

        # 4. Iterate candidates and backfill missing values
        for col in candidate_cols:
            if subset_metadata['collection_date'].notna().all(): break
                
            missing_before = subset_metadata['collection_date'].isna().sum()
            candidate_series = pd.to_datetime(subset_metadata[col], errors='coerce')
            subset_metadata['collection_date'] = subset_metadata['collection_date'].fillna(candidate_series)
            
            filled_count = missing_before - subset_metadata['collection_date'].isna().sum()
            if filled_count > 0:
                logger.info(f"Filled {filled_count} missing 'collection_date' values using column: '{col}'")
        # -------------------------------------------------------
        MetadataManager.export_tsv(subset_metadata, temp_metadata_path)
        # EnvironmentalDataCollector
        try:
            if 'run_accession' not in subset_metadata.columns: subset_metadata.reset_index(inplace=True)
                
            if "lat" in subset_metadata.columns and "lon" in subset_metadata.columns:
                subset_metadata["lat"] = pd.to_numeric(subset_metadata["lat"], errors="coerce")
                subset_metadata["lon"] = pd.to_numeric(subset_metadata["lon"], errors="coerce")
                
                # Re-format to string for Env Collector APIs now that it is populated
                subset_metadata['collection_date'] = pd.to_datetime(subset_metadata['collection_date'], errors='coerce').dt.strftime('%Y-%m-%d')
                
                valid_rows = subset_metadata.dropna(subset=["collection_date", "lat", "lon", "run_accession"])
                
                if not valid_rows.empty:
                    logger.info(f"Found {len(valid_rows)} valid samples for env data collection.")
                    env_data_path = raw_data_dir.main / "sample_env_data.json"

                    data_collector = EnvironmentalDataCollector(
                        data=valid_rows[["collection_date", "lat", "lon", "run_accession"]], 
                        config=self.config, 
                        output_file=env_data_path,
                        verbose=self.config.verbose
                    )
                    env_df = data_collector.run_apis()

                    if env_df is not None and not env_df.empty:
                        logger.info("Merging environmental data from EnvironmentalDataCollector...")
                        subset_metadata = pd.merge(subset_metadata, env_df, on=["collection_date", "lat", "lon"], how="left")
                    else: logger.warning("EnvironmentalDataCollector returned no data; skipping merge.")
                else: logger.warning(f"No valid rows with date/lat/lon for env data collection in {subset_id}.")
            else: logger.warning(f"Skipping environmental data collection for {subset_id}: lat/lon columns missing.")
        except Exception as e: logger.error(f"Environmental data collection (EnvironmentalDataCollector) failed for {subset_id}: {e}\n{traceback.format_exc()}")
        
        # Arkin Env Agents
        try:
            arkin_df = arkin_env_agents(metadata_path=temp_metadata_path, project_dir=self.project_dir)
            
            if arkin_df is not None and not arkin_df.empty:
                logger.info(f"Merging {arkin_df.shape[0]} records from Arkin Env Agents...")
                arkin_df['run_accession'] = arkin_df['associated_sample_ids'].str.split(', ')
                arkin_df = arkin_df.explode('run_accession')
                
                merge_cols = ['sample_collection_date', 'sample_lat', 'sample_lon', 'associated_sample_ids']
                arkin_df_to_merge = arkin_df.drop(columns=[col for col in merge_cols if col in arkin_df.columns], errors='ignore')

                subset_metadata = pd.merge(subset_metadata, arkin_df_to_merge, on='run_accession', how='left', suffixes=('', '_arkin'))
                logger.info("Successfully merged data from Arkin Env Agents.")
            else: logger.warning("Arkin Env Agents returned no data; skipping merge.")
        except Exception as e: logger.error(f"Arkin environmental agents enrichment failed for {subset_id}: {e}")

        # Final metadata prep
        logger.info("Finalizing metadata: cleaning formats and aggregating to one row per sample.")
        subset_metadata = self._clean_and_unpack_data(subset_metadata)
        subset_metadata = self._aggregate_rows(subset_metadata)
        logger.info(f"Final aggregated metadata shape: {subset_metadata.shape}")

        # --- QIIME + AnnData Pipeline ---
        qiime_dir = QIIME(subset_dir)
        metadata_path = self._write_sample_metadata(subset_id, subset_metadata)
        self._write_qiime_manifest(subset_id, subset_metadata, all_run_paths)

        logger.info(f"Executing QIIME2 artifact generation workflow for {subset_id}...")
        artifact_paths = execute.seqs_to_features(
            app_config=self.config, 
            subset=params, 
            qiime_dir=qiime_dir.main,
            metadata_path=metadata_path, 
            manifest_path=qiime_dir.manifest_tsv,
            anndata_dir=anndata_path.parent,
            subset_id=subset_id
        )

        logger.info(f"Logging sizes of input files for AnnData creation:")
        input_files_to_log = {
            "Feature Table (BIOM)": artifact_paths["feature_table_biom"],
            "Taxonomy (TSV)": artifact_paths["taxonomy_tsv"],
            "Rep Sequences (FASTA)": artifact_paths["rep_seqs_fasta"],
            "Phylogenetic Tree (NWK)": artifact_paths["rooted_tree_nwk"],
            "Sample Metadata (TSV)": metadata_path
        }
        for name, path in input_files_to_log.items():
            if path.exists(): logger.info(f"   - {name+':':<28} {format_bytes(path.stat().st_size)}")
            else: logger.warning(f"   - {name}: File not found at {path}")

        logger.info(f"Creating final AnnData object for {subset_id}...")
        adata = create_anndata_from_qiime_artifacts(
            feature_table_biom_path=artifact_paths["feature_table_biom"],
            taxonomy_tsv_path=artifact_paths["taxonomy_tsv"],
            rep_seqs_fasta_path=artifact_paths["rep_seqs_fasta"],
            rooted_tree_nwk_path=artifact_paths["rooted_tree_nwk"],
            metadata_path=metadata_path
        )

        logger.info(f"Writing AnnData object to {anndata_path}")
        safe_write_h5ad(adata, anndata_path)
        
        if anndata_path.exists():
            final_size = anndata_path.stat().st_size
            logger.info(f"📦 Final AnnData file size for '{subset_id}': {format_bytes(final_size)}")
            validate_anndata_file(anndata_path, subset_id)
        else: raise FileNotFoundError(f"Failed to write AnnData file to {anndata_path}")

        return anndata_path