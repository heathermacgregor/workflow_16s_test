"""
16S rRNA Analysis Pipeline 
----------------------------------------------------------------------------------------
Comprehensive workflow for analysis of 16S rRNA amplicon sequencing (microbial 
community) data from raw data to processed results.
Primarily set up to analyze how contamination from nuclear fuel cycle (NFC) activities
affects microbial community composition.
"""
# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import argparse
import itertools
import logging
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import pandas as pd
import numba

# Local Imports
parent_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(parent_dir))

from workflow_16s import constants
from workflow_16s import ena

from workflow_16s.config import get_config
from workflow_16s.logger import setup_logging 

#from workflow_16s.amplicon_data.analysis import AmpliconData
from workflow_16s.amplicon_data.downstream.downstream import Downstream
from workflow_16s.amplicon_data.downstream.results_analysis import DownstreamResultsAnalyzer
from workflow_16s.figures.html_report import generate_html_report
from workflow_16s.metadata.per_dataset import SubsetDataset
from workflow_16s.qiime.workflows.execute_workflow import (
    execute_per_dataset_qiime_workflow as execute_qiime
)
from workflow_16s.sequences.sequence_processing import process_sequences
from workflow_16s.utils.dir_utils import SubDirs
from workflow_16s.utils.file_utils import load_datasets_list, load_datasets_info
from workflow_16s.utils.general import print_data_dicts
from workflow_16s.utils.io import (
    dataset_first_match, import_metadata_tsv, import_table_biom, load_datasets_info, 
    load_datasets_list, safe_delete, write_manifest_tsv, write_metadata_tsv
)

# ========================== INITIALIZATION & CONFIGURATION ========================== #

import workflow_16s.custom_tmp_config

pd.set_option('display.max_colwidth', None)
pd.set_option('future.no_silent_downcasting', True)

os.environ['NUMBA_NUM_THREADS'] = '8'  # Match your n_jobs setting
numba.config.NUMBA_NUM_THREADS = 8

# =================================== MAIN WORKFLOW ================================== #        

def get_existing_subsets(config, logger) -> Dict[str, Dict[str, Path]]:
    """Without running upstream processing, identify existing subsets that have
    required QIIME outputs.
    
    Args:
        config : 
            Configuration dictionary.
        logger : 
            Logger instance.
        
    Returns:
        Dictionary mapping subset IDs to dictionaries of file paths.
    """
    # Get project directory structure
    project_dir = SubDirs(config["project_dir"])
    # Get taxonomy classifier
    taxonomy_config = config["qiime2"]["per_dataset"]["taxonomy"]
    classifier = taxonomy_config.get("classifier", constants.DEFAULT_CLASSIFIER)
    # Get datasets
    datasets = load_datasets_list(config["dataset_list"])
    datasets_info = load_datasets_info(config["dataset_info"])
    
    # Initialize storage for existing subsets
    existing_subsets = {}

    # Define required files and their keys
    required_files = {
        "metadata": "sample-metadata.tsv",
        "table": "table/feature-table.biom",
        "rep_seqs": "rep-seqs/dna-sequences.fasta",
        "taxonomy": f"{classifier}/taxonomy/taxonomy.tsv",
    }
    if config["target_subfragment_mode"] == "any":
        required_files["table_6"] = "table_6/feature-table.biom"

    # Process each dataset to get expected subsets
    for dataset in datasets:
        try:
            # Get dataset info
            dataset_info = dataset_first_match(dataset, datasets_info)

            # Generate potential subsets
            subsets = SubsetDataset(config)
            subsets.process(dataset, dataset_info)
            
            for subset in subsets.success:
                # Generate consistent subset ID
                sanitize = lambda s: re.sub(r"[^a-zA-Z0-9-]", "_", s)
                subset_id = (
                    subset["dataset"] + '.' 
                    + subset["instrument_platform"].lower() + '.' 
                    + subset["library_layout"].lower() + '.' 
                    + subset["target_subfragment"].lower() + '.' 
                    + f"FWD_{sanitize(subset['pcr_primer_fwd_seq'])}_" 
                    + f"REV_{sanitize(subset['pcr_primer_rev_seq'])}"
                )
                
                # Get directory paths for this subset
                subset_dirs = project_dir.subset_dirs(subset=subset)
                subset_files = {}
                all_files_exist = True
                
                # Check each required file
                for file_key, rel_path in required_files.items():
                    if file_key == "metadata":
                        file_path = subset_dirs["metadata"] / rel_path
                    else:
                        file_path = subset_dirs["qiime"] / rel_path
                    
                    if not file_path.exists():
                        all_files_exist = False
                        break
                    subset_files[file_key] = file_path
                
                if all_files_exist:
                    existing_subsets[subset_id] = subset_files
                    logger.debug(f"Found existing outputs for subset: {subset_id}")
        
        except Exception as e:
            logger.error(f"Error processing dataset {dataset} for existing subsets: {str(e)}")
    
    logger.info(f"Found {len(existing_subsets)} completed subsets")
    return existing_subsets
    

def upstream(config, logger, project_dir) -> Union[List, None]:
    """Run the "upstream" part of the workflow (raw data to feature tables).
    
    Args:
        config : 
            Configuration dictionary.
        logger : 
            Logger instance.
    """
    qiime_config = config.get("qiime2", {})
    qiime_per_dataset_config = qiime_config.get("per_dataset", {})
    qiime_hard_rerun = qiime_per_dataset_config.get("hard_rerun", False)
    classifier = qiime_per_dataset_config.get("taxonomy", {}).get(
        "classifier", constants.DEFAULT_CLASSIFIER
    )

    success_subsets, fail_subsets = [], []
    qiime_outputs = {}
    try:
        datasets = load_datasets_list(config["dataset_list"])
        datasets_info = load_datasets_info(config["dataset_info"])
        
        for dataset in datasets:
            try:
                # Partition datasets into subsets by processing requirements 
                dataset_info = dataset_first_match(dataset, datasets_info)

                subsets = SubsetDataset(config)
                subsets.process(dataset, dataset_info)

                for subset in subsets.success:
                    try:
                        sanitize = lambda s: re.sub(r"[^a-zA-Z0-9-]", "_", s)
                        
                        # Subset identifier: 
                        # dataset -> instrument_platform -> library_layout -> target_subfragment -> FWD_SEQ_REV_SEQ
                        subset_id = (
                            subset["dataset"] + '.' 
                            + subset["instrument_platform"].lower() + '.' 
                            + subset["library_layout"].lower() + '.' 
                            + subset["target_subfragment"].lower() + '.' 
                            + f"FWD_{sanitize(subset['pcr_primer_fwd_seq'])}_" 
                            + f"REV_{sanitize(subset['pcr_primer_rev_seq'])}"
                        )

                        subset_dirs = project_dir.subset_dirs(subset=subset)

                        # Write the sample metadata TSV file
                        metadata = subset["metadata"]
                        metadata_path = subset_dirs["metadata"] / "sample-metadata.tsv"
                        write_metadata_tsv(metadata, metadata_path)

                        # If hard_rerun is not enabled, skip QIIME if the necessary outputs already exist
                        if not qiime_hard_rerun:
                            required_paths = {
                                "metadata": metadata_path,
                                "manifest": manifest_path,
                                "table": subset_dirs["qiime"] / "table" / "feature-table.biom",
                                "rep_seqs": subset_dirs["qiime"] / "rep-seqs" / "dna-sequences.fasta",
                                "taxonomy": subset_dirs["qiime"] / classifier / "taxonomy" / "taxonomy.tsv",
                                "table_6": subset_dirs["qiime"] / "table_6" / "feature-table.biom",
                            }
                            if all(p.exists() for p in required_paths.values()):
                                qiime_outputs[subset_id] = required_paths
                                success_subsets.append(subset_id)
                                logger.info(
                                    f"⏭️  Skipping processing for "
                                    f"{subset_id.replace('.', '/')} "
                                    f"- existing outputs found"
                                )
                                continue

                        seq_paths, seq_stats = process_sequences(
                            config=config,
                            subset=subset,
                            subset_dirs=subset_dirs,
                            info=dataset_info,
                        )

                        # Write the manifest TSV file
                        manifest_path = subset_dirs["qiime"] / "manifest.tsv"
                        write_manifest_tsv(seq_paths, manifest_path)

                        qiime_dir = subset_dirs["qiime"]
                        qiime_outputs = execute_qiime(
                            config, subset, qiime_dir, metadata_path, manifest_path
                        )

                        qiime_outputs[subset["dataset"]] = qiime_outputs
                        success_subsets.append(subset["dataset"])

                        # Check if clean_fastq is enabled
                        clean_fastq = config.get("clean_fastq", {}).get("enabled", True)
                        dataset_type = dataset_info.get('dataset_type', '').upper()
                        if clean_fastq and dataset_type == 'ENA':
                            dir_types = ["raw_seqs", "trimmed_seqs"]
                            for dir_type in dir_types:
                                dir_path = subset_dirs[dir_type]
                                if not dir_path.exists():
                                    continue
                                for fastq_file in dir_path.glob("*.fastq.gz"):
                                    safe_delete(fastq_file)
                            logger.info(
                                f"Cleaned up intermediate files for subset: "
                                f"{subset['dataset']}"
                            )

                    except Exception as subset_error:
                        logger.error(f"Failed processing subset {subset['dataset']}: {str(subset_error)}")
                        fail_subsets.append((subset["dataset"], str(subset_error)))

            except Exception as dataset_error:
                logger.error(f"Failed processing dataset {dataset}: {str(dataset_error)}")
                fail_subsets.append((dataset, str(dataset_error)))

        n_success_subsets = len(success_subsets)
        n_total_subsets = len(success_subsets) + len(fail_subsets)
        logger.info(
            f"Processing complete! Succeeded for {n_success_subsets} of {n_total_subsets} subsets"
        )
        if fail_subsets:
            fail_subsets_report = '\n'.join(
                ["Failure details:"] + [f"    • {dataset}: {error}" for dataset, error in fail_subsets]
            )
            logger.info(fail_subsets_report)

        metadata_dfs = [import_metadata_tsv(i['metadata']) 
                        for i in qiime_outputs.values()]
        metadata_df = pd.concat(metadata_dfs)
        # Calculate the percentage of non-null values for each column
        completeness = metadata_df.sort_index(axis=1).notna().mean() * 100
        logger.info(f"\n{completeness}")

        table_type = 'table_6' if config['target_subfragment_mode'] == 'any' else 'table'
        table_dfs = [import_table_biom(i[table_type]) 
                     for i in qiime_outputs.values()]
        table_df = pd.concat(table_dfs)
        logger.info(f"Feature table shape: {table_df.shape}")
        return success_subsets
        
    except Exception as global_error:
        logger.critical(f"Fatal pipeline error: {str(global_error)}", exc_info=True)
        raise


def run_downstream(config, logger, project_dir, existing_subsets) -> None:
    """Run the "downstream" part of the workflow (feature table analysis).

    Args:
        config :           
            Configuration dictionary.
        logger :           
            Logger instance.
        existing_subsets : 
            Successful subsets from "upstream" processing, if it was performed, 
            otherwise None.
    """
    # Get existing subsets
    if existing_subsets == None:
        if config.get("downstream", {}).get("find_subsets", False):
            existing_subsets = get_existing_subsets(config, logger)
            logger.info(f"Found {len(existing_subsets)} completed subsets")
            
    # Run downstream analysis    
    try:
        amplicon_data = Downstream(
            config=config,
            project_dir=project_dir,
            existing_subsets=existing_subsets,
            verbose=False    
        )
        analyzer = DownstreamResultsAnalyzer(
            downstream_results=amplicon_data, 
            config=config, 
            verbose=True
        )
        results = analyzer.run_comprehensive_analysis(
            output_dir=str(project_dir.final / 'comprehensive_analysis')
        )
        
    except Exception as e:
        logger.error(f"Failed processing amplicon data: {str(e)}")
        
    finally:
        print_data_dicts(amplicon_data.results)

    # Generate a comprehensive HTML report
    output_path = Path(project_dir.final) / "analysis_report_ml_minimal_run.html"
    try:
        generate_html_report(
            amplicon_data=amplicon_data.results,
            output_path=output_path,
            max_features=20,
            config=config
        )
        logger.info(f"HTML report generated at: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed generating HTML report: {str(e)}")
    

class Workflow16S:
    def __init__(self, config_path: Path = constants.DEFAULT_CONFIG) -> None:
        self.config = get_config(config_path)
        self.project_dir = SubDirs(self.config["project_dir"])
        self.logger = setup_logging(self.project_dir.logs)
        self._success_subsets: Optional[List[str]] = None

    def run(self) -> None:
        """Execute the workflow based on configuration settings."""
        try:
            self._execute_upstream()
            self._execute_downstream()
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            raise WorkflowError("Workflow aborted due to errors") from e

    def _execute_upstream(self) -> None:
        """Run upstream processing if enabled in config."""
        upstream_config = self.config.get("upstream", {})
        if upstream_config.get("enabled", False):
            self.logger.info("Starting upstream processing")
            self._success_subsets = run_upstream(
                self.config, 
                self.logger, 
                self.project_dir
            )
            self.logger.info("Upstream processing completed")

    def _execute_downstream(self) -> None:
        """Run downstream processing if enabled in config."""
        downstream_config = self.config.get("downstream", {})
        if downstream_config.get("enabled", False):
            self.logger.info("Starting downstream processing")
            run_downstream(
                self.config,
                self.logger,
                self.project_dir,
                self._success_subsets
            )
            self.logger.info("Downstream processing completed")


class WorkflowError(Exception):
    """Custom exception for workflow-related errors."""
    pass
            

def main(config_path: Path = constants.DEFAULT_CONFIG) -> None:
    """Run the entire workflow."""    
    workflow = Workflow16S(config_path)
    workflow.run()


if __name__ == "__main__":
    # Get custom config.yaml file from system arguments
    parser = argparse.ArgumentParser(description="Run 16S workflow.")
    parser.add_argument(
        "--config",
        type=Path,
        default=constants.DEFAULT_CONFIG,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    main(args.config)
