# workflow_16s/utils/io/placeholder.py

from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

from workflow_16s.utils.logger import get_logger


def load_datasets_list(path: Union[str, Path]) -> List[str]:
    """Load dataset IDs from a text file, ignoring empty/whitespace lines."""
    logger = get_logger("workflow_16s")
    try:
        path = Path(path)
        if not path.is_file(): 
            raise FileNotFoundError(f"Dataset list file not found at: {path}")
        with open(path, "r") as f: 
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError as e: 
        logger.error(e)
        return []
    except Exception as e: 
        logger.error(f"Error reading dataset list file {path}: {e}")
        return []

def load_datasets_info(tsv_path: Union[str, Path]) -> pd.DataFrame:
    """Load dataset metadata from a TSV file."""
    logger = get_logger("workflow_16s")
    try:
        tsv_path = Path(tsv_path)
        if not tsv_path.is_file(): 
            raise FileNotFoundError(f"Dataset info file not found at: {tsv_path}")
        df = pd.read_csv(
            tsv_path, 
            sep="\t", 
            dtype={'ena_project_accession': str, 'dataset_id': str}
        ) 
        # Remove unnamed columns resulting from Excel spreadsheet saves
        return df.loc[:, ~df.columns.str.startswith('Unnamed')] 
    except FileNotFoundError as e: 
        logger.error(e)
        return pd.DataFrame()
    except Exception as e: 
        logger.error(f"Error reading dataset info file {tsv_path}: {e}")
        return pd.DataFrame()
    
    

def write_qiime_manifest(
    project_dir: Path,
    subset_id: str, 
    partition_metadata: pd.DataFrame, 
    run_file_paths: Dict[str, List[Path]]
) -> Tuple[str, Path]:
    """
    Write a QIIME 2 manifest file for a given partition.

    Args:
        subset_id (str): The unique identifier for the subset/partition being processed.
        partition_metadata (pd.DataFrame): The metadata for the partition being processed.
        run_file_paths (Dict[str, List[Path]]): A dictionary mapping run accessions to their corresponding FASTQ file paths.

    Raises:
        RuntimeError: If the manifest file cannot be written.
        
    Returns:
        Tuple[str, Path]: A tuple containing the library layout ('single' or 'paired') and the path to the written manifest file.
    """
    logger = get_logger("workflow_16s")
    rows = []
        
    try: 
        layout = partition_metadata['library_layout'].iloc[0].lower()
    except (KeyError, IndexError): 
        error_msg = f"Cannot determine library layout for partition {subset_id}."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

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
        current_layout = layout
        if current_layout == 'paired' and len(existing_paths) == 1:
            # If metadata says paired, but only 1 file exists (e.g. interleaved or missing R2)
            # downgrade it to single-end for processing.
            logger.warning(f"Sample {sample_id} metadata claims PAIRED, but only 1 file found. Downgrading to SINGLE-END processing.")
            current_layout = 'single'
                
        if current_layout == 'paired':
            if len(existing_paths) < 2: 
                logger.warning(f"Paired-end sample {sample_id} is missing FASTQ files. Found: {len(existing_paths)}. Skipping.")
                skip_reasons['incomplete_pairs'] += 1
                continue
            if len(existing_paths) > 2: 
                logger.warning(f"Found >2 files for paired-end sample {sample_id}, using first two.")
            rows.append({
                'sample-id': sample_id, 
                'forward-absolute-filepath': str(existing_paths[0]), 
                'reverse-absolute-filepath': str(existing_paths[1])
            })
            
        elif current_layout == 'single':
            if len(existing_paths) < 1: 
                logger.warning(f"Single-end sample {sample_id} is missing its FASTQ file. Skipping.")
                skip_reasons['missing_files'] += 1
                continue
            rows.append({
                'sample-id': sample_id, 
                'absolute-filepath': str(existing_paths[0])
            })

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

    final_layout = 'single' if any('absolute-filepath' in r for r in rows) else 'paired'
    qiime_dir = project_dir / subset_id
    qiime_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = qiime_dir / "manifest.tsv"
    
    pd.DataFrame(rows).to_csv(manifest_path, sep='\t', index=False)
    logger.info(f"QIIME2 manifest for '{subset_id}' written to {manifest_path}")
    return final_layout, manifest_path
