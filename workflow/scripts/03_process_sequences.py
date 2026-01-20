"""
03_process_sequences.py: Processes sequences and writes a QIIME2 manifest.
"""
import argparse
import yaml
import pandas as pd
from pathlib import Path

from src.workflow_16s.config_schema import AppConfig
from src.workflow_16s.upstream.sequences.processing import process_sequences
from src.workflow_16s.utils.dir_utils import Project, SubSet
from src.workflow_16s.upstream.upstream import write_manifest_tsv 

def find_dataset_info(config: AppConfig, subset_id: str) -> pd.Series:
    """Finds the metadata row for the dataset this subset belongs to."""
    dataset_id = subset_id.split('.')[0]
    df = pd.read_csv(config.paths.dataset_info, sep="\t", dtype={'ena_project_accession': str})
    
    match = df[df['dataset_id'] == dataset_id]
    if not match.empty: return match.iloc[0]
    
    match = df[df['ena_project_accession'] == dataset_id]
    if not match.empty: return match.iloc[0]
    
    # Fallback for newly discovered NFC projects
    return pd.Series({'dataset_type': 'ENA', 'ena_project_accession': dataset_id, 'dataset_id': dataset_id})


def reconstruct_subset_dict(subset_id: str, metadata: pd.DataFrame) -> dict:
    """
    Reconstructs the essential parts of the 'subset' dictionary
    by parsing the ID and using the metadata. This is a crucial
    and potentially fragile step.
    """
    parts = subset_id.split('.')
    subset_dict = {
        "dataset": parts[0],
        "instrument_platform": parts[1],
        "library_layout": parts[2],
        "target_subfragment": parts[3],
        "metadata": metadata
    }
    # This is a simplification. You might need to parse primers from the ID too.
    return subset_dict

def main():
    parser = argparse.ArgumentParser(description="Process sequences and create manifest.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subset-id", type=str, required=True)
    parser.add_argument("--subset-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    print(f"Processing sequences for subset: {args.subset_id}")
    
    with open(args.config, 'r') as f:
        config = AppConfig(**yaml.safe_load(f))

    # We need to reconstruct objects the original functions expect
    project_dir = Project(config)
    metadata_df = pd.read_csv(args.metadata, sep='\t', header=0)
    
    # Reconstruct the 'subset' dictionary and 'dataset_info' series
    subset_dict = reconstruct_subset_dict(args.subset_id, metadata_df)
    dataset_info = find_dataset_info(config, args.subset_id)
    
    # The SubSet class helps create the expected directory structure
    subset_obj = SubSet(project_dir, subset_dict)

    # Call the main sequence processing function
    seq_paths, _ = process_sequences(config, subset_dict, subset_obj, dataset_info)

    # Write the manifest file
    if isinstance(seq_paths, dict) and seq_paths:
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        write_manifest_tsv(seq_paths, args.output_manifest)
        print(f"Successfully wrote manifest to {args.output_manifest}")
    else:
        raise RuntimeError(f"Sequence processing failed for {args.subset_id}")

if __name__ == "__main__":
    main()