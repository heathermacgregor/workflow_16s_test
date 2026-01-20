import argparse
import yaml
from pathlib import Path
import pandas as pd
# Important: Make sure the workflow_16s package is in your PYTHONPATH
# or install it in the conda environment.
from src.workflow_16s.api.qiime.run import execute_per_dataset_qiime_workflow as execute_qiime
from src.workflow_16s.config_schema import AppConfig
from src.workflow_16s.utils.dir_utils import Project, SubSet, QIIME, RawData # Assuming QIIME can be initialized this way

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
    parser = argparse.ArgumentParser(description="Run QIIME2 workflow for a subset.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subset-id", type=str, required=True)
    parser.add_argument("--qiime-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = AppConfig(**config_dict)
    project_dir = Project(config)
    
    metadata_df = pd.read_csv(args.metadata, sep='\t', header=0)
    subset_dir = SubSet(project_dir, reconstruct_subset_dict(args.subset_id, metadata_df))
    # Note: The original script derived 'subset' and 'subset_dir' objects dynamically.
    # Here we simplify by passing paths directly. You may need to adjust how
    # the execute_qiime function is called to match its expected arguments.
    # This might require a small refactor in your source code to accept paths directly.
    subset = reconstruct_subset_dict(args.subset_id, metadata_df)
    print(f"Running QIIME2 for subset: {args.subset_id}")
    execute_qiime(
        config, subset, subset_dir.qiime, 
                                      RawData(subset_dir).metadata_tsv, 
                                      QIIME(subset_dir).manifest_tsv
    )
    print("QIIME2 workflow finished.")

if __name__ == "__main__":
    main()