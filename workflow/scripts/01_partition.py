"""
01_partition.py: Partitions datasets into subsets based on metadata.

This script:
1. Initializes the main 'Upstream' logic from the workflow_16s package.
2. Optionally runs the NFC facility search to discover new datasets.
3. Partitions each dataset into one or more subsets.
4. For each subset, it creates a unique ID and saves its raw metadata.
5. Writes all unique subset IDs to a master list for Snakemake to use.
"""
import argparse
import asyncio
import yaml
import re
import pandas as pd
from pathlib import Path

# Ensure 'workflow_16s' is installable or in the PYTHONPATH
from src.workflow_16s.config_schema import AppConfig
from src.workflow_16s.upstream.metadata.partition import DatasetPartition
from src.workflow_16s.api.nuclear_fuel_cycle.nfc import NFCFacilitiesHandler
from src.workflow_16s.utils.metadata_utils import export_tsv
from src.workflow_16s.utils.dir_utils import Project, SubSet, RawData

def sanitize_id_part(s: str) -> str:
    """Helper to create Snakemake-safe wildcard strings."""
    return re.sub(r"[^a-zA-Z0-9-]", "_", s) if s else "N_A"

def generate_subset_id(subset_dict: dict) -> str:
    """Recreates the unique subset ID from the original script."""
    parts = [
        subset_dict.get("dataset"),
        str(subset_dict.get("instrument_platform")).strip(),
        subset_dict.get("library_layout"),
        subset_dict.get("target_subfragment"),
        f"FWD_{sanitize_id_part(subset_dict.get('pcr_primer_fwd_seq'))}", # type: ignore
        f"REV_{sanitize_id_part(subset_dict.get('pcr_primer_rev_seq'))}", # type: ignore
    ]
    return ".".join(p.upper() for p in parts if p)

async def main():
    parser = argparse.ArgumentParser(description="Partition datasets into subsets.")
    parser.add_argument("--config", type=Path, required=True, help="Path to config.yaml")
    parser.add_argument("--output-list", type=Path, required=True, help="Output file for subset IDs")
    args = parser.parse_args()

    # --- Load Config and Initial Data ---
    with open(args.config, 'r') as f:
        config = AppConfig(**yaml.safe_load(f))

    project_dir = Project(config)
    
    with open(config.paths.dataset_list, "r") as f:
        datasets = [line.strip() for line in f if line.strip()]
    datasets_info = pd.read_csv(config.paths.dataset_info, sep="\t", dtype={'ena_project_accession': str})

    # --- Optional: NFC Data Discovery ---
    if config.nfc_facilities.enabled:
        print("NFC facility matching is enabled.")
        nfc_handler = NFCFacilitiesHandler(config)
        await nfc_handler.nfc_facilities()
        if config.nfc_facilities.fetch_nearby_samples:
            await nfc_handler.get_nearby_samples()
            nearby_samples_df = nfc_handler.nearby_samples_df
            if not nearby_samples_df.empty and 'study_accession' in nearby_samples_df.columns:
                new_projects = nearby_samples_df['study_accession'].dropna().unique().tolist()
                print(f"Found {len(new_projects)} new projects from NFC-proximal samples.")
                datasets.extend(new_projects)
                datasets = sorted(list(set(datasets)))

    # --- Partitioning ---
    all_subset_ids = []
    partitioner = DatasetPartition(config)
    for dataset_id in datasets:
        print(f"Partitioning dataset: {dataset_id}")
        try:
            # Find matching dataset info row
            match = datasets_info[datasets_info['dataset_id'] == dataset_id]
            if match.empty:
                 match = datasets_info[datasets_info['ena_project_accession'] == dataset_id]
            if match.empty:
                print(f"Warning: No info found for {dataset_id}. Treating as new ENA dataset.")
                dataset_info_dict = {'dataset_type': 'ENA', 'ena_project_accession': dataset_id, 'dataset_id': dataset_id}
            else:
                dataset_info_dict = match.iloc[0].to_dict()

            successful, _ = await partitioner.run({dataset_id: dataset_info_dict})
            
            for subset_dict in successful:
                subset_id = generate_subset_id(subset_dict)
                all_subset_ids.append(subset_id)
                
                # Save the raw metadata for this subset, which is the input for the next rule
                subset_dir = SubSet(project_dir, subset_dict)
                raw_data_dir = RawData(subset_dir)
                raw_data_dir.main.mkdir(parents=True, exist_ok=True)
                export_tsv(subset_dict["metadata"], raw_data_dir.metadata_tsv)
                print(f"  -> Created subset: {subset_id}")

        except Exception as e:
            print(f"Failed to partition dataset {dataset_id}: {e}")

    # --- Write Final List for Snakemake ---
    unique_subset_ids = sorted(list(set(all_subset_ids)))
    with open(args.output_list, 'w') as f:
        for subset_id in unique_subset_ids:
            f.write(f"{subset_id}\n")
    print(f"\nGenerated {len(unique_subset_ids)} unique subsets.")

if __name__ == "__main__":
    asyncio.run(main())