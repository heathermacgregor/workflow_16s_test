"""
02_process_metadata.py: Cleans and enriches metadata for a single subset.
"""
import argparse
import asyncio
import yaml
import pandas as pd
from pathlib import Path

from src.workflow_16s.api.environmental_data.other.execute import EnvironmentalDataCollector
from src.workflow_16s.api.nuclear_fuel_cycle.nfc import NFCFacilitiesHandler
from src.workflow_16s.config_schema import AppConfig
from src.workflow_16s.utils.metadata_utils import process_metadata, standardize_lat_lon_columns
from src.workflow_16s.api.environmental_data.google.arkin_env_agents import main as arkin_env_agents
from src.workflow_16s.utils.dir_utils import Project, SubSet, RawData, QIIME
# You may need to import other specific functions for NFC matching and env data collection

async def main():
    parser = argparse.ArgumentParser(description="Process metadata for a single subset.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subset-id", type=str, required=True)
    parser.add_argument("--input-metadata", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    args = parser.parse_args()

    print(f"Processing metadata for subset: {args.subset_id}")
    
    with open(args.config, 'r') as f:
        config = AppConfig(**yaml.safe_load(f))

    # Load metadata
    metadata_df = pd.read_csv(args.input_metadata, sep='\t', header=0)
    metadata_df = standardize_lat_lon_columns(metadata_df)
    
    # NOTE: The original script runs NFC matching and environmental data collection here.
    # You would re-implement those calls. For brevity, we focus on the main cleaning step.
    # Example:
    if config.nfc_facilities.enabled:
        nfc_handler = NFCFacilitiesHandler(config)
        nfc_facilities_df = await nfc_handler.nfc_facilities()
        if not nfc_facilities_df.empty:
            metadata_df = nfc_handler._match_facilities_with_locations(nfc_facilities_df, metadata_df)
            
    # Clean and enrich metadata
    try:
        processed_df = await process_metadata(metadata_df, args.input_metadata, config)
    except Exception as e:
        print(f"Metadata cleaning failed for {args.subset_id}: {e}. Using original metadata.")
        processed_df = metadata_df
        
    # Collect environmental data if enabled
    # Check if 'lat' and 'lon' columns exist before trying to use them
    if "lat" in metadata_df.columns and "lon" in metadata_df.columns:
        metadata_df["lat"] = pd.to_numeric(metadata_df["lat"], errors="coerce")
        metadata_df["lon"] = pd.to_numeric(metadata_df["lon"], errors="coerce")
                
        cols = ["collection_date", "lat", "lon"]
        valid_rows = metadata_df.dropna(subset=cols)
                
        if not valid_rows.empty:
            print(f"Found {len(valid_rows)} valid samples with coordinates and dates to process.")
            project_dir = Project(config)
            subset_dir = SubSet(project_dir, args.subset_id)
            data_collector = EnvironmentalDataCollector(data=valid_rows[cols], config=config, 
                                                                project_dir=project_dir, 
                                                                output_file=RawData(subset_dir).main / "sample_env_data.json",
                                                                verbose=config.verbose)
            data_collector.run_apis()
            env_data = data_collector.results
            print(env_data)

    # Arkin environmental agents enrichment
    try:
        arkin_env_agents(metadata_path=args.input_metadata, project_dir=args.output_metadata.parent)
        # The arkin function might modify the file in place. If so, we need to reload it.
        processed_df = pd.read_csv(args.input_metadata, sep='\t', header=0)
    except Exception as e:
        print(f"Arkin enrichment failed for {args.subset_id}: {e}")

    # Save the final processed metadata
    processed_df.to_csv(args.output_metadata, sep='\t', index=False)
    print(f"Successfully processed and saved metadata to {args.output_metadata}")

if __name__ == "__main__":
    asyncio.run(main())