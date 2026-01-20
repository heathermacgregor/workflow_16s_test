#!/usr/bin/env python

"""
Main executable script for 16S downstream analysis workflow.
"""

# ==================================================================================== #

import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import logging
from pathlib import Path
import argparse
import asyncio  # Required for async main
import pandas as pd  # Required for empty DataFrame

try:
    from workflow_16s.utils.logger import initialize_logging
    from workflow_16s.config_schema import load_config
    from workflow_16s.api.nuclear_fuel_cycle.nfc import NFCFacilitiesHandler
except ImportError as e:
    print(f"Error importing workflow modules: {e}", file=sys.stderr)
    print("Ensure the 'workflow_16s' package is installed correctly.", file=sys.stderr)
    sys.exit(1)
    
# ==================================================================================== #

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 16S Downstream Analysis Workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data_dir", 
        type=Path, 
        default=Path("/usr2/people/macgregor/amplicon/project_01/03_processed_data"),
        help="Input directory containing .h5ad files"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        default=Path("/usr2/people/macgregor/amplicon/project_01/04_analysis/testing_5"),
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--config", 
        type=Path,
        default=Path("/usr2/people/macgregor/amplicon/workflow_16s/config/config.yaml"),
        help="Path to configuration file (optional)"
    )
    
    parser.add_argument(
        "--n_cpus", 
        type=int, 
        default=16,
        help="Number of CPU cores to use"
    )
    
    # --- REMOVED: Obsolete picrust2 arguments ---
    # parser.add_argument("--picrust2_ref_dir", ...)
    # parser.add_argument("--picrust2_aux_dir", ...)
    return parser.parse_args()

# ==================================================================================== #

from workflow_16s.utils.logger import get_logger

async def main():
    """Main function to run the workflow."""
    args = parse_arguments()
    
    # Validate inputs
    if not args.data_dir.exists():
        print(f"Error: Data directory does not exist: {args.data_dir}", file=sys.stderr)
        sys.exit(1)
        
    # Create output directory
    args.output_dir.mkdir(exist_ok=True, parents=True)
    log_dir_path = Path("/usr2/people/macgregor/amplicon/project_01/07_logs")
    log_file_name = f"downstream_analysis_{args.output_dir.name}.log"
    log_dir_path.mkdir(exist_ok=True, parents=True)
    initialize_logging(log_dir_path)
    logger = get_logger()
    
    # Setup logging
    #logger = setup_logging(log_dir_path = log_dir_path)#, log_filename=log_file_name)
    
    logger.info("=== Starting 16S Downstream Analysis ===")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Log file: {log_dir_path}")
    
    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
            logger.info(f"Loaded configuration from: {args.config}")
        else:
            # Use default configuration
            from workflow_16s.config_schema import AppConfig
            config = AppConfig()  # This would need default values
            logger.info("Using default configuration")
        
        # Override CPU settings if provided
        if args.n_cpus:
            config.execution.threads = args.n_cpus
            logger.info(f"Using {args.n_cpus} CPU cores")
        
        # Asynchronous loading of NFC facilities 
        nfc_facilities_df = pd.DataFrame()  # Initialize as empty
        if config.nfc_facilities.enabled:
            logger.info("NFC facility processing is enabled. Fetching data...")
            try:
                nfc_handler = NFCFacilitiesHandler(config)
                # Use await to call the async function
                nfc_facilities_df = await nfc_handler.nfc_facilities()
                if not nfc_facilities_df.empty:
                    logger.info(f"Successfully loaded {len(nfc_facilities_df)} NFC facilities.")
                else:
                    logger.warning("NFC facility handler ran but returned no data.")
            except Exception as e:
                logger.error(f"Failed to load NFC facility data: {e}", exc_info=True)
                logger.warning("Continuing workflow without NFC facility data.")
        else:
            logger.info("NFC facility processing is disabled in config.")
        
        from workflow_16s.downstream.orchestrator import DownstreamWorkflow
        # Initialize and run workflow
        workflow = DownstreamWorkflow(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_cpus=args.n_cpus,
            config=config,
            nfc_facilities_df=nfc_facilities_df  # Pass the loaded df
        )
        workflow.execute()
        
        logger.info("=== Workflow Completed Successfully ===")
        
    except Exception as e:
        logger.critical(f"Workflow failed: {e}", exc_info=True)
        sys.exit(1)
        
# ==================================================================================== #

if __name__ == "__main__":
    # Use asyncio.run() to execute the async main function
    asyncio.run(main())