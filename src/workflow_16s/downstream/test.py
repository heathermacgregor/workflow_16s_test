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

import threading
import time
import psutil
import os
import logging



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

# ==================================================================================== #
# IMPROVED RESOURCE MONITOR
# ==================================================================================== #

import threading
import time
import psutil
import os
import logging
import platform
import getpass

class ResourceMonitor:
    def __init__(self, interval_seconds=300, target_dir=None):
        """
        Monitor resources every `interval_seconds` (default 300s = 5m).
        
        Args:
            target_dir (Path, optional): Directory to monitor disk usage for.
        """
        self.interval = interval_seconds
        self.target_dir = target_dir or Path.cwd()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.logger = logging.getLogger("workflow_16s")
        self.hostname = platform.node()
        self.user = getpass.getuser()

    def _get_process_tree_stats(self):
        """Calculates comprehensive stats for the entire process tree."""
        try:
            parent = psutil.Process(os.getpid())
            children = parent.children(recursive=True)
            all_procs = [parent] + children
            
            stats = {
                'count': len(all_procs),
                'children': len(children),
                'threads': 0,
                'cpu_percent_total': 0.0,
                'rss_gb': 0.0,
                'vms_gb': 0.0,
                'names': {}
            }
            
            for p in all_procs:
                try:
                    # Efficiently fetch attributes in one context switch
                    with p.oneshot():
                        stats['threads'] += p.num_threads()
                        
                        # Memory
                        mem = p.memory_info()
                        stats['rss_gb'] += mem.rss
                        stats['vms_gb'] += mem.vms
                        
                        # CPU (interval=None gets usage since last call)
                        stats['cpu_percent_total'] += p.cpu_percent(interval=None)
                        
                        # Name grouping (clean up names if needed)
                        name = p.name()
                        stats['names'][name] = stats['names'].get(name, 0) + 1
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            # Convert bytes to GB
            stats['rss_gb'] /= (1024 ** 3)
            stats['vms_gb'] /= (1024 ** 3)
            
            return stats
        except Exception:
            return None

    def _monitor_loop(self):
        self.logger.info(f"🔍 Resource Monitor started on {self.hostname} (User: {self.user})")
        
        while not self.stop_event.is_set():
            try:
                # 1. System Stats
                sys_cpu = psutil.cpu_percent(interval=1)
                sys_mem = psutil.virtual_memory()
                sys_load = os.getloadavg()
                
                # Disk Stats (for the output volume)
                try:
                    disk = psutil.disk_usage(str(self.target_dir))
                    disk_str = f"{disk.percent}% Used ({disk.free / (1024**3):.1f} GB Free)"
                except Exception:
                    disk_str = "N/A"

                # 2. Pipeline Stats
                p_stats = self._get_process_tree_stats()
                
                if p_stats:
                    # Format process names breakdown
                    breakdown = ", ".join([f"{k}: {v}" for k, v in p_stats['names'].items()])
                    
                    msg = (
                        f"\n📊 [SERVER HEALTH & PIPELINE METRICS]\n"
                        f"   ├── 🖥️ System Status ({self.hostname})\n"
                        f"   │   ├── Load (1/5/15m): {sys_load}\n"
                        f"   │   ├── CPU Usage:      {sys_cpu}% (Global)\n"
                        f"   │   ├── RAM Usage:      {sys_mem.percent}% ({sys_mem.used / (1024**3):.1f}/{sys_mem.total / (1024**3):.1f} GB)\n"
                        f"   │   └── Disk Volume:    {disk_str}\n"
                        f"   │\n"
                        f"   └── 🚀 Pipeline Consumption (PID: {os.getpid()})\n"
                        f"       ├── Hierarchy:      1 Parent + {p_stats['children']} Children = {p_stats['count']} Processes\n"
                        f"       ├── Threading:      {p_stats['threads']} Active Threads\n"
                        f"       ├── CPU Draw:       {p_stats['cpu_percent_total']:.1f}% (Cumulative across cores)\n"
                        f"       ├── Memory (RSS):   {p_stats['rss_gb']:.2f} GB (Physical RAM)\n"
                        f"       └── Breakdown:      {{{breakdown}}}"
                    )
                    self.logger.info(msg)
                
                self.stop_event.wait(self.interval - 1)
                
            except Exception as e:
                self.logger.error(f"Resource monitor error: {e}")
                self.stop_event.wait(self.interval)

    def start(self):
        self.thread.start()

    def stop(self):
        self.logger.info("Stopping Resource Monitor...")
        self.stop_event.set()
        self.thread.join(timeout=2)

# ==================================================================================== #
# MAIN INTEGRATION
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
    log_dir_path.mkdir(exist_ok=True, parents=True)
    
    initialize_logging(log_dir_path)
    logger = get_logger()
    
    logger.info("=== Starting 16S Downstream Analysis ===")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # --- 1. INITIALIZE MONITOR WITH DISK PATH ---
    monitor = ResourceMonitor(interval_seconds=300, target_dir=args.output_dir)
    monitor.start()
    
    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
            logger.info(f"Loaded configuration from: {args.config}")
        else:
            from workflow_16s.config_schema import AppConfig
            config = AppConfig()
            logger.info("Using default configuration")
        
        # Override CPU settings if provided
        if args.n_cpus:
            config.execution.threads = args.n_cpus
            logger.info(f"Using {args.n_cpus} CPU cores")
        
        # Asynchronous loading of NFC facilities 
        nfc_facilities_df = pd.DataFrame()
        if config.nfc_facilities.enabled:
            logger.info("NFC facility processing is enabled. Fetching data...")
            try:
                nfc_handler = NFCFacilitiesHandler(config)
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
        
        from workflow_16s.downstream.workflow import DownstreamWorkflow
        
        workflow = DownstreamWorkflow(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_cpus=args.n_cpus,
            config=config,
            nfc_facilities_df=nfc_facilities_df
        )
        workflow.execute()
        
        logger.info("=== Workflow Completed Successfully ===")
        
    except Exception as e:
        logger.critical(f"Workflow failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        monitor.stop()
        
# ==================================================================================== #

if __name__ == "__main__":
    # Use asyncio.run() to execute the async main function
    asyncio.run(main())