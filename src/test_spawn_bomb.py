import time
import os
import sys

print(f"--- Debugger PID: {os.getpid()} ---")
print("Open htop in another terminal. Waiting 3 seconds...")
time.sleep(3)

print("\n1. Importing standard libraries...")
import pandas as pd
import numpy as np
time.sleep(2)

print("\n2. Importing Scanpy...")
import scanpy as sc
time.sleep(2)

print("\n3. Importing workflow config...")
try:
    from workflow_16s.config_schema import load_config
    print("   -> Success")
except ImportError:
    print("   -> Failed (check path)")
    time.sleep(2)

    print("\n4. Importing Ingestion Step (Did the fix work?)...")
    # This imports the file where we fixed run_fast_load
    from workflow_16s.downstream.steps.ingestion import run_fast_load
    time.sleep(2)

    print("\n5. Importing NFC Handler (The new suspect)...")
    try:
        from workflow_16s.api.nuclear_fuel_cycle.nfc import NFCFacilitiesHandler
        print("   -> Success")
    except ImportError:
        print("   -> Failed (check path)")
        time.sleep(2)

        print("\n6. Importing Orchestrator...")
        from workflow_16s.downstream.orchestrator import DownstreamWorkflow
        time.sleep(2)

        print("\n--- DONE. If you see this without a spawn bomb, the issue is inside main() ---")
