import scanpy as sc
import os
import pandas as pd
# Import your official loader and schema
from workflow_16s.config_schema import load_config
from workflow_16s.api.environmental_data.other.execute import EnvironmentalDataCollector

# --- 1. CONFIGURATION & PATHS ---
ADATA_PATH = "/usr2/people/macgregor/amplicon/project_01/04_analysis/testing_20260212/merged_samples.h5ad"
CONFIG_PATH = "/usr2/people/macgregor/amplicon/workflow_16s/config/config_ml_only.yaml"

if not os.path.exists(ADATA_PATH):
    raise FileNotFoundError(f"❌ Cannot find AnnData at {ADATA_PATH}")

# ✅ FIX: Load as a Pydantic Object, not a dictionary
print(f"⚙️  Loading configuration from {CONFIG_PATH}...")
config_obj = load_config(CONFIG_PATH)

# --- 2. LOAD DATA ---
print(f"📖 Loading AnnData from {ADATA_PATH}...")
adata = sc.read_h5ad(ADATA_PATH)

# Ensure unique names for safe updating
if not adata.obs_names.is_unique:
    print("⚠️  Non-unique Sample IDs detected. Making names unique...")
    adata.obs_names_make_unique()

# --- 3. TARGETED RECOVERY AUDIT ---
# Identify samples with coordinates but missing ANY SoilGrids value
soil_cols = [c for c in adata.obs.columns if c.startswith('SoilGrids_') and c.endswith('_value')]

print(f"🔎 Auditing {len(adata)} samples...")

if not soil_cols:
    mask = (adata.obs['lat'].notna()) & (adata.obs['lon'].notna())
else:
    mask = (adata.obs['lat'].notna()) & (adata.obs[soil_cols].isnull().any(axis=1))

rows_to_fix = adata.obs[mask].copy()

# --- 4. EXECUTE API FETCH ---
if not rows_to_fix.empty:
    print(f"🚀 Attempting recovery for {len(rows_to_fix)} samples.")
    print("📡 This involves bulk API calls. Progress will be logged below...")
    
    # Initialize using the validated config object
    collector = EnvironmentalDataCollector(data=rows_to_fix, config=config_obj)
    
    # run_apis() handles the smart batching to avoid API bans
    recovered_env_df = collector.run_apis()
    
    if recovered_env_df is not None and not recovered_env_df.empty:
        # Standardize index to match unique obs_names
        recovered_env_df.index = recovered_env_df.index.astype(str)
        
        # Patch the metadata in place
        adata.obs.update(recovered_env_df)
        
        # Save back to the same file
        print(f"💾 Saving {len(recovered_env_df)} updated records to {ADATA_PATH}...")
        adata.write_h5ad(ADATA_PATH)
        print("✅ Recovery complete.")
    else:
        print("⚠️ API returned no data. Locations may be offshore or API is temporarily unreachable.")
else:
    print("✨ No missing soil data found for terrestrial samples.")