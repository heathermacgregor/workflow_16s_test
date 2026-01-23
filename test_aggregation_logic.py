import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse
import logging

# Setup simple logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TEST")

# FILE PATH
FILE_PATH = "/usr2/people/macgregor/amplicon/project_01/04_analysis/testing_5/merged_samples.h5ad"

def test_aggregation(adata, level='Genus'):
    print(f"\n--- Testing Aggregation Logic for {level} ---")
    
    if level not in adata.var.columns:
        print(f"❌ Column '{level}' not found!")
        return

    # ====================================================
    # 1. THE FIX: Robust Cleaning
    # ====================================================
    print("1. Cleaning Taxonomy Column...")
    # Convert to string and STRIP WHITESPACE (Fixes ' g__Luteolibacter')
    groups = adata.var[level].astype(str).str.strip()
    
    # Replace garbage with Unassigned
    groups = groups.replace(['nan', 'NaN', 'None', '', '<NA>', 'NoneType'], 'Unassigned')
    
    # Fill NAs
    groups = groups.fillna('Unassigned')
    
    unique_groups = sorted(groups.unique())
    n_groups = len(unique_groups)
    print(f"✅ Found {n_groups} unique groups after cleaning.")
    
    if n_groups < 5:
        print(f"⚠️ WARNING: Very few groups found: {unique_groups}")
    else:
        print(f"   Example groups: {unique_groups[:5]}")

    # ====================================================
    # 2. The Aggregation (Matrix Math)
    # ====================================================
    print("\n2. Performing Matrix Aggregation...")
    
    # Map groups to indices
    group_dict = {g: i for i, g in enumerate(unique_groups)}
    col_indices = [group_dict[g] for g in groups]
    row_indices = range(len(groups))
    
    # Create Grouper Matrix
    grouper = sparse.csr_matrix(
        (np.ones(len(groups)), (row_indices, col_indices)),
        shape=(len(groups), len(unique_groups))
    )
    
    # Multiply: (Samples x Features) @ (Features x Groups)
    new_X = adata.X @ grouper
    
    print(f"✅ Aggregation finished. New Matrix Shape: {new_X.shape}")
    
    # ====================================================
    # 3. Verify Result
    # ====================================================
    print("\n3. Verifying Output...")
    # Create temp AnnData to check var names
    adata_new = sc.AnnData(X=new_X, var=pd.DataFrame(index=unique_groups))
    
    if 'Unassigned' in adata_new.var_names:
        if len(adata_new.var_names) == 1:
            print("❌ FAIL: Result contains ONLY 'Unassigned'. Logic failed.")
        else:
            print(f"✅ SUCCESS: 'Unassigned' is present but valid taxa exist too.")
            print(f"   Total Taxa: {len(adata_new.var_names)}")
    else:
        print("✅ SUCCESS: Aggregation worked and 'Unassigned' isn't even in the top list (or was filtered).")

if __name__ == "__main__":
    print(f"🚀 Loading {FILE_PATH}...")
    try:
        adata = sc.read_h5ad(FILE_PATH)
        print("✅ Data loaded.")
        test_aggregation(adata, 'Genus')
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
