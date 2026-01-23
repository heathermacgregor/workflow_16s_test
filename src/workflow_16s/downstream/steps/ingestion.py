# ==================================================================================== #
#                                     INGESTION.PY
# ==================================================================================== #

import os
# Set threads to 1 to prevent library-level contention
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from typing import Tuple, Optional, List
import re
import subprocess
import hashlib
import pickle
import gc
import math
import psutil
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from scipy.sparse import issparse, csr_matrix, csc_matrix
from joblib import Parallel, delayed

# --- Logger setup ---
from workflow_16s.utils.logger import get_logger
from workflow_16s.utils.progress import get_progress_bar

# Use a safe local import or fallback if adata_utils functions are missing
try:
    from workflow_16s.downstream.utils import fix_adata_dtypes, get_resident_memory_gb
except ImportError:
    def fix_adata_dtypes(adata): pass
    def get_resident_memory_gb():
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024 / 1024
        except: return 0.0

logger = get_logger("workflow_16s")
# Suppress annoying implicit modification warnings from pandas
pd.options.mode.chained_assignment = None

def _get_file_hash(filepath: Path) -> str:
    """Fast hash of file stats to detect changes without reading content."""
    stats = filepath.stat()
    return hashlib.md5(f"{stats.st_size}_{stats.st_mtime}".encode()).hexdigest()[:8]

def _validate_cached_adata(adata) -> Tuple[bool, str]:
    """Quick sanity check on cached object."""
    if adata is None: return False, "None object"
    if not isinstance(adata, ad.AnnData): return False, "Not AnnData"
    if adata.n_obs == 0: return False, "Empty observations"
    return True, ""

def _sanitize_adata(adata: ad.AnnData) -> ad.AnnData:
    """
    Minimal sanitization to ensure merging works. 
    Does NOT force types aggressively to allow clean_metadata to do its job.
    """
    # 1. Fix Index Name Conflicts
    if adata.obs_names.name is None: adata.obs_names.name = 'sample_id'
    if adata.var_names.name is None: adata.var_names.name = 'feature_id'
    
    if not adata.obs.index.is_unique:
        adata.obs_names_make_unique()
    # Remove columns that duplicate the index name (prevents HDF5 write errors)
    for idx_name in [adata.obs_names.name, adata.var_names.name]:
        if idx_name in adata.obs.columns:
            try: del adata.obs[idx_name]
            except: pass
            
    # 2. Force Coordinates to Numeric (Critical for Maps)
    # We do this here because we often need them for spatial queries immediately
    for coord in ['lat', 'lon', 'latitude', 'longitude']:
        if coord in adata.obs.columns:
            adata.obs[coord] = pd.to_numeric(adata.obs[coord], errors='coerce')

    # 3. Strip whitespace from Index (Clean IDs)
    try: adata.var_names = adata.var_names.str.strip()
    except: pass
    try: adata.obs_names = adata.obs_names.str.strip()
    except: pass

    # NOTE: We removed the "Force Object to String" block.
    # Allowing clean_metadata to handle type inference is safer.

    return adata

def _process_single_file(f: Path, config, cache_dir: Optional[Path] = None):
    """
    Worker function: Loads, cleans, and sanitizes a single .h5ad file.
    Handles caching internally.
    """
    # 1. Check Cache
    if cache_dir:
        cache_file = cache_dir / f"{f.stem}_{_get_file_hash(f)}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as cf:
                    cached_adata = pickle.load(cf)
                is_valid, _ = _validate_cached_adata(cached_adata)
                if is_valid: 
                    return cached_adata
                else: 
                    cache_file.unlink() # Corrupt cache
            except Exception:
                if cache_file.exists(): cache_file.unlink()

    # 2. Load & Process
    try:
        adata = sc.read_h5ad(f)
        
        # A. Sanitize (Minimal fixes)
        adata = _sanitize_adata(adata)
        
        # B. Clean Metadata (Type inference happens here!)
        adata = clean_metadata(adata, config)
        
        if adata is None: return None
        
        # C. Parse & Filter
        adata = parse_taxonomy(adata)
        adata = filter_samples_and_features(adata, config)
        
        if adata is None or adata.n_obs == 0:
            return None
            
        # D. Type Fixing (Final check for HDF5 compatibility)
        fix_adata_dtypes(adata)
        
        # 3. Save to Cache
        if cache_dir:
            try:
                with open(cache_file, 'wb') as cf: 
                    pickle.dump(adata, cf, protocol=4)
            except Exception as e:
                logger.warning(f"Failed to cache {f.name}: {e}")

        return adata
        
    except Exception as e:
        logger.warning(f"Failed to load {f.name}: {e}")
        return None

def hierarchical_merge(adatas: List[ad.AnnData]) -> Optional[ad.AnnData]:
    """Merges a list of AnnData objects using concatenation."""
    if not adatas: return None
    try:
        # Outer join preserves features present in ANY dataset
        merged = ad.concat(
            adatas, 
            join="outer", 
            merge="unique", 
            uns_merge="unique", 
            fill_value=0
        )
        return merged
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        return None

def run_fast_load(workflow):
    """
    Main Ingestion Function.
    """
    logger.info("--- 1. Data Ingestion & Merging ---")
    
    # 1. Tier 1 Cache: Check for pre-merged final file
    final_cache = workflow.output_dir / "merged_samples.h5ad"
    if workflow.config.ml.load_existing and final_cache.exists():
        logger.info(f"Loading cached merged data from {final_cache.name}...")
        try:
            workflow.adata = sc.read_h5ad(final_cache)
            logger.info(f"✅ Loaded {workflow.adata.n_obs} samples.")
            return
        except Exception:
            logger.warning("Cached file corrupt. Reloading from source.")

    # 2. Setup Tier 2 Cache (Individual Files)
    # Define the directory where _process_single_file will save pickles
    preproc_cache_dir = workflow.output_dir / ".cache" / "preprocessed_files"
    preproc_cache_dir.mkdir(parents=True, exist_ok=True)

    input_files = list(workflow.data_dir.glob("*.h5ad"))
    if not input_files:
        logger.error(f"No .h5ad files found in {workflow.data_dir}")
        return

    logger.info(f"Found {len(input_files)} datasets. Starting parallel ingest...")

    # 3. Parallel Loading
    batch_size = 10
    all_adatas = []
    chunks = [input_files[i:i + batch_size] for i in range(0, len(input_files), batch_size)]
    
    for i, chunk in enumerate(chunks):
        # PASS cache_dir TO THE WORKER
        processed_chunk = Parallel(n_jobs=workflow.n_cpus)(
            delayed(_process_single_file)(
                f, 
                workflow.config, 
                cache_dir=preproc_cache_dir  # <--- HERE IS THE FIX
            ) for f in chunk
        )
        
        valid_adatas = [a for a in processed_chunk if a is not None]
        
        if valid_adatas:
            chunk_merged = hierarchical_merge(valid_adatas)
            if chunk_merged: all_adatas.append(chunk_merged)
            
            del valid_adatas
            del processed_chunk
            gc.collect()
            
        logger.info(f"Processed batch {i+1}/{len(chunks)} ({len(chunk)} files)")

    # 4. Final Merge
    if all_adatas:
        logger.info("Performing final merge...")
        final_adata = hierarchical_merge(all_adatas)
        if final_adata:
            final_adata.write_h5ad(final_cache)
            workflow.adata = final_adata
            logger.info(f"✅ Ingestion Complete: {final_adata.n_obs} samples.")
        else:
            logger.error("Final merge failed.")
    else:
        logger.error("No valid data loaded.")

def run_filter_empty(workflow):
    """
    Final Post-Ingestion cleanup.
    Ensures no empty samples or zero-count features remain after the merge.
    """
    logger.info("--- Post-Ingestion Filtering ---")
    if workflow.adata is None: 
        return

    n_samples_pre = workflow.adata.n_obs
    n_features_pre = workflow.adata.n_vars

    # 1. Filter Empty Samples (rows with 0 counts)
    # This might happen if a sample passed pre-filtering but lost all features 
    # during a merge (unlikely with outer join, but good safety)
    sc.pp.filter_cells(workflow.adata, min_counts=1)

    # 2. Filter Empty Features (cols with 0 counts across ALL samples)
    # This is common if features were filtered out in some batches but 
    # the union kept the column name with all zeros.
    sc.pp.filter_genes(workflow.adata, min_cells=1)

    n_samples_post = workflow.adata.n_obs
    n_features_post = workflow.adata.n_vars

    if n_samples_pre != n_samples_post:
        logger.info(f"Removed {n_samples_pre - n_samples_post} empty samples.")
    
    if n_features_pre != n_features_post:
        logger.info(f"Removed {n_features_pre - n_features_post} empty features (not present in any sample).")
    
    logger.info(f"Final Data Shape: {n_samples_post} samples x {n_features_post} features.")

def find_conda_env_by_substring(substring: str, logger=None) -> Optional[Path]:
    """
    Locates a conda environment path by searching for a substring (e.g., 'picrust2').
    Used to auto-discover external tool environments.
    """
    import subprocess
    import sys
    
    try:
        # Run 'conda env list' to get all environments
        # We use check_output to capture stdout
        result = subprocess.check_output(["conda", "env", "list"], text=True)
        
        for line in result.splitlines():
            # Skip comments
            if line.startswith("#") or not line.strip(): 
                continue
                
            # Parse line: "env_name   * /path/to/env"
            parts = line.split()
            if len(parts) >= 1:
                # Path is typically the last element
                env_path = parts[-1] 
                
                # Check if the path or name matches our substring
                if substring in env_path or (len(parts) > 1 and substring in parts[0]):
                    if logger: 
                        logger.info(f"Found conda env for '{substring}': {env_path}")
                    return Path(env_path)
                    
        return None
        
    except Exception as e:
        if logger: 
            logger.warning(f"Could not query conda environments to find '{substring}': {e}")
        return None