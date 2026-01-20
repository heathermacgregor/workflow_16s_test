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

# Use a safe local import or fallback if adata_utils is missing
try:
    from workflow_16s.downstream.adata_utils import fix_adata_dtypes, get_resident_memory_gb
except ImportError:
    def fix_adata_dtypes(adata): pass
    def get_resident_memory_gb():
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024 / 1024
        except: return 0.0

logger = get_logger("workflow_16s")

def log_mem(msg=""):
    """Helper to log memory usage."""
    mem = get_resident_memory_gb()
    logger.info(f"{msg} | Memory usage: {mem:.2f} GB (RES)")

def find_conda_env_by_substring(name_substring, logger):
    """Search for a Conda environment by partial name match."""
    try:
        result = subprocess.run(["conda", "info", "--envs"], capture_output=True, text=True, check=True, encoding='utf-8')
        env_list = result.stdout.strip().split('\n')
        env_name_pattern = re.compile(r"^\s*([\w\d\-_]+)\s+")
        for line in env_list:
            if line.startswith("#"): continue
            match = env_name_pattern.match(line)
            if match:
                env_name = match.group(1)
                if name_substring in env_name: return env_name
        return None
    except Exception: return None

def _get_file_hash(filepath: Path) -> str:
    """Get hash of file for cache key."""
    return hashlib.md5(f"{filepath.stem}_{filepath.stat().st_mtime}".encode()).hexdigest()[:16]

def _sanitize_adata(adata: ad.AnnData) -> ad.AnnData:
    """
    CRITICAL FIX: 
    1. Removes columns that conflict with index names.
    2. Forces coordinate columns to float.
    3. Forces object columns to string to prevent HDF5 write errors.
    """
    # 1. Fix Observation (Sample) Index Name Conflict
    if adata.obs_names.name is None:
        adata.obs_names.name = 'sample_id'
        
    idx_name = adata.obs_names.name
    if idx_name in adata.obs.columns:
        try: del adata.obs[idx_name]
        except Exception: pass

    # 2. Fix Variable (Feature) Index Name Conflict
    if adata.var_names.name is None:
        adata.var_names.name = 'feature_id'
        
    var_idx_name = adata.var_names.name
    if var_idx_name in adata.var.columns:
        try: del adata.var[var_idx_name]
        except Exception: pass

    # 3. Force Coordinates to Float
    for coord in ['lat', 'lon', 'latitude', 'longitude']:
        if coord in adata.obs.columns:
            adata.obs[coord] = pd.to_numeric(adata.obs[coord], errors='coerce')

    # 4. Force Mixed Object Columns to String
    for df in [adata.obs, adata.var]:
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            try:
                df[col] = df[col].astype(str).replace('nan', 'NaN').replace('None', 'NaN')
                if df[col].nunique() < len(df) * 0.5:
                    df[col] = df[col].astype('category')
            except Exception: pass

    return adata

def sanitize_and_save_h5ad(adata: ad.AnnData, filepath: Path):
    """Safe save wrapper that fixes index conflicts before writing."""
    adata = _sanitize_adata(adata)
    try:
        adata.write_h5ad(filepath, compression="gzip")
    except Exception as e:
        logger.warning(f"Standard save failed ({e}). Attempting aggressive cleanup...")
        adata.obs_names.name = None
        adata.var_names.name = None
        if 'sample_id' in adata.obs.columns: del adata.obs['sample_id']
        if 'feature_id' in adata.var.columns: del adata.var['feature_id']
        adata.write_h5ad(filepath)

def _validate_cached_adata(adata: ad.AnnData, cache_type: str = "file") -> tuple[bool, list[str]]:
    """Validate cached AnnData object."""
    issues = []
    if adata.n_obs == 0: issues.append("No observations")
    if adata.n_vars == 0: issues.append("No features")
    return len(issues) == 0, issues

# --- Global Merge Functions ---

def hierarchical_merge(adata_list: list, chunk_size: int = 5) -> ad.AnnData:
    """Merges a list of AnnData objects in memory. Reduced default chunk size."""
    if not adata_list: return None
    TAX_LEVELS = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    
    while len(adata_list) > 1:
        new_list = []
        log_mem(f"Merging {len(adata_list)} objects (chunk_size={chunk_size})")
        for i in range(0, len(adata_list), chunk_size):
            chunk = adata_list[i : i + chunk_size]
            if len(chunk) == 1:
                new_list.append(chunk[0])
            else:
                try:
                    # FIX: fill_value=0 ensures outer joins don't introduce NaNs in counts
                    merged_chunk = ad.concat(chunk, join='outer', merge='same', fill_value=0)
                    if issparse(merged_chunk.X): merged_chunk.X = csr_matrix(merged_chunk.X)
                    
                    # Robust Taxonomy Collection
                    valid_dfs = []
                    for obj in chunk:
                        if not obj.var.empty:
                            available = [c for c in TAX_LEVELS if c in obj.var.columns]
                            if available:
                                valid_dfs.append(obj.var[available])
                    
                    if valid_dfs:
                        combined_taxonomy = pd.concat(valid_dfs)
                        combined_taxonomy = combined_taxonomy[~combined_taxonomy.index.duplicated(keep='first')]
                        
                        for col in TAX_LEVELS:
                            if col in combined_taxonomy:
                                # FIX: Update source categories BEFORE reindexing
                                if pd.api.types.is_categorical_dtype(combined_taxonomy[col]):
                                    if 'Unassigned' not in combined_taxonomy[col].cat.categories:
                                        combined_taxonomy[col] = combined_taxonomy[col].cat.add_categories(['Unassigned'])
                                
                                merged_chunk.var[col] = combined_taxonomy[col].reindex(merged_chunk.var_names, fill_value='Unassigned')

                    new_list.append(merged_chunk)
                except Exception as e:
                    logger.error(f"Merge chunk failed: {e}")
                    merged_chunk = ad.concat(chunk, join='outer', fill_value=0)
                    new_list.append(merged_chunk)
            
            # Explicitly clear chunk to free memory
            chunk = None
            gc.collect()
            
        adata_list = new_list
        gc.collect()
        
    return adata_list[0]

def hierarchical_merge_on_disk(h5ad_paths, cache_dir, logger, chunk_size=2):
    """
    Hierarchically merge AnnData .h5ad files on disk.
    Forces sparse format to prevent memory explosion.
    """
    TAX_LEVELS = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    round_num = 0
    paths = list(h5ad_paths)
    
    while len(paths) > 1:
        new_paths = []
        logger.info(f"Disk Merge Round {round_num}: Merging {len(paths)} files...")
        
        for i in range(0, len(paths), chunk_size):
            chunk = paths[i:i+chunk_size]
            if len(chunk) == 1:
                new_paths.append(chunk[0])
                continue
                
            try:
                adatas = [sc.read_h5ad(p) for p in chunk]
                # Force sparsity on inputs
                for obj in adatas:
                    if not issparse(obj.X): obj.X = csr_matrix(obj.X)

                # FIX: fill_value=0 ensures outer joins don't introduce NaNs in counts
                merged = ad.concat(adatas, join='outer', merge='same', fill_value=0)
                # Force sparsity on output
                if not issparse(merged.X): merged.X = csr_matrix(merged.X)
                
                # Robust Taxonomy Preservation (Fixes Data Loss & Crash)
                valid_dfs = []
                for obj in adatas:
                    if not obj.var.empty:
                        available = [c for c in TAX_LEVELS if c in obj.var.columns]
                        if available:
                            valid_dfs.append(obj.var[available])

                if valid_dfs:
                    combined_taxonomy = pd.concat(valid_dfs)
                    combined_taxonomy = combined_taxonomy[~combined_taxonomy.index.duplicated(keep='first')]
                    
                    for col in TAX_LEVELS:
                        if col in combined_taxonomy:
                            if pd.api.types.is_categorical_dtype(combined_taxonomy[col]):
                                if 'Unassigned' not in combined_taxonomy[col].cat.categories:
                                    combined_taxonomy[col] = combined_taxonomy[col].cat.add_categories(['Unassigned'])
                            
                            reindexed_col = combined_taxonomy[col].reindex(merged.var_names, fill_value='Unassigned')
                            
                            if col in merged.var and pd.api.types.is_categorical_dtype(merged.var[col]):
                                if 'Unassigned' not in merged.var[col].cat.categories:
                                    merged.var[col] = merged.var[col].cat.add_categories(['Unassigned'])

                            merged.var[col] = reindexed_col
                
                merged = _sanitize_adata(merged)
                fix_adata_dtypes(merged)
                
                out_path = cache_dir / f"merge_round{round_num}_{i//chunk_size}.h5ad"
                sanitize_and_save_h5ad(merged, out_path)
                new_paths.append(out_path)
                
                del adatas, merged; gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to merge chunk {chunk}: {e}")
                raise e

        paths = new_paths
        round_num += 1
        
    return sc.read_h5ad(paths[0])

def _process_single_file(f: Path, config, cache_dir: Path = None):
    """Process a single h5ad file with threading support."""
    import gc
    from workflow_16s.downstream.steps.preprocessing import (
        clean_metadata, 
        parse_taxonomy, 
        filter_samples_and_features
    )
    try:
        if cache_dir:
            cache_file = cache_dir / f"{f.stem}_{_get_file_hash(f)}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as cf:
                        cached_adata = pickle.load(cf)
                    is_valid, _ = _validate_cached_adata(cached_adata)
                    if is_valid: return (f.stem, cached_adata, None)
                    else: cache_file.unlink()
                except Exception:
                    if cache_file.exists(): cache_file.unlink()
        
        adata = sc.read_h5ad(f)
        adata = _sanitize_adata(adata)
        adata = clean_metadata(adata, config)
        if adata is None: return (f.stem, None, "Metadata cleaning failed")
        
        if 'lat' not in adata.obs.columns: adata.obs['lat'] = np.nan
        if 'lon' not in adata.obs.columns: adata.obs['lon'] = np.nan
        for c in ['latitude', 'latitude_deg', 'LAT']:
            if c in adata.obs.columns:
                adata.obs[c] = pd.to_numeric(adata.obs[c], errors='coerce')
                adata.obs['lat'] = adata.obs['lat'].fillna(adata.obs[c])
        
        adata = parse_taxonomy(adata)
        if adata is not None: 
            adata = filter_samples_and_features(adata, config)
        
        if adata is None or adata.n_obs == 0:
            return (f.stem, None, "No observations after filtering")
        
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{f.stem}_{_get_file_hash(f)}.pkl"
            try:
                with open(cache_file, 'wb') as cf: pickle.dump(adata, cf, protocol=4)
            except: pass
            
        return (f.stem, adata, None)
    except Exception as e:
        return (f.stem, None, str(e))
    finally:
        gc.collect()

def run_fast_load(workflow):
    """Load and preprocess files using Threading (Batch-Mode) with Persistent Caching."""
    import gc
    workflow.logger.info("1. Modular Ingestion: Loading and filtering files (THREADING + BATCHED)...")
    
    # --- 1. CHECK FOR PERSISTENT CACHE ---
    # Renamed to 'merged_samples.h5ad'
    final_cache_path = workflow.output_dir / "merged_samples.h5ad"
    
    if final_cache_path.exists():
        workflow.logger.info(f"🚀 Found cached merged dataset at {final_cache_path}. Loading...")
        try:
            workflow.adata = sc.read_h5ad(final_cache_path)
            # Basic validation
            if workflow.adata.n_obs > 0:
                workflow.logger.info(f"✅ Successfully loaded cached dataset ({workflow.adata.n_obs} samples).")
                return # EXIT EARLY - SUCCESS
            else:
                workflow.logger.warning("Cached dataset was empty. Rebuilding...")
        except Exception as e:
             workflow.logger.warning(f"Failed to load cache ({e}). Rebuilding...")

    # --- 2. BUILD IF NO CACHE ---
    h5ad_files = list(workflow.data_dir.glob("*.h5ad"))
    
    if not h5ad_files:
        workflow.logger.warning(f"No h5ad files found in {workflow.data_dir}")
        return

    cache_dir = workflow.output_dir / ".cache" / "preprocessed_files"
    concat_cache_dir = workflow.output_dir / ".cache" / "concatenated"
    
    n_jobs = min(8, os.cpu_count() or 1)
    
    # Reduced batch size to manage memory
    BATCH_SIZE = 10
    total_files = len(h5ad_files)
    chunk_files = []
    
    config_dict = workflow.config.model_dump() if hasattr(workflow.config, 'model_dump') else workflow.config

    with Parallel(n_jobs=n_jobs, backend='threading', verbose=5) as parallel:
        for i in range(0, total_files, BATCH_SIZE):
            batch_files = h5ad_files[i : i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            
            workflow.logger.info(f"--- Processing Batch {batch_num} ({len(batch_files)} files) ---")
            
            batch_id = hashlib.md5("_".join(sorted([f.name for f in batch_files])).encode()).hexdigest()[:8]
            batch_cache_path = concat_cache_dir / f"batch_{batch_num}_{batch_id}.h5ad"
            
            if batch_cache_path.exists():
                workflow.logger.info(f"✅ Found cached batch: {batch_cache_path.name}")
                chunk_files.append(batch_cache_path)
                continue

            results = parallel(
                delayed(_process_single_file)(f, config_dict, cache_dir)
                for f in batch_files
            )
            
            valid_batch_adatas = []
            for f, (stem, adata, error) in zip(batch_files, results):
                if adata is not None: valid_batch_adatas.append(adata)
                elif error: workflow.logger.warning(f"Skipping {stem}: {error}")
            
            if valid_batch_adatas:
                workflow.logger.info(f"Concatenating batch {batch_num}...")
                log_mem("Before batch merge")
                # Fill value 0 prevents NaNs in counts
                batch_adata = hierarchical_merge(valid_batch_adatas, chunk_size=5)
                batch_cache_path.parent.mkdir(parents=True, exist_ok=True)
                sanitize_and_save_h5ad(batch_adata, batch_cache_path)
                chunk_files.append(batch_cache_path)
                workflow.logger.info(f"💾 Saved intermediate batch to {batch_cache_path.name}")
            
            # Explicit cleanup
            del valid_batch_adatas
            del results
            gc.collect()

    if not chunk_files:
        workflow.logger.warning("No files were successfully processed.")
        return

    workflow.logger.info(f"Merging {len(chunk_files)} batch files on disk...")
    # Use disk-based hierarchical merge for final step
    workflow.adata = hierarchical_merge_on_disk(chunk_files, concat_cache_dir, workflow.logger, chunk_size=2)
    
    if workflow.adata is not None:
        workflow.adata = _sanitize_adata(workflow.adata)
        tax_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
        missing_tax = [lvl for lvl in tax_levels if lvl not in workflow.adata.var.columns]
        if missing_tax: workflow.logger.warning(f"Missing taxonomy columns: {missing_tax}")
        else: workflow.logger.info(f"✅ All taxonomy columns present")
        
        # Ensure we have valid data
        if issparse(workflow.adata.X):
            # Fill NaNs in sparse matrix if any exist (rare but possible after concat)
            workflow.adata.X.data = np.nan_to_num(workflow.adata.X.data, nan=0.0)
        else:
            workflow.adata.X = np.nan_to_num(workflow.adata.X, nan=0.0)
            
        # --- 3. SAVE FINAL CACHE ---
        workflow.logger.info(f"💾 Saving final merged dataset to {final_cache_path}...")
        sanitize_and_save_h5ad(workflow.adata, final_cache_path)
        workflow.logger.info("✅ Merged dataset saved.")

def run_filter_empty(workflow, col='facility_match'):
    if workflow.adata is not None and col in workflow.adata.obs.columns:
        mask = workflow.adata.obs[col].isin([True, False])
        workflow.adata = workflow.adata[mask, :].copy()