# workflow_16s/src/workflow_16s/downstream/steps/ingestion.py
import os
# Set threads to 1 to prevent library-level contention
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
from typing import Optional
import warnings
import logging

import gc
import numpy as np
import pandas as pd
import scanpy as sc

from joblib import Parallel, delayed

from workflow_16s.utils.logger import get_logger
from workflow_16s.utils.progress import get_progress_bar
from workflow_16s.utils.io.anndata import (
    _process_single_file, _sanitize_obs, 
    hierarchical_merge, quick_taxonomy_check,
    safe_outer_merge
)

# Suppress annoying implicit modification warnings from pandas
pd.options.mode.chained_assignment = None

# --- SCANPY & WARNINGS LOGGING BRIDGE ---
# 1. Silence standard Scanpy INFO logs (0 = error, 1 = warning, 2 = info, 3 = hint)
sc.settings.verbosity = 1 

# 2. Tell Python to send all warnings to the logging module instead of printing them
logging.captureWarnings(True)

# NEW: Completely mute all FutureWarnings (pandas deprecations) so they don't spam the logs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="anndata.*")

# 3. Connect the warnings logger to your custom formatting
_warnings_logger = logging.getLogger("py.warnings")
_custom_logger = get_logger("workflow_16s")

# Check if we need to link them (prevent double-linking)
if not _warnings_logger.handlers and _custom_logger.handlers:
    for handler in _custom_logger.handlers:
        _warnings_logger.addHandler(handler)


def identify_trainable_samples(adata, logger=None) -> np.ndarray:
    """
    Identify samples from 'trainable' studies/groups.
    
    Trainable = Has ≥2 classes with ≥10 samples in minority class
    
    Args:
        adata: AnnData object with target classification column
        logger: Optional logger instance
        
    Returns:
        Boolean numpy array indicating trainable sample indices
    """
    if logger is None:
        logger = get_logger("workflow_16s")
    
    # Identify target column (class labels)
    target_col = None
    if 'env_broad_scale' in adata.obs.columns:
        target_col = 'env_broad_scale'
    elif hasattr(adata, 'target_col') and adata.target_col in adata.obs.columns:
        target_col = adata.target_col
    elif 'target' in adata.obs.columns:
        target_col = 'target'
    else:
        # No target column found, return all samples as trainable
        logger.debug("No target column found; treating all samples as trainable")
        return np.ones(adata.n_obs, dtype=bool)
    
    # Check class distribution
    class_counts = adata.obs[target_col].value_counts()
    
    # Minimum requirements from user spec
    min_classes = 2
    min_minority_samples = 10
    
    if len(class_counts) < min_classes:
        logger.debug(f"Data has {len(class_counts)} classes; need ≥{min_classes}. All samples trainable.")
        return np.ones(adata.n_obs, dtype=bool)
    
    if class_counts.min() < min_minority_samples:
        minority_count = class_counts.min()
        logger.debug(f"Minority class has {minority_count} samples; need ≥{min_minority_samples}. All samples trainable.")
        return np.ones(adata.n_obs, dtype=bool)
    
    # All checks passed
    trainable = np.ones(adata.n_obs, dtype=bool)
    logger.info(f" ✅ Trainability check: {len(class_counts)} classes detected, minority class has {class_counts.min()} samples. All {adata.n_obs} samples trainable.")
    
    return trainable


def load_data(workflow):
    """
    Main Ingestion Function.
    """
    logger = get_logger("workflow_16s")
    final_cache = workflow.output_dir / "merged_samples.h5ad"
    if hasattr(workflow.config.downstream, 'load_existing') and workflow.config.downstream.load_existing is True:
        # Check for existing merged cache before doing any work
        if final_cache.exists():
            logger.info(f" 💿 Found existing merged cache at {final_cache.name}. Attempting to load...")
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Observation names are not unique")
                    workflow.adata = sc.read_h5ad(final_cache)
                logger.info(f" ✅ Loaded cached file with {workflow.adata.n_obs} samples and {workflow.adata.n_vars} features.")
                # Note: Subsampling is now handled in workflow.py before preprocessing/rCLR for consistent ordering
                quick_taxonomy_check(workflow.adata)
                return
            except Exception:
                logger.warning(" ⚠️  Cached file corrupt. Will attempt to reload from source.")
        else:
            logger.info(f" 💿 No existing merged cache found at {final_cache}. Starting fresh ingestion.")
            
    # Define the directory where _process_single_file will save pickles
    preproc_cache_dir = workflow.output_dir / ".cache" / "preprocessed_files"
    preproc_cache_dir.mkdir(parents=True, exist_ok=True)

    input_files = list(workflow.data_dir.glob("*.h5ad"))
    if not input_files: logger.error(f" 🚫 No .h5ad files found in {workflow.data_dir}"); return
    logger.info(f" ✅ Found {len(input_files)} datasets. Starting parallel ingestion...")

    # Parallel Loading & Incremental Merging
    if hasattr(workflow.config.downstream, 'batch_size'): batch_size = workflow.config.downstream.batch_size
    else: batch_size = 10
    
    chunks = [input_files[i:i + batch_size] for i in range(0, len(input_files), batch_size)]
    # ingestion.py

    # 1. Check if running under orchestrated dashboard (with telemetry)
    # If so, skip Rich progress bar (can't have nested Live displays)
    # If running standalone, create progress bar
    has_telemetry = hasattr(workflow, 'telemetry')
    progress = None
    managed_context = None
    
    if not has_telemetry:
        # Running standalone - create progress bar
        progress = getattr(workflow, 'progress_bar', None)
        if progress is None:
            from workflow_16s.utils.progress import get_progress_bar
            progress = get_progress_bar()
            managed_context = progress  # Must enter context
        
        task_id = progress.add_task("📥 Ingesting Batches", total=len(chunks))
    else:
        # Running under orchestrated dashboard - skip progress bar
        from contextlib import nullcontext
        managed_context = nullcontext()
        task_id = None

    # 2. If we created a new progress bar, we must enter the context
    from contextlib import nullcontext
    context_to_use = managed_context if managed_context else nullcontext()
    
    with context_to_use:
        first_chunk = True
        for i, chunk in enumerate(chunks):
            # Process each file in the chunk in parallel, with caching of intermediate results
            processed_chunk = Parallel(n_jobs=workflow.n_cpus)(
                delayed(_process_single_file)(f, workflow.config, cache_dir=preproc_cache_dir) 
                for f in chunk
            )
            valid_adatas = [a for a in processed_chunk if a is not None]
        
            # Merge valid adatas from this chunk and incrementally merge with total
            if valid_adatas:
                # Merge in RAM to avoid disk I/O overhead
                chunk_merged = hierarchical_merge(valid_adatas) 
                if chunk_merged:
                    chunk_merged = _sanitize_obs(chunk_merged)
                    if first_chunk:
                        # Initial write to disk
                        chunk_merged.write_h5ad(final_cache)
                        first_chunk = False
                    else:
                        # Incremental merge: Read current total, add new batch, write back
                        existing_total = sc.read_h5ad(final_cache)
                        updated_total = safe_outer_merge([existing_total, chunk_merged])
                        updated_total = _sanitize_obs(updated_total) # Sanitize after merge as well
                        updated_total.write_h5ad(final_cache)
                    
                        del existing_total
                        del updated_total
        
                    del chunk_merged

            del processed_chunk
            del valid_adatas
            gc.collect()
            current_n = sc.read_h5ad(final_cache, backed='r').n_obs
            
            # Update progress bar only if using one (not under orchestrated dashboard)
            if progress and task_id is not None:
                progress.update(
                    task_id,
                    advance=1,
                    description=f"📥 Ingesting (Samples: {current_n:,})")
            else:
                # Under orchestrated dashboard - just log
                logger.debug(f"📥 Processed batch [{i+1}/{len(chunks)}]. Current samples: {current_n:,}")
            #logger.info(f" ✅ Processed and saved batch [{i+1}/{len(chunks)}]. Current total samples: {sc.read_h5ad(final_cache).n_obs}")

    # Final load
    if final_cache.exists():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Observation names are not unique")
            workflow.adata = sc.read_h5ad(final_cache)
            # TEMP
        #if workflow.is_large_scale:
            _normalize_microbeatlas(workflow)
        logger.info(f" ✅ Ingestion Complete: {workflow.adata.n_obs} samples.")
    else:
        logger.error(" 🚫 No valid data was merged.")

# workflow_16s/src/workflow_16s/downstream/steps/ingestion.py

def _normalize_microbeatlas(workflow):
    """
    Standardizes MicrobeAtlas metadata and taxonomy to match
    internal workflow expectations.
    """
    if workflow.adata is None:
        return

    logger = get_logger("workflow_16s")
    adata = workflow.adata
    
    # NOTE: Subsampling is deferred to AFTER backfill to enrich all samples with metadata
    # Trainable sample identification happens here for diagnostics
    trainable_mask = identify_trainable_samples(adata, logger=logger)

    # 1. Detect MicrobeAtlas context
    # We look for 'Environments' or 'MAP_SID' which are unique to our merge
    if 'Environments' not in adata.obs.columns and 'MAP_SID' not in adata.obs.columns:
        return

    logger.info("🕵️ MicrobeAtlas signature detected. Normalizing schema...")

    # 2. Coordinate & Physical Mapping
    # Standardizes names for SoilGrids/Meteostat backfill and plotting
    mapping = {
        'LatitudeParsed': 'latitude',
        'LongitudeParsed': 'longitude',
        'ph': 'measured_ph',
        'temp_C': 'measured_temp',
        'depth_m': 'sample_depth',
        'Environments': 'env_broad_scale' # Keep original but map to a standard name
    }

    # Rename only if the target name doesn't already exist
    rename_dict = {k: v for k, v in mapping.items() if k in adata.obs.columns and v not in adata.obs.columns}
    if rename_dict:
        adata.obs.rename(columns=rename_dict, inplace=True)
        logger.info(f" 🔗 Mapped {len(rename_dict)} MicrobeAtlas columns to internal schema.")

    # 2b. Backward Compatibility: Create short-form aliases for coordinates
    # Backfill and other modules may expect 'lat'/'lon' instead of 'latitude'/'longitude'
    if 'latitude' in adata.obs.columns and 'lat' not in adata.obs.columns:
        adata.obs['lat'] = adata.obs['latitude']
        logger.debug(" 🔄 Created backward-compatible 'lat' column from 'latitude'")
    
    if 'longitude' in adata.obs.columns and 'lon' not in adata.obs.columns:
        adata.obs['lon'] = adata.obs['longitude']
        logger.debug(" 🔄 Created backward-compatible 'lon' column from 'longitude'")
    
    # Log coordinate coverage before continuing
    if 'latitude' in adata.obs.columns and 'longitude' in adata.obs.columns:
        valid_coords = ((adata.obs['latitude'].notna()) & (adata.obs['longitude'].notna())).sum()
        coverage_pct = 100 * valid_coords / len(adata) if len(adata) > 0 else 0
        logger.info(f" 📍 Coordinate coverage: {valid_coords}/{len(adata)} samples ({coverage_pct:.1f}%)")

    # 3. Taxonomy Splitting
    # MicrobeAtlas 'var' usually has 'taxonomy' as: k__Bacteria; p__Proteobacteria...
    # We need to split this into the 7 levels defined in workflow.py
    if 'taxonomy' in adata.var.columns:
        logger.info(" 🧬 Expanding MicrobeAtlas taxonomy strings...")
        # Split by semicolon
        tax_split = adata.var['taxonomy'].str.split('; ', expand=True)

        for i, level in enumerate(workflow.TAX_LEVELS):
            if i < tax_split.shape[1]:
                # Remove prefixes (k__, p__) and set column
                adata.var[level] = tax_split[i].str.replace(r'^[a-z]__', '', regex=True)

        # Clean up 'Taxon' column for plotting
        if 'Taxon' not in adata.var.columns:
            adata.var['Taxon'] = adata.var['Genus'].fillna(adata.var['Family']).fillna('Unassigned')

    # 4. Set priority categorical column
    # If your config doesn't specify a group, use 'env_broad_scale'
    if not hasattr(workflow.config, 'group_column') or not workflow.config.group_column:
        workflow.config.group_column = 'env_broad_scale'

def cleanup(workflow):
    """
    Post-Ingestion Cleanup. Ensures no empty samples or zero-count features remain 
    after the merge.
    """
    logger = get_logger("workflow_16s")
    logger.info("--- Post-Ingestion Filtering ---")
    if workflow.adata is None: return

    n_samples_pre = workflow.adata.n_obs
    n_features_pre = workflow.adata.n_vars

    # Filter empty samples (rows with 0 counts)
    sc.pp.filter_cells(workflow.adata, min_counts=1)

    # Filter empty features (columns with 0 counts across ALL samples)
    sc.pp.filter_genes(workflow.adata, min_cells=1)

    n_samples_post = workflow.adata.n_obs
    n_features_post = workflow.adata.n_vars

    if n_samples_pre != n_samples_post: 
        logger.info(f" ⚠️ Removed {n_samples_pre - n_samples_post} empty samples.")
    if n_features_pre != n_features_post: 
        logger.info(f" ⚠️ Removed {n_features_pre - n_features_post} empty features (not present in any sample).")
    logger.info(f" ✅ Final Data Shape: {n_samples_post} samples x {n_features_post} features.")

def find_conda_env_by_substring(substring: str, logger=None) -> Optional[Path]:
    """
    Locates a conda environment path by searching for a substring (e.g., 'picrust2').
    Used to auto-discover external tool environments.
    """
    import subprocess
    
    try:
        # Get all environments from conda
        result = subprocess.check_output(["conda", "env", "list"], text=True)
        
        for line in result.splitlines():
            # Skip comments
            if line.startswith("#") or not line.strip(): continue
            # Parse line: "env_name   * /path/to/env"
            parts = line.split()
            if len(parts) >= 1:
                env_path = parts[-1] 
                if substring in env_path or (len(parts) > 1 and substring in parts[0]):
                    if logger: logger.info(f" ✅ Found conda env for '{substring}': {env_path}")
                    return Path(env_path)
        return None
        
    except Exception as e:
        if logger: logger.warning(f" ⚠️ Could not query conda environments to find '{substring}': {e}")
        return None
