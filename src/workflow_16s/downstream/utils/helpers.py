"""
Analysis Helper Utilities for the 16S Workflow.
Contains static methods for data transformation (CLR), aggregation,
metadata parsing, and plottable column finding.
"""

import re
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.sparse import csc_matrix, csr_matrix, issparse
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict
from workflow_16s.utils.logger import get_logger
import scanpy as sc
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from workflow_16s.config_schema import AppConfig

logger = get_logger("workflow_16s")

def get_cfg_value(cfg_obj, key, default=None):
    """Helper to safely get config values from dict or object."""
    if isinstance(cfg_obj, dict): return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)

def qc_metrics(adata: ad.AnnData, output_dir: Union[str, Path]) -> None:
    """Calculates and plots basic QC metrics."""
    if adata is None or adata.n_obs == 0: return
    logger.info("Calculating QC metrics...")
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    
    plot_path = Path(output_dir) / "qc_metrics.png"
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(data=adata.obs, x='total_counts', ax=axes[0], bins=30)
        sns.histplot(data=adata.obs, x='n_genes_by_counts', ax=axes[1], bins=30)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        logger.info(f"Saved QC plot: {plot_path}")
    except Exception: pass

def export_fasta(adata: ad.AnnData, config: Union[AppConfig, dict], output_dir: Union[str, Path]) -> None:
    """Exports sequences to FASTA."""
    if 'sequence' not in adata.var.columns: return
    fasta_path = Path(output_dir) / "all_features.fasta"
    try:
        with open(fasta_path, "w") as f:
            for feat_id, seq in adata.var['sequence'].dropna().items():
                f.write(f">{feat_id}\n{seq}\n")
        logger.info(f"FASTA exported to {fasta_path}")
    except Exception as e: logger.error(f"FASTA export failed: {e}")



class AnalysisUtils:
    """Contains static helper methods for analysis tasks."""
    
    # Columns to exclude from analysis/plotting automatically
    ADMIN_NOISE_COLUMNS = {
        'biosample_insdc_center_name', 'biosample_Insdc_first_public', 'biosample_investigation_type', 
        'country_facility', 'dataset', 'EnvironmentalHealth_date', 'latitude_osm', 'longitude_osm', 
        'refs', 'source_url', 'source_version', 'temporal_coverage', 'time', 'unit', 'variable', 
        'admin', 'batch', 'batch_original', 'sample_id', '#sampleid'
    }
    
    ADMIN_NOISE_PATTERNS = [re.compile(r'.*_insdc_status$')]
    
    # Priority columns that should always be included regardless of fullness threshold
    PRIORITY_COLUMNS = {
        'facility_match', 'facility_distance_km', 'facility_name', 'facility_type', 
        'latitude', 'longitude', 'lat', 'lon'
    }
    
    TAX_LEVELS_ALL = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    
    # CLR transform cache to avoid recomputation
    _clr_cache: Dict[str, pd.DataFrame] = {}
    
    @staticmethod
    def _parse_lat_lon(lat_lon_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Parses various lat/lon string formats into numeric Series."""
        if lat_lon_series is None or lat_lon_series.isnull().all(): 
            return pd.Series(dtype='float64'), pd.Series(dtype='float64')
            
        regex = r'([\d\.-]+)\s*([NS])?[\s,]+([\d\.-]+)\s*([EW])?'
        parsed = lat_lon_series.astype(str).str.extract(regex)
        
        if parsed.empty or parsed.isnull().all().all(): 
            return (pd.Series(dtype='float64', index=lat_lon_series.index), 
                    pd.Series(dtype='float64', index=lat_lon_series.index))
                    
        lat = pd.to_numeric(parsed[0], errors='coerce')
        lon = pd.to_numeric(parsed[2], errors='coerce')
        
        if 1 in parsed.columns: 
            lat[parsed[1].fillna('').str.upper() == 'S'] *= -1
        if 3 in parsed.columns: 
            lon[parsed[3].fillna('').str.upper() == 'W'] *= -1
            
        lat.index = lat_lon_series.index
        lon.index = lat_lon_series.index
        return lat, lon

    @staticmethod
    def get_analysis_adata(adata_in: ad.AnnData, level: str) -> Union[ad.AnnData, None]:
        """
        Gets an AnnData object for a specific analysis level.
        - If level is taxonomic, calls aggregate_adata_by_taxonomy.
        - If level is functional (in .obsm), creates a new AnnData object.
        """
        if level in AnalysisUtils.TAX_LEVELS_ALL: 
            logger.info(f"Getting analysis AnnData for taxonomic level: {level}")
            return AnalysisUtils.aggregate_adata_by_taxonomy(adata_in, tax_level=level)
        
        elif level in adata_in.obsm:
            logger.info(f"Getting analysis AnnData for functional level: {level}")
            try:
                func_data = adata_in.obsm[level]
                if func_data is None: 
                    logger.error(f"Data for {level} in .obsm is None")
                    return None

                # This will hold the (samples x features) data
                data_for_anndata: Union[pd.DataFrame, csc_matrix, csr_matrix, np.ndarray]

                if not isinstance(func_data, pd.DataFrame):
                    # --- BRANCH 1: Data is an array ---
                    # Assumed (features x samples) or (samples x features)
                    # We need to guess based on shape
                    logger.debug(f"Converting array data from .obsm['{level}']")
                    
                    if f"{level}_ids" in adata_in.uns: 
                        feature_names = adata_in.uns[f"{level}_ids"]
                    elif hasattr(func_data, 'shape') and func_data.shape is not None and func_data.shape[0] > 0: 
                        # Fallback feature naming
                        # If shape[1] matches n_obs, likely (features x samples)
                        if func_data.shape[1] == adata_in.n_obs:
                             feature_names = [f"{level}_{i}" for i in range(func_data.shape[0])]
                        else:
                             feature_names = [f"{level}_{i}" for i in range(func_data.shape[1])]
                    else: 
                        logger.error(f"Cannot determine feature names for {level}")
                        return None
                    
                    # Determine Orientation
                    if hasattr(func_data, 'shape') and func_data.shape is not None:
                        if func_data.shape[1] == adata_in.n_obs:
                             # (Features x Samples) -> Need Transpose
                             func_df = pd.DataFrame(func_data, index=feature_names, columns=adata_in.obs_names)
                             data_for_anndata = func_df.T
                        elif func_data.shape[0] == adata_in.n_obs:
                             # (Samples x Features) -> Correct
                             data_for_anndata = func_data
                        else:
                            logger.error(f"Shape mismatch for {level}: {func_data.shape} vs adata.n_obs {adata_in.n_obs}")
                            return None
                    else:
                        return None
                
                else:
                    # --- BRANCH 2: Data is a DataFrame ---
                    logger.debug(f"Using existing DataFrame from .obsm['{level}'].")
                    func_df = func_data.copy()
                    
                    # Validation check: Orientation
                    if func_df.shape[0] != adata_in.n_obs:
                        logger.warning(f"DataFrame in .obsm['{level}'] has {func_df.shape[0]} rows, but adata has {adata_in.n_obs} obs.")
                        logger.warning(f"Attempting to transpose, assuming (features x samples).")
                        
                        if func_df.shape[1] != adata_in.n_obs:
                            logger.error(f"Shape mismatch: DataFrame is {func_df.shape}, adata is {adata_in.n_obs} obs. Cannot orient.")
                            return None
                        # Use the transposed data
                        data_for_anndata = func_df.T
                    else:
                        # Shape matches (samples x features), use as-is
                        data_for_anndata = func_df

                # Create new AnnData object (now using data_for_anndata)
                adata_func = ad.AnnData(data_for_anndata)
                adata_func.obs = adata_in.obs.loc[adata_func.obs_names].copy()
                adata_func.var.index.name = level
                
                # IMPORTANT: Create the 'raw_counts' layer for downstream functions
                if isinstance(data_for_anndata, pd.DataFrame):
                    counts_values = data_for_anndata.values
                elif issparse(data_for_anndata):
                    counts_values = data_for_anndata.toarray()
                else:
                    counts_values = np.asarray(data_for_anndata)

                if (counts_values < 0).any():
                    logger.warning(f"Negative values found in {level}. Shifting to non-negative for 'raw_counts'.")
                    min_val = counts_values.min()
                    adata_func.layers['raw_counts'] = counts_values - min_val
                else:
                    adata_func.layers['raw_counts'] = counts_values.copy()
                    
                # Set .X to be the same as 'raw_counts' for functions that read .X
                adata_func.X = adata_func.layers['raw_counts'].copy()
                
                logger.info(f"Created functional AnnData: {adata_func.shape}")
                return adata_func
            
            except Exception as e: 
                logger.error(f"Failed to create AnnData from obsm key '{level}': {e}", exc_info=True)
                return None
        else: 
            logger.warning(f"Analysis level '{level}' not found in taxonomy or .obsm. Skipping.")
            return None
        
    @staticmethod
    def aggregate_adata_by_taxonomy(adata_in: ad.AnnData, tax_level: str = 'Genus') -> Union[ad.AnnData, None]:
        """Aggregates an AnnData object using efficient sparse matrix multiplication."""
        logger.info(f"--- Aggregating AnnData by {tax_level} ---")
        adata_copy = adata_in.copy()
        
        if tax_level not in adata_copy.var.columns: 
            logger.error(f"Tax level '{tax_level}' not in .var.")
            return None
            
        # ==================================================================
        # 1. ROBUST STANDARDIZATION (The Fix)
        # ==================================================================
        # Force conversion to string to handle Categoricals AND Objects identically.
        # This fixes the " g__X" (leading space) issue seen in your data.
        tax_series = adata_copy.var[tax_level].astype(str).str.strip()
        
        # Unify all forms of "empty" to 'Unassigned'
        # Note: We include 'nan' string because astype(str) converts np.nan to 'nan'
        tax_series = tax_series.replace(
            ['nan', 'NaN', '<NA>', 'None', '', 'NoneType'], 
            'Unassigned'
        )
        
        # Fill any remaining real NaNs
        tax_series = tax_series.fillna('Unassigned')
        
        # Update the dataframe
        adata_copy.var[tax_level] = tax_series
        # ==================================================================

        # Get Counts Matrix
        if 'raw_counts' in adata_copy.layers: 
            counts_mtx = adata_copy.layers['raw_counts']
            logger.debug("Using 'raw_counts'.")
        else: 
            logger.warning("Using '.X' for aggregation.")
            counts_mtx = adata_copy.X
            
        if sparse.issparse(counts_mtx): 
            counts_mtx = counts_mtx.tocsc()
        elif hasattr(counts_mtx, 'toarray'): 
            counts_mtx = counts_mtx.toarray()
        elif not isinstance(counts_mtx, np.ndarray): 
            counts_mtx = np.array(counts_mtx)
            
        asv_to_tax_map = adata_copy.var[tax_level]
        logger.debug("Creating sparse grouper matrix for aggregation...")
        
        # 2. Get unique taxa and their indices
        unique_taxa, group_indices = np.unique(asv_to_tax_map, return_inverse=True)

        # 3. Create the (n_groups x n_features) grouper matrix
        n_features = adata_copy.n_vars
        n_groups = len(unique_taxa)
        
        if counts_mtx is not None and hasattr(counts_mtx, 'dtype') and counts_mtx.dtype is not None:
            grouper_dtype = np.dtype(counts_mtx.dtype)
        else:
            grouper_dtype = np.float64
            
        M_grouper = csc_matrix(
            (np.ones(n_features, dtype=grouper_dtype), (group_indices, np.arange(n_features))), 
            shape=(n_groups, n_features)
        )

        # 4. Perform the aggregation with matrix multiplication
        if not isinstance(counts_mtx, csr_matrix):
            if issparse(counts_mtx):
                counts_mtx = counts_mtx.tocsr()
            else:
                counts_mtx = csr_matrix(counts_mtx)
                
        logger.debug("Performing sparse aggregation...")
        agg_mtx = counts_mtx @ M_grouper.T 

        # 5. Create the new AnnData
        new_var = pd.DataFrame(index=unique_taxa)
        new_var.index.name = tax_level
        
        if not isinstance(agg_mtx, csr_matrix):
            agg_mtx = csr_matrix(agg_mtx)

        adata_new = ad.AnnData(
            agg_mtx, 
            obs=adata_copy.obs.copy(), 
            var=new_var, 
            dtype=agg_mtx.dtype
        )
        adata_new.layers['raw_counts'] = csr_matrix(adata_new.X)
        logger.debug(f"Counts aggregation resulted in {adata_new.n_vars} groups.")
        
        # 6. Aggregate .var metadata
        # (This block remains largely the same, but relies on the cleaned taxonomy)
        tax_levels_all = AnalysisUtils.TAX_LEVELS_ALL
        levels_to_keep = tax_levels_all[:tax_levels_all.index(tax_level) + 1] if tax_level in tax_levels_all else [tax_level]
        
        logger.debug(f"Aggregating .var metadata up to {tax_level}...")
        try:
            var_meta_orig = adata_in.var.copy()
            # Use the CLEANED series for grouping metadata
            var_meta_orig['__group_key__'] = tax_series.values 
            
            agg_funcs = {}
            levels_present = [lvl for lvl in levels_to_keep if lvl in var_meta_orig.columns]
            
            for lvl in levels_present: 
                agg_funcs[lvl] = lambda series: series.dropna().iloc[0] if not series.dropna().empty else np.nan
            
            func_prefixes = ("FAPROTAX_", "CUSTOM_", "PICRUST_") 
            func_cols = [c for c in var_meta_orig.columns if c.startswith(func_prefixes) and var_meta_orig[c].dtype == bool]
            
            if func_cols: 
                for f_col in func_cols: agg_funcs[f_col] = 'any'
            
            other_cols = [c for c in var_meta_orig.columns if c not in agg_funcs and c != '__group_key__']
            for o_col in other_cols: 
                agg_funcs[o_col] = lambda series: series.dropna().iloc[0] if not series.dropna().empty else np.nan
            
            grouped_meta = var_meta_orig.groupby('__group_key__', observed=False).agg(agg_funcs)
            adata_new.var = grouped_meta.reindex(unique_taxa)
            adata_new.var.index.name = tax_level
            
        except Exception as e: 
            logger.warning(f"Could not aggregate .var metadata: {e}")
            adata_new.var_names = unique_taxa.astype(str).tolist()
            adata_new.var = pd.DataFrame(index=adata_new.var_names)
            adata_new.var.index.name = tax_level
            
        # Final formatting
        adata_new.var_names = adata_new.var_names.astype(str).tolist()
        adata_new.var.index = adata_new.var.index.astype(str)
        
        # [CRITICAL SAFETY CHECK]
        if 'Unassigned' in adata_new.var_names:
            if len(adata_new.var_names) == 1:
                logger.warning(f"⚠️ All features mapped to 'Unassigned'! Keeping it to avoid empty dataset.") 
            else:
                logger.info("Filtering 'Unassigned' taxa.")
                adata_new = adata_new[:, adata_new.var_names != 'Unassigned'].copy()
            
        logger.info(f"Aggregation complete. New shape: {adata_new.shape}")
        return adata_new
    
    @staticmethod
    def clr_transform(adata_in: ad.AnnData, pseudocount: float = 1.0, use_cache: bool = True) -> pd.DataFrame:
        """Centered log-ratio transformation with optional caching."""
        # [FIX] Force conversion to string tuple to avoid 'implicit conversion' warnings
        obs_key = hash(tuple(str(x) for x in adata_in.obs_names))
        var_key = hash(tuple(str(x) for x in adata_in.var_names))
        cache_key = f"{obs_key}_{var_key}_{pseudocount}"
        
        # Check cache
        if use_cache and cache_key in AnalysisUtils._clr_cache:
            logger.debug(f"Using cached CLR transform (cache_key={cache_key[:16]}...)")
            return AnalysisUtils._clr_cache[cache_key]
        
        logger.debug(f"Computing CLR transform (pseudocount={pseudocount})")
        
        if 'raw_counts' in adata_in.layers: 
            counts_mtx = adata_in.layers['raw_counts']
        else: 
            counts_mtx = adata_in.X
            
        if sparse.issparse(counts_mtx): 
            counts_mtx = counts_mtx.toarray()
        if 'sparse' in str(type(counts_mtx)): 
            counts_mtx = counts_mtx.toarray()
            
        counts_mtx = np.asarray(counts_mtx)
        counts_mtx_pseudo = counts_mtx + pseudocount
        log_counts = np.log(counts_mtx_pseudo)
        geom_mean_log = np.mean(log_counts, axis=1, keepdims=True)
        clr_data = log_counts - geom_mean_log
        
        clr_df = pd.DataFrame(clr_data, index=adata_in.obs_names, columns=adata_in.var_names)
        
        # Store in cache
        if use_cache:
            AnalysisUtils._clr_cache[cache_key] = clr_df
            
        return clr_df
    
    @staticmethod
    def clear_clr_cache():
        """Clear the CLR transform cache to free memory."""
        n_cached = len(AnalysisUtils._clr_cache)
        AnalysisUtils._clr_cache.clear()
        logger.info(f"Cleared CLR cache ({n_cached} entries removed)")

    # [NEW METHOD ADDED HERE TO FIX CRASH]
    @staticmethod
    def apply_transform(adata, method='clr'):
        """
        Applies normalization/transformation to an AnnData object or DataFrame.
        Returns a DataFrame of transformed features.
        """
        # CLR requires AnnData object input to access .layers/etc correctly
        if method == 'clr':
            return AnalysisUtils.clr_transform(adata)

        # For simple math transforms, we can work on the DataFrame
        if hasattr(adata, 'to_df'):
            df = adata.to_df()
        else:
            df = adata.copy()

        if method == 'log1p':
            return np.log1p(df)
        elif method == 'binary':
            return (df > 0).astype(int)
        else:
            # Default: no transform
            return df
    
    @staticmethod
    def filter_ml_targets(
        adata: ad.AnnData,
        target_cols: List[str],
        min_samples_per_group: int = 10,
        max_classes: int = 10
    ) -> Dict[str, Union[List[str], Dict]]:
        """Pre-filter ML targets to avoid small class size warnings."""
        valid_targets = []
        skipped_targets = []
        skip_reasons = {}
        
        for col in target_cols:
            if col not in adata.obs.columns:
                skip_reasons[col] = 'column not found'
                skipped_targets.append(col)
                continue
            
            y_series = adata.obs[col]
            y_clean = y_series.dropna()
            
            if len(y_clean) < 20:
                skip_reasons[col] = f'too few samples ({len(y_clean)} < 20)'
                skipped_targets.append(col)
                continue
            
            # Check if numeric and suitable for regression or classification
            is_numeric = pd.api.types.is_numeric_dtype(y_clean)
            n_unique = y_clean.nunique()
            
            if is_numeric and n_unique > max_classes:
                # Regression target - just check sample size
                valid_targets.append(col)
            elif not is_numeric or n_unique <= max_classes:
                # Classification target - check class sizes
                class_counts = y_clean.value_counts()
                if class_counts.min() < min_samples_per_group:
                    skip_reasons[col] = f'small class size (min: {class_counts.min()} < {min_samples_per_group})'
                    skipped_targets.append(col)
                    continue
                if n_unique > max_classes:
                    skip_reasons[col] = f'too many classes ({n_unique} > {max_classes})'
                    skipped_targets.append(col)
                    continue
                valid_targets.append(col)
            else:
                valid_targets.append(col)
        
        # Log summary
        if skipped_targets:
            logger.info(f"Pre-filtered ML targets: {len(valid_targets)} valid, {len(skipped_targets)} skipped")
            # Group skip reasons
            reason_groups = defaultdict(list)
            for col, reason in skip_reasons.items():
                reason_groups[reason].append(col)
            for reason, cols in reason_groups.items():
                logger.debug(f"  Skipped ({reason}): {', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}")
        
        return {'valid': valid_targets, 'skipped': skipped_targets, 'reasons': skip_reasons}

    @staticmethod
    def find_plottable_metadata(adata: ad.AnnData, fullness_threshold: float = 0.4, max_categories: int = 50) -> Dict[str, List[str]]:
        """
        Identifies plottable numeric and categorical columns in .obs.
        Consolidates exclusion logging to avoid clutter.
        """
        logger.info(f"Identifying plottable metadata (Fullness > {fullness_threshold}, Max Categories < {max_categories})...")
        categorical_cols = []
        numeric_cols = []
        excluded_cols: Dict[str, str] = {} 
        obs_df = adata.obs
        n_obs = adata.n_obs

        if n_obs == 0: 
            logger.warning("No observations.")
            return {'categorical': [], 'numeric': []}
            
        for col in obs_df.columns:
            # Check for exclusion criteria
            if col in AnalysisUtils.ADMIN_NOISE_COLUMNS: 
                excluded_cols[col] = 'admin list'
                continue
                
            if any(pattern.match(col) for pattern in AnalysisUtils.ADMIN_NOISE_PATTERNS): 
                excluded_cols[col] = 'admin pattern'
                continue
            
            # Check for priority columns (exempt from fullness threshold)
            is_priority = col in AnalysisUtils.PRIORITY_COLUMNS or col.startswith('facility_')
            
            non_null_count = obs_df[col].notna().sum()
            fullness = non_null_count / n_obs if n_obs > 0 else 0
            
            # Apply fullness threshold EXCEPT for priority columns
            if not is_priority and fullness < fullness_threshold: 
                excluded_cols[col] = f'low fullness ({fullness:.1%})'
                continue
            
            col_series = obs_df[col]
            col_dtype = col_series.dtype
            
            # For boolean columns, only count True/False (exclude NaN)
            if isinstance(col_dtype, type(pd.BooleanDtype())) or pd.api.types.is_bool_dtype(col_dtype):
                n_unique = col_series.dropna().nunique() 
            else:
                n_unique = col_series.nunique()
                
            if n_unique <= 1: 
                excluded_cols[col] = '1 unique value'
                continue
                
            # --- Classification Logic ---
            
            # 1. Check if column is NUMERIC first (BEFORE checking cardinality)
            if pd.api.types.is_numeric_dtype(col_dtype):
                # Treat low-cardinality integers as categorical (but not floats)
                if pd.api.types.is_integer_dtype(col_dtype) and n_unique < max_categories / 2: 
                    categorical_cols.append(col)
                else: 
                    numeric_cols.append(col)
                continue
            
            # 2. Boolean types
            if isinstance(col_dtype, type(pd.BooleanDtype())) or pd.api.types.is_bool_dtype(col_dtype):
                categorical_cols.append(col)
                continue
                    
            # 3. String / Object / Categorical
            if pd.api.types.is_string_dtype(col_dtype) or pd.api.types.is_object_dtype(col_dtype) or isinstance(col_dtype, pd.CategoricalDtype):
                # Now check cardinality for non-numeric types
                if n_unique > max_categories: 
                    excluded_cols[col] = f'high cardinality categorical ({n_unique})'
                    continue
                    
                # Handle 'object' dtype (could be mixed)
                if pd.api.types.is_object_dtype(col_dtype):
                    try:
                        temp_series = col_series.dropna()
                        if temp_series.empty: 
                            excluded_cols[col] = 'all NA'
                            continue
                        
                        # Sampling for speed
                        sample_size = min(100, len(temp_series))
                        sample = temp_series.iloc[:sample_size]
                        numeric_sample = pd.to_numeric(sample, errors='coerce')
                        
                        if numeric_sample.notna().sum() / len(sample) > 0.9:
                            # Verify with full column if sample looks numeric
                            numeric_check = pd.to_numeric(temp_series, errors='coerce')
                            if numeric_check.notna().sum() / len(temp_series) > 0.9:
                                numeric_cols.append(col)
                            else:
                                categorical_cols.append(col)
                        else:
                            categorical_cols.append(col)
                            
                    except Exception as e: 
                        logger.warning(f"Error checking object column '{col}': {e}. Treating as categorical.")
                        categorical_cols.append(col)
                else: 
                    categorical_cols.append(col)
            else: 
                excluded_cols[col] = f'unknown dtype ({col_dtype})'

        # --- Grouped Logging ---
        logger.info(f"Found {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns eligible for analysis.")
        
        if excluded_cols:
            reason_groups: Dict[str, List[str]] = defaultdict(list)
            for col, reason in excluded_cols.items():
                reason_groups[reason].append(col)
            
            logger.debug("--- Summary of Excluded Metadata Columns ---")
            for reason, cols in sorted(reason_groups.items()):
                count = len(cols)
                example_cols = ", ".join(cols[:5])
                if count > 5:
                    example_cols += "..."
                logger.debug(f"Reason: '{reason}' ({count} total): [{example_cols}]")
            logger.debug("--- End of Excluded Columns Summary ---")
        
        return {'categorical': sorted(categorical_cols), 'numeric': sorted(numeric_cols)}