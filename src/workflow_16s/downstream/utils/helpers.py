"""
Analysis Helper Utilities for the 16S Workflow.
Contains static methods for data transformation (CLR), aggregation,
metadata parsing, and plottable column finding.
"""

import logging
import anndata as ad
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple
from scipy import sparse
from scipy.sparse import spmatrix, csc_matrix, csr_matrix, issparse
from collections import defaultdict # Added for easier grouping

logger = logging.getLogger("workflow_16s")

class AnalysisUtils:
    """Contains static helper methods for analysis tasks."""
    ADMIN_NOISE_COLUMNS = {'biosample_insdc_center_name', 'biosample_Insdc_first_public', 'biosample_investigation_type', 'country_facility', 'dataset', 'EnvironmentalHealth_date', 'latitude_osm', 'longitude_osm', 'refs', 'source_url', 'source_version', 'temporal_coverage', 'time', 'unit', 'variable', 'admin', 'batch', 'batch_original', 'sample_id', '#sampleid'}
    ADMIN_NOISE_PATTERNS = [re.compile(r'.*_insdc_status$')]
    # Priority columns that should always be included regardless of fullness threshold
    PRIORITY_COLUMNS = {'facility_match', 'facility_distance_km', 'facility_name', 'facility_type', 'latitude', 'longitude', 'lat', 'lon'}
    TAX_LEVELS_ALL = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    
    # CLR transform cache to avoid recomputation
    _clr_cache: Dict[str, pd.DataFrame] = {}
    
    @staticmethod
    def _parse_lat_lon(lat_lon_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        if lat_lon_series is None or lat_lon_series.isnull().all(): return pd.Series(dtype='float64'), pd.Series(dtype='float64')
        regex = r'([\d\.-]+)\s*([NS])?[\s,]+([\d\.-]+)\s*([EW])?'; parsed = lat_lon_series.astype(str).str.extract(regex)
        if parsed.empty or parsed.isnull().all().all(): return (pd.Series(dtype='float64', index=lat_lon_series.index), pd.Series(dtype='float64', index=lat_lon_series.index))
        lat = pd.to_numeric(parsed[0], errors='coerce'); lon = pd.to_numeric(parsed[2], errors='coerce')
        if 1 in parsed.columns: lat[parsed[1].fillna('').str.upper() == 'S'] *= -1
        if 3 in parsed.columns: lon[parsed[3].fillna('').str.upper() == 'W'] *= -1
        lat.index = lat_lon_series.index; lon.index = lat_lon_series.index; return lat, lon

    @staticmethod
    def get_analysis_adata(adata_in: ad.AnnData, level: str) -> ad.AnnData | None:
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
                data_for_anndata: pd.DataFrame | csc_matrix | csr_matrix | np.ndarray 

                if not isinstance(func_data, pd.DataFrame):
                    # --- BRANCH 1: Data is an array ---
                    # Assumed (features x samples)
                    logger.debug(f"Converting array data from .obsm['{level}']")
                    
                    if f"{level}_ids" in adata_in.uns: 
                        feature_names = adata_in.uns[f"{level}_ids"]
                    elif hasattr(func_data, 'shape') and func_data.shape is not None and func_data.shape[0] > 0: 
                        feature_names = [f"{level}_{i}" for i in range(func_data.shape[0])] # type: ignore
                    else: 
                        logger.error(f"Cannot determine feature names for {level}")
                        return None
                    
                    if not hasattr(func_data, 'shape') or func_data.shape is None or func_data.shape[1] != adata_in.n_obs: 
                        logger.error(f"Shape mismatch for {level}: {getattr(func_data, 'shape', 'no shape')[1] if hasattr(func_data, 'shape') and func_data.shape is not None else 'unknown'} vs adata.n_obs {adata_in.n_obs}") # type: ignore
                        return None 
                        
                    # func_df is (features x samples)
                    func_df = pd.DataFrame(func_data, index=feature_names, columns=adata_in.obs_names) # type: ignore
                    logger.debug(f"Converted {level} from array to DataFrame.")
                    
                    # Transpose to (samples x features) for AnnData.X
                    data_for_anndata = func_df.T
                
                else:
                    # --- BRANCH 2: Data is a DataFrame ---
                    # Based on the error, it's already (samples x features).
                    logger.debug(f"Using existing DataFrame from .obsm['{level}'].")
                    func_df = func_data.copy()
                    
                    # Validation check: if rows don't match samples, it's probably (features x samples)
                    if func_df.shape[0] != adata_in.n_obs:
                        logger.warning(f"DataFrame in .obsm['{level}'] has {func_df.shape[0]} rows, but adata has {adata_in.n_obs} obs.")
                        logger.warning(f"Attempting to transpose, assuming (features x samples).")
                        
                        # Check if transposed shape matches
                        if func_df.shape[1] != adata_in.n_obs:
                            logger.error(f"Shape mismatch: DataFrame is {func_df.shape}, adata is {adata_in.n_obs} obs. Cannot orient.")
                            return None
                        # Use the transposed data
                        data_for_anndata = func_df.T
                    else:
                        # Shape matches (samples x features), use as-is
                        logger.debug(f"Assuming (samples x features) orientation.")
                        data_for_anndata = func_df

                # Create new AnnData object (now using data_for_anndata)
                adata_func = ad.AnnData(data_for_anndata)
                adata_func.obs = adata_in.obs.loc[adata_func.obs_names].copy()
                adata_func.var.index.name = level
                
                # IMPORTANT: Create the 'raw_counts' layer for downstream functions
                # Get the underlying values
                if isinstance(data_for_anndata, pd.DataFrame):
                    counts_values = data_for_anndata.values
                elif issparse(data_for_anndata):
                    counts_values = data_for_anndata.toarray() # type: ignore
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
                # The original KeyError will be caught here and logged
                logger.error(f"Failed to create AnnData from obsm key '{level}': {e}", exc_info=True)
                return None
        else: 
            logger.warning(f"Analysis level '{level}' not found in taxonomy or .obsm. Skipping.")
            return None
        
    @staticmethod
    def aggregate_adata_by_taxonomy(adata_in: ad.AnnData, tax_level: str = 'Genus') -> ad.AnnData | None:
        """Aggregates an AnnData object, including boolean function columns using 'any'."""
        logger.info(f"--- Aggregating AnnData by {tax_level} ---"); adata_copy = adata_in.copy()
        if tax_level not in adata_copy.var.columns: logger.error(f"Tax level '{tax_level}' not in .var."); return None
        if isinstance(adata_copy.var[tax_level].dtype, pd.CategoricalDtype):
            if 'Unassigned' not in adata_copy.var[tax_level].cat.categories: adata_copy.var[tax_level] = adata_copy.var[tax_level].cat.add_categories('Unassigned')
            adata_copy.var[tax_level] = adata_copy.var[tax_level].fillna('Unassigned')
        else: adata_copy.var[tax_level] = adata_copy.var[tax_level].astype(str).fillna('Unassigned').replace(['nan', '<NA>', 'None'], 'Unassigned')
        if 'raw_counts' in adata_copy.layers: counts_mtx = adata_copy.layers['raw_counts']; logger.debug("Using 'raw_counts'.")
        else: logger.warning("Using '.X' for aggregation."); counts_mtx = adata_copy.X
        if sparse.issparse(counts_mtx): counts_mtx = counts_mtx.tocsc()  # type: ignore[union-attr]
        elif hasattr(counts_mtx, 'toarray'): counts_mtx = counts_mtx.toarray() # type: ignore
        elif not isinstance(counts_mtx, np.ndarray): counts_mtx = np.array(counts_mtx)
        asv_to_tax_map = adata_copy.var[tax_level]
        logger.debug("Creating sparse grouper matrix for aggregation...")
        # 1. Get unique taxa and their indices
        unique_taxa, group_indices = np.unique(asv_to_tax_map, return_inverse=True)

        # 2. Create the (n_groups x n_features) grouper matrix
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

        # 3. Perform the aggregation with matrix multiplication
        #    (n_samples x n_features) @ (n_features x n_groups) = (n_samples x n_groups)
        if not isinstance(counts_mtx, csr_matrix):
            if issparse(counts_mtx):
                counts_mtx = counts_mtx.tocsr() # type: ignore
            else:
                counts_mtx = csr_matrix(counts_mtx)
                
        logger.debug("Performing sparse aggregation...")
        agg_mtx = counts_mtx @ M_grouper.T  # This is the fast, sparse multiplication

        # 4. Create the new AnnData
        new_var = pd.DataFrame(index=unique_taxa)
        new_var.index.name = tax_level
        if not isinstance(agg_mtx, csr_matrix):
            agg_mtx = csr_matrix(agg_mtx)

        # This is the key line. It passes obs and var at creation, which is robust.
        adata_new = ad.AnnData(
            agg_mtx, 
            obs=adata_copy.obs.copy(), 
            var=new_var, 
            dtype=agg_mtx.dtype
        )
        adata_new.layers['raw_counts'] = csr_matrix(adata_new.X)
        logger.debug(f"Counts aggregation resulted in {adata_new.n_vars} groups.")
        tax_levels_all = AnalysisUtils.TAX_LEVELS_ALL; levels_to_keep = tax_levels_all[:tax_levels_all.index(tax_level) + 1] if tax_level in tax_levels_all else [tax_level]
        logger.debug(f"Aggregating .var metadata up to {tax_level}...")
        try:
            var_meta_orig = adata_in.var.copy()
            if tax_level not in var_meta_orig.columns: raise KeyError(f"Column '{tax_level}' not found.")
            var_meta_orig['__group_key__'] = var_meta_orig[tax_level].fillna('Unassigned')
            agg_funcs = {}; levels_present = [lvl for lvl in levels_to_keep if lvl in var_meta_orig.columns]
            for lvl in levels_present: agg_funcs[lvl] = lambda series: series.dropna().iloc[0] if not series.dropna().empty else np.nan
            # Identify function columns by prefix (handle potential absence of prefixes)
            func_prefixes = ("FAPROTAX_", "CUSTOM_") # Add PICRUSt2 prefix if you add boolean PICRUSt2 cols later
            func_cols = [c for c in var_meta_orig.columns if c.startswith(func_prefixes) and var_meta_orig[c].dtype == bool]
            if func_cols: logger.debug(f"Aggregating {len(func_cols)} function columns using 'any'.")
            for f_col in func_cols: agg_funcs[f_col] = 'any'
            other_cols = [c for c in var_meta_orig.columns if c not in agg_funcs and c != '__group_key__']
            for o_col in other_cols: agg_funcs[o_col] = lambda series: series.dropna().iloc[0] if not series.dropna().empty else np.nan
            if not agg_funcs: raise ValueError("No columns found to aggregate in .var.")
            logger.debug(f"Aggregating .var using functions: {list(agg_funcs.keys())}")
            grouped_meta = var_meta_orig.groupby('__group_key__', observed=False).agg(agg_funcs)
            adata_new.var = grouped_meta.reindex(unique_taxa); adata_new.var.index.name = tax_level
        except Exception as e: logger.warning(f"Could not aggregate .var metadata: {e}"); adata_new.var_names = unique_taxa.astype(str).tolist(); adata_new.var = pd.DataFrame(index=adata_new.var_names); adata_new.var.index.name = tax_level
        adata_new.var_names = adata_new.var_names.astype(str).tolist(); adata_new.var.index = adata_new.var.index.astype(str)
        if 'Unassigned' in adata_new.var_names: logger.info("Filtering 'Unassigned' taxa."); adata_new = adata_new[:, adata_new.var_names != 'Unassigned'].copy()
        logger.info(f"Aggregation complete. New shape: {adata_new.shape}"); return adata_new

    @staticmethod
    def clr_transform(adata_in: ad.AnnData, pseudocount: float = 1.0, use_cache: bool = True) -> pd.DataFrame:
        """Centered log-ratio transformation with optional caching.
        
        Args:
            adata_in: AnnData object with count data
            pseudocount: Pseudocount to add before log transform
            use_cache: Whether to use/store cached results (default: True)
        
        Returns:
            DataFrame with CLR-transformed data
        """
        # Generate cache key from sample and feature names
        cache_key = f"{hash(tuple(adata_in.obs_names))}_{hash(tuple(adata_in.var_names))}_{pseudocount}"
        
        # Check cache
        if use_cache and cache_key in AnalysisUtils._clr_cache:
            logger.debug(f"Using cached CLR transform (cache_key={cache_key[:16]}...)")
            return AnalysisUtils._clr_cache[cache_key]
        
        logger.debug(f"Computing CLR transform (pseudocount={pseudocount})")
        if 'raw_counts' in adata_in.layers: counts_mtx = adata_in.layers['raw_counts']; logger.debug("Using 'raw_counts'.")
        else: counts_mtx = adata_in.X; logger.warning("Using '.X' for CLR.")
        if sparse.issparse(counts_mtx): counts_mtx = counts_mtx.toarray()  # type: ignore[union-attr]
        if 'sparse' in str(type(counts_mtx)): counts_mtx = counts_mtx.toarray()  # type: ignore[union-attr]
        counts_mtx = np.asarray(counts_mtx)
        counts_mtx_pseudo = counts_mtx + pseudocount; log_counts = np.log(counts_mtx_pseudo)
        geom_mean_log = np.mean(log_counts, axis=1, keepdims=True); clr_data = log_counts - geom_mean_log
        clr_df = pd.DataFrame(clr_data, index=adata_in.obs_names, columns=adata_in.var_names)
        
        # Store in cache
        if use_cache:
            AnalysisUtils._clr_cache[cache_key] = clr_df
            logger.debug(f"Cached CLR transform (total cached: {len(AnalysisUtils._clr_cache)})")
        
        logger.debug("CLR transform complete."); return clr_df
    
    @staticmethod
    def clear_clr_cache():
        """Clear the CLR transform cache to free memory."""
        n_cached = len(AnalysisUtils._clr_cache)
        AnalysisUtils._clr_cache.clear()
        logger.info(f"Cleared CLR cache ({n_cached} entries removed)")
    
    @staticmethod
    def filter_ml_targets(
        adata: ad.AnnData,
        target_cols: List[str],
        min_samples_per_group: int = 10,
        max_classes: int = 10
    ) -> Dict[str, List[str]]:
        """Pre-filter ML targets to avoid small class size warnings.
        
        Args:
            adata: AnnData object
            target_cols: List of potential target columns
            min_samples_per_group: Minimum samples required per class
            max_classes: Maximum number of classes for classification
        
        Returns:
            Dict with 'valid' and 'skipped' target lists, with skip reasons
        """
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

    # --- REFACTORED METHOD ---
    @staticmethod
    def find_plottable_metadata(adata: ad.AnnData, fullness_threshold: float = 0.4, max_categories: int = 50) -> Dict[str, List[str]]:
        """
        Identifies plottable numeric and categorical columns in .obs.
        
        This method is less verbose, grouping all excluded columns into a
        single debug log message at the end.
        
        FIXED:
        - Priority columns (facility_*, lat/lon) exempt from fullness threshold
        - Numeric columns checked BEFORE high cardinality exclusion
        - Boolean columns now only count True/False, exclude NaN from value counts
        """
        logger.info(f"Identifying plottable metadata (Fullness > {fullness_threshold}, Max Categories < {max_categories})...")
        categorical_cols = []
        numeric_cols = []
        # Store excluded columns and their reasons for a grouped log
        excluded_cols: Dict[str, str] = {} 
        obs_df = adata.obs
        n_obs = adata.n_obs

        if n_obs == 0: 
            logger.warning("No observations."); 
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
            
            # Count ONLY non-null unique values for booleans (exclude NaN)
            col_series = obs_df[col]
            col_dtype = col_series.dtype
            
            # For boolean columns, only count True/False
            if isinstance(col_dtype, type(pd.BooleanDtype())) or pd.api.types.is_bool_dtype(col_dtype):
                n_unique = col_series.dropna().nunique()  # Don't count NaN as a category
            else:
                n_unique = col_series.nunique()
                
            if n_unique <= 1: 
                excluded_cols[col] = '1 unique value'
                continue
                
            # --- Classification Logic (FIXED ORDER) ---
            
            # 1. Check if column is NUMERIC first (BEFORE checking cardinality)
            if pd.api.types.is_numeric_dtype(col_dtype):
                # Treat low-cardinality integers as categorical (but not floats)
                if pd.api.types.is_integer_dtype(col_dtype) and n_unique < max_categories / 2: 
                    categorical_cols.append(col)
                else: 
                    # This is numeric - add to numeric_cols (don't exclude for high cardinality!)
                    numeric_cols.append(col)
                continue  # Skip to next column
            
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
                    
                # Handle 'object' dtype, which could be mixed or numeric-as-string
                if pd.api.types.is_object_dtype(col_dtype):
                    try:
                        temp_series = col_series.dropna()
                        if temp_series.empty: 
                            excluded_cols[col] = 'all NA'
                            continue
                        
                        # OPTIMIZATION: Sample first for large columns (10x speedup)
                        sample_size = min(100, len(temp_series))
                        sample = temp_series.iloc[:sample_size]
                        numeric_sample = pd.to_numeric(sample, errors='coerce')
                        
                        # Quick check on sample
                        if numeric_sample.notna().sum() / len(sample) > 0.9:
                            # Verify with full column
                            numeric_check = pd.to_numeric(temp_series, errors='coerce')
                            if numeric_check.notna().sum() / len(temp_series) > 0.9:
                                numeric_cols.append(col)
                            else:
                                categorical_cols.append(col)
                        else:
                            categorical_cols.append(col) # Treat as categorical
                            
                    except Exception as e: 
                        # Fallback for complex object types
                        logger.warning(f"Error checking object column '{col}': {e}. Treating as categorical.")
                        categorical_cols.append(col)
                else: 
                    # String or pd.Categorical
                    categorical_cols.append(col)
            else: 
                excluded_cols[col] = f'unknown dtype ({col_dtype})'

        # --- Grouped Logging ---
        # Log the summary of what was found
        logger.info(f"Found {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns eligible for ML/FS analysis (not all will be used; see config/strict_targets).")
        
        # Log the grouped exclusions at DEBUG level
        if excluded_cols:
            # Group columns by their exclusion reason
            reason_groups: Dict[str, List[str]] = defaultdict(list)
            for col, reason in excluded_cols.items():
                reason_groups[reason].append(col)
            
            logger.debug("--- Summary of Excluded Metadata Columns ---")
            for reason, cols in sorted(reason_groups.items()):
                count = len(cols)
                # Show first 5 columns as an example to avoid huge log lines
                example_cols = ", ".join(cols[:5])
                if count > 5:
                    example_cols += "..."
                logger.debug(f"Reason: '{reason}' ({count} total): [{example_cols}]")
            logger.debug("--- End of Excluded Columns Summary ---")
        
        return {'categorical': sorted(categorical_cols), 'numeric': sorted(numeric_cols)}