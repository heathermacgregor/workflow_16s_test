"""
Analysis Helper Utilities for the 16S Workflow.
Contains static methods for data transformation (CLR), aggregation,
metadata parsing, and plottable column finding.
"""
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from scipy import sparse
from scipy.sparse import csc_matrix, csr_matrix, issparse

from workflow_16s.config import AppConfig
from workflow_16s.utils.logger import get_logger


class AnalysisUtils:
    """Contains static helper methods for analysis tasks."""
    
    # Columns to exclude from analysis/plotting automatically
    ADMIN_NOISE_COLUMNS = {
        'biosample_insdc_center_name', 'biosample_Insdc_first_public', 
        'biosample_investigation_type', 'country_facility', 'dataset', 
        'EnvironmentalHealth_date', 'latitude_osm', 'longitude_osm', 
        'refs', 'source_url', 'source_version', 'temporal_coverage', 
        'time', 'unit', 'variable', 'admin', 'batch', 'batch_original', 
        'sample_id', '#sampleid'
    }
    
    ADMIN_NOISE_PATTERNS = [re.compile(r'.*_insdc_status$')]
    
    # Priority columns that should always be included regardless of fullness threshold
    PRIORITY_COLUMNS = {
        'facility_match', 'facility_distance_km', 'facility_name', 
        'facility_type', 'latitude', 'longitude', 'lat', 'lon'
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
        logger = get_logger("workflow_16s")
        
        # ASV/OTU
        if level.upper() in ['ASV', 'FEATURE', 'OTU']:
            logger.info("Getting analysis AnnData for native ASV level")
            adata_asv = adata_in.copy()
            if 'raw_counts' not in adata_asv.layers:
                adata_asv.layers['raw_counts'] = adata_asv.X.copy()
                
            adata_asv.var.index.name = 'ASV'
            return adata_asv

        # First check if level is a known taxonomic rank, then check .obsm for functional data
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
                data_for_anndata: Union[pd.DataFrame, csc_matrix, csr_matrix, np.ndarray]
                
                # Assume data is an array if not a DataFrame, and try to convert it
                if not isinstance(func_data, pd.DataFrame):
                    logger.debug(f"Converting array data from .obsm['{level}']")
                    
                    if f"{level}_ids" in adata_in.uns: 
                        feature_names = adata_in.uns[f"{level}_ids"]
                    elif hasattr(func_data, 'shape') and func_data.shape is not None and func_data.shape[0] > 0: 
                        # If shape[1] matches n_obs, likely (features x samples)
                        if func_data.shape[1] == adata_in.n_obs:
                            feature_names = [f"{level}_{i}" for i in range(func_data.shape[0])]
                        else:
                            feature_names = [f"{level}_{i}" for i in range(func_data.shape[1])]
                    else: 
                        logger.error(f"Cannot determine feature names for {level}")
                        return None
                    
                    # Determine orientation and transpose if necessary, ensuring we end up with (samples x features)
                    if hasattr(func_data, 'shape') and func_data.shape is not None:
                        if func_data.shape[1] == adata_in.n_obs:
                            # Convert sparse matrix to dense array if needed
                            if issparse(func_data):
                                func_data_dense = func_data.toarray() # type: ignore
                            else:
                                func_data_dense = np.asarray(func_data)
                            func_df = pd.DataFrame(func_data_dense, index=feature_names, columns=adata_in.obs_names)
                            data_for_anndata = func_df.T
                        elif func_data.shape[0] == adata_in.n_obs:
                            # Convert scipy sparse arrays to matrices for compatibility
                            if hasattr(func_data, 'toarray'):
                                data_for_anndata = func_data if issparse(func_data) else np.asarray(func_data) # type: ignore
                            else:
                                data_for_anndata = np.asarray(func_data)
                        else:
                            logger.error(f"Shape mismatch for {level}: {func_data.shape} vs adata.n_obs {adata_in.n_obs}")
                            return None
                    else:
                        return None
                
                else:
                    logger.debug(f"Using existing DataFrame from .obsm['{level}'].")
                    func_df = func_data.copy()
                    
                    # Check orientation: if rows don't match n_obs, try transposing
                    if func_df.shape[0] != adata_in.n_obs:
                        logger.warning(
                            f"DataFrame in .obsm['{level}'] has {func_df.shape[0]} rows, but adata has {adata_in.n_obs} obs.\n"
                            f"Attempting to transpose, assuming (features x samples)."
                        )
                        if func_df.shape[1] != adata_in.n_obs:
                            logger.error(f"Shape mismatch: DataFrame is {func_df.shape}, adata is {adata_in.n_obs} obs. Cannot orient.")
                            return None
                        data_for_anndata = func_df.T
                    else:
                        data_for_anndata = func_df

                # Create new AnnData object 
                adata_func = ad.AnnData(data_for_anndata)
                adata_func.obs = adata_in.obs.loc[adata_func.obs_names].copy()
                adata_func.var.index.name = level
                
                # Create the 'raw_counts' layer for downstream functions
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
                logger.error(f"Failed to create AnnData from obsm key '{level}': {e}", exc_info=True)
                return None
        else: 
            logger.warning(f"Analysis level '{level}' not found in taxonomy or .obsm. Skipping.")
            return None
        
    @staticmethod
    def aggregate_adata_by_taxonomy(adata_in: ad.AnnData, tax_level: str = 'Genus') -> Union[ad.AnnData, None]:
        """
        Aggregates an AnnData object using efficient sparse matrix multiplication.
        Includes critical fixes for duplicates, whitespaces, and unassigned taxa.
        """
        logger = get_logger("workflow_16s")
        logger.info(f"--- Aggregating AnnData by {tax_level} ---")
        
        if not adata_in.obs_names.is_unique:
            logger.warning("⚠️ Duplicate sample IDs found in input! Making unique (appending -1, -2)...")
            adata_in.obs_names_make_unique()

        adata_copy = adata_in.copy()
        
        if tax_level not in adata_copy.var.columns: 
            logger.error(f"Tax level '{tax_level}' not in .var.")
            return None
            
        tax_series = adata_copy.var[tax_level].astype(str).str.strip()
        
        # Unify all forms of "empty"
        tax_series = tax_series.replace(
            ['nan', 'NaN', 'None', '', '<NA>', 'NoneType'], 
            'Unassigned'
        )
        tax_series = tax_series.fillna('Unassigned')
        
        # Update the dataframe used for mapping
        adata_copy.var[tax_level] = tax_series

        # Get Counts Matrix
        if 'raw_counts' in adata_copy.layers: 
            counts_mtx = adata_copy.layers['raw_counts']
        else: 
            counts_mtx = adata_copy.X
            
        if sparse.issparse(counts_mtx):
            counts_mtx = counts_mtx.tocsc() # type: ignore
        else:
            # If not sparse, ensure it's a numpy array
            if hasattr(counts_mtx, 'toarray'):
                counts_mtx = counts_mtx.toarray() # type: ignore
            elif not isinstance(counts_mtx, np.ndarray):
                counts_mtx = np.array(counts_mtx)
            
        asv_to_tax_map = adata_copy.var[tax_level]
        
        # Get unique taxa and their indices
        unique_taxa, group_indices = np.unique(asv_to_tax_map, return_inverse=True)

        # Create sparse grouper matrix
        n_features = adata_copy.n_vars
        n_groups = len(unique_taxa)
        
        if counts_mtx is not None and hasattr(counts_mtx, 'dtype'):
            grouper_dtype = np.dtype(counts_mtx.dtype)
        else:
            grouper_dtype = np.float64
            
        M_grouper = csc_matrix(
            (np.ones(n_features, dtype=grouper_dtype), (group_indices, np.arange(n_features))), 
            shape=(n_groups, n_features)
        )

        # Perform the aggregation
        if not isinstance(counts_mtx, csr_matrix):
            if issparse(counts_mtx):
                counts_mtx = counts_mtx.tocsr() # type: ignore
            else:
                counts_mtx = csr_matrix(counts_mtx)
                
        agg_mtx = counts_mtx @ M_grouper.T 

        # Create the new AnnData
        new_var = pd.DataFrame(index=unique_taxa)
        
        if not isinstance(agg_mtx, csr_matrix):
            agg_mtx = csr_matrix(agg_mtx)

        adata_new = ad.AnnData(
            agg_mtx, 
            obs=adata_copy.obs.copy(), 
            var=new_var, 
            dtype=agg_mtx.dtype
        )
        
        # Explicitly set the index to the taxonomy strings
        adata_new.var_names = unique_taxa.astype(str).tolist()
        adata_new.var.index.name = tax_level
        
        # Save the name as a column so downstream tools can find it
        adata_new.var[tax_level] = adata_new.var.index.values
        adata_new.layers['raw_counts'] = csr_matrix(adata_new.X)
        
        # Filter 'Unassigned' 
        if 'Unassigned' in adata_new.var_names:
            if len(adata_new.var_names) > 1:
                logger.info("Filtering 'Unassigned' taxa.")
                adata_new = adata_new[:, adata_new.var_names != 'Unassigned'].copy()
            else:
                logger.warning("⚠️ All features mapped to 'Unassigned'! Keeping it.")
        
        # Ensure obs indices are string
        adata_new.obs_names = adata_new.obs_names.astype(str).tolist()
        
        logger.info(f"Aggregation complete. New shape: {adata_new.shape}")
        return adata_new
    
    @staticmethod
    def clr_transform(adata: ad.AnnData, pseudocount: float = 1.0) -> pd.DataFrame:
        """Standard CLR (Legacy support)."""
        return AnalysisUtils.rclr_transform(adata)

    @staticmethod
    def rclr_transform(adata: ad.AnnData) -> pd.DataFrame:
        """
        Robust CLR (rCLR): Preserves sparsity by ignoring zeros in geometric mean
        and keeping zeros as zeros in the output. Matches Martino et al. (2019) mSystems.
        """
        try:
            # Access the matrix
            mat = adata.layers.get('raw_counts', adata.X)

            # Make matrix sparse
            if not issparse(mat): mat_sparse = csr_matrix(mat)
            else: mat_sparse = mat.tocsr()

            # Perform calculations ONLY on non-zero data
            log_data = np.log(mat_sparse.data)
            log_sparse = mat_sparse.copy()
            log_sparse.data = log_data

            # Mean(Log) per row (sample) - sum of logs / number of non-zero elements
            sum_log = np.array(log_sparse.sum(axis=1)).flatten()
            count_nz = np.diff(mat_sparse.indptr)

            # Avoid division by zero for empty samples
            count_nz[count_nz == 0] = 1.0
            log_gmeans = sum_log / count_nz
            
            # rCLR = log(x_i) - log(gmean_nz)
            # Subtract the mean of sample i from all non-zero elements of sample i
            for i in range(mat_sparse.shape[0]):
                start, end = log_sparse.indptr[i], log_sparse.indptr[i+1]
                log_sparse.data[start:end] -= log_gmeans[i]
            
            # Return sparse matrix instead of dense to avoid massive memory allocation
            # (463K samples × 98K features = 160GB+ if densified)
            return log_sparse
        except Exception as e:
            get_logger("workflow_16s").error(f"rCLR transform failed: {e}")
            return None

    @staticmethod
    def clr_transform_from_df(df: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
        """
        Robust CLR for DataFrames.
        Arguments 'pseudocount' is ignored for rCLR but kept for API compatibility.
        """
        mat = df.values.astype(np.float64)
        mask = mat > 0
        
        log_mat = np.zeros_like(mat)
        np.log(mat, where=mask, out=log_mat)
        
        sum_log = np.sum(log_mat, axis=1)
        count_nz = np.sum(mask, axis=1)
        count_nz[count_nz == 0] = 1.0
        
        log_gmeans = sum_log / count_nz
        
        result = log_mat - log_gmeans[:, None]
        result[~mask] = 0.0
        
        return pd.DataFrame(result, index=df.index, columns=df.columns)
    
    @staticmethod
    def clr_transform_vanilla(
        adata_in: ad.AnnData, 
        pseudocount: float = 1.0, 
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Centered log-ratio transformation with optional caching."""
        logger = get_logger("workflow_16s")
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
            counts_mtx = counts_mtx.toarray() # type: ignore
        elif 'sparse' in str(type(counts_mtx)): 
            counts_mtx = counts_mtx.toarray() # type: ignore
            
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
    def clr_transform_from_df_vanilla(
        df: pd.DataFrame, 
        pseudocount: float = 1.0
    ) -> pd.DataFrame:
        """
        Applies Centered Log-Ratio (CLR) transformation directly to a feature DataFrame.
        Useful after manual filtering or subsetting operations.
        """
        logger = get_logger("workflow_16s")
        if df is None or df.empty:
            logger.warning("Empty DataFrame passed to CLR transform.")
            return df

        # Ensure numeric types
        df_numeric = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Apply pseudocount to handle zeros in compositional data
        df_pseudo = df_numeric + pseudocount

        # Perform CLR: log(x) - mean(log(x))
        log_data = np.log(df_pseudo)
        # Compute geometric mean across features (axis=1) for each sample
        gm_log = log_data.mean(axis=1)
        
        # Subtract mean from each row
        clr_data = log_data - gm_log[:, np.newaxis]
        clr_df = pd.DataFrame(clr_data, index=df.index, columns=df.columns)

        logger.debug(f"Applied direct CLR transform to DataFrame: {clr_df.shape}")
        return clr_df
    
    @staticmethod
    def clear_clr_cache():
        """Clear the CLR transform cache to free memory."""
        logger = get_logger("workflow_16s")
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
        logger = get_logger("workflow_16s")
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
    def find_plottable_metadata(
        adata: ad.AnnData, 
        fullness_threshold: float = 0.4, 
        max_categories: int = 50,
        admin_noise_columns: Optional[List[str]] = None,  
        **kwargs,
    ) -> Dict[str, List[str]]:
        """
        Identifies plottable numeric and categorical columns in .obs.
        Consolidates exclusion logging to avoid clutter.
        """
        logger = get_logger("workflow_16s")
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
            if col in AnalysisUtils.ADMIN_NOISE_COLUMNS or (admin_noise_columns and col in admin_noise_columns): 
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
            
            # Check if column is NUMERIC first 
            if pd.api.types.is_numeric_dtype(col_dtype):
                # Treat low-cardinality integers as categorical (but not floats)
                if pd.api.types.is_integer_dtype(col_dtype) and n_unique < max_categories / 2: 
                    categorical_cols.append(col)
                else: 
                    numeric_cols.append(col)
                continue
            
            # Boolean types
            if isinstance(col_dtype, type(pd.BooleanDtype())) or pd.api.types.is_bool_dtype(col_dtype):
                categorical_cols.append(col)
                continue
                    
            # String / Object / Categorical
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
