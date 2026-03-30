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
        from workflow_16s.utils.pandas import parse_lat_lon
        return parse_lat_lon(lat_lon_series)

    @staticmethod
    def get_analysis_adata(adata_in: ad.AnnData, level: str) -> Union[ad.AnnData, None]:
        from workflow_16s.utils.anndata.misc import get_adata_level
        return get_adata_level(adata_in, level)
        
    @staticmethod
    def aggregate_adata_by_taxonomy(adata_in: ad.AnnData, tax_level: str = 'Genus') -> Union[ad.AnnData, None]:
        from workflow_16s.utils.anndata.misc import aggregate_adata_by_taxonomy
        return aggregate_adata_by_taxonomy(adata_in, tax_level)
    
    @staticmethod
    def clr_transform(adata: ad.AnnData, pseudocount: float = 1.0) -> pd.DataFrame:
        """Standard CLR (Legacy support)."""
        return AnalysisUtils.rclr_transform(adata)

    @staticmethod
    def rclr_transform(adata: ad.AnnData) -> Union[csr_matrix, None]:
        """
        Robust CLR (rCLR): Preserves sparsity by ignoring zeros in geometric mean
        and keeping zeros as zeros in the output. Matches Martino et al. (2019) mSystems.
        Returns sparse matrix to avoid memory explosion on large datasets.
        """
        try:
            # Check for raw counts layer first
            if 'raw_counts' in adata.layers: mat = adata.layers['raw_counts']
            else: mat = adata.X

            # Work with sparse matrices directly - critical for large datasets
            if sparse.issparse(mat): mat_sparse = mat.tocsr()
            else: mat_sparse = csr_matrix(mat)

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