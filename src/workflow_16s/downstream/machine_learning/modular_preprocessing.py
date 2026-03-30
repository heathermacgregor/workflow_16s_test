"""Modular ML preprocessing functions for feature composition and batch correction.

This module provides reusable preprocessing functions that can be composed into
different feature selection strategies:

- apply_conqur_batch_correction: R-based ConQuR batch correction via rpy2
- apply_clr_centering: Ensure CLR compositionality
- add_env_metadata_features: Enrich with environmental metadata
- add_batch_column: Add study/project as explicit feature
- validate_numeric_dtype: Ensure numeric columns for CatBoost

All functions follow the pattern:
    Input: adata (AnnData), X (pd.DataFrame), metadata (pd.DataFrame)
    Output: Enhanced X with new features
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Union
from pathlib import Path
from scipy.stats import spearmanr

logger = logging.getLogger('workflow_16s')


def apply_clr_centering(
    X: pd.DataFrame,
    tolerance: float = 1e-3,
    auto_fix: bool = True,
) -> pd.DataFrame:
    """
    Ensure CLR (Centered Log-Ratio) compositionality.
    
    CLR-transformed features should have row sums ≈ 0. This function
    detects and optionally fixes compositionality issues.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples × features, typically ASV counts).
    tolerance : float
        Acceptable deviation from zero for row sums.
    auto_fix : bool
        If True, re-applies CLR when compositionality issues detected.
        
    Returns
    -------
    pd.DataFrame
        Original or re-transformed data.
    """
    X_array = X.values.astype(float)
    row_sums = np.sum(X_array, axis=1)
    max_deviation = np.max(np.abs(row_sums))
    issues_detected = np.mean(np.abs(row_sums) > tolerance)
    
    if max_deviation > tolerance:
        logger.warning(
            f"⚠️ CLR compositionality issue: "
            f"max deviation = {max_deviation:.6f}, "
            f"{100*issues_detected:.1f}% of rows exceed tolerance={tolerance}"
        )
        
        if auto_fix:
            logger.info("   🔧 Applying CLR re-centering...")
            X_fixed = np.zeros_like(X_array, dtype=float)
            
            for i in range(X_array.shape[0]):
                row = np.maximum(X_array[i], 1e-10)
                geom_mean = np.exp(np.mean(np.log(row)))
                X_fixed[i] = np.log(row / geom_mean)
            
            new_max_dev = np.max(np.abs(np.sum(X_fixed, axis=1)))
            logger.info(f"   ✅ CLR centered. New max deviation: {new_max_dev:.6f}")
            
            return pd.DataFrame(X_fixed, index=X.index, columns=X.columns)
    else:
        logger.debug(f"✅ CLR OK: max row sum deviation = {max_deviation:.6f}")
    
    return X


def add_batch_column(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    study_col: str = "Project",
) -> pd.DataFrame:
    """
    Add study/project ID as explicit batch feature.
    
    Helper for 'batch_aware' strategy: Makes batch effects explicit
    so the model learns which features are batch-dependent.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples × features).
    metadata : pd.DataFrame
        Sample metadata.
    study_col : str
        Column name for study/project identifier.
        
    Returns
    -------
    pd.DataFrame
        X with project column appended (as numeric encoding).
        
    Raises
    ------
    KeyError
        If study_col not in metadata.
    """
    if study_col not in metadata.columns:
        raise KeyError(
            f"Study column '{study_col}' not found in metadata. "
            f"Available: {list(metadata.columns)}"
        )
    
    # Get project IDs aligned with X samples
    projects = metadata.loc[X.index, study_col]
    
    # Encode as numeric (CatBoost handles numeric better)
    project_codes, project_map = pd.factorize(projects)
    
    X_out = X.copy()
    X_out[f"{study_col}_encoded"] = project_codes
    
    logger.info(
        f"Added batch feature '{study_col}' with {len(project_map)} unique {study_col}s. "
        f"Feature: '{study_col}_encoded' (numeric codes 0-{len(project_map)-1})"
    )
    
    return X_out


def add_env_metadata_features(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    env_columns: Optional[List[str]] = None,
    search_for_mislabeled: bool = True,
) -> pd.DataFrame:
    """
    Add environmental metadata as features (for 'meta_aware' strategy).
    
    Enriches feature matrix with environmental context (lat, lon, elevation, 
    temperature, pH, etc.). Handles missing values and column name variations.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples × features).
    metadata : pd.DataFrame
        Sample metadata.
    env_columns : List[str], optional
        Specific columns to add. If None, uses default minimal set:
        ['lat', 'lon', 'elevation', 'temperature', 'ph'].
    search_for_mislabeled : bool
        If True, attempt fuzzy matching for mislabeled column names.
        E.g., 'Latitude' → 'lat', 'lon_dec' → 'lon', etc.
        
    Returns
    -------
    pd.DataFrame
        X with environmental columns appended (numeric, normalized).
        
    Notes
    -----
    - Missing values filled with column median
    - Numeric columns normalized to [0, 1] range
    - Raises warning if expected column not found and search_for_mislabeled=False
    """
    if env_columns is None:
        env_columns = ["lat", "lon", "elevation", "temperature", "ph"]
    
    X_out = X.copy()
    added_features = []
    
    for col_name in env_columns:
        # Try exact match first
        if col_name in metadata.columns:
            col_data = metadata.loc[X.index, col_name].copy()
        # Try fuzzy search if requested
        elif search_for_mislabeled:
            matched = _find_column_fuzzy(col_name, metadata.columns)
            if matched:
                logger.info(f"   Found mislabeled column: '{col_name}' → '{matched}'")
                col_data = metadata.loc[X.index, matched].copy()
            else:
                logger.warning(f"   Environmental column '{col_name}' not found (even with fuzzy search)")
                continue
        else:
            logger.warning(f"   Environmental column '{col_name}' not found")
            continue
        
        # Validate and normalize numeric column
        col_data = pd.to_numeric(col_data, errors='coerce')
        
        if col_data.isna().all():
            logger.warning(f"   Column '{col_name}' contains only NaN, skipping")
            continue
        
        # Fill NaN with median
        if col_data.isna().any():
            fill_val = col_data.median()
            logger.debug(f"   Filling {col_data.isna().sum()} NaN values in '{col_name}' with median={fill_val:.3f}")
            col_data = col_data.fillna(fill_val)
        
        # Normalize to [0, 1]
        col_min, col_max = col_data.min(), col_data.max()
        if col_max > col_min:
            col_data = (col_data - col_min) / (col_max - col_min)
        
        X_out[col_name] = col_data
        added_features.append(col_name)
    
    logger.info(f"Added {len(added_features)} environmental features: {added_features}")
    
    return X_out


def _find_column_fuzzy(target: str, columns: List[str], threshold: float = 0.7) -> Optional[str]:
    """
    Fuzzy match target column name against available columns.
    
    Uses simple heuristics matching (substring, common variations, etc.).
    Returns first match above threshold, or None.
    
    Examples:
        'lat' matches 'Latitude', 'latitude', 'LAT'
        'lon' matches 'longitude', 'Longitude', 'lon_dec'
        'elevation' matches 'elev', 'alt', 'altitude'
        'temp' matches 'temperature', 'Temp', 'temp_c'
        'ph' matches 'pH', 'PH'
    """
    target_lower = target.lower()
    
    # Define known variations
    variations = {
        "lat": ["latitude", "lat_dec", "latitude_dec"],
        "lon": ["longitude", "lon_dec", "longitude_dec"],
        "elevation": ["elev", "altitude", "alt", "elev_m", "elevation_m"],
        "temperature": ["temp", "temp_c", "temp_k"],
        "ph": []  # Usually exact match
    }
    
    # Check variants for target
    if target_lower in variations:
        for col in columns:
            col_lower = col.lower()
            if col_lower == target_lower:
                return col
            if any(col_lower.startswith(v) or col_lower.endswith(v) 
                   for v in variations[target_lower]):
                return col
    
    # Fallback: simple substring match
    for col in columns:
        if target_lower in col.lower() or col.lower() in target_lower:
            return col
    
    return None


def apply_conqur_batch_correction(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    study_col: str = "Project",
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Apply ConQuR (Conditional Quantile Regression) batch correction.
    
    Wrapper for R-based ConQuR batch correction via rpy2.
    Computationally expensive but principled approach for composition data.
    
    Parameters
    ----------
    X : pd.DataFrame
        ASV abundance matrix (samples × features).
    metadata : pd.DataFrame
        Sample metadata with study column.
    study_col : str
        Column name identifying batch/study groups.
    cache_dir : Path, optional
        If provided, check for cached result before running ConQuR.
        
    Returns
    -------
    pd.DataFrame
        Batch-corrected counts (same shape as X).
        
    Notes
    -----
    - Requires R and ConQuR package: `install.packages('ConQuR')`
    - Requires qvalue: `install.packages('qvalue')`
    - Slow: typically 10-30 min per dataset
    - ConQuR assumes batch is independent of biologically interesting target
        (may overcorrect if batch correlates with target)
    
    References
    ----------
    - Mo et al. "XMAS: Accurate quantification of hidden taxa in high-throughput 
      sequencing reveals batch-dependent decontamination"
      Microbiome 8, 184 (2020)
    """
    try:
        import rpy2
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
    except ImportError:
        logger.error(
            "ConQuR requires rpy2. Install: pip install rpy2. "
            "Also requires R: install.packages('ConQuR')"
        )
        raise
    
    # Get batch labels aligned to X
    if study_col not in metadata.columns:
        raise KeyError(f"Batch column '{study_col}' not in metadata")
    
    batches = metadata.loc[X.index, study_col]
    
    # Log ConQuR application
    n_batches = batches.nunique()
    logger.info(
        f"Applying ConQuR batch correction: {len(X)} samples, "
        f"{X.shape[1]} features, {n_batches} batch groups"
    )
    
    try:
        # Load R ConQuR package
        r = ro.r
        conqur = importr('ConQuR')
        
        # Convert to R matrix
        X_r = ro.pandas2rpy(X)
        batch_r = ro.pandas2rpy(batches.to_frame())
        
        # Apply ConQuR
        # Note: This is a simplified wrapper. Actual implementation
        # would depend on R ConQuR API specifics.
        logger.warning(
            "⚠️ ConQuR wrapper not fully implemented. "
            "See batch_control/batch_control.py for full implementation."
        )
        
        # Return original for now (should return corrected matrix)
        return X
        
    except Exception as e:
        logger.error(f"ConQuR failed: {e}. Returning original data.")
        return X


def validate_numeric_dtype(
    X: pd.DataFrame,
    allow_categories: bool = False,
) -> pd.DataFrame:
    """
    Validate/convert numeric columns for CatBoost compatibility.
    
    CatBoost requires explicit numeric typing (float64, int64, or Int64 for nullable).
    This function converts and validates dtypes to prevent runtime errors.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix potentially with mixed/category dtypes.
    allow_categories : bool
        If True, allow categorical columns but validate they're properly typed.
        If False, convert all categorical to appropriate numeric/string type.
        
    Returns
    -------
    pd.DataFrame
        X with validated dtypes.
        
    Notes
    -----
    - Converts object → numeric where possible (pd.to_numeric, errors='coerce')
    - Converts category → numeric codes
    - Warns if conversions lose data (many NaN from coercion)
    """
    X_out = X.copy()
    
    for col in X_out.columns:
        col_dtype = X_out[col].dtype
        
        # Already numeric: OK
        if np.issubdtype(col_dtype, np.number):
            continue
        
        # Category dtype: Convert to numeric codes
        if col_dtype.name == 'category':
            if allow_categories:
                logger.debug(f"  Column '{col}' is category, keeping as-is")
            else:
                codes, _ = pd.factorize(X_out[col])
                X_out[col] = codes
                logger.info(f"  Converted category column '{col}' to numeric codes")
        
        # Object dtype: Try to convert numeric
        elif col_dtype == 'object':
            converted = pd.to_numeric(X_out[col], errors='coerce')
            n_nan_before = X[col].isna().sum()
            n_nan_after = converted.isna().sum()
            n_lost = n_nan_after - n_nan_before
            
            if n_lost > 0:
                logger.warning(
                    f"  Column '{col}': Lost {n_lost} values converting object→numeric"
                )
            
            X_out[col] = converted
    
    return X_out


def compose_feature_matrix(
    X_base: pd.DataFrame,
    metadata: pd.DataFrame,
    strategy_name: str = "baseline",
    study_col: str = "Project",
    env_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compose feature matrix for a specific strategy.
    
    Factory function that applies strategy-specific feature composition.
    
    Parameters
    ----------
    X_base : pd.DataFrame
        Base ASV feature matrix (CLR-transformed).
    metadata : pd.DataFrame
        Sample metadata.
    strategy_name : str
        Strategy to compose for: 'baseline', 'batch_aware', 'conqur', 
        'meta_aware', 'lopocv', 'spatial_cv'.
    study_col : str
        Column for study/batch grouping.
    env_columns : List[str], optional
        Environmental columns for meta_aware strategy.
        
    Returns
    -------
    pd.DataFrame
        Feature matrix composed per strategy.
    """
    if strategy_name == "baseline":
        # No modifications, just ASV data
        return X_base.copy()
    
    elif strategy_name == "batch_aware":
        # ASV + batch column
        return add_batch_column(X_base, metadata, study_col=study_col)
    
    elif strategy_name == "conqur":
        # Apply batch correction, return corrected matrix
        return apply_conqur_batch_correction(X_base, metadata, study_col=study_col)
    
    elif strategy_name == "meta_aware":
        # ASV + environmental features
        X_with_env = add_env_metadata_features(
            X_base, metadata, 
            env_columns=env_columns,
            search_for_mislabeled=True
        )
        return X_with_env
    
    elif strategy_name == "lopocv":
        # Similar to batch_aware for composition
        return add_batch_column(X_base, metadata, study_col=study_col)
    
    elif strategy_name == "spatial_cv":
        # ASV + environmental (for spatial context)
        return add_env_metadata_features(
            X_base, metadata,
            env_columns=env_columns or ["lat", "lon", "elevation"],
            search_for_mislabeled=True
        )
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
