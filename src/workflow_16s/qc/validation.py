"""
QC System Validation and Error Handling

This module provides input validation, dependency checking,
and graceful error handling for the QC system.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import pandas as pd
import numpy as np

logger = logging.getLogger('workflow_16s')


class QCDependencyError(Exception):
    """Raised when a required dependency is missing."""
    pass


class QCValidationError(Exception):
    """Raised when input validation fails."""
    pass


def check_dependencies(modules: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Check if required dependencies are available.
    
    Args:
        modules: List of modules to check ('cutadapt', 'biopython', 'sklearn', etc.)
                If None, checks all.
    
    Returns:
        Dict of module_name -> is_available
    """
    if modules is None:
        modules = ['cutadapt', 'biopython', 'sklearn', 'scipy', 'Bio']
    
    available = {}
    
    # Check CutAdapt (command-line tool)
    if 'cutadapt' in modules:
        try:
            result = subprocess.run(['cutadapt', '--version'], 
                                  capture_output=True, 
                                  timeout=5,
                                  text=True)
            available['cutadapt'] = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            available['cutadapt'] = False
    
    # Check Python packages
    python_packages = {
        'biopython': 'Bio',
        'sklearn': 'sklearn',
        'scipy': 'scipy',
        'Bio': 'Bio'
    }
    
    for module in modules:
        if module in python_packages:
            try:
                __import__(python_packages[module])
                available[module] = True
            except ImportError:
                available[module] = False
    
    return available


def validate_config(config: Dict) -> Tuple[bool, List[str]]:
    """
    Validate QC configuration.
    
    Args:
        config: QC configuration dictionary
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Check top-level structure
    if not isinstance(config, dict):
        errors.append("Config must be a dictionary")
        return False, errors
    
    # Validate metadata_validation config
    if 'metadata_validation' in config:
        meta_config = config['metadata_validation']
        
        if 'correlation_threshold' in meta_config:
            thresh = meta_config['correlation_threshold']
            if not (0 <= thresh <= 1):
                errors.append(
                    f"metadata_validation.correlation_threshold must be 0-1, "
                    f"got {thresh}"
                )
        
        if 'max_facility_distance_km' in meta_config:
            dist = meta_config['max_facility_distance_km']
            if not (0 < dist <= 50000):
                errors.append(
                    f"metadata_validation.max_facility_distance_km must be 0-50000, "
                    f"got {dist}"
                )
    
    # Validate contamination_detection config
    if 'contamination_detection' in config:
        contam_config = config['contamination_detection']
        
        if 'method' in contam_config:
            method = contam_config['method']
            valid_methods = ['database', 'frequency', 'ubiquity', 'combined']
            if method not in valid_methods:
                errors.append(
                    f"contamination_detection.method must be one of {valid_methods}, "
                    f"got '{method}'"
                )
        
        if 'threshold' in contam_config:
            thresh = contam_config['threshold']
            if not (0 <= thresh <= 1):
                errors.append(
                    f"contamination_detection.threshold must be 0-1, got {thresh}"
                )
    
    # Validate primer_qc config
    if 'primer_qc' in config:
        primer_config = config['primer_qc']
        
        if 'max_error_rate' in primer_config:
            rate = primer_config['max_error_rate']
            if not (0 <= rate <= 1):
                errors.append(
                    f"primer_qc.max_error_rate must be 0-1, got {rate}"
                )
        
        # Check CutAdapt availability if enabled
        if primer_config.get('enabled', False) and primer_config.get('use_cutadapt', True):
            deps = check_dependencies(['cutadapt'])
            if not deps.get('cutadapt', False):
                warnings.warn(
                    "CutAdapt not found but primer_qc is enabled. "
                    "Install with: conda install -c bioconda cutadapt"
                )
    
    return len(errors) == 0, errors


def validate_metadata(df: pd.DataFrame, 
                      required_cols: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
    """
    Validate metadata DataFrame.
    
    Args:
        df: Metadata DataFrame
        required_cols: Optional list of required column names
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Check if DataFrame
    if not isinstance(df, pd.DataFrame):
        errors.append(f"Metadata must be a DataFrame, got {type(df)}")
        return False, errors
    
    # Check not empty
    if len(df) == 0:
        errors.append("Metadata is empty (0 rows)")
        return False, errors
    
    if len(df.columns) == 0:
        errors.append("Metadata has no columns")
        return False, errors
    
    # Check for required columns
    if required_cols:
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
    
    # Check index
    if df.index.duplicated().any():
        n_dup = df.index.duplicated().sum()
        errors.append(f"Metadata has {n_dup} duplicate index values")
    
    # Check for all-NaN columns
    nan_cols = df.columns[df.isna().all()].tolist()
    if nan_cols:
        warnings.warn(f"{len(nan_cols)} columns are all NaN: {nan_cols[:5]}")
    
    return len(errors) == 0, errors


def validate_adata(adata, 
                   min_samples: int = 10,
                   min_features: int = 100) -> Tuple[bool, List[str]]:
    """
    Validate AnnData object for QC.
    
    Args:
        adata: AnnData object
        min_samples: Minimum number of samples required
        min_features: Minimum number of features required
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    try:
        import anndata as ad
    except ImportError:
        errors.append("anndata package not installed")
        return False, errors
    
    # Check type
    if not isinstance(adata, ad.AnnData):
        errors.append(f"Expected AnnData object, got {type(adata)}")
        return False, errors
    
    # Check dimensions
    if adata.n_obs < min_samples:
        errors.append(
            f"Too few samples: {adata.n_obs} < {min_samples}. "
            f"QC requires at least {min_samples} samples."
        )
    
    if adata.n_vars < min_features:
        errors.append(
            f"Too few features: {adata.n_vars} < {min_features}. "
            f"QC requires at least {min_features} features."
        )
    
    # Check for metadata
    if adata.obs is None or len(adata.obs.columns) == 0:
        warnings.warn("No metadata found in adata.obs")
    
    # Check for taxonomy
    if adata.var is None or len(adata.var.columns) == 0:
        warnings.warn("No feature metadata found in adata.var")
    
    # Check data matrix
    if adata.X is None:
        errors.append("Data matrix (adata.X) is None")
        return False, errors
    
    # Check for all-zero samples
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X
    
    zero_samples = (X.sum(axis=1) == 0).sum()
    if zero_samples > 0:
        warnings.warn(
            f"{zero_samples} samples have zero total counts. "
            f"These should be filtered before QC."
        )
    
    return len(errors) == 0, errors


def safe_numeric_conversion(series: pd.Series, 
                            column_name: str = "column") -> pd.Series:
    """
    Safely convert a Series to numeric, handling errors gracefully.
    
    Args:
        series: Pandas Series to convert
        column_name: Name of column for logging
    
    Returns:
        Converted Series (numeric if possible, original if not)
    """
    try:
        # Try converting to numeric
        converted = pd.to_numeric(series, errors='coerce')
        
        # Check how many values were converted to NaN
        n_original_nan = series.isna().sum()
        n_converted_nan = converted.isna().sum()
        n_failed = n_converted_nan - n_original_nan
        
        if n_failed > len(series) * 0.5:
            # More than 50% failed conversion, probably not numeric
            logger.debug(
                f"Column '{column_name}': {n_failed}/{len(series)} values "
                f"failed numeric conversion. Keeping as-is."
            )
            return series
        
        if n_failed > 0:
            logger.debug(
                f"Column '{column_name}': {n_failed} values converted to NaN"
            )
        
        return converted
    
    except Exception as e:
        logger.warning(f"Failed to convert '{column_name}' to numeric: {e}")
        return series


def handle_missing_dependencies(module_name: str, 
                                feature: str,
                                install_command: str) -> None:
    """
    Provide helpful error message for missing dependencies.
    
    Args:
        module_name: Name of missing module
        feature: Feature that requires the module
        install_command: Command to install the module
    """
    msg = (
        f"\n{'='*80}\n"
        f"MISSING DEPENDENCY: {module_name}\n"
        f"{'='*80}\n"
        f"The '{feature}' feature requires {module_name}.\n\n"
        f"To install:\n"
        f"  {install_command}\n\n"
        f"Or disable this feature in config.yaml:\n"
        f"  {feature}:\n"
        f"    enabled: false\n"
        f"{'='*80}\n"
    )
    logger.error(msg)
    raise QCDependencyError(msg)


def validate_primer_sequences(primers: Dict[str, str]) -> Tuple[bool, List[str]]:
    """
    Validate primer sequences.
    
    Args:
        primers: Dict of primer_name -> sequence
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    if not primers:
        errors.append("No primers provided")
        return False, errors
    
    valid_bases = set('ATCGRYMKSWHBVDN')  # IUPAC codes
    
    for name, seq in primers.items():
        if not seq:
            errors.append(f"Primer '{name}' has empty sequence")
            continue
        
        # Check sequence characters
        invalid_chars = set(seq.upper()) - valid_bases
        if invalid_chars:
            errors.append(
                f"Primer '{name}' contains invalid characters: {invalid_chars}"
            )
        
        # Check length
        if len(seq) < 10:
            warnings.warn(f"Primer '{name}' is very short ({len(seq)} bp)")
        if len(seq) > 50:
            warnings.warn(f"Primer '{name}' is very long ({len(seq)} bp)")
    
    return len(errors) == 0, errors


def create_safe_output_dir(path: Path, 
                           name: str = "output") -> Path:
    """
    Safely create output directory with error handling.
    
    Args:
        path: Path to create
        name: Name for logging
    
    Returns:
        Created Path object
    
    Raises:
        QCValidationError: If directory cannot be created
    """
    try:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Test write permission
        test_file = path / '.write_test'
        try:
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            raise QCValidationError(
                f"No write permission for {name} directory: {path}"
            )
        
        return path
    
    except Exception as e:
        raise QCValidationError(
            f"Failed to create {name} directory '{path}': {e}"
        )
