"""
Performance Optimizer for Large-Scale Microbiome Analysis

Automatically detects dataset size and recommends/applies optimized parameters
to prevent excessive runtimes on large datasets.

Author: GitHub Copilot (AI Assistant)
Date: 2026-01-08
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import anndata as ad
import numpy as np

logger = logging.getLogger('workflow_16s')


@dataclass
class DatasetSizeProfile:
    """Profile of dataset dimensions and recommended optimizations."""
    n_samples: int
    n_features: int
    size_category: str  # 'small', 'medium', 'large', 'very_large', 'massive'
    recommendations: Dict[str, Any]
    warnings: list


def classify_dataset_size(n_samples: int, n_features: int) -> str:
    """
    Classify dataset into size categories.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features (ASVs/OTUs)
    
    Returns
    -------
    str
        Size category: 'small', 'medium', 'large', 'very_large', 'massive'
    """
    if n_samples < 100:
        return 'small'
    elif n_samples < 1000:
        return 'medium'
    elif n_samples < 5000:
        return 'large'
    elif n_samples < 20000:
        return 'very_large'
    else:
        return 'massive'


def get_optimal_parameters(
    adata: ad.AnnData,
    ignore_recommendations: bool = False
) -> DatasetSizeProfile:
    """
    Analyze dataset and recommend optimal parameters.
    
    This function detects when datasets are large enough that default parameters
    would cause excessive runtimes, and recommends optimizations:
    - Subsampling for rarefaction curves
    - Reduced permutations for permutation tests
    - Stratified sampling to preserve group ratios
    
    Parameters
    ----------
    adata : ad.AnnData
        Dataset to analyze
    ignore_recommendations : bool, default=False
        If True, returns recommendations but doesn't apply them
        Set via config: performance.ignore_size_recommendations = true
    
    Returns
    -------
    DatasetSizeProfile
        Profile with size category, recommendations, and warnings
    
    Examples
    --------
    >>> profile = get_optimal_parameters(adata)
    >>> if profile.warnings:
    ...     for warning in profile.warnings:
    ...         logger.warning(warning)
    >>> if profile.recommendations:
    ...     logger.info(f"Using optimized parameters: {profile.recommendations}")
    """
    n_samples = adata.n_obs
    n_features = adata.n_vars
    
    size_category = classify_dataset_size(n_samples, n_features)
    recommendations = {}
    warnings = []
    
    # === SMALL DATASETS (< 100 samples) === #
    if size_category == 'small':
        recommendations = {
            'rarefaction_subsample': None,  # Use all samples
            'permutation_n_perms': 10000,  # Full permutations
            'permutation_subsample': None,  # Use all samples
            'batch_correction_method': 'percentile',  # Fast method
        }
        logger.info(f"Dataset size: {n_samples} samples, {n_features} features (SMALL)")
        logger.info("Using standard parameters - no optimizations needed")
    
    # === MEDIUM DATASETS (100-999 samples) === #
    elif size_category == 'medium':
        recommendations = {
            'rarefaction_subsample': None,  # Use all samples
            'permutation_n_perms': 10000,  # Full permutations
            'permutation_subsample': None,  # Use all samples
            'batch_correction_method': 'percentile',  # Fast method
        }
        logger.info(f"Dataset size: {n_samples} samples, {n_features} features (MEDIUM)")
        logger.info("Using standard parameters")
    
    # === LARGE DATASETS (1,000-4,999 samples) === #
    elif size_category == 'large':
        recommendations = {
            'rarefaction_subsample': min(1000, n_samples),
            'permutation_n_perms': 5000,  # Reduce permutations
            'permutation_subsample': min(2000, n_samples),
            'batch_correction_method': 'percentile',
        }
        warnings.append(
            f"LARGE DATASET ({n_samples} samples): Optimizations recommended. "
            f"Rarefaction will use {recommendations['rarefaction_subsample']} samples, "
            f"permutation tests will use {recommendations['permutation_subsample']} samples "
            f"and {recommendations['permutation_n_perms']} permutations. "
            f"Set 'performance.ignore_size_recommendations: true' to override."
        )
        logger.warning(warnings[-1])
    
    # === VERY LARGE DATASETS (5,000-19,999 samples) === #
    elif size_category == 'very_large':
        recommendations = {
            'rarefaction_subsample': 1000,  # Subsample for rarefaction
            'permutation_n_perms': 2000,  # Reduce permutations significantly
            'permutation_subsample': 2000,  # Subsample for permutation tests
            'batch_correction_method': 'percentile',
        }
        warnings.append(
            f"VERY LARGE DATASET ({n_samples} samples): Strong optimizations recommended! "
            f"Without subsampling, rarefaction curves could take hours and permutation "
            f"tests could take days. Rarefaction will use {recommendations['rarefaction_subsample']} "
            f"samples, permutation tests will use {recommendations['permutation_subsample']} samples "
            f"and {recommendations['permutation_n_perms']} permutations. "
            f"Set 'performance.ignore_size_recommendations: true' to override."
        )
        logger.warning(warnings[-1])
    
    # === MASSIVE DATASETS (20,000+ samples) === #
    else:  # massive
        recommendations = {
            'rarefaction_subsample': 500,  # Heavy subsampling
            'permutation_n_perms': 1000,  # Minimal permutations
            'permutation_subsample': 1000,  # Heavy subsampling
            'batch_correction_method': 'percentile',
        }
        warnings.append(
            f"MASSIVE DATASET ({n_samples} samples): CRITICAL - Aggressive optimizations required! "
            f"Without heavy subsampling, some analyses could take weeks. "
            f"Rarefaction will use {recommendations['rarefaction_subsample']} samples, "
            f"permutation tests will use {recommendations['permutation_subsample']} samples "
            f"and {recommendations['permutation_n_perms']} permutations. "
            f"Consider running analyses on a compute cluster. "
            f"Set 'performance.ignore_size_recommendations: true' to override (NOT recommended)."
        )
        logger.error(warnings[-1])
    
    # Build profile
    profile = DatasetSizeProfile(
        n_samples=n_samples,
        n_features=n_features,
        size_category=size_category,
        recommendations=recommendations if not ignore_recommendations else {},
        warnings=warnings
    )
    
    return profile


def subsample_stratified(
    adata: ad.AnnData,
    n_samples: int,
    stratify_col: Optional[str] = None,
    random_state: int = 42
) -> Tuple[ad.AnnData, np.ndarray]:
    """
    Subsample dataset while preserving group ratios.
    
    Parameters
    ----------
    adata : ad.AnnData
        Dataset to subsample
    n_samples : int
        Number of samples to select
    stratify_col : str, optional
        Column in adata.obs to stratify by (e.g., 'treatment_group')
        If None, performs simple random sampling
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    tuple
        (subsampled_adata, selected_indices)
    
    Examples
    --------
    >>> # Subsample 1000 samples, preserving treatment group ratios
    >>> adata_sub, indices = subsample_stratified(
    ...     adata, n_samples=1000, stratify_col='treatment'
    ... )
    >>> logger.info(f"Subsampled {len(indices)} from {adata.n_obs} samples")
    """
    if n_samples >= adata.n_obs:
        logger.info("Requested subsample size >= dataset size, using full dataset")
        return adata, np.arange(adata.n_obs)
    
    np.random.seed(random_state)
    
    if stratify_col is None:
        # Simple random sampling
        indices = np.random.choice(adata.n_obs, size=n_samples, replace=False)
        logger.info(f"Random subsampling: {n_samples} of {adata.n_obs} samples")
    else:
        # Stratified sampling
        if stratify_col not in adata.obs.columns:
            logger.warning(
                f"Stratification column '{stratify_col}' not found, "
                f"falling back to random sampling"
            )
            indices = np.random.choice(adata.n_obs, size=n_samples, replace=False)
        else:
            groups = adata.obs[stratify_col].values
            unique_groups = np.unique(groups)
            
            # Calculate samples per group (proportional to original)
            indices = []
            for group in unique_groups:
                group_mask = groups == group
                n_group = group_mask.sum()
                n_subsample_group = int(np.round(n_samples * n_group / adata.n_obs))
                
                # Ensure we get at least 1 sample from each group
                n_subsample_group = max(1, n_subsample_group)
                
                group_indices = np.where(group_mask)[0]
                if len(group_indices) <= n_subsample_group:
                    # Use all samples from this group
                    selected = group_indices
                else:
                    # Subsample from this group
                    selected = np.random.choice(
                        group_indices, size=n_subsample_group, replace=False
                    )
                indices.extend(selected)
            
            indices = np.array(indices)
            
            # If we got too many (due to rounding), randomly drop some
            if len(indices) > n_samples:
                indices = np.random.choice(indices, size=n_samples, replace=False)
            
            logger.info(
                f"Stratified subsampling by '{stratify_col}': "
                f"{len(indices)} of {adata.n_obs} samples, "
                f"preserving ratios across {len(unique_groups)} groups"
            )
    
    # Sort indices for reproducibility
    indices = np.sort(indices)
    
    # Create subsampled AnnData
    adata_sub = adata[indices, :].copy()
    
    return adata_sub, indices


def estimate_runtime(
    n_samples: int,
    n_features: int,
    n_permutations: int = 10000,
    operation: str = 'permutation_test'
) -> str:
    """
    Estimate runtime for an operation.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    n_permutations : int, default=10000
        Number of permutations (for permutation tests)
    operation : str, default='permutation_test'
        Type of operation: 'permutation_test', 'rarefaction', 'batch_correction'
    
    Returns
    -------
    str
        Human-readable runtime estimate
    
    Examples
    --------
    >>> runtime = estimate_runtime(10000, 5000, operation='rarefaction')
    >>> logger.info(f"Estimated runtime: {runtime}")
    """
    if operation == 'permutation_test':
        # Rough estimate: ~0.1ms per sample per feature per permutation
        operations = n_samples * n_features * n_permutations
        seconds = operations * 0.0001
    elif operation == 'rarefaction':
        # Rough estimate: ~1ms per sample (multiple rarefaction depths)
        seconds = n_samples * 0.001 * 20  # ~20 depth levels
    elif operation == 'batch_correction':
        # Rough estimate: ~0.5ms per sample per feature
        seconds = n_samples * n_features * 0.0005
    else:
        return "Unknown operation"
    
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    else:
        return f"{seconds/86400:.1f} days"
