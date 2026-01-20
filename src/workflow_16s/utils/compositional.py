"""
Compositional Data Analysis Utilities
======================================

Proper handling of compositional (relative abundance) microbiome data.
Addresses the key issue that microbiome data are compositional (sum to 1)
and require special statistical treatment.

References:
    - Gloor et al. (2017) Microbiome Datasets Are Compositional
    - Aitchison (1986) The Statistical Analysis of Compositional Data
"""

# ===================================== IMPORTS ====================================== #

import logging
from typing import Union, Optional

import numpy as np
import pandas as pd
from biom.table import Table
from scipy.sparse import issparse
from skbio.stats.composition import multiplicative_replacement, closure

from workflow_16s import constants
from workflow_16s.utils.data import to_biom

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ================================ ZERO REPLACEMENT ================================== #

def handle_zeros_multiplicative(
    table: Union[Table, pd.DataFrame, np.ndarray],
    method: str = 'multiplicative'
) -> np.ndarray:
    """
    Replace zeros in compositional data using multiplicative replacement.
    
    This is the recommended approach for handling structural zeros before
    CLR transformation, as it preserves the compositional nature of the data.
    
    Args:
        table: Input table (BIOM Table, DataFrame, or numpy array)
        method: Replacement method ('multiplicative' or 'bayesian')
               - multiplicative: Martin-Fernández et al. (2003)
               - bayesian: Martín-Fernández et al. (2015) - more sophisticated
    
    Returns:
        Dense numpy array with zeros replaced, samples x features
        
    References:
        Martin-Fernández et al. (2003). Dealing with zeros and missing values in
        compositional data sets using nonparametric imputation. Mathematical Geology.
    """
    # Convert to BIOM if needed
    if isinstance(table, (pd.DataFrame, np.ndarray)):
        table = to_biom(table)
    
    # Get dense data (samples x features)
    if issparse(table.matrix_data):
        data = table.matrix_data.toarray().T  # Transpose to samples x features
    else:
        data = table.matrix_data.T
    
    # Ensure non-negative
    if np.any(data < 0):
        logger.warning("Negative values detected - setting to zero before replacement")
        data = np.maximum(data, 0)
    
    # Apply closure (normalize to sum to 1 per sample)
    data_closed = closure(data)
    
    # Replace zeros
    if method == 'multiplicative':
        data_imputed = multiplicative_replacement(data_closed)
    elif method == 'bayesian':
        # Bayesian-multiplicative replacement (requires additional dependency)
        try:
            from skbio.stats.composition import multiplicative_replacement
            data_imputed = multiplicative_replacement(data_closed, delta=0.65)
        except ImportError:
            logger.warning("Bayesian method not available, using multiplicative")
            data_imputed = multiplicative_replacement(data_closed)
    else:
        raise ValueError(f"Unknown zero replacement method: {method}")
    
    logger.debug(f"Replaced {np.sum(data_closed == 0)} zeros using {method} method")
    
    return data_imputed


def add_pseudocount(
    table: Union[Table, pd.DataFrame, np.ndarray],
    pseudocount: float = constants.DEFAULT_PSEUDOCOUNT
) -> np.ndarray:
    """
    Simple pseudocount addition (NOT RECOMMENDED for compositional data).
    
    This is the old approach - only use for backwards compatibility.
    Multiplicative replacement is statistically superior.
    
    Args:
        table: Input table
        pseudocount: Value to add to all entries
    
    Returns:
        Data with pseudocount added
    """
    logger.warning(
        "Using pseudocount addition is not recommended for compositional data. "
        "Consider using multiplicative_replacement instead."
    )
    
    if isinstance(table, Table):
        if issparse(table.matrix_data):
            data = table.matrix_data.toarray().T
        else:
            data = table.matrix_data.T
    elif isinstance(table, pd.DataFrame):
        data = table.values
    else:
        data = table
    
    return data + pseudocount


# ============================= CLR TRANSFORMATION ================================== #

def clr_transform(
    data: np.ndarray,
    handle_zeros: bool = True,
    zero_method: str = 'multiplicative'
) -> np.ndarray:
    """
    Apply centered log-ratio (CLR) transformation with proper zero handling.
    
    CLR is the recommended transformation for compositional microbiome data
    as it accounts for the compositional closure problem.
    
    Args:
        data: Input array (samples x features)
        handle_zeros: Whether to replace zeros before transformation
        zero_method: Method for zero replacement ('multiplicative' or 'bayesian')
    
    Returns:
        CLR-transformed data (samples x features)
        
    Formula:
        CLR(x_i) = log(x_i / g(x))
        where g(x) = geometric mean of x
        
    References:
        Aitchison (1986). The Statistical Analysis of Compositional Data.
        Gloor et al. (2017). Microbiome Datasets Are Compositional.
    """
    # Handle zeros if requested
    if handle_zeros:
        # Check if zeros exist
        if np.any(data == 0):
            logger.info(f"Detected zeros - applying {zero_method} replacement")
            data_processed = handle_zeros_multiplicative(data, method=zero_method)
        else:
            data_processed = data
    else:
        # Add small pseudocount to avoid log(0)
        data_processed = data + constants.DEFAULT_PSEUDOCOUNT
    
    # Calculate geometric mean for each sample (row)
    # Use log-space for numerical stability
    log_data = np.log(data_processed)
    geometric_mean_log = log_data.mean(axis=1, keepdims=True)
    
    # CLR = log(x) - log(geom_mean(x))
    clr_data = log_data - geometric_mean_log
    
    return clr_data


def clr_table(
    table: Union[Table, pd.DataFrame],
    handle_zeros: bool = True,
    zero_method: str = 'multiplicative',
    pseudocount: Optional[float] = None
) -> Table:
    """
    Apply CLR transformation to a BIOM table with proper zero handling.
    
    This is the main function to use for CLR transformation in the pipeline.
    
    Args:
        table: Input BIOM table or DataFrame
        handle_zeros: Use multiplicative replacement (recommended: True)
        zero_method: Zero replacement method
        pseudocount: If provided, overrides handle_zeros and uses old approach
    
    Returns:
        CLR-transformed BIOM Table
        
    Example:
        >>> from workflow_16s.utils.compositional import clr_table
        >>> table_clr = clr_table(table_raw, handle_zeros=True)
    """
    # Convert to BIOM if needed
    biom_table = to_biom(table)
    
    # Get dense data (samples x features)
    if issparse(biom_table.matrix_data):
        data = biom_table.matrix_data.toarray().T
    else:
        data = biom_table.matrix_data.T
    
    # Legacy pseudocount mode (not recommended)
    if pseudocount is not None:
        logger.warning(
            "Using legacy pseudocount mode. "
            "Consider handle_zeros=True for better statistical properties."
        )
        data_transformed = clr_transform(
            data + pseudocount, 
            handle_zeros=False
        )
    else:
        # Modern approach with multiplicative replacement
        data_transformed = clr_transform(
            data,
            handle_zeros=handle_zeros,
            zero_method=zero_method
        )
    
    # Transpose back to features x samples
    data_transformed = data_transformed.T
    
    # Create new BIOM table preserving metadata
    return Table(
        data=data_transformed,
        observation_ids=biom_table.ids(axis='observation'),
        sample_ids=biom_table.ids(axis='sample'),
        observation_metadata=biom_table.metadata(axis='observation'),
        sample_metadata=biom_table.metadata(axis='sample')
    )


# ============================== ILR TRANSFORMATION ================================= #

def ilr_transform(
    data: np.ndarray,
    basis: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply isometric log-ratio (ILR) transformation.
    
    ILR is an alternative to CLR that produces orthonormal coordinates.
    Useful for multivariate analysis and machine learning.
    
    Args:
        data: Input array (samples x features)
        basis: Optional orthonormal basis (features-1 x features)
              If None, uses default Helmert basis
    
    Returns:
        ILR-transformed data (samples x features-1)
        
    Note:
        ILR reduces dimensionality by 1 (from D to D-1 features)
    """
    from skbio.stats.composition import ilr
    
    # Handle zeros
    data_closed = closure(data)
    data_imputed = multiplicative_replacement(data_closed)
    
    # Apply ILR
    if basis is not None:
        ilr_data = ilr(data_imputed, basis=basis)
    else:
        ilr_data = ilr(data_imputed)
    
    return ilr_data


# ============================= PHYLO-ILR (PhILR) =================================== #

def philr_transform(
    table: Table,
    tree,
    part_weights: str = 'enorm',
    ilr_mode: str = 'weighted'
) -> pd.DataFrame:
    """
    Apply phylogenetic isometric log-ratio (PhILR) transformation.
    
    PhILR incorporates phylogenetic information into the ILR transformation,
    making it more biologically meaningful than standard CLR/ILR.
    
    Args:
        table: BIOM table
        tree: Phylogenetic tree (skbio TreeNode)
        part_weights: Weighting scheme ('enorm', 'anorm', or 'none')
        ilr_mode: ILR calculation mode ('weighted' or 'unweighted')
    
    Returns:
        PhILR-transformed DataFrame
        
    Requires:
        - Phylogenetic tree for the features
        - philr package (optional dependency)
        
    References:
        Silverman et al. (2017). A phylogenetic transform enhances analysis
        of compositional microbiota data. eLife.
    """
    try:
        from philr import philr
        from philr.philr import _balance_basis
    except ImportError:
        raise ImportError(
            "PhILR transformation requires the philr package:\n"
            "  pip install philr\n"
            "Or use CLR transformation instead."
        )
    
    # Convert to DataFrame
    df = table.to_dataframe(dense=True).T  # samples x features
    
    # Apply PhILR
    philr_df = philr.philr_transform(
        df,
        tree,
        part_weights=part_weights,
        ilr_mode=ilr_mode
    )
    
    return philr_df


# ============================== VALIDATION TOOLS =================================== #

def check_compositional(
    data: np.ndarray,
    tolerance: float = 1e-5
) -> bool:
    """
    Check if data are properly compositional (sum to 1 per sample).
    
    Args:
        data: Input array (samples x features)
        tolerance: Acceptable deviation from sum=1
    
    Returns:
        True if compositional, False otherwise
    """
    row_sums = data.sum(axis=1)
    is_compositional = np.allclose(row_sums, 1.0, atol=tolerance)
    
    if not is_compositional:
        logger.warning(
            f"Data are not compositional. Row sums range: "
            f"{row_sums.min():.6f} to {row_sums.max():.6f}"
        )
    
    return is_compositional


def diagnose_zeros(
    table: Union[Table, pd.DataFrame, np.ndarray]
) -> dict:
    """
    Diagnose zero patterns in compositional data.
    
    Args:
        table: Input table
    
    Returns:
        Dictionary with zero diagnostics
    """
    if isinstance(table, Table):
        data = table.matrix_data.toarray().T if issparse(table.matrix_data) else table.matrix_data.T
    elif isinstance(table, pd.DataFrame):
        data = table.values
    else:
        data = table
    
    n_samples, n_features = data.shape
    zeros = data == 0
    
    diagnostics = {
        'n_samples': n_samples,
        'n_features': n_features,
        'total_zeros': zeros.sum(),
        'zero_fraction': zeros.sum() / (n_samples * n_features),
        'samples_with_zeros': (zeros.any(axis=1)).sum(),
        'features_with_zeros': (zeros.any(axis=0)).sum(),
        'samples_all_zeros': (zeros.all(axis=1)).sum(),
        'features_all_zeros': (zeros.all(axis=0)).sum(),
    }
    
    logger.info(
        f"Zero diagnostics:\n"
        f"  Total zeros: {diagnostics['total_zeros']:,} "
        f"({diagnostics['zero_fraction']:.2%} of data)\n"
        f"  Samples with zeros: {diagnostics['samples_with_zeros']:,}/{n_samples}\n"
        f"  Features with zeros: {diagnostics['features_with_zeros']:,}/{n_features}"
    )
    
    return diagnostics


# ============================== ALR TRANSFORMATION ================================= #

def alr_transform(
    data: np.ndarray,
    reference_idx: Optional[int] = None,
    handle_zeros: bool = True,
    zero_method: str = 'multiplicative'
) -> np.ndarray:
    """
    Apply additive log-ratio (ALR) transformation.
    
    ALR transforms compositional data by taking the log-ratio of each component
    to a reference component. Unlike CLR, ALR is asymmetric (depends on reference
    choice) but produces interpretable ratios.
    
    Args:
        data: Input array (samples x features)
        reference_idx: Index of reference component (default: last feature)
        handle_zeros: Whether to replace zeros before transformation
        zero_method: Method for zero replacement
    
    Returns:
        ALR-transformed data (samples x features-1)
        
    Formula:
        ALR(x_i) = log(x_i / x_ref)
        
    Note:
        - Reduces dimensionality by 1
        - Interpretation: ALR_i = log(feature_i / reference_feature)
        - Positive values: feature_i > reference
        - Negative values: feature_i < reference
        
    References:
        Aitchison (1986). The Statistical Analysis of Compositional Data.
    """
    # Handle zeros if requested
    if handle_zeros and np.any(data == 0):
        logger.info(f"Applying {zero_method} zero replacement for ALR")
        data_closed = closure(data)
        data_processed = multiplicative_replacement(data_closed)
    else:
        data_processed = data + constants.DEFAULT_PSEUDOCOUNT if np.any(data == 0) else data
    
    # Select reference component
    if reference_idx is None:
        reference_idx = data_processed.shape[1] - 1  # Last feature
    
    # Extract reference
    reference = data_processed[:, reference_idx:reference_idx+1]  # Keep 2D shape
    
    # Remove reference from data
    data_without_ref = np.delete(data_processed, reference_idx, axis=1)
    
    # Calculate log-ratios
    alr_data = np.log(data_without_ref) - np.log(reference)
    
    logger.debug(f"ALR transformation: {data.shape[1]} features → {alr_data.shape[1]} features")
    
    return alr_data


# ========================= COMPOSITIONAL CORRELATIONS ============================== #

def compositional_correlation(
    data: np.ndarray,
    method: str = 'proportionality',
    transformation: str = 'clr',
    handle_zeros: bool = True
) -> np.ndarray:
    """
    Calculate correlations appropriate for compositional data.
    
    Standard Pearson correlation is invalid for compositional data due to
    spurious correlations. This function provides compositionally-aware
    alternatives.
    
    Args:
        data: Input array (samples x features)
        method: Correlation method
                - 'proportionality': φ (phi) coefficient
                - 'rho': ρ (rho) proportionality
                - 'clr_pearson': Pearson on CLR-transformed data
                - 'vlr': Variation Log Ratio
        transformation: Pre-transformation ('clr', 'alr', or 'none')
        handle_zeros: Whether to replace zeros
    
    Returns:
        Correlation matrix (features x features)
        
    References:
        Lovell et al. (2015). Proportionality: A Valid Alternative to Correlation
        for Relative Data. PLoS Comput Biol.
    """
    # Apply transformation
    if transformation == 'clr':
        data_transformed = clr_transform(data, handle_zeros=handle_zeros)
    elif transformation == 'alr':
        data_transformed = alr_transform(data, handle_zeros=handle_zeros)
    elif transformation == 'none':
        data_transformed = data
    else:
        raise ValueError(f"Unknown transformation: {transformation}")
    
    n_features = data_transformed.shape[1]
    
    if method == 'proportionality' or method == 'phi':
        # φ (phi) coefficient - variance of log-ratios
        corr_matrix = _calculate_phi(data_transformed)
        
    elif method == 'rho':
        # ρ (rho) proportionality - correlation of CLR coordinates
        corr_matrix = _calculate_rho(data_transformed)
        
    elif method == 'clr_pearson':
        # Standard Pearson on CLR-transformed data
        if transformation != 'clr':
            logger.warning("clr_pearson method should use CLR transformation")
        corr_matrix = np.corrcoef(data_transformed.T)
        
    elif method == 'vlr':
        # Variation Log Ratio
        corr_matrix = _calculate_vlr(data_transformed)
        
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    return corr_matrix


def _calculate_phi(data_clr: np.ndarray) -> np.ndarray:
    """
    Calculate φ (phi) proportionality coefficient.
    
    φ_ij = var(log(x_i/x_j)) / var(log x_i) + var(log x_j)
    
    Lower values indicate higher proportionality.
    """
    n_features = data_clr.shape[1]
    phi_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            # Variance of log-ratio
            log_ratio = data_clr[:, i] - data_clr[:, j]
            var_ratio = np.var(log_ratio)
            
            # Sum of individual variances
            var_i = np.var(data_clr[:, i])
            var_j = np.var(data_clr[:, j])
            
            # φ coefficient
            phi_ij = var_ratio / (var_i + var_j) if (var_i + var_j) > 0 else 0
            phi_matrix[i, j] = phi_ij
            phi_matrix[j, i] = phi_ij
    
    return phi_matrix


def _calculate_rho(data_clr: np.ndarray) -> np.ndarray:
    """
    Calculate ρ (rho) proportionality coefficient.
    
    ρ_ij = 1 - φ_ij
    
    Higher values indicate higher proportionality (like standard correlation).
    """
    phi_matrix = _calculate_phi(data_clr)
    rho_matrix = 1 - phi_matrix
    np.fill_diagonal(rho_matrix, 1.0)  # Perfect self-proportionality
    return rho_matrix


def _calculate_vlr(data_clr: np.ndarray) -> np.ndarray:
    """
    Calculate Variation Log Ratio matrix.
    
    VLR_ij = var(log(x_i/x_j))
    """
    n_features = data_clr.shape[1]
    vlr_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                log_ratio = data_clr[:, i] - data_clr[:, j]
                vlr_matrix[i, j] = np.var(log_ratio)
    
    return vlr_matrix


def compositional_pca(
    data: np.ndarray,
    n_components: Optional[int] = None,
    transformation: str = 'clr',
    handle_zeros: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform compositionally-aware PCA.
    
    Standard PCA on compositional data is problematic. This function applies
    appropriate transformations before PCA.
    
    Args:
        data: Input array (samples x features)
        n_components: Number of components (default: min(n_samples, n_features))
        transformation: Transformation method ('clr', 'alr', 'ilr')
        handle_zeros: Whether to replace zeros
    
    Returns:
        Tuple of (principal_components, explained_variance_ratio, loadings)
        
    References:
        Gloor et al. (2017). Microbiome Datasets Are Compositional.
    """
    from sklearn.decomposition import PCA
    
    # Apply transformation
    if transformation == 'clr':
        data_transformed = clr_transform(data, handle_zeros=handle_zeros)
    elif transformation == 'alr':
        data_transformed = alr_transform(data, handle_zeros=handle_zeros)
    elif transformation == 'ilr':
        data_transformed = ilr_transform(data)
    else:
        raise ValueError(f"Unknown transformation: {transformation}")
    
    # Perform PCA
    if n_components is None:
        n_components = min(data_transformed.shape[0], data_transformed.shape[1])
    
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_transformed)
    
    logger.info(
        f"Compositional PCA ({transformation.upper()}):\n"
        f"  Components: {n_components}\n"
        f"  Variance explained: {pca.explained_variance_ratio_.sum():.2%}"
    )
    
    return principal_components, pca.explained_variance_ratio_, pca.components_


# ========================== COMPOSITIONAL DISTANCE ================================= #

def aitchison_distance(
    data: np.ndarray,
    handle_zeros: bool = True,
    zero_method: str = 'multiplicative'
) -> np.ndarray:
    """
    Calculate Aitchison distance matrix for compositional data.
    
    The Aitchison distance is the natural metric for compositional data,
    equivalent to Euclidean distance in CLR space.
    
    Args:
        data: Input array (samples x features)
        handle_zeros: Whether to replace zeros
        zero_method: Zero replacement method
    
    Returns:
        Distance matrix (samples x samples)
        
    Formula:
        d_A(x, y) = ||CLR(x) - CLR(y)||_2
        
    Properties:
        - Subcompositionally dominant (adding/removing features preserves distances)
        - Invariant to closure (scaling doesn't affect distances)
        - Satisfies triangle inequality
        
    References:
        Aitchison (1992). On criteria for measures of compositional difference.
        Mathematical Geology.
    """
    from scipy.spatial.distance import pdist, squareform
    
    # CLR transformation
    data_clr = clr_transform(data, handle_zeros=handle_zeros, zero_method=zero_method)
    
    # Euclidean distance in CLR space = Aitchison distance
    distances = pdist(data_clr, metric='euclidean')
    distance_matrix = squareform(distances)
    
    logger.debug(f"Calculated Aitchison distance matrix: {distance_matrix.shape}")
    
    return distance_matrix
