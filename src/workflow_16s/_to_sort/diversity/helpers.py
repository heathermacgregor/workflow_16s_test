# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import os
import logging
from typing import Any, Dict, Optional, Union

# Third-Party Imports
import numpy as np
import pandas as pd
from biom import Table
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from skbio.diversity import beta_diversity
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import OrdinationResults, pcoa as PCoA, pcoa_biplot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from umap import UMAP

# Local Imports
from workflow_16s import constants

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ================================== CONSTANTS ======================================= #

NONNEGATIVE_METRICS = {
    'braycurtis', 'jaccard', 'aitchison', 'unweighted_unifrac', 'weighted_unifrac'
}
SKLEARN_METRICS = {'euclidean', 'cityblock', 'minkowski', 'cosine', 'correlation'}

# =============================== HELPER FUNCTIONS ==================================== #


def validate_table_df(df: pd.DataFrame, **kwargs):
    validate_input_data(df)
    if 'min_samples' in kwargs:
        validate_min_samples(df, kwargs['min_samples'])
    if 'n_components' in kwargs:
        safe_n_components = safe_component_limit(df, kwargs['n_components'])
        validate_component_count(safe_n_components)
        return safe_n_components


def validate_input_data(df: pd.DataFrame):
    # Validate input data
    if np.isnan(df.values).any():
        raise ValueError("Input data contains NaN values")
    if np.isinf(df.values).any():
        raise ValueError("Input data contains infinite values")


def validate_min_samples(
    df: pd.DataFrame, 
    min_samples: int = 2
) -> None:
    """Validate that the input contains sufficient samples for analysis.
    
    Ensures the input DataFrame meets minimum sample requirements for downstream
    statistical analysis and dimensionality reduction techniques.
    
    Args:
        df :          
            Input data as pandas DataFrame with samples as rows and features 
            as columns.
        min_samples : 
            Minimum required number of samples (default: 2).
        
    Raises:
        ValueError: If number of samples is less than required minimum
    """
    if len(df) < min_samples:
        raise ValueError(f"At least {min_samples} samples required")


def validate_component_count(n_components: int) -> None:
    """Validate that the requested number of components is valid.
    
    Ensures the requested number of components is a positive integer suitable
    for dimensionality reduction techniques.
    
    Args:
        n_components : 
            Requested number of components.
        
    Raises:
        ValueError: If n_components is less than 1
    """
    if n_components < 1:
        raise ValueError("n_components must be ≥ 1")


def safe_component_limit(
    df: pd.DataFrame, 
    requested: int
) -> int:
    """Determine safe number of components based on data dimensions.
    
    Computes the maximum possible components given data constraints:
    - For dimensionality reduction: min(n_samples - 1, n_features)
    - For distance-based methods: n_samples - 1
    
    Args:
        df :        
            Input data as pandas DataFrame with samples as rows and features 
            as columns.
        requested : 
            Originally requested number of components.
        
    Returns:
        Safe number of components to compute (min(requested, max_possible))
    """
    max_components = min(len(df) - 1, df.shape[1])
    return min(requested, max_components)


def create_result_df(
    data: np.ndarray, 
    index: pd.Index, 
    prefix: str, 
    n_components: int
) -> pd.DataFrame:
    """Create standardized result DataFrame with named components.
    
    Generates a DataFrame with standardized column naming convention for
    dimensionality reduction results.
    
    Args:
        data : 
            Embedding array of shape (n_samples, n_components)
        index : 
            Sample identifiers for DataFrame index
        prefix : 
            Component name prefix (e.g., 'PC', 'UMAP')
        n_components : 
            Number of components in the result
        
    Returns:
        DataFrame with named components (prefix + number) and sample index
    """
    columns = [f"{prefix}{i+1}" for i in range(n_components)]
    return pd.DataFrame(data, index=index, columns=columns)


def handle_duplicate_ids(ids: list) -> list:
    """Resolve duplicate sample IDs by appending numerical suffixes.
    
    Ensures all sample identifiers are unique by appending numerical suffixes
    to duplicate entries while preserving original identifiers when possible.
    
    Example: 
        Input: ['A', 'B', 'A'] → Output: ['A_1', 'B', 'A_2']
    
    Args:
        ids : 
            List of original sample identifiers
        
    Returns:
        List of unique identifiers with duplicates disambiguated
    """
    seen = {}
    new_ids = []
    for sample_id in ids:
        count = seen.get(sample_id, 0) + 1
        seen[sample_id] = count
        new_ids.append(f"{sample_id}_{count}" if count > 1 else sample_id)
    return new_ids

def calculate_correlation_loadings(
    original_data: pd.DataFrame, 
    embeddings: pd.DataFrame
) -> pd.DataFrame:
    """Calculate correlation loadings between original features and embedding dimensions.
    
    Computes Pearson correlations between standardized original features and
    embedding coordinates to interpret feature contributions to each dimension.
    
    Args:
        original_data : 
            Original feature data (samples × features)
        embeddings : 
            Embedding coordinates (samples × components)
        
    Returns:
        DataFrame with correlation loadings (features × components)
    """
    # Standardize the original data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(original_data.values)
    
    # Calculate correlations
    loadings = np.zeros((original_data.shape[1], embeddings.shape[1]))
    
    for i in range(original_data.shape[1]):  # For each feature
        for j in range(embeddings.shape[1]):  # For each component
            corr, _ = pearsonr(scaled_data[:, i], embeddings.iloc[:, j].values)
            loadings[i, j] = corr
    
    # Create DataFrame with proper indexing
    return pd.DataFrame(
        loadings,
        index=original_data.columns,
        columns=embeddings.columns
    )

# =============================== CORE FUNCTIONALITY ================================== #

def table_to_dataframe(table: Union[Dict, Table, pd.DataFrame]) -> pd.DataFrame:
    """Convert various table formats to standardized DataFrame (samples × features).
    
    Normalizes input data from multiple supported formats into a consistent
    pandas DataFrame structure suitable for downstream analysis.
    
    Supports:
    - BIOM Table (transposed to samples × features)
    - Dictionary of {sample_id: {feature: count}} mappings
    - Existing pandas DataFrame (returned as-is)
    
    Args:
        table : 
            Input data in supported format
        
    Returns:
        DataFrame with samples as rows and features as columns
        
    Raises:
        ValueError: For unsupported input types or empty tables
    """
    if isinstance(table, Table):
        return table.to_dataframe(dense=True).T
    if isinstance(table, dict):
        return pd.DataFrame.from_dict(table, orient='index')
    if isinstance(table, pd.DataFrame):
        return table
    raise ValueError("Unsupported input type: must be Table, dict or DataFrame")


def validate_distance_matrix(dm: DistanceMatrix):
    """Perform comprehensive validation of distance matrix.
    
    Validates and cleans distance matrices by:
        1. Handling NaN values through symmetric imputation
        2. Ensuring matrix symmetry
        3. Checking for degeneracy (all identical values)
        4. Ensuring diagonal is exactly zero
    
    Args:
        dm : 
            Input DistanceMatrix object
        
    Returns:
        Validated and cleaned DistanceMatrix
        
    Raises:
        ValueError: For invalid distance matrices that cannot be cleaned
    """
    dm_data = dm.data.copy()

    # Step 1: Handle NaNs symmetrically
    if np.isnan(dm_data).any():
        # Check for all NaNs first
        if np.isnan(dm_data).all():
            raise ValueError("Distance matrix is all NaNs")
        total_mean = np.nanmean(dm_data)  # Global mean of non-NaN values
        
        # Ensure diagonal is 0 (set to 0 if NaN)
        n = dm_data.shape[0]
        for i in range(n):
            if np.isnan(dm_data[i, i]):
                dm_data[i, i] = 0.0
        # Symmetric imputation for off-diagonal elements
        for i in range(n):
            for j in range(i + 1, n):  # Only process upper triangle
                if np.isnan(dm_data[i, j]) and not np.isnan(dm_data[j, i]):
                    dm_data[i, j] = dm_data[j, i]  # Fill with symmetric value
                elif not np.isnan(dm_data[i, j]) and np.isnan(dm_data[j, i]):
                    dm_data[j, i] = dm_data[i, j]  # Fill with symmetric value
                elif np.isnan(dm_data[i, j]) and np.isnan(dm_data[j, i]):
                    # Both NaN: set to global mean
                    dm_data[i, j] = total_mean
                    dm_data[j, i] = total_mean
        
        # Verify no NaNs remain
        if np.isnan(dm_data).any():
            raise ValueError("Distance matrix contains NaNs that couldn't be imputed")
    
    # Step 2: Check symmetry and enforce if nearly symmetric
    if not np.allclose(dm_data, dm_data.T, atol=1e-8):
        raise ValueError("Distance matrix is not symmetric")
    else:
        # Make matrix exactly symmetric
        dm_data = (dm_data + dm_data.T) / 2
    
    # Step 3: Check for degeneracy (only for matrices larger than 1x1)
    if dm_data.size > 1:
        if np.allclose(dm_data, dm_data.flat[0]):  # Check if all values are nearly identical
            raise ValueError("Distance matrix is degenerate (all values identical)")
    
    # Step 4: Ensure diagonal is exactly 0
    np.fill_diagonal(dm_data, 0.0)
    
    return DistanceMatrix(dm_data, ids=dm.ids)
