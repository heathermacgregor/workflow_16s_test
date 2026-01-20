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


def create_result_dataframe(
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
    

def distance_matrix(
    table: Union[Dict, Table, pd.DataFrame],
    metric: str = constants.DEFAULT_METRIC
) -> DistanceMatrix:
    """Compute distance matrix with enhanced validation.
    
    Calculates pairwise distance matrices between samples with comprehensive
    input validation and support for both sklearn and scikit-bio metrics.
    
    Args:
        table : 
            Input data in supported format
        metric : 
            Distance metric to use (default: constants.DEFAULT_METRIC)
        
    Returns:
        DistanceMatrix object containing pairwise distances between samples
        
    Raises:
        ValueError: For invalid input data containing NaN or infinite values
    """
    df = table_to_dataframe(table)
    validate_min_samples(df, min_samples=2)
    sample_ids = df.index.tolist()
    data = df.values
    
    # Validate input data
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values")
    
    # Special handling for compositional metrics
    if metric == 'aitchison':
        from skbio.diversity import beta_diversity
        return beta_diversity('aitchison', data, ids=sample_ids)
    
    # Compute distance matrix
    dist_array = pairwise_distances(data, metric=metric)
    
    # Ensure symmetry for metrics that should be symmetric
    if metric in {'euclidean', 'braycurtis', 'jaccard'}:
        dist_array = (dist_array + dist_array.T) / 2
    
    return DistanceMatrix(dist_array, ids=sample_ids)
    

def pcoa(
    table: Union[Dict, Table, pd.DataFrame], 
    metric: str = constants.DEFAULT_METRIC, 
    n_dimensions: Optional[int] = constants.DEFAULT_N_PCOA
) -> OrdinationResults:
    """Robust PCoA with enhanced distance matrix validation and feature loadings.
    
    Performs Principal Coordinate Analysis with comprehensive distance matrix
    validation, automatic dimension reduction, and calculation of feature loadings
    for improved interpretability.
    
    Args:
        table : 
            Input data in supported format
        metric : 
            Distance metric to use (default: constants.DEFAULT_METRIC)
        n_dimensions : 
            Number of dimensions to compute (default: constants.DEFAULT_N_PCOA)
        
    Returns:
        OrdinationResults object containing:
        - Sample coordinates in reduced space
        - Feature loadings (correlations between original features and axes)
        - Explained variance per component
        
    Raises:
        ValueError: For insufficient samples or invalid distance matrices
    """
    df = table_to_dataframe(table)
    validate_min_samples(df, min_samples=2)
    
    # Compute and validate distance matrix
    dm = distance_matrix(table, metric=metric)
    dm = validate_distance_matrix(dm)
    
    # Determine safe component count
    max_dims = min(len(df) - 1, dm.shape[0] - 1)
    n_dimensions = min(n_dimensions, max_dims) if n_dimensions else max_dims
    
    # Perform PCoA
    pcoa_result = PCoA(dm, number_of_dimensions=n_dimensions)

    # Add feature loadings via biplot
    ordination_biplot = pcoa_biplot(pcoa_result, df)
    loadings = ordination_biplot.features
    
    # Standardize output
    comp_names = [f"PCo{i+1}" for i in range(pcoa_result.samples.shape[1])]
    pcoa_result.samples.columns = comp_names
    
    # Add feature loadings to the result object
    pcoa_result.feature_loadings = loadings
    pcoa_result.feature_loadings.columns = comp_names
    
    return pcoa_result


def pca(
    table: Union[Dict, Table, pd.DataFrame],
    n_components: int = constants.DEFAULT_N_PCA
) -> Dict[str, Any]:
    """Perform Principal Component Analysis (PCA) on feature data.
    
    This method:
        1. Standardizes features (mean=0, variance=1)
        2. Computes principal components via SVD
        3. Returns component scores and loadings
    
    Args:
        table : 
            Input data in supported format
        n_components : 
            Number of principal components to compute (default: constants.DEFAULT_N_PCA)
        
    Returns:
        Dictionary with:
        - 'components': DataFrame of component scores (n_samples × n_components)
        - 'exp_var_ratio': Explained variance ratio per component
        - 'exp_var_cumul': Cumulative explained variance
        - 'loadings': Feature loadings (n_features × n_components)
        
    Raises:
        ValueError: For insufficient samples or invalid component count
    """
    df = table_to_dataframe(table)
    validate_min_samples(df, min_samples=2)
    validate_component_count(n_components)
    
    # Determine safe component count
    n_components = safe_component_limit(df, n_components)
    
    # Standardize and transform
    scaled_data = StandardScaler().fit_transform(df.values)
    pca_model = PCA(n_components=n_components)
    scores = pca_model.fit_transform(scaled_data)
    
    return {
        'components': create_result_dataframe(scores, df.index, "PC", n_components),
        'exp_var_ratio': pca_model.explained_variance_ratio_,
        'exp_var_cumul': np.cumsum(pca_model.explained_variance_ratio_),
        'loadings': pca_model.components_.T * np.sqrt(pca_model.explained_variance_)
    }


def tsne(
    table: Union[Dict, Table, pd.DataFrame],
    n_components: int = constants.DEFAULT_N_TSNE,
    random_state: int = constants.DEFAULT_RANDOM_STATE,
    n_jobs: int = constants.DEFAULT_CPU_LIMIT
) -> Dict[str, Any]:
    """Compute t-Distributed Stochastic Neighbor Embedding (t-SNE).
    
    Suitable for high-dimensional data visualization. This method:
        1. Models pairwise similarities in high-dimensional space
        2. Optimizes low-dimensional embedding to preserve local structures
    
    Args:
        table : 
            Input data in supported format
        n_components : 
            Dimension of embedding space (typically 2-3, default: constants.DEFAULT_N_TSNE)
        random_state : 
            Seed for reproducible results (default: constants.DEFAULT_RANDOM_STATE)
        n_jobs : 
            CPU cores to use (-1 for all available, default: constants.DEFAULT_CPU_LIMIT)
        
    Returns:
        Dictionary with:
        - 'components': DataFrame of t-SNE coordinates (n_samples × n_components)
        - 'loadings': Feature loadings (n_features × n_components)
        
    Raises:
        ValueError: For insufficient samples, invalid components, or data issues
    """
    df = table_to_dataframe(table)
    validate_min_samples(df, min_samples=2)
    validate_component_count(n_components)
    
    # Validate data quality
    if not np.isfinite(df.values).all():
        raise ValueError("Input data contains non-finite values")
    
    n_components = safe_component_limit(df, n_components)
    
    # Compute t-SNE embedding
    tsne_model = TSNE(
        n_components=n_components,
        random_state=random_state,
        n_jobs=n_jobs
    )
    embeddings = tsne_model.fit_transform(df.values)
    
    # Create components DataFrame
    components_df = create_result_dataframe(embeddings, df.index, "TSNE", n_components)
    
    # Calculate correlation loadings
    loadings_df = calculate_correlation_loadings(df, components_df)
    
    return {
        'components': components_df,
        'loadings': loadings_df
    }
    

def umap(
    table: Union[Dict, Table, pd.DataFrame],
    n_components: int = constants.DEFAULT_N_UMAP,
    random_state: int = constants.DEFAULT_RANDOM_STATE,
    n_jobs: int = constants.DEFAULT_CPU_LIMIT
) -> Dict[str, Any]:
    """Compute Uniform Manifold Approximation and Projection (UMAP).
    
    Preserves both local and global data structures. This method:
        1. Constructs topological representation of data
        2. Optimizes low-dimensional embedding
    
    Args:
        table : 
            Input data in supported format
        n_components : 
            Dimension of embedding space (typically 2-3, default: constants.DEFAULT_N_UMAP)
        random_state : 
            Seed for reproducible results (default: constants.DEFAULT_RANDOM_STATE)
        n_jobs : 
            CPU cores to use (default: constants.DEFAULT_CPU_LIMIT)
        
    Returns:
        Dictionary with:
        - 'components': DataFrame of UMAP coordinates (n_samples × n_components)
        - 'loadings': Feature loadings (n_features × n_components)
        
    Raises:
        ValueError: For insufficient samples or invalid components
        RuntimeError: For threading issues (handled internally)
    """
    df = table_to_dataframe(table)
    validate_min_samples(df, min_samples=2)
    validate_component_count(n_components)
    n_components = safe_component_limit(df, n_components)
    
    try:
        # Attempt UMAP with requested thread count
        reducer = UMAP(
            n_components=n_components,
            init='random',
            random_state=random_state,
            n_jobs=n_jobs
        )
        embeddings = reducer.fit_transform(df.values)
    except RuntimeError as e:
        if "threading" in str(e).lower():
            # Fallback to single-threaded execution
            reducer = UMAP(
                n_components=n_components,
                init='random',
                random_state=random_state,
                n_jobs=1
            )
            embeddings = reducer.fit_transform(df.values)
        else:
            raise
    
    # Create components DataFrame
    components_df = create_result_dataframe(embeddings, df.index, "UMAP", n_components)
    
    # Calculate correlation loadings
    loadings_df = calculate_correlation_loadings(df, components_df)
    
    return {
        'components': components_df,
        'loadings': loadings_df
    }

def mds(
    table: Union[Dict, Table, pd.DataFrame],
    metric: str = constants.DEFAULT_METRIC,
    n_components: int = constants.DEFAULT_N_MDS,
    random_state: int = constants.DEFAULT_RANDOM_STATE,
    n_jobs: int = constants.DEFAULT_CPU_LIMIT
) -> Dict[str, Any]:
    """Compute Metric Multidimensional Scaling (MDS).
    
    Projects high-dimensional data into lower dimensions while preserving
    pairwise distances as much as possible.
    
    Args:
        table : 
            Input data in supported format
        metric : 
            Distance metric to use (default: constants.DEFAULT_METRIC)
        n_components : 
            Dimension of embedding space (typically 2-3, default: constants.DEFAULT_N_MDS)
        random_state : 
            Seed for reproducible results (default: constants.DEFAULT_RANDOM_STATE)
        n_jobs : 
            CPU cores to use (default: constants.DEFAULT_CPU_LIMIT)
        
    Returns:
        Dictionary with:
        - 'components': DataFrame of MDS coordinates (n_samples × n_components)
        - 'loadings': Feature loadings (n_features × n_components)
        - 'stress': Final value of the stress (loss) function
        
    Raises:
        ValueError: For insufficient samples or invalid components
    """
    df = table_to_dataframe(table)
    validate_min_samples(df, min_samples=2)
    validate_component_count(n_components)
    n_components = safe_component_limit(df, n_components)
    
    # Compute MDS embedding
    mds_model = MDS(
        n_components=n_components,
        random_state=random_state,
        n_jobs=n_jobs,
        dissimilarity='precomputed' if metric == 'precomputed' else 'euclidean'
    )
    
    if metric == 'precomputed':
        # Use precomputed distance matrix
        if not isinstance(table, DistanceMatrix):
            raise ValueError("For precomputed metric, input must be a DistanceMatrix")
        embeddings = mds_model.fit_transform(table.data)
    else:
        # Compute distance matrix from feature data
        embeddings = mds_model.fit_transform(df.values)
    
    # Create components DataFrame
    components_df = create_result_dataframe(embeddings, df.index, "MDS", n_components)
    
    # Calculate correlation loadings
    loadings_df = calculate_correlation_loadings(df, components_df)
    
    return {
        'components': components_df,
        'loadings': loadings_df,
        'stress': mds_model.stress_
    }
