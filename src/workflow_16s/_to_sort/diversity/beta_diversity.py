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
from workflow_16s.utils.dataframe import table_to_df
from workflow_16s.diversity.helpers import (
    create_result_df, validate_distance_matrix, validate_table_df
)

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ================================== CONSTANTS ======================================= #

NONNEGATIVE_METRICS = {
    'braycurtis', 'jaccard', 'aitchison', 'unweighted_unifrac', 'weighted_unifrac'
}
SKLEARN_METRICS = {'euclidean', 'cityblock', 'minkowski', 'cosine', 'correlation'}

# =============================== HELPER FUNCTIONS ==================================== #


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
    df = table_to_df(table)
    validate_table_df(df, min_samples=2)
    sample_ids = df.index.tolist()
    data = df.values
    
    # Special handling for compositional metrics
    if metric == 'aitchison':
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
    # Compute and validate distance matrix
    dm = distance_matrix(table, metric=metric)
    dm = validate_distance_matrix(dm)
    
    # Determine safe component count
    max_dims = min(len(df) - 1, dm.shape[0] - 1)
    n_dimensions = min(n_dimensions, max_dims) if n_dimensions else max_dims
    
    # Perform PCoA
    result = PCoA(dm, number_of_dimensions=n_dimensions)

    # Add feature loadings via biplot
    loadings = pcoa_biplot(result, df).features
    result.feature_loadings = loadings
    
    # Standardize output
    components = [f"PCo{i+1}" for i in range(result.samples.shape[1])]
    result.samples.columns = components
    result.feature_loadings.columns = components
    
    return result


def pca(
    table: Union[Dict, Table, pd.DataFrame],
    n_components: int = constants.DEFAULT_N_PCA
) -> Dict[str, Any]:
    """Perform Principal Component Analysis (PCA) on feature data.
        - Standardizes features (mean=0, variance=1)
        - Computes principal components via SVD
        - Returns component scores and loadings
    
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
    df = table_to_df(table)
    safe_n_components = validate_table_df(df, min_samples=2, n_components=n_components)
    
    # Standardize and transform
    scaled_data = StandardScaler().fit_transform(df.values)
    model = PCA(n_components=safe_n_components)
    scores = model.fit_transform(scaled_data)

    components = create_result_df(scores, df.index, "PC", safe_n_components)
    exp_var = model.explained_variance_
    exp_var_ratio = model.explained_variance_ratio_
    exp_var_cumul = np.cumsum(exp_var_ratio)
    loadings = model.components_.T * np.sqrt(exp_var)

    result = {
        'components': components,
        'exp_var_ratio': exp_var_ratio,
        'exp_var_cumul': exp_var_cumul,
        'loadings': loadings
    }
    if not df.shape[0] == components.shape[0]:
        raise
    if not df.shape[1] == loadings.shape[0]:
        raise
    return result


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
    df = table_to_df(table)
    safe_n_components = validate_table_df(df, min_samples=2, n_components=n_components)
    
    # Compute t-SNE embeddings
    model = TSNE(
        n_components=safe_n_components, random_state=random_state, n_jobs=n_jobs
    )
    embeddings = model.fit_transform(df.values)
    components = create_result_df(embeddings, df.index, "TSNE", safe_n_components)
    
    # Calculate correlation loadings
    loadings = calculate_correlation_loadings(df, components)

    return {'components': components, 'loadings': loadings}
    

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
    df = table_to_df(table)
    safe_n_components = validate_table_df(df, min_samples=2, n_components=n_components)

    def reduce(df, n_components, random_state, n_jobs: int = 1):
        model = UMAP(
            n_components=n_components, init='random',
            random_state=random_state, n_jobs=n_jobs
        )
        embeddings = model.fit_transform(df.values)
        return embeddings
    try: # Attempt UMAP with requested thread count
        embeddings = reduce(df, safe_n_components, random_state, n_jobs=n_jobs)
    except RuntimeError as e: 
        if "threading" in str(e).lower(): # Fallback to single-threaded execution
            embeddings = reduce(df, safe_n_components, random_state, n_jobs=1)
        else:
            raise
    
    components = create_result_df(embeddings, df.index, "UMAP", safe_n_components)
    loadings = calculate_correlation_loadings(df, components)
    
    return {'components': components, 'loadings': loadings}


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
    df = table_to_df(table)
    safe_n_components = validate_table_df(df, min_samples=2, n_components=n_components)
    
    # Compute MDS embedding
    model = MDS(
        n_components=safe_n_components, random_state=random_state, n_jobs=n_jobs,
        dissimilarity='precomputed' if metric == 'precomputed' else 'euclidean'
    )
    
    if metric == 'precomputed': # Use precomputed distance matrix
        if not isinstance(table, DistanceMatrix):
            raise ValueError("For precomputed metric, input must be a DistanceMatrix")
        embeddings = model.fit_transform(table.data)
    else: # Compute distance matrix from feature data
        embeddings = model.fit_transform(df.values)
    
    components = create_result_df(embeddings, df.index, "MDS", safe_n_components)
    loadings = calculate_correlation_loadings(df, components)
    
    return {'components': components, 'loadings': loadings, 'stress': model.stress_}
  
