"""
Trait-based clustering utilities for ecotype detection.

Supports multiple clustering methods and model selection strategies.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

logger = logging.getLogger(__name__)


def get_clustering_methods() -> List[str]:
    """Return available clustering methods."""
    return ["kmeans", "hierarchical", "spectral"]


def cluster_by_traits(
    trait_matrix: np.ndarray,
    n_clusters: int = 3,
    method: str = "kmeans",
    random_state: int = 42,
) -> np.ndarray:
    """
    Cluster samples (or OTUs) by trait similarity.
    
    Args:
        trait_matrix: (n_samples, n_traits) matrix of trait values
        n_clusters: Number of clusters
        method: 'kmeans', 'hierarchical', or 'spectral'
        random_state: Random seed
        
    Returns:
        Cluster assignments array (0 to n_clusters-1)
    """
    if method == "kmeans":
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            algorithm='lloyd'
        )
        return clusterer.fit_predict(trait_matrix)
    
    elif method == "hierarchical":
        Z = linkage(trait_matrix, method='ward')
        return fcluster(Z, n_clusters, criterion='maxclust') - 1
    
    elif method == "spectral":
        clusterer = SpectralClustering(
            n_clusters=n_clusters,
            random_state=random_state,
            affinity='nearest_neighbors',
            n_neighbors=min(10, trait_matrix.shape[0] - 1)
        )
        return clusterer.fit_predict(trait_matrix)
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def evaluate_cluster_stability(
    trait_matrix: np.ndarray,
    n_clusters: int,
    method: str = "kmeans",
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Evaluate clustering quality using multiple metrics.
    
    Args:
        trait_matrix: (n_samples, n_traits) matrix
        n_clusters: Number of clusters
        method: Clustering method
        random_state: Random seed
        
    Returns:
        Dictionary with metrics:
        - 'silhouette': -1 to 1 (higher = better)
        - 'davies_bouldin': > 0 (lower = better)
        - 'calinski_harabasz': > 0 (higher = better)
    """
    if trait_matrix.shape[0] < n_clusters:
        return {
            'silhouette': np.nan,
            'davies_bouldin': np.nan,
            'calinski_harabasz': np.nan,
        }
    
    labels = cluster_by_traits(trait_matrix, n_clusters, method, random_state)
    
    metrics = {}
    
    # Silhouette score
    try:
        metrics['silhouette'] = silhouette_score(trait_matrix, labels)
    except:
        metrics['silhouette'] = np.nan
    
    # Davies-Bouldin index (lower = better)
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(trait_matrix, labels)
    except:
        metrics['davies_bouldin'] = np.nan
    
    # Calinski-Harabasz index (higher = better)
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(trait_matrix, labels)
    except:
        metrics['calinski_harabasz'] = np.nan
    
    return metrics


def get_optimal_cluster_count(
    trait_matrix: np.ndarray,
    n_clusters_range: Tuple[int, int] = (2, 10),
    method: str = "kmeans",
    metric: str = "silhouette",
    random_state: int = 42,
) -> Tuple[int, float]:
    """
    Find optimal number of clusters using elbow method or stability metrics.
    
    Args:
        trait_matrix: (n_samples, n_traits) matrix
        n_clusters_range: (min_k, max_k) to test
        method: Clustering method
        metric: 'silhouette', 'davies_bouldin', or 'elbow'
        random_state: Random seed
        
    Returns:
        Tuple of (optimal_n_clusters, metric_value)
    """
    min_k, max_k = n_clusters_range
    max_k = min(max_k, trait_matrix.shape[0] // 2)
    
    if min_k >= max_k:
        return (2, 0.0)
    
    results = {}
    
    for n_clusters in range(min_k, max_k + 1):
        try:
            metrics = evaluate_cluster_stability(
                trait_matrix, n_clusters, method, random_state
            )
            results[n_clusters] = metrics
        except Exception as e:
            logger.debug(f"Failed to evaluate {n_clusters} clusters: {e}")
            continue
    
    if not results:
        return (min_k, 0.0)
    
    # Select best based on chosen metric
    if metric == "silhouette":
        best_k = max(
            results.keys(),
            key=lambda k: results[k].get('silhouette', -np.inf)
        )
        best_score = results[best_k].get('silhouette', 0.0)
    
    elif metric == "davies_bouldin":
        # Lower is better, so invert
        best_k = min(
            results.keys(),
            key=lambda k: results[k].get('davies_bouldin', np.inf)
        )
        best_score = 1.0 / (1.0 + results[best_k].get('davies_bouldin', 1.0))
    
    elif metric == "calinski_harabasz":
        best_k = max(
            results.keys(),
            key=lambda k: results[k].get('calinski_harabasz', -np.inf)
        )
        best_score = results[best_k].get('calinski_harabasz', 0.0) / 1000  # Normalize
    
    elif metric == "elbow":
        # Elbow method: minimize within-cluster sum of squares
        inertias = {}
        for n_clusters in results.keys():
            labels = cluster_by_traits(trait_matrix, n_clusters, method, random_state)
            kmeans_temp = KMeans(n_clusters=n_clusters, random_state=random_state)
            kmeans_temp.fit(trait_matrix)
            inertias[n_clusters] = kmeans_temp.inertia_
        
        # Find elbow using second derivative
        ks = sorted(inertias.keys())
        if len(ks) >= 3:
            second_diffs = []
            for i in range(1, len(ks) - 1):
                diff2 = (inertias[ks[i+1]] - inertias[ks[i]]) - (inertias[ks[i]] - inertias[ks[i-1]])
                second_diffs.append((ks[i], diff2))
            best_k = max(second_diffs, key=lambda x: x[1])[0]
        else:
            best_k = ks[0] if ks else min_k
        best_score = 1.0  # Normalized
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return (best_k, best_score)


def compare_clustering_methods(
    trait_matrix: np.ndarray,
    n_clusters: int,
    methods: Optional[List[str]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compare performance of different clustering methods.
    
    Args:
        trait_matrix: (n_samples, n_traits) matrix
        n_clusters: Number of clusters
        methods: List of methods to compare (default: all)
        random_state: Random seed
        
    Returns:
        DataFrame with method comparison metrics
    """
    if methods is None:
        methods = get_clustering_methods()
    
    results = []
    for method in methods:
        try:
            metrics = evaluate_cluster_stability(
                trait_matrix, n_clusters, method, random_state
            )
            metrics['method'] = method
            results.append(metrics)
        except Exception as e:
            logger.warning(f"Failed to evaluate {method}: {e}")
    
    return pd.DataFrame(results)


def plot_clustering_results(
    trait_matrix: np.ndarray,
    labels: np.ndarray,
    method: str = "kmeans",
) -> Dict:
    """
    Generate cluster visualization data (for use with plotly, etc.).
    
    Args:
        trait_matrix: (n_samples, n_traits) matrix
        labels: Cluster assignments
        method: Clustering method used
        
    Returns:
        Dictionary with visualization info
    """
    from sklearn.decomposition import PCA
    
    # Use PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    trait_2d = pca.fit_transform(trait_matrix)
    
    # Compute cluster statistics
    unique_labels = np.unique(labels)
    cluster_stats = {}
    
    for label in unique_labels:
        mask = labels == label
        cluster_stats[int(label)] = {
            'n_samples': mask.sum(),
            'centroid': trait_2d[mask].mean(axis=0).tolist(),
            'variance': float(trait_2d[mask].var()),
        }
    
    return {
        'method': method,
        'coordinates_2d': trait_2d.tolist(),
        'labels': labels.tolist(),
        'pca_variance_explained': pca.explained_variance_ratio_.tolist(),
        'cluster_stats': cluster_stats,
    }
