import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances

logger = logging.getLogger('workflow_16s')

def check_temporal_structure(adata: ad.AnnData, time_col: str, subject_col: str) -> Dict:
    """Assess if dataset supports longitudinal analysis."""
    if time_col not in adata.obs.columns:
        raise ValueError(f"Time column '{time_col}' missing.")
    if subject_col not in adata.obs.columns:
        raise ValueError(f"Subject column '{subject_col}' missing.")
    
    counts = adata.obs.groupby(subject_col).size()
    info = {
        'n_subjects': len(counts),
        'n_timepoints': adata.obs[time_col].nunique(),
        'n_repeated': (counts > 1).sum(),
        'is_longitudinal': (counts > 1).sum() > 0
    }
    
    if not info['is_longitudinal']:
        logger.warning("No repeated measurements found.")
    else:
        logger.info(f"Found {info['n_repeated']} subjects with repeated measures.")
        
    return info

def trajectory_clustering(
    adata: ad.AnnData,
    time_col: str,
    subject_col: str,
    n_clusters: int = 4,
    features: Optional[List[str]] = None,
    method: str = 'kmeans'
) -> Dict:
    """Cluster subjects based on temporal trajectory features (mean & slope)."""
    if features is None: features = adata.var_names
    abundance = adata[:, features].to_df()
    
    profiles, ids = [], []
    
    for subject in adata.obs[subject_col].unique():
        sub_data = adata.obs[adata.obs[subject_col] == subject]
        if len(sub_data) < 2: continue # Skip single points
        
        sub_abund = abundance.loc[sub_data.index]
        times = sub_data[time_col].values
        
        # Simple feature engineering: Mean abundance + Slope of change
        means = sub_abund.mean(axis=1).mean() # Global mean
        try:
            slope, _, _, _, _ = stats.linregress(times, sub_abund.mean(axis=1))
        except: slope = 0
            
        profiles.append([means, slope])
        ids.append(subject)

    if not profiles:
        logger.warning("Not enough data for clustering.")
        return {}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(profiles)
    
    model = KMeans(n_clusters=n_clusters, random_state=42) if method == 'kmeans' else AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X_scaled)
    
    cluster_df = pd.DataFrame({'subject': ids, 'cluster': labels})
    
    # Map back to adata
    subject_map = dict(zip(ids, labels))
    adata.obs['trajectory_cluster'] = adata.obs[subject_col].map(subject_map)
    
    return {'cluster_assignments': cluster_df, 'n_clusters': n_clusters}

def calculate_temporal_stability(
    adata: ad.AnnData,
    time_col: str,
    subject_col: str,
    metric: str = 'braycurtis'
) -> pd.DataFrame:
    """Calculate subject stability (1 - avg distance between consecutive timepoints)."""
    results = []
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    for subject in adata.obs[subject_col].unique():
        mask = adata.obs[subject_col] == subject
        indices = np.where(mask)[0]
        if len(indices) < 2: continue
        
        times = adata.obs.iloc[indices][time_col].values
        sorted_idx = indices[np.argsort(times)]
        
        sub_X = X[sorted_idx]
        dists = [pairwise_distances(sub_X[i:i+2], metric=metric)[0, 1] for i in range(len(sub_X)-1)]
        
        avg_dist = np.mean(dists)
        results.append({
            'subject': subject,
            'stability_score': 1.0 - avg_dist,
            'avg_step_distance': avg_dist
        })
        
    return pd.DataFrame(results)