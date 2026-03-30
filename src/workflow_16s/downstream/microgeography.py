# downstream/microgeography.py

"""
Microgeographic Clustering: Detect spatially-cohesive microbial communities.

Answers: At sub-meter to meter scales, do microbiomes cluster by proximity?
Suggests localized environmental drivers or dispersal limitation.

Methods:
1. Haversine distance matrix from lat/lon
2. Procrustes analysis (Procrustes to physical coordinates)
3. Mantel test (correlation between biological and spatial distance)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import spearmanr
import warnings

from workflow_16s.utils.logger import get_logger

logger = get_logger("workflow_16s")


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Haversine formula: Great-circle distance between two lat/lon points (km).
    
    Args:
        lat1, lon1, lat2, lon2: Latitude and longitude in degrees
    
    Returns:
        Distance in kilometers
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def build_spatial_distance_matrix(
    metadata_df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    unit: str = "km"
) -> np.ndarray:
    """
    Build spatial distance matrix from lat/lon coordinates.
    
    Args:
        metadata_df: Metadata with lat/lon columns
        lat_col: Latitude column name
        lon_col: Longitude column name
        unit: "km", "m" (meters), or "degrees"
    
    Returns:
        N×N distance matrix
    """
    
    if lat_col not in metadata_df.columns or lon_col not in metadata_df.columns:
        raise ValueError(f"Columns {lat_col}, {lon_col} not found in metadata")
    
    # Remove samples with missing coordinates
    valid_mask = metadata_df[[lat_col, lon_col]].notna().all(axis=1)
    coords = metadata_df.loc[valid_mask, [lat_col, lon_col]].values
    
    if len(coords) < 2:
        raise ValueError("Not enough samples with valid lat/lon coordinates")
    
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[j]
            
            if unit == "km":
                dist = haversine_distance(lat1, lon1, lat2, lon2)
            elif unit == "m":
                dist = haversine_distance(lat1, lon1, lat2, lon2) * 1000
            elif unit == "degrees":
                dist = np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
            else:
                raise ValueError(f"Unknown unit: {unit}")
            
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix, valid_mask


def mantel_test(
    bio_dist: np.ndarray,
    spatial_dist: np.ndarray,
    n_permutations: int = 9999
) -> Dict:
    """
    Mantel test: correlation between biological and spatial distance matrices.
    
    Args:
        bio_dist: Biological distance matrix (e.g., Bray-Curtis)
        spatial_dist: Spatial distance matrix
        n_permutations: Number of permutations for p-value calculation
    
    Returns:
        Dict with correlation, p-value, permutation statistic
    """
    
    # Flatten upper triangles
    bio_vec = squareform(bio_dist, checks=False)
    spatial_vec = squareform(spatial_dist, checks=False)
    
    # Observed correlation
    r_obs, _ = spearmanr(bio_vec, spatial_vec)
    
    # Permutation test
    r_perms = []
    np.random.seed(42)
    
    for _ in range(n_permutations):
        # Permute biological matrix
        perm_idx = np.random.permutation(len(bio_dist))
        bio_perm = bio_dist[np.ix_(perm_idx, perm_idx)]
        bio_perm_vec = squareform(bio_perm, checks=False)
        
        r_perm, _ = spearmanr(bio_perm_vec, spatial_vec)
        r_perms.append(r_perm)
    
    r_perms = np.array(r_perms)
    
    # One-tailed p-value (correlation stronger than expected by chance)
    p_value = (np.sum(r_perms >= r_obs) + 1) / (n_permutations + 1)
    
    return {
        "r_observed": r_obs,
        "p_value": p_value,
        "r_mean_permutation": np.mean(r_perms),
        "r_std_permutation": np.std(r_perms),
        "interpretation": "Significant spatial structure" if p_value < 0.05 else "No spatial structure"
    }


def procrustes_analysis(
    bio_coords: np.ndarray,
    spatial_coords: np.ndarray
) -> Dict:
    """
    Procrustes analysis: Align two coordinate systems and measure similarity.
    
    Args:
        bio_coords: Biological coordinates (e.g., NMDS, PCA)
        spatial_coords: Physical spatial coordinates (lat/lon or geographic)
    
    Returns:
        Dict with Procrustes distance and alignment
    """
    
    try:
        from scipy.linalg import orthogonal_procrustes
    except ImportError:
        logger.warning("⚠️ scipy.linalg.orthogonal_procrustes not available")
        return {"error": "scipy version too old"}
    
    # Center both coordinate systems
    bio_centered = bio_coords - bio_coords.mean(axis=0)
    spatial_centered = spatial_coords - spatial_coords.mean(axis=0)
    
    # Find optimal rotation
    U, _ = orthogonal_procrustes(spatial_centered, bio_centered)
    
    # Rotated spatial coordinates
    spatial_rotated = spatial_centered @ U
    
    # Procrustes distance (sum of squared differences)
    procrustes_dist = np.sum((bio_centered - spatial_rotated)**2) ** 0.5
    
    # Normalize by maximum possible distance
    max_dist = np.sum(bio_centered**2) ** 0.5
    normalized_dist = procrustes_dist / max_dist if max_dist > 0 else 1.0
    
    return {
        "procrustes_distance": procrustes_dist,
        "normalized_distance": normalized_dist,
        "rotation_matrix": U,
        "interpretation": "Strong alignment" if normalized_dist < 0.3 else "Weak alignment"
    }


def spatial_clustering_coefficient(
    dist_matrix: np.ndarray,
    threshold_km: float = 1.0
) -> Dict:
    """
    Calculate clustering coefficient for spatial proximity graph.
    
    Measures local clustering in a spatial network (e.g., samples within 1 km).
    
    Args:
        dist_matrix: Spatial distance matrix
        threshold_km: Distance threshold for "neighborhood"
    
    Returns:
        Dict with clustering stats
    """
    
    # Build adjacency matrix (samples within threshold)
    adjacency = (dist_matrix > 0) & (dist_matrix <= threshold_km)
    np.fill_diagonal(adjacency, 0)
    
    # For each node, calculate local clustering coefficient
    clustering_coeffs = []
    
    for i in range(len(adjacency)):
        neighbors = np.where(adjacency[i])[0]
        
        if len(neighbors) < 2:
            clustering_coeffs.append(0.0)
            continue
        
        # Count edges between neighbors
        neighbor_subgraph = adjacency[np.ix_(neighbors, neighbors)]
        edges_in_neighborhood = np.sum(neighbor_subgraph) / 2
        
        # Possible edges
        possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
        
        if possible_edges == 0:
            clustering_coeffs.append(0.0)
        else:
            clustering_coeffs.append(edges_in_neighborhood / possible_edges)
    
    return {
        "mean_clustering_coefficient": np.mean(clustering_coeffs),
        "median_clustering_coefficient": np.median(clustering_coeffs),
        "threshold_km": threshold_km,
        "n_connected_components": len(np.unique([np.where(adjacency[i])[0] for i in range(len(adjacency)) if len(np.where(adjacency[i])[0]) > 0]))
    }


def run_microgeography_analysis(
    adata,
    beta_diversity_df: pd.DataFrame,
    output_dir: Path,
    config: Dict,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    max_distance_km: float = 100.0,
    spatial_threshold_km: float = 1.0
) -> Dict:
    """
    Main entry point for microgeographic analysis.
    
    Args:
        adata: AnnData object with metadata
        beta_diversity_df: Precomputed beta diversity matrix (Bray-Curtis, etc.)
        output_dir: Output directory
        config: Configuration dict
        lat_col: Latitude column in adata.obs
        lon_col: Longitude column in adata.obs
        max_distance_km: Max spatial distance to include
        spatial_threshold_km: Threshold for local clustering coefficient
    
    Returns:
        Dict with microgeography results
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*80)
    logger.info("MICROGEOGRAPHIC ANALYSIS")
    logger.info("="*80)
    logger.info(f"Input: {len(adata)} samples")
    
    # Check for lat/lon columns
    if lat_col not in adata.obs.columns or lon_col not in adata.obs.columns:
        logger.error(f"❌ Columns {lat_col}, {lon_col} not found in metadata")
        return {"error": "Missing geographic coordinates"}
    
    # Build spatial distance matrix
    try:
        spatial_dist, valid_mask = build_spatial_distance_matrix(
            adata.obs,
            lat_col=lat_col,
            lon_col=lon_col,
            unit="km"
        )
    except ValueError as e:
        logger.error(f"❌ {e}")
        return {"error": str(e)}
    
    # Subset to valid samples
    valid_indices = np.where(valid_mask)[0]
    bio_dist_subset = beta_diversity_df.iloc[valid_indices, valid_indices].values
    
    logger.info(f"✓ Analyzing {len(valid_indices)} samples with valid coordinates")
    logger.info(f"  Spatial range: {spatial_dist[spatial_dist > 0].min():.2f} - {spatial_dist.max():.2f} km")
    
    results = {}
    
    # 1. Mantel test
    logger.info("\n  Running Mantel test...")
    mantel_result = mantel_test(bio_dist_subset, spatial_dist, n_permutations=9999)
    logger.info(f"    Mantel r = {mantel_result['r_observed']:.4f}, p = {mantel_result['p_value']:.4f}")
    logger.info(f"    {mantel_result['interpretation']}")
    results["mantel"] = mantel_result
    
    # 2. Spatial clustering
    logger.info(f"\n  Computing spatial clustering (threshold={spatial_threshold_km} km)...")
    clustering_result = spatial_clustering_coefficient(spatial_dist, threshold_km=spatial_threshold_km)
    logger.info(f"    Mean clustering coeff: {clustering_result['mean_clustering_coefficient']:.3f}")
    results["spatial_clustering"] = clustering_result
    
    # 3. Procrustes (if NMDS available)
    if "NMDS" in adata.obsm:
        logger.info("\n  Running Procrustes analysis...")
        spatial_coords = adata.obs[[lat_col, lon_col]].values
        bio_coords = adata.obsm["NMDS"][valid_indices]
        
        procrustes_result = procrustes_analysis(bio_coords, spatial_coords)
        logger.info(f"    Procrustes distance: {procrustes_result.get('normalized_distance', np.nan):.3f}")
        logger.info(f"    {procrustes_result.get('interpretation', '')}")
        results["procrustes"] = procrustes_result
    
    # Summary
    logger.info("\n✓ Microgeography analysis complete")
    
    # Save summary
    summary_df = pd.DataFrame({
        "Metric": ["Mantel_r", "Mantel_p_value", "Spatial_clustering_coeff", "Interpretation"],
        "Value": [
            mantel_result['r_observed'],
            mantel_result['p_value'],
            clustering_result['mean_clustering_coefficient'],
            mantel_result['interpretation']
        ]
    })
    
    summary_df.to_csv(output_dir / "microgeography_summary.csv", index=False)
    logger.info(f"✓ Results saved to {output_dir}/")
    
    return results
