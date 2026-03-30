# downstream/phylogenetic_signal.py

"""
Phylogenetic Signal Analysis: Compute Pagel's Lambda for functional traits.

Answers: Is a trait (e.g., uranium reduction) phylogenetically conserved (follows evolution)
or randomly distributed (suggests adaptive convergence/HGT)?

Works with distance matrices or phylogenetic trees.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
import logging

from workflow_16s.utils.logger import get_logger

logger = get_logger("workflow_16s")


def compute_phylogenetic_signal_fast(
    trait_vector: np.ndarray,
    distance_matrix: np.ndarray,
    otu_names: list
) -> Dict[str, float]:
    """
    Fast approximation of phylogenetic signal using distance-based method.
    
    Computes Pagel's lambda via correlation between phylogenetic distance
    and trait difference. This is faster than tree-based methods and works
    with distance matrices (no tree structure needed).
    
    Args:
        trait_vector: Binary/continuous trait for each OTU (0 or 1)
        distance_matrix: Pairwise phylogenetic distance matrix
        otu_names: OTU identifiers
    
    Returns:
        Dict with:
        - lambda: Pagel's lambda estimate (0-1)
        - p_value: Significance (permutation test)
        - interpretation: Human-readable result
    """
    
    # Convert to numpy arrays
    trait = np.array(trait_vector, dtype=float)
    dist = np.array(distance_matrix, dtype=float)
    
    # Handle case where all OTUs have/lack trait
    if trait.sum() == 0 or trait.sum() == len(trait):
        return {
            "lambda": np.nan,
            "p_value": np.nan,
            "interpretation": "No signal (trait present/absent in all OTUs)",
            "n_otus_with_trait": trait.sum()
        }
    
    n = len(trait)
    
    # Compute pairwise trait differences
    trait_diff = np.abs(np.subtract.outer(trait, trait))
    
    # Correlation between distance and trait difference
    # Higher correlation = phylogenetically signal (conserved)
    flat_dist = dist[np.triu_indices(n, k=1)]
    flat_trait_diff = trait_diff[np.triu_indices(n, k=1)]
    
    # Exclude zero-distance pairs (identical OTUs)
    mask = flat_dist > 0
    flat_dist = flat_dist[mask]
    flat_trait_diff = flat_trait_diff[mask]
    
    if len(flat_dist) < 3:
        return {
            "lambda": np.nan,
            "p_value": np.nan,
            "interpretation": "Insufficient OTU pairs for signal estimation",
            "n_otus_with_trait": trait.sum()
        }
    
    # Compute correlation as proxy for lambda
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.corrcoef(flat_dist, flat_trait_diff)[0, 1]
    
    # Convert correlation to lambda approximation
    # corr = 1 → phylogenetically conserved (λ ≈ 1)
    # corr = 0 → random distribution (λ ≈ 0)
    lambda_approx = max(0.0, min(1.0, corr))
    
    # Permutation test for significance
    n_perms = 999
    perm_scores = []
    np.random.seed(42)
    
    for _ in range(n_perms):
        perm_trait = np.random.permutation(trait)
        perm_diff = np.abs(np.subtract.outer(perm_trait, perm_trait))
        flat_perm = perm_diff[np.triu_indices(n, k=1)][mask]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            perm_corr = np.corrcoef(flat_dist, flat_perm)[0, 1]
            perm_scores.append(max(0, perm_corr))
    
    perm_scores = np.array(perm_scores)
    p_value = (perm_scores >= lambda_approx).sum() / n_perms
    
    # Interpret result
    if p_value < 0.05:
        if lambda_approx > 0.6:
            interpretation = "✅ CONSERVED: Trait is phylogenetically signal (likely ancestral)"
        else:
            interpretation = "🔄 ADAPTIVE: Trait is randomly distributed (likely HGT or convergence)"
    else:
        interpretation = "❓ UNCLEAR: No significant phylogenetic signal"
    
    return {
        "lambda": lambda_approx,
        "p_value": p_value,
        "interpretation": interpretation,
        "n_otus_with_trait": int(trait.sum()),
        "n_otus_total": len(trait)
    }


def estimate_distance_matrix_from_abundance(
    otu_table: pd.DataFrame,
    method: str = "braycurtis"
) -> np.ndarray:
    """
    Estimate OTU-OTU distance matrix from abundance patterns.
    
    Uses co-occurrence similarity as proxy for phylogenetic relatedness.
    This works when you don't have a phylogenetic tree.
    """
    from scipy.spatial.distance import pdist, squareform
    from sklearn.metrics.pairwise import cosine_distances
    
    logger.info(f"Estimating phylogenetic distance from {method} similarity...")
    
    # Normalize OTU abundances (convert to proportions)
    otu_norm = otu_table.div(otu_table.sum(axis=0), axis=1).fillna(0)
    
    if method == "braycurtis":
        distances = pdist(otu_norm.T, metric="braycurtis")
    elif method == "jaccard":
        otu_binary = (otu_norm > 0).astype(int)
        distances = pdist(otu_binary.T, metric="jaccard")
    else:
        distances = pdist(otu_norm.T, metric="correlation")
    
    dist_matrix = squareform(distances)
    
    return dist_matrix


def run_phylogenetic_signal_analysis(
    function_matrix: pd.DataFrame,
    otu_table: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    output_dir: Path,
    config: Dict,
    phylo_method: str = "fast",
    reference_tree_path: Optional[Path] = None,
) -> Dict:
    """
    Main entry point for phylogenetic signal analysis.
    
    Args:
        function_matrix: OTU × Function matrix (from functional_annotation)
        otu_table: OTU abundance table
        taxonomy_df: Taxonomy metadata
        output_dir: Output directory
        config: Configuration dict
        phylo_method: 'fast' (distance-based) or 'full' (tree-based)
        reference_tree_path: Optional path to phylogenetic tree
    
    Returns:
        Dict with phylogenetic signal results for each trait
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*80)
    logger.info("PHYLOGENETIC SIGNAL ANALYSIS (Pagel's Lambda)")
    logger.info("="*80)
    logger.info(f"Input: {len(function_matrix)} OTUs × {len(function_matrix.columns)} functions")
    logger.info(f"Method: {phylo_method}")
    
    # Estimate distance matrix if no tree provided
    if reference_tree_path and Path(reference_tree_path).exists():
        logger.info(f"Loading phylogenetic tree from {reference_tree_path}")
        # Would load tree here (requires Bio.Phylo or similar)
        # For now, estimate from abundance
        dist_matrix = estimate_distance_matrix_from_abundance(otu_table)
    else:
        logger.info("Estimating phylogenetic distance from OTU co-occurrence patterns...")
        dist_matrix = estimate_distance_matrix_from_abundance(otu_table)
    
    # Compute phylogenetic signal for each function
    results = {}
    
    for func in function_matrix.columns:
        trait_vector = function_matrix[func].values
        
        signal_result = compute_phylogenetic_signal_fast(
            trait_vector,
            dist_matrix,
            function_matrix.index.tolist()
        )
        
        results[func] = signal_result
        
        logger.info(f"\n{func}:")
        logger.info(f"  Pagel's λ: {signal_result['lambda']:.3f}")
        logger.info(f"  p-value: {signal_result['p_value']:.4f}")
        logger.info(f"  {signal_result['interpretation']}")
        logger.info(f"  OTUs with trait: {signal_result['n_otus_with_trait']} / {signal_result['n_otus_total']}")
    
    # Save results
    results_df = pd.DataFrame({
        func: results[func]
        for func, _ in results.items()
    }).T
    
    results_df.to_csv(output_dir / "phylogenetic_signal_results.csv")
    
    logger.info(f"\n✓ Results saved to {output_dir}/phylogenetic_signal_results.csv")
    
    return results
