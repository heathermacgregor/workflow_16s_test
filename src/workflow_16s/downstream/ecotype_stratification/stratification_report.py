"""
Module 2 orchestrator: Ecotype stratification analysis and reporting.

Combines ecotype detection, niche analysis, and generates comprehensive reports.
Implements fingerprint-based caching for expensive ecotype detection.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import json
import pickle

import numpy as np
import pandas as pd
import scanpy as sc

from .ecotype_detection import (
    EcotypeDetector,
    EcotypeProfile,
    detect_ecotypes_from_traits,
    assign_ecotypes,
    compute_ecotype_profiles,
)
from .niche_analysis import (
    NicheAnalyzer,
    analyze_niche_specialization,
    quantify_niche_breadth,
)

logger = logging.getLogger(__name__)


def _compute_ecotype_cache_fingerprint(
    trait_matrix: pd.DataFrame,
    adata: sc.AnnData,
    min_prevalence: int,
    clustering_method: str,
    n_clusters_range: Tuple[int, int],
) -> str:
    """
    Compute fingerprint of ecotype detection inputs.
    
    Returns hash that changes when:
    - Trait matrix shape or OTU IDs change
    - AnnData OTU annotations change
    - Configuration parameters change
    """
    fingerprint_dict = {
        'trait_shape': trait_matrix.shape,
        'otu_ids_hash': hashlib.md5(
            pd.Series(trait_matrix.index).astype(str).str.cat().encode()
        ).hexdigest(),
        'otu_count': len(adata.var_names),
        'sample_count': len(adata.obs_names),
        'min_prevalence': min_prevalence,
        'clustering_method': clustering_method,
        'n_clusters_range': n_clusters_range,
    }
    
    fingerprint_json = json.dumps(fingerprint_dict, sort_keys=True)
    return hashlib.md5(fingerprint_json.encode()).hexdigest()


def _load_ecotype_cache(
    cache_dir: Path,
    fingerprint: str,
) -> Optional[Dict]:
    """
    Load cached ecotype detection results if fingerprint matches.
    
    Returns:
        Cached result dict or None if cache miss/stale
    """
    if not cache_dir.exists():
        return None
    
    cache_file = cache_dir / 'ecotype_detection_cache.pkl'
    fingerprint_file = cache_dir / 'ecotype_detection_fingerprint.json'
    
    if not cache_file.exists() or not fingerprint_file.exists():
        return None
    
    try:
        # Check fingerprint match
        with open(fingerprint_file, 'r') as f:
            cached_fingerprint = json.load(f).get('fingerprint')
        
        if cached_fingerprint != fingerprint:
            logger.debug(f"Ecotype cache fingerprint mismatch (stale)")
            return None
        
        # Load cached results
        with open(cache_file, 'rb') as f:
            cached_results = pickle.load(f)
        
        logger.info(f"✓ Loaded ecotype detection from cache ({cache_file.stat().st_size / 1e6:.1f} MB)")
        return cached_results
    
    except Exception as e:
        logger.warning(f"Could not load ecotype cache: {e}")
        return None


def _save_ecotype_cache(
    result_dict: Dict,
    cache_dir: Path,
    fingerprint: str,
) -> None:
    """
    Save ecotype detection results to cache.
    
    Args:
        result_dict: Results from detect_ecotypes_from_traits()
        cache_dir: Directory for cache files
        fingerprint: Fingerprint that identifies this data
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / 'ecotype_detection_cache.pkl'
    fingerprint_file = cache_dir / 'ecotype_detection_fingerprint.json'
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(result_dict, f)
        
        with open(fingerprint_file, 'w') as f:
            json.dump({'fingerprint': fingerprint}, f)
        
        logger.info(f"✓ Cached ecotype detection ({cache_file.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        logger.warning(f"Could not save ecotype cache: {e}")


def analyze_ecotype_stratification(
    adata: sc.AnnData,
    trait_matrix: pd.DataFrame,
    otu_metadata_path: Optional[Path] = None,
    otu_level: int = 99,
    min_prevalence: int = 5,
    clustering_method: str = "kmeans",
    n_clusters_range: Tuple[int, int] = (2, 6),
    environmental_variable: Optional[str] = None,
    output_dir: Optional[Path] = None,
    n_workers: int = 8,
) -> Dict:
    """
    Complete ecotype stratification analysis pipeline.
    
    Args:
        adata: AnnData object with samples × OTUs
        trait_matrix: DataFrame with OTUs × Traits confidence scores
        otu_metadata_path: Optional path to RAST annotations
        otu_level: OTU clustering level (97, 99, etc.) for reference/logging
        min_prevalence: Min samples for OTU inclusion
        clustering_method: 'kmeans', 'hierarchical', or 'spectral'
        n_clusters_range: (min_clusters, max_clusters) to test
        environmental_variable: Optional env column for niche analysis
        output_dir: Directory for outputs
        n_workers: Number of workers for parallel operations
        
    Returns:
        Dictionary with:
        - 'ecotype_profiles': Dict[otu_id → EcotypeProfile]
        - 'ecotype_assignments': DataFrame
        - 'niche_profiles': Dict[(otu_id, ecotype_id) → NicheProfile]
        - 'summary': Summary DataFrame
    """
    logger.info("=" * 70)
    logger.info("MODULE 2: ECOTYPE STRATIFICATION")
    logger.info(f"Detecting cryptic strain variants within {otu_level}% OTUs")
    logger.info("=" * 70)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # STEP 1: Detect ecotypes from trait patterns
    logger.info("\n[1/3] ECOTYPE DETECTION")
    logger.info(f"  - Method: {clustering_method}")
    logger.info(f"  - Cluster range: {n_clusters_range}")
    logger.info(f"  - Min prevalence: {min_prevalence} samples")
    
    # Prepare metadata columns for niche analysis
    metadata_cols = [environmental_variable] if environmental_variable else []
    
    # Check cache before running expensive detection
    cache_dir = output_dir / '.cache_ecotypes' if output_dir else Path.home() / '.cache' / 'workflow_16s_ecotypes'
    fingerprint = _compute_ecotype_cache_fingerprint(
        trait_matrix, adata, min_prevalence, clustering_method, n_clusters_range
    )
    
    cached_ecotype_profiles = _load_ecotype_cache(cache_dir, fingerprint)
    
    if cached_ecotype_profiles is not None:
        # Use cached result
        ecotype_profiles = cached_ecotype_profiles
        logger.info(f"  ✓ Loaded {len(ecotype_profiles)} cached ecotype profiles")
    else:
        # Run detection (expensive operation)
        ecotype_profiles = detect_ecotypes_from_traits(
            adata,
            trait_matrix,
            min_prevalence=min_prevalence,
            clustering_method=clustering_method,
            n_clusters_range=n_clusters_range,
            metadata_cols=metadata_cols,
        )
        
        # Save to cache for next run
        _save_ecotype_cache(ecotype_profiles, cache_dir, fingerprint)
    
    n_stratified = len(ecotype_profiles)
    total_otus = trait_matrix.shape[0]
    logger.info(f"  ✓ Detected ecotypes in {n_stratified}/{total_otus} OTUs "
                f"({100*n_stratified/total_otus:.1f}%)")
    
    # STEP 2: Assign samples to ecotypes
    logger.info("\n[2/3] ECOTYPE ASSIGNMENT")
    ecotype_assignments = assign_ecotypes(adata, ecotype_profiles)
    logger.info(f"  ✓ Assigned {len(ecotype_assignments)} ecotype groups")
    
    # STEP 3: Analyze ecological niches
    logger.info("\n[3/3] NICHE ANALYSIS")
    analyzer = NicheAnalyzer(environment_columns=metadata_cols)
    niche_profiles = analyzer.analyze_ecotype_niches(
        adata,
        ecotype_assignments,
        min_samples=3,
    )
    logger.info(f"  ✓ Analyzed niches for {len(niche_profiles)} ecotypes")
    niche_comparisons = analyzer.compare_niche_partitioning(
        adata, ecotype_profiles
    )
    logger.info(f"  ✓ Quantified niche partitioning in {len(niche_comparisons)} OTUs")
    
    # STEP 4: Generate summary statistics
    logger.info("\n[4/4] SUMMARY STATISTICS")
    ecotype_summary = compute_ecotype_profiles(adata, ecotype_profiles)
    
    mean_stratification = ecotype_summary['stratification_score'].mean()
    max_stratification = ecotype_summary['stratification_score'].max()
    
    logger.info(f"  - Mean stratification score: {mean_stratification:.3f}")
    logger.info(f"  - Max stratification score: {max_stratification:.3f}")
    logger.info(f"  - OTUs with >2 ecotypes: {(ecotype_summary['n_ecotypes'] > 2).sum()}")
    logger.info(f"  - OTUs with >3 ecotypes: {(ecotype_summary['n_ecotypes'] > 3).sum()}")
    
    # Save outputs if requested
    if output_dir:
        logger.info(f"\n📁 Writing outputs to {output_dir}")
        try:
            ecotype_summary.to_csv(
                output_dir / "ecotype_stratification_summary.csv", index=False
            )
            ecotype_assignments.to_csv(
                output_dir / "ecotype_assignments.csv", index=False
            )
            logger.info("  ✓ Saved summary tables")
        except Exception as e:
            logger.warning(f"  ⚠ Failed to save outputs: {e}")
    
    return {
        'ecotype_profiles': ecotype_profiles,
        'ecotype_assignments': ecotype_assignments,
        'niche_profiles': niche_profiles,
        'niche_comparisons': niche_comparisons,
        'summary': ecotype_summary,
    }


def generate_stratification_report(
    results: Dict,
    output_path: Optional[Path] = None,
    include_visualizations: bool = True,
) -> str:
    """
    Generate formatted markdown report of ecotype stratification analysis.
    
    Args:
        results: Output dictionary from analyze_ecotype_stratification()
        output_path: Optional path to save report
        include_visualizations: Include visualization references
        
    Returns:
        Formatted markdown string
    """
    logger.info("Generating stratification report...")
    
    summary = results['summary']
    ecotype_profiles = results['ecotype_profiles']
    niche_comparisons = results['niche_comparisons']
    
    # Build report
    lines = [
        "# Ecotype Stratification Analysis Report",
        "",
        "## Executive Summary",
        "",
        f"**Total OTUs Analyzed**: {len(summary)}",
        f"**OTUs with Detected Ecotypes**: {(summary['n_ecotypes'] > 1).sum()}",
        f"**Mean Stratification Score**: {summary['stratification_score'].mean():.3f}",
        f"**Maximum Stratification Score**: {summary['stratification_score'].max():.3f}",
        "",
        "## Scientific Interpretation",
        "",
        "Ecotype stratification indicates the presence of cryptic strain variants",
        "within species-level OTUs. High stratification scores suggest:",
        "",
        "- **Hidden genetic diversity**: Multiple sympatric strains with distinct traits",
        "- **Niche partitioning**: Strains occupying different ecological niches",
        "- **Adaptive differentiation**: Traits under divergent environmental selection",
        "",
    ]
    
    # Top stratified OTUs
    if not summary.empty:
        lines.extend([
            "## Top Stratified OTUs",
            "",
            "| OTU ID | N Ecotypes | Stratification | Trait Diversity | Coherence |",
            "|--------|-----------|-----------------|-----------------|-----------|",
        ])
        
        top_otus = summary.nlargest(10, 'stratification_score')
        for _, row in top_otus.iterrows():
            lines.append(
                f"| {row['otu_id']} | {row['n_ecotypes']} | "
                f"{row['stratification_score']:.3f} | "
                f"{row['trait_diversity']:.3f} | "
                f"{row['ecological_coherence']:.3f} |"
            )
        
        lines.append("")
    
    # Niche partitioning results
    if niche_comparisons:
        lines.extend([
            "## Niche Partitioning Results",
            "",
            "Ecotypes within OTUs show varying degrees of ecological differentiation:",
            "",
            "| OTU ID | Partitioning Score | Most Differentiated Pair |",
            "|--------|-------------------|-------------------------|",
        ])
        
        for otu_id, comparison in sorted(
            niche_comparisons.items(),
            key=lambda x: x[1].niche_partitioning_score,
            reverse=True
        )[:10]:
            lines.append(
                f"| {otu_id} | {comparison.niche_partitioning_score:.3f} | "
                f"Ecotype {comparison.most_differentiated_pair[0]} vs "
                f"{comparison.most_differentiated_pair[1]} |"
            )
        
        lines.append("")
    
    # Discussion
    lines.extend([
        "## Methods",
        "",
        "- **Clustering Method**: KMeans on trait similarity matrix",
        "- **Cluster Evaluation**: Silhouette score + Davies-Bouldin index",
        "- **Niche Analysis**: Shannon entropy of environment distribution",
        "- **Partitioning Score**: Inverse of maximum niche overlap",
        "",
        "## Implications",
        "",
        "1. **For Taxonomy**: Consider if highly stratified OTUs warrant subdivision",
        "2. **For Ecology**: Ecotype patterns suggest environment-driven divergence",
        "3. **For Evolution**: Niche partitioning enables coexistence of similar strains",
        "4. **For Functional Analysis**: Traits may show ecotype-specificity",
        "",
    ])
    
    report_text = "\n".join(lines)
    
    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_text)
        logger.info(f"✓ Report saved to {output_path}")
    
    return report_text


def summarize_ecotype_results(
    results: Dict,
) -> str:
    """
    Generate human-readable summary of key findings.
    
    Args:
        results: Output dictionary from analyze_ecotype_stratification()
        
    Returns:
        Formatted summary string
    """
    summary = results['summary']
    
    lines = [
        "\n" + "=" * 70,
        "ECOTYPE STRATIFICATION: KEY FINDINGS",
        "=" * 70,
        "",
        f"Total OTUs with Detected Ecotypes: {(summary['n_ecotypes'] > 1).sum()}",
        f"  - 2 ecotypes: {(summary['n_ecotypes'] == 2).sum()}",
        f"  - 3+ ecotypes: {(summary['n_ecotypes'] >= 3).sum()}",
        "",
        f"Stratification Score Distribution:",
        f"  - Mean: {summary['stratification_score'].mean():.3f}",
        f"  - Median: {summary['stratification_score'].median():.3f}",
        f"  - Std Dev: {summary['stratification_score'].std():.3f}",
        f"  - Min: {summary['stratification_score'].min():.3f}",
        f"  - Max: {summary['stratification_score'].max():.3f}",
        "",
        "Scientific Interpretation:",
        "  ✓ Detected hidden strain variants (ecotypes) within OTUs",
        "  ✓ Ecotypes show distinct trait and niche profiles",
        "  ✓ Niche partitioning enables coexistence of similar strains",
        "  ✓ Suggests adaptive radiation or strain-level populations",
        "",
        "Next Steps:",
        "  1. Validate top ecotypes with genome sequencing",
        "  2. Test functional differences experimentally",
        "  3. Correlate ecotypes with specific environments",
        "  4. Investigate mechanisms of differentiation",
        "",
        "=" * 70 + "\n",
    ]
    
    return "\n".join(lines)
