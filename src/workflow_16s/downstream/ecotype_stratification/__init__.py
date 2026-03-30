"""
Module 2: Ecotype Stratification

Detects cryptic ecotypes (strain variants) within 99% OTUs based on:
- Functional trait patterns
- Ecological niche specialization
- Geographic/environmental distributions

Scientific Question: Are there hidden strain variants with distinct
functional and ecological profiles within species-level OTUs?
"""

from .ecotype_detection import (
    EcotypeDetector,
    detect_ecotypes_from_traits,
    assign_ecotypes,
    compute_ecotype_profiles,
)

from .niche_analysis import (
    NicheAnalyzer,
    analyze_niche_specialization,
    quantify_niche_breadth,
)

from .trait_clustering import (
    cluster_by_traits,
    evaluate_cluster_stability,
    get_optimal_cluster_count,
    get_clustering_methods,
)

from .stratification_report import (
    analyze_ecotype_stratification,
    generate_stratification_report,
    summarize_ecotype_results,
)

__all__ = [
    # Detection
    "EcotypeDetector",
    "detect_ecotypes_from_traits",
    "assign_ecotypes",
    "compute_ecotype_profiles",
    # Niche
    "NicheAnalyzer",
    "analyze_niche_specialization",
    "quantify_niche_breadth",
    # Clustering
    "cluster_by_traits",
    "evaluate_cluster_stability",
    "get_optimal_cluster_count",
    "get_clustering_methods",
    # Reporting
    "analyze_ecotype_stratification",
    "generate_stratification_report",
    "summarize_ecotype_results",
]
