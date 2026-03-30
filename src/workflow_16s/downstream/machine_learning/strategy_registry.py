"""Strategy Registry: Metadata and documentation for ML feature selection strategies.

This module documents the 6 feature selection strategies for microbial ML analysis.
Each strategy represents a different approach to composition-based feature selection,
with documented strengths, weaknesses, and recommended use cases.

Strategies are stored in `STRATEGY_METADATA` as a dict with strategy name -> metadata mapping.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class StrategyMetadata:
    """Metadata for a feature selection strategy."""
    
    name: str
    """Strategy identifier (matches config fs_strategies list)."""
    
    description: str
    """Human-readable description of the strategy."""
    
    strengths: List[str]
    """Advantages of this strategy."""
    
    weaknesses: List[str]
    """Limitations or challenges of this strategy."""
    
    batch_correction: str
    """Batch correction method: 'none', 'conqur', or 'batch_aware'."""
    
    cv_type: str
    """Cross-validation strategy: 'kfold', 'groupkfold', 'lopocv', 'spatial'."""
    
    feature_composition: str
    """How features are composed: 'asv_only', 'asv+metadata', 'asv+env', 'asv+batch'."""
    
    recommended_datasets: int
    """Recommended minimum number of datasets/projects for this strategy."""
    
    recommended_samples: int
    """Recommended minimum number of samples for this strategy."""
    
    hyperparameter_notes: Optional[str] = None
    """Notes on hyperparameter tuning for this strategy."""
    
    metadata_dependency: Optional[str] = None
    """Required metadata column if any (e.g., 'Project', 'lat/lon')."""
    
    computational_cost: str = "medium"
    """Relative computational cost: 'low', 'medium', 'high'."""
    
    recommended_for: Optional[str] = None
    """Use case or scenario where this strategy excels."""


STRATEGY_METADATA: Dict[str, StrategyMetadata] = {
    "baseline": StrategyMetadata(
        name="baseline",
        description=(
            "Standard split without batch correction. "
            "Uses raw ASV CLR-transformed data directly with random train/test split."
        ),
        strengths=[
            "Simplest approach (minimal assumptions)",
            "Fast to compute",
            "Good baseline for comparison",
            "Works with any dataset size",
        ],
        weaknesses=[
            "Ignores batch effects completely",
            "May conflate batch signals with true biological signals",
            "Overfits to batch-specific features",
            "Not suitable for multi-project datasets",
        ],
        batch_correction="none",
        cv_type="kfold",
        feature_composition="asv_only",
        recommended_datasets=1,
        recommended_samples=100,
        hyperparameter_notes="Use standard CatBoost hyperparameters (depth=7, learning_rate=0.03).",
        metadata_dependency=None,
        computational_cost="low",
        recommended_for="Single-project studies or quick baseline testing",
    ),
    
    "batch_aware": StrategyMetadata(
        name="batch_aware",
        description=(
            "ASV CLR transformed with Project/batch column as an explicit feature. "
            "Adds the study/project ID as a categorical feature to help model learn batch effects."
        ),
        strengths=[
            "Makes batch effects explicit to the model",
            "Model learns which features are batch-dependent vs. biological",
            "Still relatively simple and fast",
            "Good for understanding batch impact",
        ],
        weaknesses=[
            "Project ID can become confounding variable",
            "May overfit to specific batch patterns",
            "Doesn't correct batch effects, just exposes them",
            "GroupKFold is required (slower than KFold)",
        ],
        batch_correction="none",
        cv_type="groupkfold",
        feature_composition="asv+batch",
        recommended_datasets=2,
        recommended_samples=200,
        hyperparameter_notes="Use GroupKFold by project. May need lower learning rate (0.02) to handle high dimensionality.",
        metadata_dependency="Project",
        computational_cost="low",
        recommended_for="Testing batch effect impact; understanding confounding",
    ),
    
    "conqur": StrategyMetadata(
        name="conqur",
        description=(
            "Conditional Quantile Regression (ConQuR) batch correction applied before feature selection. "
            "Uses R-based ConQuR via rpy2 to estimate and remove batch effects from composition data."
        ),
        strengths=[
            "Principled batch correction for compositional data",
            "Removes batch effects before feature selection",
            "Preserves biological signal better than simple standardization",
            "Well-validated method in 16S literature",
        ],
        weaknesses=[
            "Slow (R interop overhead, quantile regression)",
            "Requires rpy2 and R packages (qvalue, ConQuR)",
            "Assumes batch is independent of biological target",
            "May overcorrect if batch truly correlates with target",
        ],
        batch_correction="conqur",
        cv_type="kfold",
        feature_composition="asv_only",
        recommended_datasets=2,
        recommended_samples=500,
        hyperparameter_notes="Run once, cache result. ConQuR adds 10-30 min per dataset.",
        metadata_dependency="Project",
        computational_cost="high",
        recommended_for="Multi-project studies where batch correction is critical",
    ),
    
    "meta_aware": StrategyMetadata(
        name="meta_aware",
        description=(
            "Metadata-aware feature selection using environmental context. "
            "Adds curated environmental metadata columns (lat, lon, elevation, temperature, pH) "
            "as features alongside ASV data during ML pipeline."
        ),
        strengths=[
            "Incorporates spatial/environmental context",
            "Can discover microbe-environment associations",
            "Better for geographically-structured datasets",
            "Enriches feature space with external knowledge",
        ],
        weaknesses=[
            "Requires complete metadata (lat/lon at minimum)",
            "Environmental columns can dominate feature importance",
            "Harder to interpret (many correlated features)",
            "Metadata must be carefully validated/normalized",
        ],
        batch_correction="none",
        cv_type="kfold",
        feature_composition="asv+env",
        recommended_datasets=1,
        recommended_samples=300,
        hyperparameter_notes="May need feature scaling/normalization. CatBoost handles categorical env data.",
        metadata_dependency="lat, lon, elevation, temperature, pH",
        computational_cost="medium",
        recommended_for="Geographically-structured environmental samples",
    ),
    
    "lopocv": StrategyMetadata(
        name="lopocv",
        description=(
            "Leave-One-Project-Out Cross-Validation with batch-aware features. "
            "Each CV fold leaves out one entire project, testing generalization across datasets."
        ),
        strengths=[
            "Strict test of cross-project generalization",
            "Best for validating transferability",
            "Naturally handles multi-project datasets",
            "Prevents data leakage across projects",
        ],
        weaknesses=[
            "Can be very slow with many projects (up to N folds)",
            "Threshold can become too strict with 10+ projects",
            "May produce too few training samples per fold",
            "Requires >2 projects to be meaningful",
        ],
        batch_correction="none",
        cv_type="lopocv",
        feature_composition="asv+batch",
        recommended_datasets=3,
        recommended_samples=500,
        hyperparameter_notes="Switch to GroupKFold if >10 projects (configured in cv_strategy_threshold).",
        metadata_dependency="Project",
        computational_cost="high",
        recommended_for="Strct generalization testing, publication-quality validation",
    ),
    
    "spatial_cv": StrategyMetadata(
        name="spatial_cv",
        description=(
            "Geographic/spatial cross-validation using latitude/longitude coordinates. "
            "Divides samples into spatial regions, testing geographic generalization."
        ),
        strengths=[
            "Tests spatial generalization",
            "Appropriate for ecosystem surveys",
            "Discovers geographically-localized patterns",
            "Natural for environmental/ecological data",
        ],
        weaknesses=[
            "Requires accurate lat/lon coordinates",
            "Sensitive to grid resolution choice",
            "May not align with biological boundaries",
            "Complex to implement robustly",
        ],
        batch_correction="none",
        cv_type="spatial",
        feature_composition="asv+env",
        recommended_datasets=1,
        recommended_samples=300,
        hyperparameter_notes="Requires spatial grid setup. See spatial_cv.py for implementation.",
        metadata_dependency="lat, lon",
        computational_cost="medium",
        recommended_for="Large geographic surveys, ecosystem mapping studies",
    ),
}


def get_strategy_metadata(strategy_name: str) -> StrategyMetadata:
    """Retrieve metadata for a strategy.
    
    Parameters
    ----------
    strategy_name : str
        Strategy identifier (must be in STRATEGY_METADATA keys).
        
    Returns
    -------
    StrategyMetadata
        Metadata object for the strategy.
        
    Raises
    ------
    ValueError
        If strategy_name not found in registry.
    """
    if strategy_name not in STRATEGY_METADATA:
        available = ", ".join(STRATEGY_METADATA.keys())
        raise ValueError(
            f"Strategy '{strategy_name}' not found in registry. "
            f"Available strategies: {available}"
        )
    return STRATEGY_METADATA[strategy_name]


def list_strategies() -> List[str]:
    """Get list of all registered strategy names.
    
    Returns
    -------
    List[str]
        Strategy identifiers in STRATEGY_METADATA.
    """
    return list(STRATEGY_METADATA.keys())


def get_strategies_by_cv_type(cv_type: str) -> List[str]:
    """Filter strategies by cross-validation type.
    
    Parameters
    ----------
    cv_type : str
        CV type: 'kfold', 'groupkfold', 'lopocv', 'spatial'.
        
    Returns
    -------
    List[str]
        Strategy names using the specified CV type.
    """
    return [
        name for name, meta in STRATEGY_METADATA.items()
        if meta.cv_type == cv_type
    ]


def get_strategies_by_batch_correction(batch_method: str) -> List[str]:
    """Filter strategies by batch correction method.
    
    Parameters
    ----------
    batch_method : str
        Batch method: 'none', 'batch_aware', 'conqur'.
        
    Returns
    -------
    List[str]
        Strategy names using the specified batch correction.
    """
    return [
        name for name, meta in STRATEGY_METADATA.items()
        if meta.batch_correction == batch_method
    ]


def validate_config_strategies(strategy_list: List[str]) -> bool:
    """Validate that all strategies in config list are registered.
    
    Parameters
    ----------
    strategy_list : List[str]
        Strategy names from config.ml.grid_settings.fs_strategies.
        
    Returns
    -------
    bool
        True if all strategies found in registry.
        
    Raises
    ------
    ValueError
        If any strategy not found.
    """
    for strategy in strategy_list:
        if strategy not in STRATEGY_METADATA:
            available = ", ".join(STRATEGY_METADATA.keys())
            raise ValueError(
                f"Config strategy '{strategy}' not in registry. "
                f"Available: {available}"
            )
    return True
