# ml_strategy_config.py
"""
ML Strategy Configuration System

Defines named ML pipeline strategies combining:
- Feature Sets: taxonomy-only, +batch, +metadata, all
- Preprocessing: identity, ConQuR  
- CV Methods: k-fold, LOPOCV, spatial
- Filter Policies: none, multi-class-only, variance-filtered

Each strategy is config-driven and can be selected by semantic name.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum


class FeatureSet(Enum):
    """Feature composition for ML models."""
    TAXONOMY_ONLY = "taxonomy_only"         # ASV/OTU data only
    TAXONOMY_BATCH = "taxonomy_batch"       # ASV + batch/project column
    TAXONOMY_METADATA = "taxonomy_metadata" # ASV + environmental metadata (lat, lon, pH, temp, etc.)
    ALL = "all"                             # ASV + batch + full metadata


class PreprocessingMethod(Enum):
    """Data preprocessing approach."""
    IDENTITY = "identity"    # No correction, use rCLR as-is
    CONQUR = "conqur"        # Conditional Quantile Regression batch correction


class CVMethod(Enum):
    """Cross-validation strategy."""
    KFOLD = "kfold"           # Standard k-fold (default 5)
    LOPOCV = "lopocv"         # Leave-One-Project-Out
    SPATIAL = "spatial"       # Spatial cross-validation (if lat/lon available)


class FilterPolicy(Enum):
    """Data filtering/eligibility policy."""
    NONE = "none"                     # No filtering (use all samples)
    MULTICLASS_ONLY = "multiclass"    # Require >=2 classes per study
    VARIANCE_FILTERED = "variance"    # Require non-zero variance in target


@dataclass
class MLStrategyConfig:
    """
    A complete ML strategy definition combining feature set, preprocessing, CV, and filtering.
    """
    
    # Strategy identifier
    name: str
    """Semantic strategy name (e.g., 'taxonomyOnly_Identity_KFold_NoFilter')"""
    
    description: str
    """Human-readable description"""
    
    # Core strategy components
    feature_set: FeatureSet
    """Which features to use (taxonomy, batch, metadata, all)"""
    
    preprocessing: PreprocessingMethod
    """Preprocessing method (identity or ConQuR)"""
    
    cv_method: CVMethod
    """Cross-validation strategy"""
    
    filter_policy: FilterPolicy
    """Data filtering policy"""
    
    # Configuration parameters
    num_features: int = 200
    """Number of final features to select"""
    
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    """CatBoost and other hyperparameters"""
    
    cv_folds: int = 5
    """Number of CV folds (for k-fold)"""
    
    test_size: float = 0.2
    """Proportion of data for testing"""
    
    stratify: bool = True
    """Stratify train/test split by target variable"""
    
    min_samples_per_class: int = 10
    """Minimum samples required per class (for multiclass filter)"""
    
    min_studies: int = 1
    """Minimum number of studies/projects required"""
    
    # Metadata requirements
    required_metadata_cols: List[str] = field(default_factory=list)
    """Required metadata columns (auto-filled based on feature_set)"""
    
    # Computational cost estimate
    estimated_runtime_hours: float = 0.5
    """Estimated runtime (for planning)"""
    
    # Validation
    recommended_min_samples: int = 100
    """Recommended minimum total samples"""
    
    @property
    def should_include_batch(self) -> bool:
        """Whether batch/project column should be included."""
        return self.feature_set in [FeatureSet.TAXONOMY_BATCH, FeatureSet.ALL]
    
    @property
    def should_include_metadata(self) -> bool:
        """Whether environmental metadata should be included."""
        return self.feature_set in [FeatureSet.TAXONOMY_METADATA, FeatureSet.ALL]
    
    @property
    def requires_conqur(self) -> bool:
        """Whether ConQuR batch correction is needed."""
        return self.preprocessing == PreprocessingMethod.CONQUR
    
    @property
    def requires_multiple_studies(self) -> bool:
        """Whether multiple studies are required."""
        return self.cv_method == CVMethod.LOPOCV
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for config serialization."""
        data = asdict(self)
        data['feature_set'] = self.feature_set.value
        data['preprocessing'] = self.preprocessing.value
        data['cv_method'] = self.cv_method.value
        data['filter_policy'] = self.filter_policy.value
        return data
    
    def __str__(self) -> str:
        return f"{self.name} ({self.feature_set.name}, {self.preprocessing.name}, {self.cv_method.name}, {self.filter_policy.name})"


# Pre-defined strategy registry
STRATEGY_REGISTRY: Dict[str, MLStrategyConfig] = {}


def register_strategy(config: MLStrategyConfig) -> None:
    """Register a strategy in the global registry."""
    STRATEGY_REGISTRY[config.name] = config
    STRATEGY_REGISTRY[config.name].required_metadata_cols = _infer_required_metadata(config)


def _infer_required_metadata(config: MLStrategyConfig) -> List[str]:
    """Infer required metadata columns based on feature set and filter policy."""
    required = []
    
    # Feature set requirements
    if config.should_include_batch:
        required.extend(['Project', 'batch_original', 'dataset'])  # One of these
    
    if config.should_include_metadata:
        required.extend(['lat', 'lon', 'latitude', 'longitude', 'LatitudeParsed', 'LongitudeParsed'])
        # Other optional enriched columns will be added if available
    
    # Filter policy requirements
    if config.filter_policy == FilterPolicy.MULTICLASS_ONLY:
        required.append('target_variable')  # Generic (will be specific in run)
    
    if config.filter_policy == FilterPolicy.VARIANCE_FILTERED:
        required.append('target_variable')
    
    # ConQuR requires batch column
    if config.requires_conqur:
        required.extend(['Project', 'batch_original', 'dataset'])
    
    # LOPOCV requires batch column
    if config.cv_method == CVMethod.LOPOCV:
        required.extend(['Project', 'batch_original', 'dataset'])
    
    return list(set(required))  # Remove duplicates


# ============================================================================
# PREDEFINED STRATEGIES - Semantic Naming Convention
# ============================================================================
# Pattern: {FeatureSet}_{Preprocessing}_{CVMethod}_{FilterPolicy}
# Examples:
#   - taxonomyOnly_Identity_KFold_NoFilter
#   - taxonomyBatch_ConQuR_LOPOCV_MultiClass
#   - all_Identity_Spatial_Variance

# --- 1. BASELINE (taxonomy-only, identity, no filtering) ---

register_strategy(MLStrategyConfig(
    name="taxonomyOnly_Identity_KFold_NoFilter",
    description=(
        "Baseline: Standard taxonomy-only features, rCLR preprocessed, "
        "k-fold CV with 5 folds, no data filtering. Fastest, simplest baseline."
    ),
    feature_set=FeatureSet.TAXONOMY_ONLY,
    preprocessing=PreprocessingMethod.IDENTITY,
    cv_method=CVMethod.KFOLD,
    filter_policy=FilterPolicy.NONE,
    num_features=200,
    cv_folds=5,
    test_size=0.2,
    hyperparameters={'learning_rate': 0.03, 'depth': 7, 'iterations': 500},
    estimated_runtime_hours=0.5,
    min_samples_per_class=5,  # Relaxed since no early filtering
))

# --- 2. WITH BATCH AWARENESS (taxonomy + batch column) ---

register_strategy(MLStrategyConfig(
    name="taxonomyBatch_Identity_KFold_NoFilter",
    description=(
        "Batch-aware baseline: Taxonomy + batch/project as explicit feature. "
        "Helps model learn batch patterns. K-fold CV, no filtering."
    ),
    feature_set=FeatureSet.TAXONOMY_BATCH,
    preprocessing=PreprocessingMethod.IDENTITY,
    cv_method=CVMethod.KFOLD,
    filter_policy=FilterPolicy.NONE,
    num_features=250,  # Slightly higher (batch column adds info)
    cv_folds=5,
    test_size=0.2,
    hyperparameters={'learning_rate': 0.02, 'depth': 8, 'iterations': 600},
    estimated_runtime_hours=0.6,
    min_studies=2,
))

# --- 3. WITH CONQUR BATCH CORRECTION ---

register_strategy(MLStrategyConfig(
    name="taxonomyOnly_ConQuR_KFold_NoFilter",
    description=(
        "ConQuR-corrected: Applies ConQuR batch correction before rCLR. "
        "Removes batch effects systematically. K-fold CV, no filtering."
    ),
    feature_set=FeatureSet.TAXONOMY_ONLY,
    preprocessing=PreprocessingMethod.CONQUR,
    cv_method=CVMethod.KFOLD,
    filter_policy=FilterPolicy.NONE,
    num_features=200,
    cv_folds=5,
    test_size=0.2,
    hyperparameters={'learning_rate': 0.03, 'depth': 7, 'iterations': 500},
    estimated_runtime_hours=2.0,  # ConQuR is slow
    min_studies=2,
))

register_strategy(MLStrategyConfig(
    name="taxonomyBatch_ConQuR_KFold_NoFilter",
    description=(
        "ConQuR + batch-aware: Applies ConQuR, then adds batch as feature. "
        "Combines systematic correction with explicit batch learning."
    ),
    feature_set=FeatureSet.TAXONOMY_BATCH,
    preprocessing=PreprocessingMethod.CONQUR,
    cv_method=CVMethod.KFOLD,
    filter_policy=FilterPolicy.NONE,
    num_features=250,
    cv_folds=5,
    test_size=0.2,
    hyperparameters={'learning_rate': 0.025, 'depth': 8, 'iterations': 550},
    estimated_runtime_hours=2.2,
    min_studies=2,
))

# --- 4. WITH METADATA ENRICHMENT ---

register_strategy(MLStrategyConfig(
    name="taxonomyMetadata_Identity_KFold_NoFilter",
    description=(
        "Metadata-enriched: Taxonomy + environmental metadata (lat, lon, pH, temp, etc). "
        "No batch correction. Good for geographically-structured data."
    ),
    feature_set=FeatureSet.TAXONOMY_METADATA,
    preprocessing=PreprocessingMethod.IDENTITY,
    cv_method=CVMethod.KFOLD,
    filter_policy=FilterPolicy.NONE,
    num_features=300,  # More features due to metadata
    cv_folds=5,
    test_size=0.2,
    hyperparameters={'learning_rate': 0.025, 'depth': 8, 'iterations': 600},
    estimated_runtime_hours=0.7,
    recommended_min_samples=200,
))

# --- 5. ALL FEATURES (taxonomy + batch + metadata) ---

register_strategy(MLStrategyConfig(
    name="all_Identity_KFold_NoFilter",
    description=(
        "Kitchen sink: All features (taxonomy, batch, metadata). "
        "No preprocessing. K-fold CV. Highest complexity, risk of overfitting."
    ),
    feature_set=FeatureSet.ALL,
    preprocessing=PreprocessingMethod.IDENTITY,
    cv_method=CVMethod.KFOLD,
    filter_policy=FilterPolicy.NONE,
    num_features=400,
    cv_folds=5,
    test_size=0.2,
    hyperparameters={'learning_rate': 0.02, 'depth': 9, 'iterations': 700},
    estimated_runtime_hours=1.2,
    recommended_min_samples=500,
    min_studies=2,
))

# --- 6. LEAVE-ONE-PROJECT-OUT (LOPOCV) STRATEGIES ---

register_strategy(MLStrategyConfig(
    name="taxonomyBatch_Identity_LOPOCV_NoFilter",
    description=(
        "Project-generalization test: LOPOCV with batch as feature. "
        "Each fold holds out one entire project. Best for cross-project robustness."
    ),
    feature_set=FeatureSet.TAXONOMY_BATCH,
    preprocessing=PreprocessingMethod.IDENTITY,
    cv_method=CVMethod.LOPOCV,
    filter_policy=FilterPolicy.NONE,
    num_features=250,
    test_size=0.2,
    hyperparameters={'learning_rate': 0.025, 'depth': 8, 'iterations': 600},
    estimated_runtime_hours=3.0,  # LOPOCV can be slow
    min_studies=3,
))

register_strategy(MLStrategyConfig(
    name="taxonomyBatch_ConQuR_LOPOCV_NoFilter",
    description=(
        "Project-generalization with batch correction: LOPOCV + ConQuR. "
        "Strictest validation across projects with systematic batch removal."
    ),
    feature_set=FeatureSet.TAXONOMY_BATCH,
    preprocessing=PreprocessingMethod.CONQUR,
    cv_method=CVMethod.LOPOCV,
    filter_policy=FilterPolicy.NONE,
    num_features=275,
    test_size=0.2,
    hyperparameters={'learning_rate': 0.02, 'depth': 8, 'iterations': 600},
    estimated_runtime_hours=5.0,  # ConQuR + LOPOCV is slowest
    min_studies=3,
))

# --- 7. WITH MULTICLASS-ONLY FILTERING ---

register_strategy(MLStrategyConfig(
    name="taxonomyOnly_Identity_KFold_MultiClass",
    description=(
        "Filtered baseline: Taxonomy-only with multiclass filtering. "
        "Only studies with >=2 classes (>=10 per minority) are kept. "
        "Ensures balanced, trainable targets."
    ),
    feature_set=FeatureSet.TAXONOMY_ONLY,
    preprocessing=PreprocessingMethod.IDENTITY,
    cv_method=CVMethod.KFOLD,
    filter_policy=FilterPolicy.MULTICLASS_ONLY,
    num_features=200,
    cv_folds=5,
    test_size=0.2,
    hyperparameters={'learning_rate': 0.03, 'depth': 7, 'iterations': 500},
    estimated_runtime_hours=0.5,
    min_samples_per_class=10,
    recommended_min_samples=150,
))

register_strategy(MLStrategyConfig(
    name="taxonomyBatch_Identity_KFold_MultiClass",
    description=(
        "Batch-aware + multiclass filter: Enables robust multi-study analysis "
        "with class balance requirements."
    ),
    feature_set=FeatureSet.TAXONOMY_BATCH,
    preprocessing=PreprocessingMethod.IDENTITY,
    cv_method=CVMethod.KFOLD,
    filter_policy=FilterPolicy.MULTICLASS_ONLY,
    num_features=250,
    cv_folds=5,
    test_size=0.2,
    hyperparameters={'learning_rate': 0.025, 'depth': 8, 'iterations': 550},
    estimated_runtime_hours=0.6,
    min_samples_per_class=10,
    min_studies=2,
    recommended_min_samples=200,
))

register_strategy(MLStrategyConfig(
    name="all_Identity_KFold_MultiClass",
    description=(
        "All features + class balance: Richest feature set with strict "
        "multiclass filtering. Highest information, highest sample requirements."
    ),
    feature_set=FeatureSet.ALL,
    preprocessing=PreprocessingMethod.IDENTITY,
    cv_method=CVMethod.KFOLD,
    filter_policy=FilterPolicy.MULTICLASS_ONLY,
    num_features=400,
    cv_folds=5,
    test_size=0.2,
    hyperparameters={'learning_rate': 0.02, 'depth': 9, 'iterations': 700},
    estimated_runtime_hours=1.2,
    min_samples_per_class=10,
    min_studies=2,
    recommended_min_samples=500,
))

# --- 8. WITH VARIANCE FILTERING (for continuous targets) ---

register_strategy(MLStrategyConfig(
    name="taxonomyOnly_Identity_KFold_Variance",
    description=(
        "Taxonomy-only + variance filter: For continuous targets. "
        "Filters to studies where target has non-zero variance."
    ),
    feature_set=FeatureSet.TAXONOMY_ONLY,
    preprocessing=PreprocessingMethod.IDENTITY,
    cv_method=CVMethod.KFOLD,
    filter_policy=FilterPolicy.VARIANCE_FILTERED,
    num_features=200,
    cv_folds=5,
    test_size=0.2,
    hyperparameters={'learning_rate': 0.03, 'depth': 7, 'iterations': 500},
    estimated_runtime_hours=0.5,
    min_samples_per_class=10,
))

register_strategy(MLStrategyConfig(
    name="taxonomyMetadata_Identity_KFold_Variance",
    description=(
        "Metadata-enriched + variance filter: For geospatial continuous targets. "
        "Includes lat/lon and environmental data; filters by variance."
    ),
    feature_set=FeatureSet.TAXONOMY_METADATA,
    preprocessing=PreprocessingMethod.IDENTITY,
    cv_method=CVMethod.KFOLD,
    filter_policy=FilterPolicy.VARIANCE_FILTERED,
    num_features=300,
    cv_folds=5,
    test_size=0.2,
    hyperparameters={'learning_rate': 0.025, 'depth': 8, 'iterations': 600},
    estimated_runtime_hours=0.7,
    min_samples_per_class=10,
    recommended_min_samples=200,
))

# --- 9. SPATIAL CV (if coordinates available) ---

register_strategy(MLStrategyConfig(
    name="taxonomyMetadata_Identity_Spatial_MultiClass",
    description=(
        "Spatial cross-validation: Splits samples geographically (lat/lon) "
        "to test generalization across space. Requires coordinates and multiclass target."
    ),
    feature_set=FeatureSet.TAXONOMY_METADATA,
    preprocessing=PreprocessingMethod.IDENTITY,
    cv_method=CVMethod.SPATIAL,
    filter_policy=FilterPolicy.MULTICLASS_ONLY,
    num_features=300,
    cv_folds=5,  # Geographic regions instead of random folds
    test_size=0.2,
    hyperparameters={'learning_rate': 0.025, 'depth': 8, 'iterations': 600},
    estimated_runtime_hours=0.8,
    min_samples_per_class=15,
    recommended_min_samples=300,
))


def get_strategy(name: str) -> Optional[MLStrategyConfig]:
    """
    Retrieve a strategy by name.
    
    Args:
        name: Strategy name (e.g., 'taxonomyOnly_Identity_KFold_NoFilter')
        
    Returns:
        MLStrategyConfig or None if not found
    """
    return STRATEGY_REGISTRY.get(name)


def list_strategies(
    feature_set: Optional[FeatureSet] = None,
    preprocessing: Optional[PreprocessingMethod] = None,
    cv_method: Optional[CVMethod] = None,
    filter_policy: Optional[FilterPolicy] = None,
) -> Dict[str, MLStrategyConfig]:
    """
    List strategies with optional filtering.
    
    Args:
        feature_set: Filter by feature set
        preprocessing: Filter by preprocessing method
        cv_method: Filter by CV method
        filter_policy: Filter by filter policy
        
    Returns:
        Dict of matching strategies
    """
    results = STRATEGY_REGISTRY.copy()
    
    if feature_set:
        results = {k: v for k, v in results.items() if v.feature_set == feature_set}
    if preprocessing:
        results = {k: v for k, v in results.items() if v.preprocessing == preprocessing}
    if cv_method:
        results = {k: v for k, v in results.items() if v.cv_method == cv_method}
    if filter_policy:
        results = {k: v for k, v in results.items() if v.filter_policy == filter_policy}
    
    return results


def print_strategy_catalog() -> str:
    """Generate a human-readable catalog of all strategies."""
    lines = [
        "=" * 100,
        "ML STRATEGY CATALOG",
        "=" * 100,
        ""
    ]
    
    for name, strategy in sorted(STRATEGY_REGISTRY.items()):
        lines.extend([
            f"📋 {name}",
            f"   Description: {strategy.description}",
            f"   Spec: {strategy.feature_set.name} | {strategy.preprocessing.name} | {strategy.cv_method.name} | {strategy.filter_policy.name}",
            f"   Features: {strategy.num_features} | CV Folds: {strategy.cv_folds} | Est. Runtime: ~{strategy.estimated_runtime_hours}h",
            f"   Min Samples: {strategy.recommended_min_samples} | Min Studies: {strategy.min_studies}",
            ""
        ])
    
    lines.append("=" * 100)
    return "\n".join(lines)
