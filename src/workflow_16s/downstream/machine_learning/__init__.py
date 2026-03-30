# src/workflow_16s/downstream/machine_learning/__init__.py
"""
16S Machine Learning Discovery & Forensic Suite
===============================================
A tiered architecture for robust biomarker discovery, batch effect mitigation,
and cross-study consensus validation.
"""

from .workflows.standard_ml import run_machine_learning_analysis
from .workflows.feature_selection import run_catboost_selection
from .main import run_soil_prediction_suite
from .Project_Discovery_Dashboard import DiscoveryDashboardGenerator

# Phase 2: ML Strategy Configuration System
from .ml_strategy_config import (
    MLStrategyConfig,
    FeatureSet,
    PreprocessingMethod,
    CVMethod,
    FilterPolicy,
    STRATEGY_REGISTRY,
    get_strategy,
    list_strategies,
    print_strategy_catalog,
)

from .ml_strategy_orchestrator import MLStrategyOrchestrator

# Phase 3: ML Strategy Integration & Validation
from .ml_strategy_integration import (
    StrategyExecutor,
    run_strategies_from_config,
)

from .ml_strategy_validation import (
    StrategyValidator,
    StrategySmokeTester,
)

from .batch_control import (
    run_ml_with_batch_control, 
    create_comparison_plots,
    create_summary_report, 
    train_batch_residual_model,
    audit_biomarker_confidence,
    create_confounding_heatmap
)

from .feature_selection import (
    catboost_feature_selection,
    perform_feature_selection
)

#from .nuclear_fuel_cycle import (
#    run_facility_microbe_report
#)

from .meta_analysis import (
    perform_meta_analysis, 
    apply_meta_consensus_weighting, 
    generate_study_overlap_matrix
)

from .validation import (
    StudyEligibilityManager,
    run_comprehensive_validation,
    run_shuffle_baseline,
    validate_consensus_panel,
    BiomarkerAuditor,
    verify_run
)

from .visualization import (
    plot_shap,
    plot_feature_importances,
    plot_robustness_vs_importance,
    plot_batch_dependency,
    generate_stability_comparison,
    plot_shuffle_test,
    create_strategy_comparison_dashboard,
    generate_comprehensive_ml_report,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_predicted_vs_actual,
    plot_residuals
)

from .utils import (
    clean_feature_names, 
    resolve_feature_names, 
    align_data_robust, 
    apply_batch_centered_clr
)
from workflow_16s.downstream.machine_learning.nuclear_fuel_cycle.facility_taxa_reporter import run_facility_microbe_report

__all__ = [
    'run_machine_learning_analysis',
    'run_ml_with_batch_control',
    'create_comparison_plots',
    'create_summary_report',
    'train_batch_residual_model',
    'audit_biomarker_confidence',
    'create_confounding_heatmap',
    'run_catboost_selection',
    'run_soil_prediction_suite',
    'DiscoveryDashboardGenerator',
    'run_facility_microbe_report',  
    'StudyEligibilityManager',
    'run_comprehensive_validation',
    'run_shuffle_baseline',
    'validate_consensus_panel',
    'BiomarkerAuditor',
    'verify_run',
    'plot_shap',
    'plot_feature_importances',
    'plot_robustness_vs_importance',
    'plot_shuffle_test',
    'plot_batch_dependency',
    'generate_stability_comparison',
    'create_strategy_comparison_dashboard',
    'generate_comprehensive_ml_report',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_predicted_vs_actual',
    'plot_residuals',
    'catboost_feature_selection',
    'perform_feature_selection',
    'clean_feature_names',
    'resolve_feature_names',
    'align_data_robust',
    'apply_batch_centered_clr'
]