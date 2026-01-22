"""
workflow_16s.downstream.machine_learning package
"""

from .main import (
    prepare_batch_covariates, detect_confounding, plot_confounding_heatmap,
    train_batch_residual_model, plot_feature_importances,
    run_machine_learning_analysis, run_catboost_selection
)

from .batch_control import (
    run_ml_with_batch_control, create_comparison_plots,
    create_summary_report
)

from .feature_selection import (
    catboost_feature_selection, grid_search, filter_data, check_for_data_leakage,
    perform_feature_selection, generate_shap_report, save_feature_importances
)

from .visualization import (
    create_strategy_comparison_dashboard, create_group_fingerprint_comparison,
    create_multi_group_comparison_heatmap, create_batch_effect_impact_plot,
    generate_comprehensive_ml_report
)

__all__ = [ 
    'prepare_batch_covariates', 'detect_confounding', 'plot_confounding_heatmap',
    'train_batch_residual_model', 'plot_feature_importances',
    'run_machine_learning_analysis', 'run_catboost_selection',
    'create_strategy_comparison_dashboard', 'create_group_fingerprint_comparison',
    'create_multi_group_comparison_heatmap', 'create_batch_effect_impact_plot',
    'generate_comprehensive_ml_report', 'catboost_feature_selection',
    'grid_search', 'filter_data', 'check_for_data_leakage',
    'perform_feature_selection', 'generate_shap_report', 'save_feature_importances'
]
