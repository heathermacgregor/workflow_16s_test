# src/workflow_16s/downstream/machine_learning/visualization/__init__.py

from .features import generate_comprehensive_ml_report
from .validation_plots import plot_feature_importances, plot_shuffle_test, generate_stability_comparison
from .batch_dependency import plot_batch_dependency
from .shap_plots import plot_shap
from .evaluation_plots import (
    plot_confusion_matrix, plot_roc_curve, 
    plot_precision_recall_curve, plot_predicted_vs_actual, plot_residuals
)
from .visualization import create_strategy_comparison_dashboard, plot_robustness_vs_importance
from .interpretable_plots import InterpretablePlots, create_interpretable_plots
from .interpretability_integration import (
    generate_model_interpretability_plots,
    integrate_plots_with_ml_pipeline
)

__all__ = [
    'generate_comprehensive_ml_report',
    'plot_feature_importances',
    'plot_shuffle_test',
    'generate_stability_comparison',
    'plot_batch_dependency',
    'plot_shap',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_predicted_vs_actual',
    'plot_residuals',
    'create_strategy_comparison_dashboard',
    'plot_robustness_vs_importance',
    'InterpretablePlots',
    'create_interpretable_plots',
    'generate_model_interpretability_plots',
    'integrate_plots_with_ml_pipeline'
]