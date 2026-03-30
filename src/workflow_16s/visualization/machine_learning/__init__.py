# workflow_16s/visualization/machine_learning/__init__.py

from .batch_dependency import plot_batch_dependency
from .evaluation import (
    plot_confusion_matrix, plot_roc_curve, 
    plot_precision_recall_curve, plot_predicted_vs_actual, plot_residuals
)
from .features import generate_comprehensive_ml_report, generate_study_overlap_matrix
from .shap import plot_shap
from .strategy_comparison import (
    create_strategy_comparison_dashboard, 
    plot_robustness_vs_importance
)
from .validation import (
    plot_feature_importances, plot_shuffle_test, 
    generate_stability_comparison
)

__all__ = [
    'generate_comprehensive_ml_report',
    'generate_study_overlap_matrix',
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
    'plot_robustness_vs_importance'
]