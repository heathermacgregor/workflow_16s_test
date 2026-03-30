"""
workflow_16s.downstream.machine_learning.batch_control package

This package provides a comprehensive toolkit for detecting, quantifying, 
and mitigating technical batch effects in microbiome machine learning models.
"""

from .batch_control import (
    run_ml_with_batch_control, 
    create_comparison_plots,
    create_summary_report, 
    train_batch_residual_model,
    audit_biomarker_confidence,
    create_confounding_heatmap
)
from .covariates import (
    prepare_batch_covariates,
    calculate_batch_importance
)
from .confounding import (
    detect_confounding, 
    plot_confounding_heatmap
)

__all__ = [ 
    'run_ml_with_batch_control', 
    'create_comparison_plots',
    'create_summary_report', 
    'train_batch_residual_model', 
    'audit_biomarker_confidence',
    'create_confounding_heatmap',
    'prepare_batch_covariates', 
    'calculate_batch_importance',
    'detect_confounding', 
    'plot_confounding_heatmap'
]