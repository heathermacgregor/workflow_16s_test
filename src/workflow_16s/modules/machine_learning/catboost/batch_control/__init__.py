# workflow_16s/modules/machine_learning/catboost/batch_control/__init__.py

from .batch_control import (
    audit_biomarker_confidence,
    create_summary_report,
    run_ml_with_batch_control,
    train_batch_residual_model
)
from .confounding import detect_confounding
from .covariates import (
    calculate_batch_importance,
    prepare_batch_covariates
)

# Bridge the gap to the visualization module
from workflow_16s.visualization.machine_learning.batch_dependency import create_confounding_heatmap

__all__ = [
    'audit_biomarker_confidence',
    'create_summary_report',
    'run_ml_with_batch_control',
    'train_batch_residual_model',
    'detect_confounding',
    'calculate_batch_importance',
    'prepare_batch_covariates',
    'create_confounding_heatmap'
]
