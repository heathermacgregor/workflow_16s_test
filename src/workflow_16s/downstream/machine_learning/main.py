# src/workflow_16s/downstream/machine_learning/main.py

from .batch_control import (
    run_ml_with_batch_control, 
    prepare_batch_covariates,
    detect_confounding,
    plot_confounding_heatmap,
    train_batch_residual_model,
    create_comparison_plots
)
from .feature_selection import (
    catboost_feature_selection, 
    filter_data
)
from .meta_analysis import (
    perform_meta_analysis, 
    apply_meta_consensus_weighting, 
    generate_study_overlap_matrix
)
from .validation import (
    run_shuffle_baseline, 
    validate_consensus_panel
)
from .validation.overfitting_prevention import (
    run_comprehensive_validation
)
from .visualization import (
    generate_comprehensive_ml_report, 
    plot_feature_importances
)
from .utils import (
    clean_feature_names, 
    resolve_feature_names, 
    align_data_robust, 
    apply_batch_centered_clr,
    optimize_threshold,
    verify_model_outputs
)

# Main workflow functions
from .constants import EXPECTED_VAR_COLUMNS, MANDATORY_METADATA
from .workflows.feature_selection import run_catboost_selection
from .workflows.standard_ml import run_machine_learning_analysis
from .workflows.soil_prediction import run_soil_prediction_suite