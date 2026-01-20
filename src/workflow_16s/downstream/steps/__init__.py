"""Steps module for orchestrating the downstream analysis workflow.

This module provides high-level functions for executing different stages of the
microbial community analysis pipeline, including:
- Data ingestion and preprocessing
- Metadata backfilling from external APIs
- Comprehensive diversity and statistical analyses
- Results synthesis and reporting
"""

from .ingestion import (
    run_fast_load,
    run_filter_empty,
    find_conda_env_by_substring
)
from .preprocessing import run_preprocessing_pipeline
from .backfill import run_data_backfill
from .analysis import run_analysis_suite
from .synthesis import (
    run_results_synthesis,
    handle_strategy_impact_plot
)

# Import QC functions (optional dependency)
try:
    from .qc import run_comprehensive_qc, run_semantic_filtering
    QC_AVAILABLE = True
except ImportError:
    QC_AVAILABLE = False
    run_comprehensive_qc = None
    run_semantic_filtering = None

__all__ = [
    # Ingestion
    'run_fast_load',
    'run_filter_empty',
    'find_conda_env_by_substring',
    # Preprocessing
    'run_preprocessing_pipeline',
    # QC (optional)
    'run_comprehensive_qc',
    'run_semantic_filtering',
    # Backfill
    'run_data_backfill',
    # Analysis
    'run_analysis_suite',
    # Synthesis
    'run_results_synthesis',
    'handle_strategy_impact_plot',
]