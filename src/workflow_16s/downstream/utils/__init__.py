# workflow_16s/downstream/utils/__init__.py

"""
Utility modules for the 16S Downstream Analysis Workflow.
Includes reporting, consolidation, and auxiliary helper functions.
"""

from workflow_16s.utils.io.anndata import (
    get_cfg_value, _clean_numeric_series, _get_file_hash,
    _validate_cached_adata, _sanitize_adata, _sanitize_obs,
    fix_adata_dtypes, _process_single_file, format_bytes,
    safe_write_h5ad, safe_outer_merge, hierarchical_merge,
    create_anndata_from_qiime_artifacts, quick_taxonomy_check,
    validate_anndata_file
    
)
from .adata_biology import (
    clean_metadata, filter_low_depth_and_prevalence,
    filter_samples_and_features, parse_taxonomy
)

from .helpers import AnalysisUtils

from .metadata import (
    filter_by_prevalence, normalize_target_gene, standardize_dates
)
from .tree_handler import (
    TreeHandlingStrategy,  GracefulDegradationStrategy, 
    TreeMergingStrategy, DeNovoTreeBuildingStrategy, 
    PartialAnalysisStrategy, SubsetTreeExtractionStrategy, 
    get_tree_handling_strategy, handle_missing_tree
)

from .performance_optimizer import (
    get_optimal_parameters, subsample_stratified, estimate_runtime
)

from .reporting import generate_synthesis_report

__all__ = [
    'clean_metadata', 'filter_low_depth_and_prevalence', 
    'filter_samples_and_features', 'parse_taxonomy',
    
    'AnalysisUtils', 
    
    'filter_by_prevalence', 'normalize_target_gene', 'standardize_dates',
    
    'TreeHandlingStrategy', 'GracefulDegradationStrategy', 'TreeMergingStrategy', 
    'DeNovoTreeBuildingStrategy', 'PartialAnalysisStrategy', 'SubsetTreeExtractionStrategy', 
    'get_tree_handling_strategy', 'handle_missing_tree', 
    
    'get_optimal_parameters', 'subsample_stratified', 'estimate_runtime',
    
    'generate_synthesis_report', 
]

