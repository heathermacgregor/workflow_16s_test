# workflow_16s/downstream/utils/__init__.py

"""
Utility modules for the 16S Downstream Analysis Workflow.
Includes reporting, consolidation, and auxiliary helper functions.
"""

from .helpers import AnalysisUtils
from .reporting import generate_synthesis_report

from .adata_utils import (
    fix_adata_dtypes, get_resident_memory_gb, safe_write_h5ad, 
    inspect_adata_dtypes
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

__all__ = [
    'AnalysisUtils', 'fix_adata_dtypes', 'get_resident_memory_gb', 'safe_write_h5ad', 
    'inspect_adata_dtypes', 'generate_synthesis_report', 'TreeHandlingStrategy', 
    'GracefulDegradationStrategy', 'TreeMergingStrategy', 'DeNovoTreeBuildingStrategy', 
    'PartialAnalysisStrategy', 'SubsetTreeExtractionStrategy', 'get_tree_handling_strategy', 
    'handle_missing_tree', 'get_optimal_parameters', 'subsample_stratified', 'estimate_runtime',
]
