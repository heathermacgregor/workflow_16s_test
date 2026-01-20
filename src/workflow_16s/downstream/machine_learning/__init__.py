"""
Machine learning module for microbiome analysis.

Provides nested cross-validation and advanced feature selection methods.
"""

# Load from machine_learning.py module explicitly to avoid package conflict
import importlib.util
from pathlib import Path

_ml_py = Path(__file__).parent.parent / "machine_learning.py"
_spec = importlib.util.spec_from_file_location("workflow_16s.downstream._machine_learning_module", _ml_py)
_ml_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ml_module)

run_machine_learning_analysis = _ml_module.run_machine_learning_analysis
run_catboost_selection = _ml_module.run_catboost_selection

from .nested_cv import (
    nested_cross_validation,
    compare_with_simple_cv
)

__all__ = [
    'run_machine_learning_analysis',
    'run_catboost_selection',
    'nested_cross_validation',
    'compare_with_simple_cv',
]
