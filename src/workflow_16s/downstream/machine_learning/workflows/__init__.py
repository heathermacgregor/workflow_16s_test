# src/workflow_16s/downstream/machine_learning/workflows/__init__.py
"""
Machine Learning Orchestration Workflows
========================================
This package contains the high-level 'Brain' scripts that coordinate 
feature selection, model training, and forensic validation.
"""

# We use absolute imports to specific functions to allow the smoke test 
# and main analysis suite to find the primary entry points.
from .feature_selection import run_catboost_selection

__all__ = [
    'run_catboost_selection'
]