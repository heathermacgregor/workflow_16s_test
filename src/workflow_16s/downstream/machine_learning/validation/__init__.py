# src/workflow_16s/downstream/machine_learning/validation/__init__.py
"""
workflow_16s.downstream.machine_learning.validation package

The Scientific Integrity Layer of the 16S Discovery Pipeline.
Provides a 4-tier validation stack:
1. Eligibility: Pre-modeling checks for cohort power and target variance.
2. Internal Audit: Nested CV and Learning Curves to detect overfitting.
3. Statistical Significance: Permutation testing for empirical p-values.
4. Generalization: LOPOCV for universal biomarker validation.
"""

from .check_study_eligibility import StudyEligibilityManager
from .overfitting_prevention import run_comprehensive_validation
from .validation import run_shuffle_baseline, validate_consensus_panel
from .quality_audit import BiomarkerAuditor, verify_run
from .utils import clean_feature_names, resolve_feature_names, format_audit_results # Added these

__all__ = [
    'StudyEligibilityManager',
    'run_comprehensive_validation',
    'run_shuffle_baseline',
    'validate_consensus_panel',
    'BiomarkerAuditor',
    'verify_run',
    'clean_feature_names',
    'resolve_feature_names',
    'format_audit_results'
]