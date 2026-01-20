"""
Quality Control Module for workflow_16s

This module provides comprehensive QC and validation for:
- Metadata integrity and consistency
- Primer detection and validation
- Sample identity verification
- Contamination detection
- External data validation

Quick start:
    >>> from workflow_16s.qc import quick_qc
    >>> adata_clean = quick_qc(adata, output_dir='qc_results')
"""

from .metadata_validator import MetadataValidator, ENVOOntology
from .primer_qc import PrimerQC
from .sample_validator import SampleIdentityValidator
from .contamination_enhanced import (
    detect_contaminants_reference_based,
    detect_cross_sample_contamination,
    remove_contaminants_enhanced
)
from .pipeline import ComprehensiveQC, quick_qc
from .validation import (
    validate_config,
    validate_metadata,
    validate_adata,
    check_dependencies,
    QCValidationError,
    QCDependencyError
)

__all__ = [
    'MetadataValidator',
    'ENVOOntology',
    'PrimerQC',
    'SampleIdentityValidator',
    'detect_contaminants_reference_based',
    'detect_cross_sample_contamination',
    'remove_contaminants_enhanced',
    'ComprehensiveQC',
    'quick_qc',
]
