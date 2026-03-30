"""Validation Module

Cross-validates functional predictions against multiple data types:
1. Measured metals: Correlate KO-predicted metal resistance/uptake with measurements
2. Metatranscriptomes: Compare predicted KO abundances with gene expression

Provides confidence assessment for functional profiling results.
"""

from .measured_metals import (
    MeasuredMetalValidator,
    MetalValidationResult,
)
from .metatranscriptome_mapping import (
    MetatranscriptomeValidator,
    MetatranscriptomicResult,
    validate_with_metatranscriptome,
)

__all__ = [
    'MeasuredMetalValidator',
    'MetalValidationResult',
    'MetatranscriptomeValidator',
    'MetatranscriptomicResult',
    'validate_with_metatranscriptome',
]
