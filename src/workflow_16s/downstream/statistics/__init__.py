"""
Statistics module for microbiome analysis.

Provides effect size calculations, multiple testing correction,
and differential abundance testing methods.
"""

from .effect_sizes import (
    cohens_d,
    cliffs_delta,
    glass_delta,
    hedges_g,
    calculate_all_effect_sizes,
    effect_size_with_stats
)

from .multiple_testing import (
    apply_multiple_testing_correction,
    compare_correction_methods,
    stratified_fdr_correction,
    export_fdr_results
)

from .differential_abundance import (
    ancom_bc_wrapper,
    simple_compositional_da
)

__all__ = [
    # Effect sizes
    'cohens_d',
    'cliffs_delta',
    'glass_delta',
    'hedges_g',
    'calculate_all_effect_sizes',
    'effect_size_with_stats',
    
    # Multiple testing
    'apply_multiple_testing_correction',
    'compare_correction_methods',
    'stratified_fdr_correction',
    'export_fdr_results',
    
    # Differential abundance
    'ancom_bc_wrapper',
    'simple_compositional_da',
]
