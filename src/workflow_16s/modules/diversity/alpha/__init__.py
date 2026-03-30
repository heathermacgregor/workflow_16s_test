"""
Alpha diversity analysis module.

Provides functions for calculating within-sample diversity metrics,
rarefaction curves, and associations with metadata.
"""

from .alpha import (
    run_alpha_diversity,
)
from .rarefaction import (
    generate_rarefaction_curves,
    calculate_rarefaction_curve
)

__all__ = [
    'run_alpha_diversity'
    'generate_rarefaction_curves',
    'calculate_rarefaction_curve',
]
