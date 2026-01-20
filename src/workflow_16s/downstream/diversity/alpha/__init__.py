"""
Alpha diversity analysis module.

Provides functions for calculating within-sample diversity metrics,
rarefaction curves, and associations with metadata.
"""

from .rarefaction import (
    generate_rarefaction_curves,
    calculate_rarefaction_curve
)

__all__ = [
    'generate_rarefaction_curves',
    'calculate_rarefaction_curve',
]
