"""
Beta diversity analysis module.

Provides functions for calculating distance matrices, performing ordination,
and testing associations with metadata.
"""

from .ordination import (
    run_beta_diversity_and_stats,
    run_constrained_ordination,
    run_trajectory_analysis
)
from .dispersion import (
    run_permdisp,
    check_permanova_validity
)

__all__ = [
    'run_beta_diversity_and_stats',
    'run_constrained_ordination',
    'run_trajectory_analysis',
    'run_permdisp',
    'check_permanova_validity',
]