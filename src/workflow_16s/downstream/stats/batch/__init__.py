"""
Batch Effect Management Submodule.

Includes tools for:
- Diagnostics: PERMANOVA, Silhouette analysis, PCA variance.
- Correction: ConQuR, ComBat, Percentile Normalization.
- Visualization: Interactive PCA, Heatmaps.
"""

from .workflow import run_batch_workflow
from .diagnostics import detect_batch_effects
from .correction import (
    apply_conqur_correction,
    apply_combat_correction,
    percentile_normalization,
    add_batch_as_covariate
)
from .visualization import (
    visualize_batch_effects,
    plot_batch_pca_interactive,
    plot_silhouette_analysis,
    plot_batch_heatmap
)

__all__ = [
    'run_batch_workflow',
    'detect_batch_effects',
    'apply_conqur_correction',
    'apply_combat_correction',
    'percentile_normalization',
    'add_batch_as_covariate',
    'visualize_batch_effects',
    'plot_batch_pca_interactive',
    'plot_silhouette_analysis',
    'plot_batch_heatmap'
]