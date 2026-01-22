"""
Longitudinal Analysis Submodule.

Tools for analyzing microbiome time-series data:
- Statistical Modeling: ZIBR, MaAsLin 2
- Pattern Detection: Trajectory clustering, Temporal stability
- Visualization: Spaghetti plots, Trend lines
"""

from .analysis import run_zibr, run_maaslin2_longitudinal
from .temporal import (
    check_temporal_structure, 
    trajectory_clustering, 
    calculate_temporal_stability
)
from .visualization import plot_temporal_trajectories
from .workflow import longitudinal_analysis_workflow

__all__ = [
    'run_zibr',
    'run_maaslin2_longitudinal',
    'check_temporal_structure',
    'trajectory_clustering',
    'calculate_temporal_stability',
    'plot_temporal_trajectories',
    'longitudinal_analysis_workflow'
]