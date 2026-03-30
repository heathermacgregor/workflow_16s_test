"""
Compositional Network Analysis Module.

Methods for inferring and visualizing microbial interaction networks 
from compositional data (16S/Metagenomics).
"""

from .inference import (
    run_spiec_easi, 
    run_sparcc, 
    run_proportionality, 
    compare_network_methods
)
from .visualization import plot_network
from .workflow import network_analysis_workflow

__all__ = [
    'run_spiec_easi',
    'run_sparcc',
    'run_proportionality',
    'compare_network_methods',
    'plot_network',
    'network_analysis_workflow'
]