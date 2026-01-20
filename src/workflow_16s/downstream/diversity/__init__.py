# ==================================================================================== #
# diversity/__init__.py
# ==================================================================================== #

# Note: alpha.py module (not alpha/ package) contains run_alpha_diversity
import sys
import importlib.util
from pathlib import Path

# Load alpha.py explicitly to avoid conflict with alpha/ package
_alpha_py = Path(__file__).parent / "alpha.py"
_spec = importlib.util.spec_from_file_location("workflow_16s.downstream.diversity._alpha_module", _alpha_py)
_alpha_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_alpha_module)
run_alpha_diversity = _alpha_module.run_alpha_diversity

from .beta import run_beta_diversity_and_stats, run_constrained_ordination, run_trajectory_analysis
from .clustering import run_community_state_typing
from .statistics import run_taxa_metadata_statistics
from .network import run_network_analysis

__all__ = [
    'run_alpha_diversity',
    'run_beta_diversity_and_stats',
    'run_constrained_ordination',
    'run_trajectory_analysis',
    'run_community_state_typing',
    'run_taxa_metadata_statistics',
    'run_network_analysis',
]