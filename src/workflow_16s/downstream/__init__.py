"""
Downstream 16S Analysis Workflow Package.

This package provides modules for running a complete downstream analysis,
from data aggregation and QC to diversity, statistical, and ML analysis.
"""

#from workflow_16s.downstream.orchestrator import DownstreamWorkflow
from workflow_16s.downstream.workflow import DownstreamWorkflow
from workflow_16s.downstream.visualization import PlottingUtils

# Expose new scientific analysis modules
from workflow_16s.downstream.qc import (
    identify_contaminants, remove_contaminants,
    decontamination_workflow
)

from workflow_16s.downstream.diversity import (
    load_tree, calculate_faith_pd, calculate_unifrac,
    phylogenetic_diversity_workflow
)

from workflow_16s.downstream.networks import (
    run_spiec_easi, run_sparcc, run_proportionality,
    compare_network_methods, network_analysis_workflow
)

from workflow_16s.downstream.longitudinal import (
    check_temporal_structure, run_zibr,
    run_maaslin2_longitudinal, trajectory_clustering,
    calculate_temporal_stability, longitudinal_analysis_workflow
)

from workflow_16s.downstream.statistics.batch import (
    detect_batch_effects, 
    run_batch_workflow,
    apply_conqur_correction, 
    apply_combat_correction
)

from workflow_16s.downstream.statistics import (
    #detect_batch_effects, run_batch_workflow,
    #apply_conqur_correction, apply_combat_correction,
    run_deseq2, run_corncob, run_linda, run_aldex2, run_wilcoxon,
    compare_da_methods, consensus_da_features,  estimate_permanova_power, 
    estimate_da_power, pilot_data_power_analysis, minimal_detectable_effect, 
    plot_power_curves, power_analysis_report
)

__all__ = [
    "DownstreamWorkflow", "PlottingUtils", "detect_batch_effects",
    "run_batch_workflow", "apply_conqur_correction", "apply_combat_correction",
    "identify_contaminants", "remove_contaminants", "decontamination_workflow",
    "load_tree", "calculate_faith_pd", "calculate_unifrac",
    "phylogenetic_diversity_workflow", "run_deseq2", "run_corncob", "run_linda",
    "run_aldex2", "run_wilcoxon", "compare_da_methods", "consensus_da_features",
    "run_spiec_easi", "run_sparcc", "run_proportionality", "compare_network_methods",
    "network_analysis_workflow", "check_temporal_structure", "run_zibr",
    "run_maaslin2_longitudinal", "trajectory_clustering", "calculate_temporal_stability",
    "longitudinal_analysis_workflow", "estimate_permanova_power", "estimate_da_power",
    "pilot_data_power_analysis", "minimal_detectable_effect", "plot_power_curves",
    "power_analysis_report",
]
