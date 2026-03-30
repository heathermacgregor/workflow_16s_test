"""
Statistics module for microbiome analysis.

Provides effect size calculations, multiple testing correction,
and differential abundance testing methods.
"""

from .effect_sizes import (
    cohens_d, cliffs_delta, glass_delta, hedges_g,
    calculate_all_effect_sizes, effect_size_with_stats,
    effect_size_confidence_interval, interpret_effect_size,
    interpret_cliffs_delta, interpret_cohens_d,
    log2_fold_change
)

from .multiple_testing import (
    apply_multiple_testing_correction, compare_correction_methods,
    stratified_fdr_correction, export_fdr_results
)

from .differential_abundance import (
    consensus_da_features, compare_da_methods, run_linda, run_aldex2,
    run_deseq2, run_ancombc, run_wilcoxon, run_corncob, _check_r_package
)

from .enhanced_stats import (
    add_effect_sizes_to_stats, check_and_correct_batch_effects,
    validate_sequencing_depth, create_differential_abundance_plots,
    enhanced_differential_abundance_workflow, quick_effect_size_report
)

from .batch import (
    detect_batch_effects, run_batch_workflow
)
#from .batch_effects import (
#    detect_batch_effects, _interpret_batch_results, plot_batch_pca,
#    plot_silhouette_analysis, plot_batch_heatmap,
#    apply_conqur_correction, apply_combat_correction,
#    batch_effect_workflow#
#)

#from .batch_correction import (
#    detect_batch_effects as detect_batch_effects_bc,
#    percentile_normalization, add_batch_as_covariate,
#    visualize_batch_effects, conqur_batch_correction
#)

from .power_analysis import (
    estimate_permanova_power, estimate_da_power, generate_power_report,
    pilot_data_power_analysis, plot_power_curves, minimal_detectable_effect,
    power_analysis_report
)

from .permutation_tests import (
    permutation_ttest, permutation_ftest, permutation_test_features,
    maxt_correction, permanova, _calculate_pseudo_F,
    compare_permutation_vs_parametric
)

__all__ = [
    # Effect sizes
    'cohens_d', 'cliffs_delta', 'glass_delta', 'hedges_g',
    'calculate_all_effect_sizes', 'effect_size_with_stats',
    'effect_size_confidence_interval', 'interpret_effect_size',
    'interpret_cliffs_delta', 'interpret_cohens_d',
    'log2_fold_change',
    # Multiple testing
    'apply_multiple_testing_correction', 'compare_correction_methods',
    'stratified_fdr_correction', 'export_fdr_results',
    # Differential abundance
    'consensus_da_features', 'compare_da_methods', 'run_linda',
    'run_aldex2', 'run_deseq2', 'run_ancombc', 'run_wilcoxon',
    'run_corncob', '_check_r_package',
    # Enhanced statistics
    'add_effect_sizes_to_stats', 'check_and_correct_batch_effects',
    'validate_sequencing_depth', 'create_differential_abundance_plots',
    'enhanced_differential_abundance_workflow', 'quick_effect_size_report',
    # Batch effects
    'detect_batch_effects', 'run_batch_workflow', 'apply_conqur_correction',
    'apply_combat_correction',
    #'detect_batch_effects', '_interpret_batch_results',  'plot_batch_pca',
    #'plot_silhouette_analysis', 'plot_batch_heatmap', 'apply_conqur_correction',
    #'apply_combat_correction', 'batch_effect_workflow',
    # Batch correction
    #'detect_batch_effects_bc', 'percentile_normalization', 'add_batch_as_covariate',
    #'visualize_batch_effects', 'conqur_batch_correction',
    # Power analysis
    'estimate_permanova_power', 'estimate_da_power', 'generate_power_report',
    'pilot_data_power_analysis', 'plot_power_curves', 'minimal_detectable_effect',
    'power_analysis_report',
    # Permutation tests
    'permutation_ttest', 'permutation_ftest', 'permutation_test_features',
    'maxt_correction', 'permanova', '_calculate_pseudo_F',
    'compare_permutation_vs_parametric'
]
