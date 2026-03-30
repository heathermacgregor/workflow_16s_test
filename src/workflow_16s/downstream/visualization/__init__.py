""""""

from .dashboards import create_integrated_dashboard, create_qc_aware_diversity_dashboard
from workflow_16s.visualization.utils import PlottingUtils
from .plotting import (
    setup_plotting_theme, #PlottingUtils,
    create_custom_legend_annotations,
    plot_stacked_bar, plot_metadata_pairplot,
    plot_metadata_correlation_heatmap,
    plot_sample_facility_map, plot_sample_taxon_map
)

from .result_export import (
    export_results_to_excel,
    export_publication_tables,
    export_supplementary_data,
    create_methods_section,
    export_complete_results_package
)

from .volcano_plots import (
    create_volcano_plot, create_ma_plot,
    effect_size_volcano
)

from .sample_metadata import (
    plot_sample_distribution,
    plot_metadata_heatmap,
    plot_metadata_summary_table,
    create_geographic_map
)

from .quality_control_suite import (
    run_qc_suite,
    create_taxa_abundance_overview,
    create_feature_sparsity_plot,
    create_sample_sequencing_depth_histogram
)

# Publication-ready defaults (equivalent to ~300 DPI at typical sizes)
DEFAULT_HEIGHT = 800  # Optimized for 8" height at 100 DPI
DEFAULT_WIDTH = 1200  # Optimized for 12" width at 100 DPI
PUBLICATION_DPI = 300  # Target DPI for PNG export

__all__ = [
    'setup_plotting_theme', 'PlottingUtils',
    'create_custom_legend_annotations', 
    'plot_stacked_bar', 'plot_metadata_pairplot',
    'plot_metadata_correlation_heatmap',
    'plot_sample_facility_map', 'plot_sample_taxon_map',
    'create_volcano_plot', 'create_ma_plot',
    'effect_size_volcano', 'DEFAULT_HEIGHT', 'DEFAULT_WIDTH',
    'PUBLICATION_DPI', 'create_integrated_dashboard', 'create_qc_aware_diversity_dashboard',
    'plot_sample_distribution', 'plot_metadata_heatmap', 'plot_metadata_summary_table',
    'create_geographic_map',
    'run_qc_suite', 'create_taxa_abundance_overview', 'create_feature_sparsity_plot',
    'create_sample_sequencing_depth_histogram'
]
