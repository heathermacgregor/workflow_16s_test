"""
Functional Biogeography Module

Investigates the relationship between taxonomy and function across environments.
Answers the central scientific question: What defines adaptive functionality?

Data Integration:
- JGI/IMG database (KEGG REST API, InterPro)
- Google Earth Engine (geologic maps, SoilGrids, global coverage)
- RAST annotations (user-provided otus.X.allinfo files)
- Curated trait definitions (literature-based fallback)

Key analyses:
- Functional trait mapping (presence/absence of key genes)
- Phylogenetic signal quantification (Pagel's lambda)
- Taxonomic vs. functional conservation metrics
- Metal selection pressure analysis (geologic + element proxies)
- Integrated visualization (phylo × function × space + ecotypes)
"""

from .functional_trait_mapping import (
    MetalResistanceGeneDatabase,
    extract_traits_from_otu_metadata,
    map_traits_to_otus,
    create_trait_matrix,
    FunctionalTrait,
)

from .jgi_database import (
    JGIDatabaseClient,
    JGIGeneAnnotation,
    get_jgi_client,
)

from .earth_engine_geology import (
    EarthEngineGeologyClient,
    GEEGeologyResult,
    GEESoilElement,
    get_gee_client,
)

from .phylogenetic_signal import (
    calculate_pagels_lambda,
    calculate_phylogenetic_signal,
    assess_trait_phylogenetic_structure,
)

from .conservation_analysis import (
    analyze_functional_vs_taxonomic_conservation,
    generate_conservation_report,
    ConservationAnalyzer
)

from .metal_selection_pressure import (
    MetalSelectionPressureAnalyzer,
    MetalSelectionResult,
)

from .geologic_data import (
    MetalBearingFormations,
    USGSGeologicMapClient,
    get_geologic_client,
    GeologicUnit,
)

from .element_proxy import (
    MetalProxyAnalyzer,
    SoilElementProxyMapping,
    ElementProxyScore,
)

from .visualization import (
    PhyloFunctionVisualizer,
    SpatialTraitVisualizer,
    EcotypeFunctionVisualizer,
    DashboardBuilder,
    VisualizationConfig,
)

__all__ = [
    # Traits & Database
    'FunctionalTrait',
    'MetalResistanceGeneDatabase',
    'extract_traits_from_otu_metadata',
    'map_traits_to_otus',
    'create_trait_matrix',
    # JGI Database
    'JGIDatabaseClient',
    'JGIGeneAnnotation',
    'get_jgi_client',
    # Earth Engine (primary for geospatial)
    'EarthEngineGeologyClient',
    'GEEGeologyResult',
    'GEESoilElement',
    'get_gee_client',
    # Phylogenetic signal
    'calculate_pagels_lambda',
    'calculate_phylogenetic_signal',
    'assess_trait_phylogenetic_structure',
    # Analysis
    'analyze_functional_vs_taxonomic_conservation',
    'generate_conservation_report',
    'ConservationAnalyzer',
    # Metal Selection Pressure (Module 3)
    'MetalSelectionPressureAnalyzer',
    'MetalSelectionResult',
    # Geologic data (USGS fallback)
    'MetalBearingFormations',
    'USGSGeologicMapClient',
    'get_geologic_client',
    'GeologicUnit',
    # Element proxies
    'MetalProxyAnalyzer',
    'SoilElementProxyMapping',
    'ElementProxyScore',
    # Visualization (Module 4)
    'PhyloFunctionVisualizer',
    'SpatialTraitVisualizer',
    'EcotypeFunctionVisualizer',
    'DashboardBuilder',
    'VisualizationConfig',
]
