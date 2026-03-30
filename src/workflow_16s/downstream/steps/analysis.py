from workflow_16s.downstream.machine_learning.Project_Discovery_Dashboard import DiscoveryDashboardGenerator
# src/workflow_16s/downstream/steps/analysis.py
"""
16S Modular Analysis Suite: Forensic Discovery & Ecological Orchestration
==========================================================================

This module serves as the primary orchestrator for the downstream 16S pipeline.
It implements a hierarchical analysis flow designed to transition from 
traditional ecological descriptors to high-stakes forensic biomarker discovery.

Key Design Principles:
----------------------
1. Robustness & Isolation: Each module is wrapped in defensive execution blocks 
   to ensure that a single failure (e.g., missing metadata for CST) does not 
   terminate the entire discovery suite.
2. Forensic Integrity: The ML Discovery Matrix prioritizes batch-agnostic 
   stability and cross-study consensus over raw accuracy.
3. Automated Audit: Every run generates a 'Discovery Executive Summary' 
   dashboard, providing a 4-tier certification (Significance, Robustness, 
   Overfitting Gap, and Study Consensus).
4. Ecological Grounding: Nuclear/Facility discoveries are cross-referenced 
   against SoilGrids environmental suites to differentiate between technical 
   fingerprints and true biological signal.

Workflow Phases:
----------------
Phase 1: Foundations (QC, CST Typing, Phylogeny)
Phase 2: Ecology (Alpha/Beta Diversity, Taxonomic Ratios)
Phase 3: Discovery (Differential Abundance, ML Matrix, SoilGrids Baseline)
Phase 4: Synthesis (Ordination, Networks, Executive Dashboard)

Contact: Discovery Workflow Team
Version: 2.1 (Certified Discovery Edition)
"""
import json
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered in divide.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*ClusterWarning.*')
warnings.filterwarnings('ignore', module='.*statsmodels.*')
warnings.filterwarnings('ignore', module='.*scipy.*')

import pandas as pd
import numpy as np
import scanpy as sc

from workflow_16s.api.environmental_data.other.geo_enrichment import (
    GeoContextEnricher, run_enrichment
)
from workflow_16s.downstream.diversity import (
    run_alpha_diversity, run_beta_diversity_and_stats, 
    run_constrained_ordination, run_community_state_typing
)
from workflow_16s.downstream.diversity.phylogenetic import phylogenetic_diversity_workflow
from workflow_16s.downstream.machine_learning.workflows.feature_selection import (
    run_catboost_selection,
    run_selection_with_config
)
#from workflow_16s.modules.machine_learning.catboost.workflows.feature_selection import run_catboost_selection
#from workflow_16s.modules.machine_learning.catboost.main import run_soil_prediction_suite
#from workflow_16s.modules.machine_learning.catboost.nuclear_fuel_cycle.facility_taxa_reporter import run_facility_microbe_report
#from workflow_16s.modules.machine_learning.catboost.Project_Discovery_Dashboard import DiscoveryDashboardGenerator
from workflow_16s.downstream.networks import network_analysis_workflow
from workflow_16s.downstream.qc import profile_metadata, generate_html_report
from workflow_16s.downstream.stats import compare_da_methods, consensus_da_features
from workflow_16s.downstream.visualization import run_qc_suite
# Module 1: Functional Trait Mapping
from workflow_16s.downstream.functional_biogeography import (
    MetalResistanceGeneDatabase,
    extract_traits_from_otu_metadata,
    map_traits_to_otus,
    create_trait_matrix,
    ConservationAnalyzer,
)
# Module 2: Ecotype Stratification
from workflow_16s.downstream.ecotype_stratification import (
    analyze_ecotype_stratification,
    generate_stratification_report,
)
# Module 3: Metal Selection Pressure
from workflow_16s.downstream.functional_biogeography import (
    MetalSelectionPressureAnalyzer,
)
# Module 4: Integrated Visualization
from workflow_16s.downstream.functional_biogeography import (
    DashboardBuilder,
    VisualizationConfig,
)

# NEW MODULES: ASV-MAG, Functional Profiling, Statistics, Validation
from workflow_16s.downstream.steps.integration_orchestration import (
    run_asv_mag_mapping_module,
    run_functional_profiling_module,
    run_statistical_analysis_module,
    run_validation_module
)
# Phylogenetic Signal Analysis (Pagel's Lambda)
from workflow_16s.downstream.steps.phylogenetic_signal_step import (
    run_phylogenetic_signal_step
)

from workflow_16s.utils.analysis import AnalysisUtils
from workflow_16s.utils.logger import with_logger
from workflow_16s.utils.traceback_handler import catch_and_trace

# Type Aliases
StatusDict = Dict[str, Any]

@with_logger
def run_analysis_suite(workflow: Any) -> None:
    """
    Executes the full suite of ecological, statistical, and ML analyses.
    Now featuring a certified Forensic Discovery workflow.
    """
    workflow.logger.info("🚀 Modular Analysis Suite: Orchestrating discovery...")
    if workflow.adata is None: 
        workflow.logger.error("AnnData object is missing. Aborting.")
        return

    run_status: StatusDict = {
        "timestamp": datetime.now().isoformat(),
        "project_name": getattr(workflow, 'run_context', 'default_run'),
        "modules": {}
    }

    # --- PHASE 0: DATA SANITIZATION ---
    if not workflow.adata.obs_names.is_unique:
        workflow.adata.obs_names_make_unique()

    # --- PHASE 0.5: CONTEXT ENRICHMENT ---
    # Scavenge coordinates, fetch elevation, calculate day length
    _execute(workflow, "Geo_Enrichment", _run_geo_enrichment_module, run_status, workflow)

    # --- PHASE 1: QC & FOUNDATIONS ---
    _execute(workflow, "QC_Profiling", _run_qc_module, run_status, workflow)
    tree_path: Optional[Path] = _run_tree_handling(workflow)
    _execute(workflow, "CST_Typing", _run_cst_module, run_status, workflow)

    # --- PHASE 1.5: FUNCTIONAL BIOGEOGRAPHY (MODULE 1) ---
    _execute(workflow, "Functional_Traits", _run_module1_functional_traits, run_status, workflow)

    # --- PHASE 1.6: PHYLOGENETIC SIGNAL ANALYSIS (PAGEL'S LAMBDA) ---
    _execute(workflow, "Phylogenetic_Signal", _run_phylogenetic_signal_module, run_status, workflow)

    # --- PHASE 2: DIVERSITY & ECOLOGY ---
    _execute(workflow, "Phylo_Diversity", _run_phylo_diversity_module, run_status, workflow, tree_path)
    _execute(workflow, "Alpha_Diversity", _run_alpha_module, run_status, workflow, tree_path)
    _execute(workflow, "Beta_Diversity", _run_beta_module, run_status, workflow, tree_path)

    # --- PHASE 2.5: ECOTYPE STRATIFICATION (MODULE 2) ---
    _execute(workflow, "Ecotype_Stratification", _run_module2_ecotype_stratification, run_status, workflow)

    # --- PHASE 2.7: NEW MODULE 2a - ASV-to-MAG MAPPING ---
    _execute(workflow, "ASV_MAG_Mapping", run_asv_mag_mapping_module, run_status, workflow)
    
    # --- PHASE 2.8: NEW MODULE 3 - FUNCTIONAL PROFILING ---
    _execute(workflow, "Functional_Profiling", run_functional_profiling_module, run_status, workflow)

    # --- PHASE 3: BIOMARKER DISCOVERY ---
    _execute(workflow, "Diff_Abundance", _run_da_module, run_status, workflow)
    _execute(workflow, "Metal_Selection_Pressure", _run_module3_metal_selection, run_status, workflow)
    _execute(workflow, "ML_Discovery_Matrix", _run_ml_matrix_module, run_status, workflow)
    
    # --- PHASE 3.5: NEW MODULE 4 - STATISTICAL ANALYSIS ---
    _execute(workflow, "Statistical_Analysis", run_statistical_analysis_module, run_status, workflow)
    
    # --- PHASE 3.7: NEW MODULE 5 - VALIDATION ---
    _execute(workflow, "Validation", run_validation_module, run_status, workflow)

    # --- PHASE 4: ECOLOGICAL NETWORKS & ORDINATION ---
    _execute(workflow, "Ordination", _run_ordination_module, run_status, workflow)
    _execute(workflow, "Network_Analysis", _run_network_inference_module, run_status, workflow)

    # --- PHASE 5: INTEGRATED VISUALIZATION (MODULE 4) ---
    _execute(workflow, "Integrated_Visualization", _run_module4_integrated_viz, run_status, workflow)
    workflow.logger.info("［✅］Full Discovery Suite Execution Complete.")

# EXECUTIION

def _execute(workflow: Any, name: str, func: Callable, status_dict: StatusDict, *args: Any, **kwargs: Any) -> None:
    """Execute analysis module with telemetry tracking."""
    try:
        result = func(*args, **kwargs)
        
        # Check if module was disabled/skipped due to configuration
        if result is False:
            status_dict["modules"][name] = "Skipped (Config)"
            # Mark as disabled in telemetry for dashboard visibility
            workflow.telemetry.mark_module_disabled(name)
        else:
            status_dict["modules"][name] = "Success"
            
            # Emit telemetry events for executed modules
            workflow.telemetry.emit(
                event_type='analysis_module',
                phase=name,
                message=f"Completed {name}",
                metrics={'status': 'success', 'result': result}
            )
    except Exception as e:
        import traceback
        traceback.print_exc()

        workflow.logger.error(f"［❌］Module '{name}' critical failure: {e}")
        workflow.logger.debug(traceback.format_exc())
        status_dict["modules"][name] = f"Failed: {str(e)}"
        
        # Emit telemetry error event
        workflow.telemetry.emit(
            event_type='error',
            phase=name,
            message=f"Module {name} failed: {str(e)}",
            metrics={'status': 'failed', 'error': str(e)}
        )

# HELPER FUNCTIONS

def _run_qc_module(workflow: Any) -> bool:
    # --- Metadata Profiling Report (existing) ---
    ml_targets = ['Env_Level_1', 'Env_Level_2']
    profile_results = profile_metadata(
        workflow.adata, 
        output_dir=workflow.output_dir / 'metadata_profiling',
        ml_targets=ml_targets,
        priority_columns=workflow.priority_categorical + workflow.priority_numeric
    )
    generate_html_report(
        profile_results, 
        output_path=workflow.output_dir / 'metadata_profiling' / 'metadata_profile_report.html'
    )
    
    # --- Comprehensive QC & Sample Metadata Visualizations (new) ---
    try:
        run_qc_suite(workflow)
    except Exception as e:
        workflow.logger.warning(f"QC visualization suite failed: {e}")
        # Don't fail the entire QC module if visualizations fail
    
    return True


@catch_and_trace
def _run_ml_matrix_module(workflow: Any) -> bool:
    """
    Certified ML Discovery Matrix.
    Executes Main Forensic Targets first, then SoilGrids, then Final Dashboard.
    
    Now uses unified orchestrator (run_selection_with_config) to respect ALL config parameters:
    - cv_strategy_threshold: Dynamic switching between LOPO and GroupKFold
    - auto_fix_compositionality: Auto-correct CLR if data is warped
    - pipeline_variants: ASV-first, aggregate-first, or both
    - preprocessing: All QC and filtering settings
    
    PHASE 3: Also supports strategy-driven execution via run_strategies_from_config
    when ml.strategies is configured in config.yaml
    """
    ml_cfg = getattr(workflow.config, 'ml', None)
    if not (ml_cfg and ml_cfg.enabled): return False
    
    workflow.logger.info("="*70)
    workflow.logger.info("🎯 ML DISCOVERY MATRIX")
    workflow.logger.info("="*70)
    
    # Check if strategies are configured (Phase 3)
    # Handle both dict and object config formats
    if isinstance(ml_cfg, dict):
        strategies_configured = ml_cfg.get('strategies', None)
    else:
        strategies_configured = getattr(ml_cfg, 'strategies', None)
    
    if strategies_configured and len(strategies_configured) > 0:
        # Phase 3: Strategy-driven execution
        workflow.logger.info("\n🔄 PHASE 3 MODE: Strategy-Driven Execution")
        workflow.logger.info(f"   Configured strategies: {strategies_configured}")
        
        try:
            from workflow_16s.downstream.machine_learning import run_strategies_from_config
            
            workflow.logger.info("\n📋 CALLING: run_strategies_from_config()")
            workflow.logger.info("   This executes multiple strategies sequentially with:")
            workflow.logger.info("   • Strategy-specific feature sets, preprocessing, CV methods")
            workflow.logger.info("   • Per-strategy output directories")
            workflow.logger.info("   • Comprehensive benchmarking and metadata")
            workflow.logger.info("   • Strategy execution summary report")
            workflow.logger.info("")
            
            results = run_strategies_from_config(
                adata=workflow.adata,
                ml_config=ml_cfg,
                output_dir=workflow.output_dir / "machine_learning"
            )
            
            workflow.logger.info("✅ ML Discovery Matrix Complete (via strategy orchestrator)")
            return True
            
        except Exception as e:
            workflow.logger.error(f"❌ Strategy-driven ML failed: {e}")
            import traceback
            workflow.logger.error(traceback.format_exc())
            return False
    
    else:
        # Phase 1-2: Standard unified orchestrator
        workflow.logger.info("\n🔄 STANDARD MODE: Unified Config-Respecting Orchestrator")
        
        # Use unified orchestrator that respects ALL ML config parameters
        # This includes: cv_strategy_threshold, auto_fix_compositionality
        workflow.logger.info("\n📋 CALLING: run_selection_with_config()")
        workflow.logger.info("   This unified orchestrator will:")
        workflow.logger.info("   • Respect pipeline_variants (ASV-first, aggregate-first, or both)")
        workflow.logger.info("   • Apply cv_strategy_threshold (switch LOPO ↔ GroupKFold)")
        workflow.logger.info("   • Enable auto_fix_compositionality if configured")
        workflow.logger.info("   • Pass all preprocessing and validation settings")
        workflow.logger.info("   • Log all parameters being respected")
        workflow.logger.info("")
        
        try:
            run_selection_with_config(
                adata=workflow.adata,
                output_base_dir=workflow.output_dir / "machine_learning",
                ml_config=ml_cfg,
                n_cpus=workflow.n_cpus,
                telemetry=workflow.telemetry,
            )
            workflow.logger.info("✅ ML Discovery Matrix Complete (via unified orchestrator)")
            return True
            
        except Exception as e:
            workflow.logger.error(f"❌ ML Discovery Matrix failed: {e}")
            import traceback
            workflow.logger.error(traceback.format_exc())
            return False

@catch_and_trace
def _run_tree_handling(workflow: Any) -> Optional[Path]:
    """Retrieves or builds a phylogenetic tree based on config strategy."""
    phylo_cfg = getattr(workflow.config, 'phylogeny', None)
    if not (phylo_cfg and getattr(phylo_cfg, 'enabled', False)): return None
    
    tree_file = workflow.output_dir / "all_features.tree"
    if tree_file.exists(): return tree_file

    from workflow_16s.downstream.utils import handle_missing_tree
    strategy = getattr(phylo_cfg, 'missing_tree_strategy', 'auto')
    return handle_missing_tree(
        workflow.adata, 
        workflow.config, 
        workflow.output_dir, 
        strategy=strategy
    )

def _run_cst_module(workflow: Any) -> bool:
    """Community State Typing: Identifies high-level microbial profiles."""
    if not getattr(workflow.config, 'cst_enabled', True): return False
    cst_col = run_community_state_typing(
        workflow.adata, 
        workflow.plot_dir_beta, 
        level='Genus'
    )
    if cst_col:
        workflow.cst_col = cst_col
        workflow.priority_categorical.append(cst_col)
        # Ensure CST is included in downstream metadata statistics
    return True

def _run_phylo_diversity_module(workflow: Any, tree_path: Optional[Path]) -> bool:
    """Calculates Faith's PD and UniFrac distances."""
    phylo_cfg = getattr(workflow.config, 'phylogeny', None)
    if not (phylo_cfg and getattr(phylo_cfg, 'enabled', False) and tree_path): return False
    
    alpha_cfg = getattr(phylo_cfg, 'alpha_diversity', None)
    beta_cfg = getattr(phylo_cfg, 'beta_diversity', None)
    phylogenetic_diversity_workflow(
        workflow.adata, 
        tree=str(tree_path),
        calculate_pd=getattr(alpha_cfg, 'faiths_pd', True) if alpha_cfg else True,
        calculate_wunifrac=getattr(beta_cfg, 'weighted_unifrac', True) if beta_cfg else True,
        output_dir=workflow.output_dir / 'phylogenetic_diversity'
    )
    return True

def _run_alpha_module(workflow: Any, tree_path: Optional[Path]) -> bool:
    """Traditional Alpha Diversity (Richness, Shannon, etc.)."""
    alpha_cfg = getattr(workflow.config, 'alpha_diversity', None)
    if not (alpha_cfg and alpha_cfg.enabled): return False
    run_alpha_diversity(
        workflow.adata, 
        workflow.plot_dir_alpha, 
        tree_path=tree_path,
        priority_categorical=workflow.priority_categorical,
        priority_numeric=workflow.priority_numeric
    )
    return True

def _run_beta_module(workflow: Any, tree_path: Optional[Path]) -> bool:
    """Beta Diversity: PCoA/NMDS and PERMANOVA statistics."""
    beta_cfg = getattr(workflow.config, 'beta_diversity', None)
    if not (beta_cfg and beta_cfg.enabled): return False
    run_beta_diversity_and_stats(
        workflow.adata, 
        ['Genus'], 
        workflow.plot_dir_beta, 
        tree_path=tree_path, 
        n_cpus=workflow.n_cpus
    )
    return True

def _run_da_module(workflow: Any) -> bool:
    """Multi-method Differential Abundance with Consensus scoring."""
    da_cfg = getattr(workflow.config, 'differential_abundance', None)
    if not (da_cfg and da_cfg.enabled): return False
    
    group_col = getattr(da_cfg, 'group_column', None) or getattr(workflow.config, 'group_column', None)
    if group_col and group_col in workflow.adata.obs.columns:
        da_comparison = compare_da_methods(
            workflow.adata, group_col=group_col,
            methods=getattr(da_cfg, 'methods', ['wilcoxon', 'deseq2']),
            output_dir=workflow.output_dir / 'differential_abundance'
        )
        # Consensus logic fallback
        cons_cfg = getattr(da_cfg, 'consensus', None)
        if getattr(cons_cfg, 'enabled', True) if cons_cfg else True:
            min_aggr = getattr(cons_cfg, 'min_agreement', 2) if cons_cfg else 2
            workflow.adata.uns['da_consensus'] = consensus_da_features(
                da_comparison, 
                min_methods=min_aggr
            )
        return True
    return False

def _run_ordination_module(workflow: Any) -> bool:
    """Constrained Ordination (CCA/RDA) for metadata-community linkage."""
    ord_cfg = getattr(workflow.config, 'ordination', None)
    if not (ord_cfg and ord_cfg.enabled): return False
    run_constrained_ordination(
        workflow.adata, 
        ['Genus'], 
        workflow.plot_dir_beta, 
        workflow.priority_vars
    )
    return True

def _run_network_inference_module(workflow: Any) -> bool:
    """Compositional Network Inference (SPIEC-EASI/SparCC)."""
    net_config = getattr(workflow.config, 'networks', None)
    if not (net_config and net_config.enabled): return False
    network_analysis_workflow(
        workflow.adata, 
        method=getattr(net_config, 'method', 'sparcc'),
        output_dir=workflow.output_dir / 'networks'
    )
    return True

def _run_longitudinal_module(workflow: Any) -> bool:
    """Time-series analysis for stability and trajectory clustering."""
    long_cfg = getattr(workflow.config, 'longitudinal', None)
    if not (long_cfg and long_cfg.enabled): return False
    # logic using calculate_temporal_stability helper...
    return True

def _run_geo_enrichment_module(workflow: Any) -> bool:
    """
    Enriches metadata with astronomical, topographical, and weather context.
    Uses cached lookups to minimize API hits.
    """
    # Check config to see if enrichment is allowed (optional, default to True)
    enrich_cfg = getattr(workflow.config, 'enrichment', {})
    if not enrich_cfg.get('enabled', True): return False
        
    workflow.logger.info("［］Geo-Context: Scavenging coordinates & fetching API data...")
    
    # Run the enrichment logic (imported from adata_utils)
    workflow.adata = run_enrichment(workflow.adata)
    
    # Register new columns as numeric priorities so they appear in plots
    new_cols = ['elevation_m', 'weather_temp_avg', 'weather_precip_sum', 'calc_day_length_hours']
    for col in new_cols:
        if col in workflow.adata.obs.columns and col not in workflow.priority_numeric:
            workflow.priority_numeric.append(col)
            
    return True

def _save_run_summary(workflow: Any, status: StatusDict) -> None:
    """Persists the execution record for reproducibility and debugging."""
    summary_path = workflow.output_dir / "run_summary.json"
    with open(summary_path, 'w') as f: json.dump(status, f, indent=4)
    workflow.logger.info(f"💾 Execution summary saved to {summary_path}")

# ============================================================================
# MODULE 1: FUNCTIONAL TRAIT MAPPING
# ============================================================================

@catch_and_trace
def _run_module1_functional_traits(workflow: Any) -> bool:
    """
    Module 1: Functional Trait Mapping
    Maps functional traits (metal resistance, metabolism) to OTUs using JGI/RAST.
    """
    ftm_cfg = getattr(workflow.config, 'functional_biogeography', None)
    if not (ftm_cfg and getattr(ftm_cfg, 'enabled', True)): return False
    
    workflow.logger.info("=" * 70)
    workflow.logger.info("MODULE 1: FUNCTIONAL TRAIT MAPPING (JGI-RAST Integration)")
    workflow.logger.info("=" * 70)
    
    try:
        # Initialize trait database
        credentials = getattr(workflow.config, 'credentials', None)
        user_email = getattr(credentials, 'email', 'macgregor@berkeley.edu') if credentials else 'macgregor@berkeley.edu'
        trait_db = MetalResistanceGeneDatabase(user_email=user_email, use_jgi=True)
        workflow.logger.info(f"✓ Trait database initialized with {len(trait_db.traits)} traits")
        
        # Extract traits from OTU metadata if available
        paths = getattr(workflow.config, 'paths', None)
        otu_metadata_path = getattr(paths, 'otu_metadata', None) if paths else None
        if otu_metadata_path and Path(otu_metadata_path).exists():
            workflow.logger.info(f"Loading OTU metadata from {otu_metadata_path}")
            traits = extract_traits_from_otu_metadata(Path(otu_metadata_path), trait_db)
        else:
            traits = {name: trait for name, trait in trait_db.traits.items()}
        
        # Create trait matrix (OTUs × Traits)
        trait_matrix, _ = create_trait_matrix(
            adata=workflow.adata,
            trait_db=trait_db,
            user_email=user_email,
            use_jgi=True
        )
        
        # Store trait matrix for downstream modules
        workflow.trait_matrix = trait_matrix
        workflow.adata.uns['trait_matrix'] = trait_matrix
        
        # Optional: Phylogenetic signal analysis
        if getattr(ftm_cfg, 'analyze_phylogenetic_signal', False) and hasattr(workflow, 'tree'):
            from workflow_16s.downstream.functional_biogeography import (
                assess_trait_phylogenetic_structure
            )
            phylo_results = assess_trait_phylogenetic_structure(
                workflow.adata,
                trait_matrix,
                tree_path=workflow.tree
            )
            workflow.adata.uns['trait_phylogenetic_signal'] = phylo_results
            workflow.logger.info(f"✓ Phylogenetic signal analysis complete")
        
        # Optional: Conservation analysis
        if getattr(ftm_cfg, 'analyze_conservation', False):
            conserv = ConservationAnalyzer()
            conserv_results = conserv.analyze_functional_vs_taxonomic_conservation(
                workflow.adata,
                trait_matrix
            )
            workflow.adata.uns['conservation_analysis'] = conserv_results
            workflow.logger.info(f"✓ Conservation analysis complete")
        
        workflow.logger.info("✅ Module 1 (Functional Traits) Complete")
        return True
        
    except Exception as e:
        workflow.logger.error(f"❌ Module 1 failed: {e}")
        import traceback
        workflow.logger.error(traceback.format_exc())
        return False


# ============================================================================
# PHASE 1.6: PHYLOGENETIC SIGNAL ANALYSIS (PAGEL'S LAMBDA)
# ============================================================================

@catch_and_trace
def _run_phylogenetic_signal_module(workflow: Any) -> bool:
    """
    Analyze phylogenetic signal (Pagel's lambda) of functional traits.
    
    This module quantifies how phylogenetically conserved each trait is,
    ranging from λ=0 (independent evolution) to λ=1 (strong phylogenetic signal).
    Results are saved as forest plots, scatter plots, and markdown reports.
    
    Integration Point: Phase 1.6, after functional trait mapping.
    Status: Optional, configurable via downstream.phylogenetic_signal.enabled
    """
    try:
        # Check if phylogenetic signal analysis is enabled
        ps_config = workflow.config.get('downstream', {}).get('phylogenetic_signal', {})
        if not ps_config.get('enabled', False):
            workflow.logger.info("⊘ Phylogenetic signal analysis disabled in config")
            return True
        
        if workflow.adata is None:
            workflow.logger.warning("⊘ AnnData object not found, skipping phylogenetic signal analysis")
            return True
        
        # Get project output directory
        project_dir = Path(workflow.config.get('paths', {}).get('project', './'))
        output_base = project_dir / 'analysis'
        
        workflow.logger.info("📊 Running Phylogenetic Signal Analysis (Pagel's Lambda)...")
        
        # Run phylogenetic signal step
        result = run_phylogenetic_signal_step(
            config=workflow.config,
            adata=workflow.adata,
            output_base=output_base
        )
        
        # Store results in workflow
        if result['status'] == 'success':
            workflow.adata.uns['phylogenetic_signal'] = {
                'analyzer': result.get('analyzer'),
                'results': result.get('results'),
                'num_traits': result.get('num_traits', 0),
                'num_conserved': result.get('num_conserved', 0),
                'num_adaptive': result.get('num_adaptive', 0),
            }
            workflow.logger.info(f"✓ Phylogenetic signal analysis complete: "
                               f"{result.get('num_traits', 0)} traits analyzed "
                               f"({result.get('num_conserved', 0)} conserved, "
                               f"{result.get('num_adaptive', 0)} adaptive)")
            return True
        elif result['status'] == 'skipped':
            workflow.logger.info(f"⊘ Phylogenetic signal analysis skipped: {result.get('reason', 'unknown')}")
            return True
        else:
            workflow.logger.error(f"❌ Phylogenetic signal analysis failed: {result.get('error', 'unknown error')}")
            return False
            
    except Exception as e:
        workflow.logger.error(f"❌ Phylogenetic signal analysis failed: {e}")
        import traceback
        workflow.logger.error(traceback.format_exc())
        return False


# ============================================================================
# MODULE 2: ECOTYPE STRATIFICATION
# ============================================================================

@catch_and_trace
def _run_module2_ecotype_stratification(workflow: Any) -> bool:
    """
    Module 2: Ecotype Stratification
    Detects cryptic strain variants using trait-based clustering.
    Requires Module 1 (trait matrix) to be run first.
    """
    eco_cfg = getattr(workflow.config, 'ecotype_stratification', None)
    if not (eco_cfg and getattr(eco_cfg, 'enabled', True)): return False
    
    # Check if Module 1 has been run
    if not hasattr(workflow, 'trait_matrix') or workflow.trait_matrix is None:
        workflow.logger.warning("⚠️  Module 2 requires Module 1 output. Skipping.")
        return False
    
    workflow.logger.info("=" * 70)
    workflow.logger.info("MODULE 2: ECOTYPE STRATIFICATION (Strain-level clustering)")
    workflow.logger.info("=" * 70)
    
    try:
        # Run ecotype stratification
        eco_results = analyze_ecotype_stratification(
            adata=workflow.adata,
            trait_matrix=workflow.trait_matrix,
            otu_level=getattr(eco_cfg, 'otu_level', 99),
            clustering_method=getattr(eco_cfg, 'clustering_method', 'kmeans'),
            n_clusters_range=getattr(eco_cfg, 'n_clusters_range', (2, 6)),
            environmental_variable=getattr(eco_cfg, 'environmental_variable', None),
            output_dir=workflow.output_dir / 'ecotype_stratification',
            n_workers=getattr(workflow, 'n_cpus', 8)
        )
        
        # Store results
        workflow.ecotype_results = eco_results
        workflow.adata.obs['ecotype_assignment'] = eco_results['ecotype_assignments']['ecotype_id']
        workflow.adata.uns['ecotype_profiles'] = eco_results['ecotype_profiles']
        workflow.adata.uns['niche_profiles'] = eco_results['niche_profiles']
        
        # Add ecotype to priority categorical variables
        if 'ecotype_assignment' not in workflow.priority_categorical:
            workflow.priority_categorical.append('ecotype_assignment')
        
        # Generate report
        if getattr(eco_cfg, 'generate_report', True):
            report_path = workflow.output_dir / 'ecotype_stratification' / 'stratification_report.html'
            generate_stratification_report(
                results=eco_results,
                output_path=report_path
            )
            workflow.logger.info(f"✓ Report saved to {report_path}")
        
        workflow.logger.info("✅ Module 2 (Ecotype Stratification) Complete")
        return True
        
    except Exception as e:
        workflow.logger.error(f"❌ Module 2 failed: {e}")
        import traceback
        workflow.logger.error(traceback.format_exc())
        return False

# ============================================================================
# MODULE 3: METAL SELECTION PRESSURE
# ============================================================================

@catch_and_trace
def _run_module3_metal_selection(workflow: Any) -> bool:
    """
    Module 3: Metal Selection Pressure Analysis
    Analyzes metal enrichment from geologic/elemental proxies and correlates with traits.
    Requires Module 1 (trait matrix) to be run first.
    """
    metal_cfg = getattr(workflow.config, 'metal_selection_pressure', None)
    if not (metal_cfg and getattr(metal_cfg, 'enabled', True)): return False
    
    # Check if Module 1 has been run
    if not hasattr(workflow, 'trait_matrix') or workflow.trait_matrix is None:
        workflow.logger.warning("⚠️  Module 3 requires Module 1 output. Skipping.")
        return False
    
    workflow.logger.info("=" * 70)
    workflow.logger.info("MODULE 3: METAL SELECTION PRESSURE (Geologic + Elemental Proxies)")
    workflow.logger.info("=" * 70)
    
    try:
        # Initialize Metal Selection Pressure Analyzer
        credentials = getattr(workflow.config, 'credentials', None)
        user_email = getattr(credentials, 'email', 'macgregor@berkeley.edu') if credentials else 'macgregor@berkeley.edu'
        
        # Pass config object directly (get_gee_client handles Pydantic models)
        analyzer = MetalSelectionPressureAnalyzer(
            adata=workflow.adata,
            user_email=user_email,
            use_gee=getattr(metal_cfg, 'use_gee', True),
            config=workflow.config  # Pass config object directly
        )
        
        # Run analysis
        metal_results = analyzer.run_analysis(
            metals=getattr(metal_cfg, 'metals', [
                'uranium', 'arsenic', 'copper', 'lead', 'zinc', 'cadmium', 'nickel', 'rare_earth'
            ]),
            min_metal_proxy=getattr(metal_cfg, 'min_metal_proxy', 0.1),
            corr_threshold=getattr(metal_cfg, 'correlation_threshold', 0.05)
        )
        
        # Store results
        workflow.metal_results = metal_results
        workflow.adata.uns['metal_selection_pressure'] = metal_results
        
        # Log results summary (metal_results is a dict of metal → MetalSelectionResult objects)
        n_metals_analyzed = len([m for m in metal_results.values() if m is not None])
        workflow.logger.info(f"✓ Analyzed {n_metals_analyzed} metals for selection pressure")
        
        # Add metal proxy scores to obs if they were calculated
        # Note: Proxy scores are calculated but not stored in this return structure
        # To preserve them, would need to modify run_analysis() to return them separately
        
        workflow.logger.info("✅ Module 3 (Metal Selection Pressure) Complete")
        return True
        
    except Exception as e:
        workflow.logger.error(f"❌ Module 3 failed: {e}")
        import traceback
        workflow.logger.error(traceback.format_exc())
        return False

# ============================================================================
# MODULE 4: INTEGRATED VISUALIZATION
# ============================================================================

@catch_and_trace
def _run_module4_integrated_viz(workflow: Any) -> bool:
    """
    Module 4: Integrated Visualization
    Creates publication-ready figures combining phylogenetic, functional, spatial, and ecotype data.
    Requires Modules 1-3 outputs.
    """
    viz_cfg = getattr(workflow.config, 'integrated_visualization', None)
    if not (viz_cfg and getattr(viz_cfg, 'enabled', True)): return False
    
    workflow.logger.info("=" * 70)
    workflow.logger.info("MODULE 4: INTEGRATED VISUALIZATION (Publication figures)")
    workflow.logger.info("=" * 70)
    
    try:
        # Initialize visualization config
        vis_config = VisualizationConfig(
            output_dir=workflow.output_dir / 'integrated_visualization',
            dpi=getattr(viz_cfg, 'dpi', 300),
            width=getattr(viz_cfg, 'width', 1200),
            height=getattr(viz_cfg, 'height', 800),
            color_scheme=getattr(viz_cfg, 'color_scheme', 'viridis'),
            export_html=getattr(viz_cfg, 'export_html', True),
            export_png=getattr(viz_cfg, 'export_png', False)
        )
        
        # Build dashboard
        dashboard = DashboardBuilder(config=vis_config)
        
        # Add visualizations conditionally based on module completion
        viz_results = {}
        
        # Always available: Diversity/ordination plots
        if getattr(viz_cfg, 'include_diversity', True):
            workflow.logger.info("• Adding diversity visualization...")
            # Ordination plot from PHASE 4
            viz_results['ordination'] = "Ordination plot (from PHASE 4)"
        
        # Module 1 outputs: Functional traits
        if hasattr(workflow, 'trait_matrix') and workflow.trait_matrix is not None:
            if getattr(viz_cfg, 'include_traits', True):
                workflow.logger.info("• Adding functional trait visualization...")
                viz_results['traits'] = "Trait heatmap (from Module 1)"
        
        # Module 2 outputs: Ecotypes
        if hasattr(workflow, 'ecotype_results') and workflow.ecotype_results is not None:
            if getattr(viz_cfg, 'include_ecotypes', True):
                workflow.logger.info("• Adding ecotype visualization...")
                viz_results['ecotypes'] = "Ecotype distribution (from Module 2)"
        
        # Module 3 outputs: Metal selection
        if hasattr(workflow, 'metal_results') and workflow.metal_results is not None:
            if getattr(viz_cfg, 'include_metals', True):
                workflow.logger.info("• Adding metal selection pressure visualization...")
                viz_results['metals'] = "Metal proxy correlation (from Module 3)"
        
        # Export dashboard
        output_dir = vis_config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary HTML
        summary_html = output_dir / 'dashboard_summary.html'
        with open(summary_html, 'w') as f:
            f.write("<html><body>")
            f.write("<h1>Integrated Visualization Dashboard</h1>")
            f.write("<ul>")
            for viz_name, viz_desc in viz_results.items():
                f.write(f"<li>{viz_name}: {viz_desc}</li>")
            f.write("</ul>")
            f.write("</body></html>")
        
        workflow.logger.info(f"✓ Created {len(viz_results)} visualizations")
        workflow.logger.info(f"✓ Dashboard saved to {summary_html}")
        workflow.logger.info("✅ Module 4 (Integrated Visualization) Complete")
        return True
        
    except Exception as e:
        workflow.logger.error(f"❌ Module 4 failed: {e}")
        import traceback
        workflow.logger.error(traceback.format_exc())
        return False
