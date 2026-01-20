import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from workflow_16s.downstream.diversity import (
    run_alpha_diversity, run_beta_diversity_and_stats, run_taxa_metadata_statistics, 
    run_constrained_ordination, run_network_analysis,
    run_community_state_typing
)
from workflow_16s.downstream.machine_learning import (
    run_machine_learning_analysis, run_catboost_selection
)
from workflow_16s.downstream.steps.synthesis import handle_strategy_impact_plot

# Import new scientific analysis modules
from workflow_16s.downstream.phylogenetic_diversity import phylogenetic_diversity_workflow
from workflow_16s.downstream.differential_abundance import compare_da_methods, consensus_da_features
from workflow_16s.downstream.compositional_networks import network_analysis_workflow
from workflow_16s.downstream.metadata_profiler import profile_metadata, generate_html_report

def run_analysis_suite(workflow):
    """
    Execute the full suite of ecological and statistical analyses on microbial community data.
    
    This function orchestrates a comprehensive analysis workflow that includes:
    1. Community State Typing (CST) to identify dominant microbial profiles
    2. Alpha diversity metrics (richness, evenness, AND NEW: phylogenetic diversity)
    3. Machine learning biomarker discovery with multiple batch correction strategies
    4. Statistical testing of taxa-metadata associations
    5. NEW: Multi-method differential abundance testing with consensus framework
    6. Constrained ordination and network analysis
    7. Beta diversity analysis and significance testing (AND NEW: UniFrac distances)
    8. NEW: Compositional network inference (SPIEC-EASI, SparCC, proportionality)
    
    The function implements a comparative machine learning approach that tests three
    strategies for handling batch effects:
    - Baseline: No batch correction
    - Agnostic: Remove batch variables from features
    - Group Validated: Use group-aware k-fold cross-validation
    
    Parameters
    ----------
    workflow : DownstreamWorkflow
        The workflow instance containing:
        - adata: AnnData object with microbial abundance data and metadata
        - output_dir: Base directory for output files
        - plot_dir_*: Subdirectories for different plot types
        - catboost_output_dir: Directory for CatBoost model outputs
        - n_cpus: Number of CPU cores to use for parallel processing
        - priority_categorical: List of categorical metadata columns for priority analysis
        - priority_numeric: List of numeric metadata columns for priority analysis
        - priority_vars: Combined list of priority metadata variables
        - logger: Logger instance for status reporting
    
    Notes
    -----
    - Requires workflow.adata to be non-None
    - Phylogenetic tree (all_features.tree) is optional but recommended for phylogenetic metrics
    - Machine learning targets are hardcoded to 'facility_match' and 'facility_distance_km'
    - All analyses operate at the Genus taxonomic level
    - Results are saved to workflow-specific output directories
    - CST results are added to priority categorical variables for downstream use
    
    See Also
    --------
    run_community_state_typing : Identifies microbial community profiles
    run_alpha_diversity : Calculates within-sample diversity metrics
    phylogenetic_diversity_workflow : NEW - Calculates Faith's PD and UniFrac
    compare_da_methods : NEW - Multi-method differential abundance testing
    network_analysis_workflow : NEW - Compositional network inference
    """
    workflow.logger.info("4. Modular Analysis Suite: Executing modules...")
    if workflow.adata is None: return
    
    # ------------------------------------------------------------------
    # [CRITICAL FIX] GLOBAL DUPLICATE INDEX RESOLUTION
    # ------------------------------------------------------------------
    # Ensure indices are unique BEFORE any analysis runs.
    # This prevents "ValueError: cannot reindex on an axis with duplicate labels"
    # in Alpha Diversity, Machine Learning, and Plotting.
    if not workflow.adata.obs_names.is_unique:
        workflow.logger.warning(f"⚠️ Duplicate sample IDs detected ({len(workflow.adata.obs_names) - len(workflow.adata.obs_names.unique())} duplicates).")
        workflow.logger.warning("   Resolving by appending suffixes (e.g. SampleA -> SampleA-1)...")
        workflow.adata.obs_names_make_unique()
        workflow.logger.info("✅ Sample IDs are now unique.")
    # ------------------------------------------------------------------
    
    # 0. METADATA PROFILING (NEW)
    # Generate comprehensive metadata quality report before analysis
    # WHY: Identify data quality issues and ML warnings early
    workflow.logger.info("Generating metadata profiling report...")
    ml_targets = ['facility_match', 'facility_distance_km']
    try:
        profile_results = profile_metadata(
            workflow.adata,
            output_dir=workflow.output_dir / 'metadata_profiling',
            ml_targets=ml_targets,
            priority_columns=workflow.priority_categorical + workflow.priority_numeric
        )
        
        # Generate HTML report
        generate_html_report(
            profile_results,
            output_path=workflow.output_dir / 'metadata_profiling' / 'metadata_profile_report.html'
        )
        
        # Fail fast if critical errors detected
        if profile_results['n_errors'] > 0:
            workflow.logger.error(
                f"⚠️  {profile_results['n_errors']} critical metadata errors detected. "
                "Review metadata_profiling/ml_warnings.csv before proceeding."
            )
    except Exception as e:
        workflow.logger.warning(f"Metadata profiling failed: {e}")
    
    tree_file = workflow.output_dir / "all_features.tree"
    tree_path = tree_file if tree_file.exists() else None
    
    # Check if tree is needed for phylogenetic diversity
    phylo_config = getattr(workflow.config, 'phylogeny', None)
    phylo_enabled = getattr(phylo_config, 'enabled', False) if phylo_config else False
    
    # Handle missing or incomplete trees using configurable strategy
    if phylo_enabled and not tree_path:
        workflow.logger.info("Phylogenetic diversity enabled but no tree file found")
        workflow.logger.info("Attempting to handle missing tree...")
        
        from workflow_16s.downstream.tree_handler import handle_missing_tree
        
        # Get strategy from config or use auto
        tree_strategy = getattr(phylo_config, 'missing_tree_strategy', 'auto')
        
        workflow.logger.info(f"Tree handling strategy: {tree_strategy}")
        workflow.logger.info("Available strategies:")
        workflow.logger.info("  - auto: Automatically select best strategy")
        workflow.logger.info("  - graceful_degradation: Skip phylogenetic metrics")
        workflow.logger.info("  - tree_merging: Merge per-dataset trees")
        workflow.logger.info("  - denovo_tree_building: Build new tree from sequences")
        workflow.logger.info("  - partial_analysis: Analyze only tree-covered features")
        workflow.logger.info("  - subset_tree_extraction: Extract subtree for current features")
        
        tree_path = handle_missing_tree(
            workflow.adata,
            workflow.config,
            workflow.output_dir,
            strategy=tree_strategy
        )
        
        if tree_path:
            workflow.logger.info(f"✅ Successfully obtained tree: {tree_path}")
        else:
            workflow.logger.warning("⚠️  No tree available - phylogenetic diversity will be skipped")
    
    # 1. Community State Typing (CST)
    # Identifies dominant microbial community profiles
    cst_col = run_community_state_typing(workflow.adata, workflow.plot_dir_beta, level='Genus')
    if cst_col:
        workflow.cst_col = cst_col
        workflow.priority_categorical.append(cst_col)
        workflow._plot_cst_vs_metadata(cst_col)

    # 2. Phylogenetic Diversity (NEW - if tree available and enabled)
    # Calculates Faith's PD (alpha) and UniFrac distances (beta)
    # WHY: Phylogenetic metrics are more powerful than taxonomy-only metrics
    
    if phylo_enabled and tree_path:
        try:
            workflow.logger.info("Running phylogenetic diversity analysis...")
            alpha_config = getattr(phylo_config, 'alpha_diversity', None)
            beta_config = getattr(phylo_config, 'beta_diversity', None)
            phylo_results = phylogenetic_diversity_workflow(
                workflow.adata,
                tree=str(tree_path),
                calculate_pd=getattr(alpha_config, 'faiths_pd', True) if alpha_config else True,
                calculate_wunifrac=getattr(beta_config, 'weighted_unifrac', True) if beta_config else True,
                calculate_uwunifrac=getattr(beta_config, 'unweighted_unifrac', False) if beta_config else False,
                output_dir=workflow.output_dir / 'phylogenetic_diversity'
            )
            workflow.logger.info("Phylogenetic diversity complete.")
            # Faith's PD is now in adata.obs['faith_pd']
            # UniFrac distances are in adata.uns['weighted_unifrac'] and adata.uns['unweighted_unifrac']
        except Exception as e:
            workflow.logger.warning(f"Phylogenetic diversity analysis failed: {e}")
    elif phylo_enabled and not tree_path:
        workflow.logger.warning("Phylogenetic diversity enabled but no tree file found. Skipping.")
    
    # 3. Alpha Diversity
    # Measures within-sample richness and evenness (now includes Faith's PD if calculated)
    #run_alpha_diversity(
    #    workflow.adata, 
    #    workflow.plot_dir_alpha, 
    #    tree_path=tree_path, 
    #    priority_categorical=workflow.priority_categorical, 
    #    priority_numeric=workflow.priority_numeric
    #)
    
    # 4. Multi-Method Differential Abundance Testing (NEW - if enabled)
    # Uses multiple DA methods and finds consensus features
    # WHY: Different methods have different assumptions; consensus increases confidence
    da_config = getattr(workflow.config, 'differential_abundance', None)
    da_enabled = getattr(da_config, 'enabled', False) if da_config else False
    
    if da_enabled:
        try:
            group_col = getattr(da_config, 'group_column', None) or getattr(workflow.config, 'group_column', None)
            
            if group_col and group_col in workflow.adata.obs.columns:
                workflow.logger.info("Running multi-method differential abundance testing...")
                
                # Run comparison across multiple methods
                da_comparison = compare_da_methods(
                    workflow.adata,
                    methods=getattr(da_config, 'methods', ['wilcoxon', 'deseq2']),
                    group_col=group_col,
                    alpha=getattr(da_config, 'fdr_threshold', 0.05),
                    min_prevalence=getattr(da_config, 'min_prevalence', 0.1),
                    output_dir=workflow.output_dir / 'differential_abundance'
                )
                
                # Find consensus features
                consensus_config = getattr(da_config, 'consensus', None)
                consensus_enabled = getattr(consensus_config, 'enabled', True) if consensus_config else True
                
                if consensus_enabled:
                    min_methods = getattr(consensus_config, 'min_agreement', 2) if consensus_config else 2
                    consensus = consensus_da_features(
                        da_comparison,
                        min_methods=min_methods,
                        max_p_adj=getattr(da_config, 'fdr_threshold', 0.05)
                    )
                    
                    workflow.logger.info(
                        f"Differential abundance complete. "
                        f"Consensus features (≥{min_methods} methods): {len(consensus)}"
                    )
                    
                    # Store results in adata
                    workflow.adata.uns['da_comparison'] = da_comparison
                    workflow.adata.uns['da_consensus'] = consensus
            else:
                workflow.logger.warning(
                    f"Differential abundance enabled but group column '{group_col}' not found. Skipping."
                )
        except Exception as e:
            workflow.logger.warning(f"Differential abundance analysis failed: {e}")
    
    # 5. Machine Learning & Comparative Strategy Modules
    # Tests biomarkers under varying levels of batch control
    ml_targets = ['facility_match', 'facility_distance_km']
    catboost_strategies = [
        {"name": "baseline", "drop_batch": False, "use_group": False},
        {"name": "agnostic", "drop_batch": True, "use_group": False},
        {"name": "group_validated", "drop_batch": True, "use_group": True}
    ]

    for target in ml_targets:
        strat_results = {}
        for strat in catboost_strategies:
            s_name = strat["name"]
            out = workflow.catboost_output_dir / s_name
            out.mkdir(exist_ok=True, parents=True)
            
            run_catboost_selection(
                workflow.adata, out, level='Genus', priority_targets=[target], 
                n_cpus=workflow.n_cpus, drop_batch=strat["drop_batch"],
                use_group_kfold=strat["use_group"], batch_col='batch_original'
            )
            
            # Harvest scores for strategy impact plotting in synthesis step
            sum_p = out / f"Genus_{target}" / "results_summary.json"
            if sum_p.exists():
                with open(sum_p, 'r') as f: strat_results[s_name] = json.load(f)

        # Generate stability tables and heatmaps
        workflow._compare_catboost_strategies('Genus', target)
        handle_strategy_impact_plot(workflow, target, strat_results)
            
    # 6. Standard Machine Learning Baseline WITH Batch Covariate Control
    # Extract batch configuration from workflow config
    batch_config = None
    if hasattr(workflow.config, 'machine_learning'):
        ml_config = workflow.config.machine_learning
        if hasattr(ml_config, 'batch_covariates') and ml_config.batch_covariates.get('enabled', False):
            batch_config = ml_config.batch_covariates
            workflow.logger.info("✓ Batch covariate control enabled for machine learning")
    
    run_machine_learning_analysis(
        adata=workflow.adata,
        plot_dir_ml=workflow.plot_dir_ml,
        level='Genus',
        min_samples_per_group=10,
        max_classes=10,
        priority_targets=workflow.priority_vars,
        batch_config=batch_config
    )
    
    # 7. Core Statistical Tests
    run_taxa_metadata_statistics(
        workflow.adata, 
        ['Genus'], 
        workflow.plot_dir_stats, 
        workflow.n_cpus,
        max_taxa=getattr(workflow.config, 'max_taxa_stats', 1000),
        max_categorical=getattr(workflow.config, 'max_categorical_stats', 30)
    )
    
    # 8. Compositional Network Analysis (NEW - if enabled)
    # Infers co-occurrence networks accounting for compositionality
    # WHY: Standard correlation invalid for compositional data
    network_config = getattr(workflow.config, 'networks', None)
    network_enabled = getattr(network_config, 'enabled', False) if network_config else False
    
    if network_enabled:
        try:
            workflow.logger.info("Running compositional network inference...")
            method = getattr(network_config, 'method', 'sparcc')
            network_results = network_analysis_workflow(
                workflow.adata,
                method=method,
                min_prevalence=getattr(network_config, 'min_prevalence', 0.1),
                output_dir=workflow.output_dir / 'networks'
            )
            workflow.logger.info("Compositional network analysis complete.")
            # Network stored in adata.uns['network']
        except Exception as e:
            workflow.logger.warning(f"Compositional network analysis failed: {e}")
    
    # 9. Longitudinal Analysis (NEW - if enabled and time-series data)
    # Temporal dynamics, trajectory clustering, stability metrics
    # WHY: Many microbiome studies track communities over time (contamination events)
    longitudinal_config = getattr(workflow.config, 'longitudinal', None)
    longitudinal_enabled = getattr(longitudinal_config, 'enabled', False) if longitudinal_config else False
    
    if longitudinal_enabled:
        try:
            time_col = getattr(longitudinal_config, 'time_column', 'collection_date')
            subject_col = getattr(longitudinal_config, 'subject_column', 'location')
            
            if time_col in workflow.adata.obs.columns and subject_col in workflow.adata.obs.columns:
                workflow.logger.info("Running longitudinal analysis...")
                
                from workflow_16s.downstream.longitudinal import (
                    calculate_temporal_stability, trajectory_clustering
                )
                
                # Calculate temporal stability
                stability_df = calculate_temporal_stability(
                    workflow.adata,
                    time_col=time_col,
                    subject_col=subject_col,
                    metric=getattr(longitudinal_config, 'stability_metric', 'bray_curtis')
                )
                
                # Store results
                workflow.adata.uns['temporal_stability'] = stability_df
                
                # Cluster trajectories if requested
                trajectory_config = getattr(longitudinal_config, 'trajectory_clustering', None)
                trajectory_enabled = getattr(trajectory_config, 'enabled', False) if trajectory_config else False
                
                if trajectory_enabled:
                    n_clusters = getattr(trajectory_config, 'n_clusters', 4) if trajectory_config else 4
                    trajectory_results = trajectory_clustering(
                        workflow.adata,
                        time_col=time_col,
                        n_clusters=n_clusters
                    )
                    workflow.adata.uns['trajectory_clusters'] = trajectory_results
                
                workflow.logger.info(
                    f"Longitudinal analysis complete. "
                    f"Stability calculated for {len(stability_df)} subjects."
                )
            else:
                workflow.logger.warning(
                    f"Longitudinal analysis enabled but time column '{time_col}' "
                    f"or subject column '{subject_col}' not found. Skipping."
                )
        except Exception as e:
            workflow.logger.warning(f"Longitudinal analysis failed: {e}")
    
    # 10. Ordination & Networks (original)
    run_constrained_ordination(workflow.adata, ['Genus'], workflow.plot_dir_beta, workflow.priority_vars)
    run_network_analysis(workflow.adata, ['Genus'], workflow.plot_dir_network)
    
    # 11. Beta Diversity (now includes UniFrac if available)
    run_beta_diversity_and_stats(workflow.adata, ['Genus'], workflow.plot_dir_beta, tree_path=tree_path, n_cpus=workflow.n_cpus)