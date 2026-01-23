import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import scanpy as sc
from workflow_16s.downstream.diversity import (
    run_alpha_diversity, run_beta_diversity_and_stats, run_taxa_metadata_statistics, 
    run_constrained_ordination, run_network_analysis,
    run_community_state_typing
)
from workflow_16s.downstream.machine_learning import (
    run_machine_learning_analysis, run_catboost_selection
)
from workflow_16s.downstream.utils import AnalysisUtils
from workflow_16s.downstream.steps.synthesis import handle_strategy_impact_plot
from workflow_16s.downstream.diversity.phylogenetic import phylogenetic_diversity_workflow
from workflow_16s.downstream.statistics import compare_da_methods, consensus_da_features
from workflow_16s.downstream.networks import network_analysis_workflow
from workflow_16s.downstream.qc import profile_metadata, generate_html_report

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
    
    # Ensure indices are unique BEFORE any analysis runs.
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
    # Check counts
    # ... inside run_analysis_suite ...

    # Check counts (Validation Logic)
    # Use .info() and convert the Series to string so it logs readable text
    workflow.logger.info("--- Facility Match Counts (Nuclear Only) ---")
    workflow.logger.info("\n" + str(workflow.adata.obs['facility_match'].value_counts()))
    
    #workflow.logger.info("--- Industry Match Counts (Any Facility) ---")
    #workflow.logger.info("\n" + str(workflow.adata.obs['industry_match'].value_counts()))

    # Verify logic: No sample should be BOTH 'facility_match' (Nuclear) AND 'analog_match'
    #overlaps = workflow.adata.obs[
    #    workflow.adata.obs['facility_match'] & workflow.adata.obs['analog_match']
    #]
    
    #if len(overlaps) > 0:
    #    workflow.logger.error(f"CRITICAL LOGIC ERROR: {len(overlaps)} samples are marked as BOTH Nuclear and Analog!")
    #else:
    #    workflow.logger.info("Logic Check Passed: No overlaps between Nuclear and Analog assignments.")

    tree_file = workflow.output_dir / "all_features.tree"
    tree_path = tree_file if tree_file.exists() else None
    
    # Check if tree is needed for phylogenetic diversity
    phylo_config = getattr(workflow.config, 'phylogeny', None)
    phylo_enabled = getattr(phylo_config, 'enabled', False) if phylo_config else False
    
    # Handle missing or incomplete trees using configurable strategy
    if phylo_enabled and not tree_path:
        workflow.logger.info("Phylogenetic diversity enabled but no tree file found")
        workflow.logger.info("Attempting to handle missing tree...")
        
        from workflow_16s.downstream.utils import handle_missing_tree
        
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
    cst_enabled = False
    if cst_enabled:
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
    alpha_diversity_enabled = False
    if alpha_diversity_enabled:
        run_alpha_diversity(
            workflow.adata, 
            workflow.plot_dir_alpha, 
            tree_path=tree_path, 
            priority_categorical=workflow.priority_categorical, 
            priority_numeric=workflow.priority_numeric
        )
    
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
    
    # 5. Machine Learning Matrix Execution
    ml_config = getattr(workflow.config, 'ml', None)
    
    if ml_config and ml_config.enabled:
        # 1. Get Grid Settings
        grid = getattr(ml_config, 'grid_settings', None)
        levels = getattr(grid, 'levels', ["Genus"]) if grid else ["Genus"]
        transforms = getattr(grid, 'transformations', ["clr"]) if grid else ["clr"]
        fs_strategies = getattr(grid, 'fs_strategies', ["baseline"]) if grid else ["baseline"]
        
        # 2. Get Targets
        strict_targets = ml_config.strict_targets
        config_targets = ml_config.targets
        
        if strict_targets and config_targets:
            targets = config_targets
        else:
            targets = ['facility_match', 'facility_distance_km']

        # 3. Setup Caching Directory
        # We store intermediate files here to avoid re-aggregating 32GB of data
        cache_dir = workflow.output_dir / ".cache" / "intermediate_data"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        run_context = getattr(workflow, 'run_context', None) or 'default_run'
        
        workflow.logger.info(f"🚀 Starting ML Matrix: {len(levels)} Levels x {len(transforms)} Transforms x {len(targets)} Targets")

        # --- MATRIX LOOP ---
        for lvl in levels:
            # --- CACHE TIER 1: AGGREGATION (The Slow Part) ---
            agg_cache_path = cache_dir / f"aggregated_{lvl}.h5ad"
            adata_lvl = None
            
            if agg_cache_path.exists():
                workflow.logger.info(f"⚡ Loading cached aggregation for {lvl}...")
                try:
                    adata_lvl = sc.read_h5ad(agg_cache_path)
                except Exception as e:
                    workflow.logger.warning(f"Failed to load cached aggregation: {e}")
            
            if adata_lvl is None:
                workflow.logger.info(f"🔨 Aggregating data to {lvl} (this may take time)...")
                adata_lvl = AnalysisUtils.get_analysis_adata(workflow.adata, level=lvl)
                
                if adata_lvl is not None:
                    try:
                        adata_lvl.write_h5ad(agg_cache_path)
                        workflow.logger.info(f"💾 Cached aggregation to {agg_cache_path.name}")
                    except Exception as e:
                        workflow.logger.warning(f"Could not cache aggregation: {e}")
            
            if adata_lvl is None: 
                workflow.logger.warning(f"Skipping level {lvl}: aggregation failed.")
                continue

            for trans in transforms:
                # --- CACHE TIER 2: TRANSFORMATION (The Fast Part) ---
                trans_cache_path = cache_dir / f"transformed_{lvl}_{trans}.pkl"
                X_transformed = None
                
                if trans_cache_path.exists():
                    workflow.logger.info(f"  ⚡ Loading cached {trans} transform...")
                    try:
                        X_transformed = pd.read_pickle(trans_cache_path)
                    except Exception:
                        pass
                
                if X_transformed is None:
                    workflow.logger.info(f"  ⚗️  Applying {trans} transform...")
                    X_transformed = AnalysisUtils.apply_transform(adata_lvl, method=trans)
                    try:
                        X_transformed.to_pickle(trans_cache_path)
                    except Exception: pass
                
                # Define Output Path context
                context_path = f"{lvl}/{trans}"
                
                for target in targets:
                    
                    # B. Run Feature Selection (CatBoost)
                    catboost_base = workflow.catboost_output_dir / lvl / trans
                    catboost_base.mkdir(parents=True, exist_ok=True)
                    
                    run_catboost_selection(
                        adata=workflow.adata,       
                        X_custom=X_transformed,     
                        catboost_output_dir=catboost_base,
                        level=lvl,
                        priority_targets=[target],
                        strict_targets=True,        
                        strategies=fs_strategies,   
                        n_cpus=workflow.n_cpus,
                        batch_col='batch_original', 
                        run_context=run_context
                    )
                    
                    # C. Run Main ML Analysis (RandomForest / CatBoost)
                    ml_out = workflow.plot_dir_ml / lvl / trans
                    ml_out.mkdir(parents=True, exist_ok=True)
                    
                    batch_cfg = ml_config.batch_covariates.dict() if ml_config.batch_covariates else {}
                    batch_cfg['output_subdir'] = context_path 
                    
                    run_machine_learning_analysis(
                        adata=workflow.adata,       
                        X_custom=X_transformed,     
                        plot_dir_ml=ml_out,
                        level=lvl,
                        priority_targets=[target],
                        strict_targets=True,        
                        batch_config=batch_cfg,
                        ml_config=ml_config         
                    )

        workflow.priority_vars = targets
        workflow.logger.info("✅ ML Matrix Execution Complete.")
    
    # 5.7. Core Statistical Tests
    taxa_metadata_statistics_enabled = False
    if taxa_metadata_statistics_enabled:
        run_taxa_metadata_statistics(
            workflow.adata, 
            ['Genus'], 
            workflow.plot_dir_stats, 
            workflow.n_cpus,
            max_taxa=getattr(workflow.config, 'max_taxa_stats', 1000),
            max_categorical=getattr(workflow.config, 'max_categorical_stats', 30)
        )
    
    # 5.8. Compositional Network Analysis (NEW - if enabled)
    # Infers co-occurrence networks accounting for compositionality
    # WHY: Standard correlation invalid for compositional data
    network_config = getattr(workflow.config, 'networks', None)
    network_enabled = getattr(network_config, 'enabled', False) if network_config else False
    network_enabled = False
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
    
    # 5.9. Longitudinal Analysis (NEW - if enabled and time-series data)
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
    
    # 5.10. Ordination & Networks (original)
    ord_config = getattr(workflow.config, 'ordination', None)
    ord_enabled = getattr(ord_config, 'enabled', False) if ord_config else False
    
    net_config = getattr(workflow.config, 'legacy_networks', None)
    net_enabled = getattr(net_config, 'enabled', False) if net_config else False

    if ord_enabled:
        workflow.logger.info("Running constrained ordination (CCA/RDA)...")
        run_constrained_ordination(
            workflow.adata, 
            ['Genus'], 
            workflow.plot_dir_beta, 
            workflow.priority_vars
        )

    if net_enabled:
        workflow.logger.info("Running legacy network analysis...")
        run_network_analysis(
            workflow.adata, 
            ['Genus'], 
            workflow.plot_dir_network
        )
        
    # 5.11. Beta Diversity (now includes UniFrac if available)
    beta_config = getattr(workflow.config, 'beta_diversity', None)
    beta_enabled = getattr(beta_config, 'enabled', False) if beta_config else False
    
    # Use existing Stats config (defined as Dict in schema)
    stats_config = getattr(workflow.config, 'stats', {})
    stats_enabled = stats_config.get('enabled', False) if isinstance(stats_config, dict) else False

    if beta_enabled:
        workflow.logger.info("Running beta diversity and statistics...")
        run_beta_diversity_and_stats(
            workflow.adata, 
            ['Genus'], 
            workflow.plot_dir_beta, 
            tree_path=tree_path, 
            n_cpus=workflow.n_cpus
        )