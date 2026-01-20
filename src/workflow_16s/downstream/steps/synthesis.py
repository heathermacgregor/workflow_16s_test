import json
import pandas as pd
import plotly.express as px
from workflow_16s.downstream.utils import generate_synthesis_report

# Import enhancement modules
try:
    from workflow_16s.qc.visualization import (
        create_qc_impact_dashboard,
        create_qc_interpretation_report
    )
    QC_VIZ_AVAILABLE = True
except ImportError:
    QC_VIZ_AVAILABLE = False

try:
    from workflow_16s.downstream.statistics.top_features import (
        create_top_features_table,
        plot_top_features_heatmap,
        create_feature_consistency_plot,
        export_top_features_summary
    )
    TOP_FEATURES_AVAILABLE = True
except ImportError:
    TOP_FEATURES_AVAILABLE = False

# Import ML visualization module
try:
    from workflow_16s.downstream.ml_visualization import generate_comprehensive_ml_report
    ML_VIZ_AVAILABLE = True
except ImportError:
    ML_VIZ_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning("ML visualization module not available")


def run_results_synthesis(workflow):
    workflow.logger.info("5. Modular Synthesis: Aggregating Biomarkers...")
    targets = ['facility_match', 'facility_distance_km']
    master_list = []
    
    stats_p = workflow.plot_dir_stats / "significant_taxa_metadata_associations.csv"
    stats_df = pd.read_csv(stats_p) if stats_p.exists() else pd.DataFrame()
    
    for t in targets:
        stab_p = workflow.catboost_output_dir / f"strategy_stability_comparison_{t}.csv"
        if not stab_p.exists(): continue
        df_stab = pd.read_csv(stab_p)
        
        # High confidence = Group Validated + Statistically Significant
        robust = df_stab[df_stab['group_validated'] == '✅']['Taxon'].tolist()
        for taxon in robust:
            is_sig = not stats_df[(stats_df['metadata'] == t) & (stats_df['taxon'] == taxon)].empty if not stats_df.empty else False
            master_list.append({
                "Target": t, "Taxon": taxon, 
                "Evidence": "ML Validated" + (" & Stat-Sig" if is_sig else ""),
                "Confidence": "High" if is_sig else "Moderate"
            })
            
    if master_list:
        summary_df = pd.DataFrame(master_list)
        summary_df.to_csv(workflow.output_dir / "MASTER_BIOMARKER_SUMMARY.csv", index=False)
        _plot_master_summary(workflow, summary_df)
    
    # ========== ENHANCEMENTS: QC Visualization ==========
    if QC_VIZ_AVAILABLE and hasattr(workflow, 'qc_results') and workflow.qc_results:
        workflow.logger.info("Creating QC impact visualizations...")
        
        qc_output_dir = workflow.output_dir / "qc_visualizations"
        qc_output_dir.mkdir(exist_ok=True)
        
        try:
            # Create QC dashboard (before/after comparison)
            if hasattr(workflow, 'adata_pre_qc') and hasattr(workflow, 'adata'):
                create_qc_impact_dashboard(
                    workflow.adata_pre_qc,
                    workflow.adata,
                    workflow.qc_results,
                    qc_output_dir / "qc_impact_dashboard.html"
                )
            
            # Create interpretation report
            if hasattr(workflow, 'adata'):
                create_qc_interpretation_report(
                    workflow.adata,
                    workflow.qc_results,
                    qc_output_dir / "qc_interpretation_report.md"
                )
            
            workflow.logger.info(f"QC visualizations saved to: {qc_output_dir}")
        
        except Exception as e:
            workflow.logger.warning(f"QC visualization failed: {e}")
    
    # ========== ENHANCEMENTS: Top Features Summary ==========
    if TOP_FEATURES_AVAILABLE:
        workflow.logger.info("Creating top features summary...")
        
        # Collect all statistical test results
        stats_results = {}
        
        # Check for various stats results files
        if hasattr(workflow, 'stats_results'):
            stats_results = workflow.stats_results
        else:
            # Try to load from files
            stats_dir = workflow.plot_dir_stats
            
            for test_file in stats_dir.glob("*significant*.csv"):
                test_name = test_file.stem.replace("significant_", "")
                try:
                    df = pd.read_csv(test_file, index_col=0)
                    if not df.empty:
                        stats_results[test_name] = df
                except Exception:
                    continue
        
        if stats_results:
            try:
                # Get taxonomy if available
                taxonomy_df = None
                if hasattr(workflow, 'adata') and 'taxonomy' in workflow.adata.var.columns:
                    taxonomy_df = workflow.adata.var
                
                # Create top features table
                top_features_df = create_top_features_table(
                    stats_results,
                    n_top=30,
                    taxonomy_df=taxonomy_df,
                    sort_by='frequency'
                )
                
                if not top_features_df.empty:
                    # Export table
                    top_feat_dir = workflow.output_dir / "top_features"
                    top_feat_dir.mkdir(exist_ok=True)
                    
                    export_top_features_summary(
                        top_features_df,
                        top_feat_dir / "top_features_summary.csv"
                    )
                    
                    # Create visualizations
                    plot_top_features_heatmap(
                        top_features_df,
                        stats_results,
                        top_feat_dir / "top_features_heatmap.html"
                    )
                    
                    create_feature_consistency_plot(
                        top_features_df,
                        top_feat_dir / "feature_consistency.html"
                    )
                    
                    workflow.logger.info(f"Top features summary saved to: {top_feat_dir}")
            
            except Exception as e:
                workflow.logger.warning(f"Top features analysis failed: {e}")
    
    # ========== ENHANCEMENTS: ML Strategy Visualizations ==========
    if ML_VIZ_AVAILABLE and hasattr(workflow, 'catboost_output_dir'):
        workflow.logger.info("Creating comprehensive ML strategy visualizations...")
        
        ml_viz_dir = workflow.output_dir / "ml_visualizations"
        ml_viz_dir.mkdir(exist_ok=True)
        
        try:
            # Get ML targets
            ml_targets = ['facility_match', 'facility_distance_km']
            
            # Get additional grouping variables from config if available
            grouping_vars = []
            if hasattr(workflow, 'priority_categorical'):
                # Include relevant categorical variables
                relevant_cats = [
                    'facility_type', 'facility_status', 'contamination_status',
                    'nuclear_contamination_status', 'env_biome', 'env_feature'
                ]
                grouping_vars = [v for v in workflow.priority_categorical 
                               if v in relevant_cats and v not in ml_targets]
            
            # Generate comprehensive ML report
            ml_report = generate_comprehensive_ml_report(
                catboost_dir=workflow.catboost_output_dir,
                output_dir=ml_viz_dir,
                ml_targets=ml_targets,
                grouping_variables=grouping_vars,
                strategies=['baseline', 'agnostic', 'group_validated']
            )
            
            # Store report in workflow for HTML generation
            workflow.ml_viz_report = ml_report
            
            workflow.logger.info(f"✓ ML strategy visualizations saved to: {ml_viz_dir}")
            workflow.logger.info(f"  - Strategy comparison dashboards: {len(ml_report['strategy_comparisons'])}")
            workflow.logger.info(f"  - Group fingerprint plots: {sum(len(v) for v in ml_report['group_fingerprints'].values())}")
            workflow.logger.info(f"  - Multi-group comparisons: {len(ml_report['multi_group_comparisons'])}")
            workflow.logger.info(f"  - Batch effect visualizations: {len(ml_report['batch_effect_impacts'])}")
            
        except Exception as e:
            workflow.logger.warning(f"ML visualization failed: {e}")
            import traceback
            workflow.logger.debug(traceback.format_exc())
    
    generate_synthesis_report(workflow.output_dir)

def _plot_master_summary(workflow, df):
    fig = px.scatter(df, x="Target", y="Taxon", color="Confidence", title="Master Facility Biomarkers")
    workflow.plot_utils.save_plotly_fig(fig, workflow.output_dir / "master_biomarker_summary_plot")

def handle_strategy_impact_plot(workflow, target, results):
    """Guarded heatmap plotter to prevent dimensionality errors."""
    perf = []
    for name, data in results.items():
        scores = data.get("test_scores", {})
        if scores:
            perf.append({"Strategy": name, "MCC": scores.get("mcc", 0), "R2": scores.get("r2", 0)})
    
    if not perf: return
    df = pd.DataFrame(perf).set_index("Strategy")
    if df.empty or df.shape[1] == 0: return # Final Dimensionality Guard
    
    fig = px.imshow(df.values, x=df.columns.tolist(), y=df.index.tolist(), text_auto=True, 
                    title=f"Performance Stability: {target}", color_continuous_scale="RdBu_r")
    fig.update_traces(texttemplate="%{z:.3f}")
    workflow.plot_utils.save_plotly_fig(fig, workflow.catboost_output_dir / f"performance_comparison_{target}")