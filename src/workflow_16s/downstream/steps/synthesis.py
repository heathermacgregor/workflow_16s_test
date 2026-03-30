# file: downstream/steps/synthesis.py
# ==================================================================================== #

import json
import logging
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import List, Dict, Any

# Local Imports
from workflow_16s.downstream.utils import generate_synthesis_report

# Setup logger
logger = logging.getLogger('workflow_16s')

# Import enhancement modules with safety guards
try:
    from workflow_16s.qc.visualization import (
        create_qc_impact_dashboard,
        create_qc_interpretation_report
    )
    QC_VIZ_AVAILABLE = True
except ImportError:
    QC_VIZ_AVAILABLE = False

try:
    from workflow_16s.downstream.stats.top_features import (
        create_top_features_table,
        plot_top_features_heatmap,
        create_feature_consistency_plot,
        export_top_features_summary
    )
    TOP_FEATURES_AVAILABLE = True
except ImportError:
    TOP_FEATURES_AVAILABLE = False

try:
    from workflow_16s.downstream.machine_learning.visualization import generate_comprehensive_ml_report
    ML_VIZ_AVAILABLE = True
except ImportError:
    ML_VIZ_AVAILABLE = False
    logger.warning("ML visualization module not available")

# ============================ HELPER VISUALIZATIONS ================================= #

def plot_biomarker_stability_heatmap(
    summary_df: pd.DataFrame, 
    output_path: Path
) -> None:
    """
    Visualises which taxa are robust across multiple validation strategies.
    Pivots data to show Taxon vs Target, colored by the Stability Score.
    """
    if summary_df.empty:
        return

    # Pivot data: Genus vs Target, valued by Stability Score
    pivot_df = summary_df.pivot(
        index='Taxon', 
        columns='Target', 
        values='Stability_Score'
    ).fillna(0)
    
    fig = px.imshow(
        pivot_df,
        labels=dict(
            x="Prediction Target", 
            y="Microbial Genus", 
            color="Stability Score"
        ),
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale='Viridis',
        title="<b>Universal Biomarker Stability</b><br><sup>Higher score indicates consistency across LOPOCV, Spatial-CV, and Batch-Aware strategies</sup>"
    )
    
    fig.update_layout(
        title_x=0.5,
        template='plotly_white',
        margin=dict(l=250) # Extra margin for long Genus names
    )
    
    # Stylistic black border
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="black", width=2)
    )
    
    fig.write_html(str(output_path))

def _plot_master_summary(
    workflow,
    df: pd.DataFrame
) -> None:
    """Generates a scatter plot of biomarkers by target and confidence."""
    fig = px.scatter(
        df, 
        x="Target", 
        y="Taxon", 
        color="Confidence", 
        size="Stability_Score",
        hover_data=["Evidence"],
        title="<b>Master Forensic Biomarker Summary</b>",
        color_discrete_map={"Elite": "#FFD700", "High": "#2ca02c", "Moderate": "#ff7f0e"}
    )
    fig.update_layout(template='plotly_white')
    workflow.plot_utils.save_plotly_fig(fig, workflow.output_dir / "master_biomarker_summary_plot")

# ============================ MAIN SYNTHESIS MODULE ================================= #

def run_results_synthesis(workflow) -> None:
    """
    Final synthesis module. Aggregates results from Statistics and ML Discovery
    to find 'Universal Core' microbial signatures of the Nuclear Fuel Cycle.
    """
    workflow.logger.info("［05］Modular Synthesis: Aggregating Biomarkers & Defensive Scores...")
    
    #［01］Targets defined by the latest configuration
    targets = [
        'facility_match', 'facility_type', 'facility_status', 
        'nuclear_contamination_status', 'ph', 'temperature_c', 'salinity_psu'
    ]
    master_list = []
    
    # Load statistical significance baseline from the stats module
    stats_p = workflow.plot_dir_stats / "significant_taxa_metadata_associations.csv"
    stats_df = pd.read_csv(stats_p) if stats_p.exists() else pd.DataFrame()
    
    for t in targets:
        stab_p = workflow.catboost_output_dir / f"strategy_stability_comparison_{t}.csv"
        if not stab_p.exists(): 
            continue
            
        df_stab = pd.read_csv(stab_p)
        
        #［02］Modernized Scoring Logic (Defensive Weighting)
        for _, row in df_stab.iterrows():
            taxon = row['Taxon']
            score = 0
            evidence = []
            
            # Baseline (Random Split) - Minimal weight
            if row.get('baseline') == '✅': score += 1; evidence.append("Baseline")
            # Batch-Aware (Feature injection)
            if row.get('batch_aware') == '✅': score += 3; evidence.append("Batch-Aware")
            # LOPOCV (Cross-Project validity)
            if row.get('lopocv') == '✅': score += 5; evidence.append("LOPOCV")
            # Spatial-CV (Geographic validity)
            if row.get('spatial_cv') == '✅': score += 5; evidence.append("Spatial-CV")
            # Meta-Aware (Environmental context)
            if row.get('meta_aware') == '✅': score += 2; evidence.append("Meta-Aware")

            # Check for Statistical Significance
            is_sig = False
            if not stats_df.empty:
                # Matches if the taxon is significant for this specific metadata target
                is_sig = not stats_df[(stats_df['metadata'] == t) & (stats_df['taxon'] == taxon)].empty
            
            #［03］Filtering and Confidence Classification
            # Logic: If it only exists in Baseline (Score 1), it's likely noise/lab effect.
            # We require a score >= 4 to prove it survived defensive testing.
            if score >= 4: 
                master_list.append({
                    "Target": t, 
                    "Taxon": taxon, 
                    "Stability_Score": score,
                    "Evidence": " + ".join(evidence) + (" & Stat-Sig" if is_sig else ""),
                    # 'Elite' is the highest tier: Robust across continents and labs + Stat Sig.
                    "Confidence": "Elite" if (score >= 10 and is_sig) else ("High" if score >= 7 else "Moderate")
                })
            
    if master_list:
        summary_df = pd.DataFrame(master_list)
        summary_df = summary_df.sort_values(by=["Target", "Stability_Score"], ascending=[True, False])
        summary_df.to_csv(workflow.output_dir / "MASTER_BIOMARKER_SUMMARY.csv", index=False)
        
        # Generate synthesis plots
        _plot_master_summary(
            workflow, 
            summary_df
        )
        plot_biomarker_stability_heatmap(
            summary_df,
            workflow.output_dir / "biomarker_stability_heatmap.html"
        )
        
        workflow.logger.info(f"［🔍］Identified {len(summary_df)} robust biomarkers.")

    # QC Visualization
    if QC_VIZ_AVAILABLE and hasattr(workflow, 'qc_results') and workflow.qc_results:
        workflow.logger.info("［📊］Creating QC impact visualizations...")
        qc_output_dir = workflow.output_dir / "qc_visualizations"
        qc_output_dir.mkdir(exist_ok=True)
        
        try:
            if hasattr(workflow, 'adata_pre_qc') and hasattr(workflow, 'adata'):
                create_qc_impact_dashboard( # type: ignore
                    workflow.adata_pre_qc,
                    workflow.adata,
                    workflow.qc_results,
                    qc_output_dir / "qc_impact_dashboard.html"
                )
            
            if hasattr(workflow, 'adata'):
                create_qc_interpretation_report( # type: ignore
                    workflow.adata,
                    workflow.qc_results,
                    qc_output_dir / "qc_interpretation_report.md"
                )
        except Exception as e:
            workflow.logger.warning(f"［❌］QC visualization failed: {e}")
    
    # ========== ENHANCEMENTS: Top Features Summary ==========
    if TOP_FEATURES_AVAILABLE:
        workflow.logger.info("［📝］Creating top features summary...")
        stats_results = {}
        
        # Collect results from file system if not in memory
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
                taxonomy_df = workflow.adata.var if hasattr(workflow, 'adata') else None
                top_features_df = create_top_features_table(  # type: ignore
                    stats_results, n_top=30, taxonomy_df=taxonomy_df)
                
                if not top_features_df.empty:
                    top_feat_dir = workflow.output_dir / "top_features"
                    top_feat_dir.mkdir(exist_ok=True)
                    
                    export_top_features_summary( # type: ignore
                        top_features_df, 
                        top_feat_dir / "top_features_summary.csv"
                    )
                    plot_top_features_heatmap( # type: ignore
                        top_features_df, 
                        stats_results, 
                        top_feat_dir / "top_features_heatmap.html"
                    )
                    create_feature_consistency_plot( # type: ignore
                        top_features_df, 
                        top_feat_dir / "feature_consistency.html"
                    )
            except Exception as e:
                workflow.logger.warning(f"［❌］Top features analysis failed: {e}")
    
    # ========== ENHANCEMENTS: ML Strategy Visualizations ==========
    if ML_VIZ_AVAILABLE and hasattr(workflow, 'catboost_output_dir'):
        workflow.logger.info("［📊］Creating comprehensive ML strategy visualizations...")
        ml_viz_dir = workflow.output_dir / "ml_visualizations"
        ml_viz_dir.mkdir(exist_ok=True)
        
        current_strategies = [
            'baseline', 'batch_aware', 'lopocv', 'spatial_cv', 'meta_aware'
        ]
        
        try:
            # We pass the full target list to the viz module
            ml_report = generate_comprehensive_ml_report( # type: ignore
                catboost_dir=workflow.catboost_output_dir,
                output_dir=ml_viz_dir,
                ml_targets=targets,
                strategies=current_strategies
            )
            if isinstance(ml_report, dict):
                workflow.ml_viz_report = ml_report
                workflow.logger.info(f"［💾］ML visualizations saved to: {ml_viz_dir}")
        except Exception as e:
            workflow.logger.warning(f"［❌］ML visualization failed: {e}")
    
    # Finalize HTML report structure
    generate_synthesis_report(workflow.output_dir)

def handle_strategy_impact_plot(workflow, target, results):
    """Guarded performance comparison heatmap."""
    perf = []
    for name, data in results.items():
        scores = data.get("test_scores", {})
        if scores:
            perf.append({
                "Strategy": name, 
                "MCC": scores.get("mcc", 0), 
                "R2": scores.get("r2", 0)}
            )
    
    if not perf: return
    df = pd.DataFrame(perf).set_index("Strategy")
    if df.empty or df.shape[1] == 0: return 
    
    fig = px.imshow(
        df.values, 
        x=df.columns.tolist(),
        y=df.index.tolist(), 
        text_auto=True, 
        title=f"Performance Stability: {target}", 
        color_continuous_scale="RdBu_r"
    )
    fig.update_traces(texttemplate="%{z:.3f}")
    workflow.plot_utils.save_plotly_fig(
        fig, 
        workflow.catboost_output_dir / f"performance_comparison_{target}"
    )