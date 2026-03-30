# workflow_16s/visualization/machine_learning/features.py

import re
from asyncio.log import logger
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from workflow_16s.utils.logger import with_logger
from workflow_16s.visualization.utils import PlottingUtils


@with_logger
def generate_study_overlap_matrix(
    study_feature_map: Dict[str, List[str]],  output_dir: Path,
    n: int = 10, n_matrix: int = 30
) -> None:
    """Generates an interactive heatmap showing shared biomarkers across top studies."""
    # Select top n studies by count for display
    top_studies = sorted(
        study_feature_map.keys(), 
        key=lambda x: len(study_feature_map[x]), 
        reverse=True
    )[:n]
    all_feats = sorted(list(set([f for s in top_studies for f in study_feature_map[s]])))
    
    if not all_feats: return

    # Binary Presence/Absence matrix
    matrix = pd.DataFrame(0, index=all_feats, columns=top_studies)
    for study in top_studies:
        for feat in study_feature_map[study]:
            matrix.loc[feat, study] = 1

    matrix['Total'] = matrix.sum(axis=1)
    matrix = matrix.sort_values('Total', ascending=False).head(n_matrix) # Display top n_matrix most shared

    fig = px.imshow(
        matrix.drop(columns='Total'),
        labels=dict(x="Independent Studies", y="Top Microbial Features", color="Presence"),
        x=top_studies,
        y=[f.split('__')[-1].replace('_', ' ') for f in matrix.index],
        color_continuous_scale=[[0, 'white'], [1, '#4B0082']],
        title="Meta-Analysis Consensus: Biomarker Overlap Across Boundaries"
    )
    
    fig.update_layout(template='plotly_white', coloraxis_showscale=False, height=800)
    fig.write_html(str(output_dir / "study_consensus_matrix.html"))
    logger.info(f"Meta-consensus visualization saved to {output_dir}")

@with_logger
def plot_feature_importances(
    importance_df: pd.DataFrame, 
    plot_dir_ml: Path, 
    target_name: str, 
    level: str, 
    model_score: float, 
    score_name: str
):
    """
    Plots feature importances, prioritizing 'Adjusted Importance' from Meta-Consensus
    if available. High-confidence universal biomarkers are highlighted.
    """
    plot_utils = PlottingUtils(logger)
    # 1. Determine which importance metric to use
    use_robust = 'Adjusted_Importance' in importance_df.columns
    x_col = 'Adjusted_Importance' if use_robust else 'Importance'
    
    # 2. Prepare title and sorting
    title = f"Top Predictive Taxa ({level}) for {target_name}"
    if use_robust:
        title += "<br><sub>Ranked by Meta-Consensus Adjusted Importance</sub>"
    else:
        title += f"<br><sub>Model OOB {score_name}: {model_score:.3f}</sub>"

    importance_df = importance_df.sort_values(by=x_col, ascending=True)

    try:
        # Create bar chart
        fig = px.bar(
            importance_df, 
            x=x_col, 
            y='Taxon', 
            color='Is_Universal' if 'Is_Universal' in importance_df.columns else None,
            color_discrete_map={True: '#4B0082', False: '#636efa'}, # Universal = Deep Indigo
            title=title, 
            orientation='h'
        )
        
        # Add 'Universal' badge to labels if applicable
        if 'Is_Universal' in importance_df.columns:
            new_labels = [
                f"🌟 {row['Taxon']}" if row['Is_Universal'] else row['Taxon'] 
                for _, row in importance_df.iterrows()
            ]
            fig.update_layout(yaxis=dict(tickmode='array', tickvals=importance_df['Taxon'], ticktext=new_labels))

        fig.update_layout(
            yaxis_title=f"Taxon ({level})", 
            xaxis_title="Robustness-Adjusted Importance" if use_robust else "Feature Importance", 
            height=max(400, len(importance_df) * 25),
            template='plotly_white'
        )
        
        safe_target_name = re.sub(r'[^A-Za-z0-9_]+', '', target_name)
        plot_path = plot_dir_ml / f"feature_importance_{safe_target_name}_{level}.html"
        plot_utils.save_plotly_fig(fig, plot_path)
        
    except Exception as e: 
        logger.error(f"Failed feature importance plot for {target_name}: {e}")

@with_logger
def generate_comprehensive_ml_report(
    output_dir: Union[str, Path],
    report_name: str = "ML_Discovery_Report",
    ml_targets: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
    catboost_dir: Optional[Path] = None, # <--- ADD THIS PARAMETER
    **kwargs # Catch-all for extra synthesis arguments
) -> Path:
    """
    Compiles individual ML runs into a structured Forensic Discovery folder.
    
    This function:
    1. Aggregates top features across different CV strategies.
    2. Identifies 'Consensus Biomarkers' (taxa that appear in multiple strategies).
    3. Organizes all PNG/HTML artifacts for the final Dashboard.
    """
    plot_utils = PlottingUtils(logger)
    # Use catboost_dir if provided, otherwise fallback to output_dir
    source_dir = catboost_dir if catboost_dir else output_dir
    out_path = Path(output_dir)
    report_dir = out_path / report_name
    report_dir.mkdir(parents=True, exist_ok=True)
    
    ml_targets = ml_targets or []
    strategies = strategies or ["agnostic", "meta_analysis", "lopocv"]
    
    logger.info(f"📊 Assembling Comprehensive Discovery Report for {len(ml_targets)} targets.")

    summary_records = []

    for target in ml_targets:
        target_summary_dir = report_dir / f"Target_{target}"
        target_summary_dir.mkdir(parents=True, exist_ok=True)
        
        target_features = {}

        for strategy in strategies:
            # Look for the top_features.csv generated by each strategy
            strategy_file = out_path / strategy / f"Genus_{target}" / "top_features.csv"
            
            if strategy_file.exists():
                df = pd.read_csv(strategy_file)
                # Normalize column names for merging
                df.columns = [c.lower() for c in df.columns]
                
                if 'feature' in df.columns:
                    top_5 = df.head(5)['feature'].tolist()
                    target_features[strategy] = top_5
                    
                    # Copy main plots to the summary directory for easy dashboard access
                    _copy_key_plots(out_path / strategy / f"Genus_{target}", target_summary_dir, strategy)

        # Calculate Consensus Score
        if target_features:
            consensus_df = _calculate_consensus(target_features)
            consensus_df.to_csv(target_summary_dir / f"{target}_consensus_biomarkers.csv", index=False)
            
            logger.info(f"✅ Target '{target}': Found {len(consensus_df)} consensus biomarkers.")

    logger.info(f"✨ Discovery Report assembled: {report_dir}")
    return report_dir

def _calculate_consensus(strategy_map: Dict[str, List[str]]) -> pd.DataFrame:
    """Identifies features that appear across different ML strategies."""
    all_features = []
    for strategy, features in strategy_map.items():
        for f in features:
            all_features.append({"feature": f, "strategy": strategy})
    
    df = pd.DataFrame(all_features)
    if df.empty:
        return pd.DataFrame(columns=['feature', 'frequency', 'strategies'])
        
    consensus = df.groupby('feature').agg(
        frequency=('strategy', 'count'),
        strategies=('strategy', lambda x: ', '.join(x))
    ).reset_index().sort_values('frequency', ascending=False)
    
    return consensus

def _copy_key_plots(src_dir: Path, dest_dir: Path, strategy_prefix: str):
    """Gathers SHAP and ROC plots into a central report folder."""
    # Key plots we want to show in the final dashboard
    plot_patterns = ["*shap_beeswarm*", "*roc_curve*", "*batch_diagnostic*"]
    
    for pattern in plot_patterns:
        for plot_file in src_dir.glob(pattern):
            new_name = f"{strategy_prefix}_{plot_file.name}"
            shutil.copy2(plot_file, dest_dir / new_name)