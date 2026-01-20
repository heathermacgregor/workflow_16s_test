"""
Enhanced ML Visualization Module
=================================
Creates comprehensive, interpretable visualizations of CatBoost feature selection
results across different strategies, groupings, and batch effect approaches.

This module generates:
1. Strategy comparison plots (baseline vs agnostic vs group_validated)
2. Group fingerprint plots (facility_match vs facility_type vs contamination)
3. Batch effect impact visualizations
4. Feature stability across strategies
5. Interactive dashboards for model comparison
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


# ============================================================================
# STRATEGY COMPARISON VISUALIZATIONS
# ============================================================================

def create_strategy_comparison_dashboard(
    catboost_dir: Path,
    target_variable: str,
    output_dir: Path,
    strategies: List[str] = ['baseline', 'agnostic', 'group_validated']
) -> Optional[go.Figure]:
    """
    Create comprehensive dashboard comparing all three batch correction strategies.
    
    Parameters
    ----------
    catboost_dir : Path
        Base directory containing strategy subdirectories
    target_variable : str
        Target variable name (e.g., 'facility_match')
    output_dir : Path
        Directory to save output figures
    strategies : List[str]
        List of strategy names to compare
        
    Returns
    -------
    go.Figure or None
        Plotly figure with strategy comparison dashboard
    """
    logger.info(f"Creating strategy comparison dashboard for {target_variable}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect results for each strategy
    strategy_results = {}
    for strategy in strategies:
        strategy_dir = catboost_dir / strategy / f"Genus_{target_variable}" / "shap"
        
        # Load results summary
        summary_file = strategy_dir.parent / "results_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                strategy_results[strategy] = json.load(f)
        else:
            logger.warning(f"No results summary found for {strategy} strategy")
    
    if not strategy_results:
        logger.error(f"No strategy results found for {target_variable}")
        return None
    
    # Create 2x2 subplot dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Model Performance Comparison',
            'Feature Importance Stability',
            'Batch Effect Impact',
            'Sample Size & Class Balance'
        ),
        specs=[[{'type': 'bar'}, {'type': 'scatter'}],
               [{'type': 'heatmap'}, {'type': 'bar'}]]
    )
    
    # === Panel 1: Model Performance Comparison ===
    metrics = ['accuracy', 'mcc', 'roc_auc', 'f1']
    for strategy in strategies:
        if strategy not in strategy_results:
            continue
        test_scores = strategy_results[strategy].get('test_scores', {})
        values = [test_scores.get(m, 0) for m in metrics]
        
        fig.add_trace(
            go.Bar(
                name=strategy.replace('_', ' ').title(),
                x=metrics,
                y=values,
                text=[f"{v:.3f}" for v in values],
                textposition='auto',
            ),
            row=1, col=1
        )
    
    # === Panel 2: Feature Importance Stability ===
    # Load top features for each strategy
    top_features_by_strategy = {}
    for strategy in strategies:
        if strategy not in strategy_results:
            continue
        strategy_dir = catboost_dir / strategy / f"Genus_{target_variable}" / "shap"
        feature_file = strategy_dir / "top_features.csv"
        
        if feature_file.exists():
            df = pd.read_csv(feature_file)
            top_features_by_strategy[strategy] = set(df.head(20)['feature'].tolist())
    
    # Calculate pairwise Jaccard similarity
    if len(top_features_by_strategy) >= 2:
        strategy_names = list(top_features_by_strategy.keys())
        for i, strat1 in enumerate(strategy_names):
            for j, strat2 in enumerate(strategy_names):
                if i < j:  # Only upper triangle
                    set1 = top_features_by_strategy[strat1]
                    set2 = top_features_by_strategy[strat2]
                    jaccard = len(set1 & set2) / len(set1 | set2) if set1 or set2 else 0
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[strat1],
                            y=[strat2],
                            mode='markers+text',
                            marker=dict(size=jaccard*100, color=jaccard,
                                       colorscale='Viridis', showscale=True,
                                       colorbar=dict(title="Jaccard<br>Similarity")),
                            text=f"{jaccard:.2f}",
                            textposition='middle center',
                            showlegend=False
                        ),
                        row=1, col=2
                    )
    
    # === Panel 3: Batch Effect Impact Heatmap ===
    # Show how batch variables correlate with target for each strategy
    batch_cols = ['batch_original', 'dataset_id', 'study_accession', 'sequencing_center']
    batch_impact_data = []
    
    for strategy in strategies:
        if strategy not in strategy_results:
            continue
        feature_importance = strategy_results[strategy].get('feature_importance', {})
        
        for col in batch_cols:
            importance = feature_importance.get(col, 0)
            batch_impact_data.append({
                'Strategy': strategy.replace('_', ' ').title(),
                'Batch Variable': col,
                'Importance': importance
            })
    
    if batch_impact_data:
        df_batch = pd.DataFrame(batch_impact_data)
        pivot = df_batch.pivot(index='Batch Variable', columns='Strategy', values='Importance').fillna(0)
        
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale='RdBu_r',
                text=pivot.values.round(3),
                texttemplate='%{text}',
                showscale=True,
                colorbar=dict(title="Feature<br>Importance")
            ),
            row=2, col=1
        )
    
    # === Panel 4: Sample Size & Class Balance ===
    for strategy in strategies:
        if strategy not in strategy_results:
            continue
        sample_info = strategy_results[strategy].get('sample_info', {})
        
        fig.add_trace(
            go.Bar(
                name=strategy.replace('_', ' ').title(),
                x=['Train', 'Test'],
                y=[sample_info.get('n_train', 0), sample_info.get('n_test', 0)],
                text=[str(sample_info.get('n_train', 0)), str(sample_info.get('n_test', 0))],
                textposition='auto',
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=1000,
        width=1600,
        title_text=f"CatBoost Strategy Comparison: {target_variable.replace('_', ' ').title()}",
        title_font_size=20,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Metric", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_xaxes(title_text="Strategy", row=1, col=2)
    fig.update_yaxes(title_text="Strategy", row=1, col=2)
    fig.update_xaxes(title_text="Split", row=2, col=2)
    fig.update_yaxes(title_text="Samples", row=2, col=2)
    
    # Save figure
    output_file = output_dir / f"strategy_comparison_dashboard_{target_variable}.html"
    fig.write_html(str(output_file))
    logger.info(f"Saved strategy comparison dashboard: {output_file}")
    
    return fig


# ============================================================================
# GROUP FINGERPRINT VISUALIZATIONS
# ============================================================================

def create_group_fingerprint_comparison(
    catboost_dir: Path,
    grouping_variables: List[str],
    output_dir: Path,
    strategy: str = 'agnostic'
) -> Dict[str, go.Figure]:
    """
    Create fingerprint comparison plots for different grouping variables.
    
    Shows how microbial biomarkers differ across:
    - facility_match (True vs False)
    - facility_type (different facility types)
    - contamination status
    - etc.
    
    Parameters
    ----------
    catboost_dir : Path
        Base directory containing CatBoost results
    grouping_variables : List[str]
        List of grouping variables to compare
    output_dir : Path
        Directory to save output figures
    strategy : str
        Which strategy to use for comparison (default: 'agnostic')
        
    Returns
    -------
    Dict[str, go.Figure]
        Dictionary mapping grouping variable to figure
    """
    logger.info(f"Creating group fingerprint comparisons using {strategy} strategy...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    for grouping_var in grouping_variables:
        logger.info(f"Processing {grouping_var}...")
        
        # Load top features for this grouping
        strategy_dir = catboost_dir / strategy / f"Genus_{grouping_var}" / "shap"
        
        if not strategy_dir.exists():
            logger.warning(f"No results found for {grouping_var} in {strategy} strategy")
            continue
        
        # Load feature importance
        feature_file = strategy_dir / "top_features.csv"
        if not feature_file.exists():
            logger.warning(f"No feature file found: {feature_file}")
            continue
        
        df = pd.read_csv(feature_file)
        top_n = min(30, len(df))
        df_top = df.head(top_n)
        
        # Create horizontal bar plot
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=df_top['feature'],
            x=df_top['importance'],
            orientation='h',
            marker=dict(
                color=df_top['importance'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="SHAP<br>Importance")
            ),
            text=df_top['importance'].round(3),
            textposition='auto',
        ))
        
        fig.update_layout(
            title=f"Microbial Fingerprint: {grouping_var.replace('_', ' ').title()}<br><sub>Top {top_n} Features ({strategy} strategy)</sub>",
            xaxis_title="SHAP Importance",
            yaxis_title="Genus",
            height=max(600, top_n * 25),
            width=1000,
            yaxis=dict(autorange="reversed")  # Top feature at top
        )
        
        # Save figure
        output_file = output_dir / f"fingerprint_{grouping_var}_{strategy}.html"
        fig.write_html(str(output_file))
        logger.info(f"Saved fingerprint plot: {output_file}")
        
        figures[grouping_var] = fig
    
    return figures


def create_multi_group_comparison_heatmap(
    catboost_dir: Path,
    grouping_variables: List[str],
    output_dir: Path,
    strategy: str = 'agnostic',
    top_n: int = 20
) -> Optional[go.Figure]:
    """
    Create heatmap comparing top features across multiple grouping variables.
    
    Shows which genera are important across different comparisons (facility match,
    facility type, contamination status, etc.)
    
    Parameters
    ----------
    catboost_dir : Path
        Base directory containing CatBoost results
    grouping_variables : List[str]
        List of grouping variables to compare
    output_dir : Path
        Directory to save output figure
    strategy : str
        Which strategy to use (default: 'agnostic')
    top_n : int
        Number of top features to include per grouping
        
    Returns
    -------
    go.Figure or None
        Plotly heatmap figure
    """
    logger.info(f"Creating multi-group comparison heatmap...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect top features for each grouping
    all_features = set()
    importance_matrix = {}
    
    for grouping_var in grouping_variables:
        strategy_dir = catboost_dir / strategy / f"Genus_{grouping_var}" / "shap"
        feature_file = strategy_dir / "top_features.csv"
        
        if not feature_file.exists():
            logger.warning(f"No feature file found for {grouping_var}")
            continue
        
        df = pd.read_csv(feature_file).head(top_n)
        
        # Store importances
        importance_matrix[grouping_var] = df.set_index('feature')['importance'].to_dict()
        all_features.update(df['feature'].tolist())
    
    if not importance_matrix:
        logger.error("No feature data found for any grouping variable")
        return None
    
    # Create DataFrame for heatmap
    df_matrix = pd.DataFrame(
        {group: [importance_matrix[group].get(feat, 0) for feat in all_features]
         for group in importance_matrix.keys()},
        index=list(all_features)
    )
    
    # Sort by total importance across all groupings
    df_matrix['total'] = df_matrix.sum(axis=1)
    df_matrix = df_matrix.sort_values('total', ascending=False).drop('total', axis=1)
    
    # Take top N features overall
    df_matrix = df_matrix.head(top_n * 2)  # Show more features for cross-comparison
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=df_matrix.values,
        x=[col.replace('_', ' ').title() for col in df_matrix.columns],
        y=df_matrix.index,
        colorscale='Viridis',
        text=df_matrix.values.round(3),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="SHAP<br>Importance")
    ))
    
    fig.update_layout(
        title=f"Microbial Biomarker Cross-Comparison<br><sub>{len(df_matrix)} top features across {len(importance_matrix)} groupings ({strategy} strategy)</sub>",
        xaxis_title="Grouping Variable",
        yaxis_title="Genus",
        height=max(800, len(df_matrix) * 20),
        width=max(1000, len(importance_matrix) * 150),
        yaxis=dict(autorange="reversed")
    )
    
    # Save figure
    output_file = output_dir / f"multi_group_comparison_{strategy}.html"
    fig.write_html(str(output_file))
    logger.info(f"Saved multi-group comparison: {output_file}")
    
    return fig


# ============================================================================
# BATCH EFFECT IMPACT VISUALIZATIONS
# ============================================================================

def create_batch_effect_impact_plot(
    catboost_dir: Path,
    target_variable: str,
    output_dir: Path
) -> Optional[go.Figure]:
    """
    Visualize how batch effects are handled differently across strategies.
    
    Shows:
    1. Batch variable importance in baseline strategy (batch included)
    2. How removing batch affects other feature importance
    3. Group validation impact on model performance
    
    Parameters
    ----------
    catboost_dir : Path
        Base directory containing strategy results
    target_variable : str
        Target variable name
    output_dir : Path
        Directory to save output figure
        
    Returns
    -------
    go.Figure or None
        Plotly figure showing batch effect impact
    """
    logger.info(f"Creating batch effect impact visualization for {target_variable}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results for all three strategies
    strategies = ['baseline', 'agnostic', 'group_validated']
    strategy_data = {}
    
    for strategy in strategies:
        strategy_dir = catboost_dir / strategy / f"Genus_{target_variable}" / "shap"
        feature_file = strategy_dir / "top_features.csv"
        
        if feature_file.exists():
            df = pd.read_csv(feature_file)
            strategy_data[strategy] = df
        else:
            logger.warning(f"No feature file found for {strategy} strategy")
    
    if len(strategy_data) < 2:
        logger.error("Need at least 2 strategies to compare batch effects")
        return None
    
    # Create comparison figure
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[s.replace('_', ' ').title() for s in strategies],
        specs=[[{'type': 'bar'}] * 3]
    )
    
    batch_keywords = ['batch', 'dataset', 'study', 'sequencing', 'instrument', 'library', 'pcr']
    
    for col_idx, strategy in enumerate(strategies, start=1):
        if strategy not in strategy_data:
            continue
        
        df = strategy_data[strategy]
        
        # Identify batch-related features
        df['is_batch'] = df['feature'].str.lower().str.contains('|'.join(batch_keywords), regex=True)
        
        # Take top 15 features
        df_top = df.head(15)
        
        colors = ['red' if is_batch else 'blue' for is_batch in df_top['is_batch']]
        
        fig.add_trace(
            go.Bar(
                y=df_top['feature'],
                x=df_top['importance'],
                orientation='h',
                marker=dict(color=colors),
                text=df_top['importance'].round(3),
                textposition='auto',
                showlegend=False
            ),
            row=1, col=col_idx
        )
        
        fig.update_yaxes(autorange="reversed", row=1, col=col_idx)
        fig.update_xaxes(title_text="SHAP Importance", row=1, col=col_idx)
    
    fig.update_layout(
        height=700,
        width=1800,
        title_text=f"Batch Effect Impact Across Strategies: {target_variable.replace('_', ' ').title()}<br><sub>Red = Batch-related features, Blue = Biological features</sub>",
        title_font_size=18
    )
    
    # Save figure
    output_file = output_dir / f"batch_effect_impact_{target_variable}.html"
    fig.write_html(str(output_file))
    logger.info(f"Saved batch effect impact plot: {output_file}")
    
    return fig


# ============================================================================
# MAIN ORCHESTRATION FUNCTION
# ============================================================================

def generate_comprehensive_ml_report(
    catboost_dir: Path,
    output_dir: Path,
    ml_targets: List[str] = ['facility_match', 'facility_distance_km'],
    grouping_variables: Optional[List[str]] = None,
    strategies: List[str] = ['baseline', 'agnostic', 'group_validated']
) -> Dict[str, Any]:
    """
    Generate comprehensive ML visualization report.
    
    Creates all visualizations for interpretable ML results across strategies,
    groups, and batch effect approaches.
    
    Parameters
    ----------
    catboost_dir : Path
        Base directory containing CatBoost results
    output_dir : Path
        Directory to save all output figures
    ml_targets : List[str]
        List of ML target variables
    grouping_variables : Optional[List[str]]
        Additional grouping variables beyond ML targets
    strategies : List[str]
        List of strategy names to process
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing all generated figures and metadata
    """
    logger.info("=" * 80)
    logger.info("Generating Comprehensive ML Visualization Report")
    logger.info("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all grouping variables
    if grouping_variables is None:
        grouping_variables = []
    all_groupings = list(set(ml_targets + grouping_variables))
    
    report = {
        'strategy_comparisons': {},
        'group_fingerprints': {},
        'multi_group_comparisons': {},
        'batch_effect_impacts': {},
        'metadata': {
            'strategies': strategies,
            'targets': ml_targets,
            'groupings': all_groupings
        }
    }
    
    # 1. Create strategy comparison dashboards for each target
    logger.info("\n1. Creating strategy comparison dashboards...")
    for target in ml_targets:
        fig = create_strategy_comparison_dashboard(
            catboost_dir, target, output_dir / 'strategy_comparisons', strategies
        )
        if fig:
            report['strategy_comparisons'][target] = fig
    
    # 2. Create group fingerprint plots
    logger.info("\n2. Creating group fingerprint plots...")
    for strategy in ['agnostic', 'group_validated']:  # Skip baseline for clarity
        figs = create_group_fingerprint_comparison(
            catboost_dir, all_groupings, output_dir / 'group_fingerprints', strategy
        )
        report['group_fingerprints'][strategy] = figs
    
    # 3. Create multi-group comparison heatmaps
    logger.info("\n3. Creating multi-group comparison heatmaps...")
    for strategy in ['agnostic', 'group_validated']:
        fig = create_multi_group_comparison_heatmap(
            catboost_dir, all_groupings, output_dir / 'multi_group_comparisons',
            strategy, top_n=15
        )
        if fig:
            report['multi_group_comparisons'][strategy] = fig
    
    # 4. Create batch effect impact visualizations
    logger.info("\n4. Creating batch effect impact visualizations...")
    for target in ml_targets:
        fig = create_batch_effect_impact_plot(
            catboost_dir, target, output_dir / 'batch_effects'
        )
        if fig:
            report['batch_effect_impacts'][target] = fig
    
    logger.info("\n" + "=" * 80)
    logger.info("ML Visualization Report Complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"- {len(report['strategy_comparisons'])} strategy comparison dashboards")
    logger.info(f"- {sum(len(v) for v in report['group_fingerprints'].values())} group fingerprint plots")
    logger.info(f"- {len(report['multi_group_comparisons'])} multi-group comparison heatmaps")
    logger.info(f"- {len(report['batch_effect_impacts'])} batch effect impact plots")
    logger.info("=" * 80 + "\n")
    
    return report
