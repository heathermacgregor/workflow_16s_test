# src/workflow_16s/downstream/machine_learning/visualization/validation_plots.py

import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import numpy as np
import re
from pathlib import Path
from typing import Optional

from .base_style import update_font_sizes, DEFAULT_HEIGHT, DEFAULT_WIDTH_RECTANGLE
from workflow_16s.visualization import apply_common_layout, save_fig

from workflow_16s.utils.logger import get_logger
from workflow_16s.visualization.utils import PlottingUtils


def plot_shuffle_test(json_path: Path, output_dir: Path) -> Optional[Path]:
    """
    Creates a histogram comparing the Null Distribution (shuffled) vs Real Model Performance.
    
    This is the definitive proof that the model is learning real biology rather than
    finding patterns in random noise.
    
    Parameters
    ----------
    json_path : Path
        Path to the JSON file containing shuffle test results.
    output_dir : Path
        Directory to save the output HTML plot.
        
    Returns
    -------
    Optional[Path]
        Path to the saved HTML plot, or None if generation failed.
    """
    logger = get_logger("workflow_16s")
    plot_utils = PlottingUtils(logger)
    if not json_path.exists():
        logger.warning(f"Shuffle test data not found at {json_path}")
        return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        shuffled_scores = np.array(data['shuffled_scores'])
        real_score = data['real_mcc']
        p_val = data['p_value']
        target = data['target']
        
        # Create distribution plot for the null hypothesis
        fig = ff.create_distplot(
            [shuffled_scores], ['Random Chance (Null Distribution)'], 
            bin_size=0.05, show_hist=True, show_rug=True, colors=['#95a5a6']
        )
        
        # Add the 'Real Model' line
        fig.add_shape(
            type="line", x0=real_score, y0=0, x1=real_score, y1=1, yref="paper",
            line=dict(color="#e74c3c", width=3, dash="solid")
        )
        
        fig.add_annotation(
            x=real_score, y=0.95, yref="paper",
            text=f"Real Model: {real_score:.3f}<br>(p={p_val:.4f})",
            showarrow=True, arrowhead=2, ax=0, ay=-40,
            font=dict(color="#c0392b", size=12)
        )
        
        fig.update_layout(
            title=f"Statistical Significance Test: {target.replace('_', ' ').title()}",
            xaxis_title="Model Performance (MCC)",
            yaxis_title="Density",
            template="plotly_white", showlegend=False, height=450
        )
        
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"shuffle_plot_{target}.html"
        fig.write_html(out_file, include_plotlyjs='cdn')
        logger.info(f"✅ Significance plot generated: {out_file.name}")
        return out_file
        
    except Exception as e:
        logger.error(f"Failed to generate shuffle plot: {e}")
        return None

def generate_stability_comparison(catboost_dir: Path, output_dir: Path) -> Optional[Path]:
    """
    Generates a heatmap comparing feature stability across all automated sub-targets.
    
    Identifies 'Universal' biomarkers (predictive across all facilities) 
    vs 'Facility-Specific' indicators.
    
    Parameters
    ----------
    catboost_dir : Path
        Directory containing CatBoost ML results.
    output_dir : Path
        Directory to save the output HTML plot.
        
    Returns
    -------
    Optional[Path]
        Path to the saved HTML plot, or None if generation failed.
    """
    logger = get_logger("workflow_16s")
    plot_utils = PlottingUtils(logger)
    importance_files = list(catboost_dir.glob("**/top_features.csv"))
    if not importance_files:
        logger.warning("No feature importance files found for stability heatmap.")
        return None

    data_frames = []
    for f in importance_files:
        parts = f.parts
        try:
            # Dynamic path extraction: .../strategy/level_target/shap/top_features.csv
            strategy = parts[-4]
            target = parts[-3].split('_', 1)[1] # remove 'Genus_' or 'Family_'
            
            df = pd.read_csv(f)
            df['Strategy'] = strategy
            df['Target'] = target
            data_frames.append(df)
        except Exception:
            continue
    
    if not data_frames:
        return None
    
    master_df = pd.concat(data_frames)
    
    # Prioritize Agnostic or LOPOCV strategies as they reflect the most robust signal
    robust_df = master_df[master_df['Strategy'].isin(['agnostic', 'lopocv'])]
    if robust_df.empty:
        robust_df = master_df

    # Pivot: Features on Y-axis, Targets on X-axis
    pivot_df = robust_df.pivot_table(
        index='feature', columns='Target', values='importance', fill_value=0
    )
    
    # Sort by total impact to bring the most "Universal" features to the top
    pivot_df['Total_Impact'] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values('Total_Impact', ascending=False).head(40)
    pivot_df = pivot_df.drop(columns=['Total_Impact'])
    
    # Clean indices for plot readability
    pivot_df.index = [i.split('__')[-1].replace('_', ' ') for i in pivot_df.index]

    fig = px.imshow(
        pivot_df, 
        aspect='auto', 
        color_continuous_scale='Viridis',
        labels=dict(x="Forensic Target", y="Taxonomic Feature", color="SHAP Importance"),
        title="Biomarker Stability Across Facility Targets (Robust Strategies Only)"
    )
    
    fig.update_layout(height=max(600, len(pivot_df) * 20), template='plotly_white')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "biomarker_stability_heatmap.html"
    fig.write_html(out_file, include_plotlyjs='cdn')
    logger.info(f"✅ Stability heatmap generated: {out_file.name}")
    return out_file

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
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with feature importances. 
    plot_dir_ml : Path
        Directory to save the plot.
    target_name : str
        Name of the target variable.
    level : str
        Taxonomic level (e.g., 'Genus', 'Family').
    model_score : float
        Overall model performance score (e.g., MCC, R2).
    score_name : str
        Name of the performance metric.
    """
    logger = get_logger("workflow_16s")
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

def plot_strategy_resilience(results_df: pd.DataFrame, output_path: Path, target_name: str) -> go.Figure:
    """
    Plots the performance delta between ML strategies.
    A high 'Baseline' vs low 'Agnostic' indicates batch effect contamination.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with strategy performance results.
    output_path : Path
        Path to save the output HTML plot.
    target_name : str
        Name of the target variable.
    
    Returns
    -------
    go.Figure
        The generated Plotly figure.
    """
    logger = get_logger("workflow_16s")
    plot_utils = PlottingUtils(logger)
    # results_df expected columns: ['strategy', 'metric_score', 'std_dev']
    fig = go.Figure()

    colors = {
        'baseline': '#95a5a6',    # Grey: The "Naive" approach
        'agnostic': '#3498db',    # Blue: The "Forensic" approach
        'lopocv': '#2ecc71',      # Green: The "Universal" approach
        'shuffle': '#e74c3c'      # Red: The "Chance" level
    }

    for strategy in results_df['strategy'].unique():
        df_sub = results_df[results_df['strategy'] == strategy]
        fig.add_trace(go.Bar(
            name=strategy.capitalize(),
            x=[strategy],
            y=df_sub['metric_score'],
            error_y=dict(type='data', array=df_sub['std_dev'], visible=True),
            marker_color=colors.get(strategy, '#bdc3c7')
        ))

    fig = apply_common_layout(
        fig, "Discovery Strategy", "Performance Metric (MCC/R2)", 
        f"Strategy Resilience Audit: {target_name}"
    )
    
    # Add a 'DANGER ZONE' line based on shuffle baseline
    if 'shuffle' in results_df['strategy'].values:
        chance_val = results_df[results_df['strategy'] == 'shuffle']['metric_score'].values[0]
        fig.add_shape(
            type="line", line=dict(color="Red", width=2, dash="dash"),
            x0=-0.5, x1=len(results_df['strategy'].unique())-0.5, y0=chance_val, y1=chance_val
        )
        fig.add_annotation(x=0, y=chance_val, text="Random Chance Baseline", showarrow=False, yshift=10)

    fig = update_font_sizes(fig)
    plot_utils.save_plotly_fig(fig, output_path)
    return fig