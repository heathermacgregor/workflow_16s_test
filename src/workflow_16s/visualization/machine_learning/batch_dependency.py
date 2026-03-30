# workflow_16s/visualization/machine_learning/batch_dependency.py

import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from workflow_16s.constants import BATCH_KEYWORDS
from workflow_16s.utils.logger import get_logger
from workflow_16s.visualization.utils import PlottingUtils

def _calculate_eta_squared(feature_values: pd.Series, batch_labels: pd.Series) -> float:
    """Calculates Eta-squared (proportion of variance explained by batch)."""
    df = pd.DataFrame({'val': feature_values, 'batch': batch_labels}).dropna()
    if df.empty or df['batch'].nunique() < 2:
        return 0.0
        
    grand_mean = df['val'].mean()
    ss_total = ((df['val'] - grand_mean) ** 2).sum()
    
    if ss_total == 0:
        return 0.0
        
    ss_between = 0
    for _, group in df.groupby('batch'):
        group_mean = group['val'].mean()
        ss_between += len(group) * ((group_mean - grand_mean) ** 2)
        
    return float(ss_between / ss_total)

def plot_batch_dependency(
    importance_df: pd.DataFrame, 
    X_df: pd.DataFrame,
    batch_series: pd.Series,
    output_path: Path, 
    target_name: str,
    top_n: int = 20
):
    """
    Visualizes the feature importance of top taxa alongside their Batch Confounding Score.
    
    A successful forensic model relies on features that have high ML importance 
    but low variance explained by technical batches (low Eta-squared).
    """
    logger = get_logger("workflow_16s")
    
    # 1. Get Top N Features
    top_features = importance_df.nlargest(top_n, 'importance').copy()
    
    # 2. Calculate Batch Confounding (Eta-squared) for each top feature
    confounding_scores = []
    for feature in top_features['feature']:
        if feature in X_df.columns:
            score = _calculate_eta_squared(X_df[feature], batch_series)
        else:
            score = 0.0
        confounding_scores.append(score)
        
    top_features['batch_confounding'] = confounding_scores
    
    # 3. Create Dual-Axis Plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar chart for ML Importance
    fig.add_trace(
        go.Bar(
            x=top_features['feature'],
            y=top_features['importance'],
            name="Feature Importance",
            marker_color='#3498db',
            opacity=0.8
        ),
        secondary_y=False,
    )

    # Line/Scatter for Batch Confounding
    fig.add_trace(
        go.Scatter(
            x=top_features['feature'],
            y=top_features['batch_confounding'],
            name="Batch Confounding (Eta²)",
            mode='lines+markers',
            marker=dict(color='#e74c3c', size=8),
            line=dict(color='#e74c3c', width=2)
        ),
        secondary_y=True,
    )

    # 4. Formatting
    fig.update_layout(
        title=f"Feature Importance vs. Batch Confounding: {target_name}<br><sub>Are the top predictors actually just technical artifacts?</sub>",
        template='plotly_white',
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Tree-based Importance", secondary_y=False)
    fig.update_yaxes(title_text="Variance Explained by Batch (0-1)", range=[0, 1.05], secondary_y=True)
    fig.update_xaxes(tickangle=45)

    # 5. Save and Audit
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    
    # Audit Warning if the top 5 features are highly confounded (>30% variance from batch)
    top_5_confounding = top_features.head(5)['batch_confounding'].mean()
    logger.info(f"🛡️ Integrity Audit for {target_name}: Top 5 features have {top_5_confounding:.1%} mean batch confounding.")
    
    if top_5_confounding > 0.3:
        logger.warning(f"⚠️ HIGH CONFOUNDING: The primary drivers of the {target_name} model are heavily influenced by batch effects!")
        
def plot_confounding_heatmap(results: Dict[str, Any], plot_dir: Path, target_col: str, threshold: float):
    """Create bar-style heatmap of batch-target associations.
    
    METHODOLOGY:
    - Bars colored by risk level (High, Moderate, Low)
    - Annotations indicate association values and types
    
    Parameters
    ----------
    results : Dict[str, Any]
        Confounding statistics for batch variables.
    plot_dir : Path
        Directory to save the plot.
    target_col : str
        Target variable name.
    threshold : float
        Threshold for high confounding.
    """
    logger = get_logger("workflow_16s")
    plot_utils = PlottingUtils(logger)
    stats = results['statistics']
    batch_cols = list(stats.keys())
    values = [stats[col]['value'] for col in batch_cols]
    types = [stats[col]['type'] for col in batch_cols]
    
    # Color mapping
    colors = ['#ef553b' if v >= threshold else '#fec031' if v >= threshold*0.7 else '#636efa' for v in values]
    
    fig = go.Figure(data=go.Bar(
        x=batch_cols, y=values,
        marker_color=colors,
        text=[f"{v:.2f} ({t})" for v, t in zip(values, types)],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Confounding Diagnostic: {target_col}",
        yaxis_title="Association Strength (0-1)",
        yaxis_range=[0, 1.1],
        template="plotly_white",
        height=450
    )
    
    # Risk Lines
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="High Risk")
    fig.add_hline(y=threshold*0.7, line_dash="dash", line_color="orange", annotation_text="Moderate Risk")
    
    safe_target = re.sub(r'\W+', '', target_col)
    plot_path = plot_dir / f"confounding_diagnostic_{safe_target}.html"
    plot_utils.save_plotly_fig(fig, plot_path)

def create_confounding_heatmap(
    X_taxa, 
    batch_covariates, 
    top_taxa, 
    plot_dir, 
    target_name, 
    level
):
    """Significance-masked heatmap (p < 0.05)."""
    logger = get_logger("workflow_16s")
    taxa_subset = X_taxa[top_taxa]
    corr_matrix = pd.DataFrame(index=top_taxa, columns=batch_covariates.columns)
    p_matrix = pd.DataFrame(index=top_taxa, columns=batch_covariates.columns)

    for taxon in top_taxa:
        for b_var in batch_covariates.columns:
            b_vals = batch_covariates[b_var].astype('category').cat.codes if batch_covariates[b_var].dtype == 'object' else batch_covariates[b_var]
            rho, p = spearmanr(taxa_subset[taxon], b_vals, nan_policy='omit')
            corr_matrix.loc[taxon, b_var] = rho # type: ignore
            p_matrix.loc[taxon, b_var] = p # type: ignore

    masked_corr = corr_matrix.astype(float).where(p_matrix.astype(float) < 0.05, np.nan)
    fig = px.imshow(
        masked_corr,
        labels=dict(x="Batch Variable", y="Microbial Taxon", color="Significant ρ"),
        x=batch_covariates.columns, y=[t.split('__')[-1] for t in top_taxa],
        color_continuous_scale='RdBu_r', range_color=[-1, 1],
        title=f"Confounding Diagnostic (p < 0.05): {target_name}"
    )
    safe_target = re.sub(r'\W+', '', target_name)
    fig.write_html(str(plot_dir / f"confounding_heatmap_{safe_target}.html"))
    logger.info(f"[✔] Confounding heatmap saved for target '{target_name}' at {plot_dir / f'confounding_heatmap_{safe_target}.html'}")
    
def create_comparison_plots(
    all_results: Dict[str, Dict], 
    plot_dir: Path, 
    level: str
):
    logger = get_logger("workflow_16s")
    targets = list(all_results.keys())
    baseline_s = [all_results[t]['models']['baseline']['test_score'] for t in targets]
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Model Accuracy", "Batch Contribution"))
    fig.add_trace(go.Bar(name='Baseline', x=targets, y=baseline_s), row=1, col=1)
    fig.write_html(str(plot_dir / f"batch_control_comparison_{level}.html"))
    logger.info(f"[✔] Comparison plots saved to {plot_dir / f'batch_control_comparison_{level}.html'}")