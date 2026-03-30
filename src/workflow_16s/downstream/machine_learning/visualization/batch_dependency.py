# src/workflow_16s/downstream/machine_learning/visualization/batch_dependency.py

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from ..constants import BATCH_KEYWORDS

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from workflow_16s.utils.logger import get_logger

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