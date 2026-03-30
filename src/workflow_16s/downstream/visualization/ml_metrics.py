"""ML-specific visualizations for optimization, metrics, and feature relationships.

Generates:
- Optuna trial progress and parameter importance
- Model metrics (accuracy, F1, AUC, MSE, etc.)
- Confusion matrices
- Feature vs target plots (violin, scatter, box)
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger('workflow_16s')


def plot_optuna_trial_progress(
    trials_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    height: int = 600,
    width: int = 1000,
) -> go.Figure:
    """
    Plot Optuna trial progress and best value over iterations.
    
    Parameters
    ----------
    trials_df : pd.DataFrame
        DataFrame with columns: (trial_id/number, value, state, datetime)
        Typically from study.trials_dataframe().
    output_path : Path, optional
        Save HTML plot.
    height : int
        Figure height.
    width : int
        Figure width.
    
    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if trials_df.empty:
        logger.warning("No trials to plot")
        return None
    
    # Normalize column names
    if 'number' not in trials_df.columns and 'trial_id' in trials_df.columns:
        trials_df = trials_df.rename(columns={'trial_id': 'number'})
    
    if 'value' not in trials_df.columns:
        logger.error("'value' column not found in trials_df")
        return None
    
    # Filter completed trials
    completed = trials_df[trials_df['state'] == 'COMPLETE'].copy()
    completed = completed.sort_values('number')
    
    if len(completed) == 0:
        logger.warning("No completed trials found")
        return None
    
    # Calculate best value so far
    completed['best_value'] = completed['value'].cummin()
    
    fig = go.Figure()
    
    # Trial values
    fig.add_trace(go.Scatter(
        x=completed['number'],
        y=completed['value'],
        mode='markers+lines',
        name='Trial Value',
        marker=dict(size=8, color='lightblue', opacity=0.6),
        line=dict(color='lightblue', width=1),
    ))
    
    # Best value so far
    fig.add_trace(go.Scatter(
        x=completed['number'],
        y=completed['best_value'],
        mode='lines',
        name='Best Value (cumulative)',
        line=dict(color='darkblue', width=3, dash='solid'),
    ))
    
    fig.update_layout(
        title="Optuna Trial Progress",
        xaxis_title="Trial Number",
        yaxis_title="Objective Value",
        height=height,
        width=width,
        hovermode='closest',
        template='plotly_white',
    )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Optuna trial progress saved: {output_path}")
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    height: int = 600,
    width: int = 700,
) -> go.Figure:
    """
    Plot confusion matrix as heatmap.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    class_names : List[str], optional
        Class label names.
    output_path : Path, optional
        Save HTML.
    height : int
        Figure height.
    width : int
        Figure width.
    
    Returns
    -------
    go.Figure
        Plotly heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]
    
    # Normalize for visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=class_names,
        y=class_names,
        text=cm,  # Show actual counts on hover
        texttemplate='%{text}',
        textposition='inside',
        colorscale='Blues',
        hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{text}<br>Normalized: %{z:.2%}<extra></extra>',
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="True",
        height=height,
        width=width,
        template='plotly_white',
    )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Confusion matrix saved: {output_path}")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Optional[Path] = None,
    height: int = 600,
    width: int = 700,
) -> go.Figure:
    """
    Plot ROC curve for binary classification.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_score : np.ndarray
        Predicted scores/probabilities.
    output_path : Path, optional
        Save HTML.
    height : int
        Figure height.
    width : int
        Figure width.
    
    Returns
    -------
    go.Figure
        Plotly ROC curve.
    """
    try:
        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
    except Exception as e:
        logger.error(f"Could not compute ROC curve: {e}")
        return None
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC={auc:.3f})',
        line=dict(color='#1f77b4', width=3),
        fill='tonexty',
    ))
    
    # Diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random (AUC=0.5)',
        line=dict(color='gray', width=2, dash='dash'),
    ))
    
    fig.update_layout(
        title=f"ROC Curve (AUC = {auc:.3f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=height,
        width=width,
        template='plotly_white',
        hovermode='closest',
    )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ ROC curve saved: {output_path}")
    
    return fig


def plot_feature_vs_target_categorical(
    X_features: pd.DataFrame,
    y_target: pd.Series,
    feature_list: Optional[List[str]] = None,
    max_features: int = 6,
    output_path: Optional[Path] = None,
    height: int = 1000,
    width: int = 1200,
) -> go.Figure:
    """
    Plot top features vs categorical target using violin plots.
    
    Parameters
    ----------
    X_features : pd.DataFrame
        Feature matrix.
    y_target : pd.Series
        Categorical target variable.
    feature_list : List[str], optional
        Features to plot. If None, uses all columns.
    max_features : int
        Maximum features to display.
    output_path : Path, optional
        Save HTML.
    height : int
        Figure height.
    width : int
        Figure width.
    
    Returns
    -------
    go.Figure
        Plotly subplots with violin plots.
    """
    if feature_list is None:
        feature_list = X_features.columns.tolist()[:max_features]
    else:
        feature_list = feature_list[:max_features]
    
    n_features = len(feature_list)
    n_rows = (n_features + 2) // 3
    
    fig = make_subplots(
        rows=n_rows,
        cols=3,
        subplot_titles=feature_list,
        specs=[[{'type': 'box'} for _ in range(3)] for _ in range(n_rows)],
    )
    
    for idx, feature in enumerate(feature_list):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        try:
            feature_data = pd.to_numeric(X_features[feature], errors='coerce')
            
            for target_val in y_target.unique():
                mask = y_target == target_val
                
                fig.add_trace(
                    go.Violin(
                        x=[str(target_val)] * mask.sum(),
                        y=feature_data[mask],
                        name=str(target_val),
                        box_visible=True,
                        meanline_visible=True,
                    ),
                    row=row,
                    col=col,
                )
        
        except Exception as e:
            logger.warning(f"Could not plot '{feature}': {e}")
    
    fig.update_yaxes(title_text="Value")
    fig.update_xaxes(title_text="Target Class")
    
    fig.update_layout(
        title_text="Top Features vs Target (Categorical)",
        height=height,
        width=width,
        showlegend=True,
        template='plotly_white',
    )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Feature vs target (violin) saved: {output_path}")
    
    return fig


def plot_feature_vs_target_continuous(
    X_features: pd.DataFrame,
    y_target: pd.Series,
    feature_list: Optional[List[str]] = None,
    max_features: int = 6,
    output_path: Optional[Path] = None,
    height: int = 1000,
    width: int = 1200,
) -> go.Figure:
    """
    Plot top features vs continuous target using scatter plots.
    
    Parameters
    ----------
    X_features : pd.DataFrame
        Feature matrix.
    y_target : pd.Series
        Continuous target variable.
    feature_list : List[str], optional
        Features to plot.
    max_features : int
        Maximum features to display.
    output_path : Path, optional
        Save HTML.
    height : int
        Figure height.
    width : int
        Figure width.
    
    Returns
    -------
    go.Figure
        Plotly subplots with scatter plots.
    """
    if feature_list is None:
        feature_list = X_features.columns.tolist()[:max_features]
    else:
        feature_list = feature_list[:max_features]
    
    n_features = len(feature_list)
    n_rows = (n_features + 2) // 3
    
    fig = make_subplots(
        rows=n_rows,
        cols=3,
        subplot_titles=feature_list,
    )
    
    for idx, feature in enumerate(feature_list):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        try:
            feature_data = pd.to_numeric(X_features[feature], errors='coerce')
            target_data = pd.to_numeric(y_target, errors='coerce')
            
            # Calculate correlation
            valid_mask = feature_data.notna() & target_data.notna()
            if valid_mask.sum() > 1:
                corr = np.corrcoef(feature_data[valid_mask], target_data[valid_mask])[0, 1]
            else:
                corr = np.nan
            
            fig.add_trace(
                go.Scatter(
                    x=feature_data,
                    y=target_data,
                    mode='markers',
                    name=feature,
                    marker=dict(size=6, opacity=0.6),
                    text=[f"Feature: {f:.3f}<br>Target: {t:.3f}" 
                          for f, t in zip(feature_data, target_data)],
                    hovertemplate='%{text}<extra></extra>',
                ),
                row=row,
                col=col,
            )
            
            fig.update_xaxes(title_text=f"{feature}<br>(r={corr:.2f})", row=row, col=col)
            fig.update_yaxes(title_text="Target", row=row, col=col)
        
        except Exception as e:
            logger.warning(f"Could not plot '{feature}': {e}")
    
    fig.update_layout(
        title_text="Top Features vs Target (Continuous - Scatter)",
        height=height,
        width=width,
        showlegend=False,
        template='plotly_white',
    )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Feature vs target (scatter) saved: {output_path}")
    
    return fig


def plot_feature_importance_bars(
    feature_importance: pd.Series,
    top_n: int = 20,
    output_path: Optional[Path] = None,
    height: int = 600,
    width: int = 1000,
) -> go.Figure:
    """
    Plot top features as horizontal bar chart.
    
    Parameters
    ----------
    feature_importance : pd.Series
        Feature importance scores (index=feature names, values=importance).
    top_n : int
        Number of top features to show.
    output_path : Path, optional
        Save HTML.
    height : int
        Figure height.
    width : int
        Figure width.
    
    Returns
    -------
    go.Figure
        Plotly bar chart.
    """
    top_features = feature_importance.nlargest(top_n)
    
    fig = go.Figure(data=go.Bar(
        x=top_features.values,
        y=top_features.index,
        orientation='h',
        marker=dict(color=top_features.values, colorscale='Viridis'),
        text=top_features.values,
        textposition='outside',
        hovertemplate='%{y}<br>Importance: %{x:.4f}<extra></extra>',
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=height,
        width=width,
        template='plotly_white',
    )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Feature importance saved: {output_path}")
    
    return fig


def plot_model_metrics_summary(
    metrics: Dict[str, float],
    output_path: Optional[Path] = None,
    height: int = 500,
    width: int = 900,
) -> go.Figure:
    """
    Plot model metrics as horizontal bar chart.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of metric names and values (should be 0-1 range).
    output_path : Path, optional
        Save HTML.
    height : int
        Figure height.
    width : int
        Figure width.
    
    Returns
    -------
    go.Figure
        Plotly bar chart.
    """
    metrics_series = pd.Series(metrics).sort_values()
    
    fig = go.Figure(data=go.Bar(
        x=metrics_series.values,
        y=metrics_series.index,
        orientation='h',
        marker=dict(
            color=metrics_series.values,
            colorscale='RdYlGn',
            cmin=0,
            cmax=1,
        ),
        text=[f"{v:.3f}" for v in metrics_series.values],
        textposition='outside',
        hovertemplate='%{y}<br>%{x:.4f}<extra></extra>',
    ))
    
    fig.update_layout(
        title="Model Metrics Summary",
        xaxis_title="Score",
        yaxis_title="Metric",
        xaxis=dict(range=[0, 1]),
        height=height,
        width=width,
        template='plotly_white',
    )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Model metrics summary saved: {output_path}")
    
    return fig
