# workflow_16s/visualization/machine_learning/evaluation.py
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import List, Union, Any
from workflow_16s.visualization import apply_common_layout, save_fig
from .style import update_font_sizes, DEFAULT_HEIGHT, DEFAULT_WIDTH_SQUARE

def plot_confusion_matrix(cm: np.ndarray, output_path: Union[str, Path], class_names: List[str], height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH_SQUARE) -> Any:
    """Create an interactive confusion matrix plot using Plotly."""
    annotations_text = []
    total = cm.sum()
    for i in range(cm.shape[0]):
        row_text = []
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = f"{count/total:.1%}" if total > 0 else "0%"
            row_text.append(f"<b>{count}</b><br>({percentage})")
        annotations_text.append(row_text)

    x_labels = [f'Predicted {name}' for name in class_names]
    y_labels = [f'Actual {name}' for name in class_names]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=x_labels,
        y=y_labels,
        colorscale='Blues',
        hoverinfo='z',
        text=annotations_text,
        texttemplate="%{text}",
        textfont={"size": 14},
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
    ))
    fig.update_layout(title_x=0.5, font=dict(size=12))
    fig.update_yaxes(autorange="reversed")
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1, line=dict(color="black", width=2)) # type: ignore
    fig = apply_common_layout(fig, 'Predicted Label', 'Actual Label', 'Confusion Matrix') 
    fig = update_font_sizes(fig)
    fig.update_layout(autosize=True, height=height, width=width, xaxis=dict(side='bottom'))    
    save_fig(fig, Path(output_path), formats=['png', 'html', 'json'], verbose=False)
    return fig


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, output_path: Union[str, Path], height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH_SQUARE) -> go.Figure:
    """Plot Predicted vs Actual values with stylised reference line and boundary."""
    fig = go.Figure()

    # Scatter points - matching your color palette
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode='markers',
        marker=dict(color='#1f77b4', opacity=0.6, size=10, line=dict(width=1, color='white')),
        name='Samples',
        hovertemplate="Actual: %{x}<br>Predicted: %{y}<extra></extra>"
    ))

    # Perfect prediction reference line
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode='lines',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        name='Perfect Fit'
    ))

    # Stylistic additions to match Confusion Matrix
    fig.update_layout(title_x=0.5, font=dict(size=12))
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1, line=dict(color="black", width=2))
    
    # Standard Workflow Wrappers
    fig = apply_common_layout(fig, 'Actual Values', 'Predicted Values', 'Regression: Predicted vs Actual')
    fig = update_font_sizes(fig)
    fig.update_layout(autosize=True, height=height, width=width)
    
    save_fig(fig, Path(output_path), formats=['png', 'html', 'json'], verbose=False)
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, output_path: Union[str, Path], height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH_SQUARE) -> go.Figure:
    """Plot residuals with stylized horizontal zero-line and boundary."""
    residuals = y_true - y_pred
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode='markers',
        marker=dict(color='#d62728', opacity=0.6, size=10, line=dict(width=1, color='white')),
        name='Residuals',
        hovertemplate="Predicted: %{x}<br>Error: %{y}<extra></extra>"
    ))

    # Zero error line
    fig.add_trace(go.Scatter(
        x=[y_pred.min(), y_pred.max()], y=[0, 0],
        mode='lines',
        line=dict(color='black', width=2, dash='solid'),
        showlegend=False
    ))

    # Stylistic additions to match Confusion Matrix
    fig.update_layout(title_x=0.5, font=dict(size=12))
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1, line=dict(color="black", width=2))

    # Standard Workflow Wrappers
    fig = apply_common_layout(fig, 'Predicted Values', 'Residual (Actual - Predicted)', 'Residual Analysis')
    fig = update_font_sizes(fig)
    fig.update_layout(autosize=True, height=height, width=width)

    save_fig(fig, Path(output_path), formats=['png', 'html', 'json'], verbose=False)
    return fig

        
def plot_roc_curve(fpr, tpr, roc_auc, output_path: Union[str, Path], height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH_SQUARE) -> go.Figure:
    """Plot ROC curve using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})', line=dict(width=3, color='#1f77b4')))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random (AUC = 0.50)', line=dict(dash='dash', color='#444')))
    fig = update_font_sizes(fig)
    fig.update_layout(autosize=True, height=height, width=width, xaxis=dict(range=[-0.05, 1.05]), yaxis=dict(range=[-0.05, 1.05]), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),)
    fig = apply_common_layout(fig, 'False Positive Rate', 'True Positive Rate', 'Receiver Operating Characteristic') 
    save_fig(fig, Path(output_path), formats=['png', 'html', 'json'], verbose=False)
    return fig
    

def plot_precision_recall_curve(precision, recall, average_precision, output_path: Union[str, Path], show: bool = False, verbose: bool = False, height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH_SQUARE) -> go.Figure:
    """Plot Precision-Recall curve using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR curve (AP = {average_precision:.2f})', line=dict(width=3, color='#ff7f0e'), fill='tozeroy'))
    fig = update_font_sizes(fig)
    fig.update_layout(autosize=True, height=height, width=width, xaxis=dict(range=[-0.05, 1.05]), yaxis=dict(range=[-0.05, 1.05]), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    fig = apply_common_layout(fig, 'Recall', 'Precision', 'Precision-Recall Curve')
    save_fig(fig, Path(output_path), formats=['png', 'html', 'json'], verbose=False)
    return fig