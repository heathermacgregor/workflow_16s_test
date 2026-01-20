# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import colorcet as cc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import shap
from matplotlib import (
    colors as mcolors,
    pyplot as plt
)
from matplotlib.colors import LogNorm
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, leaves_list

# Local Imports
from workflow_16s.figures.figures import (
    attach_legend_to_figure, largecolorset, plot_legend, plotly_show_and_save, save_plotly_html
)
from workflow_16s.figures.merged import _apply_common_layout

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')
sns.set_style('whitegrid')  # Set seaborn style globally
warnings.filterwarnings("ignore") # Suppress warnings

DEFAULT_HEIGHT = 1100
DEFAULT_WIDTH_SQUARE = 1200
DEFAULT_WIDTH_RECTANGLE = 1600
DEFAULT_TITLE_FONT_SIZE = 24
DEFAULT_AXIS_TITLE_FONT_SIZE = 20
DEFAULT_TICKS_LABEL_FONT_SIZE = 16

# ==================================== FUNCTIONS ===================================== #

def update_font_sizes(fig):
    fig.update_layout(
        title=dict(font=dict(size=DEFAULT_TITLE_FONT_SIZE)),
        xaxis=dict(
            title=dict(font=dict(size=DEFAULT_AXIS_TITLE_FONT_SIZE)),
            tickfont=dict(size=DEFAULT_TICKS_LABEL_FONT_SIZE), 
        ),
        yaxis=dict(
            title=dict(font=dict(size=DEFAULT_AXIS_TITLE_FONT_SIZE)),
            tickfont=dict(size=DEFAULT_TICKS_LABEL_FONT_SIZE), 
        )
    )    
    return fig
    
    
def plot_confusion_matrix(
    cm_flipped: np.ndarray,
    output_path: Union[str, Path],
    class_names: List[str] = ['Positive', 'Negative'],
    show: bool = False,
    verbose: bool = True,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH_SQUARE
) -> Any:
    """
    Create an interactive confusion matrix plot using Plotly.
    
    Args:
        cm_flipped:  Confusion matrix (2x2) as [[TN, FP], [FN, TP]].
        output_path: Output path for saving the plot (without extension).
        class_names: Names for the classes [Actual, Predicted].
        show:        Whether to display the plot.
        verbose:     Verbosity flag.
    """
    # Create annotation text with values and percentages
    annotations = []
    total = cm_flipped.sum()
    for i in range(cm_flipped.shape[0]):
        for j in range(cm_flipped.shape[1]):
            count = cm_flipped[i, j]
            percentage = f"{count/total:.1%}" if total > 0 else "0%"
            annotations.append(
                f"<b>{count}</b><br>({percentage})"
            )
    
    # Reshape annotations to match matrix shape
    annotations = np.array(annotations).reshape(cm_flipped.shape).tolist()
    
    # Create heatmap
    fig = ff.create_annotated_heatmap(
        z=cm_flipped,
        annotation_text=annotations,
        colorscale='Blues',
        x=[f'Predicted {name}' for name in class_names],
        y=[f'Actual {name}' for name in class_names],
        hoverinfo='z',
        showscale=True
    )
    
    fig.update_layout(title_x=0.5, font=dict(size=12))
    
    # Customize hover text
    fig.update_traces(
        hovertemplate=(
            "<b>Actual</b>: %{y}<br>"
            "<b>Predicted</b>: %{x}<br>"
            "<b>Count</b>: %{z}<br>"
            "<b>Percentage</b>: %{text}"
        ),
        texttemplate="%{text}",
        textfont_size=14
    )
    
    # Reverse y-axis to match typical confusion matrix orientation
    fig.update_yaxes(autorange="reversed")
    
    # Add border lines
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="black", width=2)
    )
    fig = _apply_common_layout(
        fig, 
        'Predicted Label', 'Actual Label', 
        '<b>Confusion Matrix</b>'
    ) 
    # Update font sizes
    fig = update_font_sizes(fig)
    fig.update_layout(
        autosize=True,
        height=height, width=width,
        xaxis=dict(side='bottom')
    )    
    plotly_show_and_save(
        fig, show, output_path, 
        ['png', 'html'], verbose
    )
    return fig

        
def plot_roc_curve(
    fpr, 
    tpr, 
    roc_auc, 
    output_path: Union[str, Path],
    show: bool = False,
    verbose: bool = False,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH_SQUARE
) -> go.Figure:
    """
    Plot ROC curve using Plotly.
    
    Args:
        fpr:         False Positive Rates.
        tpr:         True Positive Rates.
        roc_auc:     Area Under ROC Curve.
        output_path: Output path for saving the plot.
        show:        Whether to display the plot.
        verbose:     Whether to log output.
    """
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, 
        y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.2f})',
        line=dict(width=3, color='#1f77b4')
    ))
    
    # Random chance line
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='Random (AUC = 0.50)',
        line=dict(dash='dash', color='#444')
    ))
    
    # Update layout
    fig = update_font_sizes(fig)
    fig.update_layout(
        autosize=True,
        height=height,
        width=width,
        xaxis=dict(range=[-0.05, 1.05]),
        yaxis=dict(range=[-0.05, 1.05]),
        legend=dict(
            yanchor="top", y=0.99, 
            xanchor="left", x=0.01
        ),
    )
    fig = _apply_common_layout(
        fig, 
        'False Positive Rate', 'True Positive Rate', 
        'Receiver Operating Characteristic'
    ) 
    plotly_show_and_save(
        fig, show, output_path, 
        ['png', 'html'], verbose
    )
    return fig
    

def plot_precision_recall_curve(
    precision, 
    recall, 
    average_precision, 
    output_path: Union[str, Path],
    show: bool = False,
    verbose: bool = False,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH_SQUARE
) -> go.Figure:
    """
    Plot Precision-Recall curve using Plotly.
    
    Args:
        precision:         Precision values.
        recall:            Recall values.
        average_precision: Average precision score.
        output_path:       Output path for saving the plot.
        show:              Whether to display the plot.
        verbose:           Verbosity flag.
    """
    fig = go.Figure()
    
    # Precision-Recall curve
    fig.add_trace(go.Scatter(
        x=recall, 
        y=precision,
        mode='lines',
        name=f'PR curve (AP = {average_precision:.2f})',
        line=dict(width=3, color='#ff7f0e'),
        fill='tozeroy'
    ))
    
    # Update layout
    fig = update_font_sizes(fig)
    fig.update_layout(
        autosize=True,
        height=height, width=width,
        xaxis=dict(range=[-0.05, 1.05]),
        yaxis=dict(range=[-0.05, 1.05]),
        legend=dict(
            yanchor="top", y=0.99, 
            xanchor="left", x=0.01
        )
    )
    fig = _apply_common_layout(
        fig, 'Recall', 'Precision', 
        'Precision-Recall Curve'
    )
    plotly_show_and_save(
        fig, show, output_path, 
        ['png', 'html'], verbose
    )
    return fig


# TODO: Remove if unused
def plot_feature_importance(
    feature_importances: pd.Series,
    output_path: Union[str, Path],
    threshold: float = 0.5,
    show: bool = False,
    verbose: bool = False
) -> go.Figure:
    """
    Plot horizontal bar chart of feature importances using Plotly.
    
    Args:
        feature_importances: Series with feature importances.
        threshold:           Minimum importance to display.
        output_path:         Output path for saving the plot.
        show:                Whether to display the plot.
        verbose:             Verbosity flag.
    """
    # Filter and sort features
    filtered = feature_importances[feature_importances > threshold]
    sorted_features = filtered.sort_values(ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=sorted_features.index,
        x=sorted_features.values,
        orientation='h',
        marker_color='#2ca02c'
    ))
    
    fig.update_layout(
        title='Feature Importances',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        template='plotly_white',
        height=600 + len(sorted_features) * 10,  # Dynamic height
        width=800,
        margin=dict(l=150)  # Extra margin for long feature names
    )
    fig = _apply_common_layout(
        fig, 'Importance Score', 'Features', 'Feature Importances'
    )
    
    fig.update_layout(yaxis=dict(showticklabels=True))
    plotly_show_and_save(fig, show, output_path, ['png', 'html'], verbose)
    if verbose:
        logger.info(f"Feature importance plot saved to: {output_path}")
    return fig


# SHAP    
def simplify_feature_name(taxon: str) -> str:
    """Simplify a feature name by selecting the most specific meaningful part."""
    parts = taxon.split(";")
    last = parts[-1].strip().lower()
    if last in {"__unclassified", "__uncultured", "__"}:
        return ";".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    return parts[-1]
    

def generate_unique_simplified_labels(feature_names: List[str]) -> List[str]:
    """Generate simplified labels while ensuring uniqueness."""
    simplified_labels = []
    used_labels = set()
    for f in feature_names:
        label = simplify_feature_name(f)
        base_label = label
        suffix = 1
        while label in used_labels:
            label = f"{base_label}_{suffix}"
            suffix += 1
        simplified_labels.append(label)
        used_labels.add(label)
    return simplified_labels


def shap_summary_bar(
    shap_values: np.array,
    feature_names: List[str],
    max_display: int = 20,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH_RECTANGLE
) -> Tuple[go.Figure, List[str]]:
    """
    Convert SHAP summary bar plot to a Plotly figure.

    Args:
        shap_values:   SHAP values array (n_samples, n_features).
        feature_names: List of full feature names.
        max_display:   Maximum number of features to display.

    Returns:
        Horizontal bar plot of mean absolute SHAP values.
    """
    # Compute mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Select top features
    top_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
    top_features_full = [feature_names[i] for i in top_indices]
    top_values = mean_abs_shap[top_indices]

    # Generate simplified labels and ensure uniqueness
    simplified_labels = generate_unique_simplified_labels(top_features_full)

    # Create horizontal bar plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=simplified_labels,
        x=top_values,
        orientation='h',
        marker_color='#1e88e5',
        hovertext=top_features_full,
        hoverinfo='text+x'
    ))
    
    fig = update_font_sizes(fig)
    fig = _apply_common_layout(
        fig, 
        'Mean |SHAP Value|', 'Features', 
        "SHAP Summary Bar Plot"
    )

    # Layout adjustments
    fig = update_font_sizes(fig)
    fig.update_layout(
        autosize=True,
        showlegend=False,
        height=height, width=width,
        yaxis=dict(
            title=dict(standoff=100), # Distance between title and ticks
            automargin=True,
            showticklabels=True
        )
    )
    return fig, top_features_full 
    

def shap_beeswarm(
    shap_values: np.array, 
    feature_values: np.array, 
    feature_names: List, 
    max_display: int = 20,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH_RECTANGLE
) -> go.Figure:
    """
    Convert SHAP beeswarm plot to a Plotly figure with simplified red-blue color scheme.
    
    Args:
        shap_values:    SHAP values array (n_samples, n_features).
        feature_values: Feature values array (n_samples, n_features).
        feature_names:  List of feature names.
        max_display:    Maximum number of features to display.
    
    Returns:
        Beeswarm plot of SHAP values.
    """
    # Compute mean absolute SHAP for feature ordering
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
    top_features_full = [feature_names[i] for i in top_indices]
    
    # Generate simplified labels and ensure uniqueness
    simplified_labels = generate_unique_simplified_labels(top_features_full)
    
    # Prepare figure
    fig = go.Figure()
    y_offset = 0.3  # Vertical spread for jitter
    
    # Define color scheme
    low_color = '#1e88e5'  # Blue for low values
    high_color = '#ff0d57'  # Red for high values
    
    # Add scatter traces for each feature
    np.random.seed(42)  # Consistent jitter
    for idx, feature_idx in enumerate(top_indices):
        shap_vals = shap_values[:, feature_idx]
        feat_vals = feature_values[:, feature_idx]
        
        # Generate jittered y-coordinates
        jitter = np.random.uniform(-y_offset, y_offset, size=len(shap_vals))
        y_pos = idx + jitter
        
        # Normalize feature values for coloring
        vmin, vmax = np.min(feat_vals), np.max(feat_vals)
        normalized_vals = (feat_vals - vmin) / (vmax - vmin + 1e-8)
        
        # Create colors using linear interpolation in RGB space
        colors = []
        for nv in normalized_vals:
            # Interpolate between blue (low) and red (high)
            r = int(30 + (255 - 30) * nv)
            g = int(136 + (13 - 136) * nv)
            b = int(229 + (87 - 229) * nv)
            colors.append(f'rgb({r},{g},{b})')
        
        # Add trace
        fig.add_trace(go.Scatter(
            x=shap_vals,
            y=y_pos,
            mode='markers',
            marker=dict(size=5, color=colors),
            name=feature_names[feature_idx],
            hoverinfo='text',
            text=[
                f"<b>Feature</b>: {feature_names[feature_idx]}<br>"
                f"<b>SHAP</b>: {shap_val:.4f}<br>"
                f"<b>Value</b>: {fv:.4f}"
                for shap_val, fv in zip(shap_vals, feat_vals)
            ],
            showlegend=False
        ))
    
    # Add zero line
    fig.add_shape(
        type='line',
        x0=0, y0=-0.5, x1=0, y1=len(top_features_full) - 0.5,
        line=dict(color='gray', width=1, dash='dash')
    )
    
    # Calculate padding for x-axis
    all_shap_vals = shap_values[:, top_indices]
    x_min = min(np.min(all_shap_vals), 0)
    x_max = max(np.max(all_shap_vals), 0)
    x_padding = 0.05 * (x_max - x_min)
    fig = update_font_sizes(fig)
    fig = _apply_common_layout(
        fig, 
        'SHAP Value', 
        'Features', 
        'SHAP Beeswarm Plot'
    )
    # Update layout with dynamic axis scaling
    fig.update_layout(
        autosize=True,
        hovermode='closest',
        height=height, width=width,
        xaxis=dict(
            range=[x_min - x_padding, 
                   x_max + x_padding], 
        ),
        yaxis=dict(
            title=dict(standoff=100),
            automargin=True,
            showticklabels=True,
            tickvals=list(range(len(top_features_full))),
            ticktext=simplified_labels,
            range=[-0.5, 
                   len(top_features_full) - 0.5]
        )
    )
    return fig
    

def shap_dependency_plot(
    shap_values: np.array, 
    feature_values: np.array, 
    feature_names: List[str], 
    feature: str, 
    max_points: int = 1000,
    interaction_feature: Optional[Union[str, None]] = None,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH_RECTANGLE
) -> go.Figure:
    """
    Create a SHAP dependency plot for a single feature with optional interaction coloring.
    
    Args:
        shap_values:     SHAP values array.
        feature_values:  Feature values array.
        feature_names:   List of feature names.
        feature:         Feature to plot.
        max_points:      Maximum points to show (downsample if exceeded).
        interaction_feature: Feature to use for coloring points (None, 'auto', or feature name).
    
    Returns:
        Dependency plot figure.
    """
    # Validate main feature
    if feature not in feature_names:
        raise ValueError(f"Feature '{feature}' not found in feature_names")
    idx = feature_names.index(feature)
    
    # Simplify feature name for display
    feature_display = simplify_feature_name(feature)
    
    # Extract main feature data
    x = feature_values[:, idx]
    y = shap_values[:, idx]
    
    # Prepare interaction feature data
    color_data = None
    color_title = None
    color_title_display = None  # For simplified display name
    auto_interaction = False
    
    # Handle interaction feature selection
    if interaction_feature:
        if interaction_feature == 'auto':
            auto_interaction = True
            # Find strongest interaction feature using variance explained
            best_j = None
            max_ss_between = -1
            current_shap = y
            
            # Use sampling for large datasets
            sample_size = min(10000, len(current_shap))
            if len(current_shap) > sample_size:
                sample_idx = np.random.choice(len(current_shap), sample_size, replace=False)
                current_shap_sample = current_shap[sample_idx]
                feature_values_sample = feature_values[sample_idx]
            else:
                current_shap_sample = current_shap
                feature_values_sample = feature_values
            
            # Iterate through features to find best interaction
            for j in range(len(feature_names)):
                if j == idx: 
                    continue
                try:
                    # Create bins for grouping
                    bins = np.percentile(feature_values_sample[:, j], np.linspace(0, 100, 11))
                    bins = np.unique(bins)
                    if len(bins) < 2: 
                        continue
                    
                    # Group data and calculate between-group variance
                    groups = np.digitize(feature_values_sample[:, j], bins)
                    ss_between = 0
                    overall_mean = np.mean(current_shap_sample)
                    
                    for group_id in np.unique(groups):
                        mask = groups == group_id
                        group_data = current_shap_sample[mask]
                        if len(group_data) == 0: 
                            continue
                        group_mean = np.mean(group_data)
                        ss_between += len(group_data) * (group_mean - overall_mean)**2
                    
                    # Update best feature if variance is higher
                    if ss_between > max_ss_between:
                        max_ss_between = ss_between
                        best_j = j
                except:
                    continue
            
            # Fallback if no valid feature found
            if best_j is None:
                for j in range(len(feature_names)):
                    if j != idx:
                        best_j = j
                        break
            color_title = feature_names[best_j]
            color_title_display = simplify_feature_name(color_title)
            color_data = feature_values[:, best_j]
        else:
            # Use specified interaction feature
            if interaction_feature not in feature_names:
                raise ValueError(f"Interaction feature '{interaction_feature}' not found")
            color_title = interaction_feature
            color_title_display = simplify_feature_name(interaction_feature)
            color_data = feature_values[:, feature_names.index(interaction_feature)]

    # Downsample if needed
    if len(x) > max_points:
        indices = np.random.choice(len(x), max_points, replace=False)
        x = x[indices]
        y = y[indices]
        if color_data is not None:
            color_data = color_data[indices]

    # Create figure
    fig = go.Figure()
    
    # Configure marker color based on interaction
    marker_config = {
        'size': 6,
        'opacity': 0.6,
        'showscale': True
    }
    hover_template = "<b>Value</b>: %{x:.4f}<br><b>SHAP</b>: %{y:.4f}"
    
    if color_data is not None:
        marker_config.update({
            'color': color_data,
            'colorscale': 'Viridis',
            'colorbar': {
                'title': {
                    'text': color_title_display, 
                    'side': 'right', 
                    'font': {'size': 20}
                }, 
                'tickfont': {'size': 16}
            },
            
        })
        hover_template += f"<br><b>{color_title}</b>: %{{marker.color:.4f}}"  # Full name in hover
    else:
        marker_config.update({
            'color': y,
            'colorscale': 'RdBu',
            'colorbar': {
                'title': {
                    'text': 'SHAP Value', 
                    'side': 'right', 
                    'font': {'size': 20}
                }, 
                'tickfont': {'size': 16}
            },
        })
    
    hover_template += "<extra></extra>"
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=x, 
        y=y, 
        mode='markers',
        marker=marker_config,
        name=feature_display,  # Use simplified name for legend
        hovertemplate=hover_template
    ))
    
    # Add trend line
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(y, x, frac=0.3, it=2)
        fig.add_trace(go.Scatter(
            x=smoothed[:, 0],
            y=smoothed[:, 1],
            mode='lines',
            line=dict(color='black', width=3),
            name='Trend'
        ))
    except ImportError:
        df = pd.DataFrame({'x': x, 'y': y}).sort_values('x')
        df['rolling'] = df['y'].rolling(50, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df['x'],
            y=df['rolling'],
            mode='lines',
            line=dict(color='black', width=3),
            name='Trend'
        ))
    
    # Calculate axis padding
    x_padding = 0.05 * (x.max() - x.min())
    y_padding = 0.05 * (y.max() - y.min())
    
    # Update layout with dynamic axis scaling
    title_suffix = " with interaction" if auto_interaction else ""    
    fig = update_font_sizes(fig)
    fig = _apply_common_layout(
        fig, 
        f'Feature Value: {feature_display}', 
        'SHAP Value', 
        f'SHAP Dependency Plot: {feature_display}{title_suffix}'
    )
    
    fig.update_layout(
        autosize=True,
        height=height, width=width,
        xaxis=dict(
            showgrid=False,
            mirror=True,
            range=[x.min() - x_padding, 
                   x.max() + x_padding],
        ),
        yaxis=dict(
            showgrid=False,
            mirror=True,
            range=[y.min() - y_padding, 
                   y.max() + y_padding]
        )
    )
    return fig
    

def shap_heatmap(
    shap_values: np.array,
    feature_values: np.array,
    feature_names: List[str],
    max_display: int = 20,
    max_samples: int = 1000,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH_RECTANGLE
) -> go.Figure:
    """
    Create a SHAP heatmap showing SHAP values across instances and features.

    Args:
        shap_values:     SHAP values array (n_samples, n_features).
        feature_values:  Feature values array (n_samples, n_features).
        feature_names:   List of feature names.
        max_display:     Maximum number of features to display.
        max_samples:     Maximum number of samples to display.

    Returns:
        Heatmap figure with clustered instances and features.
    """
    # Downsample instances if needed
    if len(shap_values) > max_samples:
        sample_idx = np.random.choice(len(shap_values), max_samples, replace=False)
        shap_values_sampled = shap_values[sample_idx]
        feature_values_sampled = feature_values[sample_idx]
    else:
        sample_idx = np.arange(len(shap_values))
        shap_values_sampled = shap_values
        feature_values_sampled = feature_values

    # Select top features by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values_sampled).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    
    # Cluster FEATURES using correlation of SHAP values
    corr_matrix = np.corrcoef(shap_values_sampled[:, top_indices].T)
    np.fill_diagonal(corr_matrix, 1.0)  # Ensure diagonal is 1 before absolute
    dist_matrix = 1 - np.abs(corr_matrix)
    feature_linkage = linkage(dist_matrix, method='complete', optimal_ordering=True)
    feature_order = leaves_list(feature_linkage)
    clustered_feature_names = [top_features[i] for i in feature_order]
    clustered_indices = top_indices[feature_order]  # Original feature indices in clustered order

    # Cluster INSTANCES using SHAP values of clustered features
    instance_linkage = linkage(shap_values_sampled[:, clustered_indices], method='complete')
    instance_order = leaves_list(instance_linkage)

    # Prepare clustered SHAP values and feature values
    clustered_shap = shap_values_sampled[instance_order][:, clustered_indices]
    clustered_feature_vals = feature_values_sampled[instance_order][:, clustered_indices]
    
    # Generate hover text
    hover_text = []
    for i, instance_idx in enumerate(instance_order):
        row_text = []
        for j, feat_idx in enumerate(clustered_indices):
            row_text.append(
                f"<b>Feature</b>: {feature_names[feat_idx]}<br>"
                f"<b>SHAP</b>: {clustered_shap[i, j]:.4f}<br>"
                f"<b>Value</b>: {clustered_feature_vals[i, j]:.4f}<br>"
                f"<b>Instance</b>: {sample_idx[instance_order[i]]}"
            )
        hover_text.append(row_text)

    # Create heatmap
    fig = go.Figure(go.Heatmap(
        z=clustered_shap,
        x=clustered_feature_names,
        y=[f"Instance {sample_idx[i]}" for i in instance_order],
        colorscale='Viridis',
        zmid=0,
        hoverinfo="text",
        text=hover_text,
        colorbar=dict(
            title=dict(text='SHAP Value', font=dict(size=14)),
        )
    ))

    # Apply layout adjustments
    fig = update_font_sizes(fig)
    fig = _apply_common_layout(
        fig, 
        "Features (clustered by similarity)", 
        "Instances (clustered by similarity)", 
        "SHAP Feature Importance"
    )
    
    # Update layout with dynamic axis scaling
    fig.update_layout(
        autosize=True,
        hovermode='closest',
        height=height, width=width,
        xaxis=dict(
            automargin=True,
            tickangle=-45, 
        ),
        yaxis=dict(
            title=dict(standoff=100),
            automargin=True,
            showticklabels=False
        )
    )
    return fig


def shap_force_plot(
    base_value: float,
    shap_values: np.array,
    feature_values: np.array,
    feature_names: List[str],
    instance_index: int = 0,
    max_display: int = 12,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH_RECTANGLE
) -> go.Figure:
    """
    Create a waterfall-style force plot showing feature contributions for a single instance.

    Args:
        base_value:      Base value (expected model output).
        shap_values:     SHAP values array (n_samples, n_features).
        feature_values:  Feature values array (n_samples, n_features).
        feature_names:   List of feature names.
        instance_index:  Index of instance to visualize.
        max_display:     Maximum number of features to display.

    Returns:
        Waterfall plot figure showing contribution breakdown.
    """
    # Select instance data
    instance_shap = shap_values[instance_index]
    instance_feature_values = feature_values[instance_index]
    prediction = base_value + instance_shap.sum()
    
    # Sort features by absolute SHAP value
    sorted_indices = np.argsort(np.abs(instance_shap))[::-1]
    top_indices = sorted_indices[:max_display]
    other_idx = sorted_indices[max_display:]
    
    # Prepare data
    top_features = [feature_names[i] for i in top_indices]
    top_shap = instance_shap[top_indices]
    top_values = instance_feature_values[top_indices]
    
    # Calculate "other" contribution
    other_contrib = instance_shap[other_idx].sum() if len(other_idx) > 0 else 0
    
    # Create cumulative values for waterfall
    cumulative_values = [base_value]
    current_value = base_value
    for shap_val in top_shap:
        current_value += shap_val
        cumulative_values.append(current_value)
    
    if other_contrib != 0:
        cumulative_values.append(current_value + other_contrib)
    
    # Prepare waterfall positions and labels
    y_labels = ['Base Value'] + top_features
    if other_contrib != 0:
        y_labels += [f'Other ({len(other_idx)} features)']
    y_labels += ['Prediction']
    
    # Create figure
    fig = go.Figure()
    
    # Add base value marker
    fig.add_trace(go.Scatter(
        x=[base_value],
        y=['Base Value'],
        mode='markers',
        marker=dict(size=18, color='#999999'),
        hoverinfo='text',
        text=f"<b>Base Value</b>: {base_value:.4f}"
    ))
    
    # Add feature contribution bars
    for i in range(len(top_features)):
        start_val = cumulative_values[i]
        end_val = cumulative_values[i+1]
        contribution = top_shap[i]
        
        fig.add_trace(go.Bar(
            x=[contribution],
            y=[top_features[i]],
            base=[start_val],
            orientation='h',
            marker=dict(
                color='#ff0d57' if contribution > 0 else '#1e88e5',
                line=dict(width=0)
            ),
            hoverinfo='text',
            text=(
                f"<b>Feature</b>: {top_features[i]}<br>"
                f"<b>Value</b>: {top_values[i]:.4f}<br>"
                f"<b>SHAP</b>: {contribution:.4f}<br>"
                f"<b>Cumulative</b>: {end_val:.4f}"
            )
        ))
    
    # Add other contributions if needed
    if other_contrib != 0:
        start_val = cumulative_values[-2]
        end_val = cumulative_values[-1]
        
        fig.add_trace(go.Bar(
            x=[other_contrib],
            y=[f'Other ({len(other_idx)} features)'],
            base=[start_val],
            orientation='h',
            marker=dict(
                color='#999999',
                line=dict(width=0)
            ),
            hoverinfo='text',
            text=f"<b>Sum of {len(other_idx)} other features</b>: {other_contrib:.4f}"
        ))
    
    # Add prediction marker
    fig.add_trace(go.Scatter(
        x=[prediction],
        y=['Prediction'],
        mode='markers',
        marker=dict(size=18, symbol='diamond', color='#000000'),
        hoverinfo='text',
        text=f"<b>Final Prediction</b>: {prediction:.4f}"
    ))
    
    # Add connector lines
    for i in range(len(cumulative_values)-1):
        fig.add_trace(go.Scatter(
            x=[cumulative_values[i], cumulative_values[i+1]],
            y=[y_labels[i], y_labels[i+1]],
            mode='lines',
            line=dict(color='#aaaaaa', width=1, dash='dot'),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Custom layout adjustments
    fig.update_layout(
        barmode='stack',
        showlegend=False,
        hovermode='closest',
        height=50 * len(y_labels),  # Dynamic height based on features
        yaxis=dict(
            categoryorder='array',
            categoryarray=list(reversed(y_labels))
        ),
        shapes=[
            # Base value line
            dict(
                type='line',
                x0=base_value,
                x1=base_value,
                y0=-1,
                y1=len(y_labels),
                line=dict(color='#999999', width=2, dash='dot'))
        ]
    )
    
    # Add annotations for cumulative values
    for i, val in enumerate(cumulative_values):
        fig.add_annotation(
            x=val,
            y=y_labels[i],
            xref='x',
            yref='y',
            text=f"{val:.4f}",
            showarrow=False,
            xanchor='left' if val >= base_value else 'right',
            yanchor='middle',
            font=dict(size=12),
            xshift=10 if val >= base_value else -10
        )
        
    fig = update_font_sizes(fig)
    fig = _apply_common_layout(
        fig, 
        'Model Output Value', 
        None, 
        f"SHAP Force Plot - Instance {instance_index}"
    )
    fig.update_layout(
        autosize=True,
        width=width,
        yaxis=dict(automargin=True)
    )
    return fig


# SHAP Waterfall Plot
def shap_waterfall_plot(
    base_value: float,
    shap_values: np.array,
    feature_values: np.array,
    feature_names: List[str],
    instance_index: int = 0,
    max_display: int = 10,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH_RECTANGLE
) -> go.Figure:
    """
    Create a waterfall plot showing cumulative feature contributions.

    Args:
        base_value:      Base value (expected model output).
        shap_values:     SHAP values array (n_samples, n_features).
        feature_values:  Feature values array (n_samples, n_features).
        feature_names:   List of feature names.
        instance_index:  Index of instance to visualize.
        max_display:     Maximum number of features to display.

    Returns:
        Waterfall plot figure showing contribution steps.
    """
    # Select instance data
    instance_shap = shap_values[instance_index]
    instance_feature_values = feature_values[instance_index]
    prediction = base_value + np.sum(instance_shap)
    
    # Sort features by absolute SHAP value
    sorted_indices = np.argsort(np.abs(instance_shap))[::-1]
    top_indices = sorted_indices[:max_display]
    other_idx = sorted_indices[max_display:]
    
    # Calculate other features contribution
    other_contrib = np.sum(instance_shap[other_idx]) if len(other_idx) > 0 else 0.0
    
    # Prepare waterfall data
    cumulative = base_value
    steps = [('Base Value', base_value, base_value)]
    
    for idx in top_indices:
        new_value = cumulative + instance_shap[idx]
        steps.append((
            feature_names[idx],
            instance_shap[idx],
            new_value
        ))
        cumulative = new_value
    
    if other_contrib != 0:
        steps.append((
            f"{len(other_idx)} Other Features",
            other_contrib,
            cumulative + other_contrib
        ))
        cumulative += other_contrib
    
    steps.append(('Prediction', 0, cumulative))
    
    # Generate simplified labels
    feature_labels = [simplify_feature_name(step[0]) for step in steps[:-1]]
    feature_labels.append(steps[-1][0])
    
    # Prepare hover text safely
    hover_text = []
    for i, step in enumerate(steps):
        # For features, try to get the value
        feat_val = "N/A"
        if step[0] in feature_names:
            feat_idx = feature_names.index(step[0])
            feat_val = f"{instance_feature_values[feat_idx]:.4f}"
        elif step[0] == "Base Value" or step[0] == "Prediction":
            feat_val = "N/A"
        
        # Format contribution safely
        cont_str = f"{step[1]:+.4f}" if (i > 0 and i < len(steps)-1) else ""
        
        text = (
            f"<b>{step[0]}</b><br>"
            f"<b>Value</b>: {feat_val}<br>"
            f"<b>Contribution</b>: {cont_str}<br>"
            f"<b>Cumulative</b>: {step[2]:.4f}"
        )
        hover_text.append(text)
    
    # Format bar text safely
    bar_text = []
    for i, step in enumerate(steps):
        if i > 0 and i < len(steps)-1:
            bar_text.append(f"{step[1]:+.4f}")
        else:
            bar_text.append(f"{step[2]:.4f}")
    
    # Create figure
    fig = go.Figure(go.Waterfall(
        name="",
        orientation="v",
        measure=["absolute"] + ["relative"] * (len(steps)-2) + ["total"],
        x=feature_labels,
        textposition="outside",
        text=bar_text,
        y=[step[1] for step in steps],
        connector={"line":{"color":"rgb(63, 63, 63)"}},
        increasing={"marker":{"color":"#1e88e5"}},
        decreasing={"marker":{"color":"#ff0d57"}},
        totals={"marker":{"color":"#000000"}},
        hoverinfo='text',
        hovertext=hover_text
    ))
    
    # Apply common layout
    fig = update_font_sizes(fig)
    fig = _apply_common_layout(
        fig,
        "Features",
        "Model Output Value",
        "SHAP Waterfall Plot"
    )
    
    # Custom layout adjustments
    fig.update_layout(
        height=height,
        showlegend=False,
        waterfallgap=0.3
    )
    return fig
    

def plot_shap(
    base_value: float,
    shap_values: np.array, 
    feature_values: np.array, 
    feature_names: list, 
    n_features: int = 20, 
    output_dir: Union[str, Path] = None,
    interaction_feature: Optional[Union[str, None]] = 'auto',
    show: bool = False,
    verbose: bool = False,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH_RECTANGLE
) -> dict:
    """
    Generate both SHAP bar plot, beeswarm plot, and dependency plots as Plotly figures.
    
    Args:
        shap_values:    SHAP values array.
        feature_values: Feature values array.
        feature_names:  List of feature names.
        n_features:     Maximum features to display.
        output_dir:
        show:
        verbose:        Verbosity flag.
    
    Returns:
        Tuple of bar_plot_fig, beeswarm_plot_fig, dependency_plot_figs.
    """
    output_dir = Path(output_dir) / 'figs'
    bar_fig, top_full_features = shap_summary_bar(
        shap_values, feature_names, n_features
    )
    plotly_show_and_save(
        bar_fig, show, 
        output_dir / f"shap.summary.bar.{n_features}", 
        ['png', 'html'], verbose
    )
    beeswarm_fig = shap_beeswarm(
        shap_values, feature_values, feature_names, n_features
    )
    plotly_show_and_save(
        beeswarm_fig, show, 
        output_dir / f"shap.summary.beeswarm.{n_features}",
        ['png', 'html'], verbose
    )
    heatmap_fig = shap_heatmap(
        shap_values,
        feature_values,
        feature_names,
        max_display=n_features,
        max_samples=1000
    )
    plotly_show_and_save(
        heatmap_fig, show, 
        output_dir / f"shap.summary.heatmap.{n_features}",
        ['png', 'html'], verbose
    )
    force_fig = shap_force_plot(
        base_value,
        shap_values,
        feature_values,
        feature_names,
        instance_index=0,
        max_display=12
    )
    plotly_show_and_save(
        force_fig, show, 
        output_dir / f"shap.summary.force.{n_features}",
        ['png', 'html'], verbose
    )
    waterfall_fig = shap_waterfall_plot(
        base_value,
        shap_values,
        feature_values,
        feature_names,
        instance_index=0,
        max_display=10
    )
    plotly_show_and_save(
        waterfall_fig, show, 
        output_dir / f"shap.summary.waterfall.{n_features}",
        ['png', 'html'], verbose
    )
    # Create dependency plots for top features
    dependency_figs = {}
    if n_features > 0:
        for feature in top_full_features[:n_features]:
            try:
                dep_fig = shap_dependency_plot(
                    shap_values, 
                    feature_values, 
                    feature_names, 
                    feature,  # Use full feature name here
                    10000, 
                    interaction_feature='auto'
                )
                #plotly_show_and_save(dep_fig, show, output_dir / f"shap.dependency.{feature}", ['png', 'html'], verbose)
                save_plotly_html(
                    fig=dep_fig, 
                    filepath=output_dir / f"shap.dependency.{feature}",
                    verbose=verbose
                )
                dependency_figs[feature] = dep_fig
            except Exception as e:
                logger.error(f"Error creating dependency plot for {feature}: {str(e)}")
    return {
        'bar_fig': bar_fig, 
        'beeswarm_fig': beeswarm_fig, 
        'heatmap_fig': heatmap_fig,
        'force_fig': force_fig,
        'waterfall_fig': waterfall_fig,
        'dependency_figs': dependency_figs
    }
