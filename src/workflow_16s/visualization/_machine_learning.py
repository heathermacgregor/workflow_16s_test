# src/workflow_16s/visualization/_machine_learning.py
# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

# Third-Party Imports
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list

# Local Imports
from workflow_16s.visualization import apply_common_layout, save_fig

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
    save_fig(fig, Path(output_path), formats=['png', 'html', 'json'])
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
    
    save_fig(fig, Path(output_path), formats=['png', 'html', 'json'])
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

    save_fig(fig, Path(output_path), formats=['png', 'html', 'json'])
    return fig

        
def plot_roc_curve(fpr, tpr, roc_auc, output_path: Union[str, Path], height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH_SQUARE) -> go.Figure:
    """Plot ROC curve using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})', line=dict(width=3, color='#1f77b4')))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random (AUC = 0.50)', line=dict(dash='dash', color='#444')))
    fig = update_font_sizes(fig)
    fig.update_layout(autosize=True, height=height, width=width, xaxis=dict(range=[-0.05, 1.05]), yaxis=dict(range=[-0.05, 1.05]), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),)
    fig = apply_common_layout(fig, 'False Positive Rate', 'True Positive Rate', 'Receiver Operating Characteristic') 
    save_fig(fig, Path(output_path), formats=['png', 'html', 'json'])
    return fig
    

def plot_precision_recall_curve(precision, recall, average_precision, output_path: Union[str, Path], show: bool = False, verbose: bool = False, height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH_SQUARE) -> go.Figure:
    """Plot Precision-Recall curve using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR curve (AP = {average_precision:.2f})', line=dict(width=3, color='#ff7f0e'), fill='tozeroy'))
    fig = update_font_sizes(fig)
    fig.update_layout(autosize=True, height=height, width=width, xaxis=dict(range=[-0.05, 1.05]), yaxis=dict(range=[-0.05, 1.05]), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    fig = apply_common_layout(fig, 'Recall', 'Precision', 'Precision-Recall Curve')
    save_fig(fig, Path(output_path), formats=['png', 'html', 'json'])
    return fig


def simplify_feature_name(taxon: str) -> str:
    parts = taxon.split(";")
    last = parts[-1].strip().lower()
    if last in {"__unclassified", "__uncultured", "__"}:
        return ";".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    return parts[-1]
    

def generate_unique_simplified_labels(feature_names: List[str]) -> List[str]:
    simplified_labels = []; used_labels = set()
    for f in feature_names:
        label = simplify_feature_name(f); base_label = label; suffix = 1
        while label in used_labels:
            label = f"{base_label}_{suffix}"; suffix += 1
        simplified_labels.append(label); used_labels.add(label)
    return simplified_labels


def shap_summary_bar(shap_values: np.ndarray, feature_names: List[str], max_display: int = 20, height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH_RECTANGLE) -> Tuple[go.Figure, List[str]]:
    """Convert SHAP summary bar plot to a Plotly figure."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
    top_features_full = [feature_names[i] for i in top_indices]
    top_values = mean_abs_shap[top_indices]
    simplified_labels = generate_unique_simplified_labels(top_features_full)

    fig = go.Figure()
    fig.add_trace(go.Bar(y=simplified_labels, x=top_values, orientation='h', marker_color='#1e88e5', hovertext=top_features_full, hoverinfo='text+x'))
    fig = update_font_sizes(fig)
    fig = apply_common_layout(fig, 'Mean |SHAP Value|', 'Features', "SHAP Summary Bar Plot")
    fig.update_layout(autosize=True, showlegend=False, height=height+100, width=width, yaxis=dict(title=dict(standoff=100), automargin=True, showticklabels=True))
    return fig, top_features_full 
    

def shap_beeswarm(shap_values: np.ndarray, feature_values: np.ndarray, feature_names: List, max_display: int = 20, height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH_RECTANGLE) -> go.Figure:
    """Convert SHAP beeswarm plot to a Plotly figure with simplified red-blue color scheme."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
    top_features_full = [feature_names[i] for i in top_indices]
    simplified_labels = generate_unique_simplified_labels(top_features_full)
    
    fig = go.Figure()
    y_offset = 0.3
    
    np.random.seed(42)
    for idx, feature_idx in enumerate(top_indices):
        shap_vals = shap_values[:, feature_idx]
        feat_vals = feature_values[:, feature_idx]
        jitter = np.random.uniform(-y_offset, y_offset, size=len(shap_vals))
        y_pos = idx + jitter
        
        # --- ROBUST FIX for Strings ---
        try:
            feat_vals_numeric = feat_vals.astype(float)
            vmin, vmax = np.nanmin(feat_vals_numeric), np.nanmax(feat_vals_numeric)
            denominator = vmax - vmin
            if denominator == 0: denominator = 1e-8
            normalized_vals = (feat_vals_numeric - vmin) / denominator
            colors = []
            for nv in normalized_vals:
                if np.isnan(nv): colors.append('#999999')
                else:
                    r = int(30 + (255 - 30) * nv)
                    g = int(136 + (13 - 136) * nv)
                    b = int(229 + (87 - 229) * nv)
                    colors.append(f'rgb({r},{g},{b})')
            hover_text = [f"<b>Feature</b>: {feature_names[feature_idx]}<br><b>SHAP</b>: {shap_val:.4f}<br><b>Value</b>: {fv:.4f}" for shap_val, fv in zip(shap_vals, feat_vals_numeric)]

        except (ValueError, TypeError):
            colors = '#999999' # Neutral grey for strings
            hover_text = [f"<b>Feature</b>: {feature_names[feature_idx]}<br><b>SHAP</b>: {shap_val:.4f}<br><b>Value</b>: {fv}" for shap_val, fv in zip(shap_vals, feat_vals)]
        
        fig.add_trace(go.Scatter(x=shap_vals, y=y_pos, mode='markers', marker=dict(size=5, color=colors), name=feature_names[feature_idx], hoverinfo='text', text=hover_text, showlegend=False))
    
    fig.add_shape(type='line', x0=0, y0=-0.5, x1=0, y1=len(top_features_full) - 0.5, line=dict(color='gray', width=1, dash='dash'))
    
    all_shap_vals = shap_values[:, top_indices]
    x_min = min(np.min(all_shap_vals), 0)
    x_max = max(np.max(all_shap_vals), 0)
    x_padding = 0.05 * (x_max - x_min)
    fig = update_font_sizes(fig)
    fig = apply_common_layout(fig, 'SHAP Value', 'Features', 'SHAP Beeswarm Plot')
    fig.update_layout(autosize=True, hovermode='closest', height=height, width=width, xaxis=dict(range=[x_min - x_padding, x_max + x_padding]), yaxis=dict(title=dict(standoff=100), automargin=True, showticklabels=True, tickvals=list(range(len(top_features_full))), ticktext=simplified_labels, range=[-0.5, len(top_features_full) - 0.5]))
    return fig
    

def shap_dependency_plot(shap_values: np.ndarray, feature_values: np.ndarray, feature_names: List[str], feature: str, max_points: int = 1000, interaction_feature: Optional[Union[str, None]] = None, height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH_RECTANGLE) -> go.Figure:
    """Create a SHAP dependency plot for a single feature with optional interaction coloring."""
    if feature not in feature_names: raise ValueError(f"Feature '{feature}' not found in feature_names")
    idx = feature_names.index(feature)
    feature_display = simplify_feature_name(feature)
    x = feature_values[:, idx]
    y = shap_values[:, idx]
    
    color_data = None; color_title = None; color_title_display = None; auto_interaction = False
    
    if interaction_feature:
        if interaction_feature == 'auto':
            auto_interaction = True; best_j = None; max_ss_between = -1; current_shap = y
            sample_size = min(10000, len(current_shap))
            if len(current_shap) > sample_size:
                sample_idx = np.random.choice(len(current_shap), sample_size, replace=False)
                current_shap_sample = current_shap[sample_idx]; feature_values_sample = feature_values[sample_idx]
            else:
                current_shap_sample = current_shap; feature_values_sample = feature_values
            
            for j in range(len(feature_names)):
                if j == idx: continue
                
                # --- FIX: Handle Categorical Interaction Candidates ---
                try:
                    col_data = feature_values_sample[:, j]
                    is_numeric = False
                    try:
                        col_data.astype(float)
                        is_numeric = True
                    except (ValueError, TypeError):
                        is_numeric = False

                    groups = None
                    if is_numeric:
                        # Use percentile bins for numeric
                        bins = np.percentile(col_data.astype(float), np.linspace(0, 100, 11))
                        bins = np.unique(bins)
                        if len(bins) < 2: continue
                        groups = np.digitize(col_data.astype(float), bins)
                    else:
                        # Use unique values for categorical
                        # Limit to features with < 50 categories to avoid overfitting/noise
                        unique_vals, groups = np.unique(col_data.astype(str), return_inverse=True)
                        if len(unique_vals) > 50 or len(unique_vals) < 2: continue
                    
                    # Calculate between-group variance
                    ss_between = 0; overall_mean = np.mean(current_shap_sample)
                    for group_id in np.unique(groups):
                        mask = groups == group_id; group_data = current_shap_sample[mask]
                        if len(group_data) == 0: continue
                        group_mean = np.mean(group_data); ss_between += len(group_data) * (group_mean - overall_mean)**2
                    
                    if ss_between > max_ss_between: max_ss_between = ss_between; best_j = j
                except Exception: continue
                # ---------------------------------------------------
            
            if best_j is None:
                for j in range(len(feature_names)):
                    if j != idx: best_j = j; break
            
            if best_j is None: interaction_feature = None
            else:
                color_title = feature_names[best_j]; color_title_display = simplify_feature_name(color_title); color_data = feature_values[:, best_j]
        else:
            if interaction_feature not in feature_names: raise ValueError(f"Interaction feature '{interaction_feature}' not found")
            color_title = interaction_feature; color_title_display = simplify_feature_name(interaction_feature); color_data = feature_values[:, feature_names.index(interaction_feature)]

    if len(x) > max_points:
        indices = np.random.choice(len(x), max_points, replace=False); x = x[indices]; y = y[indices]
        if color_data is not None: color_data = color_data[indices]

    fig = go.Figure()
    marker_config = {'size': 6, 'opacity': 0.6, 'showscale': True}
    hover_template = "<b>Value</b>: %{x}<br><b>SHAP</b>: %{y:.4f}"
    
    if color_data is not None:
        # --- Robust Color Mapping for Plotting ---
        try:
            color_data_numeric = color_data.astype(float)
            marker_config.update({'color': color_data_numeric, 'colorscale': 'Viridis', 'colorbar': {'title': {'text': color_title_display, 'side': 'right', 'font': {'size': 20}}, 'tickfont': {'size': 16}}})
            hover_template += f"<br><b>{color_title}</b>: %{{marker.color:.4f}}"
        except (ValueError, TypeError):
             # Map strings to integers for color scale
             unique_cats = np.unique(color_data.astype(str))
             cat_map = {cat: i for i, cat in enumerate(unique_cats)}
             color_data_mapped = np.array([cat_map[str(val)] for val in color_data])
             # Use discrete colors if few categories
             marker_config.update({'color': color_data_mapped, 'colorscale': 'Turbo', 'colorbar': {'title': {'text': color_title_display, 'side': 'right'}, 'tickvals': list(cat_map.values()), 'ticktext': list(cat_map.keys())}})
             hover_template += f"<br><b>{color_title}</b>: %{{text}}"
        # -----------------------------------------
    else:
        marker_config.update({'color': y, 'colorscale': 'RdBu', 'colorbar': {'title': {'text': 'SHAP Value', 'side': 'right', 'font': {'size': 20}}, 'tickfont': {'size': 16}}})
    
    hover_template += "<extra></extra>"
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=marker_config, name=feature_display, hovertemplate=hover_template, text=color_data.astype(str) if color_data is not None else None))
    
    # Trend line (skip if x is not numeric)
    try:
        x_float = x.astype(float)
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(y, x_float, frac=0.3, it=2)
        fig.add_trace(go.Scatter(x=smoothed[:, 0], y=smoothed[:, 1], mode='lines', line=dict(color='black', width=3), name='Trend'))
    except (ImportError, ValueError, TypeError): pass
    
    # Axis padding
    try:
        x_float = x.astype(float)
        x_padding = 0.05 * (x_float.max() - x_float.min()); y_padding = 0.05 * (y.max() - y.min())
        xaxis_dict = dict(showgrid=False, mirror=True, range=[x_float.min() - x_padding, x_float.max() + x_padding])
    except:
        xaxis_dict = dict(showgrid=False, mirror=True) # Categorical x-axis
        y_padding = 0.05 * (y.max() - y.min())
    
    title_suffix = " with interaction" if auto_interaction else ""    
    fig = update_font_sizes(fig)
    fig = apply_common_layout(fig, f'Feature Value: {feature_display}', 'SHAP Value', f'SHAP Dependency Plot: {feature_display}{title_suffix}')
    fig.update_layout(autosize=True, height=height, width=width, xaxis=xaxis_dict, yaxis=dict(showgrid=False, mirror=True, range=[y.min() - y_padding, y.max() + y_padding]))
    return fig
    

def shap_heatmap(shap_values: np.ndarray, feature_values: np.ndarray, feature_names: List[str], max_display: int = 20, max_samples: int = 1000, height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH_RECTANGLE) -> go.Figure:
    """Create a SHAP heatmap showing SHAP values across instances and features."""
    if len(shap_values) > max_samples:
        sample_idx = np.random.choice(len(shap_values), max_samples, replace=False); shap_values_sampled = shap_values[sample_idx]; feature_values_sampled = feature_values[sample_idx]
    else:
        sample_idx = np.arange(len(shap_values)); shap_values_sampled = shap_values; feature_values_sampled = feature_values

    mean_abs_shap = np.abs(shap_values_sampled).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    
    corr_matrix = np.corrcoef(shap_values_sampled[:, top_indices].T)
    np.fill_diagonal(corr_matrix, 1.0)
    dist_matrix = 1 - np.abs(corr_matrix)
    dist_matrix = np.nan_to_num(dist_matrix, nan=0.0)
    
    feature_linkage = linkage(dist_matrix, method='complete', optimal_ordering=True)
    feature_order = leaves_list(feature_linkage)
    clustered_feature_names = [top_features[i] for i in feature_order]
    clustered_indices = top_indices[feature_order]

    instance_linkage = linkage(shap_values_sampled[:, clustered_indices], method='complete')
    instance_order = leaves_list(instance_linkage)

    clustered_shap = shap_values_sampled[instance_order][:, clustered_indices]
    clustered_feature_vals = feature_values_sampled[instance_order][:, clustered_indices]
    
    hover_text = []
    for i, instance_idx in enumerate(instance_order):
        row_text = []
        for j, feat_idx in enumerate(clustered_indices):
            val_display = "N/A"
            try:
                val = float(clustered_feature_vals[i, j])
                val_display = f"{val:.4f}"
            except (ValueError, TypeError):
                val_display = str(clustered_feature_vals[i, j])
            row_text.append(f"<b>Feature</b>: {feature_names[feat_idx]}<br><b>SHAP</b>: {clustered_shap[i, j]:.4f}<br><b>Value</b>: {val_display}<br><b>Instance</b>: {sample_idx[instance_order[i]]}")
        hover_text.append(row_text)

    fig = go.Figure(go.Heatmap(z=clustered_shap, x=clustered_feature_names, y=[f"Instance {sample_idx[i]}" for i in instance_order], colorscale='Viridis', zmid=0, hoverinfo="text", text=hover_text, colorbar=dict(title=dict(text='SHAP Value', font=dict(size=14)),)))
    fig = update_font_sizes(fig)
    fig = apply_common_layout(fig, "Features (clustered by similarity)", "Instances (clustered by similarity)", "SHAP Feature Importance")
    fig.update_layout(autosize=True, hovermode='closest', height=height, width=width, xaxis=dict(automargin=True, tickangle=-45), yaxis=dict(title=dict(standoff=100), automargin=True, showticklabels=False))
    return fig


def shap_force_plot(base_value: float, shap_values: np.ndarray, feature_values: np.ndarray, feature_names: List[str], instance_index: int = 0, max_display: int = 12, height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH_RECTANGLE) -> go.Figure:
    """Create a waterfall-style force plot showing feature contributions for a single instance."""
    instance_shap = shap_values[instance_index]
    instance_feature_values = feature_values[instance_index]
    prediction = base_value + instance_shap.sum()
    
    sorted_indices = np.argsort(np.abs(instance_shap))[::-1]
    top_indices = sorted_indices[:max_display]
    other_idx = sorted_indices[max_display:]
    
    top_features = [feature_names[i] for i in top_indices]
    top_shap = instance_shap[top_indices]
    top_values = instance_feature_values[top_indices]
    other_contrib = instance_shap[other_idx].sum() if len(other_idx) > 0 else 0
    
    cumulative_values = [base_value]; current_value = base_value
    for shap_val in top_shap:
        current_value += shap_val; cumulative_values.append(current_value)
    if other_contrib != 0: cumulative_values.append(current_value + other_contrib)
    
    y_labels = ['Base Value'] + top_features
    if other_contrib != 0: y_labels += [f'Other ({len(other_idx)} features)']
    y_labels += ['Prediction']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[base_value], y=['Base Value'], mode='markers', marker=dict(size=18, color='#999999'), hoverinfo='text', text=f"<b>Base Value</b>: {base_value:.4f}"))
    
    for i in range(len(top_features)):
        start_val = cumulative_values[i]; end_val = cumulative_values[i+1]; contribution = top_shap[i]
        try: val_str = f"{float(top_values[i]):.4f}"
        except: val_str = str(top_values[i])
        fig.add_trace(go.Bar(x=[contribution], y=[top_features[i]], base=[start_val], orientation='h', marker=dict(color='#ff0d57' if contribution > 0 else '#1e88e5', line=dict(width=0)), hoverinfo='text', text=(f"<b>Feature</b>: {top_features[i]}<br><b>Value</b>: {val_str}<br><b>SHAP</b>: {contribution:.4f}<br><b>Cumulative</b>: {end_val:.4f}")))
    
    if other_contrib != 0:
        start_val = cumulative_values[-2]; end_val = cumulative_values[-1]
        fig.add_trace(go.Bar(x=[other_contrib], y=[f'Other ({len(other_idx)} features)'], base=[start_val], orientation='h', marker=dict(color='#999999', line=dict(width=0)), hoverinfo='text', text=f"<b>Sum of {len(other_idx)} other features</b>: {other_contrib:.4f}"))
    
    fig.add_trace(go.Scatter(x=[prediction], y=['Prediction'], mode='markers', marker=dict(size=18, symbol='diamond', color='#000000'), hoverinfo='text', text=f"<b>Final Prediction</b>: {prediction:.4f}"))
    
    for i in range(len(cumulative_values)-1):
        fig.add_trace(go.Scatter(x=[cumulative_values[i], cumulative_values[i+1]], y=[y_labels[i], y_labels[i+1]], mode='lines', line=dict(color='#aaaaaa', width=1, dash='dot'), hoverinfo='none', showlegend=False))
    
    fig.update_layout(barmode='stack', showlegend=False, hovermode='closest', height=50 * len(y_labels), yaxis=dict(categoryorder='array', categoryarray=list(reversed(y_labels))), shapes=[dict(type='line', x0=base_value, x1=base_value, y0=-1, y1=len(y_labels), line=dict(color='#999999', width=2, dash='dot'))])
    
    annotations = [dict(x=val, y=y_labels[i], xref='x', yref='y', text=f"{val:.4f}", showarrow=False, xanchor='left' if val >= base_value else 'right', yanchor='middle', font=dict(size=12), xshift=10 if val >= base_value else -10) for i, val in enumerate(cumulative_values)]
    for annotation in annotations: fig.add_annotation(**annotation) # type: ignore
        
    fig = update_font_sizes(fig)
    fig = apply_common_layout(fig, 'Model Output Value', "", f"SHAP Force Plot - Instance {instance_index}")
    fig.update_layout(autosize=True, width=width, yaxis=dict(automargin=True))
    return fig


def shap_waterfall_plot(base_value: float, shap_values: np.ndarray, feature_values: np.ndarray, feature_names: List[str], instance_index: int = 0, max_display: int = 10, height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH_RECTANGLE) -> go.Figure:
    """Create a waterfall plot showing cumulative feature contributions."""
    instance_shap = shap_values[instance_index]
    instance_feature_values = feature_values[instance_index]
    prediction = base_value + np.sum(instance_shap)
    
    sorted_indices = np.argsort(np.abs(instance_shap))[::-1]
    top_indices = sorted_indices[:max_display]
    other_idx = sorted_indices[max_display:]
    
    other_contrib = np.sum(instance_shap[other_idx]) if len(other_idx) > 0 else 0.0
    
    cumulative = base_value
    steps = [('Base Value', base_value, base_value)]
    
    for idx in top_indices:
        new_value = cumulative + instance_shap[idx]
        steps.append((feature_names[idx], instance_shap[idx], new_value))
        cumulative = new_value
    
    if other_contrib != 0:
        steps.append((f"{len(other_idx)} Other Features", other_contrib, cumulative + other_contrib))
        cumulative += other_contrib
    
    steps.append(('Prediction', 0, cumulative))
    
    feature_labels = [simplify_feature_name(step[0]) for step in steps[:-1]]
    feature_labels.append(steps[-1][0])
    
    hover_text = []
    for i, step in enumerate(steps):
        feat_val = "N/A"
        if step[0] in feature_names:
            feat_idx = feature_names.index(step[0])
            try: feat_val = f"{float(instance_feature_values[feat_idx]):.4f}"
            except: feat_val = str(instance_feature_values[feat_idx])
        elif step[0] == "Base Value" or step[0] == "Prediction":
            feat_val = "N/A"
        
        cont_str = f"{step[1]:+.4f}" if (i > 0 and i < len(steps)-1) else ""
        text = (f"<b>{step[0]}</b><br><b>Value</b>: {feat_val}<br><b>Contribution</b>: {cont_str}<br><b>Cumulative</b>: {step[2]:.4f}")
        hover_text.append(text)
    
    bar_text = []
    for i, step in enumerate(steps):
        if i > 0 and i < len(steps)-1: bar_text.append(f"{step[1]:+.4f}")
        else: bar_text.append(f"{step[2]:.4f}")
    
    fig = go.Figure(go.Waterfall(name="", orientation="v", measure=["absolute"] + ["relative"] * (len(steps)-2) + ["total"], x=feature_labels, textposition="outside", text=bar_text, y=[step[1] for step in steps], connector={"line":{"color":"rgb(63, 63, 63)"}}, increasing={"marker":{"color":"#1e88e5"}}, decreasing={"marker":{"color":"#ff0d57"}}, totals={"marker":{"color":"#000000"}}, hoverinfo='text', hovertext=hover_text))
    fig = update_font_sizes(fig)
    fig = apply_common_layout(fig, "Features", "Model Output Value", "SHAP Waterfall Plot")
    fig.update_layout(height=height, showlegend=False, waterfallgap=0.3)
    return fig
    

def shap_interaction_heatmap(shap_interaction_values: np.ndarray, feature_names: List[str], max_display: int = 15, height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH_SQUARE) -> go.Figure:
    """
    Create SHAP interaction heatmap showing pairwise feature interactions.
    
    Args:
        shap_interaction_values: 3D array [n_samples, n_features, n_features] or 2D mean interaction matrix
        feature_names: List of feature names
        max_display: Maximum number of features to display
        height: Figure height in pixels
        width: Figure width in pixels
    
    Returns:
        Plotly figure object
    """
    # Handle 3D interaction values (average across samples)
    if shap_interaction_values.ndim == 3:
        # Mean absolute interaction strength across all samples
        interaction_matrix = np.abs(shap_interaction_values).mean(axis=0)
    else:
        interaction_matrix = shap_interaction_values
    
    # Select top features by total interaction strength
    interaction_strength = interaction_matrix.sum(axis=1)
    top_indices = np.argsort(interaction_strength)[-max_display:][::-1]
    
    # Extract submatrix
    submatrix = interaction_matrix[top_indices][:, top_indices]
    top_features = [feature_names[i] for i in top_indices]
    simplified_labels = generate_unique_simplified_labels(top_features)
    
    # Create hover text
    hover_text = []
    for i in range(len(top_features)):
        row_text = []
        for j in range(len(top_features)):
            row_text.append(
                f"<b>{top_features[i]}</b> ↔ <b>{top_features[j]}</b><br>"
                f"Interaction: {submatrix[i, j]:.4f}"
            )
        hover_text.append(row_text)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=submatrix,
        x=simplified_labels,
        y=simplified_labels,
        colorscale='Viridis',
        hoverinfo='text',
        text=hover_text,
        colorbar=dict(title=dict(text='Mean |Interaction|', font=dict(size=14)))
    ))
    
    fig = update_font_sizes(fig)
    fig = apply_common_layout(fig, 'Feature', 'Feature', 'SHAP Feature Interactions')
    fig.update_layout(
        autosize=True,
        height=height,
        width=width,
        xaxis=dict(tickangle=-45, side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    
    return fig


def plot_shap(base_value: float, shap_values: np.ndarray, feature_values: np.ndarray, feature_names: list, n_features: int = 20, output_dir: Optional[Union[str, Path]] = None, interaction_feature: Optional[Union[str, None]] = 'auto', show: bool = False, verbose: bool = False, height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH_RECTANGLE, is_multiclass_avg: bool = False, shap_interaction_values: Optional[np.ndarray] = None) -> dict:
    """Generate both SHAP bar plot, beeswarm plot, dependency plots, and optional interaction plots."""
    if output_dir is None:
        can_save = False; output_path_base = None
    else:
        can_save = True; output_path_base = Path(output_dir) / 'figs'
        output_path_base.parent.mkdir(parents=True, exist_ok=True)

    bar_fig, top_full_features = shap_summary_bar(shap_values, feature_names, n_features, height=height, width=width)
    if can_save and output_path_base:
        save_fig(bar_fig, output_path_base / f"shap.summary.bar.{n_features}", formats=['png', 'html', 'json'])
        
    if is_multiclass_avg:
        logger.warning("Skipping SHAP beeswarm, heatmap, force, waterfall, and dependency plots for multiclass target.")
        beeswarm_fig = None; heatmap_fig = None; force_fig = None; waterfall_fig = None; dependency_figs = {}; interaction_fig = None
    else:
        beeswarm_fig = shap_beeswarm(shap_values, feature_values, feature_names, n_features, height=height, width=width)
        if can_save and output_path_base:
            save_fig(beeswarm_fig, output_path_base / f"shap.summary.beeswarm.{n_features}", formats=['png', 'html', 'json'])
            
        heatmap_fig = shap_heatmap(shap_values, feature_values, feature_names, max_display=n_features, max_samples=1000, height=height, width=width)
        if can_save and output_path_base:
            save_fig(fig=heatmap_fig, output_path=output_path_base / f"shap.summary.heatmap.{n_features}", formats=['png', 'html', 'json'])
            
        force_fig = shap_force_plot(base_value, shap_values, feature_values, feature_names, instance_index=0, max_display=12, height=height, width=width)
        if can_save and output_path_base:
            save_fig(fig=force_fig, output_path=output_path_base / f"shap.summary.force.{n_features}", formats=['png', 'html', 'json'])
            
        waterfall_fig = shap_waterfall_plot(base_value, shap_values, feature_values, feature_names, instance_index=0, max_display=10, height=height, width=width)
        if can_save and output_path_base:
            save_fig(fig=waterfall_fig, output_path=output_path_base / f"shap.summary.waterfall.{n_features}", formats=['png', 'html', 'json'])
            
        # Generate interaction heatmap if interaction values provided
        interaction_fig = None
        if shap_interaction_values is not None:
            try:
                logger.info("Generating SHAP interaction heatmap...")
                interaction_fig = shap_interaction_heatmap(
                    shap_interaction_values, 
                    feature_names, 
                    max_display=min(n_features, 15),  # Limit for readability
                    height=height, 
                    width=DEFAULT_WIDTH_SQUARE
                )
                if can_save and output_path_base:
                    save_fig(fig=interaction_fig, output_path=output_path_base / f"shap.interactions.heatmap.{n_features}", formats=['png', 'html', 'json'])
            except Exception as e:
                logger.error(f"Error creating SHAP interaction heatmap: {str(e)}")
            
        dependency_figs = {}
        if n_features > 0:
            for feature in top_full_features[:n_features]:
                try:
                    dep_fig = shap_dependency_plot(shap_values, feature_values, feature_names, feature, 10000, interaction_feature='auto', height=height, width=width)
                    if can_save and output_path_base:
                        save_fig(fig=dep_fig, output_path=output_path_base / f"shap.dependency.{feature}", formats=['png', 'html', 'json'])
                    dependency_figs[feature] = dep_fig
                except Exception as e:
                    logger.error(f"Error creating dependency plot for {feature}: {str(e)}")

    return {'bar_fig': bar_fig, 'beeswarm_fig': beeswarm_fig, 'heatmap_fig': heatmap_fig, 'force_fig': force_fig, 'waterfall_fig': waterfall_fig, 'dependency_figs': dependency_figs, 'interaction_fig': interaction_fig}