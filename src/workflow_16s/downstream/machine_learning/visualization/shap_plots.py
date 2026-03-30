# src/workflow_16s/downstream/machine_learning/visualization/shap_plots.py

import numpy as np
import plotly.graph_objects as go

from pathlib import Path
from typing import List, Optional, Tuple, Union
from scipy.cluster.hierarchy import linkage, leaves_list

from .taxa_labels import (
    generate_unique_simplified_labels, 
    simplify_feature_name
)
from .base_style import (
    update_font_sizes, 
    DEFAULT_HEIGHT, 
    DEFAULT_WIDTH_RECTANGLE, 
    DEFAULT_WIDTH_SQUARE
)
from workflow_16s.visualization import (
    apply_common_layout, save_fig
)
from workflow_16s.utils.logger import get_logger
from workflow_16s.visualization.utils import PlottingUtils


def shap_summary_bar(
    shap_values: Union[np.ndarray, List[np.ndarray]], 
    feature_names: List[str], 
    max_display: int = 20, 
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH_RECTANGLE
) -> Tuple[go.Figure, List[str]]:
    """
    Convert SHAP summary bar plot to a Plotly figure.
    """
    logger = get_logger("workflow_16s")
    plot_utils = PlottingUtils(logger)
    if isinstance(shap_values, list):
        # Average the absolute values across all classes
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Ensure mean_abs_shap is 1D (flatten if needed)
    mean_abs_shap = np.asarray(mean_abs_shap).ravel()
    
    # Validate that we have consistent dimensions
    if len(mean_abs_shap) != len(feature_names):
        logger.warning(f"⚠️ Feature count mismatch: mean_abs_shap has {len(mean_abs_shap)} features, but feature_names has {len(feature_names)}")
        # Truncate to match
        n_features = min(len(mean_abs_shap), len(feature_names))
        mean_abs_shap = mean_abs_shap[:n_features]
        feature_names = feature_names[:n_features]
    
    # Get top feature indices safely - limit to available features
    n_display = min(max_display, len(feature_names))
    top_indices = np.argsort(mean_abs_shap)[-n_display:][::-1]
    # Convert to Python ints - handle case where indices might be arrays
    top_indices = [int(i.item() if hasattr(i, 'item') else i) for i in top_indices]
    
    # Final safety check: ensure all indices are valid
    top_indices = [i for i in top_indices if i < len(feature_names)]
    
    if not top_indices:
        logger.warning(f"⚠️ No valid feature indices for SHAP bar plot")
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(text="No valid features for plot", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig, []

    top_features_full = [feature_names[i] for i in top_indices]
    top_values = mean_abs_shap[top_indices]
    simplified_labels = generate_unique_simplified_labels(top_features_full)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=simplified_labels, 
            x=top_values, 
            orientation='h', 
            marker_color='#1e88e5', 
            hovertext=[f"<b>{label}</b><br>Impact: {val:.4f}" for label, val in zip(simplified_labels, top_values)],
            hoverinfo='text',
            showlegend=False
        )
    )
    fig.update_layout(
        autosize=True, 
        showlegend=False, 
        height=height+100, 
        width=width,
        font=dict(size=11),
        yaxis=dict(
            title=None,
            automargin=True, 
            showticklabels=True,
            tickfont=dict(size=10)
        ),
        xaxis=dict(
            title=dict(text='Mean |SHAP Value|', font=dict(size=12)),
            tickfont=dict(size=10)
        ),
        margin=dict(l=150, r=40, t=40, b=40)
    )
    return fig, top_features_full
    

def shap_beeswarm(
    shap_values: np.ndarray, 
    feature_values: np.ndarray, 
    feature_names: List[str], 
    max_display: int = 20, 
    height: int = DEFAULT_HEIGHT, 
    width: int = DEFAULT_WIDTH_RECTANGLE
) -> go.Figure:
    """
    Convert SHAP beeswarm plot to a Plotly figure with simplified red-blue color scheme.
    """
    logger = get_logger("workflow_16s")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
    top_indices = [int(i) for i in top_indices]
    
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
            hover_text = [f"<b>SHAP</b>: {shap_val:.2f}<br><b>Value</b>: {fv:.2f}" for shap_val, fv in zip(shap_vals, feat_vals_numeric)]

        except (ValueError, TypeError):
            colors = '#999999' # Neutral grey for strings
            hover_text = [f"<b>SHAP</b>: {shap_val:.2f}<br><b>Value</b>: {fv}" for shap_val, fv in zip(shap_vals, feat_vals)]
        
        fig.add_trace(
            go.Scatter(
                x=shap_vals, 
                y=y_pos, 
                mode='markers', 
                marker=dict(size=4, color=colors), 
                name=feature_names[feature_idx], 
                hoverinfo='text', 
                text=hover_text, 
                showlegend=False
            )
        )
    
    fig.add_shape(type='line', x0=0, y0=-0.5, x1=0, y1=len(top_features_full) - 0.5, line=dict(color='#ccc', width=1, dash='dash'))
    
    all_shap_vals = shap_values[:, top_indices]
    x_min = min(np.min(all_shap_vals), 0)
    x_max = max(np.max(all_shap_vals), 0)
    x_padding = 0.05 * (x_max - x_min)
    
    fig.update_layout(
        autosize=True, 
        hovermode='closest', 
        height=height, 
        width=width, 
        font=dict(size=10),
        xaxis=dict(range=[x_min - x_padding, x_max + x_padding], title=dict(text='SHAP Value', font=dict(size=11)), tickfont=dict(size=9)),
        yaxis=dict(
            title=None,
            automargin=True, 
            showticklabels=True, 
            tickvals=list(range(len(top_features_full))), 
            ticktext=simplified_labels,
            tickfont=dict(size=9),
            range=[-0.5, len(top_features_full) - 0.5]
        ),
        margin=dict(l=150, r=40, t=40, b=40),
        title=dict(text='SHAP Beeswarm', font=dict(size=13), x=0.5)
    )
    return fig
    

def shap_dependency_plot(
    shap_values: np.ndarray, 
    feature_values: np.ndarray, 
    feature_names: List[str], 
    feature: str, 
    max_points: int = 1000, 
    interaction_feature: Optional[Union[str, None]] = None, 
    height: int = DEFAULT_HEIGHT, 
    width: int = DEFAULT_WIDTH_RECTANGLE
) -> go.Figure:
    """
    Create a SHAP dependency plot for a single feature with optional interaction coloring.
    """
    logger = get_logger("workflow_16s")
    plot_utils = PlottingUtils(logger)
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
                
                # Handle Categorical Interaction Candidates
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
            
            if best_j is None:
                for j in range(len(feature_names)):
                    if j != idx: best_j = j; break
            
            if best_j is None: interaction_feature = None
            else:
                color_title = feature_names[best_j]; color_title_display = simplify_feature_name(color_title)
                color_data = feature_values[:, best_j]
        else:
            if interaction_feature not in feature_names: raise ValueError(f"Interaction feature '{interaction_feature}' not found")
            color_title = interaction_feature; color_title_display = simplify_feature_name(interaction_feature)
            color_data = feature_values[:, feature_names.index(interaction_feature)]

    if len(x) > max_points:
        indices = np.random.choice(len(x), max_points, replace=False); x = x[indices]; y = y[indices]
        if color_data is not None: color_data = color_data[indices]

    fig = go.Figure()
    marker_config = {'size': 5, 'opacity': 0.5, 'showscale': True}
    hover_template = "Value: %{x:.2f}<br>SHAP: %{y:.2f}"
    
    if color_data is not None:
        try:
            color_data_numeric = color_data.astype(float)
            marker_config.update({'color': color_data_numeric, 'colorscale': 'Viridis', 'colorbar': {'title': {'text': color_title_display[:12], 'side': 'right', 'font': {'size': 10}}, 'tickfont': {'size': 9}}})
        except (ValueError, TypeError):
            unique_cats = np.unique(color_data.astype(str))
            cat_map = {cat: i for i, cat in enumerate(unique_cats)}
            color_data_mapped = np.array([cat_map[str(val)] for val in color_data])
            marker_config.update({'color': color_data_mapped, 'colorscale': 'Turbo', 'colorbar': {'title': {'text': color_title_display[:12], 'side': 'right', 'font': {'size': 10}}, 'tickfont': {'size': 9}}})
    else:
        marker_config.update({'color': y, 'colorscale': 'RdBu', 'colorbar': {'title': {'text': 'SHAP', 'side': 'right', 'font': {'size': 10}}, 'tickfont': {'size': 9}}})
    
    hover_template += "<extra></extra>"
    fig.add_trace(
        go.Scatter(
            x=x, 
            y=y, 
            mode='markers', 
            marker=marker_config, 
            name=feature_display, 
            hovertemplate=hover_template, 
            text=color_data.astype(str) if color_data is not None else None
        )
    )
    
    try:
        x_float = x.astype(float)
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(y, x_float, frac=0.3, it=2)
        fig.add_trace(
            go.Scatter(
                x=smoothed[:, 0], 
                y=smoothed[:, 1], 
                mode='lines', 
                line=dict(color='black', width=3), 
                name='Trend'
            )
        )
    except (ImportError, ValueError, TypeError): pass
    
    try:
        x_float = x.astype(float)
        x_padding = 0.05 * (x_float.max() - x_float.min()); y_padding = 0.05 * (y.max() - y.min())
        xaxis_dict = dict(
            showgrid=False, 
            mirror=True, 
            range=[x_float.min() - x_padding, x_float.max() + x_padding]
        )
    except:
        xaxis_dict = dict(
            showgrid=False, 
            mirror=True
        ) 
        y_padding = 0.05 * (y.max() - y.min())
    
    title_suffix = " (interaction)" if auto_interaction else ""    
    title = f'{feature_display}{title_suffix}'
    
    fig.update_layout(
        autosize=True, 
        height=height, 
        width=width, 
        font=dict(size=10),
        title=dict(text=title, font=dict(size=12), x=0.5),
        xaxis=dict(
            **xaxis_dict,
            title=dict(text='Feature Value', font=dict(size=11)),
            tickfont=dict(size=9)
        ), 
        yaxis=dict(
            title=dict(text='SHAP Value', font=dict(size=11)),
            showgrid=False, 
            mirror=True,
            tickfont=dict(size=9),
            range=[y.min() - y_padding, y.max() + y_padding]
        ),
        coloraxis_colorbar_title_text='',
        margin=dict(l=60, r=80, t=60, b=60),
        hovermode='closest'
    )
    return fig
    

def shap_heatmap(
    shap_values: np.ndarray, 
    feature_values: np.ndarray, 
    feature_names: List[str], 
    max_display: int = 20, 
    max_samples: int = 1000, 
    height: int = DEFAULT_HEIGHT, 
    width: int = DEFAULT_WIDTH_RECTANGLE
) -> go.Figure:
    """Create a SHAP heatmap showing SHAP values across instances and features."""
    logger = get_logger("workflow_16s")
    if len(shap_values) > max_samples:
        sample_idx = np.random.choice(len(shap_values), max_samples, replace=False); shap_values_sampled = shap_values[sample_idx]; feature_values_sampled = feature_values[sample_idx]
    else:
        sample_idx = np.arange(len(shap_values)); shap_values_sampled = shap_values; feature_values_sampled = feature_values

    mean_abs_shap = np.abs(shap_values_sampled).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
    top_indices = [int(i) for i in top_indices]
    
    top_features = [feature_names[i] for i in top_indices]
    simplified_feature_names = generate_unique_simplified_labels(top_features)
    
    corr_matrix = np.corrcoef(shap_values_sampled[:, top_indices].T)
    np.fill_diagonal(corr_matrix, 1.0)
    dist_matrix = 1 - np.abs(corr_matrix)
    dist_matrix = np.nan_to_num(dist_matrix, nan=0.0)
    
    feature_linkage = linkage(dist_matrix, method='complete', optimal_ordering=True)
    feature_order = leaves_list(feature_linkage)
    clustered_feature_names = [simplified_feature_names[i] for i in feature_order]
    
    # Use numpy indexing from the numpy array of top_indices
    clustered_indices = np.array(top_indices)[feature_order]

    instance_linkage = linkage(shap_values_sampled[:, clustered_indices], method='complete')
    instance_order = leaves_list(instance_linkage)

    clustered_shap = shap_values_sampled[instance_order][:, clustered_indices]
    clustered_feature_vals = feature_values_sampled[instance_order][:, clustered_indices]
    
    hover_text = []
    for i, instance_idx in enumerate(instance_order):
        row_text = []
        for j, feat_idx in enumerate(clustered_indices):
            row_text.append(f"SHAP: {clustered_shap[i, j]:.2f}")
        hover_text.append(row_text)

    fig = go.Figure(
        go.Heatmap(
            z=clustered_shap, 
            x=clustered_feature_names, 
            y=[f"S{sample_idx[i]}" for i in instance_order], 
            colorscale='Viridis', 
            zmid=0, 
            hoverinfo="text", 
            text=hover_text, 
            colorbar=dict(title=dict(text='SHAP', font=dict(size=11)), tickfont=dict(size=9))
        )
    )
    
    fig.update_layout(
        autosize=True, 
        hovermode='closest', 
        height=height, 
        width=width,
        font=dict(size=10),
        title=dict(text='SHAP Heatmap (Clustered)', font=dict(size=12), x=0.5),
        xaxis=dict(
            title=dict(text='Features', font=dict(size=11)),
            automargin=True, 
            tickangle=-45,
            tickfont=dict(size=9)
        ), 
        yaxis=dict(
            title=dict(text='Samples', font=dict(size=11)),
            automargin=True,
            tickfont=dict(size=8),
            showticklabels=False
        ),
        margin=dict(l=60, r=80, t=50, b=50)
    )
    return fig


def shap_force_plot(
    base_value: float, 
    shap_values: np.ndarray, 
    feature_values: np.ndarray, 
    feature_names: List[str], 
    instance_index: int = 0, 
    max_display: int = 12, 
    height: int = DEFAULT_HEIGHT, 
    width: int = DEFAULT_WIDTH_RECTANGLE
) -> go.Figure:
    """
    Create a waterfall-style force plot showing feature contributions for a single instance.
    """
    logger = get_logger("workflow_16s")
    instance_shap = shap_values[instance_index]
    instance_feature_values = feature_values[instance_index]
    prediction = base_value + instance_shap.sum()
    
    sorted_indices = np.argsort(np.abs(instance_shap))[::-1]
    sorted_indices = [int(i) for i in sorted_indices]
    
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
    fig.add_trace(
        go.Scatter(
            x=[base_value], 
            y=['Base Value'], 
            mode='markers', 
            marker=dict(size=14, color='#999999'), 
            hoverinfo='text', 
            text=f"Base: {base_value:.3f}"
        )
    )
    
    for i in range(len(top_features)):
        start_val = cumulative_values[i]; end_val = cumulative_values[i+1]; contribution = top_shap[i]
        try: val_str = f"{float(top_values[i]):.2f}"
        except: val_str = str(top_values[i])
        
        # Use simplified feature names
        feat_display = simplify_feature_name(top_features[i])
        
        fig.add_trace(
            go.Bar(
                x=[contribution], 
                y=[feat_display], 
                base=[start_val], 
                orientation='h', 
                marker=dict(color='#ff0d57' if contribution > 0 else '#1e88e5', line=dict(width=0)), 
                hoverinfo='text', 
                text=(f"{feat_display}<br>SHAP: {contribution:.2f}")
            )
        )
    
    if other_contrib != 0:
        start_val = cumulative_values[-2]; end_val = cumulative_values[-1]
        fig.add_trace(
            go.Bar(
                x=[other_contrib], 
                y=[f'+{len(other_idx)} more'], 
                base=[start_val], 
                orientation='h', 
                marker=dict(color='#999999', line=dict(width=0)), 
                hoverinfo='text', 
                text=f"Sum of {len(other_idx)} features: {other_contrib:.2f}"
            )
        )
    
    fig.add_trace(
        go.Scatter(
            x=[prediction], 
            y=['Output'], 
            mode='markers', 
            marker=dict(
                size=14, 
                symbol='diamond', 
                color='#000000'
            ), 
            hoverinfo='text', 
            text=f"Prediction: {prediction:.2f}"
        )
    )
    
    for i in range(len(cumulative_values)-1):
        fig.add_trace(
            go.Scatter(
                x=[cumulative_values[i], cumulative_values[i+1]], 
                y=[y_labels[i], y_labels[i+1]], 
                mode='lines', 
                line=dict(color='#aaaaaa', width=1, dash='dot'), 
                hoverinfo='none', 
                showlegend=False
            )
    )
    
    fig.update_layout(
        barmode='stack', 
        showlegend=False, 
        hovermode='closest', 
        height=35 * len(y_labels) + 100, 
        font=dict(size=10),
        title=dict(text=f'Force Plot - Sample {instance_index}', font=dict(size=12), x=0.5),
        yaxis=dict(
            categoryorder='array', 
            categoryarray=list(reversed(y_labels)),
            tickfont=dict(size=10)
        ),
        xaxis=dict(
            title=dict(text='Model Output', font=dict(size=11)),
            tickfont=dict(size=9)
        ),
        margin=dict(l=80, r=40, t=60, b=40),
        shapes=[dict(
            type='line', 
            x0=base_value, 
            x1=base_value, 
            y0=-1, 
            y1=len(y_labels), 
            line=dict(color='#999999', width=1, dash='dash')
        )]
    )
        
    fig = update_font_sizes(fig)
    fig = apply_common_layout(fig, 'Model Output Value', "", f"SHAP Force Plot - Instance {instance_index}")
    fig.update_layout(autosize=True, width=width, yaxis=dict(automargin=True))
    return fig


def shap_waterfall_plot(
    base_value: float, 
    shap_values: np.ndarray, 
    feature_values: np.ndarray, 
    feature_names: List[str], 
    instance_index: int = 0, 
    max_display: int = 10, 
    height: int = DEFAULT_HEIGHT, 
    width: int = DEFAULT_WIDTH_RECTANGLE
) -> go.Figure:
    """
    Create a waterfall plot showing cumulative feature contributions.
    """
    logger = get_logger("workflow_16s")
    plot_utils = PlottingUtils(logger)
    instance_shap = shap_values[instance_index]
    instance_feature_values = feature_values[instance_index]
    prediction = base_value + np.sum(instance_shap)
    
    sorted_indices = np.argsort(np.abs(instance_shap))[::-1]
    sorted_indices = [int(i) for i in sorted_indices]
    
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
    
    fig = go.Figure(
        go.Waterfall(
            name="", 
            orientation="v", 
            measure=["absolute"] + ["relative"] * (len(steps)-2) + ["total"], 
            x=feature_labels, 
            textposition="outside", 
            text=bar_text, 
            y=[step[1] for step in steps], 
            connector={
                "line":{"color":"rgb(63, 63, 63)"}}, 
            increasing={"marker":{"color":"#1e88e5"}}, 
            decreasing={"marker":{"color":"#ff0d57"}}, 
            totals={"marker":{"color":"#000000"}}, 
            hoverinfo='text', 
            hovertext=hover_text
        )
    )
    fig = update_font_sizes(fig)
    fig = apply_common_layout(fig, "Features", "Model Output Value", "SHAP Waterfall Plot")
    fig.update_layout(height=height, showlegend=False, waterfallgap=0.3)
    return fig
    

def shap_interaction_heatmap(
    shap_interaction_values: np.ndarray, 
    feature_names: List[str], 
    max_display: int = 15, 
    height: int = DEFAULT_HEIGHT, 
    width: int = DEFAULT_WIDTH_SQUARE
) -> go.Figure:
    """
    Create SHAP interaction heatmap showing pairwise feature interactions.
    """
    logger = get_logger("workflow_16s")
    plot_utils = PlottingUtils(logger)
    # Handle 3D interaction values (average across samples)
    if shap_interaction_values.ndim == 3:
        # Mean absolute interaction strength across all samples
        interaction_matrix = np.abs(shap_interaction_values).mean(axis=0)
    else:
        interaction_matrix = shap_interaction_values
    
    # Select top features by total interaction strength
    interaction_strength = interaction_matrix.sum(axis=1)
    top_indices = np.argsort(interaction_strength)[-max_display:][::-1]
    top_indices = [int(i) for i in top_indices]
    
    # Extract submatrix
    submatrix = interaction_matrix[np.ix_(top_indices, top_indices)]
    
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
        colorbar=dict(
            title=dict(
                text='Mean |Interaction|', 
                font=dict(size=14)
            )
        )
    ))
    
    fig = update_font_sizes(fig)
    fig = apply_common_layout(fig, 'feature', 'feature', 'SHAP Feature Interactions')
    fig.update_layout(
        autosize=True,
        height=height,
        width=width,
        xaxis=dict(tickangle=-45, side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    
    return fig


def plot_shap(
    base_value: float, 
    shap_values: np.ndarray, 
    feature_values: np.ndarray, 
    feature_names: list, 
    n_features: int = 20, 
    output_dir: Optional[Union[str, Path]] = None, 
    interaction_feature: Optional[Union[str, None]] = 'auto', 
    show: bool = False, 
    verbose: bool = False, 
    height: int = DEFAULT_HEIGHT, 
    width: int = DEFAULT_WIDTH_RECTANGLE, 
    is_multiclass_avg: bool = False, 
    shap_interaction_values: Optional[np.ndarray] = None
) -> dict:
    """
    Generate both SHAP bar plot, beeswarm plot, dependency plots, and optional interaction plots.
    """
    logger = get_logger("workflow_16s")
    plot_utils = PlottingUtils(logger)
    if output_dir is None:
        can_save = False; output_path_base = None
    else:
        can_save = True; output_path_base = Path(output_dir) / 'figs'
        output_path_base.parent.mkdir(parents=True, exist_ok=True)

    bar_fig, top_full_features = shap_summary_bar(
        shap_values, 
        feature_names, 
        n_features, 
        height=height, 
        width=width
    )
    if can_save and output_path_base:
        save_fig(
            bar_fig, 
            output_path_base / f"shap.summary.bar.{n_features}", 
            formats=['png', 'html', 'json']
        )
        
    if is_multiclass_avg:
        logger.warning("Skipping SHAP beeswarm, heatmap, force, waterfall, and dependency plots for multiclass target.")
        beeswarm_fig = None; heatmap_fig = None; force_fig = None; waterfall_fig = None; dependency_figs = {}; interaction_fig = None
    else:
        beeswarm_fig = shap_beeswarm(
            shap_values, 
            feature_values, 
            feature_names, 
            n_features, 
            height=height, 
            width=width
        )
        if can_save and output_path_base:
            save_fig(
                fig=beeswarm_fig, 
                output_path=output_path_base / f"shap.summary.beeswarm.{n_features}", 
                formats=['png', 'html', 'json']
            )
            
        heatmap_fig = shap_heatmap(
            shap_values, 
            feature_values, 
            feature_names, 
            max_display=n_features, 
            max_samples=1000, 
            height=height, 
            width=width
        )
        if can_save and output_path_base:
            save_fig(
                fig=heatmap_fig, 
                output_path=output_path_base / f"shap.summary.heatmap.{n_features}", 
                formats=['png', 'html', 'json']
            )
            
        force_fig = shap_force_plot(
            base_value, 
            shap_values, 
            feature_values, 
            feature_names, 
            instance_index=0, 
            max_display=12, 
            height=height, 
            width=width
        )
        if can_save and output_path_base:
            save_fig(
                fig=force_fig, 
                output_path=output_path_base / f"shap.summary.force.{n_features}", 
                formats=['png', 'html', 'json']
            )
            
        waterfall_fig = shap_waterfall_plot(
            base_value, 
            shap_values, 
            feature_values, 
            feature_names, 
            instance_index=0, 
            max_display=10, 
            height=height, 
            width=width
        )
        if can_save and output_path_base:
            save_fig(
                fig=waterfall_fig, 
                output_path=output_path_base / f"shap.summary.waterfall.{n_features}", 
                formats=['png', 'html', 'json']
            )
            
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
                    save_fig(
                        fig=interaction_fig, 
                        output_path=output_path_base / f"shap.interactions.heatmap.{n_features}", 
                        formats=['png', 'html', 'json']
                    )
            except Exception as e:
                logger.error(f"Error creating SHAP interaction heatmap: {str(e)}")
            
        dependency_figs = {}
        if n_features > 0:
            for feature in top_full_features[:n_features]:
                try:
                    dep_fig = shap_dependency_plot(
                        shap_values, feature_values, feature_names, feature, 10000, 
                        interaction_feature='auto', height=height, width=width
                    )
                    if can_save and output_path_base:
                        save_fig(fig=dep_fig, output_path=output_path_base / f"shap.dependency.{feature}", formats=['png', 'html', 'json'])
                    dependency_figs[feature] = dep_fig
                except Exception as e:
                    logger.error(f"Error creating dependency plot for {feature}: {str(e)}")

    return {
        'bar_fig': bar_fig, 
        'beeswarm_fig': beeswarm_fig, 
        'heatmap_fig': heatmap_fig,
        'force_fig': force_fig, 
        'waterfall_fig': waterfall_fig, 
        'dependency_figs': dependency_figs, 
        'interaction_fig': interaction_fig
    }