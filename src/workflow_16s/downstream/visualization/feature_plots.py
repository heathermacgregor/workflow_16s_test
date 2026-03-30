"""Feature relationship and importance visualizations.

Generates:
- Feature correlation heatmaps
- Feature vs Feature scatter/heatmaps
- Distribution plots for top features
- Variance explained plots
"""

import logging
from pathlib import Path
from typing import Optional, List, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.cluster.hierarchy as scipy_cluster
from scipy.spatial.distance import squareform
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger('workflow_16s')


def plot_feature_correlation_heatmap(
    X_features: pd.DataFrame,
    top_n_features: int = 30,
    output_path: Optional[Path] = None,
    height: int = 900,
    width: int = 1000,
    cluster: bool = True,
) -> go.Figure:
    """
    Plot correlation heatmap of top features with optional clustering.
    
    Parameters
    ----------
    X_features : pd.DataFrame
        Feature matrix.
    top_n_features : int
        Number of most variant features to show.
    output_path : Path, optional
        Save HTML.
    height : int
        Figure height.
    width : int
        Figure width.
    cluster : bool
        Apply hierarchical clustering to reorder features.
    
    Returns
    -------
    go.Figure
        Plotly heatmap.
    """
    # Select top variance features
    feature_var = X_features.var(axis=0).nlargest(top_n_features)
    X_subset = X_features[feature_var.index]
    
    # Compute correlation
    corr_matrix = X_subset.corr()
    
    # Hierarchical clustering
    if cluster and len(corr_matrix) > 1:
        try:
            # Convert correlation to distance
            dist_matrix = 1 - corr_matrix.values
            dist_matrix[dist_matrix < 0] = 0  # Ensure non-negative
            condensed_dist = squareform(dist_matrix)
            
            # Hierarchical clustering
            z = scipy_cluster.linkage(condensed_dist, method='average')
            dendro = scipy_cluster.dendrogram(z, no_plot=True)
            
            # Reorder based on clustering
            feature_order = [corr_matrix.index[i] for i in dendro['leaves']]
            corr_matrix = corr_matrix.loc[feature_order, feature_order]
        except Exception as e:
            logger.warning(f"Could not cluster features: {e}")
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
    ))
    
    fig.update_layout(
        title=f"Feature Correlation Heatmap (Top {top_n_features} by Variance)",
        height=height,
        width=width,
        template='plotly_white',
    )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Feature correlation heatmap saved: {output_path}")
    
    return fig


def plot_feature_distributions(
    X_features: pd.DataFrame,
    top_n_features: int = 8,
    output_path: Optional[Path] = None,
    height: int = 1000,
    width: int = 1200,
) -> go.Figure:
    """
    Plot distributions of top features as histograms/violin plots.
    
    Parameters
    ----------
    X_features : pd.DataFrame
        Feature matrix.
    top_n_features : int
        Number of most variant features to show.
    output_path : Path, optional
        Save HTML.
    height : int
        Figure height.
    width : int
        Figure width.
    
    Returns
    -------
    go.Figure
        Plotly subplots with histograms.
    """
    # Select top variance features
    feature_var = X_features.var(axis=0).nlargest(top_n_features)
    feature_list = feature_var.index.tolist()
    
    n_features = len(feature_list)
    n_rows = (n_features + 2) // 3
    
    fig = make_subplots(
        rows=n_rows,
        cols=3,
        subplot_titles=[f"{f} (var={feature_var[f]:.2e})" for f in feature_list],
    )
    
    for idx, feature in enumerate(feature_list):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        try:
            feature_data = pd.to_numeric(X_features[feature], errors='coerce').dropna()
            
            fig.add_trace(
                go.Histogram(
                    x=feature_data,
                    nbinsx=30,
                    name=feature,
                    showlegend=False,
                    marker_color='steelblue',
                ),
                row=row,
                col=col,
            )
            
            fig.update_xaxes(title_text=feature, row=row, col=col)
            fig.update_yaxes(title_text="Count", row=row, col=col)
        
        except Exception as e:
            logger.warning(f"Could not plot distribution for '{feature}': {e}")
    
    fig.update_layout(
        title_text=f"Top {top_n_features} Feature Distributions",
        height=height,
        width=width,
        showlegend=False,
        template='plotly_white',
    )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Feature distributions saved: {output_path}")
    
    return fig


def plot_feature_pairs_scatter(
    X_features: pd.DataFrame,
    feature_pairs: Optional[List[tuple]] = None,
    max_pairs: int = 6,
    output_path: Optional[Path] = None,
    height: int = 1000,
    width: int = 1200,
    sample_size: Optional[int] = None,
) -> go.Figure:
    """
    Plot scatter plots for top feature pairs (highest correlations).
    
    Parameters
    ----------
    X_features : pd.DataFrame
        Feature matrix.
    feature_pairs : List[tuple], optional
        Specific pairs to plot. If None, uses top correlated pairs.
    max_pairs : int
        Number of feature pairs to show.
    output_path : Path, optional
        Save HTML.
    height : int
        Figure height.
    width : int
        Figure width.
    sample_size : int, optional
        If provided, sample this many points from each pair.
    
    Returns
    -------
    go.Figure
        Plotly subplots with scatter plots.
    """
    # Find top correlated pairs if not provided
    if feature_pairs is None:
        corr_matrix = X_features.corr()
        
        # Get upper triangle indices
        pairs_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                feat1 = corr_matrix.columns[i]
                feat2 = corr_matrix.columns[j]
                corr_val = abs(corr_matrix.iloc[i, j])
                
                if corr_val > 0.001:  # Skip near-zero correlations
                    pairs_corr.append((feat1, feat2, corr_val))
        
        # Sort by correlation and take top pairs
        pairs_corr = sorted(pairs_corr, key=lambda x: x[2], reverse=True)[:max_pairs]
        feature_pairs = [(p[0], p[1]) for p in pairs_corr]
    else:
        feature_pairs = feature_pairs[:max_pairs]
    
    if not feature_pairs:
        logger.warning("No feature pairs to plot")
        return None
    
    n_pairs = len(feature_pairs)
    n_rows = (n_pairs + 2) // 3
    
    fig = make_subplots(
        rows=n_rows,
        cols=3,
        subplot_titles=[f"{p[0]} vs {p[1]}" for p in feature_pairs],
    )
    
    for idx, (feat1, feat2) in enumerate(feature_pairs):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        try:
            x_data = pd.to_numeric(X_features[feat1], errors='coerce')
            y_data = pd.to_numeric(X_features[feat2], errors='coerce')
            
            # Sample if needed
            valid_mask = x_data.notna() & y_data.notna()
            if sample_size and valid_mask.sum() > sample_size:
                sample_idx = np.random.choice(np.where(valid_mask)[0], sample_size, replace=False)
                x_data = x_data[sample_idx]
                y_data = y_data[sample_idx]
            
            corr = np.corrcoef(x_data.dropna(), y_data[x_data.notna()])[0, 1] if x_data.notna().sum() > 1 else np.nan
            
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='markers',
                    marker=dict(size=4, opacity=0.5, color='steelblue'),
                    text=f"{feat1}: {{x}}<br>{feat2}: {{y}}",
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            
            fig.update_xaxes(title_text=feat1, row=row, col=col)
            fig.update_yaxes(title_text=feat2, row=row, col=col)
        
        except Exception as e:
            logger.warning(f"Could not plot pair ({feat1}, {feat2}): {e}")
    
    fig.update_layout(
        title_text="Top Feature Pairs (by Correlation)",
        height=height,
        width=width,
        showlegend=False,
        template='plotly_white',
    )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Feature pairs scatter plot saved: {output_path}")
    
    return fig


def plot_cumulative_variance(
    X_features: pd.DataFrame,
    max_features: int = 50,
    output_path: Optional[Path] = None,
    height: int = 600,
    width: int = 1000,
) -> go.Figure:
    """
    Plot cumulative variance explained by features (sorted by variance).
    
    Parameters
    ----------
    X_features : pd.DataFrame
        Feature matrix.
    max_features : int
        Maximum number of features to consider.
    output_path : Path, optional
        Save HTML.
    height : int
        Figure height.
    width : int
        Figure width.
    
    Returns
    -------
    go.Figure
        Plotly line plot.
    """
    # Compute variance for each feature
    variances = X_features.var(axis=0).sort_values(ascending=False).iloc[:max_features]
    
    # Cumulative variance
    cumsum_var = variances.cumsum()
    total_var = cumsum_var.iloc[-1]
    cumsum_pct = (cumsum_var / total_var * 100).values
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=np.arange(1, len(cumsum_pct) + 1),
        y=cumsum_pct,
        mode='lines+markers',
        name='Cumulative Variance',
        line=dict(color='darkblue', width=3),
        marker=dict(size=6),
        fill='tozeroy',
    ))
    
    # Add 95% threshold line
    fig.add_hline(y=95, line_dash="dash", line_color="red", 
                  annotation_text="95%", annotation_position="right")
    
    fig.update_layout(
        title="Cumulative Variance Explained by Features",
        xaxis_title="Number of Features (ranked by variance)",
        yaxis_title="Cumulative Variance (%)",
        height=height,
        width=width,
        template='plotly_white',
        hovermode='closest',
    )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Cumulative variance plot saved: {output_path}")
    
    return fig


def plot_top_features_boxplot(
    X_features: pd.DataFrame,
    group_by: Optional[pd.Series] = None,
    top_n_features: int = 10,
    output_path: Optional[Path] = None,
    height: int = 800,
    width: int = 1200,
) -> go.Figure:
    """
    Plot boxplots for top features, optionally grouped by a categorical variable.
    
    Parameters
    ----------
    X_features : pd.DataFrame
        Feature matrix.
    group_by : pd.Series, optional
        Categorical variable for grouping.
    top_n_features : int
        Number of top variance features to show.
    output_path : Path, optional
        Save HTML.
    height : int
        Figure height.
    width : int
        Figure width.
    
    Returns
    -------
    go.Figure
        Plotly subplots with boxplots.
    """
    # Select top features
    top_features = X_features.var(axis=0).nlargest(top_n_features).index.tolist()
    
    if group_by is None:
        # Single column per feature
        n_features = len(top_features)
        n_rows = (n_features + 2) // 3
        
        fig = make_subplots(
            rows=n_rows,
            cols=3,
            subplot_titles=top_features,
        )
        
        for idx, feature in enumerate(top_features):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            try:
                feature_data = pd.to_numeric(X_features[feature], errors='coerce').dropna()
                
                fig.add_trace(
                    go.Box(y=feature_data, name=feature, showlegend=False),
                    row=row,
                    col=col,
                )
                
                fig.update_yaxes(title_text=feature, row=row, col=col)
            
            except Exception as e:
                logger.warning(f"Could not plot '{feature}': {e}")
        
        title = f"Top {top_n_features} Features - Boxplots"
    
    else:
        # Grouped boxplots
        n_features = len(top_features)
        n_rows = (n_features + 2) // 3
        
        fig = make_subplots(
            rows=n_rows,
            cols=3,
            subplot_titles=top_features,
        )
        
        for idx, feature in enumerate(top_features):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            try:
                for group in group_by.unique():
                    mask = group_by == group
                    feature_data = pd.to_numeric(X_features.loc[mask, feature], 
                                                 errors='coerce').dropna()
                    
                    fig.add_trace(
                        go.Box(y=feature_data, name=str(group), showlegend=(idx == 0)),
                        row=row,
                        col=col,
                    )
                
                fig.update_yaxes(title_text=feature, row=row, col=col)
            
            except Exception as e:
                logger.warning(f"Could not plot '{feature}': {e}")
        
        title = f"Top {top_n_features} Features - Grouped by {group_by.name}"
    
    fig.update_layout(
        title_text=title,
        height=height,
        width=width,
        template='plotly_white',
    )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Top features boxplot saved: {output_path}")
    
    return fig

def plot_feature_importance_bars(
    importance_df: pd.DataFrame,
    feature_col: str = 'feature',
    importance_col: str = 'importance',
    top_n_features: int = 30,
    output_path: Optional[Path] = None,
    height: int = 800,
    width: int = 1000,
) -> go.Figure:
    """
    Plot a horizontal bar chart of the top feature importances.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame containing feature names and their importance scores.
    feature_col : str
        Column name containing the feature names.
    importance_col : str
        Column name containing the importance scores.
    top_n_features : int
        Number of top features to display.
    output_path : Path, optional
        Save HTML.
    height : int
        Figure height.
    width : int
        Figure width.
        
    Returns
    -------
    go.Figure
        Plotly horizontal bar chart.
    """
    # Handle case where features might be the index
    if feature_col not in importance_df.columns:
        if importance_col in importance_df.columns:
            importance_df = importance_df.reset_index().rename(columns={'index': feature_col, 'Feature': feature_col})
        else:
            logger.error(f"Could not find required columns in importance_df. Available: {importance_df.columns}")
            return go.Figure()

    # Sort values and grab the top N
    df_sorted = importance_df.sort_values(by=importance_col, ascending=True).tail(top_n_features)

    fig = px.bar(
        df_sorted,
        x=importance_col,
        y=feature_col,
        orientation='h',
        color=importance_col,
        color_continuous_scale='Blues',
        title=f"Top {top_n_features} Feature Importances"
    )

    fig.update_layout(
        height=height,
        width=width,
        template='plotly_white',
        yaxis={'categoryorder': 'total ascending'},
        coloraxis_showscale=False,  # Hide color bar to save space
        xaxis_title="Importance Score",
        yaxis_title="Feature"
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Feature importance bar plot saved: {output_path}")

    return fig
