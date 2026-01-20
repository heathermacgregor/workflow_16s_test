# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third Party Imports
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns

# Local Imports
from workflow_16s import constants
from workflow_16s.figures.figures import (
    attach_legend_to_figure, largecolorset, plot_legend, plotly_show_and_save    
)
from workflow_16s.constants import DEFAULT_COLOR_COL, DEFAULT_SYMBOL_COL
from workflow_16s.figures.tools import PlotlyScatterPlot, fig_to_html, fig_to_json

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')
sns.set_style('whitegrid')  # Set seaborn style globally

# ================================= GLOBAL VARIABLES ================================= #

# ================================== PLOT FUNCTIONS ================================== #

def volcano_plot(
    results_df: pd.DataFrame,
    pvalue_threshold: float = 0.05,
    effect_size_col: str = 'log2_fold_change',
    pvalue_col: str = 'pvalue',
    output_dir: Union[Path, None] = None,
    title: str = "Volcano Plot - Differential Abundance"
) -> go.Figure:
    """Generate a volcano plot for differential abundance results.
    
    Args:
        results_df: DataFrame with statistical test results
        pvalue_threshold: Significance threshold for p-values
        effect_size_col: Column name for effect size values
        pvalue_col: Column name for p-values
        output_dir: Directory to save outputs
        title: Plot title
        
    Returns:
        Figure object
    """
    # Create a copy to avoid modifying the original
    plot_df = results_df.copy()
    
    # Calculate -log10(p-value)
    plot_df['neg_log10_pvalue'] = -np.log10(plot_df[pvalue_col])
    
    # Determine significance
    plot_df['significant'] = plot_df[pvalue_col] < pvalue_threshold
    
    # Create the plot
    fig = px.scatter(
        plot_df,
        x=effect_size_col,
        y='neg_log10_pvalue',
        color='significant',
        hover_data=plot_df.columns.tolist(),
        title=title,
        labels={
            effect_size_col: 'Effect Size (Log2 Fold Change)',
            'neg_log10_pvalue': '-Log10(P-value)'
        }
    )
    
    # Add significance threshold line
    threshold_y = -np.log10(pvalue_threshold)
    fig.add_hline(
        y=threshold_y, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"P-value = {pvalue_threshold}"
    )
    
    # Customize layout
    fig.update_layout(
        legend_title_text='Significant',
        xaxis_title="Effect Size (Log2 Fold Change)",
        yaxis_title="-Log10(P-value)"
    )
    
    # Save if output directory provided
    if output_dir:
        output_path = output_dir / "volcano_plot"
        fig_to_json(fig, output_path)
      
    return fig


def core_microbiome_barplot(
    core_results: Dict[str, pd.DataFrame],
    output_dir: Union[Path, None] = None,
    title: str = "Core Microbiome Composition"
) -> go.Figure:
    """Generate a stacked bar chart for core microbiome results.
    
    Args:
        core_results: Dictionary with group names as keys and DataFrames as values
        output_dir: Directory to save outputs
        title: Plot title
        
    Returns:
        Figure object
    """
    # Prepare data for plotting
    plot_data = []
    for group, df in core_results.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            # Get top features for this group
            top_features = df.head(10)  # Show top 10 features per group
            for _, row in top_features.iterrows():
                plot_data.append({
                    'Group': group,
                    'Feature': row.get('feature', 'Unknown'),
                    'Abundance': row.get('abundance', 0),
                    'Prevalence': row.get('prevalence', 0)
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    if plot_df.empty:
        logger.warning("No core microbiome data to plot")
        return go.Figure()
    
    # Create the plot
    fig = px.bar(
        plot_df,
        x='Group',
        y='Abundance',
        color='Feature',
        title=title,
        hover_data=['Prevalence'],
        barmode='stack'
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title="Sample Groups",
        yaxis_title="Relative Abundance (%)",
        legend_title="Microbial Features"
    )
    
    # Save if output directory provided
    if output_dir:
        output_path = output_dir / "core_microbiome_barplot"
        fig_to_html(fig, output_path)
        fig_to_json(fig, output_path)
    
    return fig


def network_plot(
    edges_df: pd.DataFrame,
    network_stats: Dict[str, Any],
    output_dir: Union[Path, None] = None,
    title: str = "Microbial Co-occurrence Network"
) -> go.Figure:
    """Generate a network visualization for microbial correlations.
    
    Args:
        edges_df: DataFrame with network edges (source, target, correlation)
        network_stats: Dictionary with network statistics
        output_dir: Directory to save outputs
        title: Plot title
        
    Returns:
        Figure object
    """
    if edges_df.empty:
        logger.warning("No network edges to plot")
        return go.Figure()
    
    # Create node list from edges
    all_nodes = list(set(edges_df['source'].tolist() + edges_df['target'].tolist()))
    node_trace = go.Scatter(
        x=[], y=[], 
        text=all_nodes,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            size=10,
            color='lightblue'
        )
    )
    
    # Create edge traces
    edge_traces = []
    for _, row in edges_df.iterrows():
        # This is a simplified representation
        # In a real implementation, you would use a proper layout algorithm
        edge_trace = go.Scatter(
            x=[row['source'], row['target']],
            y=[1, 1],  # Placeholder y-values
            mode='lines',
            line=dict(
                width=abs(row['correlation']) * 5,
                color='red' if row['correlation'] > 0 else 'blue'
            ),
            hoverinfo='text',
            text=f"Correlation: {row['correlation']:.3f}"
        )
        edge_traces.append(edge_trace)
    
    # Create figure
    fig = go.Figure(data=[node_trace] + edge_traces)
    
    # Add title and annotations
    fig.update_layout(
        title=title,
        showlegend=False,
        annotations=[
            dict(
                text=f"Nodes: {network_stats.get('unique_nodes', 0)}<br>"
                     f"Edges: {network_stats.get('total_edges', 0)}<br>"
                     f"Positive: {network_stats.get('positive_edges', 0)}<br>"
                     f"Negative: {network_stats.get('negative_edges', 0)}",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.05, y=0.95,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
        ]
    )
    
    # Save if output directory provided
    if output_dir:
        output_path = output_dir / "network_plot"
        fig_to_html(fig, output_path)
        fig_to_json(fig, output_path)
    
    return fig


def correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    output_dir: Union[Path, None] = None,
    title: str = "Feature-Environment Correlation Heatmap"
) -> go.Figure:
    """Generate a heatmap for correlation results.
    
    Args:
        correlation_matrix: DataFrame with correlation values
        output_dir: Directory to save outputs
        title: Plot title
        
    Returns:
        Figure object
    """
    # Create the heatmap
    fig = px.imshow(
        correlation_matrix,
        aspect='auto',
        color_continuous_scale='RdBu_r',
        title=title,
        labels=dict(color="Correlation")
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title="Environmental Variables",
        yaxis_title="Microbial Features"
    )
    
    # Save if output directory provided
    if output_dir:
        output_path = output_dir / "correlation_heatmap"
        fig_to_html(fig, output_path)
        fig_to_json(fig, output_path)
    
    return fig


def statistical_results_table(
    results_df: pd.DataFrame,
    output_dir: Union[Path, None] = None,
    title: str = "Statistical Results Summary"
) -> go.Figure:
    """Generate an interactive table for statistical results.
    
    Args:
        results_df: DataFrame with statistical test results
        output_dir: Directory to save outputs
        title: Table title
        
    Returns:
        Figure object with table
    """
    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=results_df.columns.tolist(),
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[results_df[col] for col in results_df.columns],
            fill_color='lavender',
            align='left'
        )
    )])
    
    # Add title
    fig.update_layout(title=title)
    
    # Save if output directory provided
    if output_dir:
        output_path = output_dir / "results_table"
        fig_to_html(fig, output_path)
        fig_to_json(fig, output_path)
    
    return fig


def create_statistical_summary_dashboard(
    statistical_results: Dict[str, Any],
    output_dir: Union[Path, None] = None
) -> go.Figure:
    """Create a comprehensive dashboard of statistical results.
    
    Args:
        statistical_results: Dictionary containing all statistical results
        output_dir: Directory to save outputs
        
    Returns:
        Figure object with dashboard
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Volcano Plot", 
            "Core Microbiome", 
            "Correlation Heatmap", 
            "Network Statistics"
        )
    )
    
    # Add volcano plot if available
    if 'differential_abundance' in statistical_results:
        volcano_df = statistical_results['differential_abundance']
        volcano_fig = volcano_plot(volcano_df, title="")
        for trace in volcano_fig.data:
            fig.add_trace(trace, row=1, col=1)
    
    # Add core microbiome plot if available
    if 'core_microbiome' in statistical_results:
        core_data = statistical_results['core_microbiome']
        core_fig = core_microbiome_barplot(core_data, title="")
        for trace in core_fig.data:
            fig.add_trace(trace, row=1, col=2)
    
    # Add correlation heatmap if available
    if 'correlations' in statistical_results:
        corr_matrix = statistical_results['correlations']
        heatmap_fig = correlation_heatmap(corr_matrix, title="")
        for trace in heatmap_fig.data:
            fig.add_trace(trace, row=2, col=1)
    
    # Add network statistics if available
    if 'networks' in statistical_results:
        network_stats = statistical_results['networks'].get('network_stats', {})
        stats_df = pd.DataFrame([network_stats])
        table_fig = statistical_results_table(stats_df, title="")
        for trace in table_fig.data:
            fig.add_trace(trace, row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        title_text="Statistical Analysis Dashboard",
        showlegend=False
    )
    
    # Save if output directory provided
    if output_dir:
        output_path = output_dir / "statistical_dashboard"
        fig_to_html(fig, output_path)
        fig_to_json(fig, output_path)
    
    return fig
  
