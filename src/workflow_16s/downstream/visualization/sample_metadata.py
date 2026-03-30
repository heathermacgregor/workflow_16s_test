"""
Sample Metadata Visualization Module

Creates publication-ready visualizations of sample-level metadata
with geographic, temporal, and categorical information.

Features:
- Geographic distribution maps (requires folium)
- Categorical variable distributions
- Numeric variable distributions
- Metadata correlation heatmaps
- Sample attribute summaries
- Custom heather template styling

Example:
    >>> from workflow_16s.downstream.visualization.sample_metadata import (
    ...     plot_sample_distribution,
    ...     plot_metadata_heatmap,
    ...     create_geographic_map
    ... )
    >>> 
    >>> # Plot categorical variables
    >>> fig = plot_sample_distribution(
    ...     adata,
    ...     categorical_cols=['env_category_type', 'project_accession'],
    ...     output_path='sample_distribution.html'
    ... )
    >>> fig.show()
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

logger = logging.getLogger('workflow_16s')

# Ensure heather template is available
try:
    if 'heather' not in pio.templates:
        heather = go.layout.Template(
            layout=go.Layout(
                colorway=['#8B7BA4', '#A89CC4', '#C4B8D1', '#6B6B7A', '#9A8E99', '#7A9E8F', '#B8A89C'],
                paper_bgcolor='#FAFAF9',
                plot_bgcolor='#FFFFFF',
                font=dict(family="Arial, sans-serif", size=12, color='#3D3D3D'),
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#E8E8E6', showline=True, linewidth=1),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#E8E8E6', showline=True, linewidth=1),
            )
        )
        pio.templates['heather'] = heather
except Exception as e:
    logger.warning(f"Could not register heather template: {e}")


def plot_sample_distribution(
    adata: ad.AnnData,
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> go.Figure:
    """
    Create dashboard showing distribution of sample metadata.
    
    Generates subplots for each categorical and numeric variable,
    displaying how samples are distributed across metadata categories.
    
    Args:
        adata: AnnData object with metadata in .obs
        categorical_cols: List of categorical column names to plot
                         If None, auto-detect from dtype
        numeric_cols: List of numeric column names to plot
                     If None, auto-detect from dtype
        output_path: Optional path to save HTML output
        
    Returns:
        Plotly figure with multiple subplots
        
    Example:
        >>> fig = plot_sample_distribution(
        ...     adata,
        ...     categorical_cols=['env_category_type', 'project_accession']
        ... )
        >>> fig.show()
    """
    # Auto-detect column types if not provided
    if categorical_cols is None:
        categorical_cols = [
            col for col in adata.obs.columns 
            if adata.obs[col].dtype == 'object' and col not in ['sample_id', 'feature_id']
        ]
    
    if numeric_cols is None:
        numeric_cols = [
            col for col in adata.obs.columns 
            if pd.api.types.is_numeric_dtype(adata.obs[col]) 
            and not col.endswith('_id')
        ]
    
    # Limit to available columns
    categorical_cols = [col for col in categorical_cols if col in adata.obs.columns]
    numeric_cols = [col for col in numeric_cols if col in adata.obs.columns]
    
    n_cat = len(categorical_cols)
    n_num = len(numeric_cols)
    n_plots = n_cat + n_num
    
    if n_plots == 0:
        logger.warning("No categorical or numeric columns found")
        return None
    
    # Create subplots
    n_cols = min(3, max(1, n_plots))
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"<b>{col}</b>" for col in categorical_cols + numeric_cols],
        specs=[[{"type": "bar"} if i % 2 == 0 else {"type": "histogram"} 
                for i in range(n_cols)] 
               for _ in range(n_rows)],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )
    
    # Color palette for categorical data
    heather_colors = [
        '#8B7BA4', '#A89CC4', '#C4B8D1', '#6B6B7A', '#9A8E99',
        '#7A9E8F', '#B8A89C', '#C9A876', '#A85A5A'
    ]
    
    plot_idx = 0
    
    # Add categorical plots
    for col in categorical_cols:
        row = plot_idx // n_cols + 1
        col_pos = plot_idx % n_cols + 1
        
        value_counts = adata.obs[col].value_counts()
        
        fig.add_trace(
            go.Bar(
                x=value_counts.index.tolist(),
                y=value_counts.values.tolist(),
                marker=dict(
                    color='#8B7BA4',
                    line=dict(color='#6B6B7A', width=1.5)
                ),
                text=value_counts.values,
                textposition='outside',
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',
                name=col
            ),
            row=row, col=col_pos
        )
        
        fig.update_yaxes(title_text="<b>Count</b>", row=row, col=col_pos)
        plot_idx += 1
    
    # Add numeric plots
    for col in numeric_cols:
        row = plot_idx // n_cols + 1
        col_pos = plot_idx % n_cols + 1
        
        fig.add_trace(
            go.Histogram(
                x=adata.obs[col].dropna(),
                marker=dict(
                    color='#A89CC4',
                    line=dict(color='#6B6B7A', width=0.5)
                ),
                nbinsx=30,
                showlegend=False,
                hovertemplate='Value: %{x:.2f}<br>Count: %{y}<extra></extra>',
                name=col
            ),
            row=row, col=col_pos
        )
        
        fig.update_yaxes(title_text="<b>Frequency</b>", row=row, col=col_pos)
        plot_idx += 1
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>Sample Metadata Distribution</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#2D2D2D')
        ),
        height=max(400, 400 * n_rows),
        width=1400,
        showlegend=False,
        template='heather',
        hovermode='closest',
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Sample distribution plot saved: {output_path}")
    
    return fig


def plot_metadata_heatmap(
    adata: ad.AnnData,
    numeric_cols: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> go.Figure:
    """
    Create correlation heatmap of numeric metadata variables.
    
    Shows correlations between numeric sample attributes (e.g., read depth,
    latitude, longitude, diversity metrics).
    
    Args:
        adata: AnnData object with metadata in .obs
        numeric_cols: List of numeric columns to include
                     If None, auto-detect
        output_path: Optional path to save HTML output
        
    Returns:
        Plotly heatmap figure
        
    Example:
        >>> fig = plot_metadata_heatmap(
        ...     adata,
        ...     numeric_cols=['read_depth', 'latitude', 'longitude', 'shannon']
        ... )
        >>> fig.show()
    """
    # Auto-detect numeric columns
    if numeric_cols is None:
        numeric_cols = [
            col for col in adata.obs.columns 
            if pd.api.types.is_numeric_dtype(adata.obs[col]) 
            and not col.endswith('_id')
        ]
    
    numeric_cols = [col for col in numeric_cols if col in adata.obs.columns]
    
    if len(numeric_cols) < 2:
        logger.warning(f"Need at least 2 numeric columns, found {len(numeric_cols)}")
        return None
    
    # Calculate correlation matrix
    corr_matrix = adata.obs[numeric_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=[
            [0.0, '#A85A5A'],    # Negative: soft red
            [0.5, '#FAFAF9'],    # Zero: off-white
            [1.0, '#7A9E8F']     # Positive: sage green
        ],
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(
            title="<b>Correlation</b>",
            thickness=15,
            len=0.7
        ),
        hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Sample Metadata Correlation Heatmap</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#2D2D2D')
        ),
        xaxis_title="",
        yaxis_title="",
        height=600,
        width=800,
        template='heather',
        hovermode='closest',
        margin=dict(l=120, r=100, t=100, b=120)
    )
    
    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Metadata correlation heatmap saved: {output_path}")
    
    return fig


def plot_metadata_summary_table(
    adata: ad.AnnData,
    output_path: Optional[Union[str, Path]] = None
) -> go.Figure:
    """
    Create summary table of metadata statistics.
    
    Generates a table with descriptive statistics for all metadata variables.
    
    Args:
        adata: AnnData object with metadata in .obs
        output_path: Optional path to save HTML output
        
    Returns:
        Plotly table figure
        
    Example:
        >>> fig = plot_metadata_summary_table(adata)
        >>> fig.show()
    """
    summary_data = []
    
    for col in adata.obs.columns:
        if col in ['sample_id', 'feature_id']:
            continue
        
        series = adata.obs[col]
        
        if pd.api.types.is_numeric_dtype(series):
            summary_data.append({
                'Column': col,
                'Type': 'Numeric',
                'Mean': f"{series.mean():.2f}",
                'Std': f"{series.std():.2f}",
                'Min': f"{series.min():.2f}",
                'Max': f"{series.max():.2f}",
                'Missing': series.isna().sum()
            })
        else:
            summary_data.append({
                'Column': col,
                'Type': 'Categorical',
                'N_Unique': series.nunique(),
                'Mode': series.mode().values[0] if len(series.mode()) > 0 else 'N/A',
                'Most_Common_%': f"{series.value_counts().iloc[0] / len(series) * 100:.1f}%",
                'Missing': series.isna().sum()
            })
    
    # Convert to DataFrame for easier table creation
    summary_df = pd.DataFrame(summary_data)
    
    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{col}</b>" for col in summary_df.columns],
            fill_color='#8B7BA4',
            align='left',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[summary_df[col].tolist() for col in summary_df.columns],
            fill_color='#FAFAF9',
            align='left',
            font=dict(color='#3D3D3D', size=11),
            line=dict(color='#E8E8E6', width=1)
        )
    )])
    
    fig.update_layout(
        title=dict(
            text="<b>Sample Metadata Summary</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#2D2D2D')
        ),
        height=400 + len(summary_df) * 30,
        width=1200,
        margin=dict(l=40, r=40, t=100, b=40)
    )
    
    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Metadata summary table saved: {output_path}")
    
    return fig


def create_geographic_map(
    adata: ad.AnnData,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    color_by: Optional[str] = None,
    marker_size: int = 8,
    output_path: Optional[Union[str, Path]] = None
) -> go.Figure:
    """
    Create interactive geographic map of samples.
    
    Requires valid latitude/longitude columns in metadata.
    
    Args:
        adata: AnnData object with geographic coordinates
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        color_by: Optional column name to color samples by
        marker_size: Size of point markers (default: 8)
        output_path: Optional path to save HTML output
        
    Returns:
        Plotly map figure
        
    Example:
        >>> fig = create_geographic_map(
        ...     adata,
        ...     color_by='env_category_type'
        ... )
        >>> fig.show()
    """
    # Check for required columns
    if lat_col not in adata.obs.columns or lon_col not in adata.obs.columns:
        logger.warning(f"Cannot find {lat_col} or {lon_col} columns")
        return None
    
    # Prepare data
    map_data = adata.obs[[lat_col, lon_col]].copy()
    if color_by and color_by in adata.obs.columns:
        map_data['color_by'] = adata.obs[color_by]
    
    # Remove rows with missing coordinates
    map_data = map_data.dropna(subset=[lat_col, lon_col])
    
    if len(map_data) == 0:
        logger.warning("No valid coordinate pairs found")
        return None
    
    # Create map figure
    if color_by and 'color_by' in map_data.columns:
        fig = px.scatter_geo(
            map_data,
            lat=lat_col,
            lon=lon_col,
            color='color_by',
            hover_name=map_data.index,
            color_discrete_sequence=['#8B7BA4', '#A89CC4', '#C4B8D1', '#7A9E8F', '#C9A876', '#A85A5A'],
            size_max=marker_size
        )
    else:
        fig = px.scatter_geo(
            map_data,
            lat=lat_col,
            lon=lon_col,
            hover_name=map_data.index
        )
    
    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor='#F0F0F0',
        oceancolor='#E0E8F0',
        coastlinecolor='#C9C9C9'
    )
    
    fig.update_layout(
        title=dict(
            text="<b>Geographic Distribution of Samples</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#2D2D2D')
        ),
        height=700,
        width=1200,
        template='heather',
        hovermode='closest'
    )
    
    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Geographic map saved: {output_path}")
    
    return fig
