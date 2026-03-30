"""Sample map visualizations for geographic and metadata context.

Generates interactive Plotly maps showing:
- Sample locations colored by metadata (categorical or numerical)
- Train/test split geographical distribution
- Metadata heatmaps by location
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger('workflow_16s')


def create_categorical_sample_map(
    metadata: pd.DataFrame,
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    color_col: str = 'Project',
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    height: int = 600,
    width: int = 1000,
) -> go.Figure:
    """
    Create sample map with categorical coloring.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata with lat/lon columns and categorical column to color by.
    lat_col : str
        Latitude column name.
    lon_col : str
        Longitude column name.
    color_col : str
        Categorical column to color samples by.
    title : str, optional
        Map title. Auto-generated if None.
    output_path : Path, optional
        Save HTML map to this path.
    height : int
        Figure height in pixels.
    width : int
        Figure width in pixels.
    
    Returns
    -------
    go.Figure
        Interactive Plotly map.
    """
    if lat_col not in metadata.columns or lon_col not in metadata.columns:
        logger.error(f"Required columns not found: {lat_col}, {lon_col}")
        return None
    
    if color_col not in metadata.columns:
        logger.warning(f"Color column '{color_col}' not found, using Project if available")
        color_col = 'Project' if 'Project' in metadata.columns else None
    
    # Create map
    fig = px.scatter_mapbox(
        metadata,
        lat=lat_col,
        lon=lon_col,
        color=color_col,
        hover_data=metadata.columns.tolist(),
        title=title or f"Sample Map (colored by {color_col})",
        mapbox_style="carto-positron",
        height=height,
        width=width,
    )
    
    fig.update_layout(
        mapbox=dict(zoom=2, center=dict(lat=0, lon=0)),
        hovermode='closest',
        font=dict(size=10),
    )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Categorical sample map saved: {output_path}")
    
    return fig


def create_numerical_sample_map(
    metadata: pd.DataFrame,
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    color_col: str = 'elevation',
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    colorscale: str = 'Viridis',
    height: int = 600,
    width: int = 1000,
) -> go.Figure:
    """
    Create sample map with numerical gradient coloring.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata with lat/lon and numerical columns.
    lat_col : str
        Latitude column name.
    lon_col : str
        Longitude column name.
    color_col : str
        Numerical column for color gradient.
    title : str, optional
        Map title.
    output_path : Path, optional
        Save HTML map.
    colorscale : str
        Plotly colorscale (e.g., 'Viridis', 'Plasma', 'RdBu').
    height : int
        Figure height.
    width : int
        Figure width.
    
    Returns
    -------
    go.Figure
        Interactive Plotly map.
    """
    if lat_col not in metadata.columns or lon_col not in metadata.columns:
        logger.error(f"Required columns not found: {lat_col}, {lon_col}")
        return None
    
    if color_col not in metadata.columns:
        logger.warning(f"Color column '{color_col}' not found")
        return None
    
    # Convert to numeric
    color_data = pd.to_numeric(metadata[color_col], errors='coerce')
    
    fig = go.Figure(data=go.Scattermapbox(
        lat=metadata[lat_col],
        lon=metadata[lon_col],
        mode='markers',
        marker=dict(
            size=8,
            color=color_data,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=color_col),
            opacity=0.7
        ),
        text=[f"{color_col}: {v:.2f}" for v in color_data],
        hovertemplate='<b>Location</b><br>Lat: %{lat}<br>Lon: %{lon}<br>%{text}<extra></extra>',
    ))
    
    fig.update_layout(
        mapbox=dict(style="carto-positron", zoom=2, center=dict(lat=0, lon=0)),
        title=title or f"Sample Map (gradient: {color_col})",
        height=height,
        width=width,
        hovermode='closest',
        font=dict(size=10),
    )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Numerical sample map saved: {output_path}")
    
    return fig


def create_train_test_split_map(
    metadata: pd.DataFrame,
    train_indices: pd.Index,
    test_indices: pd.Index,
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    title: str = "Train/Test Split (Geographic Distribution)",
    output_path: Optional[Path] = None,
    height: int = 600,
    width: int = 1000,
) -> go.Figure:
    """
    Create sample map showing train/test split distribution.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Full metadata.
    train_indices : pd.Index
        Training sample indices.
    test_indices : pd.Index
        Test sample indices.
    lat_col : str
        Latitude column.
    lon_col : str
        Longitude column.
    title : str
        Map title.
    output_path : Path, optional
        Save HTML map.
    height : int
        Figure height.
    width : int
        Figure width.
    
    Returns
    -------
    go.Figure
        Interactive Plotly map.
    """
    if lat_col not in metadata.columns or lon_col not in metadata.columns:
        logger.error(f"Required columns not found: {lat_col}, {lon_col}")
        return None
    
    # Create split indicator
    split = pd.Series('Other', index=metadata.index)
    split.loc[train_indices] = 'Training'
    split.loc[test_indices] = 'Testing'
    
    # Create map with custom colors
    fig = go.Figure()
    
    # Training samples (blue)
    train_mask = split == 'Training'
    fig.add_trace(go.Scattermapbox(
        lat=metadata.loc[train_mask, lat_col],
        lon=metadata.loc[train_mask, lon_col],
        mode='markers',
        name='Training',
        marker=dict(size=8, color='#1f77b4', opacity=0.7),
        hovertemplate='<b>Training Sample</b><br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>',
    ))
    
    # Test samples (orange)
    test_mask = split == 'Testing'
    fig.add_trace(go.Scattermapbox(
        lat=metadata.loc[test_mask, lat_col],
        lon=metadata.loc[test_mask, lon_col],
        mode='markers',
        name='Testing',
        marker=dict(size=10, color='#ff7f0e', symbol='diamond', opacity=0.8),
        hovertemplate='<b>Test Sample</b><br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>',
    ))
    
    fig.update_layout(
        mapbox=dict(style="carto-positron", zoom=2, center=dict(lat=0, lon=0)),
        title=title,
        height=height,
        width=width,
        hovermode='closest',
        legend=dict(x=0.01, y=0.99),
        font=dict(size=10),
    )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Train/test map saved: {output_path}")
    
    return fig


def create_metadata_sample_maps(
    metadata: pd.DataFrame,
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    output_dir: Optional[Path] = None,
    exclude_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Dict[str, go.Figure]:
    """
    Create sample maps for all metadata columns.
    
    Automatically detects categorical vs numerical columns and uses
    appropriate visualization (discrete colors vs gradients).
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Full metadata.
    lat_col : str
        Latitude column.
    lon_col : str
        Longitude column.
    output_dir : Path, optional
        Save HTML maps here.
    exclude_cols : List[str], optional
        Columns to skip.
    categorical_cols : List[str], optional
        Columns to treat as categorical. Auto-detected if None.
    
    Returns
    -------
    Dict[str, go.Figure]
        Maps keyed by column name.
    """
    if exclude_cols is None:
        exclude_cols = [lat_col, lon_col, 'index', 'sample_id']
    
    if categorical_cols is None:
        categorical_cols = []
    
    figures = {}
    
    for col in metadata.columns:
        if col in exclude_cols or col in [lat_col, lon_col]:
            continue
        
        logger.info(f"Creating sample map for '{col}'...")
        
        try:
            # Check if categorical or numerical
            if col in categorical_cols or metadata[col].dtype == 'object' or metadata[col].dtype.name == 'category':
                fig = create_categorical_sample_map(
                    metadata,
                    lat_col=lat_col,
                    lon_col=lon_col,
                    color_col=col,
                    title=f"Sample Map: {col}",
                    output_path=output_dir / f"sample_map_{col}.html" if output_dir else None,
                )
            else:
                # Numerical
                fig = create_numerical_sample_map(
                    metadata,
                    lat_col=lat_col,
                    lon_col=lon_col,
                    color_col=col,
                    title=f"Sample Map: {col} (gradient)",
                    output_path=output_dir / f"sample_map_{col}_gradient.html" if output_dir else None,
                )
            
            if fig:
                figures[col] = fig
        
        except Exception as e:
            logger.warning(f"Could not create map for '{col}': {e}")
    
    logger.info(f"✅ Created {len(figures)} sample maps")
    return figures


def create_multi_column_overview(
    metadata: pd.DataFrame,
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    columns_to_plot: Optional[List[str]] = None,
    max_cols: int = 4,
    output_path: Optional[Path] = None,
    height: int = 1200,
    width: int = 1600,
) -> Optional[go.Figure]:
    """
    Create subplot layout showing multiple metadata columns.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata.
    lat_col : str
        Latitude column.
    lon_col : str
        Longitude column.
    columns_to_plot : List[str], optional
        Columns to include. If None, uses first 8 columns.
    max_cols : int
        Maximum columns per row.
    output_path : Path, optional
        Save HTML.
    height : int
        Figure height.
    width : int
        Figure width.
    
    Returns
    -------
    go.Figure
        Subplot figure with multiple maps.
    """
    if columns_to_plot is None:
        columns_to_plot = [c for c in metadata.columns.tolist()[:8] 
                          if c not in [lat_col, lon_col]]
    
    n_plots = len(columns_to_plot)
    n_rows = (n_plots + max_cols - 1) // max_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=max_cols,
        specs=[[{'type': 'geo'} for _ in range(max_cols)] for _ in range(n_rows)],
        subplot_titles=columns_to_plot,
    )
    
    for idx, col in enumerate(columns_to_plot):
        row = idx // max_cols + 1
        col_pos = idx % max_cols + 1
        
        try:
            color_data = pd.to_numeric(metadata[col], errors='coerce')
            
            fig.add_trace(
                go.Scattergeo(
                    lat=metadata[lat_col],
                    lon=metadata[lon_col],
                    mode='markers',
                    marker=dict(
                        sizemode='area',
                        size=5,
                        color=color_data if color_data.notna().any() else metadata[col].astype('category').cat.codes,
                        showscale=(idx == 0),
                        colorscale='Viridis',
                    ),
                    name=col,
                    showlegend=False,
                ),
                row=row,
                col=col_pos,
            )
            
            # Update axis
            fig.update_geos(
                projection_type='natural earth',
                row=row,
                col=col_pos,
            )
        
        except Exception as e:
            logger.warning(f"Could not add trace for '{col}': {e}")
    
    fig.update_layout(
        title_text=f"Metadata Overview Map ({len(columns_to_plot)} columns)",
        height=height,
        width=width,
    )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Multi-column overview saved: {output_path}")
    
    return fig
