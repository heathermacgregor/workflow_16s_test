# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third Party Imports
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

import textwrap

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s import constants
from workflow_16s.figures.figures import (
    plotly_show_and_save,
    largecolorset,
    plot_legend,
    attach_legend_to_figure,
    _validate_metadata,
    _prepare_visualization_data,
    _create_colordict,
    _save_figure_and_legend,
    _pts_in_trace,
    _create_base_scatter_plot,
    _apply_common_layout
)

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')
sns.set_style('whitegrid')  # Set seaborn style globally
warnings.filterwarnings("ignore") # Suppress warnings

# ================================= GLOBAL VARIABLES ================================= #

DEFAULT_HEIGHT = 1000
DEFAULT_WIDTH = 1100

DEFAULT_COLOR_COL = 'dataset_name'
DEFAULT_SYMBOL_COL = 'nuclear_contamination_status'

DEFAULT_METRIC = 'braycurtis'

DEFAULT_PROJECTION = 'natural earth'
DEFAULT_LATITUDE_COL = 'latitude_deg'
DEFAULT_LONGITUDE_COL = 'longitude_deg'
DEFAULT_SIZE_MAP = 5
DEFAULT_OPACITY_MAP = 0.3

DEFAULT_FEATURE_TYPE = 'ASV'

DEFAULT_FEATURE_TYPE_ANCOM = 'l6'
DEFAULT_COLOR_COL_ANCOM = 'p'

# ================================ VISUALIZATIONS ================================== #

def create_heatmap(
    data: pd.DataFrame,
    feature_type: str = constants.DEFAULT_FEATURE_TYPE,
    output_dir: Union[Path, None] = None,
    show: bool = False
) -> go.Figure:
    """
    Generate feature abundance heatmap.
    
    Args:
        data:         Abundance matrix (features x samples).
        feature_type: Type of features (ASV, OTU, etc.).
        output_dir:   Directory to save outputs.
        show:         Display figure interactively.
        
    Returns:
        Plotly heatmap figure
    """
    fig = px.imshow(
        data,
        color_continuous_scale='viridis',
        labels={'x': 'Samples', 'y': feature_type, 'color': 'Abundance'},
        title=f"{feature_type} Abundance Heatmap"
    )
    
    fig.update_layout(
        template='heather',
        height=1200,
        xaxis_showticklabels=False,
        yaxis_showticklabels=False
    )
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_stem = f"heatmap.{feature_type.lower()}"
        plotly_show_and_save(fig, show, output_dir / file_stem)
    
    return fig


def create_ubiquity_plot(
    cm: np.ndarray, 
    pm: np.ndarray, 
    ubi_c: np.ndarray, 
    ubi_p: np.ndarray, 
    contaminated: List[str], 
    pristine: List[str], 
    transformation: str = None,
    output_dir: Union[Path, None] = None,
    show: bool = False
) -> go.Figure:
    """
    Generate ubiquity comparison plot.
    
    Args:
        cm:             Mean abundances in contaminated samples.
        pm:             Mean abundances in pristine samples.
        ubi_c:          Ubiquity values in contaminated samples.
        ubi_p:          Ubiquity values in pristine samples.
        contaminated:   IDs of contaminated samples.
        pristine:       IDs of pristine samples.
        transformation: Data transformation applied.
        output_dir:     Directory to save outputs.
        show:           Display figure interactively.
        
    Returns:
        Plotly scatter figure
    """
    # Calculate marker sizes and hover text
    sizes = [(v/len(contaminated)) + (ubi_p[i]/len(pristine)) for i, v in enumerate(ubi_c)]
    text = [
        f'Ubiq C = {v/len(contaminated):.3g}<br>Ubiq P = {ubi_p[i]/len(pristine):.3g}'
        f'<br>Mean C = {cm[i]:.3g}<br>Mean P = {pm[i]:.3g}'
        for i, v in enumerate(ubi_c)
    ]
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cm,
        y=pm,
        mode='markers',
        marker_size=10 * np.array(sizes),
        text=text
    ))
    
    fig.update_layout(
        template='heather',
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        title='Feature Ubiquity Comparison',
        xaxis_title='Contaminated',
        yaxis_title='Pristine'
    )

    if output_dir:
        plot_dir = output_dir / 'ubiquity'
        plot_dir.mkdir(parents=True, exist_ok=True)
        file_stem = f"ubiquity.{transformation or 'raw'}"
        plotly_show_and_save(fig, show, plot_dir / file_stem)
    
    return fig





def create_correlation_heatmap(
    data: pd.DataFrame,
    feature_type: str = constants.DEFAULT_FEATURE_TYPE,
    output_dir: Union[Path, None] = None,
    show: bool = False
) -> go.Figure:
    """
    Generate correlation matrix heatmap.
    
    Args:
        data:         Correlation matrix DataFrame.
        feature_type: Type of features (ASV, OTU, etc.).
        output_dir:   Directory to save outputs.
        show:         Display figure interactively.
        
    Returns:
        Plotly heatmap figure.
    """
    fig = px.imshow(
        data, 
        color_continuous_scale='bluered', 
        title=f"{feature_type} Correlation Matrix"
    )
    
    fig.update_layout(
        template='heather',
        height=1200,
        coloraxis_colorbar=dict(
            thickness=30,
            len=0.85,
            x=1.05,
            y=0.5,
            yanchor='middle',
            tickfont=dict(size=14)
    ))

    if output_dir:
        plot_dir = output_dir / 'correlation'
        plot_dir.mkdir(parents=True, exist_ok=True)
        file_stem = f"correlation.{feature_type.lower()}"
        plotly_show_and_save(fig, show, plot_dir / file_stem)
    
    return fig



def heatmap_feature_abundance(
    table: pd.DataFrame, 
    show: bool = False,
    output_dir: Union[str, Path] = None,
    feature_type: str = constants.DEFAULT_FEATURE_TYPE,
) -> go.Figure:
    """API endpoint for feature abundance heatmap"""
    output_path = Path(output_dir) if output_dir else None
    
    return create_heatmap(
        data=table,
        feature_type=feature_type,
        output_dir=output_path,
        show=show
    )


def plot_ubiquity(
    cm: np.ndarray, 
    pm: np.ndarray, 
    ubi_c: np.ndarray, 
    ubi_p: np.ndarray, 
    contaminated: List[str], 
    pristine: List[str], 
    show: bool = False,
    output_dir: Union[str, Path] = None,
    transformation: str = None
) -> go.Figure:
    """API endpoint for ubiquity plot"""
    output_path = Path(output_dir) if output_dir else None
    
    return create_ubiquity_plot(
        cm=cm,
        pm=pm,
        ubi_c=ubi_c,
        ubi_p=ubi_p,
        contaminated=contaminated,
        pristine=pristine,
        transformation=transformation,
        output_dir=output_path,
        show=show
    )


def create_feature_abundance_map(
    metadata: pd.DataFrame,
    feature_abundance: pd.DataFrame,
    feature_name: str,
    nfc_facilities_data: Optional[pd.DataFrame] = None,
    lat_col: str = constants.DEFAULT_LATITUDE_COL,
    lon_col: str = constants.DEFAULT_LONGITUDE_COL,
    projection: str = constants.DEFAULT_PROJECTION,
    output_dir: Union[Path, None] = None,
    show: bool = False,
    verbose: bool = False,
    size: int = DEFAULT_SIZE_MAP,
    opacity: float = DEFAULT_OPACITY_MAP,
    color_scale: str = 'Viridis',
    log_transform: bool = True
) -> go.Figure:
    """
    Generate interactive geographical map colored by feature abundance.
    
    Args:
        metadata:          DataFrame containing geographic coordinates and sample IDs.
        feature_abundance: DataFrame with feature abundances (samples x features).
        feature_name:      Name of the feature to visualize.
        nfc_facilities_data: Optional DataFrame containing NFC facilities data.
        lat_col:           Column containing latitude values.
        lon_col:           Column containing longitude values.
        projection:        Map projection type.
        output_dir:        Directory to save outputs.
        show:              Display figure interactively.
        verbose:           Enable debug logging.
        size:              Sample marker size.
        opacity:           Sample marker opacity.
        color_scale:       Plotly color scale name.
        log_transform:    Apply log10 transformation to abundance values.
        
    Returns:
        Plotly figure object.
    """
    # Preprocess sample data
    metadata = metadata.copy()
    
    # Validate inputs
    _validate_metadata(metadata, [lat_col, lon_col, '#sampleid'])
    
    # Prepare abundance data
    if feature_name not in feature_abundance.columns:
        raise ValueError(f"Feature '{feature_name}' not found in abundance data")
    
    abundance_data = feature_abundance[[feature_name]].copy()
    abundance_data.index = abundance_data.index.astype(str).str.strip().str.lower()
    
    # Prepare metadata
    metadata['#sampleid'] = metadata['#sampleid'].astype(str).str.strip().str.lower()
    metadata.set_index('#sampleid', inplace=True)
    
    # Merge metadata with abundance data
    merged = metadata.join(abundance_data, how='inner')
    
    if merged.empty:
        raise ValueError("No matching samples between metadata and abundance data")
    
    # Apply log transformation if requested
    abundance_values = merged[feature_name]
    if log_transform:
        # Handle zeros by adding a small pseudocount
        min_nonzero = abundance_values[abundance_values > 0].min() / 10
        abundance_values = np.log10(abundance_values.replace(0, min_nonzero))
        hover_text = f"Log10({feature_name})"
        colorbar_title = f"Log10 Abundance of {feature_name}"
    else:
        hover_text = feature_name
        colorbar_title = f"Abundance of {feature_name}"
    
    # Create visualization
    fig = go.Figure()
    
    # Add sample trace with continuous coloring
    fig.add_trace(
        go.Scattergeo(
            lon=merged[lon_col],
            lat=merged[lat_col],
            text=merged.index + f"<br>{hover_text}: " + abundance_values.round(4).astype(str),
            marker=dict(
                size=size,
                opacity=opacity,
                color=abundance_values,
                colorscale=color_scale,
                colorbar=dict(
                    title=colorbar_title,
                    thickness=20,
                    len=0.75
                ),
                showscale=True
            ),
            name='Samples',
            hoverinfo='text',
            hovertemplate='<b>%{text}</b><extra></extra>'
        )
    )

    # Add facilities layer if provided
    if nfc_facilities_data is not None:
        facilities = nfc_facilities_data.dropna(subset=['latitude_deg', 'longitude_deg']).copy()
        facility_text = []
        for _, row in facilities.iterrows():
            try:
                facility_text.append(f"{row['facility']} \n{row['country']}  \n{row['facility_type']} \n{row['facility_capacity']} \n{row['facility_status']} \n{row['facility_start_year']}-{row['facility_end_year']}")
            except KeyError:
                facility_text.append(f"{row['facility']}")
        
        fig.add_trace(
            go.Scattergeo(
                lon=facilities['longitude_deg'],
                lat=facilities['latitude_deg'],
                text=facility_text,
                marker=dict(
                    size=12,
                    color='black',
                    symbol='star',
                    line=dict(width=1, color='yellow')
                ),
                name='NFC Facilities',
                hoverinfo='text',
                hovertemplate='<b>%{text}</b><extra></extra>'
            )
        )

    # Add sample count annotation
    fig.add_annotation(
        text=f"n = {len(merged)} samples",
        xref="paper", yref="paper",
        x=0.99, y=0.01,
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(size=12, color="black"),
        bgcolor="rgba(255,255,255,0.4)",
    )
    
    # Configure map
    fig.update_geos(
        projection_type=projection,
        resolution=50,
        showcoastlines=True, coastlinecolor="#b5b5b5",
        showland=True, landcolor="#e8e8e8",
        showlakes=True, lakecolor="#fff",
        showrivers=True, rivercolor="#fff",
    )
    
    fig.update_layout(
        title_text=f"Geographical Distribution of {feature_name}",
        title_x=0.5,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # Save output
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_feature_name = feature_name.replace('/', '_').replace(' ', '_')
        file_stem = f"feature_map.{safe_feature_name}"
        plotly_show_and_save(fig, show, output_dir / file_stem, ['png', 'html'], verbose)
        
    return fig

def _prep_data_violin(
    metadata,
    table,
    feature,
):
    # Convert BIOM Table to Pandas DataFrame
    table_df = table_to_df(table)
  
    # Verify feature exists
    if feature not in table_df.columns:
        logger.warning(f"Feature '{feature}' not found in table")
        return None

    # Create a DataFrame with just this feature
    feature_df = table_df[[feature]].copy()
                        
    # Normalize IDs for matching
    feature_df.index = feature_df.index.astype(str).str.strip().str.lower()
    metadata_ids = metadata['#sampleid'].astype(str).str.strip().str.lower()
                        
    # Align metadata with feature table
    common_ids = feature_df.index.intersection(metadata_ids)
    if len(common_ids) == 0:
        logger.warning("No matching samples between feature table and metadata")
        return None
      
    # Add group column using aligned IDs
    group_map = metadata.set_index(metadata_ids)[col]
    feature_df[col] = feature_df.index.map(group_map)
                        
    # Remove samples without group assignment
    feature_df = feature_df.dropna(subset=[col])
                        
    # Create output directory
    feature_output_dir = output_dir / col / str(val) / table_type / level
    feature_output_dir.mkdir(parents=True, exist_ok=True)
  

def create_violin_plot(
    data: pd.DataFrame,
    feature: str,
    group_col: str = constants.DEFAULT_SYMBOL_COL,
    output_dir: Union[Path, None] = None,
    show: bool = False,
    verbose: bool = False
) -> Union[go.Figure, None]:
    """
    Generate feature distribution violin plot with robust NaN handling.
    
    Args:
        data:       DataFrame containing feature abundances and metadata.
        feature:    Feature name to visualize.
        status_col: Column containing contamination status.
        output_dir: Directory to save outputs.
        sub_dir:    Subdirectory for output.
        show:       Display figure interactively.
        verbose:    Enable debug logging.
        
    Returns:
        Plotly violin figure or None if no valid data
    """
    # Check if feature exists in data
    if feature not in data.columns:
        logger.warning(
          f"Feature '{feature}' not found in data columns. "
          f"First 5 available features: {data.columns.tolist()[:5]}"
        )
        return None

    # Create working copy
    plot_data = data.reset_index()
    initial_count = len(plot_data)
    
    # Check for NaNs
    nan_group = plot_data[group_col].isna().sum()
    nan_feature = plot_data[feature].isna().sum()
    
    # Remove NaNs
    plot_data_remove = plot_data.dropna(subset=[feature, status_col])
    final_count = len(plot_data_remove)
    if verbose:
        logger.info(
                f"Violin plot preprocessing for '{feature}': "
                f"Initial samples={initial_count}, "
                f"NaNs in status={nan_status}, "
                f"NaNs in feature={nan_feature}, "
                f"Final samples={final_count}"
        )

    # Handle empty data case
    if plot_data.empty:
        logger.warning(f"No valid data for '{feature}' after NaN removal")
        return None
    hover_data = ['index', 'dataset_name']
    hover_data = [i for i in hover_data if i in plot_data]
    
    # Create plot
    fig = px.violin(
        plot_data, 
        y=feature, 
        x=status_col,
        box=True,
        points="all",
        title='<br>'.join(textwrap.wrap(f"{feature} Distribution", width=40)),
        hover_data=['index']#, 'dataset_name']
    )
    
    # Customize layout
    fig.update_layout(
        template='heather',
        xaxis_title=xaxis_title,
        yaxis_title="Abundance (CLR)",
        width=800
    )

    # Save plot if requested
    if output_dir:
        plot_dir = output_dir / sub_dir
        plot_dir.mkdir(parents=True, exist_ok=True)
        file_stem = f'{status_col}.{feature}'.lower()
        plotly_show_and_save(fig, show, plot_dir / file_stem)
    
    return fig

# API ENDPOINTS
def violin_feature(
    df: pd.DataFrame, 
    feature: str, 
    output_dir: Union[str, Path], 
    sub_output_dir: str = 'faprotax',
    status_col: str = constants.DEFAULT_SYMBOL_COL, 
    show: bool = False
) -> go.Figure:
    """API endpoint for violin plot"""
    output_path = Path(output_dir) if output_dir else None
    
    return create_violin_plot(
        data=df,
        feature=feature,
        status_col=status_col,
        output_dir=output_path,
        sub_dir=sub_output_dir,
        show=show
    )

