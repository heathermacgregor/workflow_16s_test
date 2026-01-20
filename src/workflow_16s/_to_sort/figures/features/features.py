# ===================================== IMPORTS ====================================== #

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm

import seaborn as sns
sns.set_style('whitegrid')

import plotly.express as px
import plotly.io as pio
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import workflow_16s.figures.figures
from workflow_16s.figures.figures import (
    PlotlyFigure, plotly_show_and_save, largecolorset, marker_color_map, plot_legend
)

# ================================= GLOBAL VARIABLES ================================= #
# ==================================== FUNCTIONS ===================================== #

def heatmap_feature_abundance(
    table: pd.DataFrame, 
    show: bool = False,
    output_dir: Union[str, Path] = None,
    feature_type: str = "ASV",
) -> go.Figure:
    """
    Args:
        table:
        show:
        output_dir:
        feature_type:

    Returns:
        fig
    """
    
    fig = px.imshow(
        table,
        color_continuous_scale='viridis',
        labels={
            'x': 'Samples', 
            'y': feature_type, 
            'color': 'Abundance'
        },
        title=f"Heatmap of {feature_type} Abundance"
    )
    
    fig.layout.template = 'heather'
    
    fig.update_layout(
        height=1200, 
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )
    
    plotly_show_and_save(
        fig=fig,
        show=show,
        output_dir=output_dir,
        output_path=f"heatmap_{feature_type}"  
    )
    
    return fig

def pcoa(
    components, 
    proportion_explained, 
    metadata: pd.DataFrame,
    metric: str = 'braycurtis',
    color_col: str = 'dataset_name', 
    symbol_col: str = 'nuclear_contamination_status',
    show: bool = False,
    output_dir: Union[str, Path] = None, 
    transformation: str = None,
    x: int = 1, 
    y: int = 2
):
    """
    Args:
        components:
        proportion_explained:
        metadata:
        metric:
        color_col:
        symbol_col:
        show:
        output_dir:
        transformation:
        x:
        y:
    """
    exp_var = proportion_explained
    # Ensure the metadata DataFrame has the required columns
    if color_col not in metadata.columns or symbol_col not in metadata.columns:
        raise ValueError(f"Columns '{color_col}' and '{symbol_col}' not found in metadata.")

    # Find the intersection of A's index and B's columns
    matching_values = metadata.index.intersection(components.index)
    
    # Subset A and B to keep only the matching rows and columns
    metadata = metadata.loc[matching_values]
    components = components.loc[matching_values]
    
    # Drop rows with NaN values in the color and symbol columns
    placeholder = 'unknown'
    metadata = metadata.dropna(subset=[color_col, symbol_col], how='any').fillna(placeholder)

    # Join df with metadata on the index (ensuring they are aligned)
    data = components.join(metadata[[color_col, symbol_col]], how='inner')
    
    # Check for NaN values (for debugging purposes)
    if data.isna().any().any():
        logger.debug("Warning: There are NaN values in the data.")
        logger.debug(data.isna().sum())  # Print count of NaNs for each column
    
    colordict, colormap = marker_color_map(
        data, 
        color_col, 
        continuous_color_set=False
    )
    
    data['index'] = metadata.index
    
    # Create the scatter plot with index included in hover data
    fig = px.scatter(
        data, 
        x=f'PC{x}', 
        y=f'PC{y}', 
        color=color_col, 
        hover_data=['index', color_col], # Include the index as 'index'
        symbol=symbol_col, 
        color_discrete_map=colordict  # Use the color mapping
    )
            
    fig.update_xaxes(
        title_text=f"PCo{x} ({round(100 * exp_var[x-1], 2)}%)", 
        showline=True, 
        linewidth=7, 
        linecolor='black', 
        automargin=True, 
        mirror=True
    )
    fig.update_yaxes(
        title_text=f"PCo{y} ({round(100 * exp_var[y-1], 2)}%)", 
        showline=True, 
        linewidth=7, 
        linecolor='black', 
        automargin=True, 
        mirror=True
    )
    fig.update_layout(
        legend=dict(
            x=0, 
            y=1, 
            traceorder='normal', 
            font=dict(size=12), 
            bgcolor='rgba(0,0,0,0)'
        )
    )
        
    height=1000
    width=1100
    fig.update_layout(
        height=height, 
        width=width, 
        showlegend=False, 
        plot_bgcolor='#fff', 
        font_size=45,
        # Hide internal x and y axes by setting zeroline=False
        xaxis=dict(showticklabels=False, zeroline=True), 
        yaxis=dict(showticklabels=False, zeroline=True)
    )                                 

    if transformation:
        output_path = f'pcoa_{transformation}_{metric}_{x}-{y}_{color_col}_{symbol_col}'
    else:
        output_path = f'pcoa_{metric}_{x}-{y}_{color_col}_{symbol_col}'
        
    plotly_show_and_save(
        fig=fig,
        show=show,
        output_dir=Path(output_dir) / 'pcoa',
        output_path=output_path   
    )
    fig_legend = plot_legend(
        colordict, 
        color_col, 
        output_path=output_dir / f'legend_{color_col}.png'
    )
    return fig, fig_legend

def pca(
    components, 
    proportion_explained, 
    metadata: pd.DataFrame,
    color_col: str = 'dataset_name', 
    symbol_col: str = 'nuclear_contamination_status',
    show: bool = False,
    output_dir: Union[str, Path] = None, 
    transformation: str = None,
    x: int = 1, 
    y: int = 2
) -> Any:
    """
    Args:
        components:
        proportion_explained:
        metadata:
        color_col:
        symbol_col:
        show:
        output_dir:
        transformation:
        x:
        y:
    """
    # Ensure the metadata DataFrame has the required columns
    if color_col not in metadata.columns or symbol_col not in metadata.columns:
        raise ValueError(f"Columns '{color_col}' and '{symbol_col}' not found in metadata.")

    # Find the intersection of A's index and B's columns
    matching_values = metadata.index.intersection(components.index)
    
    # Subset A and B to keep only the matching rows and columns
    metadata = metadata.loc[matching_values]
    components = components.loc[matching_values]
    
    # Drop rows with NaN values in the color and symbol columns
    placeholder = 'unknown'
    metadata = metadata.dropna(
        subset=[color_col, symbol_col], 
        how='any'
    ).fillna(placeholder)

    # Join df with metadata on the index (ensuring they are aligned)
    data = components.join(
        metadata[[color_col, symbol_col]], 
        how='inner'
    )
    
    # Check for NaN values (for debugging purposes)
    if data.isna().any().any():
        logger.debug("Warning: There are NaN values in the data.")
        logger.debug(data.isna().sum())  # Print count of NaNs for each column
    
    colordict, colormap = marker_color_map(
        data, 
        color_col, 
        continuous_color_set=False
    )
    
    # Create a single plot
    fig = go.Figure()
    fig.layout.template = 'heather'
    
    data['index'] = metadata.index
    
    # Create the scatter plot with index included in hover data
    scatter = px.scatter(
        data, 
        x=f'PC{x}', 
        y=f'PC{y}', 
        color=color_col, 
        hover_data=['index', color_col],  # Include the index as 'index'
        symbol=symbol_col, 
        color_discrete_map=colordict  # Use the color mapping
    )
    
    scatter.update_traces(
        marker_size=8, 
        marker=dict(line=dict(width=0.1, color='black'))
    )

    # Add traces from scatter plot to the figure
    for trace in scatter.data:
        fig.add_trace(trace)

    # Update axes titles
    fig.update_xaxes(
        title_text=f'PC{x} ({round(100 * proportion_explained[x-1], 2)}%)', 
        showline=True, 
        linewidth=4, 
        linecolor='black', 
        mirror=True
    )
    fig.update_yaxes(
        title_text=f'PC{y} ({round(100 * proportion_explained[y-1], 2)}%)', 
        showline=True, 
        linewidth=4, 
        linecolor='black', 
        mirror=True
    )
    
    # Update layout properties
    fig.update_layout(
        height=800, 
        width=800, 
        showlegend=False,
        legend=dict(
            orientation="v",
            x=1.05, 
            y=1, 
            yanchor="top", 
            xanchor="left", 
            tracegroupgap=0,
            itemclick='toggleothers'
        ),
        font_family="Helvetica", 
        font_color="black", 
        font_size=15, 
        title_font_family="Helvetica", 
        title_font_color="black", 
        title_font_size=25, 
        title={
            'text': f'PCA', 
            'x': 0.5, 
            'xanchor': 'center', 
            'yanchor': 'top'
        },
        legend_title_font_color="black"
    )

    if transformation:
        output_path = f'pca_{metric}_{x}-{y}_{color_col}_{symbol_col}_{transformation}'
    else:
        output_path = f'pca_{metric}_{x}-{y}_{color_col}_{symbol_col}'

    plotly_show_and_save(
        fig=fig,
        show=show,
        output_dir=Path(output_dir) / 'pca',
        output_path=output_path   
    )

    legend_fig = plot_legend(
        colordict, 
        color_col, 
        output_path=output_dir / f'legend_{color_col}.png'
    )
        
    return fig, legend_fig

def mds(
    df: pd.DataFrame, 
    metadata: pd.DataFrame,
    group_col: str, 
    symbol_col: str,
    show: bool = False,
    output_dir: Union[str, Path] = None, 
    transformation: str = None,
    mode: str = 'UMAP',
    x: int = 1, 
    y: int = 2
) -> Any:
    """
    Args:
        df:
        metadata:
        group_col:
        symbol_col:
        show:
        output_dir:
        transformation:
        mode:
        x:
        y:
    """
    # Ensure the metadata DataFrame has the required columns
    if group_col not in metadata.columns or symbol_col not in metadata.columns:
        raise ValueError(f"Columns '{group_col}' and '{symbol_col}' not found in metadata.")

    df.index = df.index.astype(str)
    metadata.index = metadata.index.astype(str)
    
    # Drop rows with NaN values in the column
    metadata = metadata.dropna(subset=[group_col, symbol_col])
    
    # Filter the df to include only the IDs in filtered metadata
    df = df.loc[df.index.intersection(metadata.index).tolist()]

    # Join df with metadata on the index (ensuring they are aligned)
    data = df.join(metadata[[group_col, symbol_col]], how='inner')
    
    # Check for NaN values (for debugging purposes)
    if data.isna().any().any():
        logger.debug("Warning: There are NaN values in the data.")
        logger.debug(data.isna().sum())  # Print count of NaNs for each column
    
    # Ensure that the necessary columns for x and y exist in df
    if f'{mode}1' not in data.columns or f'{mode}2' not in data.columns:
        raise ValueError(f"Columns '{mode}1' and '{mode}2' not found in the data.")
    
    logger.debug(data.shape)
    
    colordict, colormap = marker_color_map(
        data, 
        group_col, 
        continuous_color_set=False
    )
    =
    # Create a single plot
    fig = go.Figure()
    fig.layout.template = 'heather'
    
    data['index'] = data.index
    
    # Create the scatter plot with index included in hover data
    scatter = px.scatter(
        data, 
        x=f'{mode}{x}', 
        y=f'{mode}{y}', 
        color=group_col, 
        hover_data=['index', group_col],  # Include the index as 'index'
        symbol=symbol_col, 
        color_discrete_map=colordict  # Use the color mapping
    )
    
    scatter.update_traces(
        marker_size=8, 
        marker=dict(line=dict(width=0.1, color='black'))
    )

    # Add traces from scatter plot to the figure
    for trace in scatter.data:
        fig.add_trace(trace)

    # Update axes titles
    fig.update_xaxes(
        title_text=f'{mode}{x}', 
        showline=True, 
        linewidth=4, 
        linecolor='black', 
        mirror=True
    )
    fig.update_yaxes(
        title_text=f'{mode}{y}', 
        showline=True, 
        linewidth=4, 
        linecolor='black', 
        mirror=True
    )
    
    # Update layout properties
    fig.update_layout(
        height=800, 
        width=800, 
        showlegend=False,
        legend=dict(
            orientation="v",
            x=1.05, 
            y=1, 
            yanchor="top", 
            xanchor="left", 
            tracegroupgap=0, 
            itemclick='toggleothers'
            ),
        font_family="Helvetica", 
        font_color="black", 
        font_size=15, 
        title_font_family="Helvetica", 
        title_font_color="black", title_font_size=25,
        title={'text': f'{mode}', 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        legend_title_font_color="black"
    )

    if transformation:
        output_path = f'{mode}_{x}-{y}_{group_col}_{symbol_col}_{transformation}'
    else:
        output_path = f'{mode}_{x}-{y}_{group_col}_{symbol_col}'

    plotly_show_and_save(
        fig=fig,
        show=show,
        output_dir=Path(output_dir) / mode,
        output_path=output_path   
    )
    legend_fig = plot_legend(
        colordict, 
        group_col, 
        output_path=output_dir / f'legend_{color_col}.png'
    )
    return fig, legend_fig


def plot_ubiquity(
    cm, 
    pm, 
    ubi_c, 
    ubi_p, 
    contaminated, 
    pristine, 
    show: bool = False,
    output_dir: Union[str, Path] = None,
    transformation: str = None
):
    """
    Args:
        cm:
        pm:
        ubi_c:
        ubi_p:
        contaminated:
        pristine:
        show:
        output_dir:
        transformation:
    """
    sizes = np.array(
        [(v / len(contaminated)) + (ubi_p[i] / len(pristine)) 
         for i, v in enumerate(ubi_c)]
    )
    
    text = [
        'Ubiq C = {0:.3g}'.format(v / len(contaminated)) + 
        '<BR>' +
        'Ubiq P = {0:.3g}'.format(ubi_p[i] / len(pristine)) + 
        '<BR><BR>' +
        'Mean C = {0:.3g}'.format(cm[i]) + 
        '<BR>' +
        'Mean P = {0:.3g}'.format(pm[i]) 
        #+ '<BR>' +
        #'Sample = ' + 
        #self.taxonomy.loc[ridict[list(self.cmms.keys())[i]]]['taxstring'] + 
        #' (' + str(list(self.cmms.keys())[i]) +')'
        for i, v in enumerate(ubi_c)
    ]
        
    fig=go.Figure()
    fig.layout.template = 'heather'
    
    fig.add_trace(
        go.Scatter(
            x=cm,
            y=pm,
            mode='markers',
            marker_size=10*sizes,
            text=text
        )
    ) #marker_color=colors,
    
    fig.update_traces(
        textposition='top center'
    )
    fig.update_xaxes(
        title_text='Contaminated'
    )
    fig.update_yaxes(
        title_text='Pristine'
    )
    fig.update_layout(
        height=600,
        width=800,
        title_text='Enrichment of significant changers',
        title_x=0.5,
        font_family='Arial',
        font_color='black',
        title_font_family='Arial',
        title_font_color='black',
        legend_title_font_color='black'
    )
    if transformation:
        output_path = f'ubiquity_{transformation}'
    else:
        output_path = f'ubiquity'

    plotly_show_and_save(
        fig=fig,
        show=show,
        output_dir=Path(output_dir) / 'ubiquity',
        output_path=output_path   
    )
    return fig


def violin_feature(
    df: pd.DataFrame, 
    feature: str, 
    output_dir: Union[str, Path], 
    sub_output_dir: str = 'faprotax',
    status_col: str = 'nuclear_contamination_status', 
    show: bool = False
):
    """
    Args:
        df:
        feature:
        output_dir:
        sub_output_dir:
        status_col:
        show:
    """
    # Ensure 'index' is available for hover data
    df_with_index = df.reset_index()
    
    # Plot with additional hover data
    fig = px.violin(
        df_with_index, 
        y=feature, 
        x=status_col,
        box=True,  # Adds a box plot inside the violin
        points="all",  # Shows individual points
        title=f"Violin Plot of {feature.replace('_', ' ').title()} by Contamination Status",
        labels={
            status_col: "Contamination Status",
            feature: feature.replace('_', ' ').title()
        },
        hover_data={'index': True, 'dataset_name': True} # Include index and dataset_name
    )  
    
    fig.layout.template = 'heather' 

    plotly_show_and_save(
        fig=fig,
        show=show,
        output_dir=Path(output_dir) / sub_output_dir,
        output_path=f'violin_{feature}_{status_col}'   
    )
    return fig
    
def ancom(
    data,
    min_W,
    color_col: str = 'p',
    show: bool = False,
    output_dir: Union[str, Path] = None,
    reverse_x_axis: bool = True,
    feature_type: str = "l6",
) -> go.Figure:
    """
    Args:
        data:
        min_W:
        color_col:
        show:
        output_dir:
        reverse_x_axis:
        feature_type:

    Returns:
        fig
    """

    colordict, colormap = marker_color_map(
        data, 
        color_col, 
        continuous_color_set=False
    )
    if reverse_x_axis:
        data['clr'] = [-i for i in data['clr']]
    
    fig = px.scatter(
        data, 
        x='clr', 
        y='W', 
        hover_data=['Feature']
    )
    
    fig.update_traces(
        marker=dict(
            symbol='circle', 
            color=colormap, 
            size=15, 
            opacity=0.8, 
            line=dict(color='rgba(0,0,0,0)', width=3)
        )
    ) 
    
    # Add a dashed line to indicate where values become significant.
    fig.add_shape(
        x0=data['clr'].min(), 
        y0=min_W,
        x1=data['clr'].max(), 
        y1=min_W,
        line=dict(color='Black', dash='dash', width=4)
    )
    
    fig.add_shape(
        type="rect",
        x0=data['clr'].min()*1.01, 
        y0=data['W'].max()*1.01,
        x1=0, 
        y1=min_W,
        fillcolor="lightblue",  # Background color on one side of the y-axis
        line=dict(width=0),
        layer="below"
    )
    
    fig.layout.template = 'heather'
    fig.update_layout(
        width=1100, 
        height=1000,
        paper_bgcolor='#fff'
    )
    fig.update_xaxes(
        title_text='clr', 
        showline=True, 
        linewidth=7, 
        linecolor='black', 
        mirror=True, 
        automargin=True
    )
    fig.update_yaxes(
        title_text='W', 
        showline=True, 
        linewidth=7, 
        linecolor='black', 
        mirror=True, 
        automargin=True
    )

    plotly_show_and_save(
        fig=fig,
        show=show,
        output_dir=output_dir,
        output_path=f"ancom_{feature_type}"  
    )
    fig_legend = plot_legend(
        colordict, 
        color_col, 
        output_path=output_dir / f'ancom_{feature_type}_legend_{color_col}.png'
    )
    return fig

def plot_correlation_matrix(
    data,
    show: bool = False,
    output_dir: Union[str, Path] = None,
    feature_type: str = "ASV"
) -> go.Figure:
    """
    Args:
        data:
        show:
        output_dir:
        feature_type:

    Returns:
        fig
    """
    fig = px.imshow(
        data, 
        color_continuous_scale='bluered', 
        title=f"Correlation Matrix of {feature_type}s"
    )
    
    fig.layout.template = 'heather'
    
    fig.update_layout(
        height=1200, 
        font_family="Helvetica", 
        font_color="black", 
        font_size=7,
        title_font_family="Helvetica", 
        title_font_color="black", 
        title_font_size=25, 
        legend_title_font_color="black",
        coloraxis_colorbar=dict(
            thickness=30,  # Adjust thickness of the color bar (controls width)
            len=0.85,  # Adjust length of the color bar (controls relative height)
            x=1.05,  # Position color bar to the right of the plot
            y=0.5,  # Center the color bar vertically
            yanchor='middle',  # Align the color bar to the middle of the plot
            tickfont=dict(size=14)  # Change the size here
            )
    )

    plotly_show_and_save(
        fig=fig,
        show=show,
        output_dir=output_dir,
        output_path=f"correlation_matrix_{feature_type}"  
    )
    return fig


