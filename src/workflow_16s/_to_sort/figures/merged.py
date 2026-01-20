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
import seaborn as sns
import textwrap

# Local Imports
from workflow_16s import constants
from workflow_16s.figures.figures import (
    attach_legend_to_figure, largecolorset, plot_legend, plotly_show_and_save    
)

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')
sns.set_style('whitegrid')  # Set seaborn style globally

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

# ================================== CORE HELPERS =================================== #

def _validate_col_in_metadata(
    metadata: pd.DataFrame, 
    required_cols: List[str]
) -> None:
    """Validate presence of required columns in metadata.
    
    Args:
        metadata:      DataFrame containing sample metadata.
        required_cols: List of column names required for visualization.
        
    Raises:
        ValueError: If any required columns are missing.
    """
    missing = [col for col in required_cols if col not in metadata.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _prepare_visualization_data(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    color_col: str,
    symbol_col: str,
    placeholder: str = 'unknown',
    verbose: bool = False
) -> pd.DataFrame:
    """Prepare merged component and metadata data for visualization.
    
    Args:
        data:        DataFrame with data.
        metadata:    DataFrame with sample metadata.
        color_col:   Column to use for point coloring.
        symbol_col:  Column to use for point symbols.
        placeholder: Value to use for missing metadata.
        verbose:     Enable debug logging.
        
    Returns:
        Merged DataFrame ready for visualization
        
    Raises:
        ValueError: If no common samples exist between datasets
    """
    # Create copies to avoid modifying originals
    data_copy = data.copy()
    meta_copy = metadata.copy()
    
    # Standardize indices to lowercase strings with whitespace trimming
    data_copy.index = data_copy.index.astype(str).str.strip().str.lower()
    
    # Handle metadata index - prefer '#sampleid' column if available
    if '#sampleid' in meta_copy.columns:
        meta_copy['#sampleid'] = meta_copy['#sampleid'].astype(str).str.strip().str.lower()
        meta_copy.index = meta_copy['#sampleid']
        if verbose:
            logger.debug("Set metadata index from '#sampleid' column")
    else:
        if verbose:
            logger.warning(
                "Metadata missing '#sampleid' column - using existing index"
            )
        meta_copy.index = meta_copy.index.astype(str).str.strip().str.lower()
    
    # ------------------ DUPLICATE HANDLING ------------------ #
    # Identify and remove duplicate indices
    data_duplicates = data_copy.index.duplicated(keep='first')
    meta_duplicates = meta_copy.index.duplicated(keep='first')
    
    if verbose:
        if data_duplicates.any():
            dup_samples = data_copy.index[data_duplicates].unique()
            logger.warning(
                f"Found {len(dup_samples)} duplicate samples in data: "
                f"{list(dup_samples)[:5]}{'...' if len(dup_samples) > 5 else ''}"
            )
        if meta_duplicates.any():
            dup_samples = meta_copy.index[meta_duplicates].unique()
            logger.warning(
                f"Found {len(dup_samples)} duplicate samples in metadata: "
                f"{list(dup_samples)[:5]}{'...' if len(dup_samples) > 5 else ''}"
            )
    
    # Remove duplicates keeping first occurrence
    data_copy = data_copy[~data_duplicates]
    meta_copy = meta_copy[~meta_duplicates]
    # -------------------------------------------------------- #
    
    if verbose:
        # Log sample IDs for debugging
        logger.debug(f"data index (first 5): {data_copy.index.tolist()[:5]}")
        logger.debug(f"Metadata index (first 5): {meta_copy.index.tolist()[:5]}")
    
    # Find common samples
    common_idx = data_copy.index.intersection(meta_copy.index)
    if verbose:
        logger.info(
            f"Found {len(common_idx)} common samples after duplicate removal"
        )
    
    # Handle no common samples case with detailed diagnostics
    if len(common_idx) == 0:
        data_samples = set(data_copy.index)
        meta_samples = set(meta_copy.index)
        
        data_only = data_samples - meta_samples
        meta_only = meta_samples - data_samples

        logger.critical(
            "CRITICAL ERROR: No common samples between data and metadata!"
        )
        logger.critical(
            f"data-only samples ({len(data_only)}): "
            f"{list(data_only)[:5]}{'...' if len(data_only) > 5 else ''}"
        )
        logger.critical(
            f"Metadata-only samples ({len(meta_only)}): "
            f"{list(meta_only)[:5]}{'...' if len(meta_only) > 5 else ''}"
        )
        
        # Look for partial matches
        partial_matches = []
        for data_id in list(data_samples)[:10]:  # Check first 10
            for meta_id in meta_samples:
                if data_id in meta_id or meta_id in data_id:
                    partial_matches.append(f"{data_id} ~ {meta_id}")
                    break
        
        if partial_matches:
            logger.critical(f"Possible partial matches: {partial_matches[:5]}")
        
        raise ValueError("No common samples between data and metadata")
    
    # Filter to common samples
    meta_filtered = meta_copy.loc[common_idx].copy()
    data_filtered = data_copy.loc[common_idx].copy()
    
    # Remove existing color/symbol columns from data to prevent duplicates
    for col in [color_col, symbol_col]:
        if col in data_filtered.columns:
            if verbose:
                logger.warning(
                    f"Removing existing '{col}' column from data "
                    f"to prevent duplication"
                )
            data_filtered = data_filtered.drop(columns=col)
    # ========================================================= #
    
    # Handle missing metadata columns
    for col in [color_col, symbol_col]:
        if col not in meta_filtered.columns:
            if verbose:
                logger.warning(
                    f"Column '{col}' missing from metadata. "
                    f"Creating placeholder column."
                )
            meta_filtered[col] = placeholder
    
    # Merge data with metadata
    merged = data_filtered.join(
        meta_filtered[[color_col, symbol_col]], 
        how='inner'
    )
    
    # Fill missing values in metadata columns
    for col in [color_col, symbol_col]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(placeholder)
    
    if verbose:
        logger.debug(f"Merged data shape: {merged.shape}")
        logger.debug(f"Final columns: {merged.columns.tolist()}")
        
        # Verify no duplicate indices in final output
        if merged.index.duplicated().any():
            dupes = merged.index[merged.index.duplicated()].tolist()
            logger.error(
                f"DUPLICATE INDEX IN FINAL OUTPUT: {dupes[:5]}{'...' if len(dupes)>5 else ''}"
            )
        else:
            logger.debug("No duplicate indices in final merged data")
            
        # Verify no duplicate columns
        duplicate_cols = [col for col in merged.columns if col in [color_col, symbol_col]]
        if len(duplicate_cols) > 1:
            logger.error(
                f"DUPLICATE COLUMNS DETECTED: {duplicate_cols}"
            )
        else:
            logger.debug(f"Single column for each: {color_col}, {symbol_col}")
        merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged
    

def _create_colordict(
    data: Union[pd.Series, pd.DataFrame], 
    color_set: List[str] = largecolorset
) -> Dict[str, str]:
    """
    Create consistent color mapping for categories.
    
    Args:
        data:      Series or single-column DataFrame containing categorical values.
        color_set: List of colors to use for mapping.
        
    Returns:
        Dictionary mapping categories to colors.
    """
    # Handle DataFrame input (extract first column)
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            data = data.iloc[:, 0]
    
    categories = sorted(data.astype(str).unique())
    return {c: color_set[i % len(color_set)] for i, c in enumerate(categories)}


def _save_figure_and_legend(
    fig: go.Figure,
    colordict: Dict[str, str],
    color_col: str,
    output_dir: Path,
    file_stem: str,
    show: bool,
    verbose: bool
) -> None:
    """
    Save figure and corresponding legend.
    
    Args:
        fig:        Plotly figure to save.
        colordict:  Color mapping dictionary.
        color_col:  Name of the coloring column.
        output_dir: Directory to save outputs.
        file_stem:  Base filename without extension.
        show:       Display figure interactively.
        verbose:    Enable debug logging.
    """
    legend_fig = plot_legend(colordict)
    combined_fig = attach_legend_to_figure(fig, legend_fig)
    plotly_show_and_save(fig, show, output_dir / file_stem, ['png', 'html'], verbose)
    #plot_legend(colordict, color_col, output_dir / f"{file_stem}.legend.png")


def _pts_in_trace(trace):
    for key in ("x", "y", "z", "values"):
        arr = trace.get(key)
        if arr is None:
            continue
        # 2‑D arrays → total cells
        if (isinstance(arr, (list, tuple))
                and arr
                and isinstance(arr[0], (list, tuple))):
            return sum(len(row) for row in arr)
        try:
            return len(arr)
        except TypeError:
            pass
    return 0



def _create_base_scatter_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    symbol_col: str,
    colordict: Dict[str, str],
    hover_data: List[str]
) -> go.Figure:
    """
    Create standardized scatter plot configuration.
    
    Args:
        data:       DataFrame containing visualization data.
        x_col:      Column name for x-axis values.
        y_col:      Column name for y-axis values.
        color_col:  Column name for coloring points.
        symbol_col: Column name for point symbols.
        colordict:  Color mapping dictionary.
        hover_data: Additional columns to show in hover info.
        
    Returns:
        Configured Plotly scatter plot
    """
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color=color_col,
        symbol=symbol_col,
        color_discrete_map=colordict,
        hover_data=hover_data,
        opacity=0.8,
        size_max=10
    )
    n_pts = data.shape[0]
    fig.add_annotation(
        text=f"n = {n_pts}",
        xref="paper", yref="paper",        # relative to full plot
        x=0.99, y=0.01,                    # bottom‑right corner
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(size=18, color="black"),
        bgcolor="rgba(255,255,255,0.4)",
    )
    return fig


def _apply_common_layout(
    fig: go.Figure,
    x_title: str,
    y_title: str,
    title: str = None,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH
) -> go.Figure:
    """
    Apply consistent layout to figures.
    
    Args:
        fig:     Plotly figure to configure.
        x_title: Label for x-axis.
        y_title: Label for y-axis.
        title:   Overall plot title.
        height:  Figure height in pixels.
        width:   Figure width in pixels.
        
    Returns:
        Configured Plotly figure.
    """
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title
    )
    return fig

# ================================ VISUALIZATIONS ================================== #

def create_geographical_map(
    metadata: pd.DataFrame,
    nfc_facilities_data: Optional[pd.DataFrame] = None,
    color_col: str = DEFAULT_COLOR_COL,
    lat_col: str = DEFAULT_LATITUDE_COL,
    lon_col: str = DEFAULT_LONGITUDE_COL,
    projection: str = DEFAULT_PROJECTION,
    output_dir: Union[Path, None] = None,
    show: bool = False,
    verbose: bool = False,
    size: int = DEFAULT_SIZE_MAP,
    opacity: float = DEFAULT_OPACITY_MAP,
) -> Tuple[go.Figure, Dict]:
    """
    Generate interactive geographical map with distinct layers for samples and NFC facilities.
    
    Args:
        metadata:      DataFrame containing geographic coordinates.
        color_col:     Column to use for coloring sample points.
        lat_col:       Column containing latitude values.
        lon_col:       Column containing longitude values.
        projection:    Map projection type.
        output_dir:    Directory to save outputs.
        show:          Display figure interactively.
        verbose:       Enable debug logging.
        size:          Sample marker size.
        opacity:       Sample marker opacity.
        facilities_df: DataFrame containing NFC facilities data (requires 'facility_latitude_deg' 
                       and 'facility_longitude_deg' columns)
        
    Returns:
        Tuple containing figure and color mapping dictionary.
    """
    # Preprocess sample data
    metadata = metadata.copy()
    metadata[color_col] = metadata[color_col].fillna('other').replace('', 'other')
    metadata = metadata.sort_values(color_col)
    
    # Count samples per category
    cat_counts = metadata[color_col].value_counts().reset_index()
    cat_counts.columns = [color_col, 'sample_count']
    metadata = metadata.merge(cat_counts, on=color_col, how='left')
    
    # Create visualization
    colordict = _create_colordict(metadata[color_col])
    n_pts = metadata.shape[0]

    # Create base map with samples
    fig = go.Figure()
    
    # Add sample trace
    fig.add_trace(
        go.Scattergeo(
            lon=metadata[lon_col],
            lat=metadata[lat_col],
            text=metadata[color_col],
            marker=dict(
                size=size,
                opacity=opacity,
                color=metadata[color_col].map(colordict),
            ),
            name='Samples',
            hoverinfo='text',
            hovertemplate='<b>%{text}</b><extra></extra>'
        )
    )

    # Add facilities layer if provided
    facilities_df = nfc_facilities_data
    if facilities_df and (color_col == 'nuclear_contamination_status' or color_col == 'facility_match'):
        # Clean facilities data
        print(facilities_df)
        facilities = facilities_df.dropna(subset=['latitude_deg', 'longitude_deg']).copy()
        
        # Create text array without relying on index alignment
        facility_text = []
        for _, row in facilities.iterrows():
            try:
                facility_text.append(
                    f"{row['facility']} \n{row['country']}  \n{row['facility_type']} "
                    f"\n{row['facility_capacity']} \n{row['facility_status']} "
                    f"\n{row['facility_start_year']}-{row['facility_end_year']}"
                )
            except KeyError:
                facility_text.append(f"{row['facility']}")
        # Add facility trace with distinct style
        fig.add_trace(
            go.Scattergeo(
                lon=facilities['longitude_deg'],
                lat=facilities['latitude_deg'],
                text=facility_text,  # Use safe text array
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
        text=f"n = {n_pts}",
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
    
    # Save output
    if output_dir:
        file_stem = f"sample_map.{color_col}"
        output_dir.mkdir(parents=True, exist_ok=True)
        plotly_show_and_save(fig, show, output_dir / file_stem, ['png', 'html'], verbose)
        
    return fig, colordict


def create_ordination_plot(
    components: pd.DataFrame,
    metadata: pd.DataFrame,
    ordination_type: str,
    proportion_explained: np.ndarray = None,
    color_col: str = DEFAULT_COLOR_COL,
    symbol_col: str = DEFAULT_SYMBOL_COL,
    dimensions: Tuple[int, int] = (1, 2),
    transformation: str = None,
    output_dir: Union[Path, None] = None,
    show: bool = False,
    verbose: bool = False
) -> Tuple[go.Figure, Dict]:
    """
    Generate ordination plot (PCA/PCoA/MDS).
    
    Args:
        components:           DataFrame with ordination results.
        metadata:             DataFrame with sample metadata.
        ordination_type:      Type of ordination (PCA, PCoA, MDS).
        proportion_explained: Variance explained per dimension.
        color_col:            Column to use for coloring points.
        symbol_col:           Column to use for point symbols.
        dimensions:           Tuple of dimensions to plot (x,y).
        transformation:       Data transformation applied.
        output_dir:           Directory to save outputs.
        show:                 Display figure interactively.
        verbose:              Enable debug logging.
        
    Returns:
        Tuple containing figure and color mapping dictionary.
    """
    # Validate inputs
    _validate_col_in_metadata(metadata, [color_col, symbol_col, '#sampleid'])
    if not isinstance(color_col, str):
        raise TypeError(f"color_col must be a string, got {type(color_col)}")
    
    # Prepare data
    data = _prepare_visualization_data(
        components, metadata, color_col, symbol_col, verbose=verbose
    )
    data['sample_id'] = data.index
    
    # Create colormap
    colordict = _create_colordict(data[color_col])
    
    # Determine axis columns and titles
    prefix_map = {
        'PCA': 'PC',
        'PCoA': 'PCo',
        'MDS': ordination_type
    }
    prefix = prefix_map.get(ordination_type, ordination_type)
    
    x_dim, y_dim = dimensions
    x_col = f'{prefix}{x_dim}'
    y_col = f'{prefix}{y_dim}'
    
    # Verify dimension columns exist
    if x_col not in data.columns:
        available_dims = [col for col in data.columns if col.startswith(prefix)]
        raise ValueError(
            f"Column '{x_col}' not found. Available: "
            f"{available_dims[:5]}{'...' if len(available_dims) > 5 else ''}"
        )
    
    if y_col not in data.columns:
        available_dims = [col for col in data.columns if col.startswith(prefix)]
        raise ValueError(
            f"Column '{y_col}' not found. Available: "
            f"{available_dims[:5]}{'...' if len(available_dims) > 5 else ''}"
        )
    
    # Create axis titles
    if proportion_explained is not None and len(proportion_explained) >= max(x_dim, y_dim):
        x_title = f"{x_col} ({proportion_explained[x_dim-1]*100:.1f}%)"
        y_title = f"{y_col} ({proportion_explained[y_dim-1]*100:.1f}%)"
    else:
        x_title, y_title = x_col, y_col
    data = data.loc[:, ~data.columns.duplicated()]
    hover_data = ['sample_id', color_col, symbol_col]
    # Create plot
    fig = _create_base_scatter_plot(
        data,
        x_col,
        y_col,
        color_col,
        symbol_col,
        colordict,
        hover_data=hover_data
    )
    
    # Apply layout
    title = f'{ordination_type}: {transformation.title() if transformation else "Raw Data"}'
    fig = _apply_common_layout(fig, x_title, y_title, title)

    fig.update_layout(
        width=1600,
        title=dict(font=dict(size=24)),
        xaxis=dict(title=dict(font=dict(size=20)), scaleanchor="y", scaleratio=1.0),
        yaxis=dict(title=dict(font=dict(size=20)))
    )
    
    # Save output
    if output_dir:
        plot_dir = output_dir / ordination_type.lower()
        plot_dir.mkdir(parents=True, exist_ok=True)
        file_stem = f"{ordination_type.lower()}.{transformation or 'raw'}.{x_dim}-{y_dim}.{color_col}"
        plotly_show_and_save(fig, show, plot_dir / file_stem, ['png', 'html'], verbose=True)
        
    return fig, colordict


def create_heatmap(
    data: pd.DataFrame,
    feature_type: str = DEFAULT_FEATURE_TYPE,
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
        height=constants.DEFAULT_HEIGHT,
        width=constants.DEFAULT_WIDTH,
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


def create_violin_plot(
    data: pd.DataFrame,
    feature: str,
    status_col: str = DEFAULT_SYMBOL_COL,
    output_dir: Union[Path, None] = None,
    sub_dir: str = 'violin',
    xaxis_title: str = "Contamination Status",
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
        if verbose:
            logger.warning(f"Feature '{feature}' not found in data columns. Available features: {data.columns.tolist()[:5]}")
        return None

    # Create working copy
    plot_data = data.reset_index()
    initial_count = len(plot_data)
    
    # Check for NaNs
    nan_status = plot_data[status_col].isna().sum()
    nan_feature = plot_data[feature].isna().sum()
    
    # Remove NaNs
    plot_data_remove = plot_data.dropna(subset=[feature, status_col])
    final_count = len(plot_data_remove)
    
    logger.debug(
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


def create_ancom_plot(
    data: pd.DataFrame,
    min_W: float,
    feature_type: str = "l6",
    output_dir: Union[Path, None] = None,
    show: bool = False,
    reverse_x_axis: bool = True
) -> Tuple[go.Figure, Dict]:
    """Generate ANCOM volcano plot.
    
    Args:
        data:           ANCOM results DataFrame.
        min_W:          Significance threshold for W statistic.
        feature_type:   Taxonomic level (l2-l7).
        output_dir:     Directory to save outputs.
        show:           Display figure interactively.
        reverse_x_axis: Flip CLR values direction.
        
    Returns:
        Tuple containing figure and empty dictionary
    """
    # Optionally reverse CLR values
    if reverse_x_axis:
        data = data.assign(clr=-data['clr'])
    
    fig = px.scatter(
        data, 
        x='clr', 
        y='W', 
        hover_data=['Feature'],
        color='p',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        template='heather',
        width=constants.DEFAULT_WIDTH,
        height=constants.DEFAULT_HEIGHT,
        xaxis_title='CLR',
        yaxis_title='W statistic'
    )
    
    # Add significance threshold
    fig.add_shape(
        type='line',
        y0=min_W,
        y1=min_W,
        x0=data['clr'].min(),
        x1=data['clr'].max(),
        line=dict(color='black', dash='dash', width=4)
    )

    if output_dir:
        plot_dir = output_dir / 'ancom'
        plot_dir.mkdir(parents=True, exist_ok=True)
        file_stem = f"ancom.{feature_type.lower()}"
        plotly_show_and_save(fig, show, plot_dir / file_stem)
    
    return fig, {}


def create_correlation_heatmap(
    data: pd.DataFrame,
    feature_type: str = DEFAULT_FEATURE_TYPE,
    output_dir: Union[Path, None] = None,
    show: bool = False
) -> go.Figure:
    """Generate correlation matrix heatmap.
    
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
        coloraxis_colorbar=dict(thickness=30, len=0.85, x=1.05, y=0.5, yanchor='middle',
                                tickfont=dict(size=14))
    )

    if output_dir:
        plot_dir = output_dir / 'correlation'
        plot_dir.mkdir(parents=True, exist_ok=True)
        file_stem = f"correlation.{feature_type.lower()}"
        plotly_show_and_save(fig, show, plot_dir / file_stem)
    
    return fig


# ================================ API ENDPOINTS ==================================== #

# Simplified API functions using the new modular components
def sample_map_categorical(
    metadata: pd.DataFrame, 
    nfc_facilities_data: Optional[pd.DataFrame] = None,
    show: bool = False,
    output_dir: Union[str, Path, None] = None, 
    projection_type: str = constants.DEFAULT_PROJECTION, 
    height: int = constants.DEFAULT_HEIGHT, 
    size: int = constants.DEFAULT_SIZE_MAP, 
    opacity: float = constants.DEFAULT_OPACITY_MAP,
    lat: str = constants.DEFAULT_LATITUDE_COL, 
    lon: str = constants.DEFAULT_LONGITUDE_COL,
    color_col: str = constants.DEFAULT_COLOR_COL,
    limit_axes: bool = False,
    verbose: bool = False
) -> Tuple[go.Figure, Dict]:
    """API endpoint for geographical sample map"""
    # Convert output_dir to Path if provided
    output_path = Path(output_dir) if output_dir else None
    
    return create_geographical_map(
        metadata=metadata,
        nfc_facilities_data=nfc_facilities_data,
        color_col=color_col,
        lat_col=lat,
        lon_col=lon,
        projection=projection_type,
        output_dir=output_path,
        show=show,
        verbose=verbose,
        size=size,
        opacity=opacity
    )


def pca(
    components: pd.DataFrame, 
    proportion_explained: np.ndarray, 
    metadata: pd.DataFrame,
    color_col: str = constants.DEFAULT_COLOR_COL, 
    color_map: Dict = None,
    symbol_col: str = constants.DEFAULT_SYMBOL_COL,
    show: bool = False,
    output_dir: Union[str, Path] = None, 
    transformation: str = None,
    x: int = 1, 
    y: int = 2,
    verbose: bool = False
) -> Tuple[go.Figure, Dict]:
    """API endpoint for PCA plot"""
    output_path = Path(output_dir) if output_dir else None
    
    return create_ordination_plot(
        components=components,
        metadata=metadata,
        ordination_type='PCA',
        proportion_explained=proportion_explained,
        color_col=color_col,
        symbol_col=symbol_col,
        dimensions=(x, y),
        transformation=transformation,
        output_dir=output_path,
        show=show,
        verbose=verbose
    )


def pcoa(
    components: pd.DataFrame, 
    proportion_explained: np.ndarray, 
    metadata: pd.DataFrame,
    metric: str = constants.DEFAULT_METRIC,
    color_map: Dict = None,
    color_col: str = constants.DEFAULT_COLOR_COL, 
    symbol_col: str = constants.DEFAULT_SYMBOL_COL,
    show: bool = False,
    output_dir: Union[str, Path] = None, 
    transformation: str = None,
    x: int = 1, 
    y: int = 2,
    verbose: bool = False
) -> Tuple[go.Figure, Dict]:
    """API endpoint for PCoA plot"""
    output_path = Path(output_dir) if output_dir else None
    
    return create_ordination_plot(
        components=components,
        metadata=metadata,
        ordination_type='PCoA',
        proportion_explained=proportion_explained,
        color_col=color_col,
        symbol_col=symbol_col,
        dimensions=(x, y),
        transformation=transformation,
        output_dir=output_path,
        show=show,
        verbose=verbose
    )


def mds(
    df: pd.DataFrame, 
    metadata: pd.DataFrame,
    color_col: str, 
    symbol_col: str,
    show: bool = False,
    output_dir: Union[str, Path] = None, 
    transformation: str = None,
    mode: str = 'UMAP',
    x: int = 1, 
    y: int = 2,
    verbose: bool = False
) -> Tuple[go.Figure, Dict]:
    """API endpoint for MDS plot"""
    output_path = Path(output_dir) if output_dir else None
    
    return create_ordination_plot(
        components=df,
        metadata=metadata,
        ordination_type=mode,
        color_col=color_col,
        symbol_col=symbol_col,
        dimensions=(x, y),
        transformation=transformation,
        output_dir=output_path,
        show=show,
        verbose=verbose
    )


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


def ancom(
    data: pd.DataFrame,
    min_W: float,
    output_dir: Union[str, Path] = None,
    color_col: str = constants.DEFAULT_COLOR_COL_ANCOM,
    show: bool = False,
    reverse_x_axis: bool = True,
    feature_type: str = constants.DEFAULT_FEATURE_TYPE_ANCOM
) -> Tuple[go.Figure, Any]:
    """API endpoint for ANCOM plot"""
    output_path = Path(output_dir) if output_dir else None
    
    return create_ancom_plot(
        data=data,
        min_W=min_W,
        feature_type=feature_type,
        output_dir=output_path,
        show=show,
        reverse_x_axis=reverse_x_axis
    )


def plot_correlation_matrix(
    data: pd.DataFrame,
    show: bool = False,
    output_dir: Union[str, Path] = None,
    feature_type: str = constants.DEFAULT_FEATURE_TYPE
) -> go.Figure:
    """API endpoint for correlation matrix"""
    output_path = Path(output_dir) if output_dir else None
    
    return create_correlation_heatmap(
        data=data,
        feature_type=feature_type,
        output_dir=output_path,
        show=show
    )


import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def create_alpha_diversity_boxplot(
    alpha_df: pd.DataFrame,
    metadata: pd.DataFrame,
    group_column: str,
    metric: str,
    output_dir: Path,
    show: bool = True,
    verbose: bool = False,
    add_points: bool = True,
    add_stat_annot: bool = True,
    test_type: str = "nonparametric"
) -> go.Figure:
    """
    Create a boxplot for alpha diversity metric with optional statistical annotations.
    
    Args:
        alpha_df: DataFrame of alpha diversity metrics
        metadata: Sample metadata DataFrame
        group_column: Column in metadata defining groups
        metric: Alpha diversity metric to plot
        output_dir: Directory to save plot
        show: Whether to display the plot
        verbose: Enable verbose logging
        add_points: Add individual points to boxplot
        add_stat_annot: Add statistical annotations
        test_type: Type of statistical test ('parametric' or 'nonparametric')
    
    Returns:
        Plotly Figure object
    """
    try:
        merged = _prepare_visualization_data(
            alpha_df[[metric]], metadata, 'dataset_name', group_column, verbose=verbose
        )
        
        if merged.empty:
            if verbose:
                logger.warning(f"No data for {metric} after merging with metadata")
            return go.Figure()
        
        # Create boxplot
        fig = go.Figure()
        
        # Add box traces for each group
        groups = merged[group_column].unique()
        for group in groups:
            group_data = merged[merged[group_column] == group][metric]
            fig.add_trace(go.Box(
                y=group_data,
                name=str(group),
                boxpoints='all' if add_points else False,
                jitter=0.3,
                pointpos=-1.8,
                marker=dict(size=4),
            ))
        
        # Update layout
        fig.update_layout(
            title=f"{metric.replace('_', ' ').title()} by {group_column}",
            yaxis_title=metric.replace('_', ' ').title(),
            xaxis_title=group_column,
            template="heather",
            showlegend=False
        )
        fig = _apply_common_layout(
            fig, group_column, 
            metric.replace('_', ' ').title(), 
            f"{metric.replace('_', ' ').title()} by '{group_column}'"
        )
        # Add statistical annotations if requested
        if add_stat_annot and len(groups) > 1:
            try:
                # Perform statistical test
                if test_type == "parametric":
                    if len(groups) == 2:
                        # T-test
                        group1 = merged[merged[group_column] == groups[0]][metric]
                        group2 = merged[merged[group_column] == groups[1]][metric]
                        _, p_val = stats.ttest_ind(group1, group2, equal_var=False)
                        test_name = "T-test"
                    else:
                        # ANOVA
                        group_data = [merged[merged[group_column] == g][metric] for g in groups]
                        _, p_val = stats.f_oneway(*group_data)
                        test_name = "ANOVA"
                else:
                    if len(groups) == 2:
                        # Mann-Whitney
                        group1 = merged[merged[group_column] == groups[0]][metric]
                        group2 = merged[merged[group_column] == groups[1]][metric]
                        _, p_val = stats.mannwhitneyu(group1, group2)
                        test_name = "Mann-Whitney"
                    else:
                        # Kruskal-Wallis
                        group_data = [merged[merged[group_column] == g][metric] for g in groups]
                        _, p_val = stats.kruskal(*group_data)
                        test_name = "Kruskal-Wallis"
                
                # Add annotation
                fig.add_annotation(
                    x=0.5,
                    y=1.05,
                    xref="paper",
                    yref="paper",
                    text=f"{test_name} p = {p_val:.4f}",
                    showarrow=False,
                    font=dict(size=14)
                )
            except Exception as e:
                if verbose:
                    logger.error(f"Failed to add stats annotation: {e}")
        fig.update_layout(
            title=dict(font=dict(size=24)),
            xaxis=dict(title=dict(font=dict(size=20)), scaleanchor="y", scaleratio=1.0, tickfont=dict(size=16), showticklabels=True),
            yaxis=dict(title=dict(font=dict(size=20)))
        )
        # Save plot
        output_path = output_dir / f"alpha_boxplot_{metric}.html"
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        
        return fig
    
    except Exception as e:
        logger.error(f"Failed to create boxplot for {metric}: {e}")
        return go.Figure()
    

def create_alpha_diversity_stats_plot(
    stats_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    show: bool = False,
    verbose: bool = False,
    effect_size_threshold: float = 0.5
) -> go.Figure:
    """
    Create interactive visualization for statistical results.
    
    Args:
        stats_df:              DataFrame from analyze_alpha_diversity().
        output_dir:            Save directory.
        show:                  Display interactive plot.
        verbose:               Enable debug logging.
        effect_size_threshold: Threshold for meaningful effect size.
    
    Returns:
        Plotly Figure object
    """
    # Transform p-values
    stats_df['-log10(p_value)'] = -np.log10(stats_df['p_value'])
    stats_df['significant'] = stats_df['p_value'] < 0.05
    stats_df['meaningful_effect'] = stats_df['effect_size'].abs() > effect_size_threshold
    
    # Create figure with dual axes
    fig = go.Figure()
    
    # P-value bars
    fig.add_trace(go.Bar(
        x=stats_df['metric'],
        y=stats_df['-log10(p_value)'],
        marker_color=np.where(stats_df['significant'], '#EF553B', '#636EFA'),
        name='-log10(p-value)',
        text=stats_df.apply(lambda x: f"p={x['p_value']:.2e}", axis=1),
        textposition='auto'
    ))
    
    # Effect size markers
    fig.add_trace(go.Scatter(
        x=stats_df['metric'],
        y=stats_df['effect_size'],
        yaxis='y2',
        mode='markers+text',
        marker=dict(
            size=16,
            color=np.where(stats_df['meaningful_effect'], '#00CC96', '#AB63FA'),
            symbol=np.where(stats_df['effect_size'] > 0, 'triangle-up', 'triangle-down')
        ),
        text=stats_df['effect_size'].round(2),
        textfont=dict(size=10),  
        name='Effect Size'
    ))
    
    # Layout configuration
    fig.update_layout(
        title="Alpha Diversity Statistical Summary",
        template='heather',
        height=800,
        width=1200,
        font_size=16,
        xaxis_title="Diversity Metric",
        yaxis_title="-log10(p-value)",
        yaxis2=dict(
            title="Effect Size",
            overlaying="y",
            side="right",
            range=[stats_df['effect_size'].abs().max() * -1.5, 
                   stats_df['effect_size'].abs().max() * 1.5]
        ),
        legend=dict(x=1.1, y=1.0),
        hovermode="x unified"
    )
    fig = _apply_common_layout(fig, "Diversity Metric", "-log10(p-value)", "Alpha Diversity Statistical Summary")
    fig.update_layout(
        title=dict(font=dict(size=24)),
        xaxis=dict(title=dict(font=dict(size=20)), scaleanchor="y", scaleratio=1.0, tickfont=dict(size=16), showticklabels=True),
        yaxis=dict(title=dict(font=dict(size=20)))
    )
    # Add significance thresholds
    fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red")
    fig.add_hline(y=effect_size_threshold, line_dash="dot", line_color="green", yref="y2")
    fig.add_hline(y=-effect_size_threshold, line_dash="dot", line_color="green", yref="y2")
    
    # Improve text formatting
    fig.update_traces(texttemplate='%{text:.2e}')
    
    # Save output
    if output_dir:
        output_dir = Path(output_dir) / 'alpha_diversity'
        output_dir.mkdir(parents=True, exist_ok=True)
        file_stem = "statistics"
        plotly_show_and_save(fig, show, output_dir / file_stem, ['html', 'png'], verbose)
        
    return fig


def plot_alpha_correlations(
    corr_results: Dict[str, pd.DataFrame],
    output_dir: Optional[Path] = None,
    top_n: int = 10,
    height: int = 800,
    width: int = 1000
) -> Dict[str, go.Figure]:
    """
    Visualize top correlations for each alpha diversity metric.
    
    Args:
        corr_results: Output from analyze_alpha_correlations()
        output_dir: Directory to save plots
        top_n: Number of top correlations to display
        height: Figure height
        width: Figure width
        
    Returns:
        Dictionary of Plotly figures per metric
    """
    figs = {}
    
    for metric, df in corr_results.items():
        if df.empty:
            continue
            
        # Prepare data for visualization
        df = df.head(top_n).copy()
        df['abs_strength'] = df.apply(
            lambda x: abs(x['spearman_rho']) if x['type'] == 'numerical' else x['eta_squared'],
            axis=1
        )
        df['direction'] = df.apply(
            lambda x: "positive" if (x['type'] == 'numerical' and x['spearman_rho'] > 0) else "negative",
            axis=1
        )
        
        # Create figure
        fig = go.Figure()
        
        # Add bars with conditional coloring
        colors = {'numerical': 'rgba(54, 162, 235, 0.6)', 'categorical': 'rgba(255, 99, 132, 0.6)'}
        for _, row in df.iterrows():
            fig.add_trace(go.Bar(
                x=[row['metadata_column']],
                y=[row['abs_strength']],
                name=row['type'],
                marker_color=colors[row['type']],
                hoverinfo='text',
                hovertext=(
                    f"<b>{row['metadata_column']}</b><br>"
                    f"Type: {row['type']}<br>"
                    + (f"ρ = {row['spearman_rho']:.3f}<br>p = {row['spearman_p']:.4f}" 
                       if row['type'] == 'numerical' 
                       else f"η² = {row['eta_squared']:.3f}<br>p = {row['kruskal_p']:.4f}")
                )
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Top {top_n} Associations with {metric.replace('_', ' ').title()}",
            yaxis_title="Association Strength (|ρ| or η²)",
            barmode='group',
            height=height,
            width=width,
            template="plotly_white",
            hoverlabel=dict(bgcolor="white", font_size=12),
            legend_title="Variable Type"
        )
        
        fig.add_trace(go.Scatter(
            x=df['metadata_column'],
            y=df['abs_strength'] * 1.05,
            text=df.apply(lambda x: "★" if (x['spearman_p'] < 0.05 or x['kruskal_p'] < 0.05) else "", axis=1),
            mode="text",
            showlegend=False,
            textfont=dict(color="red", size=16)
        ))
        fig = _apply_common_layout(fig, "", "Association Strength (|ρ| or η²)", f"Top {top_n} Associations with {metric.replace('_', ' ').title()}")
        figs[metric] = fig
        
        # Save output
        if output_dir:
            #output_dir = Path(output_dir) / 'alpha_diversity'
            #output_dir.mkdir(parents=True, exist_ok=True)
            file_stem = "statistics"
            plotly_show_and_save(fig, show, output_dir / file_stem, ['html', 'png'], verbose)
                
    return figs

def create_feature_abundance_map(
    metadata: pd.DataFrame,
    feature_abundance: pd.DataFrame,
    feature_name: str,
    nfc_facilities_data: Optional[pd.DataFrame] = None,
    lat_col: str = DEFAULT_LATITUDE_COL,
    lon_col: str = DEFAULT_LONGITUDE_COL,
    projection: str = DEFAULT_PROJECTION,
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
    _validate_col_in_metadata(metadata, [lat_col, lon_col, '#sampleid'])
    
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
