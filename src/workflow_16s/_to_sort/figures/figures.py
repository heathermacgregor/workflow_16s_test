# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import math
import logging
from pathlib import Path
from typing import Dict, List, Union

# Third Party Imports
import colorcet as cc
import json
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Local Imports
from workflow_16s import constants

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ================================= GLOBAL VARIABLES ================================= #

largecolorset = list(
  cc.glasbey + cc.glasbey_light + cc.glasbey_warm + cc.glasbey_cool + cc.glasbey_dark
)

# Define the plot template
pio.templates["heather"] = go.layout.Template(
  layout={
    'height': constants.DEFAULT_HEIGHT,
    'width': constants.DEFAULT_WIDTH,
    'title': {
      'font': {
        'family': 'HelveticaNeue-CondensedBold, Helvetica, Sans-serif',
        'size': 45,
        'color': '#000' # Black
      }
    },
    'title_x': 0.5,
    'font': {
      'family': 'Helvetica Neue, Helvetica, Sans-serif',
      'size': 26,
      'color' : '#000'
    },
    'paper_bgcolor': 'rgba(0, 0, 0, 0)', # Transparent
    'plot_bgcolor': '#fff', # White
    'colorway': largecolorset,
    'xaxis': {
      'showgrid': False,
      'zeroline': True,
      'showline': True,
      'linewidth': 3,
      'linecolor': 'black',
      'automargin': True,
      'mirror': True
    },
    'yaxis': {
      'showgrid': False,
      'zeroline': True,
      'showline': True,
      'linewidth': 3,
      'linecolor': 'black',
      'automargin': True,
      'mirror': True
    },
    'showlegend': False
  }
)
pio.templates.default = "heather" 

# ==================================== FUNCTIONS ===================================== #

def save_plotly_html(
    fig: go.Figure, 
    filepath: Union[str, Path],
    verbose: bool = True
) -> None:
    """Save a Plotly figure to an HTML file.

    Args:
        fig : The Plotly figure to save.
        filepath : Path where the HTML file will be saved.
        verbose : Verbosity flag.
    """
    # Convenience for optional INFO logging
    log_ok = (lambda msg: logger.debug(msg)) if verbose else (lambda *_: None)
    filepath = Path(filepath).expanduser().resolve()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_html(filepath, include_plotlyjs="cdn")
        log_ok(f"Saved figure to '{filepath}'")
    except Exception as e:
        logger.error(f"Failed to save figure: {str(e)}")


def load_plotly_html(
    filepath: Union[str, Path],
    verbose: bool = True
) -> go.Figure:
    """Load a Plotly figure from an HTML file.

    Args:
        filepath : Path to the saved HTML file.
        verbose : Verbosity flag.

    Returns:
        The reloaded Plotly figure.
    """
    # Convenience for optional INFO logging
    log_ok = (lambda msg: logger.debug(msg)) if verbose else (lambda *_: None)
  
    with open(filepath, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Extract the JSON figure data from the HTML
    start_marker = "window.PLOTLYENV=window.PLOTLYENV || {};Plotly.newPlot("
    start_idx = html_content.find(start_marker)
    if start_idx == -1:
        raise ValueError("Could not locate Plotly figure data in the HTML file.")
    
    start_idx = html_content.find("{", start_idx)  # JSON starts here
    end_idx = html_content.find(");", start_idx)   # JSON ends here
    json_str = html_content[start_idx:end_idx]

    fig_dict = json.loads(json_str)
    fig = pio.from_json(json.dumps(fig_dict))
    log_ok(f"Imported figure from '{filepath}'")
    return fig


def plotly_show_and_save(
    fig,
    show: bool = False,
    output_path: Union[str, Path] = None,
    save_as: List[str] = ['png', 'html'],
    scale: int = 3,
    engine: str = 'kaleido',
    verbose: bool = False,
    **write_kwargs
):
    """Save a Plotly figure to PNG and/or HTML formats.
    
    Args:
        fig:            Plotly Figure object to be saved/displayed.
        show:           Whether to display the figure (default: False).
        output_path:    Base output path for files. Actual files will have format-
                        specific extensions appended (.png, .html). Directory will 
                        be created if needed.
        save_as:        List of formats to save ('png', 'html'). 
                        Default: ['png', 'html']
        scale:          DPI‑like scale factor for raster outputs.
        engine:         Backend used for static image export ('kaleido', 'orca').
        verbose:        If True, logs success messages; errors are always logged.
        **write_kwargs: Extra args forwarded to `fig.write_image` / `fig.write_html`.
    
    Notes:
        - Saving PNG files requires kaleido: install with `pip install -U kaleido`.
        - File extensions are automatically handled (e.g., 'plot' becomes 'plot.png').
    """
    if output_path:
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convenience for optional INFO logging
        log_ok = (lambda msg: logger.debug(msg)) if verbose else (lambda *_: None)

        static_exts = {"png", "jpg", "jpeg", "pdf", "svg", "eps"}
        for ext in list(static_exts) + ['html']:
            output_path = str(output_path).removesuffix(f'.{ext}')
        
        for ext in static_exts.intersection(save_as):
            target = f"{output_path}.{ext}"
            try:
                fig.write_image(
                    str(target),
                    format=ext,
                    scale=scale,
                    engine=engine,
                    **write_kwargs,
                )
                log_ok(f"Saved figure to '{target}'.")
            except Exception as e:
                logger.error(
                    f"Failed to save figure: {str(e)}. "
                    "Make sure the export engine is installed "
                    "(e.g. `pip install -U kaleido`)."
                )
        
        if 'html' in save_as:
            target = f"{output_path}.html"
            try:
                fig.write_html(str(target), **write_kwargs)
                log_ok(f"Saved figure to '{target}'.")
            except Exception as e:
                logger.error(f"Failed to save figure: {str(e)}")


def marker_color_map(
    df: pd.DataFrame, 
    col: str, 
    continuous_color_set: bool = False
) -> None:
    """
    Generate color mappings for markers based on a DataFrame column.
    
    Creates either continuous (numeric) or categorical color mappings for 
    data visualization. For continuous data, uses a viridis colormap and 
    normalizes values. For categorical data, assigns unique colors from a 
    large color set.
    
    Args:
        df:                   DataFrame containing the data to color-map.
        col:                  Column name in `df` to base color mapping on.
        continuous_color_set: If True, treats column as continuous/numeric data. 
                              If False (default), treats as categorical data.
    
    Returns:
        tuple: 
            - For continuous: (ScalarMappable, list) 
                - ScalarMappable: Matplotlib mappable object for colorbar 
                  creation.
                - list: RGBA color strings for each data point.
            - For categorical: (dict, pd.Series)
                - dict: Color mapping dictionary {category: rgba_color}.
                - Series: RGBA color strings for each data point (aligned 
                  with df index).
    
    Raises:
        ValueError: If `continuous_color_set=True` but column contains 
        non-numeric data.
    
    Notes:
        - Handles NaN values by assigning transparent black (rgba(0,0,0,0)).
        - Continuous mode adds a 'marker_color' column to the input DataFrame.
        - Categorical mode does not modify the input DataFrame.
    """
    nan_color = (0, 0, 0, 0)
    if continuous_color_set:
        try:
            cmap = plt.cm.viridis      
            norm = mcolors.Normalize(
              vmin=df[col].min(), vmax=df[col].max()
            )    
            scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)    
            # Map column data to colors
            color_map_col = [
              scalar_mappable.to_rgba(value) if not np.isnan(value) 
              else nan_color for value in df[col]]         
            color_map_col = [
              'rgba({}, {}, {}, {})'.format(x[0], x[1], x[2], x[3]) 
              for x in color_map_col
            ]
            df['marker_color'] = color_map_colw
            return scalar_mappable, color_map_col
        
        except Exception as e:
            error_message = (
              f"ERROR: 'continuous_color_set' can only be used on "
              f"numeric data. Please check your input for '{col}'."
            )
            logger.error(error_message)
            raise ValueError(error_message)
            
    else:
        color_map_dict = {
            value: color if pd.notnull(value) else 'rgba({0}, {0}, {0}, {0})' 
            for value, color in zip(df[col].unique(), largecolorset)
        }
        color_map_col = df[col].map(color_map_dict)
        return color_map_dict, color_map_col
        

def create_color_mapping(
    metadata: pd.DataFrame, 
    color_col: str
) -> Dict:
    """
    Create a color mapping for unique values in a metadata column.
    
    Args:
        metadata:  Metadata containing the color column.
        color_col: Column to use for color mapping.
    
    Returns:
        A dictionary mapping unique values to colors.
    """
    unique_values = metadata[color_col].unique()
    return {
      value: largecolorset[i % len(largecolorset)] 
      for i, value in enumerate(unique_values)
    }


def plot_legend(
    color_dict: Dict[str, str],
    max_height: int = 600  # pixels (default max height before multi-column)
) -> go.Figure:
    """
    Creates a Plotly legend figure from a dictionary of labels and colors.
    
    Args:
        color_dict:  Dictionary with labels as keys and hex colors as values.
        max_height:  Maximum pixel height before creating additional columns.
    
    Returns:
        Plotly Figure object containing only the legend
    """
    if not color_dict:
        return go.Figure()  # Return empty figure if no items
    
    # Reverse the order of legend items
    items = list(color_dict.items())[::-1]
    n = len(items)
    
    # Calculate layout parameters
    row_height = 30  # pixels per row
    max_rows = max(1, min(n, max_height // row_height))
    n_cols = (n + max_rows - 1) // max_rows  # Ceiling division
    n_rows = min(n, max_rows)
    
    # Create figure
    fig = go.Figure()
    for label, color in items:
        fig.add_trace(
            go.Scatter(
                x=[None], 
                y=[None],
                mode='markers',
                marker=dict(color=color, size=15),
                name=label,
                showlegend=True
            )
        )
    
    # Configure legend layout
    fig.update_layout(
        legend=dict(
            title=None,
            orientation='v',
            itemsizing='constant',
            itemwidth=30,
            traceorder='normal',  # Maintains reversed item order
            bordercolor='black',
            borderwidth=1
        ),
        template="heather",
        width=200 * n_cols,  # Adjust width based on columns
        height=row_height * n_rows + 100,  # Dynamic height
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    # Hide axes
    fig.update_xaxes(
      showgrid=False, showticklabels=False, zeroline=False, mirror=True
    )
    fig.update_yaxes(
      showgrid=False, showticklabels=False, zeroline=False, mirror=True
    )
    return fig


def attach_legend_to_figure(
    main_fig: go.Figure,
    legend_fig: go.Figure,
    main_width: float = 0.8,
    legend_width: float = 0.2
) -> go.Figure:
    """
    Attaches a legend figure to the right of a main Plotly figure.
    Dynamically handles both geographic (geo) and Cartesian (xy) main plots.
    
    Args:
        main_fig:     The main Plotly figure (geo or xy plot).
        legend_fig:   The legend Plotly figure.
        main_width:   Width proportion for main figure (0-1).
        legend_width: Width proportion for legend (0-1).
    
    Returns:
        Combined Plotly figure with main plot and legend side-by-side.
    """
    # Determine subplot type based on main figure traces
    main_type = "geo" if any(trace.type in ["scattergeo", "choropleth"] for trace in main_fig.data) else "xy"
    
    # Create subplot figure with dynamic types
    combined_fig = make_subplots(
        rows=1, 
        cols=2,
        column_widths=[main_width, legend_width],
        specs=[[{"type": main_type}, {"type": "xy"}]],
        horizontal_spacing=0.01
    )
    
    # Add main figure traces
    for trace in main_fig.data:
        combined_fig.add_trace(trace, row=1, col=1)
    
    # Add legend figure traces
    for trace in legend_fig.data:
        combined_fig.add_trace(trace, row=1, col=2)
    
    # Update layout from main figure
    layout_updates = {
        "title": main_fig.layout.title,
        "template": main_fig.layout.template,
        "showlegend": False,
        "margin": main_fig.layout.margin
    }
    
    # Conditionally add geo layout if applicable
    if main_type == "geo" and hasattr(main_fig.layout, "geo"):
        layout_updates["geo"] = main_fig.layout.geo
    
    combined_fig.update_layout(**layout_updates)
    
    # Configure legend column axes
    combined_fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, row=1, col=2)
    combined_fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=1, col=2)
    
    # Handle dimensions
    main_width_val = main_fig.layout.width or 800
    main_height_val = main_fig.layout.height or 600
    legend_width_val = legend_fig.layout.width or 200
    
    combined_fig.update_layout(
        height=main_height_val,
        width=main_width_val + legend_width_val
    )
    return combined_fig


import math
from pathlib import Path
from typing import List, Union

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def combine_figures_as_subplots(
    figures: List[go.Figure],
    figures_per_row: int = 2,
    shared_xaxes: bool = False,
    shared_yaxes: bool = False,
    subplot_titles: List[str] = None,
    vertical_spacing: float = 0.1,
    horizontal_spacing: float = 0.05,
    show: bool = False,
    output_path: Union[str, Path] = None,
    verbose: bool = False
) -> go.Figure:
    """
    Combines multiple Plotly go.Figure objects into a subplot figure, preserving 
    legends, color bars, and the original x/y axis ratio for each subplot.

    Parameters:
        figures (List[go.Figure]): List of Plotly figures to combine.
        figures_per_row (int): Number of figures per row.
        shared_xaxes (bool): Share x-axis across subplots.
        shared_yaxes (bool): Share y-axis across subplots.
        subplot_titles (List[str]): Optional list of subplot titles.
        vertical_spacing (float): Vertical spacing between subplots (0-1).
        horizontal_spacing (float): Horizontal spacing between subplots (0-1).
        show (bool): Whether to display the figure.
        output_path (Union[str, Path]): Path to save the figure.
        verbose (bool): Whether to print verbose output.

    Returns:
        go.Figure: A single Plotly figure containing all input figures as subplots.
    """
    total_figs = len(figures)
    cols = figures_per_row
    rows = math.ceil(total_figs / cols)

    if subplot_titles is None:
        subplot_titles = [fig.layout.title.text if fig.layout.title and fig.layout.title.text else ""
                          for fig in figures]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        subplot_titles=subplot_titles,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing
    )

    # Store axis domain information for each subplot
    axis_domains = {}
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            axis_num = (r - 1) * cols + c
            xaxis_name = f'xaxis{axis_num}' if axis_num > 1 else 'xaxis'
            yaxis_name = f'yaxis{axis_num}' if axis_num > 1 else 'yaxis'
            
            if xaxis_name in fig.layout and 'domain' in fig.layout[xaxis_name]:
                x_domain = fig.layout[xaxis_name].domain
            else:
                x_domain = [0.0, 1.0]
                
            if yaxis_name in fig.layout and 'domain' in fig.layout[yaxis_name]:
                y_domain = fig.layout[yaxis_name].domain
            else:
                y_domain = [0.0, 1.0]
                
            axis_domains[(r, c)] = (x_domain, y_domain)

    for idx, subfig in enumerate(figures):
        row = idx // cols + 1
        col = idx % cols + 1
        x_domain, y_domain = axis_domains[(row, col)]
        
        for trace in subfig.data:
            fig.add_trace(trace, row=row, col=col)
            added_trace = fig.data[-1]
            
            # Position color bars relative to subplot
            if hasattr(added_trace, 'marker') and hasattr(added_trace.marker, 'colorbar') and added_trace.marker.colorbar is not None:
                added_trace.marker.colorbar.update(
                    x=x_domain[1] + 0.01,
                    y=(y_domain[0] + y_domain[1]) / 2
                )
            elif hasattr(added_trace, 'colorbar') and added_trace.colorbar is not None:
                added_trace.colorbar.update(
                    x=x_domain[1] + 0.01,
                    y=(y_domain[0] + y_domain[1]) / 2
                )
        
        # Preserve axis titles
        if 'xaxis' in subfig.layout and subfig.layout.xaxis.title.text:
            fig.update_xaxes(title_text=subfig.layout.xaxis.title.text, row=row, col=col)
        if 'yaxis' in subfig.layout and subfig.layout.yaxis.title.text:
            fig.update_yaxes(title_text=subfig.layout.yaxis.title.text, row=row, col=col)
        
        # Preserve aspect ratio using scaleanchor
        if hasattr(subfig.layout, 'yaxis') and hasattr(subfig.layout.yaxis, 'scaleanchor'):
            fig.update_yaxes(
                scaleanchor=subfig.layout.yaxis.scaleanchor,
                scaleratio=subfig.layout.yaxis.scaleratio,
                row=row, 
                col=col
            )
    
    # Preserve legends by showing them and adjusting their group assignments
    for trace in fig.data:
        if hasattr(trace, 'showlegend'):
            trace.showlegend = True
        if hasattr(trace, 'legendgroup'):
            trace.legendgroup = f"group_{trace.legendgroup}"
    
    plotly_show_and_save(fig, show, output_path, ['png', 'html'], verbose)
    return fig


def _validate_metadata(
    metadata: pd.DataFrame, 
    required_cols: List[str]
) -> None:
    """
    Validate presence of required columns in metadata.
    
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
    components: pd.DataFrame,
    metadata: pd.DataFrame,
    color_col: str,
    symbol_col: str,
    placeholder: str = 'unknown',
    verbose: bool = False
) -> pd.DataFrame:
    """
    Prepare merged component and metadata data for visualization.
    
    Args:
        components:  DataFrame with ordination components.
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
    comp_copy = components.copy()
    meta_copy = metadata.copy()
    
    # Standardize indices to lowercase strings with whitespace trimming
    comp_copy.index = comp_copy.index.astype(str).str.strip().str.lower()
    
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
    comp_duplicates = comp_copy.index.duplicated(keep='first')
    meta_duplicates = meta_copy.index.duplicated(keep='first')
    
    if verbose:
        if comp_duplicates.any():
            dup_samples = comp_copy.index[comp_duplicates].unique()
            logger.warning(
                f"Found {len(dup_samples)} duplicate samples in components: "
                f"{list(dup_samples)[:5]}{'...' if len(dup_samples) > 5 else ''}"
            )
        if meta_duplicates.any():
            dup_samples = meta_copy.index[meta_duplicates].unique()
            logger.warning(
                f"Found {len(dup_samples)} duplicate samples in metadata: "
                f"{list(dup_samples)[:5]}{'...' if len(dup_samples) > 5 else ''}"
            )
    
    # Remove duplicates keeping first occurrence
    comp_copy = comp_copy[~comp_duplicates]
    meta_copy = meta_copy[~meta_duplicates]
    # -------------------------------------------------------- #
    
    if verbose:
        # Log sample IDs for debugging
        logger.debug(f"Components index (first 5): {comp_copy.index.tolist()[:5]}")
        logger.debug(f"Metadata index (first 5): {meta_copy.index.tolist()[:5]}")
    
    # Find common samples
    common_idx = comp_copy.index.intersection(meta_copy.index)
    if verbose:
        logger.info(
            f"Found {len(common_idx)} common samples after duplicate removal"
        )
    
    # Handle no common samples case with detailed diagnostics
    if len(common_idx) == 0:
        comp_samples = set(comp_copy.index)
        meta_samples = set(meta_copy.index)
        
        comp_only = comp_samples - meta_samples
        meta_only = meta_samples - comp_samples

        logger.critical(
            "CRITICAL ERROR: No common samples between components and metadata!"
        )
        logger.critical(
            f"Components-only samples ({len(comp_only)}): "
            f"{list(comp_only)[:5]}{'...' if len(comp_only) > 5 else ''}"
        )
        logger.critical(
            f"Metadata-only samples ({len(meta_only)}): "
            f"{list(meta_only)[:5]}{'...' if len(meta_only) > 5 else ''}"
        )
        
        # Look for partial matches
        partial_matches = []
        for comp_id in list(comp_samples)[:10]:  # Check first 10
            for meta_id in meta_samples:
                if comp_id in meta_id or meta_id in comp_id:
                    partial_matches.append(f"{comp_id} ~ {meta_id}")
                    break
        
        if partial_matches:
            logger.critical(f"Possible partial matches: {partial_matches[:5]}")
        
        raise ValueError("No common samples between components and metadata")
    
    # Filter to common samples
    meta_filtered = meta_copy.loc[common_idx].copy()
    comp_filtered = comp_copy.loc[common_idx].copy()
    
    # ======== CRITICAL FIX: PREVENT DUPLICATE COLUMNS ======== #
    # Remove existing color/symbol columns from components to prevent duplicates
    for col in [color_col, symbol_col]:
        if col in comp_filtered.columns:
            if verbose:
                logger.warning(
                    f"Removing existing '{col}' column from components "
                    f"to prevent duplication"
                )
            comp_filtered = comp_filtered.drop(columns=col)
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
    
    # Merge components with metadata
    merged = comp_filtered.join(
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
            #raise ValueError("Color data must be a single column")
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
    height: int = constants.DEFAULT_HEIGHT,
    width: int = constants.DEFAULT_WIDTH
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

