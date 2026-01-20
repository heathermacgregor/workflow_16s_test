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
from workflow_16s.constants import DEFAULT_COLOR_COL, DEFAULT_SYMBOL_COL
from workflow_16s.figures.tools import PlotlyScatterPlot

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')
sns.set_style('whitegrid')  # Set seaborn style globally

# ================================= GLOBAL VARIABLES ================================= #

def beta_diversity_plot(
    components: pd.DataFrame,
    metadata: pd.DataFrame,
    ordination_type: str,
    proportion_explained: np.ndarray = None,
    color_col: str = DEFAULT_COLOR_COL,
    symbol_col: str = DEFAULT_SYMBOL_COL,
    dimensions: Tuple[int, int] = (1, 2),
    transformation: str = None,
    output_path: Union[Path, None] = None,
) -> go.Figure:
    """Generate ordination plot (PCA/PCoA/MDS).
        
    Args:
        components:           DataFrame with ordination results.
        metadata:             DataFrame with sample metadata.
        ordination_type:      Type of ordination (PCA, PCoA, MDS).
        proportion_explained: Variance explained per dimension.
        color_col:            Column to use for coloring points.
        symbol_col:           Column to use for point symbols.
        dimensions:           Tuple of dimensions to plot (x,y).
        transformation:       Data transformation applied.
        output_path:          Path to save outputs.
            
    Returns:
        Figure.
    """        
    title = f'{ordination_type}: {transformation.title() if transformation else "Raw Data"}'
    # Determine axis columns and titles
    prefix_map = {
        'PCA': 'PC',
        'PCoA': 'PCo',
        'MDS': ordination_type
    }
    prefix = prefix_map.get(ordination_type, ordination_type)

    x_dim, y_dim = dimensions
    x_col, y_col = f'{prefix}{x_dim}', f'{prefix}{y_dim}'
    
    if proportion_explained is not None and len(proportion_explained) >= max(x_dim, y_dim):
        x_title = f"{x_col} ({proportion_explained[x_dim-1]*100:.1f}%)"
        y_title = f"{y_col} ({proportion_explained[y_dim-1]*100:.1f}%)"
    else:
        x_title, y_title = x_col, y_col
           
    # Create plot
    obj = PlotlyScatterPlot(data, metadata, color_col, symbol_col)
    obj.create_fig(x_col, y_col, x_title, y_title, hover_data=['sample_id', color_col, symbol_col])
    if output_path: 
        obj.save(f"{output_path}.json")
    return obj.fig
