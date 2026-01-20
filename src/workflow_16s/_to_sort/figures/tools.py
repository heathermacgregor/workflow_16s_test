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


def fig_to_json(fig, output_path, verbose: bool = True):
    # Convenience for optional INFO logging
    log_ok = (lambda msg: logger.debug(msg)) if verbose else (lambda *_: None)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = f"{output_path}.json"
    try:
        fig.write_json(output_path, engine="json")
        log_ok(f"Saved figure to '{output_path}'")
    except Exception as e:
        logger.error(f"Failed to save figure: {str(e)}")

def fig_to_html(fig, output_path, verbose: bool = True):
    log_ok = (lambda msg: logger.debug(msg)) if verbose else (lambda *_: None)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = f"{output_path}.html"
    try:
        fig.write_html(output_path, include_plotlyjs="cdn")
        log_ok(f"Saved figure to '{output_path}'")
    except Exception as e:
        logger.error(f"Failed to save figure: {str(e)}")

def json_to_fig(output_path):
    return pio.read_json(output_path)
  
from functools import wraps
from contextlib import contextmanager


class DataPrepError(Exception):
    """Raised when data preparation encounters unrecoverable issues."""
    pass

@contextmanager
def prep_context(verbose: bool = False):
    """Context manager for data preparation with optional verbose logging."""
    if verbose:
        level = logger.getEffectiveLevel()
        logger.setLevel(logging.DEBUG)
    try:
        yield
    finally:
        if verbose:
            logger.setLevel(level)

def prep_step(description: str):
    """Decorator to log preparation steps."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"→ {description}")
            result = func(*args, **kwargs)
            logger.debug(f"✓ {description}")
            return result
        return wrapper
    return decorator

class DataPrep:
    """Elegant data preparation for visualization with fluent interface."""
    
    def __init__(self, data: pd.DataFrame, metadata: pd.DataFrame, verbose: bool = False):
        self.data = data.copy()
        self.metadata = metadata.copy()
        self.verbose = verbose
        self.placeholder = 'unknown'
    
    def with_placeholder(self, value: str) -> 'DataPrep':
        """Set placeholder value for missing data."""
        self.placeholder = value
        return self
    
    @prep_step("Normalizing sample indices")
    def _normalize_indices(self) -> 'DataPrep':
        """Normalize indices to lowercase strings, handling special cases."""
        # Data index
        self.data.index = self.data.index.astype(str).str.strip().str.lower()
        
        # Metadata index - prefer #sampleid if available
        if '#sampleid' in self.metadata.columns:
            self.metadata.index = (self.metadata['#sampleid']
                                 .astype(str).str.strip().str.lower())
        else:
            self.metadata.index = self.metadata.index.astype(str).str.strip().str.lower()
        
        return self
    
    @prep_step("Removing duplicate samples")
    def _remove_duplicates(self) -> 'DataPrep':
        """Remove duplicate indices, keeping first occurrence."""
        initial_data_size = len(self.data)
        initial_meta_size = len(self.metadata)
        
        self.data = self.data[~self.data.index.duplicated(keep='first')]
        self.metadata = self.metadata[~self.metadata.index.duplicated(keep='first')]
        
        if self.verbose:
            data_removed = initial_data_size - len(self.data)
            meta_removed = initial_meta_size - len(self.metadata)
            if data_removed:
                logger.info(f"Removed {data_removed} duplicate data samples")
            if meta_removed:
                logger.info(f"Removed {meta_removed} duplicate metadata samples")
        
        return self
    
    @prep_step("Finding sample intersection")
    def _find_intersection(self) -> 'DataPrep':
        """Find and validate common samples between datasets."""
        self.common_samples = self.data.index.intersection(self.metadata.index)
        
        if len(self.common_samples) == 0:
            self._diagnose_mismatch()
            raise DataPrepError("No common samples found between data and metadata")
        
        if self.verbose:
            logger.info(f"Found {len(self.common_samples)} common samples")
        
        return self
    
    def _diagnose_mismatch(self):
        """Provide diagnostic information for sample mismatches."""
        data_samples = set(self.data.index[:10])  # Sample for diagnosis
        meta_samples = set(self.metadata.index)
        
        # Look for partial matches
        matches = [(d, m) for d in data_samples for m in meta_samples 
                  if d in m or m in d]
        
        logger.error("Sample ID mismatch detected")
        logger.error(f"Data samples: {list(data_samples)}")
        logger.error(f"Metadata samples: {list(meta_samples)[:10]}")
        if matches:
            logger.error(f"Potential matches: {matches[:3]}")
    
    @prep_step("Preparing metadata columns")
    def _prepare_columns(self, required_cols: List[str]) -> 'DataPrep':
        """Ensure required columns exist with appropriate defaults."""
        for col in required_cols:
            if col not in self.metadata.columns:
                if self.verbose:
                    logger.warning(f"Missing column '{col}' - using placeholder")
                self.metadata[col] = self.placeholder
        
        return self
    
    @prep_step("Merging datasets")
    def _merge(self, color_col: str, symbol_col: str) -> pd.DataFrame:
        """Merge data with metadata on common samples."""
        # Filter to common samples
        data_filtered = self.data.loc[self.common_samples]
        meta_filtered = self.metadata.loc[self.common_samples]
        
        # Remove conflicting columns from data
        conflicts = [col for col in [color_col, symbol_col] if col in data_filtered.columns]
        if conflicts:
            data_filtered = data_filtered.drop(columns=conflicts)
        
        # Merge and fill missing values
        merged = data_filtered.join(meta_filtered[[color_col, symbol_col]])
        merged[[color_col, symbol_col]] = merged[[color_col, symbol_col]].fillna(self.placeholder)
        
        if self.verbose:
            logger.info(f"Final dataset shape: {merged.shape}")
        
        return merged
    
    def prepare(self, color_col: str, symbol_col: str) -> pd.DataFrame:
        """Execute the complete preparation pipeline."""
        with prep_context(self.verbose):
            return (self
                   ._normalize_indices()
                   ._remove_duplicates()
                   ._find_intersection()
                   ._prepare_columns([color_col, symbol_col])
                   ._merge(color_col, symbol_col))


class PlotlyScatterPlot:
    def __init__(
        self, 
        data: pd.DataFrame,
        metadata: pd.DataFrame,
        color_col: str,
        symbol_col: str,
        placeholder: str = 'unknown',
        verbose: bool = False
    ):
        """
        Args:
            data: Primary dataset with samples as index
            metadata: Sample metadata with matching identifiers
            color_col: Column for visualization colors
            symbol_col: Column for visualization symbols
            placeholder: Value for missing metadata (default: 'unknown')
            verbose: Enable detailed logging (default: False)
        """
        self.data = data
        self.metadata = metadata
        self.color_col = color_col
        self.symbol_col = symbol_col
        self.placeholder = placeholder
        self.verbose = verbose

        self.df = pd.DataFrame()
        self.colordict = {}

        self._prep_data()
        self._color_mapping()
      
    def _prep_data(self) -> pd.DataFrame:
        """Elegantly prepare data for visualization by merging datasets on common samples.
        
        Handles the common data preparation tasks:
            • Index normalization and cleanup
            • Duplicate sample removal  
            • Sample intersection validation
            • Missing metadata column creation
            • Clean dataset merging
        """
        missing = [col for col in ['#sample_id', self.color_col, self.symbol_col] if col not in self.metadata.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        self.df = DataPrep(self.data, self.metadata, self.verbose).with_placeholder(self.placeholder).prepare(self.color_col, self.symbol_col)
        self.df['#sampleid'] = self.df.index
        self.df = self.df.loc[:, ~self.df.columns.duplicated()]
    
    def _color_mapping(self, color_set: List[str] = largecolorset) -> Dict[str, str]:
        """Create consistent color mapping for categories.
        
        Args:
            color_set: List of colors to use for mapping.
        """
        # Handle DataFrame input (extract first column)
        if isinstance(self.df, pd.DataFrame):
            if self.df.shape[1] != 1:
                data = self.df.iloc[:, 0]
        
        categories = sorted(data.astype(str).unique())
        self.colordict = {c: color_set[i % len(color_set)] for i, c in enumerate(categories)}

    def create_fig(
        self, 
        x_col: str,
        y_col: str,
        x_title: str,
        y_title: str,
        hover_data: List[str]
    ):
        self._base_scatter_plot(x_col, y_col, hover_data)
        self._apply_common_layout(x_title, y_title)

    def save(self, output_path):
        fig_to_html(self.fig, output_path)
        fig_to_json(self.fig, output_path)
        
    def load(self, output_path) -> go.Figure:
        return pio.read_json(output_path)
      
    def _base_scatter_plot(
        self,
        x_col: str,
        y_col: str,
        hover_data: List[str]
    ) -> go.Figure:
        """Create standardized scatter plot configuration.
        
        Args:
            x_col:      Column name for x-axis values.
            y_col:      Column name for y-axis values.
            hover_data: Additional columns to show in hover info.
        """
        self.fig = px.scatter(
            self.df,
            x=x_col,
            y=y_col,
            color=self.color_col,
            symbol=self.symbol_col,
            color_discrete_map=self.colordict,
            hover_data=hover_data,
            opacity=0.8,
            size_max=10
        )
        n_pts = data.shape[0]
        self.fig.add_annotation(
            text=f"n = {n_pts}",
            xref="paper", yref="paper",        # relative to full plot
            x=0.99, y=0.01,                    # bottom‑right corner
            xanchor="right", yanchor="bottom",
            showarrow=False,
            font=dict(size=18, color="black"),
            bgcolor="rgba(255,255,255,0.4)",
        )

    def _apply_common_layout(
        self,
        x_title: str,
        y_title: str,
    ) -> go.Figure:
        """Apply consistent layout to figures.
        
        Args:
            x_title: Label for x-axis.
            y_title: Label for y-axis.
        """
        self.fig.layout.template = 'heather'
        self.fig.update_layout(
            xaxis_title=x_title,
            yaxis_title=y_title
        )
        '''
        fig.update_layout(
            width=1600,
            title=dict(font=dict(size=24)),
            xaxis=dict(title=dict(font=dict(size=20)), scaleanchor="y", scaleratio=1.0),
            yaxis=dict(title=dict(font=dict(size=20)))
        )
        '''
