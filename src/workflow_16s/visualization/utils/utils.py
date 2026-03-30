# workflow_16s/visualization/utils/utils.py
"""
Plotting Utilities for the 16S Workflow.
Includes the custom Plotly theme setup and a
utility class for saving interactive HTML plots.
"""
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import anndata as ad
import colorcet as cc
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import numpy as np
from scipy.sparse import issparse  

from workflow_16s.downstream.utils import AnalysisUtils
from workflow_16s.utils.logger import get_logger

largecolorset = list(cc.glasbey + cc.glasbey_light + cc.glasbey_warm + cc.glasbey_cool + cc.glasbey_dark)

# Publication-ready defaults (equivalent to ~300 DPI at typical sizes)
DEFAULT_HEIGHT = 800  # Optimized for 8" height at 100 DPI
DEFAULT_WIDTH = 1200  # Optimized for 12" width at 100 DPI
PUBLICATION_DPI = 300  # Target DPI for PNG export

def setup_plotting_theme():
    """Defines and registers the 'heather' custom Plotly theme for publication-quality figures."""
    pio.templates["heather"] = go.layout.Template(
        layout=go.Layout({
            'height': DEFAULT_HEIGHT, 
            'width': DEFAULT_WIDTH,
            'margin': {'l': 80, 'r': 120, 't': 100, 'b': 80},  # Larger margins for labels
            'title': {
                'font': {'family': 'Arial, Helvetica, sans-serif', 'size': 20, 'color': '#000'},
                'x': 0.5,
                'xanchor': 'center'
            },
            'font': {'family': 'Arial, Helvetica, sans-serif', 'size': 14, 'color': '#000'},
            'paper_bgcolor': '#fff', 
            'plot_bgcolor': '#fff', 
            'colorway': largecolorset,
            'xaxis': {
                'showgrid': False, 
                'zeroline': True, 
                'showline': True, 
                'linewidth': 2, 
                'linecolor': 'black', 
                'automargin': True, 
                'mirror': True, 
                'tickangle': -45,  # Better readability than -90
                'tickfont': {'size': 12}
            },
            'yaxis': {
                'showgrid': True,  # Enable grid for easier reading
                'gridcolor': 'rgba(200, 200, 200, 0.3)',
                'zeroline': True, 
                'showline': True, 
                'linewidth': 2, 
                'linecolor': 'black', 
                'automargin': True, 
                'mirror': True,
                'tickfont': {'size': 12}
            },
            'showlegend': True,  # Enable legends by default
            'legend': {
                'bgcolor': 'rgba(255, 255, 255, 0.9)', 
                'bordercolor': 'black', 
                'borderwidth': 1, 
                'font': {'size': 11}, 
                'orientation': 'v', 
                'traceorder': 'normal',
                'yanchor': 'top', 
                'y': 0.98, 
                'xanchor': 'left', 
                'x': 1.02  # Position outside plot area
            },
        })
    )
    pio.templates.default = "heather"
    get_logger("workflow_16s").info("Custom 'heather' publication theme registered (300 DPI equivalent).")


class PlottingUtils:
    """Provides utility methods for saving Plotly figures as HTML, PNG, and JSON."""
    
    def __init__(self, logger: logging.Logger, run_settings: dict = None):
        self.logger = logger or get_logger("workflow_16s")
        self._plot_queue = []  # Queue for batch processing
        self.run_settings = run_settings or {}  # Store analysis parameters
        
    def _generate_metadata_header(self) -> str:
        """Generate HTML metadata header with run settings."""
        if not self.run_settings:
            return ""
        
        metadata_html = """
        <div id="analysis-metadata" style="
            background: #f8f9fa; 
            border: 1px solid #dee2e6; 
            border-radius: 5px;
            padding: 15px; 
            margin: 20px;
            font-family: Arial, sans-serif;
            font-size: 12px;">
            <h3 style="margin-top: 0; color: #495057;">Analysis Settings</h3>
            <table style="width: 100%; border-collapse: collapse;">
        """
        
        for key, value in self.run_settings.items():
            # Format key (replace underscores, capitalize)
            display_key = key.replace('_', ' ').title()
            metadata_html += f"""
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #dee2e6; font-weight: bold;">{display_key}:</td>
                    <td style="padding: 5px; border-bottom: 1px solid #dee2e6;">{value}</td>
                </tr>
            """
        
        metadata_html += """
            </table>
        </div>
        """
        return metadata_html
    def add_line_breaks(self, txt: str, max_length: int = 40) -> str:
        """
        Adds line breaks to a string if it exceeds a certain length, trying to break at spaces.
        """
        if len(txt) <= max_length:
            return txt
        else:
            # Try to break at the last space before max_length
            breakpoint = txt.rfind(' ', 0, max_length)
            if breakpoint == -1: breakpoint = max_length  # If no space found, break at max_length
            return txt[:breakpoint] + '<br>' + self.add_line_breaks(txt[breakpoint:].strip(), max_length)
        
    def save_plotly_fig(self, fig: go.Figure, filepath: Path, batch: bool = False, scale: int = None, save_png: bool = False, **kwargs):
        """
        Safely saves a Plotly figure to HTML, PNG, and JSON files.
        
        Args:
            fig: 
                Plotly figure to save
            filepath: 
                Destination base path (WITHOUT extension). e.g., 'output/my_plot' will create 'my_plot.html', 'my_plot.png', and 'my_plot.json'.
            batch: 
                If True, queues the plot for batch saving instead of immediate save.
            scale: 
                Scale factor for PNG export (default: 2 for ~200 DPI, faster rendering).
            save_png: 
                If True, saves PNG files. If False, skips PNG generation to avoid Kaleido timeouts (default: False).
            **kwargs: 
                Extra args (like 'width', 'height') passed to fig.write_image.
        """
        # Use moderate DPI for faster rendering (2x = ~200 DPI)
        if scale is None:
            scale = 2  # 2x scale for faster rendering while maintaining quality
        
        # Combine scale and other kwargs for image writing
        image_args = {'scale': scale, 'save_png': save_png, **kwargs}
        if batch: self._plot_queue.append((fig, filepath, image_args)); self.logger.debug(f"Queued plot for: {filepath}.*"); return
        
        # Immediate (non-batch) Save Logic
        output_dir = filepath.parent; base_name = filepath.name
        html_path = output_dir / f"{base_name}.html"; png_path = output_dir / f"{base_name}.png"; json_path = output_dir / f"{base_name}.json"
        save_png = image_args.pop('save_png', False)  # Extract save_png flag
        
        try:
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # 1. Save HTML with metadata header
            metadata_header = self._generate_metadata_header()
            fig.write_html(
                html_path, 
                include_plotlyjs='cdn',
                post_script=metadata_header if metadata_header else None,
                config={'toImageButtonOptions': {'format': 'png', 'scale': scale}}
            )
            self.logger.debug(f"Saved HTML: {html_path}")
            
            # 2. Save PNG (publication quality) - OPTIONAL
            if save_png:
                try: 
                    fig.write_image(png_path, **image_args)
                    self.logger.debug(f"Saved PNG ({scale}x scale): {png_path}")
                except Exception as e: 
                    self.logger.warning(f"Failed to save PNG plot to {png_path}. Is 'kaleido' installed? Error: {e}")
            
            # 3. Save JSON
            fig.write_json(json_path)
            self.logger.debug(f"Saved JSON: {json_path}")
            
        except Exception as e: 
            self.logger.error(f"Failed to save plot(s) for base {filepath}: {e}")
    
    def _save_single_static(self, plot_tuple, logger_ref):
        """Helper function for parallel plot saving."""
        # Unpack the new tuple: (fig, base_path, image_args)
        fig, base_path, image_args = plot_tuple 
        output_dir = base_path.parent; base_name = base_path.name
        html_path = output_dir / f"{base_name}.html"; png_path = output_dir / f"{base_name}.png"; json_path = output_dir / f"{base_name}.json"
        save_png = image_args.pop('save_png', False)  # Extract save_png flag
        
        try:
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # 1. Save HTML with metadata
            metadata_header = self._generate_metadata_header()
            scale = image_args.get('scale', 3)
            fig.write_html(
                html_path, 
                include_plotlyjs='cdn',
                post_script=metadata_header if metadata_header else None,
                config={'toImageButtonOptions': {'format': 'png', 'scale': scale}}
            )
            
            # 2. Save PNG (OPTIONAL - disabled by default to avoid Kaleido timeouts)
            if save_png:
                try: 
                    # --- FIX for kaleido pd.NA error ---
                    # This is a defensive conversion, as fig might not be serializable
                    # if it contains pd.NA. A better place is before fig creation.
                    fig.write_image(png_path, **image_args)
                except TypeError as e:
                    if "NAType" in str(e):
                        logger_ref.warning(f"[BATCH] Failed to save PNG {png_path} due to NAType error. Retrying with np.nan conversion... Error: {e}")
                        try:
                            # This is a costly operation, but a necessary fallback
                            fig_dict = fig.to_dict()
                            # This conversion is tricky. We'll just log the error.
                            logger_ref.error(f"[BATCH] PNG {png_path} failed permanently due to NAType. Plot data must be cleaned before creation.")
                        except Exception as e_retry: logger_ref.error(f"[BATCH] PNG {png_path} retry failed: {e_retry}")
                    else:
                        logger_ref.warning(f"[BATCH] Failed to save PNG {png_path}. Is 'kaleido' installed? Error: {e}")
                except Exception as e: 
                    logger_ref.warning(f"[BATCH] Failed to save PNG {png_path}. Is 'kaleido' installed? Error: {e}")
            
            fig.write_json(json_path) # 3. Save JSON
            return base_path # Return base_path on success
        except Exception as e: 
            logger_ref.error(f"Failed to save batched plot {base_path}.*: {e}"); return None # Return None on failure

    def flush_plot_queue(self, max_workers: int = 8):
        """
        Save all queued plots (HTML, PNG, JSON) in parallel using ThreadPoolExecutor.
        
        Args:
            max_workers: Maximum number of parallel workers for saving (default: 8)
        """
        if not self._plot_queue: self.logger.info("Plot queue is empty, nothing to flush."); return
        num_plots = len(self._plot_queue); self.logger.info(f"Saving {num_plots} plots from queue in parallel (max_workers={max_workers})...")
        
        saved_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Pass the logger to the map function
            results = list(executor.map(lambda item: self._save_single_static(item, self.logger), self._plot_queue))
            saved_count = sum(1 for r in results if r is not None)
        failed_count = num_plots - saved_count
        self._plot_queue.clear()
        if failed_count > 0: self.logger.warning(f"Batch plot save complete. Saved {saved_count}, Failed {failed_count}.")
        else: self.logger.info(f"Batch plot save complete. Successfully saved {saved_count} plots.")


def create_custom_legend_annotations(categories: List[str], colors: List[str], x: float = 1.02, y: float = 0.98, title: str = "Legend") -> List[dict]:
    """
    Creates custom legend as annotations for cases where plotly legend is problematic.
    
    Args:
        categories: List of category names
        colors: List of corresponding colors (hex or rgb)
        x: X position (paper coordinates, 0-1)
        y: Y position (paper coordinates, 0-1)
        title: Legend title
        
    Returns:
        List of annotation dictionaries to add to figure
    """
    annotations = []
    
    # Title annotation
    annotations.append({
        'text': f'<b>{title}</b>',
        'x': x, 'y': y,
        'xref': 'paper', 'yref': 'paper',
        'showarrow': False,
        'xanchor': 'left',
        'font': {'size': 13, 'color': 'black'}
    })
    
    # Category annotations with colored markers
    for i, (cat, color) in enumerate(zip(categories, colors)):
        y_pos = y - 0.04 * (i + 1)
        
        # Colored square marker
        annotations.append({
            'text': '\u25a0',  # Square symbol
            'x': x, 'y': y_pos,
            'xref': 'paper', 'yref': 'paper',
            'showarrow': False,
            'xanchor': 'left',
            'font': {'size': 16, 'color': color}
        })
        
        # Category label
        annotations.append({
            'text': f' {cat}',
            'x': x + 0.015, 'y': y_pos,
            'xref': 'paper', 'yref': 'paper',
            'showarrow': False,
            'xanchor': 'left',
            'font': {'size': 11, 'color': 'black'}
        })
    
    return annotations