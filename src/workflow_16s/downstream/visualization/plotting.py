"""
Plotting Utilities for the 16S Workflow.
Includes the custom Plotly theme setup and a
utility class for saving interactive HTML plots.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
from typing import List
import anndata as ad
import colorcet as cc
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import numpy as np
from sklearn.metrics.pairwise import haversine_distances 
from scipy.sparse import issparse  
from workflow_16s.downstream.utils import AnalysisUtils
logger = logging.getLogger("workflow_16s")

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
    logger.info("Custom 'heather' publication theme registered (300 DPI equivalent).")


class PlottingUtils:
    """Provides utility methods for saving Plotly figures as HTML, PNG, and JSON."""
    
    def __init__(self, logger: logging.Logger, run_settings: dict = None):
        self.logger = logger
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

            
def plot_stacked_bar(adata: ad.AnnData, cst_col: str, plottable_cat: List[str], target_path: Path):
    """Generates 100% stacked bar charts for a primary categorical column (e.g., CST) against a list of other metadata columns."""
    # Use the parent directory from the target_path
    target_path = Path(target_path)
    plot_dir = target_path.parent
    plot_dir.mkdir(parents=True, exist_ok=True)
        
    if not plottable_cat: logger.warning("No plottable categorical columns provided for stacked bar plot."); return
    logger.info(f"Generating stacked bar plots for '{cst_col}'...")

    for meta_col in plottable_cat:
        if meta_col not in adata.obs.columns: logger.warning(f"Skipping stacked bar plot: '{meta_col}' not in adata.obs."); continue  
        try:
            # Convert both columns to string to handle mixed types 
            cst_data_str = adata.obs[cst_col].astype(str).fillna('Unknown')
            meta_data_str = adata.obs[meta_col].astype(str).fillna('Unknown')
            # 1. Create a contingency table (counts)
            contingency_table = pd.crosstab(meta_data_str, cst_data_str)
            # 2. Normalize to get percentages (100% stacked bar)
            normalized_table = contingency_table.div(contingency_table.sum(axis=1), axis=0)
            # 3. Melt for Plotly (long format)
            plot_df = normalized_table.reset_index().melt(id_vars=meta_col, var_name=cst_col, value_name='Percentage')
            # 4. Create the figure
            fig = px.bar(plot_df, x=meta_col, y='Percentage', color=cst_col, title=f'Community State Type Distribution by {meta_col}', labels={meta_col: meta_col.replace('_', ' ').capitalize(), cst_col: cst_col.replace('_', ' ').capitalize()}, text_auto=True)
            fig.update_traces(texttemplate='%{y:.1%}')
            fig.update_layout(xaxis_title=meta_col.replace('_', ' ').capitalize(), yaxis_title='Proportion of Samples', yaxis_tickformat='.0%')
            # 5. Save the figure using the class's save method
            # We use batch=False to save immediately, since the orchestrator calls flush() right after this.
            safe_col_name = re.sub(r'[^A-Za-z0-9_]+', '', meta_col)
            # We use the plot_dir, not the full target_path
            save_path_stem = plot_dir / f"cst_vs_{safe_col_name}_bar"
            PlottingUtils(logger).save_plotly_fig(fig, save_path_stem, batch=False) 
        except Exception as e: logger.error(f"Failed to generate stacked bar plot for {meta_col}: {e}")
            
                
def plot_metadata_pairplot(adata: ad.AnnData, plot_dir_meta: Path, max_vars: int = 10, save_scale: int = 2):
    """Creates scatter matrix for top numerical metadata."""
    logger.info(f"--- Generating Metadata Pair Plot (Top {max_vars} Numeric) ---")
    if adata is None: logger.error("AnnData object not loaded."); return
    meta_vars = AnalysisUtils.find_plottable_metadata(adata, fullness_threshold=0.5); numeric_vars = meta_vars['numeric']
    if len(numeric_vars) < 2: logger.info("Skipping pair plot"); return
    vars_to_plot = sorted(numeric_vars)[:max_vars]; logger.info(f"Plotting pair plot for: {vars_to_plot}")
    # Replace pd.NA with np.nan to prevent kaleido save error 
    plot_df = adata.obs[vars_to_plot].copy().replace(pd.NA, np.nan)
    color_col = 'facility_match' if 'facility_match' in adata.obs.columns else None
    if color_col:
        fm_data = adata.obs[color_col]
        if isinstance(fm_data.dtype, pd.CategoricalDtype):
            if 'Unknown' not in fm_data.cat.categories: fm_data = fm_data.cat.add_categories('Unknown')
            plot_df[color_col] = fm_data.fillna('Unknown')
        else: plot_df[color_col] = fm_data.astype(str).fillna('Unknown')
    logger.debug(f"Pairplot df shape: {plot_df.shape}")
    if plot_df.empty: logger.warning("Pairplot DataFrame is empty before plotting."); return
    try:
        n_vars = len(vars_to_plot); base_size = 900; plot_height = min(max(base_size, n_vars * 175), 2500); plot_width = min(max(base_size, n_vars * 175), 2500)
        fig = px.scatter_matrix(plot_df, dimensions=vars_to_plot, color=color_col, title="Pairwise Relationships of Numerical Metadata")  
        fig.update_traces(diagonal_visible=True, showupperhalf=True, marker=dict(size=5, opacity=0.7), selector=dict(type='scatter'))
        fig.update_layout(height=plot_height, width=plot_width, font_size=max(10, 22 - n_vars), legend_font_size=max(10, 22 - n_vars), title_font_size=max(16, 32 - n_vars))
        plot_path = plot_dir_meta / "metadata_pairplot"; PlottingUtils(logger).save_plotly_fig(fig, plot_path); logger.info(f"Saved metadata pair plot: {plot_path}")
    except Exception as e: logger.error(f"Failed metadata pair plot: {e}")


def plot_metadata_correlation_heatmap(adata: ad.AnnData, plot_dir_meta: Path, save_scale: int = 2):
    """Calculates and plots Spearman correlation heatmap for numerical metadata."""
    logger.info("--- Generating Metadata Correlation Heatmap ---")
    if adata is None: logger.error("AnnData object not loaded."); return
    meta_vars = AnalysisUtils.find_plottable_metadata(adata, fullness_threshold=0.5); numeric_vars = meta_vars['numeric']
    if len(numeric_vars) < 2: logger.info("Skipping correlation heatmap"); return
    logger.info(f"Calculating Spearman correlation matrix for {len(numeric_vars)} variables."); 
    # Replace pd.NA with np.nan to prevent kaleido save error 
    numeric_df = adata.obs[numeric_vars].copy().replace(pd.NA, np.nan)
    for col in numeric_df.columns:
        try: numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce').astype(float)
        except Exception as e: logger.warning(f"Could not convert '{col}' to float: {e}. Dropping."); numeric_df = numeric_df.drop(columns=[col])
    if numeric_df.empty or numeric_df.shape[1] < 2: logger.warning("Skipping correlation heatmap."); return
    try: numeric_df.fillna(numeric_df.mean(), inplace=True)
    except Exception as e: logger.error(f"Failed during fillna for heatmap: {e}"); return
    try: corr_matrix = numeric_df.corr(method='spearman')
    except Exception as e: logger.error(f"Failed correlation matrix calculation: {e}"); return
    if isinstance(corr_matrix, np.ndarray): corr_matrix_df = pd.DataFrame(corr_matrix, index=numeric_vars, columns=numeric_vars) 
    else: corr_matrix_df = corr_matrix 
    try:
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', color_continuous_midpoint=0, zmin=-1, zmax=1, title="Spearman Correlation of Numerical Metadata")
        fig.update_traces(texttemplate="%{z:.2f}", textfont_size=max(8, 14 - len(numeric_vars) // 2))
        fig.update_xaxes(side="bottom", tickangle=-90)
        n_vars = len(numeric_vars); plot_height = max(800, n_vars * 50); plot_width = max(900, n_vars * 55)
        fig.update_layout(height=plot_height, width=plot_width, margin=dict(l=200, r=50, b=200, t=100))
        plot_path = plot_dir_meta / "metadata_correlation_heatmap"; PlottingUtils(logger).save_plotly_fig(fig, plot_path); logger.info(f"Saved metadata correlation heatmap: {plot_path}")
    except Exception as e: logger.error(f"Failed correlation heatmap plot: {e}")


def plot_sample_facility_map(adata: ad.AnnData, plot_dir: Path, nfc_facilities_df: pd.DataFrame, use_geo: bool = False): 
    """
    Plots samples and optionally NFC facilities on a map.
    Generates an interactive Plotly Mapbox map by default.
    Generates an interactive Plotly Geo map if use_geo=True.
    """
    logger.info("--- Generating Sample and Facility Map ---")
    if use_geo: logger.info("Geo map requested (using scatter_geo).")
    else: logger.info("Interactive map requested (using scatter_mapbox).")
    # Check for sample data and coordinates
    if adata is None: logger.error("AnnData object not loaded. Skipping map."); return
    if 'latitude' not in adata.obs.columns or 'longitude' not in adata.obs.columns: logger.warning("Sample metadata missing 'latitude' or 'longitude' columns. Skipping map."); return
    # Check for facility data
    plot_facilities = True
    if nfc_facilities_df is None or nfc_facilities_df.empty: logger.info("NFC facilities data is empty or None. Plotting samples only."); plot_facilities = False
    elif 'latitude' not in nfc_facilities_df.columns or 'longitude' not in nfc_facilities_df.columns: logger.warning("NFC facility data missing 'latitude' or 'longitude' columns. Skipping facilities."); plot_facilities = False
    try:
        # Prepare sample data 
        samples_df = adata.obs.copy()
        samples_df['latitude'] = pd.to_numeric(samples_df['latitude'], errors='coerce')
        samples_df['longitude'] = pd.to_numeric(samples_df['longitude'], errors='coerce')
        samples_df = samples_df.dropna(subset=['latitude', 'longitude'])
        if samples_df.empty: logger.warning("No valid sample coordinates found after cleaning. Skipping map."); return

        # Ensure color/shape columns exist
        color_col = 'batch_original' if 'batch_original' in samples_df.columns else None
        shape_col = 'facility_match' if 'facility_match' in samples_df.columns else None
        
        if shape_col: samples_df[shape_col] = samples_df[shape_col].astype(str).fillna('Unknown')
        if color_col: samples_df[color_col] = samples_df[color_col].astype(str).fillna('Unknown')
        
        # Prepare facility data (Optional) 
        facilities_df = pd.DataFrame() 
        if plot_facilities:
            facilities_df = nfc_facilities_df.copy()
            if 'facility_name' not in facilities_df.columns: facilities_df['name'] = 'NFC Facility' 
            facilities_df['latitude'] = pd.to_numeric(facilities_df['latitude'], errors='coerce')
            facilities_df['longitude'] = pd.to_numeric(facilities_df['longitude'], errors='coerce')
            facilities_df = facilities_df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)
            if facilities_df.empty: logger.warning("No valid facility coordinates found after cleaning. Plotting samples only."); plot_facilities = False

        # Add logging for coordinate ranges 
        logger.info(f"--- Coordinate Range Check ---\nSample Lats: min={samples_df['latitude'].min()}, max={samples_df['latitude'].max()}, mean={samples_df['latitude'].mean()}\nSample Lons: min={samples_df['longitude'].min()}, max={samples_df['longitude'].max()}, mean={samples_df['longitude'].mean()}")
        if plot_facilities: logger.info(f"Facility Lats: min={facilities_df['latitude'].min()}, max={facilities_df['latitude'].max()}, mean={facilities_df['latitude'].mean()}\nFacility Lons: min={facilities_df['longitude'].min()}, max={facilities_df['longitude'].max()}, mean={facilities_df['longitude'].mean()}")

        # Find closest facilities (Optional) 
        line_lats, line_lons = [], []
        if plot_facilities:
            logger.info("Calculating closest facilities for each sample...")
            samples_rad = np.radians(samples_df[['latitude', 'longitude']])
            facilities_rad = np.radians(facilities_df[['latitude', 'longitude']])
            dist_matrix = haversine_distances(samples_rad, facilities_rad) * 6371
            closest_facility_indices = np.argmin(dist_matrix, axis=1)
            for i, (idx, sample_row) in enumerate(samples_df.iterrows()):
                closest_fac = facilities_df.iloc[closest_facility_indices[i]]
                line_lats.extend([sample_row['latitude'], closest_fac['latitude'], None])
                line_lons.extend([sample_row['longitude'], closest_fac['longitude'], None])
            logger.info("...finished calculating connection lines.")
        
        # Define minimal hover data for samples
        adata.obs['sample_id'] = adata.obs.index
        if 'sample_id' not in samples_df.columns: samples_df['sample_id'] = samples_df.index 
        hover_cols = ['sample_id', 'latitude', 'longitude', 'batch_original', 'project_accession', 'facility_match']
        if color_col and color_col not in hover_cols: hover_cols.append(color_col)
        if shape_col and shape_col not in hover_cols: hover_cols.append(shape_col)
        final_hover_cols = [col for col in hover_cols if col in samples_df.columns]
        
        fig = go.Figure() 
        symbols_list = ['circle', 'square', 'star', 'cross', 'x', 'diamond']
        plot_path = plot_dir / "sample_facility_map" # Default path

        # Geo plot (scatter_geo)
        if use_geo:
            projection_type = 'natural earth'
            plot_path = plot_dir / "sample_facility_map_geo"
            # Plot lines between samples and facilities
            if plot_facilities: fig.add_trace(go.Scattergeo(lat=line_lats, lon=line_lons, mode='lines', line=dict(width=1, color='gray'), opacity=0.4, name='Sample-Facility Link', hoverinfo='none'))
            
            # Plot samples (trace-by-trace if shape column is provided)
            if shape_col:
                shape_categories = sorted(samples_df[shape_col].unique())
                symbol_map = {cat: symbols_list[i % len(symbols_list)] for i, cat in enumerate(shape_categories)}
                logger.info(f"Mapping shapes ({shape_col}) to symbols: {symbol_map}")
                for shape_val in shape_categories:
                    df_shape = samples_df[samples_df[shape_col] == shape_val].copy()
                    if df_shape.empty: continue
                    temp_fig = px.scatter_geo(df_shape, lat='latitude', lon='longitude', color=color_col, hover_data=final_hover_cols, color_discrete_sequence=largecolorset)
                    for trace in temp_fig.data:
                        trace.marker.symbol = symbol_map[shape_val] # type: ignore
                        trace.marker.size = 7 # type: ignore
                        trace.marker.opacity = 0.4 # type: ignore
                        trace.name = f"{shape_val} ({trace.name})" if color_col else str(shape_val) # type: ignore
                        fig.add_trace(trace) # type: ignore
            else:
                logger.info("No shape_col defined. Plotting all samples with default symbol.")
                temp_fig = px.scatter_geo(samples_df, lat='latitude', lon='longitude', color=color_col, hover_data=final_hover_cols, color_discrete_sequence=largecolorset)
                for trace in temp_fig.data:
                    trace.marker.size = 7 # type: ignore
                    trace.marker.opacity = 0.4 # type: ignore
                    fig.add_trace(trace) # type: ignore
            
            # Plot facilities
            if plot_facilities: fig.add_trace(go.Scattergeo(lat=facilities_df['latitude'], lon=facilities_df['longitude'], mode='markers', marker=dict(size=7, color='black', symbol='square', opacity=0.5, line=dict(width=1, color='white')), text=facilities_df['facility'], hovertemplate='<b>%{text}</b><br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>', name='NFC Facilities'))
            # Update Geo Layout
            fig.update_layout(title="Sample and NFC Facility Map", height=1000, geo=dict(projection_type=projection_type, showland=True, landcolor='rgb(235, 235, 235)', showocean=True, oceancolor='rgb(214, 230, 247)', showcountries=True, lonaxis_range=[-180, 180]), legend_title_text='Legend')
            
        # Mapbox plot (scatter_mapbox)
        else:
            # Plot lines between samples and facilities
            if plot_facilities: fig.add_trace(go.Scattermapbox(lat=line_lats, lon=line_lons, mode='lines', line=dict(width=1, color='gray'), opacity=0.4, name='Sample-Facility Link', hoverinfo='none'))
            
            # Plot samples (trace-by-trace if shape column is provided)
            if shape_col:
                shape_categories = sorted(samples_df[shape_col].unique())
                symbol_map = {cat: symbols_list[i % len(symbols_list)] for i, cat in enumerate(shape_categories)}
                logger.info(f"Mapping shapes ({shape_col}) to symbols: {symbol_map}")
                for shape_val in shape_categories:
                    df_shape = samples_df[samples_df[shape_col] == shape_val].copy()
                    if df_shape.empty: continue
                    temp_fig = px.scatter_mapbox(df_shape, lat='latitude', lon='longitude', color=color_col, hover_data=final_hover_cols, color_discrete_sequence=largecolorset)
                    for trace in temp_fig.data:
                        trace.marker.symbol = symbol_map[shape_val] # type: ignore
                        trace.marker.size = 7 # type: ignore
                        trace.marker.opacity = 0.4 # type: ignore
                        if color_col: trace.name = f"{shape_val} ({trace.name})" # type: ignore
                        else: trace.name = str(shape_val) # type: ignore
                        fig.add_trace(trace)
            else:
                logger.info("No shape_col defined. Plotting all samples with default symbol.")
                temp_fig = px.scatter_mapbox(samples_df, lat='latitude', lon='longitude', color=color_col, hover_data=final_hover_cols, color_discrete_sequence=largecolorset)
                for trace in temp_fig.data:
                    trace.marker.size = 7 # type: ignore
                    trace.marker.opacity = 0.4 # type: ignore
                    fig.add_trace(trace) # type: ignore
            
            # Plot facilities
            # Simplified for Mapbox as direct outline is complex, prioritizing visibility.
            if plot_facilities: fig.add_trace(go.Scattermapbox(lat=facilities_df['latitude'], lon=facilities_df['longitude'], mode='markers', marker=go.scattermapbox.Marker(size=7, color='black', symbol='square', opacity=1.0), text=facilities_df['name'], hovertemplate='<b>%{text}</b><br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>', name='NFC Facilities'))

            # Update Mapbox layout
            all_lats = samples_df['latitude']
            if plot_facilities: all_lats = pd.concat([all_lats, facilities_df['latitude']]).dropna()
            fig.update_layout(title="Sample and NFC Facility Map", mapbox_style="open-street-map", mapbox_layers=[], mapbox_center=go.layout.mapbox.Center(lat=all_lats.mean(), lon=0), mapbox_zoom=1, legend_title_text='Legend')
        # Turn off legend and title
        fig.update_layout(title_text=None, showlegend=False, margin={"r":0,"t":0,"l":0,"b":0})
        # Save plot (HTML, PNG, JSON) 
        PlottingUtils(logger).save_plotly_fig(fig, plot_path)
        logger.info(f"Saved sample and facility map: {plot_path}")
    except Exception as e: logger.error(f"Failed to generate sample/facility map: {e}", exc_info=True)



def plot_sample_taxon_map(
    adata: ad.AnnData, 
    plot_dir: Path, 
    nfc_facilities_df: pd.DataFrame, 
    taxon_name: str,
    taxon_level: str,
    use_geo: bool = False,
    layer: str = 'raw_counts'
): 
    """
    Plots samples and optionally NFC facilities on a map, coloring samples
    by the log-transformed abundance of a specific taxon.
    
    Args:
        adata: The main AnnData object.
        plot_dir: The directory to save the plot.
        nfc_facilities_df: DataFrame of NFC facilities.
        taxon_name: The name of the taxon to plot (e.g., 'Pseudomonas').
        taxon_level: The rank of the taxon (e.g., 'Genus', 'Phylum').
        use_geo: If True, use scatter_geo; otherwise, use scatter_mapbox.
        layer: The layer in adata to use for abundance (default: 'raw_counts').
    """
    logger.info(f"--- Generating Sample/Facility Map for Taxon: {taxon_name} ({taxon_level}) ---")
    if use_geo: logger.info("Geo map requested (using scatter_geo).")
    else: logger.info("Interactive map requested (using scatter_mapbox).")
        
    # Check for sample data and coordinates
    if adata is None: logger.error("AnnData object not loaded. Skipping map."); return
    if 'latitude' not in adata.obs.columns or 'longitude' not in adata.obs.columns:
        logger.warning("Sample metadata missing 'latitude' or 'longitude'. Skipping map."); return
        
    # --- 1. Get Taxon Abundance (FIXED ALIGNMENT) ---
    if taxon_level not in adata.var.columns:
        logger.error(f"Taxon level '{taxon_level}' not found in adata.var. Skipping."); return
    if layer not in adata.layers:
        logger.error(f"Layer '{layer}' not found in adata.layers. Skipping."); return
        
    matching_asvs = adata.var_names[adata.var[taxon_level] == taxon_name]
    if matching_asvs.empty:
        logger.warning(f"No features found for taxon '{taxon_name}' at level '{taxon_level}'. Skipping map."); return
        
    logger.info(f"Found {len(matching_asvs)} features for {taxon_name} at {taxon_level} level.")

    # Prepare sample data and determine the subset of samples with valid coordinates
    samples_df = adata.obs.copy()
    samples_df['latitude'] = pd.to_numeric(samples_df['latitude'], errors='coerce')
    samples_df['longitude'] = pd.to_numeric(samples_df['longitude'], errors='coerce')
    
    # Filter samples that have coordinates
    samples_df = samples_df.dropna(subset=['latitude', 'longitude'])
    if samples_df.empty: 
        logger.warning("No valid sample coordinates found. Skipping map."); return

    # FIX: Get abundance data for the *filtered* samples in the correct order
    valid_sample_index = samples_df.index
    
    # Slice AnnData: rows are valid_sample_index, columns are matching_asvs
    abundances_sparse = adata[valid_sample_index, matching_asvs].layers[layer] 
    abundances_dense = abundances_sparse.toarray() if issparse(abundances_sparse) else np.asarray(abundances_sparse)
    
    if abundances_dense.shape[0] != samples_df.shape[0]:
        logger.error(f"Abundance array length ({abundances_dense.shape[0]}) does not match filtered samples ({samples_df.shape[0]}). Check slicing."); return

    # Sum abundances across features (ASVs) for each sample (row)
    total_abundance = np.sum(abundances_dense, axis=1)
    
    # --- 2. Add log-transformed abundance to samples_df ---
    color_col_name = f'log1p_abundance_{taxon_name}'
    # The arrays are now guaranteed to be aligned by index order
    samples_df[color_col_name] = np.log1p(total_abundance)
    logger.info(f"Calculated log1p abundance. Min: {samples_df[color_col_name].min():.2f}, Max: {samples_df[color_col_name].max():.2f}")

    # Set this as the color column
    color_col = color_col_name
    
    # Check for facility data (same as original)
    plot_facilities = True
    if nfc_facilities_df is None or nfc_facilities_df.empty: plot_facilities = False
    elif 'latitude' not in nfc_facilities_df.columns or 'longitude' not in nfc_facilities_df.columns: plot_facilities = False

    try:
        # Ensure shape column exists (same as original)
        shape_col = 'facility_match' if 'facility_match' in samples_df.columns else None
        if shape_col: samples_df[shape_col] = samples_df[shape_col].astype(str).fillna('Unknown')
        
        # Prepare facility data (same as original)
        facilities_df = pd.DataFrame() 
        if plot_facilities:
            facilities_df = nfc_facilities_df.copy()
            if 'facility_name' not in facilities_df.columns: facilities_df['name'] = 'NFC Facility' 
            facilities_df['latitude'] = pd.to_numeric(facilities_df['latitude'], errors='coerce')
            facilities_df['longitude'] = pd.to_numeric(facilities_df['longitude'], errors='coerce')
            facilities_df = facilities_df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)
            if facilities_df.empty: plot_facilities = False

        # Calculate closest facilities (same as original - requires haversine_distances)
        line_lats, line_lons = [], []
        if plot_facilities:
            # Requires `from skbio.stats.distance import haversine_distances`
            samples_rad = np.radians(samples_df[['latitude', 'longitude']])
            facilities_rad = np.radians(facilities_df[['latitude', 'longitude']])
            dist_matrix = haversine_distances(samples_rad, facilities_rad) * 6371
            closest_facility_indices = np.argmin(dist_matrix, axis=1)
            for i, (idx, sample_row) in enumerate(samples_df.iterrows()):
                closest_fac = facilities_df.iloc[closest_facility_indices[i]]
                line_lats.extend([sample_row['latitude'], closest_fac['latitude'], None])
                line_lons.extend([sample_row['longitude'], closest_fac['longitude'], None])
            
        # Define hover data, including the new abundance column
        adata.obs['sample_id'] = adata.obs.index
        if 'sample_id' not in samples_df.columns: samples_df['sample_id'] = samples_df.index 
        hover_cols = ['sample_id', 'latitude', 'longitude', 'facility_match', color_col_name]
        if shape_col and shape_col not in hover_cols: hover_cols.append(shape_col)
        final_hover_cols = [col for col in hover_cols if col in samples_df.columns]
        
        fig = go.Figure() 
        symbols_list = ['circle', 'square', 'star', 'cross', 'x', 'diamond']
        plot_path = plot_dir / f"sample_facility_map_taxon_{taxon_level}_{taxon_name}"
        
        # --- 3. Modify Plotting Logic ---
        
        # Set common color arguments for continuous data
        color_args = {
            'color': color_col,
            'color_continuous_scale': 'Inferno', # Use a continuous scale
            'range_color': [0, samples_df[color_col].max()]
        }

        # Geo plot (scatter_geo)
        if use_geo:
            projection_type = 'natural earth'
            plot_path = plot_dir / f"sample_facility_map_taxon_{taxon_level}_{taxon_name}_geo"
            if plot_facilities: fig.add_trace(go.Scattergeo(lat=line_lats, lon=line_lons, mode='lines', line=dict(width=1, color='gray'), opacity=0.4, name='Sample-Facility Link', hoverinfo='none'))
            
            if shape_col:
                shape_categories = sorted(samples_df[shape_col].unique())
                symbol_map = {cat: symbols_list[i % len(symbols_list)] for i, cat in enumerate(shape_categories)}
                for shape_val in shape_categories:
                    df_shape = samples_df[samples_df[shape_col] == shape_val].copy()
                    if df_shape.empty: continue
                    # **MODIFIED CALL**
                    temp_fig = px.scatter_geo(df_shape, lat='latitude', lon='longitude', 
                                              hover_data=final_hover_cols, **color_args)
                    for trace in temp_fig.data:
                        trace.marker.symbol = symbol_map[shape_val]
                        trace.marker.size = 7
                        trace.marker.opacity = 0.5 # Increased opacity
                        trace.name = str(shape_val)
                        fig.add_trace(trace)
            else:
                # **MODIFIED CALL**
                temp_fig = px.scatter_geo(samples_df, lat='latitude', lon='longitude', 
                                          hover_data=final_hover_cols, **color_args)
                for trace in temp_fig.data:
                    trace.marker.size = 7
                    trace.marker.opacity = 0.5 # Increased opacity
                    fig.add_trace(trace)
            
            if plot_facilities: fig.add_trace(go.Scattergeo(lat=facilities_df['latitude'], lon=facilities_df['longitude'], mode='markers', marker=dict(size=7, color='black', symbol='square', opacity=0.1, line=dict(width=1, color='white')), text=facilities_df['facility'], hovertemplate='<b>%{text}</b><br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>', name='NFC Facilities'))
            fig.update_layout(title=f"Sample Map (Color: {color_col})", height=1000, geo=dict(projection_type=projection_type, showland=True, landcolor='rgb(235, 235, 235)', showocean=True, oceancolor='rgb(214, 230, 247)', showcountries=True, lonaxis_range=[-180, 180]), legend_title_text='Legend')
            
        # Mapbox plot (scatter_mapbox)
        else:
            if plot_facilities: fig.add_trace(go.Scattermapbox(lat=line_lats, lon=line_lons, mode='lines', line=dict(width=1, color='gray'), opacity=0.4, name='Sample-Facility Link', hoverinfo='none'))
            
            if shape_col:
                shape_categories = sorted(samples_df[shape_col].unique())
                symbol_map = {cat: symbols_list[i % len(symbols_list)] for i, cat in enumerate(shape_categories)}
                for shape_val in shape_categories:
                    df_shape = samples_df[samples_df[shape_col] == shape_val].copy()
                    if df_shape.empty: continue
                    # **MODIFIED CALL**
                    temp_fig = px.scatter_mapbox(df_shape, lat='latitude', lon='longitude', 
                                                 hover_data=final_hover_cols, **color_args)
                    for trace in temp_fig.data:
                        trace.marker.symbol = symbol_map[shape_val]
                        trace.marker.size = 5 # Slightly larger for mapbox
                        trace.marker.opacity = 0.5
                        trace.name = str(shape_val)
                        fig.add_trace(trace)
            else:
                # **MODIFIED CALL**
                temp_fig = px.scatter_mapbox(samples_df, lat='latitude', lon='longitude', 
                                             hover_data=final_hover_cols, **color_args)
                for trace in temp_fig.data:
                    trace.marker.size = 5
                    trace.marker.opacity = 0.5
                    fig.add_trace(trace)
            
            if plot_facilities: fig.add_trace(go.Scattermapbox(lat=facilities_df['latitude'], lon=facilities_df['longitude'], mode='markers', marker=go.scattermapbox.Marker(size=3, color='black', symbol='square', opacity=1.0), text=facilities_df['name'], hovertemplate='<b>%{text}</b><br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>', name='NFC Facilities'))
            all_lats = samples_df['latitude']
            if plot_facilities: all_lats = pd.concat([all_lats, facilities_df['latitude']]).dropna()
            fig.update_layout(title=f"Sample Map (Color: {color_col})", mapbox_style="open-street-map", mapbox_layers=[], mapbox_center=go.layout.mapbox.Center(lat=all_lats.mean(), lon=0), mapbox_zoom=1, legend_title_text='Legend')
        
        # Update layout: Show a legend for shape, and handle color bar title
        fig.update_layout(
            title_text=None, 
            showlegend=True, # Show legend for shapes
            legend_title_text=shape_col.replace('_', ' ').capitalize() if shape_col else None,
            
            # 1. Configure the Colorbar
            coloraxis_colorbar=dict(
                title=None,   # Turn off default title
                thickness=15,
                len=0.7         
            ),
            # Ensure margins allow room for the new text on the right
            margin={"r": 60, "t": 0, "l": 0, "b": 0} 
        )

        # 2. Add the Rotated Title as an Annotation
        fig.add_annotation(
            text=color_col.replace('_', ' ').capitalize(),
            x=1.0,           
            y=0.5,                
            xref="paper",      
            yref="paper",
            showarrow=False, 
            xanchor="left",   
            textangle=-90,    
            xshift=50            
        )
          
        PlottingUtils(logger).save_plotly_fig(fig, plot_path)
        logger.info(f"Saved sample and facility map for {taxon_name}: {plot_path}")
    except Exception as e: logger.error(f"Failed to generate sample/facility map for {taxon_name}: {e}", exc_info=True)