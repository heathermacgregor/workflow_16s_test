# workflow_16s/visualization/maps.py

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
from sklearn.metrics.pairwise import haversine_distances 
 
from workflow_16s.downstream.utils import AnalysisUtils
from workflow_16s.visualization.utils import PlottingUtils
from workflow_16s.utils.logger import with_logger

@with_logger
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
    if adata is None: 
        logger.error("AnnData object not loaded. Skipping map."); return
    if 'latitude' not in adata.obs.columns or 'longitude' not in adata.obs.columns: 
        logger.warning("Sample metadata missing 'latitude' or 'longitude' columns. Skipping map."); return
    # Check for facility data
    plot_facilities = True
    if nfc_facilities_df is None or nfc_facilities_df.empty: 
        logger.info("NFC facilities data is empty or None. Plotting samples only."); plot_facilities = False
    elif 'latitude' not in nfc_facilities_df.columns or 'longitude' not in nfc_facilities_df.columns: 
        logger.warning("NFC facility data missing 'latitude' or 'longitude' columns. Skipping facilities."); plot_facilities = False
    try:
        # Prepare sample data 
        samples_df = adata.obs.copy()
        samples_df['latitude'] = pd.to_numeric(samples_df['latitude'], errors='coerce')
        samples_df['longitude'] = pd.to_numeric(samples_df['longitude'], errors='coerce')
        samples_df = samples_df.dropna(subset=['latitude', 'longitude'])
        if samples_df.empty: 
            logger.warning("No valid sample coordinates found after cleaning. Skipping map."); return

        # Ensure color/shape columns exist
        color_col = 'batch_original' if 'batch_original' in samples_df.columns else None
        shape_col = 'facility_match' if 'facility_match' in samples_df.columns else None
        
        if shape_col: samples_df[shape_col] = samples_df[shape_col].astype(str).fillna('Unknown')
        if color_col: samples_df[color_col] = samples_df[color_col].astype(str).fillna('Unknown')
        
        # Prepare facility data (Optional) 
        facilities_df = pd.DataFrame() 
        if plot_facilities:
            facilities_df = nfc_facilities_df.copy()
            if 'facility_name' not in facilities_df.columns: 
                facilities_df['name'] = 'NFC Facility' 
            facilities_df['latitude'] = pd.to_numeric(facilities_df['latitude'], errors='coerce')
            facilities_df['longitude'] = pd.to_numeric(facilities_df['longitude'], errors='coerce')
            facilities_df = facilities_df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)
            if facilities_df.empty: 
                logger.warning("No valid facility coordinates found after cleaning. Plotting samples only."); plot_facilities = False

        # Add logging for coordinate ranges 
        logger.info(f"--- Coordinate Range Check ---\nSample Lats: min={samples_df['latitude'].min()}, max={samples_df['latitude'].max()}, mean={samples_df['latitude'].mean()}\nSample Lons: min={samples_df['longitude'].min()}, max={samples_df['longitude'].max()}, mean={samples_df['longitude'].mean()}")
        if plot_facilities: 
            logger.info(f"Facility Lats: min={facilities_df['latitude'].min()}, max={facilities_df['latitude'].max()}, mean={facilities_df['latitude'].mean()}\nFacility Lons: min={facilities_df['longitude'].min()}, max={facilities_df['longitude'].max()}, mean={facilities_df['longitude'].mean()}")

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
            if plot_facilities: 
                fig.add_trace(go.Scattergeo(lat=line_lats, lon=line_lons, mode='lines', line=dict(width=1, color='gray'), opacity=0.4, name='Sample-Facility Link', hoverinfo='none'))
            
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


@with_logger
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