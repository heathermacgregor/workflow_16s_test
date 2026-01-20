# ==================================================================================== #

# Standard Imports
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

# Third-Party Imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Local Imports
from workflow_16s.visualization import create_colordict

# ==================================================================================== #

logger = logging.getLogger("workflow_16s")

# ==================================================================================== #

def add_sample_count_annotation(fig: go.Figure, n_pts: int):
    """Add a sample count annotation to the bottom-right corner of the plot."""
    fig.add_annotation(text=f"n = {n_pts}", xref="x domain", yref="y domain",
                       x=0.99, y=0.01, xanchor="right", yanchor="bottom",
                       showarrow=False, font=dict(size=12, color="black"),
                       bgcolor="rgba(255,255,255,0.4)")
    return fig
    
def configure_map(fig: go.Figure, center_lon: Optional[float] = None, 
                  center_lat: Optional[float] = None, 
                  projection_scale: Optional[int] = None, 
                  projection: str = 'natural earth'):
    """Configure the map settings for a Scattergeo plot."""
    fig.update_geos(projection_type=projection, resolution=50, showcoastlines=True, 
                    coastlinecolor="#b5b5b5", showland=True, landcolor="#e8e8e8",
                    showlakes=True, lakecolor="#fff", showrivers=True, rivercolor="#fff")
    if center_lon is not None and center_lat is not None and projection_scale is not None:
        fig.update_geos(center=dict(lon=center_lon, lat=center_lat),
                        projection_scale=projection_scale)
    return fig


def update_layout(fig, map_style='open-street-map'):
    if map_style == 'open-street-map': fig.update_layout(map_style="open-street-map")
    elif map_style == 'usgs':
        fig.update_layout(map_style="white-bg", 
                          map_layers=[{"below": 'traces', "sourcetype": "raster", 
                                       "sourceattribution": "United States Geological Survey",
                                       "source": ["https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"]}])
    fig.update_layout(height=1000, margin=dict(l=0,r=0,t=0,b=0),
                      legend=dict(yanchor="top", y=1, xanchor="left", x=1.02))
    return fig
    
def save(fig, file_stem):
    fig.write_html(f"{file_stem}.html")
    fig.write_json(f"{file_stem}.json")
    fig.write_image(f"{file_stem}.png", scale=2)
    
# ==================================================================================== #

def plot_facility_with_samples_geo(facility_row: pd.Series, nearby_samples: pd.DataFrame,
                                   output_dir: Union[str, Path] = "facility_plots_geo",
                                   max_distance_km: float = 50, show: bool = False,
                                   projection: str = 'natural earth',
                                   map_style: str = "usgs") -> Tuple[str, str]:
    """Plot a single facility with nearby samples using Scattergeo, including 
    distance circle and detailed hover info.
    
    Args:
        facility_row:    Series with facility data including 'latitude' and 
                         'longitude' fields.
        nearby_samples:  DataFrame with nearby samples including 'latitude_deg', 
                         'longitude_deg', and distance fields.
        output_dir:      Directory to save the output files.
        max_distance_km: Maximum distance in km for the distance circle.
        show:            Whether to display the plot interactively.
        projection:      Map projection type for Scattergeo.
        map_style:       Map style, either 'open-street-map' or 'usgs'.
        
    Returns:
        html_path: Path to the saved HTML file.
        json_path: Path to the saved JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    facility_id = facility_row.get('facility', facility_row.name)
    safe_id = str(facility_id).replace(" ", "_").replace("/", "_")
    file_stem = output_dir / f"facility_{safe_id}_geo"

    fig = go.Figure()

    # Facility marker
    fig.add_trace(go.Scattergeo(name='Facility', lat=[facility_row['lat']],
                                lon=[facility_row['lon']],
                                text=[f"{facility_id}\n{facility_row.get('country','')}"],
                                hovertemplate='<b>%{text}</b><extra></extra>',
                                marker=dict(size=15, color='red', symbol='star')))
    
    def _hovertext(row):
        sample_col = "#sampleid" if "#sampleid" in row.index else row.index[0]
        return (f"ID = {row[sample_col]}\n"
                f"Dataset = {row.get('dataset', '')}\n"
                f"Dataset Name = {row.get('dataset_name', '')}\n"
                f"facility_match = {row.get('facility_match', '')}\n"
                f"nuclear_contamination_status = {row.get('nuclear_contamination_status', '')}\n"
                f"({row['lat']:.4f}, {row['lon']:.4f})\n"
                f"Distance: {row['facility_distance_km']:.4f} km\n"
                f"Match: {row['facility_match']}")
        
    # Nearby samples
    if not nearby_samples.empty:
        hover_text = nearby_samples.apply(lambda row: _hovertext(row), axis=1)
        fig.add_trace(go.Scattergeo(name='Samples', lat=nearby_samples['lat'],
                                    lon=nearby_samples['lon'], text=hover_text, 
                                    hovertemplate='<b>%{text}</b><extra></extra>',
                                    marker=dict(size=8, color='blue', opacity=0.7)))

    # Distance circle
    circle_points = 50
    EARTH_RADIUS_KM = 6371
    lat = facility_row['latitude']
    lon = facility_row['longitude']
    angles = np.linspace(0, 2*np.pi, circle_points)
    d = max_distance_km / EARTH_RADIUS_KM
    fig.add_trace(go.Scattergeo(name=f"{max_distance_km} km radius",
                                lat=lat + (d * 180/np.pi) * np.sin(angles),
                                lon=lon + (d * 180/np.pi) * np.cos(angles) / np.cos(lat * np.pi/180),
                                mode='lines', line=dict(color='red', width=2, dash='dash'),
                                hoverinfo='skip'))
    
    fig = add_sample_count_annotation(fig, len(nearby_samples))
    fig = configure_map(fig, facility_row['lon'], facility_row['late'], 
                        50, projection)
    fig = update_layout(fig, map_style)
    save(fig, file_stem)
    if show: fig.show()

    return f"{file_stem}.html", f"{file_stem}.json"

# ==================================================================================== #

def plot_all_facilities_and_samples_geo(metadata: pd.DataFrame,
                                        nfc_facilities_data: Optional[pd.DataFrame] = None,
                                        color_col: str = "dataset_name",
                                        output_dir: Union[str, Path] = "all_facilities_geo",
                                        show: bool = False, projection: str = 'natural earth',
                                        map_style: str = "usgs") -> Tuple[Optional[go.Figure], Optional[dict]]:
    """Plot all samples and optionally all NFC facilities.
    
    Args:
        metadata:            DataFrame with sample metadata including 'lat' 
                             and 'lon' columns.
        nfc_facilities_data: Optional DataFrame with NFC facility data including 
                             'latitude' and 'longitude' columns.
        color_col:           Column in metadata to use for color coding samples.
        output_dir:          Directory to save the output files.
        show:                Whether to display the plot interactively.
        projection:          Map projection type for Scattergeo.
        map_style:           Map style, either 'open-street-map' or 'usgs'.
    
    Returns:
        fig:       The Plotly Figure object.
        colordict: Dictionary mapping unique values in color_col to colors.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    file_stem = output_dir / f"all_facilities_map.{color_col}"

    metadata = metadata.copy()
    # Clean your main samples DataFrame 
    lat_is_valid = metadata['lat'].between(-90, 90)
    lon_is_valid = metadata['lon'].between(-180, 180)
    metadata_cleaned = metadata[lat_is_valid & lon_is_valid]

    # Clean your facilities DataFrame 
    if nfc_facilities_data is not None:
        lat_is_valid_fac = nfc_facilities_data['lat'].between(-90, 90)
        lon_is_valid_fac = nfc_facilities_data['lon'].between(-180, 180)
        nfc_facilities_data_cleaned = nfc_facilities_data[lat_is_valid_fac & lon_is_valid_fac]
    else:
        nfc_facilities_data_cleaned = None
        if color_col not in metadata: metadata[color_col] = 'other'
    metadata[color_col] = metadata[color_col].fillna('other').replace('', 'other')
    metadata[color_col] = metadata[color_col].astype(str)
    colordict = create_colordict(metadata[color_col])
    logger.info(f"Color mapping for '{color_col}': {colordict}")

    fig = go.Figure()
    
    # Sample points
    def _hovertext(row):
        sample_col = "#sampleid" if "#sampleid" in row.index else row.index[0]
        return (f"ID = {row[sample_col]}\n"
                f"Dataset = {row.get('dataset', '')}\n"
                f"Dataset Name = {row.get('dataset_name', '')}\n"
                f"facility_match = {row.get('facility_match', '')}\n"
                f"nuclear_contamination_status = {row.get('nuclear_contamination_status', '')}\n"
                f"({row['lat']:.4f}, {row['lon']:.4f})\n"
                f"Distance: {row['facility_distance_km']:.4f} km\n"
                f"Match: {row['facility_match']}")
        
    hover_text = metadata.apply(lambda row: _hovertext(row), axis=1)
    fig.add_trace(go.Scattergeo(name='Samples', lat=metadata['lat'],
                                lon=metadata['lon'], text=hover_text, 
                                hovertemplate="<b>%{text}</b><extra></extra>",
                                marker=dict(size=3, opacity=0.5, 
                                            color=metadata[color_col].map(colordict))))
    def _get_text(row):
        return f"{row['facility']}\n{row.get('country','')}\n{row.get('data_source', '')}"
    
    # Facility points
    if nfc_facilities_data is not None and not nfc_facilities_data.empty:
        facilities = nfc_facilities_data.dropna(subset=['lat','lon']).copy()
        facility_text = facilities.apply(lambda row: _get_text(row), axis=1)
        fig.add_trace(go.Scattergeo(name='Facilities', lat=facilities['lat'],
                                    lon=facilities['lon'], text=facility_text,
                                    hovertemplate="<b>%{text}</b><extra></extra>",
                                    marker=dict(size=3, opacity=0.25, color='black', 
                                                symbol='star', 
                                                line=dict(width=0, color='yellow'))))
    
    fig = add_sample_count_annotation(fig, len(metadata))
    fig = configure_map(fig, projection=projection)
    fig = update_layout(fig, map_style)
    save(fig, file_stem)
    if show: fig.show()
    return fig, colordict

# ==================================================================================== #

def plot_all_facilities_and_fetched_samples_geo(fetched_samples: pd.DataFrame,
                                                nfc_facilities_data: Optional[pd.DataFrame] = None,
                                                color_col: str = "study_accession",
                                                output_dir: Union[str, Path] = "all_facilities_fetched_samples_geo",
                                                show: bool = False,
                                                projection: str = 'natural earth',
                                                map_style: str = "usgs") -> Tuple[Optional[go.Figure], Optional[dict]]:
    """Plot all fetched ENA biosamples and optionally all NFC facilities.
    
    Args:
        fetched_samples:     DataFrame with fetched sample metadata including 
                             'latitude_deg' and 'longitude_deg' columns.
        nfc_facilities_data: Optional DataFrame with NFC facility data including 
                             'latitude' and 'longitude' columns.
        color_col:           Column in fetched_samples to use for color coding samples.
        output_dir:          Directory to save the output files.
        show:                Whether to display the plot interactively.
        projection:          Map projection type for Scattergeo.
        map_style:           Map style, either 'open-street-map' or 'usgs'.
        
    Returns:
        fig:       The Plotly Figure object.
        colordict: Dictionary mapping unique values in color_col to colors.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    file_stem = output_dir / f"all_facilities_fetched_samples_map.{color_col}"

    fetched_samples = fetched_samples.copy()
    
    # Check for coordinate columns and rename if necessary for compatibility
    if 'lat' not in fetched_samples.columns or 'lon' not in fetched_samples.columns:
        if 'latitude' in fetched_samples.columns and 'longitude' in fetched_samples.columns:
            fetched_samples = fetched_samples.rename(columns={'latitude': 'lat', 
                                                              'longitude': 'lon'})
        else:
            logger.error("Fetched samples DataFrame is missing coordinate columns ('latitude'/'longitude' or 'lat'/'lon'). Cannot create plot.")
            return None, None

    # Handle cases where the color column might be missing
    if color_col not in fetched_samples.columns:
        logger.warning(f"Color column '{color_col}' not found in fetched samples. Defaulting to a single color.")
        fetched_samples[color_col] = 'Fetched Sample'
    
    fetched_samples[color_col] = fetched_samples[color_col].fillna('N/A').replace('', 'N/A')
    fetched_samples[color_col] = fetched_samples[color_col].astype(str)
    
    colordict = create_colordict(fetched_samples[color_col])
    logger.info(f"Color mapping for '{color_col}': {colordict}")
    
    fig = go.Figure()

    def _get_text(row):
        return f"{row['facility']}<br>{row.get('country','')}<br>{row.get('data_source', '')}"
    
    # Add facility points to the plot
    if nfc_facilities_data is not None and not nfc_facilities_data.empty:
        facilities = nfc_facilities_data.dropna(subset=['lat','lon']).copy()
        facility_text = facilities.apply(lambda row: _get_text(row), axis=1)
        fig.add_trace(go.Scattergeo(name='Facilities', lat=facilities['lat'],
                                    lon=facilities['lon'], text=facility_text,
                                    hovertemplate="<b>%{text}</b><extra></extra>",
                                    marker=dict(size=3, opacity=0.25, color='black', 
                                                symbol='star')))
        
    # Add fetched biosample points to the plot
    def _hovertext(row):
        sample_id = row.get('sample_accession', row.get('run_accession', row.name))
        return (f"ID: {sample_id}<br>"
                f"Study: {row.get('study_title', 'N/A')} ({row.get('study_accession', 'N/A')})<br>"
                f"Info: {row.get('title', 'N/A')}<br>"
                f"Source: {row.get('envo_biome_0', 'N/A')} - "
                f"{row.get('envo_biome_1', 'N/A')} - "
                f"{row.get('envo_biome_2', 'N/A')}<br>"
                f"Coords: ({row['lat']:.4f}, {row['lon']:.4f})")

    hover_text = fetched_samples.apply(_hovertext, axis=1)
    fig.add_trace(go.Scattergeo(name='Samples', lat=fetched_samples['lat'],
                                lon=fetched_samples['lon'], text=hover_text,
                                hovertemplate="<b>%{text}</b><extra></extra>",
                                marker=dict(size=3, opacity=0.25, 
                                            color=fetched_samples[color_col].map(colordict))))
    
    fig = add_sample_count_annotation(fig, len(fetched_samples))
    fig = configure_map(fig, projection=projection)
    fig = update_layout(fig, map_style)
    save(fig, file_stem)
    if show: fig.show()
    return fig, colordict