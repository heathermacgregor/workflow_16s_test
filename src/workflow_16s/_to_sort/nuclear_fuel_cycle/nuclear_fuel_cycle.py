# ===================================== IMPORTS ====================================== #

# Standard Imports
import logging
import os
import requests
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Third Party Imports
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree
from tqdm import tqdm 

# Local Imports
from workflow_16s.constants import DEFAULT_USER_AGENT, MINDAT_API_KEY, REFERENCES_DIR
from workflow_16s.nuclear_fuel_cycle import dnfsb, mindat, wikipedia, other_databases, utils 
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc
from workflow_16s.figures.figures import largecolorset

# ==================================================================================== #

logger = logging.getLogger("workflow_16s")

# ==================================================================================== #

mindat_columns_to_keep = [
    "facility", "country", "latitude", "longitude", "elements", "refs", "wikipedia", 
    "data_source"
]
wikipedia_columns_to_keep = [
    "facility", "country", "facility_start_year", "facility_end_year", "lat_lon", 
    "location", "data_source", "wikipedia", "wikitable"
]

# ==================================================================================== #

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

def plot_facility_with_samples_geo(
    facility_row: pd.Series,
    nearby_samples: pd.DataFrame,
    output_dir: Union[str, Path] = "facility_plots_geo",
    max_distance_km: float = 50,
    show: bool = False
):
    """
    Plot a single facility with nearby samples using Scattergeo,
    including distance circle and detailed hover info.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    facility_id = facility_row.get('facility', facility_row.name)
    safe_id = str(facility_id).replace(" ", "_").replace("/", "_")
    file_stem = output_dir / f"facility_{safe_id}_geo"

    fig = go.Figure()

    # Facility marker
    fig.add_trace(
        go.Scattergeo(
            lon=[facility_row['longitude_deg']],
            lat=[facility_row['latitude_deg']],
            text=[f"{facility_id}\n{facility_row.get('country','')}"],
            marker=dict(size=15, color='red', symbol='star'),
            name='Facility',
            hovertemplate='<b>%{text}</b><extra></extra>'
        )
    )

    # Nearby samples
    if not nearby_samples.empty:
        sample_col = "#sampleid" if "#sampleid" in nearby_samples.columns else nearby_samples.columns[0]
        hover_text = nearby_samples.apply(lambda row: f"{row[sample_col]}\nLat: {row['latitude_deg']:.4f}\nLon: {row['longitude_deg']:.4f}\nDistance: {row['facility_distance_km']:.4f}\nMatch: {row['facility_match']}", axis=1)
        fig.add_trace(
            go.Scattergeo(
                lon=nearby_samples['longitude_deg'],
                lat=nearby_samples['latitude_deg'],
                text=hover_text,
                marker=dict(size=8, color='blue', opacity=0.7),
                name='Samples',
                hovertemplate='<b>%{text}</b><extra></extra>'
            )
        )

    # Distance circle
    circle_points = 50
    EARTH_RADIUS_KM = 6371
    lat = facility_row['latitude_deg']
    lon = facility_row['longitude_deg']
    angles = np.linspace(0, 2*np.pi, circle_points)
    d = max_distance_km / EARTH_RADIUS_KM
    circle_lat = lat + (d * 180/np.pi) * np.sin(angles)
    circle_lon = lon + (d * 180/np.pi) * np.cos(angles) / np.cos(lat * np.pi/180)
    fig.add_trace(
        go.Scattergeo(
            lon=circle_lon,
            lat=circle_lat,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name=f"{max_distance_km} km radius",
            hoverinfo='skip'
        )
    )

    # Layout
    fig.update_layout(
        title=dict(text=f"Facility {facility_id} and Nearby Samples", x=0.5),
        geo=dict(
            projection_type='natural earth',
            showcountries=True,
            landcolor="#e8e8e8",
            showland=True,
            showcoastlines=True,
            coastlinecolor="#b5b5b5",
        ),
        showlegend=True,
        height=600,
        margin=dict(l=0,r=0,t=50,b=0)
    )

    # Save
    fig.write_html(f"{file_stem}.html")
    fig.write_json(f"{file_stem}.json")
    if show:
        fig.show()

    return f"{file_stem}.html", f"{file_stem}.json"

# ==================================================================================== #

def plot_all_facilities_and_samples_geo(
    metadata: pd.DataFrame,
    nfc_facilities_data: Optional[pd.DataFrame] = None,
    color_col: str = "dataset_name",
    output_dir: Union[str, Path] = "all_facilities_geo",
    show: bool = False
):
    """
    Plot all samples and optionally all NFC facilities using Scattergeo.
    Provides hover info and color coding.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    file_stem = output_dir / "all_facilities_map"

    metadata = metadata.copy()
    metadata[color_col] = metadata.get(color_col, 'other').fillna('other').replace('', 'other')
    colordict = _create_colordict(metadata[color_col])

    fig = go.Figure()

    # Sample points
    hover_text = metadata.apply(lambda row: f"ID = {row.get('#sampleid', row.name)}\nDataaset = {row.get('dataset', '')}\nDataset Name = {row.get('dataset_name', '')}\nfacility_match = {row.get('facility_match', '')}\nnuclear_contamination_status = {row.get('nuclear_contamination_status', '')}\n{row['latitude_deg']:.4f}, {row['longitude_deg']:.4f}", axis=1)
    fig.add_trace(
        go.Scattergeo(
            lon=metadata['longitude_deg'],
            lat=metadata['latitude_deg'],
            text=hover_text,
            marker=dict(size=6, opacity=0.7, color=metadata[color_col].map(colordict)),
            name='Samples',
            hovertemplate="<b>%{text}</b><extra></extra>"
        )
    )

    # Facility points
    if nfc_facilities_data is not None and not nfc_facilities_data.empty:
        facilities = nfc_facilities_data.dropna(subset=['latitude_deg','longitude_deg']).copy()
        facility_text = facilities.apply(lambda row: f"{row['facility']}\n{row.get('country','')}\n{row.get('data_source', '')}", axis=1)
        fig.add_trace(
            go.Scattergeo(
                lon=facilities['longitude_deg'],
                lat=facilities['latitude_deg'],\
                text=facility_text,
                marker=dict(size=12, opacity=0.5, color='black', symbol='star', line=dict(width=0, color='yellow')),
                name='Facilities',
                hovertemplate="<b>%{text}</b><extra></extra>"
            )
        )

    # Layout
    fig.update_layout(
        title=dict(text="All Facilities and Samples", x=0.5),
        geo=dict(
            projection_type='natural earth',
            showcountries=True,
            landcolor="#e8e8e8",
            showland=True,
            showcoastlines=True,
            coastlinecolor="#b5b5b5",
        ),
        showlegend=True,
        height=800,
        margin=dict(l=0,r=0,t=50,b=0)
    )

    # Save
    fig.write_html(f"{file_stem}.html")
    fig.write_json(f"{file_stem}.json")
    if show:
        fig.show()

    return fig, colordict

# ==================================================================================== #

class NFCFacilitiesHandler:
    """Handler for managing Nuclear Fuel Cycle (NFC) facilities data.
    
    This class handles the retrieval, geocoding, and matching of NFC facilities
    with sample metadata based on geographical coordinates.
    
    Attributes:
        config:                 Configuration dictionary containing settings.
        output_dir:             Directory path for output files.
        mindat_api_key:         API key for MinDat API access.
        user_agent:             User agent string for web requests.
        verbose:                Flag for verbose logging.
        databases:              List of database configurations.
        database_names:         Names of enabled databases.
        max_distance_km:        Maximum distance in km for facility matching.
        use_local:              Flag to use locally cached data.
        facilities_output_path: Path to facilities output CSV.
        matches_output_path:    Path to facility matches output TSV.
    """
    def __init__(
        self, 
        config: Dict,        
        output_dir: Optional[Union[str, Path]] = REFERENCES_DIR, 
        mindat_api_key: str = MINDAT_API_KEY,
        user_agent: str = DEFAULT_USER_AGENT
    ):
        """Initialize NFC facilities handler.
        
        Args:
            config:        Configuration dictionary containing NFC facilities settings.
            output_dir:     Output directory path. Defaults to REFERENCES_DIR.
            mindat_api_key: Mindat API key. Defaults to MINDAT_API_KEY.
            user_agent:     User agent string for web requests. Defaults to DEFAULT_USER_AGENT.
        """
        self.config = config
        enabled = self.config.get("nfc_facilities", {}).get("enabled", False) 
        if not enabled:
            return

        self.verbose = self.config.get("verbose", False)
        
        self.databases = self.config.get("nfc_facilities", {}).get("databases", [{'name': "NFCIS"}, {'name': "GEM"}])
        self.database_names = [db['name'] for db in self.databases]
        
        self.max_distance_km = self.config.get("nfc_facilities", {}).get("max_distance_km", 50)

        self.use_local = self.config.get("nfc_facilities", {}).get('use_local', False)

        self.output_dir = Path(output_dir) / "nfc_facilities"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.facilities_output_path = Path(self.output_dir) / 'facilities_raw.tsv'
        self.facilities_geocoded_output_path = Path(self.output_dir) / 'facilities.tsv'
        self.load_existing_geocoded = True
        self.matches_output_path = Path(self.output_dir) / f"facility_matches_{self.max_distance_km}km.tsv"
        self.matches_only_output_path = Path(self.output_dir) / f"facility_matches_only_{self.max_distance_km}km.tsv"
        
        self.mindat_api_key = mindat_api_key
        self.user_agent = user_agent
        
    def log(self, msg):
        """Log message with debug level if verbose mode is enabled."""
        return (lambda msg: logger.debug(msg)) if self.verbose else (lambda *_: None)
        
    def run(self, metadata: pd.DataFrame):
        """Execute the complete NFC facilities processing pipeline."""
        if self.use_local and self.facilities_output_path.exists():
            df = pd.read_csv(self.facilities_output_path, sep='\t')
        else:
            df = self._get_geocoded_data()
        toggle = False   
        if self.use_local and self.matches_output_path.exists() and toggle:   
            updated_metadata = pd.read_csv(self.matches_output_path, sep='\t')
        else:
            logger.info("Matching facilities with samples")
            updated_metadata = self._match_facilities_with_samples(facilities_df=df, samples_df=metadata)
    
        return df, updated_metadata

    def _get_geocoded_data(self):
        """Retrieve and geocode facilities data from configured databases."""
        # Load data from existing file if enabled
        if self.facilities_geocoded_output_path.exists() and self.load_existing_geocoded:
            logger.info("Loading geocoded data")
            df = pd.read_csv(self.facilities_geocoded_output_path, sep='\t')
            return df
        df = self._get_data()
        df = self._geocode(df)
        self.nfc_facilities = df
        return df
        
    def _get_data(self):
        """Aggregate facilities data from all enabled databases."""
        database_dfs = []
        if "DFNSB" in self.database_names:
            dfnsb_results = dfnsb.load_dnfsb_facilities(output_dir=self.output_dir)
            database_dfs.append(dfnsb_results)
        if "GEM" in self.database_names or "NFCIS" in self.database_names:
            other_databases_results = other_databases.load_nfc_facilities(config=self.config, output_dir=self.output_dir)
            database_dfs.append(other_databases_results)
        if "MinDat" in self.database_names:
            mindat_results, _ = mindat.world_uranium_mines(self.config, self.mindat_api_key, self.output_dir)
            database_dfs.append(mindat_results[mindat_columns_to_keep])
        if "Wikipedia" in self.database_names:
            wikipedia_results = wikipedia.world_nfc_facilities(config=self.config, output_dir=self.output_dir)
            database_dfs.append(wikipedia_results[wikipedia_columns_to_keep])
        
        dfs = [df for df in database_dfs if isinstance(df, pd.DataFrame)]            
        facilities_df = pd.concat(dfs, axis=0)
        facilities_df = facilities_df.sort_values(by='facility')
        facilities_df = facilities_df.reindex(sorted(facilities_df.columns), axis=1)
        facilities_df['country'] = facilities_df['country'].str.replace('USA', 'United States of America')
        if self.output_dir:
            facilities_df.to_csv(self.facilities_output_path, sep='\t', index=False)
        return facilities_df

    def _geocode(self, df: pd.DataFrame):
        """Geocode facility locations using nominatim OpenStreetMap.
        
        Args:
            df: Input DataFrame with facility and country information.
            
        Returns:
            pd.DataFrame: DataFrame with added latitude and longitude columns.
        """        
        # Prepare geocoding
        df['__query__'] = df['facility'].fillna('') + ', ' + df['country'].fillna('')
        unique_queries = df['__query__'].unique()
    
        # Geocode unique queries with progress
        coords = {}
        with get_progress_bar() as progress:
            task = progress.add_task( 
                _format_task_desc("Geocoding unique locations"), 
                total=len(unique_queries)
            )
            for q in unique_queries:
                coords[q] = utils._geocode_query(q, self.user_agent)
                time.sleep(1) 
                progress.update(task, advance=1)
    
        # Map coords back to DataFrame
        df['latitude_osm']  = df['__query__'].map(lambda q: coords[q][0])
        df['longitude_osm'] = df['__query__'].map(lambda q: coords[q][1])
        df.drop(columns='__query__', inplace=True)
        
        df['latitude_deg'] = df['latitude'].combine_first(df['latitude_osm'])
        df['longitude_deg'] = df['longitude'].combine_first(df['longitude_osm'])
        
        if self.output_dir:
            df.to_csv(self.facilities_geocoded_output_path, sep='\t', index=False)
            
        return df

    # ==================================================================================== #
    
    def _match_facilities_with_samples(
        self,
        facilities_df: pd.DataFrame,
        samples_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Match samples with nearby facilities, generate per-facility and global maps,
        and save matched results.
        """
        # Run existing matching logic
        matched_df = self._match_facilities_with_locations(facilities_df, samples_df)
    
        # Directory for plots
        per_facility_dir = self.output_dir / "per_facility_maps"
        all_facility_dir = self.output_dir / "all_facilities_map"
        per_facility_dir.mkdir(exist_ok=True)
        all_facility_dir.mkdir(exist_ok=True)
    
        # Build mapping of facility_id -> nearby samples
        facility_samples_map = {}
        for _, row in matched_df[matched_df['facility_match'] == True].iterrows():
            facility_id = row.get('facility', None)
            if facility_id is None:
                continue
            facility_samples_map.setdefault(facility_id, []).append(row)
        
        # Generate per-facility plots
        for _, facility in facilities_df.iterrows():
            facility_id = facility.get('facility', None)
            if facility_id is None:
                continue
            nearby_samples = pd.DataFrame(facility_samples_map.get(facility_id, []))
            if nearby_samples.empty:
                continue
            try:
                plot_facility_with_samples_geo(
                    facility_row=facility,
                    nearby_samples=nearby_samples,
                    output_dir=per_facility_dir,
                    max_distance_km=self.max_distance_km,
                    show=False
                )
            except Exception as e:
                logger.warning(f"Error creating plot for facility {facility_id}: {e}")
    
        # Generate global map of all samples and all facilities
        plot_all_facilities_and_samples_geo(
            metadata=matched_df,
            nfc_facilities_data=facilities_df,
            color_col="dataset_name",
            output_dir=all_facility_dir,
            show=False
        )
    
        # Save matched results
        result_cols = ['#sampleid','latitude_deg','longitude_deg','facility_match','facility','facility_distance_km']
        matched_df[result_cols] = matched_df[result_cols].copy()  # ensure all columns exist
        matched_df.to_csv(self.matches_output_path, sep='\t', index=False)
    
        return matched_df

    def _match_facilities_with_locations(
        self,
        facilities: pd.DataFrame,
        samples: pd.DataFrame,
        max_distance_km: float = 100
    ) -> pd.DataFrame:
        """
        Match locations to nearby facilities within a specified distance threshold.
        Handles missing coordinates by preserving original rows.
        Returns a DataFrame with facility_match column indicating successful matches.
        Automatically detects sample ID columns like '#sampleid', 'sample_id', or 'sample id'
        (case-insensitive) and standardizes them to '#sampleid'.
        """
        EARTH_RADIUS_KM = 6371
    
        # ------------------------
        # Detect and standardize sample ID column
        # ------------------------
        sample_id_cols = [col for col in samples.columns if col.lower().replace(" ", "") in {"#sampleid", "sampleid", "sample_id", "sample id"}]
        if sample_id_cols:
            logger.info(sample_id_cols)
            # Use the first matching column
            samples = samples.rename(columns={sample_id_cols[0]: "#sampleid"})
        else:
            # If not found, create a default sample_id from index
            samples["sample_id"] = samples.index.astype(str)
    
        # Create copy with original index preserved as a column
        samples = samples.copy()
        samples["original_index"] = samples.index
    
        # Split into valid/invalid samples
        valid_mask = samples[['latitude_deg', 'longitude_deg']].notnull().all(axis=1)
        valid_samples = samples[valid_mask]
        invalid_samples = samples[~valid_mask]
        logger.info(len(valid_samples))
        logger.info(len(invalid_samples))
    
        # Prepare facilities data
        valid_facilities = facilities.dropna(subset=["latitude_deg", "longitude_deg"]).copy()
        valid_facilities = valid_facilities.rename(columns={
            "latitude_deg": "latitude_deg_facility",
            "longitude_deg": "longitude_deg_facility",
            "country": "country_facility"
        })
    
        # Initialize results list
        results = []
    
        # Process valid samples if we have both valid samples and facilities
        if not valid_samples.empty and not valid_facilities.empty:
            # Convert coordinates to radians for BallTree
            sample_coords = np.radians(valid_samples[["latitude_deg", "longitude_deg"]])
            facility_coords = np.radians(valid_facilities[["latitude_deg_facility",
                                                           "longitude_deg_facility"]])
    
            # Build BallTree for efficient distance queries
            tree = BallTree(facility_coords, metric="haversine")
            radius_radians = max_distance_km / EARTH_RADIUS_KM
    
            # Query for facilities within radius
            indices, distances = tree.query_radius(
                sample_coords,
                r=radius_radians,
                return_distance=True
            )
    
            # Build results for each valid sample
            with get_progress_bar() as progress:
                task_desc = "Processing valid samples"
                task_id = progress.add_task(_format_task_desc(task_desc), total=len(valid_samples))
                for i, (sample_idx, sample) in enumerate(valid_samples.iterrows()):
                    sample_results = []
                    sample_distances = distances[i] * EARTH_RADIUS_KM  # Convert to km
                    sample_indices = indices[i]
    
                    # Inside the valid_samples processing loop:
                    closest_distance = None
                    closest_facility = None
                    
                    for j, distance_km in zip(sample_indices, sample_distances):
                        if closest_distance is None or distance_km < closest_distance:
                            closest_distance = distance_km
                            closest_facility = valid_facilities.iloc[j]
                    
                    if closest_facility is not None:
                        result_row = {
                            **sample.to_dict(),
                            **closest_facility.to_dict(),
                            "facility_distance_km": closest_distance,
                            "facility_match": True
                        }
                    else:
                        result_row = {
                            **sample.to_dict(),
                            **{col: np.nan for col in valid_facilities.columns},
                            "facility_distance_km": np.nan,
                            "facility_match": False
                        }
                    
                    results.append(result_row)  # Only one result per sample
                    progress.update(task_id, advance=1)
    
        # Add invalid samples to results (no match possible)
        for _, sample in invalid_samples.iterrows():
            result_row = {
                **sample.to_dict(),
                **{col: np.nan for col in valid_facilities.columns},
                "facility_distance_km": np.nan,
                "facility_match": False
            }
            results.append(result_row)
    
        # Create DataFrame and restore original index
        result_df = pd.DataFrame(results)
        result_df.set_index("original_index", inplace=True)
        result_df.index.name = None
    
        return result_df

# ==================================================================================== #

def update_nfc_facilities_data(
    config: Dict, 
    metadata: pd.DataFrame, 
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function to update NFC facilities data and match with samples.
    
    Args:
        config:   Configuration dictionary.
        metadata: Sample metadata DataFrame with coordinates.
        verbose:  Verbosity flag.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - NFC facilities DataFrame
            - Updated metadata with facility matches
    """
    handler = NFCFacilitiesHandler(config=config)
    nfc_facilities, updated_metadata = handler.run(metadata=metadata)
    if verbose:
        logger.info(nfc_facilities)
        logger.info(updated_metadata)
    return nfc_facilities, updated_metadata
    
