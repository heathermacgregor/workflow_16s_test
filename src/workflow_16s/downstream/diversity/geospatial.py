# diversity/geospatial.py
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from skbio.stats.distance import mantel, DistanceMatrix
from sklearn.metrics.pairwise import haversine_distances
from workflow_16s.downstream.visualization import PlottingUtils
from workflow_16s.utils.logger import get_logger

logger = get_logger("workflow_16s")
plot_utils = PlottingUtils(logger)

def run_geospatial_decay(adata, dist_matrix, plot_dir_beta: Path):
    """Calculates community dissimilarity vs physical distance (km)."""
    logger.info("--- Starting Geospatial Distance Decay Analysis ---")
    # 1. Prepare Lat/Lon (convert to radians for haversine)
    coords = adata.obs[['latitude', 'longitude']].dropna()
    common_ids = coords.index.intersection(dist_matrix.ids)
    if len(common_ids) < 10: return
    
    coords_rad = np.radians(coords.loc[common_ids].values)
    geo_dist = haversine_distances(coords_rad) * 6371 # Earth radius in km
    
    # 2. Extract distance vectors (flatten upper triangle)
    upper_idx = np.triu_indices(len(common_ids), k=1)
    vec_geo = geo_dist[upper_idx]
    vec_microbe = dist_matrix.filter(common_ids).data[upper_idx]
    
    # 3. Mantel Test
    r, p, _ = mantel(dist_matrix.filter(common_ids), DistanceMatrix(geo_dist, ids=common_ids))
    
    # 4. Plotting
    plot_df = pd.DataFrame({'Physical Distance (km)': vec_geo, 'Community Dissimilarity': vec_microbe})
    fig = px.scatter(plot_df.sample(min(5000, len(plot_df))), x='Physical Distance (km)', y='Community Dissimilarity', 
                     trendline="ols", title=f"Distance Decay Analysis (Mantel r={r:.3f}, p={p:.2e})")
    plot_utils.save_plotly_fig(fig, plot_dir_beta / "geospatial_distance_decay", batch=False)