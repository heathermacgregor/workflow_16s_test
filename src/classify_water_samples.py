import sys
import scanpy as ad
import pandas as pd
import numpy as np
import geopandas as gpd
from global_land_mask import globe

def classify_water_bodies_fast(h5ad_path, output_path):
    print(f"Loading {h5ad_path}...")
    adata = ad.read_h5ad(h5ad_path)
    adata.obs_names_make_unique()
    
    # 1. Ultra-fast NaN dropping and coercion
    # Copying only the columns we need to a temporary df to save memory
    obs_df = adata.obs[['lat', 'lon']].copy()
    obs_df['lat'] = pd.to_numeric(obs_df['lat'], errors='coerce')
    obs_df['lon'] = pd.to_numeric(obs_df['lon'], errors='coerce')
    
    # Drop rows with missing coordinates directly
    valid_obs = obs_df.dropna(subset=['lat', 'lon']).copy()
    
    if valid_obs.empty:
        raise ValueError("No valid coordinates (lat/lon) found in the dataset!")

    # Pre-allocate water_type array (defaulting to 'unknown')
    valid_obs['water_type'] = 'unknown'

    # 2. Vectorized Land/Ocean Masking (Millisecond execution)
    print(f"Checking {len(valid_obs)} coordinates against global land mask...")
    lats = valid_obs['lat'].values
    lons = valid_obs['lon'].values
    is_on_land = globe.is_land(lats, lons)
    
    # Assign Oceans
    valid_obs.loc[~is_on_land, 'water_type'] = 'Ocean'

    # 3. Vectorized Spatial Join for Lakes
    land_mask = is_on_land
    if land_mask.any():
        print(f"Checking {land_mask.sum()} land-based samples for lakes...")
        
        # ⚡ THE SPEED UP: Vectorized geometry creation (No Python loops)
        land_samples = valid_obs[land_mask]
        geometries = gpd.points_from_xy(land_samples['lon'], land_samples['lat'])
        gdf_samples = gpd.GeoDataFrame(land_samples, geometry=geometries, crs="EPSG:4326")
        
        # Load lake polygons
        # ⚡ THE SPEED UP & FIX: Load high-res lake polygons directly from Natural Earth
        print("Fetching high-resolution lake polygons from Natural Earth...")
        try:
            # Geopandas can read zipped shapefiles directly over HTTP!
            lakes_url = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_lakes.zip"
            world_lakes = gpd.read_file(lakes_url)
        except Exception as e:
            print(f"\n⚠️ HTTP Download Failed (Firewall issue?): {e}")
            print("Please manually download the file using this command in your terminal:")
            print("wget https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_lakes.zip")
            print("Then update 'lakes_url' in this script to point to your local 'ne_10m_lakes.zip' file.")
            sys.exit(1)
        
        # ⚡ THE SPEED UP: sjoin automatically uses spatial indexing (R-Tree)
        # We only keep the index to map it back instantly
        lake_indices = gdf_samples.sjoin(world_lakes, how="inner", predicate='intersects').index
        valid_obs.loc[lake_indices, 'water_type'] = 'Lake'

    # 4. Map back to original AnnData and Filter (Vectorized)
    # Map the valid results back; anything missing stays as NaN, fill with 'unknown'
    adata.obs['water_type'] = valid_obs['water_type']
    adata.obs['water_type'] = adata.obs['water_type'].fillna('unknown')

    # Filter keeping only Ocean and Lake
    mask = adata.obs['water_type'].isin(['Lake', 'Ocean'])
    filtered_adata = adata[mask].copy()
    
    # 5. Output Results
    counts = filtered_adata.obs['water_type'].value_counts()
    print("\n--- Fast Classification Results ---")
    print(f"Oceans:  {counts.get('Ocean', 0)}")
    print(f"Lakes:   {counts.get('Lake', 0)}")
    print(f"Dropped: {len(adata) - len(filtered_adata)} (Land / Unknown / Missing)")
    
    filtered_adata.write_h5ad(output_path)
    print(f"Filtered h5ad saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        classify_water_bodies_fast(sys.argv[1], sys.argv[2])
    else:
        classify_water_bodies_fast("/usr2/people/macgregor/amplicon/project_01/04_analysis/testing_20260212/merged_samples.h5ad", "/usr2/people/macgregor/amplicon/aquatic/aquatic_only.h5ad")
        print("Usage: python classify_water_samples.py <input.h5ad> <output.h5ad>")
    