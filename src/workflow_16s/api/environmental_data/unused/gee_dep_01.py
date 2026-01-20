# ==================================================================================== #

import os

import ast
import ee
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from scipy.stats import spearmanr, f_oneway
from scipy.spatial import cKDTree

from workflow_16s.api.environmental_data.google import find_global_datasets

# ==================================================================================== #

def process_asset(asset_id):
    """Checks the type of a GEE asset and processes it accordingly.

    Args:
        asset_id (str): The asset ID to process.
    """
    try:
        # 1. Fetch the asset's metadata first
        asset_info = ee.data.getAsset(asset_id)
        asset_type = asset_info['type']
        
        print(f"--- Processing asset: {asset_id} ---")
        print(f"Detected asset type: {asset_type}")

        # 2. Use conditional logic to handle the asset based on its type
        if asset_type == 'Image':
            # Load the asset as an Image
            image = ee.Image(asset_id)
            
            # --- Your Image-specific processing goes here ---
            # For example, get the projection info or clip it to a region.
            print("Processing as a single Image.")
            proj = image.projection().getInfo()
            print(f"Image projection: {proj['crs']}")

        elif asset_type == 'IMAGECOLLECTION':
            # Load the asset as an ImageCollection
            collection = ee.ImageCollection(asset_id)
            
            # --- Your Collection-specific processing goes here ---
            # For example, get the size of the collection or filter it.
            print("Processing as an Image Collection.")
            size = collection.size().getInfo()
            first_image = collection.first() # Get one image for further work
            print(f"Collection contains {size} images.")
            
        elif asset_type == 'FEATURECOLLECTION':
            # Load the asset as a FeatureCollection (for vector data)
            table = ee.FeatureCollection(asset_id)
            
            # --- Your FeatureCollection-specific processing goes here ---
            print("Processing as a Feature Collection (Table).")
            count = table.size().getInfo()
            print(f"Feature Collection contains {count} features.")

        else:
            print(f"Warning: Asset type '{asset_type}' is not supported by this function.")

    except ee.EEException as e:
        print(f"An error occurred with asset '{asset_id}': {e}")

        
def get_earth_engine_embeddings(
    df: pd.DataFrame,
    year: int,
    project_id: str,
    collection_name: str = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
) -> pd.DataFrame:
    """
    Gets Google Earth Engine satellite data for each coordinate from a specified collection.

    Args:
        df: DataFrame with 'latitude' and 'longitude' columns.
        year: The year to filter the image collection.
        project_id: Your Google Cloud Project ID for Earth Engine authentication.
        collection_name: The name of the Earth Engine ImageCollection to use.

    Returns:
        The input DataFrame with a new 'embedding' column containing band values.
    """
    try:
        ee.Initialize(project=project_id)
        print(f"✅ Successfully initialized Earth Engine for project: {project_id}")
    except Exception as e:
        print(f"❌ Error initializing Earth Engine: {e}")
        return df

    print(f"Fetching data from Earth Engine collection: '{collection_name}' for the year {year}...")

    # Fetch the asset's metadata
    asset_info = ee.data.getAsset(collection_name)['type']

    # The metadata is a dictionary; get the value associated with the 'type' key
    asset_type = asset_info

    print(f"The asset '{collection_name}' is of type: {asset_type}")
    
    # Filter the specified collection by date
    collection = ee.ImageCollection(collection_name).filterDate(f'{year}-01-01', f'{year}-12-31')
    
    if collection.size().getInfo() == 0:
        print(f"⚠️ No Earth Engine data found for '{collection_name}' in {year}.")
        return df

    # Create a single composite image for the year
    image = collection.mosaic()
    
    def get_embedding(row):
        """Extracts all band values for a given coordinate."""
        try:
            point = ee.Geometry.Point(row['longitude'], row['latitude'])
            # Use reduceRegion to get all band values at the point
            embedding_dict = image.reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=10).getInfo()
            print(embedding_dict)
            return list(embedding_dict.values())
        except Exception as e:
            # Handle cases where EE might fail for a specific point
            print(f"Could not retrieve data for point ({row['latitude']}, {row['longitude']}): {e}")
            return None
            
    df['embedding'] = df.apply(get_embedding, axis=1)
    return df


def is_valid_embedding(embedding_list):
    """
    Checks if an embedding is a valid, non-empty list of numbers without NaNs.
    Returns True if valid, False otherwise.
    """
    # Ensure the input is a list and is not empty
    if not isinstance(embedding_list, list) or not embedding_list:
        return False
    
    try:
        # Attempt to convert to a float array and check for NaNs.
        # This will fail if the list contains non-numeric types (like strings).
        return not np.isnan(np.array(embedding_list, dtype=np.float64)).any()
    except (ValueError, TypeError):
        # The conversion failed, meaning it contains non-numeric data.
        return False
    
    
def cluster_facilities(df: pd.DataFrame, n_clusters: int = 3, facility_col: str = 'facility') -> pd.DataFrame:
    """Applies K-Means clustering to the embedding vectors."""
    df_embed = df.dropna(subset=['embedding']).copy()
    if df_embed.empty:
        print("No valid embeddings found to perform clustering.")
        return df
    # This checks inside each list for NaNs and keeps only the rows that have no NaNs.
    df_embed = df_embed[df_embed['embedding'].apply(is_valid_embedding)]

    # Now, check again if the DataFrame is empty after the more thorough cleaning
    if df_embed.empty:
        print("No valid embeddings without NaN values found to perform clustering.")
        return df
        
    X = np.array(df_embed['embedding'].tolist())
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df_embed['environment_cluster'] = kmeans.fit_predict(X)
    
    
    # Merge cluster labels back into the original DataFrame
    return df.merge(df_embed[[facility_col, 'environment_cluster']], on=facility_col, how='left')


def find_nearest_facility_info(samples_df: pd.DataFrame, facilities_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each sample, efficiently finds the nearest facility and its info
    (distance and environment cluster).
    """
    if 'latitude_deg' not in samples_df.columns or 'longitude_deg' not in samples_df.columns:
        raise ValueError("Samples DataFrame must contain 'latitude_deg' and 'longitude_deg' columns.")
    
    # Use cKDTree for efficient nearest-neighbor search
    facility_coords = facilities_df[['latitude', 'longitude']].values
    sample_coords = samples_df[['latitude_deg', 'longitude_deg']].values
    
    kdtree = cKDTree(facility_coords)
    distances_deg, indices = kdtree.query(sample_coords, k=1)

    # Add results to the samples DataFrame
    samples_df['nearest_facility_idx'] = indices
    
    # Calculate geodesic distance in kilometers for more accuracy
    distances_km = [
        geodesic(
            (sample_row.latitude_deg, sample_row.longitude_deg),
            facility_coords[sample_row.nearest_facility_idx]
        ).kilometers
        for sample_row in samples_df.itertuples()
    ]
    samples_df['distance_to_nfc_km'] = distances_km
    
    # Map the cluster information from the facilities DataFrame
    cluster_map = facilities_df['environment_cluster'].to_dict()
    samples_df['nearest_facility_cluster'] = samples_df['nearest_facility_idx'].map(cluster_map)
    
    return samples_df.drop(columns=['nearest_facility_idx'])


def link_samples_to_facilities(metadata_df, nfc_clustered_df, alpha_df):
    metadata_with_facility_info = find_nearest_facility_info(metadata_df, nfc_clustered_df)
    return pd.merge(metadata_with_facility_info, alpha_df, left_on='#sampleid', right_index=True)


def plot_correlation_diversity_metric_with_numerical_col(df, output_dir, metric, col: str = 'distance_to_nfc_km', col_name: str = 'Distance'):
    # Analysis A: Correlation with Distance
    corr, p_val_corr = spearmanr(df[col], df[metric])
    print(f"Analysis A ({col_name} vs {metric}): Spearman Correlation={corr:.3f}, P-value={p_val_corr:.4f}")
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df, x=col, y=metric, line_kws={"color":"red"})
    plt.title(f'Distance to Facility vs. {metric.capitalize()} Diversity')
    plt.savefig(f"{output_dir}distance_vs_{metric}.png", dpi=300, bbox_inches='tight')
    plt.close()
  
    
def anova_plot_by_categorical_col(df, output_dir, metric, col: str = 'nearest_facility_cluster', col_name: str = 'Cluster'):
    if df[col].nunique() > 1:
        groups = [group[metric].values for name, group in df.groupby(col)]
        f_stat, p_val_anova = f_oneway(*groups)
        print(f"Analysis B ({col_name} vs {metric}): ANOVA F-statistic={f_stat:.2f}, P-value={p_val_anova:.4f}")
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df, x=col, y=metric, palette='viridis')
        plt.title(f'{col_name} vs. {metric.capitalize()} Diversity')
        plt.xlabel(col_name)
        plt.ylabel(f'{metric.capitalize()} Index')
        plt.savefig(f"{output_dir}{col}_vs_{metric}.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print(f"Only one cluster found for metric '{metric}'. Skipping ANOVA.")
    print(f"Plots for '{metric}' saved to '{output_dir}'")

# ==================================================================================== #

def main():
    """Main function to run the entire analysis workflow."""
    
    # --- Configuration ---
    MY_GCP_PROJECT_ID = 'wired-day-365517'
    YEARS_OF_INTEREST = [2020, 2021, 2022, 2023, 2024]
    
    try:
        ee.Initialize(project=MY_GCP_PROJECT_ID)
        print(f"✅ Successfully initialized Earth Engine for project: {MY_GCP_PROJECT_ID}")
    except Exception as e:
        print(f"❌ Error initializing Earth Engine: {e}")
        return df
    
    # --- File Paths ---
    CATALOG_FILE_PATH = '/usr2/people/macgregor/amplicon/workflow_16s/src/workflow_16s/api/environmental_data/google_earth_engine/resources/catalog.json'
    METADATA_PATH = '/usr2/people/macgregor/amplicon/test/data/merged/metadata/raw/genus.tsv'
    ALPHA_DIVERSITY_PATH = '/usr2/people/macgregor/amplicon/test/results/alpha_diversity/nuclear_contamination_status/raw/genus/alpha_diversity.tsv'
    NFC_FACILITIES_PATH = '/usr2/people/macgregor/amplicon/test/data/nfc/facilities.tsv'
    
    # Specify the Earth Engine collection to use for the main analysis.
    # The script will first list other potential global datasets it finds.
    found_datasets = find_global_datasets(CATALOG_FILE_PATH)
    # --- Step 0: Discover Available Global Datasets ---
    # --- Step 1: Load Data & Perform Cleaning ---
    print("--- Step 1: Loading and cleaning data files ---")
    metadata_df = pd.read_csv(METADATA_PATH, sep='\t', low_memory=False)
    alpha_df = pd.read_csv(ALPHA_DIVERSITY_PATH, sep='\t', low_memory=False, index_col=0)
    nfc_df = pd.read_csv(NFC_FACILITIES_PATH, sep='\t', low_memory=False)
        
    metadata_df = metadata_df[metadata_df['dataset_id'] == 'ENA_PRJEB21351']
    metadata_df_cleaned = metadata_df.dropna(subset=['latitude_deg', 'longitude_deg'])
    metadata_df_cleaned = metadata_df_cleaned[metadata_df_cleaned['latitude_deg'].between(-90, 90)]
    nfc_df = nfc_df.dropna(subset=['latitude', 'longitude'])
    nfc_df = nfc_df[nfc_df['latitude'].between(-90, 90)]
    print(f"Using {len(metadata_df_cleaned)} cleaned samples and {len(nfc_df)} cleaned facilities.")
    # --- Iterate Over Each Found Global Dataset ---
    for dataset in found_datasets:
        EE_COLLECTION = dataset['id']
        collection_type = ee.data.getAsset(EE_COLLECTION)['type']
        print(f"{EE_COLLECTION} - {collection_type}")
        if not collection_type == 'IMAGE_COLLECTION':
            continue
        if EE_COLLECTION == "COPERNICUS/MARINE/SATELLITE_OCEAN_COLOR/V6":
            continue
        for YEAR_OF_INTEREST in YEARS_OF_INTEREST:
            OUTPUT_DIR = f'/usr2/people/macgregor/amplicon/test/results/nfc_impact_analysis/{EE_COLLECTION.replace("/", "_").replace(":", "_")}/{YEAR_OF_INTEREST}/'
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # --- Dynamic Cache Path ---
            sanitized_collection_name = EE_COLLECTION.replace('/', '_').replace(':', '_')
            EMBEDDINGS_CACHE_PATH = f"{OUTPUT_DIR}nfc_facilities_embeddings_{sanitized_collection_name}_{YEAR_OF_INTEREST}.csv"
            CLUSTERS_CACHE_PATH = f"{OUTPUT_DIR}nfc_facilities_embeddings_{sanitized_collection_name}_{YEAR_OF_INTEREST}_clusters.csv"        

            # --- Step 2: Get Embeddings (with Caching) & Cluster Facilities ---
            print("\n--- Step 2: Getting embeddings and clustering facilities ---")
            if os.path.exists(EMBEDDINGS_CACHE_PATH):
                print(f"Embeddings file found at '{EMBEDDINGS_CACHE_PATH}'. Loading from disk.")
                nfc_with_embeddings = pd.read_csv(EMBEDDINGS_CACHE_PATH)
                nfc_with_embeddings['embedding'] = nfc_with_embeddings['embedding'].apply(ast.literal_eval)
            else:
                print(f"Embeddings file not found. Fetching from Google Earth Engine...")
                nfc_with_embeddings = get_earth_engine_embeddings(
                    df=nfc_df,
                    year=YEAR_OF_INTEREST,
                    project_id=MY_GCP_PROJECT_ID,
                    collection_name=EE_COLLECTION
                )
                if 'embedding' in nfc_with_embeddings.columns and not nfc_with_embeddings['embedding'].isnull().all():
                    nfc_with_embeddings.to_csv(EMBEDDINGS_CACHE_PATH, index=False)
                    print(f"Embeddings saved to '{EMBEDDINGS_CACHE_PATH}' for future use.")
                else:
                    print("Failed to fetch any embeddings. Skipping clustering and saving.")
            try:        
                nfc_clustered = cluster_facilities(nfc_with_embeddings, n_clusters=10, facility_col='facility')
                nfc_clustered.to_csv(CLUSTERS_CACHE_PATH, index=False)
                print(f"Clustered facility data saved to {CLUSTERS_CACHE_PATH}")
            except Exception as e:
                print(f"Clustering failed. Ensure embeddings were fetched correctly. {e}")
                continue
            try:
                if nfc_clustered:
                    # --- Step 3: Link Samples to Facilities ---
                    print("\n--- Step 3: Linking samples to nearest facilities ---")
                    final_df = link_samples_to_facilities(metadata_df_cleaned, nfc_clustered, alpha_df)
                    print(final_df.head())
                

                    # --- Step 4: Run Analyses and Generate Plots ---
                    print("\n--- Step 4: Running analyses and generating plots ---")
                    for metric in ['shannon', 'observed_features', 'simpson', 'pielou_evenness', 'heip_evenness']:
                        if metric in final_df.columns:
                            analysis_df = final_df.dropna(subset=['distance_to_nfc_km', metric, 'distance_to_nfc_km'])
                            if analysis_df.empty:
                                print(f"No data available for analysis on metric '{metric}'. Skipping.")
                                continue

                            # Analysis A: Correlation with Distance
                            plot_correlation_diversity_metric_with_numerical_col(analysis_df, OUTPUT_DIR, metric, 'distance_to_nfc_km', 'Distance')

                            # Analysis B: ANOVA by Environment Cluster
                            anova_plot_by_categorical_col(analysis_df, OUTPUT_DIR, metric, 'nearest_facility_cluster', 'Cluster')
                        else:
                            print(f"'{metric}' column not found. Skipping analysis.")
            except:
                print("Downstream failed. Ensure embeddings were fetched correctly.")
                continue
                
    print("\n✅ Workflow complete.")
    
# ==================================================================================== #

if __name__ == "__main__":
    main()