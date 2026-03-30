import scanpy as ad
import pandas as pd

# Load the original massive dataset (before we filtered it)
print("Loading original merged dataset...")
adata = ad.read_h5ad("/usr2/people/macgregor/amplicon/project_01/04_analysis/testing_20260212/merged_samples.h5ad")

# Fill missing text with empty strings so we can search them safely
obs_df = adata.obs.fillna('')

# Convert everything to lowercase for easy searching
text_cols = [col for col in obs_df.columns if obs_df[col].dtype == 'object']

print("Hunting for the word 'lake' in the metadata...")
# Create a boolean mask that is True if 'lake' appears in ANY text column
is_lake_text = pd.Series(False, index=obs_df.index)

for col in text_cols:
    is_lake_text = is_lake_text | obs_df[col].str.lower().str.contains('lake')

# How many did we find?
rescued_lakes = obs_df[is_lake_text]
print(f"\nFound {len(rescued_lakes)} samples with 'lake' explicitly in their metadata!")

# Let's see WHY the spatial script missed them
missing_coords = rescued_lakes[rescued_lakes['lat'] == ''].shape[0]
print(f" - {missing_coords} of these are completely missing lat/lon coordinates.")
print(f" - {len(rescued_lakes) - missing_coords} have coordinates, but they fell outside our lake polygons (Shoreline/Small lake issue).")