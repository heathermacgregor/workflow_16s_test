import ee
import pandas as pd
from tqdm import tqdm
import time

def map_biomes():
    # 1. Initialize with explicit project ID
    PROJECT_ID = 'wired-day-365517'
    try:
        ee.Initialize(project=PROJECT_ID)
        print(f"✅ GEE Initialized successfully on project: {PROJECT_ID}")
    except Exception as e:
        print(f"❌ GEE Initialization Error: {e}")
        print("💡 Try running: earthengine authenticate")
        return

    # 2. Load Metadata
    # Path updated to your expanded file
    path = 'project_01/04_analysis/testing_20260212/merged_metadata_environmental.tsv'
    try:
        df = pd.read_csv(path, sep='\t', low_memory=False)
    except FileNotFoundError:
        print(f"❌ Could not find file at {path}")
        return

    # 3. Filter and Validate Coordinates
    # We need Soil + Real Numbers for Lat/Lon
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    
    target_mask = (df['broad_class'] == 'soil') & df['lat'].notna() & df['lon'].notna()
    work_df = df[target_mask].copy()

    if work_df.empty:
        print("🤷 No soil samples with valid numeric coordinates found. Check your lat/lon columns.")
        return

    print(f"🌍 Found {len(work_df)} valid soil samples. Starting spatial join...")

    # 4. RESOLVE Ecoregions (2017)
    eco_fc = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
    
    batch_size = 250  # Smaller batches are more stable for testing
    results = []

    for i in tqdm(range(0, len(work_df), batch_size), desc="GEE Batches"):
        batch = work_df.iloc[i:i+batch_size]
        
        # Build features safely
        features = []
        for idx, row in batch.iterrows():
            # GEE expects [Lon, Lat]
            geom = ee.Geometry.Point([float(row['lon']), float(row['lat'])])
            features.append(ee.Feature(geom, {'run_id': str(row['run_accession'])}))
        
        fc = ee.FeatureCollection(features)
        
        # Spatial Join Logic
        joined_fc = ee.Join.saveFirst(matchKey='eco').apply(
            fc, eco_fc, ee.Filter.intersects(leftField='.geo', rightField='.geo')
        )
        
        try:
            # Get data from cloud
            data = joined_fc.getInfo()['features']
            for feat in data:
                props = feat['properties']
                if 'eco' in props:
                    eco_props = props['eco']['properties']
                    results.append({
                        'run_accession': props['run_id'],
                        'terrestrial_biome': eco_props.get('BIOME_NAME', 'Unknown'),
                        'terrestrial_ecoregion': eco_props.get('ECO_NAME', 'Unknown')
                    })
        except Exception as e:
            print(f"\n⚠️ Batch {i//batch_size} failed: {e}")
            time.sleep(2) # Brief pause before next batch

    # 5. Save Results
    if results:
        res_df = pd.DataFrame(results)
        # Avoid duplicate columns on merge
        cols_to_keep = [c for c in df.columns if c not in ['terrestrial_biome', 'terrestrial_ecoregion']]
        final_df = df[cols_to_keep].merge(res_df, on='run_accession', how='left')
        
        final_df.to_csv(path, sep='\t', index=False)
        print(f"\n✅ Done! Enriched {len(res_df)} samples.")
        print(final_df['terrestrial_biome'].value_counts().head())
    else:
        print("❌ No spatial matches found. Are your coordinates in the right range?")

if __name__ == "__main__":
    map_biomes()