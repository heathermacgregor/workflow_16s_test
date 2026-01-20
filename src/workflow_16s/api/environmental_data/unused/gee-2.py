import ee
ee.Authenticate()
ee.Initialize(project='wired-day-365517')
import pandas as pd 
from ee.ee_exception import EEException
from ee.feature import Feature
from ee.featurecollection import FeatureCollection
from ee.geometry import Geometry
from ee.image import Image
from ee.imagecollection import ImageCollection
from ee.reducer import Reducer
from ee.terrain import Terrain


def fetch_environmental_data_gee(df):
    """Fetches environmental data using Google Earth Engine for a DataFrame of points."""
    print("Fetching environmental data via Google Earth Engine...")
    try:
        # Using the high-volume endpoint is good practice for larger jobs
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        print("GEE initialized successfully.")
    except EEException as e:
        print(f"Could not initialize GEE. Please ensure you have authenticated. Error: {e}")
        return df

    # Convert pandas DataFrame to a GEE FeatureCollection
    features = []
    for index, row in df.iterrows():
        geom = Geometry.Point([row['longitude_deg'], row['latitude_deg']])
        feature = Feature(geom, {'sample_id': index})
        features.append(feature)
    
    ee_points = FeatureCollection(features)

    # --- Define GEE Image objects for different datasets ---
    
    # 1. USGS Elevation and Slope
    srtm = Image('USGS/SRTMGL1_003')
    elevation = srtm.select('elevation')
    slope = Terrain.slope(srtm)

    # 2. USDA gSSURGO Soil Properties
    gssurgo = Image('USDA/NRCS/gSSURGO/v0')
    awc = gssurgo.select('awc_0_5')
    ph = gssurgo.select('ph_0_5')

    # 3. SoilGrids Global Soil Data
    soilgrids_ph = Image("projects/soilgrids-isric/soildata/v2/phh2o_0-5cm_mean")
    soilgrids_soc = Image("projects/soilgrids-isric/soildata/v2/soc_0-5cm_mean")
    soilgrids_clay = Image("projects/soilgrids-isric/soildata/v2/clay_0-5cm_mean")
    
    # 4. NOAA PRISM Climate Normals
    prism = Image('OREGONSTATE/PRISM/Norm81m').rename(['ppt', 'tmean'])
    
    # 5. Land Cover (NLCD 2021)
    nlcd = Image('USGS/NLCD_RELEASES/2021_REL/NLCD').select('landcover')
    
    # 6. Atmospheric Pollution (TROPOMI NO2, 1-year mean)
    # Using a recent full year for stable averages.
    no2_collection = ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2') \
        .select('tropospheric_NO2_column_number_density') \
        .filterDate('2023-01-01', '2023-12-31')
    no2_mean = no2_collection.mean()
    
    # 7. Population Density (WorldPop 2020)
    population = Image('WORLDPOP/GP/100m/pop/USA_2020').rename('population_density')
    
    # 8. Distance to Permanent Water (JRC Global Surface Water)
    water_mask = Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').gt(50)
    distance_to_water = water_mask.fastDistanceTransform().sqrt() \
        .multiply(Image.pixelArea().sqrt())

    # --- Combine all layers into a single multi-band image ---
    
    final_image = elevation.addBands(slope).addBands(awc).addBands(ph) \
        .addBands(soilgrids_ph).addBands(soilgrids_soc).addBands(soilgrids_clay) \
        .addBands(prism).addBands(nlcd).addBands(no2_mean) \
        .addBands(population).addBands(distance_to_water)
        
    # Rename bands for clarity
    final_image = final_image.rename([
        'elevation_m', 'slope_deg', 'gssurgo_awc_0_5cm', 'gssurgo_ph_0_5cm',
        'soilgrids_ph', 'soilgrids_soc_g_kg', 'soilgrids_clay_g_kg',
        'prism_ppt_mm', 'prism_tmean_c', 'nlcd_landcover_class',
        'tropomi_no2_mol_m2', 'population_density', 'distance_to_water_m'
    ])

    # Sample the raster values at each point
    sampled_data = final_image.reduceRegions(
        collection=ee_points, reducer=Reducer.mean(), scale=500
    ).getInfo()

    # Process results and merge back into the main DataFrame
    results = {
        feature['properties']['sample_id']: feature['properties']
        for feature in sampled_data['features']
    }
        
    env_df = pd.DataFrame.from_dict(results, orient='index').drop(columns='sample_id', errors='ignore')
    df = df.join(env_df, how='left')
    
    # SoilGrids data has a scale factor
    if 'soilgrids_ph' in df.columns:
        df['soilgrids_ph'] = df['soilgrids_ph'] / 10.0
    
    print("✅ GEE data extraction complete.")
    return df