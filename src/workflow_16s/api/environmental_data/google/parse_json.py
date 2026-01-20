# ==================================================================================== #

import csv
import json
import logging
import os
import warnings
from datetime import datetime

import ee
from rich.progress import Progress

from workflow_16s.utils.progress import get_progress_bar 

# ==================================================================================== #

logger = logging.getLogger("workflow_16s")

# ==================================================================================== #

def format_date_for_gee(date_str):
    """Parses a date string from common formats and returns 'YYYY-MM-DD'.
    Returns None if the format is not recognized.
    """
    # List of common date formats to try
    formats_to_try = [
        '%Y-%m-%d',        # 2023-12-25
        '%m/%d/%Y',        # 12/25/2023
        '%d-%b-%Y',        # 25-Dec-2023
        '%Y/%m/%d',        # 2023/12/25
        '%d %b %Y',        # 25 Dec 2023
        '%Y%m%d',          # 20231225
    ]
    for fmt in formats_to_try:
        try:
            # Attempt to parse the date string
            dt_object = datetime.strptime(str(date_str), fmt)
            return dt_object.strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            continue # Try the next format
    logger.warning(f"Could not parse date: '{date_str}'. Unrecognized format.")
    return None


def parse_json_file(file_path):
    """Parses a JSON file and returns its contents as a Python dictionary.
    Returns None if the file is not found or invalid.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.error(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error: The file '{file_path}' is not a valid JSON file.")
        return None

def read_existing_entries(file_path):
    """Reads a TSV file and returns a set of existing (lat, lon, date) entries
    to avoid re-processing them.
    """
    existing_entries = set()
    try:
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            # Check for header before skipping
            if os.path.getsize(file_path) > 0:
                next(reader) # Skip header
            # The columns are lat, lon, date, ...
            for row in reader:
                if row: # Make sure row is not empty
                    existing_entries.add((row[0], row[1], row[2]))
    except (FileNotFoundError, StopIteration):
        # The file doesn't exist yet or is empty, which is fine.
        pass
    return existing_entries


def get_band_values(asset_id, bands, point, date, scale=30):
    """
    Gets band/property values and asset type from a GEE asset for a given point and date.
    """
    asset_type = None
    try:
        # Proactively get asset metadata to determine its type.
        asset_info = ee.data.getAsset(asset_id)
        asset_type = asset_info['type']

        if asset_type == 'IMAGE_COLLECTION':
            collection = ee.ImageCollection(asset_id)
            start_date = ee.Date(date)
            end_date = start_date.advance(1, 'day')
            filtered_collection = collection.filterDate(start_date, end_date).filterBounds(point)
            image_to_sample = filtered_collection.first()
            if image_to_sample is None or image_to_sample.getInfo() is None:
                return None, asset_type
            selected_bands = image_to_sample.select(bands)
            band_values = selected_bands.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=scale
            ).getInfo()
            return band_values, asset_type

        elif asset_type == 'IMAGE':
            logger.info(f"   - Note: Asset '{asset_id}' is a single Image. Ignoring date filter.")
            image_to_sample = ee.Image(asset_id)
            if image_to_sample is None or image_to_sample.getInfo() is None:
                return None, asset_type
            selected_bands = image_to_sample.select(bands)
            band_values = selected_bands.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=scale
            ).getInfo()
            return band_values, asset_type
        
        elif asset_type == 'TABLE':
            logger.info(f"   - Note: Asset '{asset_id}' is a Table. Finding nearest feature.")
            collection = ee.FeatureCollection(asset_id)
            start_date = ee.Date(date)
            end_date = start_date.advance(1, 'day')
            # Filter by date and location. This works even if the table has no time property.
            filtered_collection = collection.filterDate(start_date, end_date).filterBounds(point)
            feature = filtered_collection.first()
            if feature is None or feature.getInfo() is None:
                return None, asset_type
            # Extract the requested properties (which are the 'bands' for a table)
            properties = feature.get(bands).getInfo()
            return properties, asset_type

        else:
            logger.warning(f"   - Skipping asset '{asset_id}' because it is a {asset_type}, not a supported raster or table type.")
            return None, asset_type

    except ee.EEException as e:
        # This catches errors during GEE API calls (e.g., asset not found, invalid bands)
        logger.error(f"   - GEE Error for asset '{asset_id}': {e}")
        return None, asset_type
    except Exception as e:
        # This catches other potential errors (e.g., network issues)
        logger.error(f"   - A general error occurred for asset {asset_id}: {e}")
        return None, asset_type

# ==================================================================================== #

if __name__ == "__main__":
    # --- Logging Configuration ---
    LOG_FILE = 'gee_processing.log'
    # Clear the log file at the beginning of the run if it exists.
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=LOG_FILE,
        filemode='w' # 'w' to overwrite the file each time
    )
    # Capture and log warnings, including DeprecationWarning
    logging.captureWarnings(True)
    warnings.simplefilter('always', DeprecationWarning)
    warnings.simplefilter('always', FutureWarning)

    # Suppress GEE's own verbose logging to keep our log file clean.
    ee_logger = logging.getLogger('ee')
    ee_logger.setLevel(logging.WARNING)
    # ---------------------------

    # --- Example Usage ---
    # Create a dummy config for demonstration
    # You MUST replace 'your-gee-project' with your actual GEE project ID
    # You MUST have a 'credentials.json' file for Google Drive access
    from workflow_16s.config import get_config # type: ignore

    config = get_config()
    
    import pandas as pd
    SAMPLES_PATH = '/usr2/people/macgregor/amplicon/test/data/merged/metadata/final_metadata.tsv'
    # 1. Load the data
    samples_df = pd.read_csv(SAMPLES_PATH, sep='\t', low_memory=False)
    logger.info(f"Initial shape: {samples_df.shape}")

    # 2. First, handle NaN-to-None conversion for all OTHER property columns.
    samples_df = samples_df.astype(object).where(pd.notna(samples_df), None)
    
    # 3. Find a valid date in each row, searching all columns if 'collection_date' is empty.
    def find_date_in_row(row):
        # Prioritize 'collection_date' if it exists and is parsable
        if 'collection_date' in row.index and row['collection_date'] is not None:
            if format_date_for_gee(row['collection_date']):
                return row['collection_date']
        
        # Otherwise, search all columns in the row
        for value in row:
            if value is not None:
                if format_date_for_gee(value):
                    return value # Return the first parsable date found
        return None

    logger.info("Searching for a valid date in each row...")
    samples_df['found_date'] = samples_df.apply(find_date_in_row, axis=1)
    
    # 4. Perform strict cleaning on coordinates and the newly found date.
    samples_df['latitude_deg'] = pd.to_numeric(samples_df['latitude_deg'], errors='coerce')
    samples_df['longitude_deg'] = pd.to_numeric(samples_df['longitude_deg'], errors='coerce')
    samples_df.dropna(subset=['latitude_deg', 'longitude_deg', 'found_date'], inplace=True)
    
    samples_df = samples_df[samples_df['latitude_deg'].between(-90, 90)]
    samples_df = samples_df[samples_df['longitude_deg'].between(-180, 180)]

    # 5. Reset the index and select/rename columns for final processing.
    logger.info("Resetting the index...")
    samples_df.reset_index(inplace=True, drop=True)
    
    samples_df.rename(
        columns={'latitude_deg': 'lat', 'longitude_deg': 'lon', 'found_date': 'date'},
        inplace=True
    )
    
    cols_to_keep = ['lat', 'lon', 'date']
    samples_df = samples_df[cols_to_keep]
    
    # 6. Immediately verify the result.
    logger.info("\n--- Verifying data AFTER cleaning and date processing ---")
    import io
    buf = io.StringIO()
    samples_df.info(buf=buf)
    logger.info(buf.getvalue())

    logger.info(f"Using {len(samples_df)} cleaned locations.")
    samples_df.dropna(inplace=True)
    samples_df.drop_duplicates(inplace=True)
    locations_and_dates = samples_df.to_dict(orient='records')

    JSON_FILE = '/usr2/people/macgregor/amplicon/workflow_16s/src/workflow_16s/api/environmental_data/google/resources/asset_metadata.json'
    OUTPUT_TSV_FILE = '/usr2/people/macgregor/amplicon/workflow_16s/src/workflow_16s/api/environmental_data/google/resources/earth_engine_data.tsv'
    # ------------------------------

    # Initialize the Earth Engine API
    try:
        ee.Initialize(project='wired-day-365517')
        logger.info("Successfully initialized Google Earth Engine.")
    except Exception as e:
        logger.error(f"Error initializing Earth Engine: {e}")
        logger.error("Please make sure you have authenticated with 'earthengine authenticate'")
        exit()

    existing_entries = read_existing_entries(OUTPUT_TSV_FILE)
    logger.info(f"Found {len(existing_entries)} existing location-date entries. They will be skipped.")

    asset_data = parse_json_file(JSON_FILE)
    if not asset_data:
        exit()

    with open(OUTPUT_TSV_FILE, 'a', newline='') as f, get_progress_bar() as progress:
        writer = csv.writer(f, delimiter='\t')

        # Write the new header row only if the file is new/empty
        if f.tell() == 0:
            writer.writerow(['latitude', 'longitude', 'date', 'asset_name', 'asset_type', 'band', 'value', 'band_description', 'band_units'])

        
        task = progress.add_task("[cyan]Processing Locations...", total=len(locations_and_dates))

        for location in locations_and_dates:
            progress.update(task, advance=1)
            lat, lon, original_date_str = location['lat'], location['lon'], location['date']
            
            # Format the date for GEE
            gee_date_str = format_date_for_gee(original_date_str)
            if gee_date_str is None:
                logger.warning(f"Skipping entry due to invalid date: Lat={lat}, Lon={lon}, Date='{original_date_str}'")
                continue # Skip to the next location

            # Use the original date string for checking existence and for writing to the file
            if (str(lat), str(lon), str(original_date_str)) in existing_entries:
                logger.info(f"Skipping already processed entry: Lat={lat}, Lon={lon}, Date={original_date_str}")
                continue
            
            logger.info(f"Processing: Lat={lat}, Lon={lon}, Date={original_date_str} (Formatted as {gee_date_str})")
            point = ee.Geometry.Point(float(lon), float(lat))

            for asset_name, asset_info in asset_data.items():
                
                if not isinstance(asset_info, dict):
                    logger.warning(f"   - Warning: Skipping asset '{asset_name}' due to malformed data.")
                    continue

                asset_id = asset_info.get('asset_id')
                bands_info = asset_info.get('bands')

                if not asset_id or not bands_info:
                    continue

                bands = list(bands_info.keys())
                
                logger.info(f"   - Requesting asset: {asset_name}")

                try:
                    with warnings.catch_warnings():
                        # Temporarily treat deprecation warnings as exceptions to be caught
                        warnings.simplefilter("error", DeprecationWarning)
                        values, asset_type = get_band_values(asset_id, bands, point, gee_date_str)
                except DeprecationWarning as e:
                    logger.warning(f"   - Skipping asset '{asset_name}' due to DeprecationWarning: {e}")
                    continue # Skip to the next asset in the loop

                asset_type_str = asset_type if asset_type else 'UNKNOWN'

                if values:
                    for band, value in values.items():
                        band_details = bands_info.get(band, {})
                        description = band_details.get('description', 'N/A')
                        
                        if description:
                            description = description.replace('\n', ' ')
                        
                        units = band_details.get('units', 'N/A')
                        
                        # Use original_date_str for the output file to maintain consistency
                        row = [lat, lon, original_date_str, asset_name, asset_type_str, band, value, description, units]
                        writer.writerow(row)
                    logger.info(f"     -> Success: Wrote {len(values)} band(s) to file.")
                else:
                    logger.info(f"     -> No data found or asset was skipped for this date/location.")

    logger.info(f"Processing complete. Data saved to '{OUTPUT_TSV_FILE}'. Log saved to '{LOG_FILE}'.")
    # Final print to console to let user know where to find the log
    print(f"\nProcessing complete. See full logs in '{LOG_FILE}'.")


