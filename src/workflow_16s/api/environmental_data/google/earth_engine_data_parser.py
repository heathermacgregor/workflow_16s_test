import json
import csv
import argparse
import os
import pandas as pd

def process_and_describe_data(long_format_path, json_path, output_basename):
    """
    Converts the long-format GEE data to a wide format and generates a
    corresponding data dictionary (column descriptions).

    Args:
        long_format_path (str): Path to the input long-format TSV file.
        json_path (str): Path to the input asset_metadata.json file.
        output_basename (str): Base name for the output files.
    """
    # --- Part 1: Convert long-format data to wide format ---
    wide_output_path = f"{output_basename}_wide.csv"
    print(f"\n--- Part 1: Converting to Wide Format ---")
    
    # Check if the long-format input file exists
    if not os.path.exists(long_format_path):
        print(f"Error: Long-format input file not found at '{long_format_path}'")
        return

    print(f"Reading long-format data from '{long_format_path}'...")
    try:
        df = pd.read_csv(long_format_path, sep='\t')

        # Create a new column for pivoting that combines asset and band names
        df['column_name'] = df['asset_name'] + '_' + df['band']

        # Pivot the table to convert from long to wide format
        print("Pivoting data from long to wide format...")
        wide_df = df.pivot_table(
            index=['latitude', 'longitude', 'date'],
            columns='column_name',
            values='value'
        ).reset_index()
        
        # Clean up the column names that result from pivoting
        wide_df.columns.name = None

        # Save the wide-format data to a CSV file
        wide_df.to_csv(wide_output_path, index=False)
        print(f"Successfully created wide-format data file at '{wide_output_path}'")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{long_format_path}'")
        return
    except Exception as e:
        print(f"An error occurred during the conversion to wide format: {e}")
        return

    # --- Part 2: Generate Column Descriptions ---
    descriptions_output_path = f"{output_basename}_descriptions.csv"
    print(f"\n--- Part 2: Generating Column Descriptions ---")
    
    # Check if the input JSON file exists
    if not os.path.exists(json_path):
        print(f"Error: Input JSON file not found at '{json_path}'")
        return

    print(f"Reading asset metadata from '{json_path}'...")
    try:
        with open(json_path, 'r') as f:
            asset_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading or parsing the JSON file: {e}")
        return

    # Prepare to write the output CSV file for the descriptions
    try:
        with open(descriptions_output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write the header row
            writer.writerow(['column_name', 'description', 'units', 'gee_asset'])

            print("Generating column descriptions...")
            # Add descriptions for the standard index columns
            writer.writerow(['latitude', 'Latitude of the sample location', 'Decimal Degrees', 'N/A'])
            writer.writerow(['longitude', 'Longitude of the sample location', 'Decimal Degrees', 'N/A'])
            writer.writerow(['date', 'The date of the sample or observation', 'YYYY-MM-DD', 'N/A'])
            
            # Iterate through each asset in the JSON file
            for asset_name, asset_info in asset_data.items():
                if not isinstance(asset_info, dict) or 'bands' not in asset_info:
                    continue

                # Iterate through each band within the asset
                bands_info = asset_info.get('bands', {})
                for band_name, band_details in bands_info.items():
                    # Construct the column name as it appears in the wide-format file
                    column_name = f"{asset_name}_{band_name}"
                    
                    # Extract the description and units, providing defaults if missing
                    description = band_details.get('description', 'N/A').replace('\n', ' ')
                    units = band_details.get('units', 'N/A')
                    asset_id = asset_info.get('asset_id', 'N/A')

                    # Write the extracted information as a new row in the CSV
                    writer.writerow([column_name, description, units, asset_id])
                    
        print(f"Successfully created column descriptions file at '{descriptions_output_path}'")

    except IOError as e:
        print(f"Error writing to the output file: {e}")


if __name__ == "__main__":
    # Run the main function with the provided file paths
    process_and_describe_data("/usr2/people/macgregor/amplicon/workflow_16s/src/workflow_16s/api/environmental_data/google/resources/earth_engine_data.tsv", 
                              "/usr2/people/macgregor/amplicon/workflow_16s/src/workflow_16s/api/environmental_data/google/resources/asset_metadata.json", 
                              "/usr2/people/macgregor/amplicon/workflow_16s/src/workflow_16s/api/environmental_data/google/resources/earth_engine_data_parsed")
