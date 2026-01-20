import pandas as pd
import csv
import ast
import glob
import os

def parse_observations_tsv(file_path):
    """
    Parses a single tab-separated values (TSV) file with observation data.

    This function reads the specified TSV file, correctly interpreting a
    string-formatted dictionary in the 'attributes' column and converting
    each row into a Python dictionary object. It's designed to be robust
    against errors in the attributes column.

    Args:
        file_path (str): The path to the .tsv file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a
              row of data from the file.
    """
    parsed_data = []
    print(f"--> Parsing file: {file_path}")
    try:
        with open(file_path, mode='r', encoding='utf-8') as tsv_file:
            reader = csv.DictReader(tsv_file, delimiter='\t')
            for row in reader:
                # ast.literal_eval safely parses the string representation
                # of the dictionary in the 'attributes' column.
                if 'attributes' in row and row['attributes']:
                    try:
                        row['attributes'] = ast.literal_eval(row['attributes'])
                    except (ValueError, SyntaxError):
                        # If parsing fails, keep it as a string but log a warning.
                        print(f"    Warning: Could not parse attributes string in row for {row.get('observation_id')}")
                parsed_data.append(row)
    except FileNotFoundError:
        print(f"    Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"    An unexpected error occurred while parsing {file_path}: {e}")
    return parsed_data

def process_and_filter_data(file_paths):
    """
    Loads, combines, filters, and cleans data from multiple TSV files.

    Args:
        file_paths (list): A list of file paths to process.

    Returns:
        pandas.DataFrame: A cleaned and filtered DataFrame.
    """
    # 1. Parse all specified files and combine them
    all_data = []
    for path in file_paths:
        all_data.extend(parse_observations_tsv(path))

    if not all_data:
        print("No data was parsed. Exiting.")
        return pd.DataFrame()

    print("\n--> Converting parsed data into a DataFrame...")
    df = pd.DataFrame(all_data)

    # Convert numeric and date columns, coercing errors to 'Not a Number/Time'
    # This prevents crashes if data is malformed.
    numeric_cols = ['latitude', 'longitude', 'sample_latitude', 'sample_longitude']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Filter rows based on proximity
    print("--> Filtering data based on location proximity...")
    # Keep rows where lat/lon are not specified (cannot be filtered)
    # or where the observation is within 0.1 degrees of the sample.
    df = df[
        (df['latitude'].isnull()) |
        (df['longitude'].isnull()) |
        (df['sample_latitude'].isnull()) |
        (df['sample_longitude'].isnull()) |
        (df['latitude'].sub(df['sample_latitude']).abs() <= 0.1) |
        (df['longitude'].sub(df['sample_longitude']).abs() <= 0.1)
    ]

    # 3. Filter NASA_POWER data by date
    print("--> Filtering NASA_POWER data to match collection dates...")
    # Using errors='coerce' will turn any unparseable dates into NaT (Not a Time)
    df['time_date'] = pd.to_datetime(df['time'], errors='coerce', utc=True).dt.date
    df['collection_date'] = pd.to_datetime(df['sample_collection_date'], errors='coerce', utc=True).dt.date
    
    # This condition correctly keeps all non-NASA data, AND the NASA data where dates match.
    df = df[(df['dataset'] != 'NASA_POWER') | (df['time_date'] == df['collection_date'])]

    # 4. Clean up the DataFrame
    print("--> Cleaning up final DataFrame...")
    # Flatten the attributes dictionary into separate columns for easier analysis
    if 'attributes' in df.columns:
        # handle non-dict attributes gracefully
        attributes_df = df['attributes'].apply(lambda x: x if isinstance(x, dict) else {}).apply(pd.Series)
        attributes_df = attributes_df.add_prefix('attr_') # Add prefix to avoid name collisions

        # Drop the original attributes column and join the flattened one
        df = df.drop('attributes', axis=1).join(attributes_df)
    
    # Remove columns that are entirely empty
    df = df.dropna(axis=1, how='all')
    
    # Clean up temporary date columns
    df = df.drop(columns=['time_date', 'collection_date'], errors='ignore')

    return df

def create_sample_summary(df):
    """
    Pivots the detailed observation DataFrame to create a summary DataFrame
    with one row per sample and variables as columns.

    Args:
        df (pandas.DataFrame): The cleaned DataFrame from process_and_filter_data.

    Returns:
        pandas.DataFrame: A summary DataFrame describing each sample.
    """
    print("\n--> Creating sample summary DataFrame...")

    # Define columns that uniquely identify a sample
    id_vars = ['sample_id', 'sample_latitude', 'sample_longitude', 'sample_collection_date']
    
    # Check if essential columns exist to perform the pivot
    required_cols = id_vars + ['variable', 'value']
    if not all(col in df.columns for col in required_cols):
        print("    Warning: Missing required columns ('sample_id', 'variable', 'value', etc.) for summary. Skipping pivot.")
        return pd.DataFrame()

    # Filter to only the necessary columns and drop rows where variable or value is null
    summary_subset = df[required_cols].dropna(subset=['variable', 'value'])

    # In case of multiple observations for the same variable per sample, we take the first one.
    summary_subset = summary_subset.drop_duplicates(subset=id_vars + ['variable'])

    # Pivot the table to create the "wide" format
    try:
        sample_summary_df = summary_subset.pivot_table(
            index=id_vars,
            columns='variable',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        # Clean up the column names that come from the pivot operation
        sample_summary_df.columns.name = None
        print("    Sample summary created successfully.")
        return sample_summary_df
    except Exception as e:
        print(f"    Error during pivoting process: {e}")
        return pd.DataFrame()

# --- Main execution block ---
if __name__ == "__main__":
    # Define the directory and pattern for the files to process.
    data_directory = '/usr2/people/macgregor/amplicon/project_01/01_raw_data/environmental_context'
    file_pattern = f"{data_directory}/env_data_*.tsv"

    print("Starting data processing pipeline...")
    print(f"Searching for files matching pattern: {file_pattern}")

    # Use glob to find all files matching the pattern
    files_to_process = glob.glob(file_pattern)

    if not files_to_process:
        print("\nNo files found matching the pattern. Exiting.")
    else:
        print(f"\nFound {len(files_to_process)} files to process.")
        
        # Run the full processing and filtering pipeline on ALL files at once
        filtered_df = process_and_filter_data(files_to_process)

        if not filtered_df.empty:
            print("\nFiltering and cleaning finished successfully for all files.")
            print("---------------------------------")
            
            # Create the sample-centric summary DataFrame from the combined filtered data
            sample_summary = create_sample_summary(filtered_df)
            
            if not sample_summary.empty:
                print("\nCombined Sample Summary DataFrame Info:")
                sample_summary.info()
                
                print("\nFirst 5 rows of the Combined Sample Summary DataFrame:")
                print(sample_summary.head())

                # Define the output file path
                output_path = os.path.join(data_directory, '_summary.tsv')
                
                # Write the summary DataFrame to a TSV file
                try:
                    sample_summary.to_csv(output_path, sep='\t', index=False)
                    print(f"\nSuccessfully wrote summary to: {output_path}")
                except Exception as e:
                    print(f"\nError writing summary file: {e}")
            else:
                print("\nCould not generate a combined sample summary.")
        else:
            print(f"\nNo data remained after filtering for all processed files.")

