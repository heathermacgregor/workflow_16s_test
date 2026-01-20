import os
import google.generativeai as genai
import google.api_core.exceptions
import json
import pandas as pd
from tqdm import tqdm
import time

# --- Configuration ---
# REMOVED HARDCODED KEY - THIS IS A MAJOR SECURITY RISK. Use environment variables.
INPUT_PATH = '/usr2/people/macgregor/amplicon/test/data/merged/metadata/raw/genus.tsv'
FINAL_OUTPUT_PATH = '/usr2/people/macgregor/amplicon/test/data/merged/metadata/raw/genus_gemini_enhanced_batch.tsv'
TMP_OUTPUT_PATH = '/usr2/people/macgregor/amplicon/test/data/merged/metadata/raw/genus_gemini_progress.jsonl'
BATCH_SIZE = 10 # Process 50 rows per API call. Adjust as needed.

def configure_api_key():
    """Configures the API key from environment variables."""
    try:
        api_key = os.getenv("GEMINI_API_KEY") # Ensure env var is named GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        print("API key configured successfully.")
    except Exception as e:
        print(f"Error configuring API key: {e}")
        exit()

def get_metadata_df(path):
    """Reads the metadata TSV into a DataFrame."""
    print(f"Reading data from {path}...")
    return pd.read_csv(path, sep='\t', low_memory=False)

def create_batch_prompt(batch_of_dicts):
    """Creates a robust, one-shot prompt for a batch of rows."""
    # Convert the list of dictionaries to a JSON string for the prompt
    input_json_str = json.dumps(batch_of_dicts, indent=2)

    return f"""
You are a highly intelligent data standardization engine. Your task is to process a list of JSON objects and return a new list of JSON objects with a clean, specified schema.

Follow these rules for each object in the list:
- Column names with units (e.g. "uranium_ppm") should be split into "uranium" and "uranium_unit" columns. If no unit is given, the unit can be 'unknown'.
- Convert latitude and longitude to separate numeric columns in decimal degrees.
- If a value is missing or cannot be determined, use null.
- Ensure the output is a valid JSON list of objects.

### Example ###

#### Input Data (List of Objects): ####
```json
[
  {{
    "accession": "SAMN999999",
    "collection_date": "Mar 15, 2024",
    "lat_lon": "36.618 N 121.902 W",
    "uranium_ppm": "5.2"
  }},
  {{
    "accession": "SAMN888888",
    "collection_date": "2023-01-20",
    "lat_lon": "40.7128, -74.0060",
    "uranium_ppm": "1.1"
  }}
]
Desired Standardized Output (List of Objects):
JSON

[
  {{
    "sample_id": "SAMN999999",
    "collection_date": "2024-03-15",
    "latitude": 36.618,
    "longitude": -121.902,
    "uranium": 5.2,
    "uranium_unit": "ppm"
  }},
  {{
    "sample_id": "SAMN888888",
    "collection_date": "2023-01-20",
    "latitude": 40.7128,
    "longitude": -74.0060,
    "uranium": 1.1,
    "uranium_unit": "ppm"
  }}
]
Your Task
Now, standardize the following list of objects using the exact same logic and format. Your response MUST be only the raw JSON list, with no surrounding text or explanations.

New Input Data:
JSON

{input_json_str}
Standardized Output:
"""

def main():
    """Main function to run the data enhancement process."""
    configure_api_key()
    df = get_metadata_df(INPUT_PATH)
    # Convert NaN to None for proper JSON serialization
    df = df.where(pd.notnull(df), None)
    all_rows = df.to_dict(orient='records')
    
    # --- RESUMABILITY LOGIC: START ---
    processed_ids = set()
    try:
        if os.path.exists(TMP_OUTPUT_PATH):
            print(f"Found existing progress file: {TMP_OUTPUT_PATH}. Reading processed IDs...")
            with open(TMP_OUTPUT_PATH, 'r') as f:
                for line in f:
                    # Assumes the standardized dict will have a 'sample_id' or 'accession' key
                    data = json.loads(line)
                    if data.get('sample_id'):
                        processed_ids.add(data['sample_id'])
                    elif data.get('original_data', {}).get('accession'): # Handles error logs
                        processed_ids.add(data['original_data']['accession'])

            print(f"Found {len(processed_ids)} already processed samples.")
    except Exception as e:
        print(f"Warning: Could not read progress file. Starting from scratch. Error: {e}")

    # Filter out rows that have already been processed
    rows_to_process = [row for row in all_rows if row.get('accession') not in processed_ids]

    if not rows_to_process:
        print("All rows have already been processed.")
    else:
        print(f"Total rows: {len(all_rows)}. Already processed: {len(processed_ids)}. Remaining: {len(rows_to_process)}.")
    # --- RESUMABILITY LOGIC: END ---

    # Split the list into batches
    batches = [rows_to_process[i:i + BATCH_SIZE] for i in range(0, len(rows_to_process), BATCH_SIZE)]
    model = genai.GenerativeModel('gemini-1.5-pro')

    if batches:
        print(f"Starting data enhancement with {len(batches)} batches...")
        for batch in tqdm(batches, desc="Processing batches"):
            try:
                prompt = create_batch_prompt(batch)
                response = model.generate_content(prompt)
                
                cleaned_json_string = response.text.strip().replace("```json", "").replace("```", "")
                list_of_processed_dicts = json.loads(cleaned_json_string)
                
                # --- SAVE PROGRESS IMMEDIATELY ---
                with open(TMP_OUTPUT_PATH, 'a') as f:
                    for item in list_of_processed_dicts:
                        f.write(json.dumps(item) + '\n')
                
                time.sleep(1.5) # Respect rate limits

            except (google.api_core.exceptions.ResourceExhausted, Exception) as e:
                print(f"\nError processing a batch: {e}. Logging errors and continuing.")
                with open(TMP_OUTPUT_PATH, 'a') as f:
                    for row in batch:
                        error_log = {'error': str(e), 'original_data': row}
                        f.write(json.dumps(error_log) + '\n')

    # --- FINAL STEP: Create the final TSV from the complete log file ---
    print("\nData enhancement complete.")
    print(f"Consolidating all results from {TMP_OUTPUT_PATH}...")

    try:
        final_df = pd.read_json(TMP_OUTPUT_PATH, lines=True)
        # You might want to handle the 'error' and 'original_data' columns here
        # For now, we'll just save everything
        print(f"Saving {len(final_df)} total rows to {FINAL_OUTPUT_PATH}...")
        final_df.to_csv(FINAL_OUTPUT_PATH, sep='\t', index=False)
        print("Process finished successfully.")
    except Exception as e:
        print(f"Could not create final TSV file from log. Error: {e}")
        print("The processed data is still available in the .jsonl file.")
    
if __name__ == '__main__':
    main()