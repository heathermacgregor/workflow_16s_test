import requests
import argparse
import sys
from collections import defaultdict
import time
import math
import pandas as pd

# ENA Portal API endpoint
ENA_API_URL = "https://www.ebi.ac.uk/ena/portal/api/search"

def batch_fetch_ena_data(result_type, query_key, accessions, email):
    """
    Performs a batch fetch from the ENA API for a list of accessions,
    handling large lists by splitting them into chunks.

    Args:
        result_type (str): The type of data to fetch (e.g., 'read_experiment', 'read_run').
        query_key (str): The field to query on (e.g., 'sample_accession').
        accessions (list): A list of accession strings to query for.
        fields (list): The fields to return from the API.
        email (str): The user's email for API identification.

    Returns:
        list: A list of dictionaries from the API, or an empty list on failure.
    """
    if not accessions:
        return []

    all_results = []
    unique_accessions = list(set(accessions))
    
    # Set chunk size to 100 as requested
    CHUNK_SIZE = 100
    num_chunks = math.ceil(len(unique_accessions) / CHUNK_SIZE)

    for i in range(0, len(unique_accessions), CHUNK_SIZE):
        chunk = unique_accessions[i:i + CHUNK_SIZE]
        current_chunk_num = (i // CHUNK_SIZE) + 1
        print(f"  Fetching {result_type} data for chunk {current_chunk_num} of {num_chunks} ({len(chunk)} accessions)...")
        
        query = " OR ".join(f'{query_key}="{acc}"' for acc in chunk)

        params = {
            "result": result_type,
            "query": query,
            "fields": "all",
            "format": "json",
            "limit": 0
        }
        headers = {"User-Agent": f"PythonClient/1.0 ({email})"}

        response = None
        try:
            response = requests.get(ENA_API_URL, params=params, headers=headers, timeout=120)
            response.raise_for_status()
            if response.status_code == 204: # No content
                print("    ... no results found for this chunk.")
                continue
            
            chunk_results = response.json()
            all_results.extend(chunk_results)
            print(f"    ... found {len(chunk_results)} results for this chunk.")
            time.sleep(0.2)

        except requests.exceptions.HTTPError as http_err:
            print(f"\nHTTP error occurred while fetching chunk for {result_type}: {http_err}", file=sys.stderr)
            if response:
                print(f"Response text: {response.text}", file=sys.stderr)
            continue
        except requests.exceptions.RequestException as req_err:
            print(f"\nAn error occurred while fetching {result_type}: {req_err}", file=sys.stderr)
            break
    
    return all_results


def find_nearby_samples(latitude, longitude, radius, email, no_host_filter):
    """
    Finds ENA samples within a specified radius of a given latitude and longitude.
    """
    fields = ["accession", "scientific_name", "collection_date", "location", "description", "host"]
    query = f"geo_circ({latitude},{longitude},{radius})"

    if no_host_filter:
        query += " AND NOT host=*"
    params = {"result": "sample", "query": query, "fields": ",".join(fields), "format": "json", "limit": 0}
    headers = {"User-Agent": f"PythonClient/1.0 ({email})"}

    response = None
    try:
        print(f"Querying ENA for samples within {radius}km of ({latitude}, {longitude})...")
        print(ENA_API_URL)
        print(params)
        print(headers)
        response = requests.get(ENA_API_URL, params=params, headers=headers, timeout=60)
        response.raise_for_status()
        if response.status_code == 204:
            return []
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}", file=sys.stderr)
        if response:
            print(f"Response text: {response.text}", file=sys.stderr)
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred with the request: {req_err}", file=sys.stderr)
    except ValueError as json_err:
        print(f"Failed to decode JSON response: {json_err}", file=sys.stderr)
        if response:
            print(f"Response text: {response.text}", file=sys.stderr)
        
    return None
        
    return None
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find nearby ENA samples and their associated data for a given latitude and longitude.",
        epilog="Example: python find_ena_samples.py 37.8719 -122.2585 --radius 10 --amplicon"
    )

    parser.add_argument("latitude", type=float, help="Latitude of the location to search.")
    parser.add_argument("longitude", type=float, help="Longitude of the location to search.")
    parser.add_argument("--radius", type=int, default=50, help="Search radius in kilometers. Default is 50km.")
    parser.add_argument("--email", type=str, default="macgregor@berkeley.edu", help="Email address for API identification.")
    parser.add_argument("--no-host", action="store_true", help="Filter out host-associated samples.")
    parser.add_argument("--amplicon", action="store_true", help="Filter for amplicon sequencing data.")
    
    args = parser.parse_args()

    samples = find_nearby_samples(args.latitude, args.longitude, args.radius, args.email, args.no_host)

    if samples is None:
        print("Failed to retrieve sample data from ENA.", file=sys.stderr)
        sys.exit(1)

    if not samples:
        print("No samples found at the specified location.")
        sys.exit(0)

    # Safely get sample accessions, skipping records missing the 'accession' key
    sample_accessions = [s['accession'] for s in samples if 'accession' in s]
    print(f"\nFound {len(sample_accessions)} valid samples. Now fetching associated experiments...")
    experiments = batch_fetch_ena_data("read_experiment", "sample_accession", sample_accessions, args.email)
    print(f"Finished fetching. Found a total of {len(experiments)} experiments.")

    if experiments:
        exp_accessions = [e['experiment_accession'] for e in experiments if 'experiment_accession' in e]
        runs = batch_fetch_ena_data("read_run", "experiment_accession", exp_accessions, args.email)

        study_accessions = list(set(e['study_accession'] for e in experiments if 'study_accession' in e))
        studies = batch_fetch_ena_data("study", "study_accession", study_accessions, args.email)
    else:
        runs, studies = [], []

    if not samples:
        print("\nNo samples found matching all criteria.")
    else:
        print("\nProcessing data and generating DataFrame...")
        sample_lookup = {s['accession']: s for s in samples if 'accession' in s}
        study_lookup = {p['study_accession']: p for p in studies if 'study_accession' in p}
        exp_to_runs = defaultdict(list)
        for run in runs:
            if 'experiment_accession' in run:
                exp_to_runs[run['experiment_accession']].append(run)

        all_data_rows = []
        for exp in experiments:
            sample = sample_lookup.get(exp.get('sample_accession'))
            study = study_lookup.get(exp.get('study_accession'))
            exp_runs = exp_to_runs.get(exp.get('experiment_accession'), [])

            # Dynamically create base row with prefixed columns to avoid name collisions
            base_row = {}
            if sample:
                for key, value in sample.items(): base_row[f"sample_{key}"] = value
            if study:
                for key, value in study.items(): base_row[f"study_{key}"] = value
            for key, value in exp.items(): base_row[f"experiment_{key}"] = value

            if not exp_runs:
                all_data_rows.append(base_row)
            else:
                for run in exp_runs:
                    row = base_row.copy()
                    for key, value in run.items(): row[f"run_{key}"] = value
                    all_data_rows.append(row)
        
        df = pd.DataFrame(all_data_rows)
        
        if args.amplicon:
            print("\nFiltering DataFrame for AMPLICON library strategy...")
            # Use the prefixed column name for filtering
            amplicon_col = 'experiment_library_strategy'
            if amplicon_col in df.columns and not df.empty:
                df = df[df[amplicon_col] == 'AMPLICON'].copy()
                print(f"Filtered down to {len(df)} rows.")
            else:
                print(f"Cannot filter: '{amplicon_col}' column not found or DataFrame is empty.")
                df = pd.DataFrame()
                
        if not df.empty:
            print("\nSorting results by experiment data completeness...")
            exp_cols = [c for c in df.columns if c.startswith('experiment_')]
            if exp_cols:
                # Calculate a score based on how many experiment columns are not empty
                df['completeness_score'] = df[exp_cols].notna().sum(axis=1)
                df = df.sort_values(by='completeness_score', ascending=False).drop(columns=['completeness_score'])
                print("...sorting complete.")
            else:
                print("...no experiment columns found to sort by.")


        if df.empty:
            print("\nNo data to display after processing and filtering.")
        else:
            print("\nAnalyzing retrieved columns...")
            non_nan_study_cols = [c for c in df.columns if c.startswith('study_') and df[c].notna().any()]
            non_nan_exp_cols = [c for c in df.columns if c.startswith('experiment_') and df[c].notna().any()]
            non_nan_run_cols = [c for c in df.columns if c.startswith('run_') and df[c].notna().any()]

            print("\n--- Columns with Data ---")
            print("Study fields containing data:")
            if non_nan_study_cols:
                print("  " + "\n  ".join(non_nan_study_cols))
            else:
                print("  None")

            print("\nExperiment fields containing data:")
            if non_nan_exp_cols:
                print("  " + "\n  ".join(non_nan_exp_cols))
            else:
                print("  None")

            print("\nRun fields containing data:")
            if non_nan_run_cols:
                print("  " + "\n  ".join(non_nan_run_cols))
            else:
                print("  None")
                
            # Filter for rows where 'experiment_target_gene' has a value
            if not df.empty:
                target_gene_col = 'experiment_target_gene'
                if target_gene_col in df.columns:
                    print(f"\nFiltering DataFrame for non-empty '{target_gene_col}' entries...")
                    df = df[df[target_gene_col].notna() & (df[target_gene_col] != '')].copy()
                    print(f"Filtered down to {len(df)} rows.")
                else:
                    print(f"\nSkipping target gene filter: '{target_gene_col}' column not found.")
            print("-------------------------\n")
            print(f"\nDisplaying {len(df)} final results as a DataFrame:")
            pd.set_option('display.max_rows', 100)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(df[['experiment_depth', 'experiment_description', 'experiment_extraction_protocol', 
                      'experiment_restriction_enzyme_target_sequence',
                      'experiment_sample_prep_interval', 'experiment_sample_prep_interval_units',
                      'experiment_sequencing_method',
                      'experiment_sequencing_primer_catalog',
                      'experiment_sequencing_primer_provider',
                      'experiment_target_gene',
                      'experiment_library_pcr_isolation_protocol']].head(20))
            print(df['experiment_target_gene'].value_counts())
            print(df['experiment_sequencing_primer_catalog'].value_counts())
            #print(sorted(df.columns.tolist()))
