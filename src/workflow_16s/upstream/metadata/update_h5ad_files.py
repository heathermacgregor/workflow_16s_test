import logging
import re
import pandas as pd
import anndata as ad
import requests
import io
import os
import yaml # Added
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Tuple, Set, Dict, Any
from Bio import Entrez
from pydantic import ValidationError # Added

# --- IMPORTS FROM YOUR WORKFLOW ---
# These are the real handlers, replacing the placeholders
try:
    from workflow_16s.api.environmental_data.other.execute import EnvironmentalDataCollector
    from workflow_16s.api.environmental_data.google.arkin_env_agents import main as arkin_env_agents
    from workflow_16s.api.nuclear_fuel_cycle.nfc import NFCFacilitiesHandler
    from workflow_16s.config_schema import AppConfig
    from workflow_16s.downstream.adata_utils import safe_write_h5ad
except ImportError:
    print("CRITICAL: Failed to import 'workflow_16s' package.")
    print("Please run this script from the root of your project (the directory containing the 'workflow_16s' folder)")
    print("or ensure the 'workflow_16s' package is installed in your environment.")
    exit(1)

# --- Constants ---

# *** REQUIRED: Set your email for NCBI Entrez API ***
# The Entrez API (for publication fetching) requires your email.
ENTREZ_EMAIL = "macgregor@berkeley.edu"
if ENTREZ_EMAIL == "your-email@example.com":
    print("WARNING: Please update ENTREZ_EMAIL in the script before running.")
Entrez.email = ENTREZ_EMAIL

# ENA API endpoint (Using the Search API, which is more robust for queries)
ENA_API_URL = "https://www.ebi.ac.uk/ena/portal/api/search"
# ENA_FIELDS list removed. We will now fetch all available columns.

# Regex patterns for lat/lon
LAT_LON_PATTERNS = {
    'lat': [r'^lat$', r'.*latitude.*', r'^x$'],
    'lon': [r'^lon$', r'.*longitude.*', r'^y$'],
    'lat_lon': [r'.*lat.*lon.*', r'.*lon.*lat.*']
}

# Exclusion keywords (from partition.py)
EXCLUSION_KEYWORDS = [
    'metagenome', 'metatranscriptome', 'virus', 'viral', 'phage', 'eukaryote',
    'metabolome', 'metabolomic', 'proteome', 'proteomic', 'virome', 'viromic',
    'synthetic', 'mock community', 'control', 'contaminant', 'human'
]

# Host-associated keywords (from partition.py)
HOST_ASSOCIATED_KEYWORDS = [
    'human', 'mouse', 'rat', 'bovine', 'gut', 'fecal', 'oral', 'skin',
    'microbiome', 'host-associated', 'clinical'
]

# --- 1. LOGGING SETUP ---
def setup_logging():
    """Configures logging to file and console."""
    # Remove any existing root handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_file = 'processing.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging configured. Output will be saved to {log_file}")
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("Bio").setLevel(logging.WARNING)

# --- 2. ENA METADATA FETCHER (with Caching) ---
def get_ena_metadata(project_id: str, cache_dir: Path) -> Optional[pd.DataFrame]:
    """
    Fetches metadata for a given ENA project, using a local file cache.
    """
    cache_file = cache_dir / f"{project_id}_ena.tsv"

    # 1. Check cache
    if cache_file.exists():
        logging.info(f"Using cached ENA metadata for {project_id} from {cache_file}")
        try:
            ena_df = pd.read_csv(cache_file, sep='\t')
            return ena_df
        except Exception as e:
            logging.warning(f"Cache file {cache_file} is corrupted, refetching. Error: {e}")

    # 2. Fetch from ENA (if cache miss or corrupt)
    logging.info(f"Fetching ENA metadata for {project_id} (no valid cache)...")
    params = {
        'query': f'study_accession="{project_id}"',
        'result': 'read_run',
        'format': 'tsv'
        # 'fields' parameter removed to fetch all available columns
    }
    try:
        response = requests.get(ENA_API_URL, params=params, timeout=30)
        response.raise_for_status()

        tsv_data = response.text
        if not tsv_data or len(tsv_data.splitlines()) <= 1:
            logging.warning(f"No metadata returned from ENA for {project_id}.")
            return None

        # 3. Save to cache
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(tsv_data)
        logging.info(f"Saved new ENA metadata to cache: {cache_file}")

        # Read TSV data into a pandas DataFrame
        ena_df = pd.read_csv(io.StringIO(tsv_data), sep='\t')
        logging.info(f"Successfully fetched {len(ena_df)} records from ENA.")
        return ena_df

    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP Error fetching ENA data: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching ENA data: {e}")
        return None

# --- 3. METADATA MERGING & CLEANING ---
def find_merge_column(obs_df: pd.DataFrame, ena_df: pd.DataFrame) -> Optional[str]:
    """
    Finds the best common column to merge on.
    Prefers 'run_accession', then 'sample_accession', then 'sample_alias'.
    """
    potential_cols = ['run_accession', 'sample_accession', 'sample_alias']
    obs_cols = set(obs_df.columns)
    ena_cols = set(ena_df.columns)

    for col in potential_cols:
        if col in obs_cols and col in ena_cols:
            # Try to make data types compatible for comparison
            try:
                obs_unique = obs_df[col].astype(str).is_unique
                ena_unique = ena_df[col].astype(str).is_unique
                if obs_unique and ena_unique:
                    logging.info(f"Identified '{col}' as the unique merge column.")
                    return col
                else:
                    logging.warning(f"Column '{col}' exists but is not unique. Skipping.")
            except Exception as e:
                logging.warning(f"Could not compare column '{col}' due to type issues: {e}")

    logging.error("Could not find a suitable unique column to merge ENA data.")
    return None

def merge_metadata(obs_df: pd.DataFrame, ena_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Merges the existing .obs DataFrame with the new ENA DataFrame.
    """
    merge_col = find_merge_column(obs_df, ena_df)
    if not merge_col:
        return None

    # Ensure merge columns are string type for a robust merge
    obs_df[merge_col] = obs_df[merge_col].astype(str)
    ena_df[merge_col] = ena_df[merge_col].astype(str)
    
    obs_df_indexed = obs_df.set_index(merge_col, drop=False)
    ena_df_indexed = ena_df.set_index(merge_col, drop=False)

    # Combine dataframes, giving preference to the new ENA data for common columns
    combined_df = obs_df_indexed.combine_first(ena_df_indexed)

    # Add columns that were only in ENA
    new_cols = ena_df_indexed.columns.difference(obs_df_indexed.columns)
    combined_df[new_cols] = ena_df_indexed[new_cols]

    # Reset index to restore the merge column (now as a column)
    if merge_col not in combined_df.columns:
         combined_df[merge_col] = combined_df.index

    combined_df = combined_df.reset_index(drop=True)

    logging.info("Successfully merged existing metadata with ENA data.")
    return combined_df

def report_metadata_stats(df: pd.DataFrame, context: str):
    """Logs a report of DataFrame columns, types, and fullness."""
    logging.info(f"--- Metadata Report ({context}) ---")
    logging.info(f"Total records: {len(df)}")
    if df.empty:
        logging.info("DataFrame is empty.")
        return

    report = []
    for col in df.columns:
        col_type = str(df[col].dtype)
        non_null_count = df[col].count()
        non_null_pct = (non_null_count / len(df)) * 100
        report.append({
            'Column': col,
            'Type': col_type,
            'Fullness': f"{non_null_count}/{len(df)} ({non_null_pct:.1f}%)"
        })

    report_df = pd.DataFrame(report)
    logging.info("\n" + report_df.to_string())
    logging.info(f"--- End Report ({context}) ---")

def clean_lat_lon(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Finds all lat/lon columns, extracts data into 'lat' and 'lon',
    and returns the list of old columns to be deleted.
    """
    new_lat = pd.Series(index=df.index, dtype=float)
    new_lon = pd.Series(index=df.index, dtype=float)
    cols_to_drop = []

    def compile_regex(patterns):
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    lat_regexes = compile_regex(LAT_LON_PATTERNS['lat'])
    lon_regexes = compile_regex(LAT_LON_PATTERNS['lon'])
    lat_lon_regexes = compile_regex(LAT_LON_PATTERNS['lat_lon'])

    lat_cols = []
    lon_cols = []
    combo_cols = []

    for col in df.columns:
        if any(r.search(col) for r in lat_lon_regexes):
            combo_cols.append(col)
        elif any(r.search(col) for r in lat_regexes):
            lat_cols.append(col)
        elif any(r.search(col) for r in lon_regexes):
            lon_cols.append(col)

    # Process single lat/lon columns first
    for col in lat_cols:
        logging.info(f"Found latitude column: '{col}'")
        lat_data = pd.to_numeric(df[col], errors='coerce')
        new_lat = new_lat.fillna(lat_data)
        cols_to_drop.append(col)

    for col in lon_cols:
        logging.info(f"Found longitude column: '{col}'")
        lon_data = pd.to_numeric(df[col], errors='coerce')
        new_lon = new_lon.fillna(lon_data)
        cols_to_drop.append(col)

    # Process combined lat/lon columns
    for col in combo_cols:
        logging.info(f"Found combined lat/lon column: '{col}'")
        cols_to_drop.append(col)
        pat = r'([-+]?\d*\.\d+|\d+)[,;\s]+([-+]?\d*\.\d+|\d+)'
        extracted = df[col].astype(str).str.extract(pat)

        if not extracted.empty:
            lat_data = pd.to_numeric(extracted[0], errors='coerce')
            lon_data = pd.to_numeric(extracted[1], errors='coerce')
            new_lat = new_lat.fillna(lat_data)
            new_lon = new_lon.fillna(lon_data)

    # Add/update the new standardized columns
    # Check if 'lat'/'lon' already exists and has data, otherwise use new data
    if 'lat' in df.columns and df['lat'].count() > 0:
        df['lat'] = df['lat'].fillna(new_lat)
    else:
        df['lat'] = new_lat

    if 'lon' in df.columns and df['lon'].count() > 0:
        df['lon'] = df['lon'].fillna(new_lon)
    else:
        df['lon'] = new_lon

    cols_to_drop = list(set(cols_to_drop) - {'lat', 'lon'})
    logging.info(f"Standardized 'lat' and 'lon' columns. Old columns to be dropped: {cols_to_drop}")
    return df, cols_to_drop

def standardize_collection_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the best possible collection date column, standardizes it, 
    and renames it to 'collection_date'.
    """
    potential_cols = []
    for col in df.columns:
        if 'date' in col.lower() or 'timestamp' in col.lower():
            potential_cols.append(col)

    if 'collection_date' in df.columns and df['collection_date'].count() > 0:
        logging.info("Using existing 'collection_date' column.")
        col_to_use = 'collection_date'
    else:
        best_col = None
        for col in potential_cols:
            if col in df.columns and df[col].count() > 0:
                best_col = col
                break # Found the first non-empty date column
        
        if best_col:
            logging.info(f"Standardizing 'collection_date' from column: '{best_col}'")
            col_to_use = best_col
        else:
            logging.warning("No valid collection date column found. 'collection_date' will be empty.")
            df['collection_date'] = pd.NaT
            return df

    # Standardize the chosen column
    try:
        standardized_date = pd.to_datetime(df[col_to_use], errors='coerce')
        # Check if parsing failed for all rows
        if standardized_date.isnull().all():
            logging.warning(f"Column '{col_to_use}' could not be parsed as dates. 'collection_date' will be empty.")
            df['collection_date'] = pd.NaT
        else:
            df['collection_date'] = standardized_date
            logging.info(f"Successfully standardized 'collection_date'. {df['collection_date'].count()} valid dates found.")
            # Drop the original column if it's not 'collection_date'
            if col_to_use != 'collection_date':
                logging.info(f"Dropping old date column: '{col_to_use}'")
                df = df.drop(columns=[col_to_use])
    except Exception as e:
        logging.error(f"Error parsing date column '{col_to_use}': {e}. 'collection_date' will be empty.")
        df['collection_date'] = pd.NaT

    return df

# --- 4. METADATA FILTERING (from partition.py) ---
def find_keyword_matches(text: str, keywords: List[str]) -> Set[str]:
    """Finds which keywords are present in a given text string."""
    if not isinstance(text, str):
        return set()
    text_lower = text.lower()
    matches = set()
    for keyword in keywords:
        # Use regex to find whole words
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
            matches.add(keyword)
    return matches

def filter_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies keyword-based filtering and tagging (from partition.py).
    """
    logging.info("Applying metadata filtering and tagging...")
    text_cols = ['sample_description', 'sample_title', 'experiment_title']
    
    # Ensure text columns exist and are string type
    for col in text_cols:
        if col not in df.columns:
            df[col] = pd.Series(dtype=str)
        # *** THIS IS THE FIX ***
        # Convert to string *first* to handle Categorical types, then fill NA
        df[col] = df[col].astype(str).fillna('')

    # Combine text fields for searching
    df['combined_text'] = df[text_cols].apply(lambda row: ' '.join(row), axis=1)

    # 1. Exclusion filtering
    df['exclusion_keywords'] = df['combined_text'].apply(
        lambda x: find_keyword_matches(x, EXCLUSION_KEYWORDS)
    )
    df['is_excluded'] = df['exclusion_keywords'].apply(len) > 0
    excluded_count = df['is_excluded'].sum()
    logging.info(f"Flagged {excluded_count} records for exclusion based on keywords.")

    # 2. Host-associated tagging
    df['host_keywords'] = df['combined_text'].apply(
        lambda x: find_keyword_matches(x, HOST_ASSOCIATED_KEYWORDS)
    )
    df['is_host_associated'] = df['host_keywords'].apply(len) > 0
    host_count = df['is_host_associated'].sum()
    logging.info(f"Flagged {host_count} records as likely host-associated.")

    df = df.drop(columns=['combined_text', 'exclusion_keywords', 'host_keywords'])
    return df

# --- 5. PUBLICATION FETCHER (from partition.py) ---
def get_publication_info(study_id: str) -> Dict[str, Any]:
    """
    Fetches publication info (PMID, Title) from NCBI Entrez.
    """
    logging.info(f"Fetching publication info for {study_id}...")
    try:
        # 1. Search for the study ID in PubMed
        handle = Entrez.esearch(
            db="pubmed",
            term=f"{study_id}[BioProject]",
            retmax=5
        )
        record = Entrez.read(handle)
        handle.close()
        
        pmid_list = record.get("IdList", [])
        if not pmid_list:
            logging.info(f"No publications found for {study_id}.")
            return {}

        # 2. Get summary for the first found PMID
        pmid = pmid_list[0]
        handle = Entrez.esummary(db="pubmed", id=pmid)
        summary = Entrez.read(handle)
        handle.close()

        if not summary:
            return {}

        pub_data = summary[0]
        title = pub_data.get("Title", "N/A")
        
        logging.info(f"Found publication for {study_id}: PMID {pmid} - {title[:50]}...")
        return {
            "publication_pmid": pmid,
            "publication_title": title,
            "publication_journal": pub_data.get("Source", "N/A"),
            "publication_date": pub_data.get("PubDate", "N/A")
        }

    except Exception as e:
        logging.error(f"Failed to fetch publication info for {study_id}: {e}")
        return {}

# --- 6. REAL ENRICHMENT FUNCTIONS (from processor.py) ---
# All placeholder functions have been replaced with these.

def enrich_env_data(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Enriches metadata with environmental data using EnvironmentalDataCollector."""
    
    # 1. Check for required columns BEFORE initialization
    # The __init__ requires 'lat', 'lon', AND 'collection_date'
    required_cols = ['lat', 'lon', 'collection_date']
    if not all(col in df.columns for col in required_cols):
        logging.warning("Skipping environmental enrichment: 'lat', 'lon', or 'collection_date' columns missing.")
        return df
    
    # 2. Check if data is null (which __init__ will drop anyway)
    if df[required_cols].isnull().all().all():
        logging.warning("Skipping environmental enrichment: 'lat'/'lon'/'collection_date' columns are empty.")
        return df

    logging.info("Enriching with environmental data (EnvironmentalDataCollector)...")
    try:
        # 3. Initialize the collector HERE, inside the function
        # We pass 'df' as the 'data' argument and 'config'
        env_data_collector = EnvironmentalDataCollector(data=df, config=config)
        
        # 4. Call run_apis() (which uses the 'self.data' it was initialized with)
        env_results_df = env_data_collector.run_apis()
        
        if not env_results_df.empty:
            # 5. Merge the results back.
            # The 'env_results_df' will have 'lat', 'lon', 'collection_date'
            # We must merge on these keys.
            
            # Find columns that are in both, *except* for our merge keys
            common_cols = [col for col in env_results_df.columns if col in df.columns]
            cols_to_drop = list(set(common_cols) - set(required_cols))
            
            if cols_to_drop:
                logging.info(f"Dropping overlapping columns before merge: {cols_to_drop}")
                df = df.drop(columns=cols_to_drop)
            
            # Ensure data types of merge keys are compatible
            df['collection_date'] = pd.to_datetime(df['collection_date'], errors='coerce').dt.strftime('%Y-%m-%d')
            env_results_df['collection_date'] = pd.to_datetime(env_results_df['collection_date'], errors='coerce').dt.strftime('%Y-%m-%d')
            
            for col in ['lat', 'lon']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                env_results_df[col] = pd.to_numeric(env_results_df[col], errors='coerce')

            df = df.merge(env_results_df, on=required_cols, how='left')
            logging.info(f"Successfully merged {len(env_results_df)} rows of environmental data.")
            
        logging.info("Environmental data enrichment complete.")
    except ValueError as e:
       # Catch the ValueError from __init__ if cols are still bad
       logging.error(f"Environmental data enrichment failed (ValueError, likely in init): {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Environmental data enrichment failed: {e}", exc_info=True)
    return df

def enrich_arkin_agents(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Enriches metadata with Arkin agents data."""
    if 'lat' not in df.columns or 'lon' not in df.columns or df[['lat', 'lon']].isnull().all().all():
        logging.warning("Skipping Arkin agents enrichment: 'lat'/'lon' columns missing or empty.")
        return df
    
    logging.info("Enriching with Arkin agents data...")
    try:
        arkin_results_df = arkin_env_agents(df, config)
        if not arkin_results_df.empty:
            common_cols = [col for col in arkin_results_df.columns if col in df.columns]
            df = df.drop(columns=common_cols)
            df = df.merge(arkin_results_df, left_index=True, right_index=True, how='left')
        logging.info("Arkin agents enrichment complete.")
    except Exception as e:
        logging.error(f"Arkin agents enrichment failed: {e}", exc_info=True)
    return df

def enrich_nfc(df: pd.DataFrame, nfc_handler: NFCFacilitiesHandler) -> pd.DataFrame:
    """Enriches metadata with NFC (Nuclear Fuel Cycle) data."""
    if 'lat' not in df.columns or 'lon' not in df.columns or df[['lat', 'lon']].isnull().all().all():
        logging.warning("Skipping NFC enrichment: 'lat'/'lon' columns missing or empty.")
        return df
    
    logging.info("Enriching with NFC (Nuclear Fuel Cycle) data...")
    try:
        nfc_results_df = nfc_handler.process_dataframe(df)
        if not nfc_results_df.empty:
            common_cols = [col for col in nfc_results_df.columns if col in df.columns]
            df = df.drop(columns=common_cols)
            df = df.merge(nfc_results_df, left_index=True, right_index=True, how='left')
        logging.info("NFC enrichment complete.")
    except Exception as e:
        logging.error(f"NFC enrichment failed: {e}", exc_info=True)
    return df


# --- 7. MAIN FILE PROCESSING ORCHESTRATOR ---

def process_file(
    file_path: Path, 
    cache_dir: Path, 
    config: AppConfig, 
    nfc_handler: NFCFacilitiesHandler
):
    """
    Runs the full processing pipeline for a single .h5ad file.
    """
    logging.info(f"--- Processing {file_path.name} ---")

    # 1. Extract Project ID
    match = re.match(r'(PRJ[EDN][AB]\d+)', file_path.name, re.IGNORECASE)
    if not match:
        logging.error(f"Could not extract ENA project ID from filename: {file_path.name}. Skipping.")
        return
    project_id = match.group(1).upper()

    # 2. Fetch ENA Metadata (from cache or API)
    ena_df = get_ena_metadata(project_id, cache_dir)
    if ena_df is None:
        logging.warning(f"No ENA metadata for {project_id}. Proceeding with original data.")
        ena_df = pd.DataFrame()

    # 3. Read AnnData
    try:
        adata = ad.read_h5ad(file_path)
        obs_df = adata.obs.copy()
    except Exception as e:
        logging.error(f"Failed to read .h5ad file {file_path.name}: {e}. Skipping.")
        return

    # 4. Report Original Metadata
    report_metadata_stats(obs_df, f"{file_path.name} (Original)")

    # 5. Merge Metadata
    if not ena_df.empty:
        merged_df = merge_metadata(obs_df, ena_df)
        if merged_df is None:
            logging.error("Failed to merge metadata, proceeding with original data.")
            merged_df = obs_df
    else:
        merged_df = obs_df

    # 6. Fetch Publication Info
    pub_info = get_publication_info(project_id)
    if pub_info:
        for key, value in pub_info.items():
            merged_df[key] = value
        logging.info(f"Added publication info to metadata.")

    # 7. Apply Filters & Tagging
    merged_df = filter_metadata(merged_df)

    # 8. Clean Lat/Lon
    merged_df, cols_to_drop = clean_lat_lon(merged_df)

    # 9. Standardize Collection Date
    merged_df = standardize_collection_date(merged_df)

    # 10. Enrich Metadata (REAL ENRICHMENT)
    merged_df = enrich_env_data(merged_df, config)
    merged_df = enrich_arkin_agents(merged_df, config) # arkin agents takes the full config
    merged_df = enrich_nfc(merged_df, nfc_handler)

    # 11. Final Cleanup
    if cols_to_drop:
        merged_df = merged_df.drop(columns=cols_to_drop, errors='ignore')
        logging.info(f"Dropped processed columns: {cols_to_drop}")
    
    # Ensure all columns are compatible with HDF5
    for col in merged_df.columns:
        if merged_df[col].dtype == 'object':
            # Convert mixed types or complex objects to string
            if merged_df[col].apply(type).nunique() > 1:
                merged_df[col] = merged_df[col].astype(str)
        elif 'datetime' in str(merged_df[col].dtype):
             merged_df[col] = merged_df[col].astype(str)


    # 12. Report Modified Metadata
    report_metadata_stats(merged_df, f"{file_path.name} (Modified)")

    # 13. Update and Save AnnData
    try:
        adata.obs = merged_df
        safe_write_h5ad(adata, file_path)
        logging.info(f"Successfully updated and saved changes to {file_path.name}")
    except Exception as e:
        logging.error(f"Failed to save modified .h5ad file {file_path.name}: {e}", exc_info=True)

    logging.info(f"--- Finished {file_path.name} ---")

# --- 8. SCRIPT EXECUTION ---

if __name__ == "__main__":
    setup_logging() # Log file 'processing.log' will be created in the script's run directory.

    script_run_dir = Path().cwd()
    
    # --- Load Configuration ---
    # Use the specific, absolute path provided by the user
    config_path = Path("/usr2/people/macgregor/amplicon/workflow_16s/config/config.yaml")
    config: AppConfig
    if not config_path.exists():
        logging.critical(f"Configuration file not found at: {config_path.resolve()}")
        logging.critical("Please ensure this path is correct and accessible.")
        exit(1)
        
    try:
        logging.info(f"Loading configuration from: {config_path.resolve()}")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = AppConfig(**config_dict)
        logging.info("Configuration loaded and validated successfully.")
    except (yaml.YAMLError, ValidationError) as e:
        logging.critical(f"Failed to load or validate configuration file: {e}")
        exit(1)
    except Exception as e:
        logging.critical(f"An unexpected error occurred loading config: {e}")
        exit(1)

    # --- Initialize Handlers ---
    try:
        logging.info("Initializing enrichment handlers...")
        nfc_handler = NFCFacilitiesHandler(config=config)
        # env_data_collector is now initialized inside process_file
        logging.info("Enrichment handlers initialized.")
    except Exception as e:
        logging.critical(f"Failed to initialize handlers: {e}", exc_info=True)
        exit(1)

    # --- Define the target directory for .h5ad files ---
    target_data_dir = Path("/usr2/people/macgregor/amplicon/project_01/03_processed_data/specific_h5ad_files")

    # --- Setup cache directory relative to the script's run location ---
    cache_dir = script_run_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    logging.info(f"Script running from: {script_run_dir.resolve()}")
    logging.info(f"Using cache directory: {cache_dir.resolve()}")
    logging.info(f"Scanning for .h5ad files in target: {target_data_dir.resolve()}")

    if not target_data_dir.is_dir():
        logging.critical(f"Target data directory does not exist or is not a directory: {target_data_dir}")
        exit() # Exit the script if the data directory isn't found

    files_to_process = list(target_data_dir.glob('*.h5ad'))

    if not files_to_process:
        logging.warning(f"No .h5ad files found in target directory: {target_data_dir}")
    else:
        logging.info(f"Found {len(files_to_process)} .h5ad files to process.")

        for file_path in tqdm(files_to_process, desc="Processing files"):
            try:
                # Pass all required objects to the processing function
                process_file(
                    file_path, 
                    cache_dir, 
                    config, 
                    nfc_handler
                )
            except Exception as e:
                logging.error(f"An unexpected error occurred while processing {file_path.name}: {e}", exc_info=True)