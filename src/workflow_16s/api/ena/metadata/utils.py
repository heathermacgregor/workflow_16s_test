# workflow_16s/api/ena/metadata/utils.py

# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

# Third Party Imports
import pandas as pd
from Bio import Entrez

# Local Imports
from workflow_16s.utils.logger import get_logger

# ==================================================================================== #

logger = get_logger()

# ==================================================================================== #

def process_and_structure_data(samples: List, runs: List, biosamples: Dict) -> pd.DataFrame:
    """Combines all fetched data sources into a single structured DataFrame."""
    if not runs: return pd.DataFrame() # Runs are essential

    runs_df = pd.DataFrame(runs)
    # Ensure samples list is not None before creating DataFrame
    samples_df = pd.DataFrame(samples) if samples else pd.DataFrame(columns=['sample_accession']) # Add default empty DF

    # Ensure essential columns exist even if empty
    if 'sample_accession' not in runs_df.columns:
        logger.warning("Run data missing 'sample_accession'. Cannot merge samples effectively.")
        # Attempt merge on experiment_accession if available, otherwise return runs_df
        if 'experiment_accession' in runs_df.columns and 'experiment_accession' in samples_df.columns:
            df = pd.merge(runs_df, samples_df, on='experiment_accession', how='left', suffixes=('', '_sample'))
            logger.debug("Merged samples based on 'experiment_accession'.")
        else:
            logger.warning("Cannot reliably merge samples without 'sample_accession' or 'experiment_accession'. Returning run data only.")
            return runs_df # Cannot merge samples reliably
    else:
        # Standard merge: runs with samples
        df = pd.merge(runs_df, samples_df, on='sample_accession', how='left', suffixes=('', '_sample'))
        logger.debug("Merged samples based on 'sample_accession'.")

    # Merge BioSamples data if available
    if biosamples:
        # Important: Filter biosamples dict for keys present in df['sample_accession']
        relevant_biosamples = {k: v for k, v in biosamples.items() if k in df['sample_accession'].unique()}
        if relevant_biosamples:
            biosamples_df = pd.DataFrame.from_dict(relevant_biosamples, orient='index')
            # Ensure biosamples_df is not empty and the merge key exists in df
            if not biosamples_df.empty and 'sample_accession' in df.columns:
                df = pd.merge(df, biosamples_df, left_on='sample_accession', right_index=True, how='left')
                logger.debug(f"Merged {len(biosamples_df)} BioSample records.")
            elif 'sample_accession' not in df.columns: logger.warning("Cannot merge BioSamples data as 'sample_accession' column is missing after initial merge.")
        else: logger.debug("No relevant BioSample records found for the current sample accessions.")

    return df

def get_tax_name(tax_id: int) -> str:
    """Fetches the scientific name for a given NCBI Taxonomy ID (Synchronous)."""
    # Note: This is synchronous and potentially slow if called many times.
    # Consider batching or using the async fetcher if performance is critical.
    try:
        # Add slight delay to respect NCBI rate limits even for single calls
        import time
        time.sleep(0.4) # ~3 requests per second limit
        handle = Entrez.efetch(db="taxonomy", id=str(tax_id), retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        if records and isinstance(records, list) and len(records) > 0:
            if isinstance(records[0], dict) and "ScientificName" in records[0]:  return records[0]["ScientificName"]
            else: logger.warning(f"Unexpected record structure for tax ID {tax_id}: {records[0]}"); return "Name parse error"
        else: logger.warning(f"No records found for tax ID {tax_id}."); return "Name not found"
    except Exception as e: logger.warning(f"Entrez error fetching tax name for {tax_id}: {e}"); return "Fetch error"


def apply_filters(
    df: pd.DataFrame, amplicon_filter: bool, no_host_filter: bool
) -> pd.DataFrame:
    """Filters DataFrame for amplicon and non-host samples with logging."""
    if df.empty: return df

    filtered_df = df.copy()
    initial_count = len(filtered_df)
    logger.info(f"Applying filters to {initial_count} initial records...")

    # Apply amplicon filter (using experiment_library_strategy if available)
    strategy_col = 'library_strategy' # Assuming this comes from runs or experiments
    if amplicon_filter and strategy_col in filtered_df.columns:
        count_before = len(filtered_df)
        # Handle potential lists in library_strategy if multiple runs merged weirdly
        # Convert to string first to handle potential non-string types safely
        mask = filtered_df[strategy_col].astype(str).str.upper() == 'AMPLICON'
        filtered_df = filtered_df[mask]
        count_after = len(filtered_df)
        removed_count = count_before - count_after
        if removed_count > 0:
            logger.info(f" ⤷ Amplicon filter ('{strategy_col}') removed {removed_count} non-amplicon records.")
    elif amplicon_filter: logger.warning(f"Amplicon filter requested but '{strategy_col}' column not found.")

    # Apply no-host filter (using tax_id if available)
    # This assumes 'tax_id' comes from the initial sample query
    tax_id_col = 'tax_id'
    if no_host_filter and tax_id_col in filtered_df.columns:
        count_before = len(filtered_df)
        # Create a mask for rows to KEEP (tax_id is NaN, empty, or NOT human (9606))
        # Convert to numeric, coercing errors to NaN. This handles strings, NaNs etc.
        numeric_tax_id = pd.to_numeric(filtered_df[tax_id_col], errors='coerce')
        # Keep NaNs (likely environmental) and non-human IDs
        # Also handle potential 0 tax_ids if they mean "unclassified" or similar
        host_mask = numeric_tax_id.isna() | ((numeric_tax_id != 9606) & (numeric_tax_id != 0))

        # Log removed hosts (optional, can be slow if many unique hosts)
        removed_hosts_df = filtered_df[~host_mask] # Select rows that were removed
        removed_count = len(removed_hosts_df)
        if removed_count > 0:
            unique_removed_ids = removed_hosts_df[tax_id_col].dropna().unique()
            # Filter out non-numeric IDs before logging
            loggable_ids = [int(tid) for tid in unique_removed_ids if pd.notna(tid) and str(tid).isdigit()]
            logger.info(f" ⤷ No-host filter ('{tax_id_col}') removed {removed_count} host-associated records (Tax IDs: {loggable_ids})")
            # Example of getting names (can be slow):
            # for tid in loggable_ids[:3]: # Log first few examples
            #      logger.info(f"   - Host example: {get_tax_name(tid)} (ID: {tid})")

        filtered_df = filtered_df[host_mask] # Apply the mask to keep desired rows

    elif no_host_filter: logger.warning(f"No-host filter requested but '{tax_id_col}' column not found.")

    final_count = len(filtered_df)
    total_removed = initial_count - final_count
    logger.info(f"➤ Filtering complete. Total removed: {total_removed}, Remaining: {final_count}.")

    return filtered_df


def save_results_by_location(df: pd.DataFrame, output_dir: Path):
    """Saves the final DataFrame, splitting files by query location."""
    if df.empty: logger.warning("No data to save."); return
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to: '{output_dir.resolve()}'")

    # Check if query columns exist before grouping
    group_cols = ['query_lat', 'query_lon']
    if not all(col in df.columns for col in group_cols):
        logger.warning("Query location columns ('query_lat', 'query_lon') not found. Saving all results to a single file.")
        filename = "all_locations_data.csv"
        filepath = output_dir / filename
        logger.info(f" ⤷ Saving {len(df)} records to '{filepath}'")
        try: df.to_csv(filepath, index=False)
        except Exception as e: logger.error(f"Failed to save file {filepath}: {e}")
        return

    # Group and save per location
    for (lat, lon), group_df in df.groupby(group_cols):
        # Sanitize lat/lon for filename - ensure they are strings first
        lat_str = str(lat).replace('.', '_').replace('-', 'neg')
        lon_str = str(lon).replace('.', '_').replace('-', 'neg')
        filename = f"data_lat_{lat_str}_lon_{lon_str}.csv"
        filepath = output_dir / filename
        logger.info(f" ⤷ Saving {len(group_df)} records for ({lat}, {lon}) to '{filepath}'")
        try: group_df.to_csv(filepath, index=False)
        except Exception as e: logger.error(f"Failed to save file {filepath}: {e}")