# workflow_16s/api/ena/utils.py

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from Bio import Entrez

from workflow_16s.utils.logger import get_logger, with_logger

from .constants import (
    HOST_KEYWORDS, PATTERNS_TO_EXCLUDE
)
logger = get_logger("workflow_16s")
def optimize_dataframe_operations(
    samples: List[Dict], 
    experiments: List[Dict], 
    runs: List[Dict],
    studies: List[Dict], 
    biosamples_info: Dict[str, Dict]
) -> pd.DataFrame:
    """Optimized DataFrame creation using vectorized operations."""
    if not experiments: return pd.DataFrame()

    samples_df = pd.DataFrame(samples).add_prefix('sample_')
    experiments_df = pd.DataFrame(experiments).add_prefix('experiment_')
    studies_df = pd.DataFrame(studies).add_prefix('study_')

    if runs:
        runs_df = pd.DataFrame(runs).add_prefix('run_')
        runs_grouped = runs_df.groupby('run_experiment_accession').first()
    else:
        runs_grouped = pd.DataFrame()

    result_df = experiments_df.copy()

    # Optimized Merges
    if not samples_df.empty and 'sample_accession' in samples_df.columns:
        result_df = result_df.merge(
            samples_df, 
            left_on='experiment_sample_accession',
            right_on='sample_accession', 
            how='left'
        )

    if not studies_df.empty and 'study_study_accession' in studies_df.columns:
        result_df = result_df.merge(
            studies_df, 
            left_on='experiment_study_accession',
            right_on='study_study_accession', 
            how='left'
        )

    if not runs_grouped.empty:
        result_df = result_df.merge(
            runs_grouped, 
            left_on='experiment_experiment_accession',
            right_index=True, 
            how='left'
        )

    if biosamples_info:
        biosamples_df = pd.DataFrame.from_dict(biosamples_info, orient='index')
        if not biosamples_df.empty:
            result_df = result_df.merge(
                biosamples_df, 
                left_on='experiment_sample_accession',
                right_index=True, 
                how='left'
            )

    return result_df

@with_logger
def apply_filters_vectorized(df: pd.DataFrame, amplicon: bool = True) -> pd.DataFrame:
    """Fully vectorized filtering to prevent CPU hangs on large datasets."""
    if df.empty: return df

    logger.debug(f"Starting filtering with {len(df)} rows")
    
    # Amplicon filter
    if amplicon and 'experiment_library_strategy' in df.columns:
        count_before = len(df)
        df = df[df['experiment_library_strategy'].str.upper() == 'AMPLICON'].copy()
        removed = count_before - len(df)
        if removed > 0: logger.debug(f" ⟶ Amplicon filter removed {removed} rows")

    if df.empty: return df

    # 16S filter  
    count_before = len(df)
    target_cols = [
        'experiment_title', 
        'study_study_title', 
        'study_study_abstract', 
        'sample_description', 
        'experiment_design_description'
    ]
    cols_to_search = [c for c in target_cols if c in df.columns]
    
    if cols_to_search:
        # Search only relevant columns
        mask: pd.Series = df[cols_to_search].fillna('').apply(
            lambda x: x.str.contains('16S', case=False, na=False)
        ).any(axis=1)
        df = df[mask].copy()
        removed = count_before - len(df)
        if removed > 0: logger.debug(f" ⟶ '16S' keyword filter removed {removed} rows")

    if df.empty: return df

    # Host filter
    count_before = len(df)
    pattern = '|'.join(HOST_KEYWORDS)
    
    host_col = next(
        (c for c in [
            'biosample_scientific_name', 
            'sample_scientific_name', 
            'sample_host'
        ] if c in df.columns), 
        None
    )
    
    if host_col:
        mask = ~df[host_col].str.contains(pattern, case=False, na=False)
        df = df[mask].copy()
        removed = count_before - len(df)
        if removed > 0: 
            logger.debug(f" ⟶ Host filter (col: {host_col}) removed {removed} rows")

    # Metagenome filter
    if not df.empty:
        count_before = len(df)
        meta_pattern = '|'.join(PATTERNS_TO_EXCLUDE)
        
        meta_cols = [
            'sample_scientific_name', 
            'biosample_scientific_name', 
            'sample_description', 
            'study_study_title'
        ]
        present_cols = [c for c in meta_cols if c in df.columns]
        
        if present_cols:
            # Search only relevant columns
            exclude_mask: pd.Series = df[present_cols].fillna('').apply(
                lambda x: x.str.contains(meta_pattern, case=False, na=False)
            ).any(axis=1)
            df = df[~exclude_mask].copy()
            removed = count_before - len(df)
            if removed > 0: logger.debug(f" ⟶ Metagenome filter removed {removed} rows")

    logger.debug(f"Filtering complete. Remaining: {len(df)}")
    return df

@with_logger
def process_and_save_by_location(
    df: pd.DataFrame, 
    output_dir: Union[str, Path] = "results"
) -> None:
    """Saves the final DataFrame, splitting it by the original query location."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"\nSaving results to directory: '{output_path.resolve()}'")

    # Ensure query columns exist before grouping
    group_cols = ['query_lat', 'query_lon']
    if not all(col in df.columns for col in group_cols):
        logger.warning("Query columns ('query_lat', 'query_lon') not found. Saving all results to single file.")
        filepath = output_path / "all_location_data.csv"
        logger.info(f" ⟶ Saving {len(df)} records to '{filepath}'")
        df.to_csv(filepath, index=False)
        return

    # Group and save per location
    for (lat, lon), group_df in df.groupby(group_cols):
        # Sanitize lat/lon for filename
        lat_str = str(lat).replace('.', '_').replace('-', 'neg')
        lon_str = str(lon).replace('.', '_').replace('-', 'neg')
        filename = f"data_lat_{lat_str}_lon_{lon_str}.csv"
        filepath = output_path / filename
        logger.debug(f" ⟶ Saving {len(group_df)} records for ({lat}, {lon}) to '{filepath}'")
        group_df.to_csv(filepath, index=False)# workflow_16s/api/ena/metadata/utils.py

@with_logger
def process_and_structure_data(samples: List, runs: List, biosamples: Dict) -> pd.DataFrame:
    """Combines all fetched data sources into a single structured DataFrame."""
    if not runs: 
        return pd.DataFrame() # Runs are essential

    runs_df = pd.DataFrame(runs)
    samples_df = pd.DataFrame(samples) if samples else pd.DataFrame(columns=['sample_accession'])

    # --- NEW: ROBUST ID STANDARDIZATION ---
    # Public databases frequently change the column names for sample IDs.
    # We hunt for known fallbacks and rename them to 'sample_accession' to save the merge.
    potential_sample_ids = ['sample_accession', 'secondary_sample_accession', 'biosample', 'accession']
    
    for df_to_check in [runs_df, samples_df]:
        if not df_to_check.empty and 'sample_accession' not in df_to_check.columns:
            fallback_col = next((col for col in potential_sample_ids if col in df_to_check.columns), None)
            if fallback_col:
                logger.debug(f"Remapping column '{fallback_col}' to 'sample_accession' for merging.")
                df_to_check.rename(columns={fallback_col: 'sample_accession'}, inplace=True)

    # Do the exact same thing for experiment accessions
    potential_exp_ids = ['experiment_accession', 'study_accession']
    for df_to_check in [runs_df, samples_df]:
        if not df_to_check.empty and 'experiment_accession' not in df_to_check.columns:
            fallback_col = next((col for col in potential_exp_ids if col in df_to_check.columns), None)
            if fallback_col:
                logger.debug(f"Remapping column '{fallback_col}' to 'experiment_accession' for merging.")
                df_to_check.rename(columns={fallback_col: 'experiment_accession'}, inplace=True)

    # Ensure essential columns exist
    has_sample_acc = 'sample_accession' in runs_df.columns and 'sample_accession' in samples_df.columns
    has_exp_acc = 'experiment_accession' in runs_df.columns and 'experiment_accession' in samples_df.columns

    # 1. Merge Runs and Samples
    if has_sample_acc:
        # OPTIMIZATION: Deduplicate to prevent Cartesian product explosion
        samples_df = samples_df.drop_duplicates(subset=['sample_accession'])
        df = pd.merge(runs_df, samples_df, on='sample_accession', how='left', suffixes=('', '_sample'))
        logger.debug("Merged samples based on 'sample_accession'.")
        
    elif has_exp_acc:
        samples_df = samples_df.drop_duplicates(subset=['experiment_accession'])
        df = pd.merge(runs_df, samples_df, on='experiment_accession', how='left', suffixes=('', '_sample'))
        logger.debug("Merged samples based on 'experiment_accession'.")
        
    else:
        logger.warning("Cannot reliably merge samples without 'sample_accession' or 'experiment_accession'. Returning run data only.")
        df = runs_df 

    # 2. Merge BioSamples data if available
    if biosamples and 'sample_accession' in df.columns:
        
        # OPTIMIZATION: Create a set ONCE for O(1) extremely fast lookups
        valid_accessions = set(df['sample_accession'].dropna())
        
        # Filter dict using the pre-calculated set (and drop empty values)
        relevant_biosamples = {k: v for k, v in biosamples.items() if k in valid_accessions and v}
        
        if relevant_biosamples:
            biosamples_df = pd.DataFrame.from_dict(relevant_biosamples, orient='index')
            
            # OPTIMIZATION: Add suffixes to prevent column name collisions from BioSamples
            df = pd.merge(df, biosamples_df, left_on='sample_accession', right_index=True, how='left', suffixes=('', '_biosample'))
            logger.debug(f"Merged {len(biosamples_df)} BioSample records.")
        else: 
            logger.debug("No relevant BioSample records found for the current sample accessions.")
            
    elif biosamples: 
        logger.warning("Cannot merge BioSamples data as 'sample_accession' column is missing after initial merge.")

    return df

@with_logger
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

@with_logger
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

@with_logger
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