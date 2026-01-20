# workflow_16s/api/ena/utils.py

# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Dict, List, Union

# Third Party Imports
import pandas as pd

# Local Imports
# This relative import should work if constants.py is in the same directory (ena/)
from workflow_16s.api.ena.constants import PATTERNS_TO_EXCLUDE
from workflow_16s.utils.logger import get_logger

# ==================================================================================== #

logger = get_logger()

# ==================================================================================== #

def optimize_dataframe_operations(
    samples: List[Dict], experiments: List[Dict], runs: List[Dict],
    studies: List[Dict], biosamples_info: Dict[str, Dict]
) -> pd.DataFrame:
    """Optimized DataFrame creation using vectorized operations."""
    if not experiments: return pd.DataFrame()

    samples_df = pd.DataFrame(samples).add_prefix('sample_')
    experiments_df = pd.DataFrame(experiments).add_prefix('experiment_')
    studies_df = pd.DataFrame(studies).add_prefix('study_')

    if runs:
        runs_df = pd.DataFrame(runs).add_prefix('run_')
        # Aggregate runs: take first if unique, list if multiple runs per experiment
        runs_grouped = runs_df.groupby('run_experiment_accession').agg(
            # Use a lambda that handles potential non-unique single values correctly
            lambda x: x.iloc[0] if x.nunique() <= 1 else list(x.unique())
        )
    else:
        runs_grouped = pd.DataFrame() # Ensure it exists even if empty

    result_df = experiments_df.copy()

    # Merge Samples
    if not samples_df.empty and 'sample_accession' in samples_df.columns and 'experiment_sample_accession' in result_df.columns:
        result_df = result_df.merge(samples_df, left_on='experiment_sample_accession',
                                    right_on='sample_accession', how='left', suffixes=('', '_sample_dup'))
        # Drop redundant sample_accession if it exists
        if 'sample_accession_sample_dup' in result_df.columns:
            result_df.drop(columns=['sample_accession_sample_dup'], inplace=True)


    # Merge Studies
    if not studies_df.empty and 'study_study_accession' in studies_df.columns and 'experiment_study_accession' in result_df.columns:
        result_df = result_df.merge(studies_df, left_on='experiment_study_accession',
                                    right_on='study_study_accession', how='left', suffixes=('', '_study_dup'))
        # Drop redundant study_accession if it exists
        if 'study_study_accession_study_dup' in result_df.columns:
            result_df.drop(columns=['study_study_accession_study_dup'], inplace=True)

    # Merge Runs
    if not runs_grouped.empty and 'experiment_experiment_accession' in result_df.columns:
        result_df = result_df.merge(runs_grouped, left_on='experiment_experiment_accession',
                                    right_index=True, how='left') # Use right_index for merge on index

    # Merge BioSamples
    if biosamples_info and 'experiment_sample_accession' in result_df.columns:
        # Filter biosamples_info for keys actually present in the experiments
        relevant_keys = result_df['experiment_sample_accession'].dropna().unique()
        filtered_biosamples = {k: v for k, v in biosamples_info.items() if k in relevant_keys}
        if filtered_biosamples:
            biosamples_df = pd.DataFrame.from_dict(filtered_biosamples, orient='index')
            if not biosamples_df.empty:
                result_df = result_df.merge(biosamples_df, left_on='experiment_sample_accession',
                                            right_index=True, how='left') # Use right_index

    return result_df


def apply_filters_vectorized(df: pd.DataFrame, amplicon: bool = True) -> pd.DataFrame:
    """Apply all filters using vectorized operations for better performance."""
    if df.empty: return df

    logger.debug(f"Starting filtering with {len(df)} rows")
    original_count = len(df)

    # --- Amplicon Filter ---
    # Use 'experiment_library_strategy' for amplicon filtering
    strategy_col = 'experiment_library_strategy'
    if amplicon and strategy_col in df.columns:
        count_before = len(df)
        # Ensure comparison is robust against non-string types or case variations
        mask = df[strategy_col].astype(str).str.upper() == 'AMPLICON'
        df = df[mask].copy() # Use .copy() to avoid SettingWithCopyWarning
        removed = count_before - len(df)
        if removed > 0: logger.debug(f" -> Amplicon filter removed {removed} rows")
    elif amplicon:
        logger.warning(f"Amplicon filter requested but '{strategy_col}' column missing.")

    # --- 16S Filter ---
    # Search across all columns for '16S' (case-insensitive)
    if not df.empty:
        count_before = len(df)
        # Convert to string and check for '16S' substring
        mask = df.astype(str).apply(lambda col: col.str.contains('16S', case=False, na=False)).any(axis=1)
        df = df[mask].copy()
        removed = count_before - len(df)
        if removed > 0: logger.debug(f" -> '16S' keyword filter removed {removed} rows")

    # --- Host Filter (Simplified: Exclude Human 'Homo sapiens' and common model organisms) ---
    # Uses 'sample_scientific_name' or 'biosample_scientific_name' if available
    host_col = None
    if 'biosample_scientific_name' in df.columns: host_col = 'biosample_scientific_name'
    elif 'sample_scientific_name' in df.columns: host_col = 'sample_scientific_name' # Fallback
    elif 'sample_host' in df.columns: host_col = 'sample_host' # Another fallback

    if not df.empty and host_col:
        count_before = len(df)
        # List common hosts/models to exclude (adjust as needed)
        exclude_hosts = ['homo sapiens', 'mus musculus', 'rattus norvegicus', 'metazoa', 'neotoma']
        pattern = '|'.join(exclude_hosts)
        # Keep rows where the host column does NOT contain these patterns (case-insensitive)
        mask = ~df[host_col].astype(str).str.contains(pattern, case=False, na=False)
        df = df[mask].copy()
        removed = count_before - len(df)
        if removed > 0: logger.debug(f" -> Host filter (col: {host_col}) removed {removed} rows")
    elif not host_col:
        logger.debug(" -> Host filter skipped: No suitable host column found.")


    # --- Metagenome Filter (using PATTERNS_TO_EXCLUDE) ---
    # Checks multiple columns where metagenome info might appear
    metagenome_cols_to_check = [
        'sample_scientific_name', 'biosample_scientific_name',
        'sample_description', 'biosample_description',
        'study_study_title', 'study_study_abstract' # Added study fields
    ]
    cols_present = [col for col in metagenome_cols_to_check if col in df.columns]

    if not df.empty and cols_present:
        count_before = len(df)
        pattern = '|'.join(PATTERNS_TO_EXCLUDE) # Uses the imported list
        # Create a combined mask: True if ANY checked column contains an excluded pattern
        combined_mask = pd.Series(False, index=df.index)
        for col in cols_present:
            combined_mask |= df[col].astype(str).str.contains(pattern, case=False, na=False)

        # Keep rows where the combined mask is FALSE (i.e., NO excluded patterns found)
        df = df[~combined_mask].copy()
        removed = count_before - len(df)
        if removed > 0: logger.debug(f" -> Metagenome exclusion filter removed {removed} rows")

    final_count = len(df)
    total_removed = original_count - final_count
    logger.debug(f"Filtering complete. Removed {total_removed} rows, {final_count} remaining.")

    return df

def process_and_save_by_location(df, output_dir: Union[str, Path] = "results"):
    """Saves the final DataFrame, splitting it by the original query location."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"\nSaving results to directory: '{output_path.resolve()}'")

    # Ensure query columns exist before grouping
    group_cols = ['query_lat', 'query_lon']
    if not all(col in df.columns for col in group_cols):
        logger.warning("Query columns ('query_lat', 'query_lon') not found. Saving all results to single file.")
        filepath = output_path / "all_location_data.csv"
        logger.info(f" -> Saving {len(df)} records to '{filepath}'")
        df.to_csv(filepath, index=False)
        return

    # Group and save per location
    for (lat, lon), group_df in df.groupby(group_cols):
        # Sanitize lat/lon for filename
        lat_str = str(lat).replace('.', '_').replace('-', 'neg')
        lon_str = str(lon).replace('.', '_').replace('-', 'neg')
        filename = f"data_lat_{lat_str}_lon_{lon_str}.csv"
        filepath = output_path / filename
        logger.debug(f"   -> Saving {len(group_df)} records for ({lat}, {lon}) to '{filepath}'")
        group_df.to_csv(filepath, index=False)
        
        