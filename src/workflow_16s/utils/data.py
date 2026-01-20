# ==================================================================================== #

# Standard Imports
import logging
import re
from typing import Dict, Iterable, List, Optional, Tuple, Union

# Third-Party Imports
import pandas as pd
import numpy as np
from biom.table import Table
from pandarallel import pandarallel
from skbio.stats.composition import clr as CLR

# Local Imports
from workflow_16s.constants import (
    MIN_REL_ABUNDANCE, MIN_SAMPLES, MIN_COUNTS, PSEUDOCOUNT, GROUP_THRESHOLD, 
    PREVALENCE_THRESHOLD, DEFAULT_GROUP_COLUMN, DEFAULT_N, TAXONOMIC_LEVELS,
    SAMPLE_ID_COLUMN
)
from workflow_16s.utils.biom_utils import to_biom
from workflow_16s.utils.progress import get_progress_bar

# ==================================================================================== #

FEATURE_PATTERNS = {
    "taxonomic": re.compile(
        r'^d__[\w]+(;p__[\w]+)?(;c__[\w]+)?(;o__[\w]+)?'
        r'(;f__[\w]+)?(;g__[\w]+)?(;s__[\w]+)?$'
    ),
    "hashes": re.compile(r'^[a-f0-9]{32}$|^[a-f0-9]{64}$'),
    "raw_sequences": re.compile(r'^[ACGTRYSWKMBDHVN]+$', re.IGNORECASE)
}

# ==================================================================================== #

logger = logging.getLogger("workflow_16s")

# ================================ TABLE FILTERING =================================== #

def filter(
    table: Union[dict, Table, pd.DataFrame], min_rel_abundance: float = MIN_REL_ABUNDANCE, 
    min_samples: int = MIN_SAMPLES, min_counts: int = MIN_COUNTS
) -> Table:
    """
    Filter features and samples with strict type enforcement.
    
    Applies two-step filtering:
        1. Feature filtering (min_rel_abundance and min_samples)
        2. Sample filtering (min_counts)
    
    Args:
        table:             Input table
        min_rel_abundance: Minimum relative abundance (%) for feature retention.
        min_samples:       Minimum samples where feature must appear.
        min_counts:        Minimum total counts per sample.
        
    Returns:
        Filtered BIOM Table.
    """
    table = to_biom(table)
    table = filter_features(table, min_rel_abundance, min_samples)
    table = filter_samples(table, min_counts)
    return table
    

def filter_features(
    table: Union[dict, Table, pd.DataFrame], min_rel_abundance: float, 
    min_samples: int
) -> Table:
    """
    Filter features based on prevalence and abundance.
    
    Args:
        table:             BIOM Table to filter.
        min_rel_abundance: Minimum relative abundance (%).
        min_samples:       Minimum samples where feature must appear.
        
    Returns:
        Filtered BIOM Table.
    """
    table = to_biom(table)
    min_abs_abundance = min_rel_abundance / 100
    
    # Convert to DataFrame for vectorized operations
    table = to_biom(table)
    df = table.to_dataframe().astype(float)
    
    # Calculate filtering criteria
    max_per_feature = df.max(axis=1)
    non_zero_per_feature = (df > 0).sum(axis=1)
    
    # Create feature mask
    feature_mask = (max_per_feature >= min_abs_abundance) & (non_zero_per_feature >= min_samples)
    
    # Apply filtering
    feature_ids = table.ids(axis='observation')
    ids_to_keep = [fid for fid, keep in zip(feature_ids, feature_mask) if keep]
    
    return table.filter(ids_to_keep, axis='observation')
    

def filter_samples(
    table: Union[dict, Table, pd.DataFrame], min_counts: int
) -> Table:
    """
    Filter samples based on minimum total counts.
    
    Args:
        table:      BIOM Table to filter.
        min_counts: Minimum total counts per sample.
        
    Returns:
        Filtered BIOM Table.
    """
    # Convert to DataFrame for vectorized operations
    table = to_biom(table)
    df = table.to_dataframe().astype(float)
    
    # Calculate total counts per sample
    total_per_sample = df.sum(axis=0)
    
    # Create sample mask
    sample_mask = total_per_sample >= min_counts
    
    # Apply filtering
    sample_ids = table.ids(axis='sample')
    ids_to_keep = [sid for sid, keep in zip(sample_ids, sample_mask) if keep]
    
    return table.filter(ids_to_keep, axis='sample')


# ========================== TABLE NORMALIZATION & TRANSFORM ========================= #

def normalize(table: Union[dict, Table, pd.DataFrame], axis: int = 1) -> Table:
    """
    Normalize table to relative abundance with strict type enforcement.
    
    Args:
        table: Input table.
        axis:  Normalization axis (0=features, 1=samples).
        
    Returns:
        Normalized BIOM Table.
        
    Raises:
        ValueError: For invalid axis values.
    """
    if axis not in (0, 1):
        raise ValueError(f"Invalid axis: {axis}. Must be 0 (features) or 1 (samples)")

    biom_table = to_biom(table)
    # Sample-wise normalization (convert to relative abundance)
    if axis == 1: return biom_table.norm(axis='sample')
    # Feature-wise normalization
    else: return biom_table.norm(axis='observation')
        

def clr(
    table: Union[dict, Table, pd.DataFrame],
    pseudocount: Optional[float] = None,
    handle_zeros: bool = True,
    zero_method: str = 'multiplicative'
) -> Table:
    """
    Apply centered log-ratio (CLR) transformation with proper zero handling.
    
    **UPDATED (Jan 2026):** Now uses multiplicative replacement by default
    for statistically correct compositional data analysis.
    
    Args:
        table:        Input table.
        pseudocount:  DEPRECATED - Legacy pseudocount. If provided, overrides handle_zeros.
        handle_zeros: Use multiplicative replacement (recommended: True).
        zero_method:  Zero replacement method ('multiplicative' recommended).
        
    Returns:
        CLR-transformed BIOM Table.
        
    References:
        Martin-Fernández et al. (2003). Dealing with zeros in compositional data.
        Gloor et al. (2017). Microbiome Datasets Are Compositional.
    """
    # Use new compositional-aware implementation
    from workflow_16s.utils.compositional import clr_table
    
    return clr_table(
        table=table,
        handle_zeros=handle_zeros,
        zero_method=zero_method,
        pseudocount=pseudocount if pseudocount is not None else None
    )


# TODO: Integrate into workflow
def classify_feature_format(
    cols: Iterable[str], verbose: bool = False
) -> Dict[str, int]:
    """
    Classify feature IDs into taxonomic, hash, sequence, or unknown types.
    
    Uses regex patterns to identify:
        - Taxonomic strings (e.g., 'd__Bacteria;p__Firmicutes')
        - QIIME-style hashes (32/64 character hex strings)
        - IUPAC nucleotide sequences
        - Unknown patterns
    
    Args:
        cols:    Feature IDs to classify.
        verbose: Verbosity flag.
        
    Returns:
        Dictionary with counts for each category.
    """
    counts = {k: 0 for k in FEATURE_PATTERNS}
    counts["unknown"] = 0

    if verbose:
        for col in map(str, cols):
            col = col.strip()
            if not col: counts["unknown"] += 1; continue
                
            matched = False
            for name, pattern in FEATURE_PATTERNS.items():
                if pattern.match(col): counts[name] += 1; matched = True; break
                    
            if not matched: counts["unknown"] += 1
                
        # Print classification summary
        total = sum(counts.values())
        if total > 0:
            dominant = max(counts, key=lambda k: counts[k])
            confidence = counts[dominant] / total
            logger.info(f"Feature classification: {dominant} ({confidence:.0%} confidence)")
    else: 
        with get_progress_bar() as progress:
            task = progress.add_task("Classifying feature IDs...", total=len(list(map(str, cols))))
            for col in map(str, cols):
                try: 
                    col = col.strip()
                    if not col: counts["unknown"] += 1; continue
                        
                    matched = False
                    for name, pattern in FEATURE_PATTERNS.items():
                        if pattern.match(col): counts[name] += 1; matched = True; break
                            
                    if not matched: counts["unknown"] += 1
                except Exception as e:
                    logger.error(f"Classification failed for feature ID {col}: {e!r}")
                finally: progress.update(task, advance=1)
        
    return counts


# TODO: Integrate into workflow
def trim_and_merge_asvs(
    asv_table: pd.DataFrame, asv_seqs: List[str], trim_len: int = 250, 
    n_workers: int = 8, verbose: bool = False, progress: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Trim ASV sequences and merge identical trimmed variants with Rich progress tracking.
    
    Args:
        asv_table:        Feature table (features x samples).
        asv_sequences:    Raw sequences corresponding to features.
        trim_length:      Number of bases to keep from sequence start.
        n_workers:        Number of parallel workers.
        verbose:          Verbosity flag.
        progress:         Whether to show progress bar.
        
    Returns:
        Tuple:
            - Merged feature table (features x samples)
            - Series of trimmed sequences
            - Temporary feature IDs
    """
    # Validate inputs
    if len(asv_seqs) != asv_table.shape[0]:
        raise ValueError(f"ASV count mismatch: {len(asv_seqs)} sequences vs {asv_table.shape[0]} features")
    if trim_len < 1: raise ValueError(f"Invalid trim length: {trim_len}")

    # Initialize parallel processing WITHOUT pandarallel progress
    pandarallel.initialize(nb_workers=n_workers, progress_bar=False)
    
    # Get progress bar context manager (assume this is defined elsewhere)
    with get_progress_bar() as progress_bar:
        # Sequence trimming
        trim_task = progress_bar.add_task(
            "Trimming sequences", total=len(asv_seqs), visible=progress
        )
        
        # Process in chunks to update progress
        chunk_size = max(100, len(asv_seqs) // 100)  # 100 chunks max
        trimmed_seqs = []
        
        for start_idx in range(0, len(asv_seqs), chunk_size):
            end_idx = start_idx + chunk_size
            chunk = asv_seqs[start_idx:end_idx]
            
            # Process chunk in parallel
            try:
                chunk_trimmed = (pd.Series(chunk).apply(lambda x: x[:trim_len]))
                trimmed_seqs.extend(chunk_trimmed)
            except Exception as e:
                logger.error(f"Sequence trimming failed for chunk starting at index {start_idx}: {e!r}")
                trimmed_seqs.extend([''] * len(chunk))  # Placeholder for failed trims
            finally: progress_bar.update(
                trim_task,advance=len(chunk),
                description=f"Trimming sequences ({start_idx+len(chunk)}/{len(asv_seqs)})"
            )
        
        trimmed_seqs = pd.Series(trimmed_seqs)
        
        # BIOM table creation
        table_task = progress_bar.add_task(
            "Creating BIOM table", total=1, visible=progress
        )
        
        # Create BIOM table with temporary feature IDs
        obs_ids = [f"TMP_FEATURE_{i}" for i in range(len(asv_seqs))]
        biom_table = Table(
            asv_table.values.astype(np.uint32),
            observation_ids=obs_ids,
            sample_ids=asv_table.columns.tolist(),
            observation_metadata=[{'trimmed_seq': s} for s in trimmed_seqs]
        )
        progress_bar.update(table_task, advance=1)
        
        # Collapsing features
        collapse_task = progress_bar.add_task("Collapsing sequences", total=1, visible=progress)
        
        # Collapse features by trimmed sequences
        merged_biom = biom_table.collapse(
            lambda id_, md: md['trimmed_seq'], axis='observation', norm=False, min_group_size=1,
            include_collapsed_metadata=False
        )
        progress_bar.update(collapse_task, advance=1)

    # Convert to DataFrame
    merged_df = merged_biom.to_dataframe(dense=True).astype(np.uint32)
    
    # Log reduction statistics
    orig = len(asv_seqs)
    new = merged_df.shape[0]
    if verbose: logger.info(f"Feature reduction: {orig} → {new} ({new/orig:.1%}) after {trim_len}bp trim")
    return merged_df, trimmed_seqs, obs_ids


def collapse_taxa(
    table: Union[pd.DataFrame, Table], target_level: str, progress = None, task_id = None
) -> Table:
    """
    Collapse feature table to specified taxonomic level.
    
    Args:
        table:        Input BIOM Table or DataFrame.
        target_level: Taxonomic level to collapse to (phylum/class/order/family).
        output_dir:   Directory to save collapsed table.
    
    Returns:
        Collapsed BIOM Table.
    
    Raises:
        ValueError: For invalid target_level.
    """
    
    table = table.copy()
    table = to_biom(table)
        
    if target_level not in TAXONOMIC_LEVELS:
        raise ValueError(f"Invalid `target_level`: {target_level}. Expected one of {list(TAXONOMIC_LEVELS.keys())}")

    level_idx = TAXONOMIC_LEVELS[target_level]

    # Create taxonomy mapping
    id_map = {}
    sub_task = None
    if progress is not None:
        sub_task = progress.add_task( 
            "Feature:", parent=task_id,
            total=len(table.ids(axis='observation').astype(str))
        )
    for taxon in table.ids(axis='observation').astype(str):
        try:
            new_desc = f"Feature: {taxon}"
            if len(new_desc) > DEFAULT_N:
                new_desc = f"{new_desc[:DEFAULT_N-3]}..."
            if progress is not None and sub_task is not None:
                progress.update(sub_task, description=new_desc)
            parts = taxon.split(';')
            truncated = ';'.join(parts[:level_idx + 1]) if len(parts) >= level_idx + 1 else 'Unclassified'
            id_map[taxon] = truncated
        except Exception as e:
            logger.error(f"Mapping failed for taxon {taxon}: {e!r}")
        finally: pass
    collapsed_table = table.collapse(
        lambda id, _: id_map.get(id, 'Unclassified'), norm=False, axis='observation',
        include_collapsed_metadata=False
    ).remove_empty()
    if progress is not None and sub_task is not None: progress.remove_task(sub_task)
    return collapsed_table


def presence_absence(table: Union[Table, pd.DataFrame]) -> Table:
    """Convert table to presence/absence format and filter by abundance.
    
    Args:
        table: Input BIOM Table or DataFrame.
    
    Returns:
        Presence/absence BIOM Table filtered by abundance.
    """
    table = table.copy()
    table = to_biom(table)
    
    # Filter by abundance
    feature_sums = np.array(table.sum(axis='observation')).flatten()
    sorted_idx = np.argsort(feature_sums)[::-1]
    cumulative = np.cumsum(feature_sums[sorted_idx]) / feature_sums.sum()
    stop_idx = np.searchsorted(cumulative, 0.99) + 1
    keep_ids = [table.ids(axis='observation')[i] for i in sorted_idx[:stop_idx]]
    
    # Convert to presence/absence
    pa_table = table.pa(inplace=False)
    pa_table_filtered = pa_table.filter(keep_ids, axis='observation')
    pa_df_filtered = pa_table_filtered.to_dataframe(dense=True)

    return Table(
        pa_df_filtered.values, pa_df_filtered.index, pa_df_filtered.columns, 
        table_id='Presence Absence Table'
    )


def filter_presence_absence(
    table: Table, metadata: pd.DataFrame, col: str = DEFAULT_GROUP_COLUMN, 
    prevalence_threshold: float = PREVALENCE_THRESHOLD, 
    group_threshold: float = GROUP_THRESHOLD
) -> Table:
    """
    Filter presence/absence table based on prevalence and group differences.
    
    Args:
        table:                Input BIOM Table.
        metadata:             Sample metadata DataFrame.
        col:                  Metadata column to group by.
        prevalence_threshold: Minimum prevalence across all samples.
        group_threshold:      Minimum prevalence difference between groups.
    
    Returns:
        Filtered BIOM Table.
    """
    df = table.to_dataframe(dense=True).T
    metadata = metadata.set_index("run_accession.1")
    df_with_meta = df.join(metadata[[col]], how='inner')

    # Apply prevalence filter
    if prevalence_threshold:
        species_data = df_with_meta.drop(columns=[col])
        prev = species_data.mean(axis=0)
        filtered_species = prev[prev >= prevalence_threshold].index
        df_with_meta = df_with_meta[filtered_species.union(pd.Index([col]))]

    # Apply group filter
    if group_threshold:
        groups = df_with_meta.groupby(col)
        if True not in groups.groups or False not in groups.groups:
            raise ValueError(f"Metadata column `{col}` must have True/False groups")
        sum_per_group = groups.sum(numeric_only=True)
        n_samples = groups.size()
        percentages = sum_per_group.div(n_samples, axis=0)
        mask = (percentages.loc[True] >= group_threshold) & (percentages.loc[False] >= group_threshold)
        selected_species = mask.index[mask] # type: ignore ????????????
        df_with_meta = df_with_meta[selected_species.union(pd.Index([col]))]

    return Table(
        df_with_meta.drop(columns=[col]).values.T,
        df_with_meta.columns.tolist(), df_with_meta.index.tolist(),
        table_id='Filtered Presence/Absence Table'
    )


def sync_samples(table: Table, metadata: pd.DataFrame) -> Tuple[Table, pd.DataFrame]:
    """
    Aligns a BIOM table and a metadata DataFrame case-insensitively,
    retaining only common samples.

    Args:
        table:    The BIOM table object containing feature counts.
        metadata: The pandas DataFrame containing sample metadata.

    Returns:
        A tuple of (filtered_table, filtered_metadata) containing only samples
        present in both inputs, matched case-insensitively.
    
    Raises:
        KeyError: If no valid sample ID column is found in the metadata.
    """
    potential_id_columns = ['#SampleID', 'sample_id', 'SampleID', 'sample-id', 
                            'sampleid', '#sampleid']
    found_id_column = next(
        (col for col in potential_id_columns if col in metadata.columns),
        None
    )

    if not found_id_column:
        raise KeyError(
            "Could not find a valid sample ID column in the metadata.\n"
            f" ⤷ Tried looking for: {potential_id_columns}\n"
            f" ⤷ Available columns are: {metadata.columns.tolist()}"
        )
    logger.debug(f"Found sample ID column: '{found_id_column}'. Renaming to '{SAMPLE_ID_COLUMN}'.")
    metadata.rename(columns={found_id_column: SAMPLE_ID_COLUMN}, inplace=True)
    found_id_column = SAMPLE_ID_COLUMN
    
    # Case-insensitive matching
    # Create lookup maps from lowercase IDs to original cased IDs
    metadata_id_map = {str(orig_id).lower(): str(orig_id) for orig_id in metadata[found_id_column]}
    table_id_map = {str(orig_id).lower(): str(orig_id) for orig_id in table.ids(axis='sample')}

    # Warn if case-variant duplicates exist within a single file (e.g., 'SampleA' and 'samplea')
    if len(metadata_id_map) < len(metadata):
        logger.warning("Metadata contains duplicate sample IDs when ignoring case. One version will be kept per match.")
    if table.shape:
        n_samples = table.shape[1]
        if len(table_id_map) < n_samples:
            logger.warning("Feature table contains duplicate sample IDs when ignoring case. One version will be kept per match.")

    # Find the intersection of the lowercase keys
    common_lowercase_ids = set(metadata_id_map.keys()).intersection(table_id_map.keys())

    if not common_lowercase_ids:
        logger.warning("No common samples found between the table and metadata. Returning empty objects.")
        return table.filter([], axis='sample', inplace=False), metadata.iloc[0:0]

    # Map the common lowercase IDs back to their original cased versions
    table_ids_to_keep = [table_id_map[lc_id] for lc_id in common_lowercase_ids]
    metadata_ids_to_keep = [metadata_id_map[lc_id] for lc_id in common_lowercase_ids]

    # Filter the data using the original cased IDs
    table.filter(table_ids_to_keep, axis='sample', inplace=True)
    
    if metadata.index.name != found_id_column:
        metadata = metadata.set_index(found_id_column, drop=False)
    
    metadata = metadata.reindex(metadata_ids_to_keep)

    logger.info(f"Synced table and metadata case-insensitively, retaining {len(common_lowercase_ids)} common samples.")
    return table, metadata


def merge_table_with_metadata(
    table: Table, metadata: pd.DataFrame, potential_id_columns: Union[List[str], None] = None
) -> pd.DataFrame:
    """
    Converts a BIOM table to a DataFrame and merges it with metadata.

    This function performs a case-insensitive join on the sample IDs and
    automatically searches for common sample ID column names in the metadata.

    Args:
        table: The BIOM table object containing feature counts.
        metadata: The pandas DataFrame containing sample metadata.
        potential_id_columns: An optional list of possible sample ID column names.

    Returns:
        A pandas DataFrame containing the merged feature data and metadata.
        
    Raises:
        KeyError: If no valid sample ID column is found in the metadata.
    """
    if potential_id_columns is None:
        potential_id_columns = ['#SampleID', 'sample_id', 'SampleID', 'sample-id', 
                                'sampleid', '#sampleid']

    # Find the correct sample ID column in the metadata
    sample_id_col = next(
        (col for col in potential_id_columns if col in metadata.columns), 
        None
    )
    if not sample_id_col:
        raise KeyError(
            "Could not find a valid sample ID column in the metadata.\n"
            f" ⤷ Tried looking for: {potential_id_columns}\n"
            f" ⤷ Available columns are: {metadata.columns.tolist()}"
        )

    # Convert the BIOM table to a DataFrame with samples as rows
    features_df = table.to_dataframe(dense=True).T

    # Prepare for case-insensitive merge by creating temporary lowercase columns
    metadata_copy = metadata.copy()
    features_df_copy = features_df.copy()
    
    metadata_copy['_tmp_join_id'] = metadata_copy[sample_id_col].str.lower()
    features_df_copy['_tmp_join_id'] = features_df_copy.index.str.lower()

    # Perform the merge using the temporary lowercase ID column
    # Only samples present in both are kept
    merged_df = pd.merge(
        metadata_copy, features_df_copy, on='_tmp_join_id', how='inner'
    )

    # Clean up by removing the temporary column
    merged_df.drop(columns=['_tmp_join_id'], inplace=True)
    
    # Set the original sample ID column as the index for clarity
    merged_df.set_index(sample_id_col, inplace=True)

    return merged_df


def merge_dataframes_on_sample_id(
    features_df: pd.DataFrame, metadata_df: pd.DataFrame,
    potential_id_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Merges a features DataFrame with a metadata DataFrame using a case-
    insensitive join on a shared sample ID column.

    The function automatically detects the sample ID column in the metadata and
    uses the index of the features DataFrame for the join.

    Args:
        features_df:          DataFrame where rows are samples and columns are features. The sample IDs are expected to be in the index.
        metadata_df:          DataFrame containing sample metadata, with one column serving as the sample identifier.
        potential_id_columns: An optional list of possible sample ID column names to search for in the metadata.

    Returns:
        A single pandas DataFrame containing the merged feature data and metadata.
        Only samples present in both input DataFrames are retained.

    Raises:
        KeyError: If no valid sample ID column is found in the metadata.
    """
    if potential_id_columns is None:
        potential_id_columns = ['#SampleID', 'sample_id', 'SampleID', 'sample-id',
                                'sampleid', '#sampleid']

    # Find the correct sample ID column in the metadata DataFrame
    metadata_id_col = next((col for col in potential_id_columns if col in metadata_df.columns), None)
    if not metadata_id_col:
        raise KeyError(
            "Could not find a valid sample ID column in the metadata.\n"
            f" ⤷ Tried looking for: {potential_id_columns}\n"
            f" ⤷ Available columns are: {metadata_df.columns.tolist()}"
        )

    # Create temporary copies to avoid modifying original DataFrames
    features_copy = features_df.copy()
    metadata_copy = metadata_df.copy()

    # Prepare for case-insensitive merge by creating temporary lowercase join keys
    # For features, the key comes from its index. For metadata, from the found column.
    features_copy['_tmp_join_key'] = features_copy.index.astype(str).str.lower()
    metadata_copy['_tmp_join_key'] = metadata_copy[metadata_id_col].astype(str).str.lower()

    # Perform the merge using the temporary lowercase keys
    # Only samples present in both are kept
    merged_df = pd.merge(
        metadata_copy, features_copy, on='_tmp_join_key', how='inner'
    )

    # Clean up the final DataFrame
    # Remove the temporary join key
    merged_df.drop(columns=['_tmp_join_key'], inplace=True)
    # Set the original, cased sample ID column as the index for clarity
    merged_df.set_index(metadata_id_col, inplace=True)

    return merged_df


def add_metadata_column(features_df: pd.DataFrame, metadata_df: pd.DataFrame,
                        column_to_add: str, new_column_name: Optional[str] = None,
                        potential_id_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Adds a single column from a metadata DataFrame to a features DataFrame
    using a case-insensitive match on the sample ID.

    Args:
        features_df:          The DataFrame to which the column will be added. Its index should contain the sample IDs.
        metadata_df:          The DataFrame from which to pull the column.
        column_to_add:        The name of the column in metadata_df to add.
        new_column_name:      Optional new name for the added column. If None, the original name is used.
        potential_id_columns: Optional list of possible sample ID column names to search for in the metadata.

    Returns:
        A new DataFrame with the added metadata column. Samples in features_df
        that have no match in the metadata will have a NaN value in the new column.
        
    Raises:
        KeyError: If column_to_add or a valid sample ID column is not found.
    """
    if potential_id_columns is None:
        potential_id_columns = ['#SampleID', 'sample_id', 'SampleID', 'sample-id', 
                                'sampleid', '#sampleid']
    
    # Validate that the column to add exists
    if column_to_add not in metadata_df.columns:
        raise KeyError(f"Column '{column_to_add}' not found in the metadata DataFrame.")

    # Find the correct sample ID column in the metadata
    sample_id_col = next((col for col in potential_id_columns if col in metadata_df.columns), None)
    if not sample_id_col:
        raise KeyError(f"Could not find a valid sample ID column in the metadata.")

    # Create a case-insensitive lookup map (a pandas Series)
    # Index = lowercase sample IDs, Values = the metadata column data
    mapping_series = pd.Series(
        metadata_df[column_to_add].values, index=metadata_df[sample_id_col].str.lower()
    )
    
    # Create a new DataFrame to avoid modifying the original
    result_df = features_df.copy()
    
    # Determine the name for the new column
    col_name = new_column_name if new_column_name is not None else column_to_add
    
    # Use .map() to create the new column
    # It looks up each lowercase sample ID from the features index in the mapping series
    result_df[col_name] = result_df.index.str.lower().map(mapping_series)

    return result_df