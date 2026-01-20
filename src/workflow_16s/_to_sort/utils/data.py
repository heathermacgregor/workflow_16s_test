# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# Third-Party Imports
import numpy as np
import pandas as pd
from biom import Table as Table
from pandarallel import pandarallel
from scipy.sparse import issparse
from skbio.stats.composition import clr as CLR

# Local Imports
from workflow_16s import constants
from workflow_16s.constants import SAMPLE_ID_COLUMN, DEFAULT_GROUP_COLUMN, DEFAULT_META_ID_COLUMN
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc
from workflow_16s.utils.biom import sample_id_map

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ================================= DEFAULT VALUES =================================== #

FEATURE_PATTERNS = {
    "taxonomic": re.compile(
        r'^d__[\w]+(;p__[\w]+)?(;c__[\w]+)?(;o__[\w]+)?'
        r'(;f__[\w]+)?(;g__[\w]+)?(;s__[\w]+)?$'
    ),
    "hashes": re.compile(r'^[a-f0-9]{32}$|^[a-f0-9]{64}$'),
    "raw_sequences": re.compile(r'^[ACGTRYSWKMBDHVN]+$', re.IGNORECASE)
}

# ================================ TABLE CONVERSION ================================== #

def table_to_df(table: Union[Dict, Table, pd.DataFrame]) -> pd.DataFrame:
    """Convert various table formats to samples × features DataFrame.
    
    Handles:
    - Pandas DataFrame (returns unchanged)
    - BIOM Table (transposes to samples × features)
    - Dictionary (converts to DataFrame)
    
    Args:
        table: Input table in various formats.
        
    Returns:
        DataFrame in samples × features orientation.
        
    Raises:
        TypeError: For unsupported input types
    """
    if isinstance(table, pd.DataFrame):  # samples × features
        return table
    if isinstance(table, Table):     # features × samples
        return table.to_dataframe(dense=True).T
    if isinstance(table, dict):          # samples × features
        return pd.DataFrame(table)
    raise TypeError("Input must be BIOM Table, dict, or DataFrame.")


def to_biom(table: Union[dict, Table, pd.DataFrame]) -> Table:
    """Convert various table formats to BIOM Table with features × samples 
    orientation.
    
    Args:
        table: Input table in various formats.
        
    Returns:
        BIOM Table in features × samples orientation.
        
    Raises:
        ValueError: For unsupported input types.
    """
    if isinstance(table, Table):
        return table
    if isinstance(table, dict):
        return Table.from_json(table)
    if isinstance(table, pd.DataFrame):
        # Ensure features x samples orientation
        return Table(
            table.values,
            observation_ids=table.index.tolist(),
            sample_ids=table.columns.tolist(),
            observation_metadata=None,
            sample_metadata=None
        )
    raise ValueError(f"Unsupported table type: {type(table)}")


# ================================ TABLE OPERATIONS ================================== #

def align_table_and_metadata(
    table: Table,
    metadata: pd.DataFrame,
    sample_id_col: str = SAMPLE_ID_COLUMN
) -> Tuple[Table, pd.DataFrame]:
    """Align BIOM table with metadata using sample IDs.
    
    Args:
        table:         BIOM feature table.
        metadata:      Sample metadata DataFrame.
        sample_id_col: Metadata column containing sample IDs.
    
    Returns:
        Tuple of (filtered BIOM table, filtered metadata DataFrame)
    
    Raises:
        ValueError: For duplicate lowercase sample IDs in BIOM table.
    """
    # Handle empty metadata
    if metadata.empty:
        return Table(np.array([]), [], []), pd.DataFrame(columns=[sample_id_col])
    
    biom_mapping = sample_id_map(table)
    shared_ids = [id for id in metadata[sample_id_col] if id in biom_mapping]
    
    # Handle no shared IDs
    if not shared_ids:
        return Table(np.array([]), [], []), pd.DataFrame(columns=[sample_id_col])
    
    filtered_metadata = metadata[metadata[sample_id_col].isin(shared_ids)]
    original_ids = [biom_mapping[id] for id in filtered_metadata[sample_id_col]]
    filtered_table = table.filter(original_ids, axis='sample', inplace=False)
    return filtered_table, filtered_metadata

def add_col(
    table: Union[pd.DataFrame, Table],
    metadata: pd.DataFrame,
    sample_id_col: str = SAMPLE_ID_COLUMN,
    group_cols: List[str] = ['nuclear_contamination_status']
):
    """"""
    if not isinstance(table, Table):
        table = to_biom(table)
    table, metadata = align_table_and_metadata(table, metadata, sample_id_col)
    logger.info(table.shape)
    logger.info(type(table))
    logger.info(metadata.shape)
    logger.info(type(metadata))
    table = table_to_df(table)
    for col in group_cols:
        if col in metadata.columns:
            table[col] = metadata[col]
    logger.info(table.shape)
    logger.info(type(table))
    return table


def merge_table_with_meta(
    table: pd.DataFrame,
    meta: pd.DataFrame,
    group_col: str = DEFAULT_GROUP_COLUMN,
    meta_id_col: Optional[str] = DEFAULT_META_ID_COLUMN,
    verbose: bool = False
) -> pd.DataFrame:
    """Merge feature table with metadata column using direct ID matching.
    
    Features:
    - Automatic orientation detection
    - Duplicate ID detection
    - Case normalization
    - Transposition when needed
    
    Args:
        table:       Feature table (Samples × features) or (features × Samples).
        meta:        Metadata table.
        group_col:   Metadata column to add.
        meta_id_col: Column in metadata containing sample IDs.
        verbose:     Verbosity flag.
        
    Returns:
        Table with added group_column (Samples × features+1).
        
    Raises:
        ValueError: For duplicate IDs or mismatched samples.
    """
    # Identify sample IDs in metadata
    if meta_id_col:
        if verbose:
            print(f"Using metadata column '{meta_id_col}' for sample IDs")
        
        if meta_id_col not in meta.columns:
            raise ValueError(f"Column '{meta_id_col}' not found in metadata")
        
        # Extract and normalize metadata sample IDs
        meta_ids = meta[meta_id_col].astype(str).str.strip().str.lower()
    else:
        if verbose:
            print("Using metadata index for sample IDs")
        meta_ids = meta.index.astype(str).str.strip().str.lower()

    # Check for duplicates in normalized metadata IDs
    duplicate_mask = meta_ids.duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = meta_ids[duplicate_mask].unique()
        n_duplicates = len(duplicates)
        example_duplicates = duplicates[:5]
        
        # Find original values for duplicates
        if meta_id_col:
            original_values = meta.loc[duplicate_mask, meta_id_col].unique()
        else:
            original_values = meta.index[duplicate_mask].unique()
        
        example_originals = original_values[:5]
        
        raise ValueError(
            f"Found {n_duplicates} duplicate sample IDs in metadata\n"
            f"Duplicate normalized IDs: {example_duplicates}\n"
            f"Original values: {example_originals}"
        )
    
    # Assume samples are rows (standard orientation)
    table_ids = table.index.astype(str).str.strip().str.lower()
    # Check intersection
    shared_ids = set(table_ids) & set(meta_ids)
    
    # If no overlap, transpose table (features as rows)
    if not shared_ids:
        if verbose:
            print("No shared IDs found - transposing table")
        table = table.T
        table_ids = table.index.astype(str).str.strip().str.lower()
        shared_ids = set(table_ids) & set(meta_ids)
        
        # If still no matches, raise error
        if not shared_ids:
            table_examples = sorted(table_ids)[:5]
            meta_examples = sorted(meta_ids)[:5]
            raise ValueError(
                "No common sample IDs found\n"
                f"Table IDs: {table_examples}\n"
                f"Metadata IDs: {meta_examples}"
            )
    
    if verbose:
        print(f"Found {len(shared_ids)} shared sample IDs")
    
    # Create normalized ID to group mapping
    if meta_id_col:
        group_map = (
            meta
            .assign(norm_id=meta_ids)
            .set_index("norm_id")[group_col]
        )
    else:
        # Use normalized index directly
        group_map = meta.set_index(meta_ids)[group_col]
    
    # Create normalized table index
    table_normalized_index = table.index.astype(str).str.strip().str.lower()
    # Map group values using normalized IDs
    table[group_col] = table_normalized_index.map(group_map)
    
    # Validate mapping
    if table[group_col].isna().any():
        missing_count = table[group_col].isna().sum()
        missing_samples = table.index[table[group_col].isna()][:5].tolist()
        raise ValueError(
            f"{missing_count} samples missing '{group_col}' values\n"
            f"First 5: {missing_samples}"
        )
    
    return table


def update_table_and_meta(
    table: Table,
    meta: pd.DataFrame,
    sample_col: str = DEFAULT_META_ID_COLUMN
) -> Tuple[Table, pd.DataFrame]:
    """Align BIOM table with metadata using sample IDs.
    
    Args:
        table:         BIOM feature table.
        metadata_df:   Sample metadata DataFrame.
        sample_column: Metadata column containing sample IDs.
    
    Returns:
        Tuple of (filtered BIOM table, filtered metadata DataFrame)
    
    Raises:
        ValueError: For duplicate lowercase sample IDs in BIOM table.
    """
    norm_meta = _normalize_metadata(meta, sample_col)
    biom_mapping = _create_biom_id_mapping(table)
    
    shared_ids = [sid for sid in norm_meta[sample_col] if sid in biom_mapping]
    filtered_meta = norm_meta[norm_meta[sample_col].isin(shared_ids)]
    original_ids = [biom_mapping[sid] for sid in filtered_meta[sample_col]]
    
    return table.filter(original_ids, axis='sample', inplace=False), filtered_meta


def _normalize_metadata(
    meta: pd.DataFrame, 
    sample_col: str
) -> pd.DataFrame:
    """Normalize sample IDs and remove duplicates.
    
    Args:
        metadata_df: Sample metadata DataFrame.
        sample_column: Column containing sample IDs.
    
    Returns:
        Normalized metadata with lowercase IDs and duplicates removed.
    """
    df = meta.copy()
    df[sample_col] = df[sample_col].astype(str).str.lower()
    return df.drop_duplicates(subset=[sample_col])


def _create_biom_id_mapping(table: Table) -> Dict[str, str]:
    """Create lowercase to original-case ID mapping for BIOM table samples.
    
    Args:
        table: BIOM feature table.
    
    Returns:
        Dictionary mapping lowercase IDs to original-case IDs.
    
    Raises:
        ValueError: If duplicate lowercase IDs are detected.
    """
    mapping: Dict[str, str] = {}
    for orig_id in table.ids(axis='sample'):
        lower_id = orig_id.lower()
        if lower_id in mapping:
            raise ValueError(
                f"Duplicate lowercase sample ID: '{lower_id}' "
                f"(from '{orig_id}' and '{mapping[lower_id]}')"
            )
        mapping[lower_id] = orig_id
    return mapping


# ================================ TABLE FILTERING =================================== #

def filter(
    table: Union[dict, Table, pd.DataFrame],
    min_rel_abundance: float = constants.DEFAULT_MIN_REL_ABUNDANCE,
    min_samples: int = constants.DEFAULT_MIN_SAMPLES,
    min_counts: int = constants.DEFAULT_MIN_COUNTS,
) -> Table:
    """Filter features and samples with strict type enforcement.
    
    Applies two-step filtering:
    1. Feature filtering (min_rel_abundance and min_samples)
    2. Sample filtering (min_counts)
    
    Args:
        table: Input table
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
    table: Table, 
    min_rel_abundance: float, 
    min_samples: int
) -> Table:
    """Filter features based on prevalence and abundance.
    
    Args:
        table:             BIOM Table to filter.
        min_rel_abundance: Minimum relative abundance (%).
        min_samples:       Minimum samples where feature must appear.
        
    Returns:
        Filtered BIOM Table.
    """
    min_abs_abundance = min_rel_abundance / 100
    
    # Convert to DataFrame for vectorized operations
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
    table: Table, 
    min_counts: int
) -> Table:
    """Filter samples based on minimum total counts.
    
    Args:
        table:      BIOM Table to filter.
        min_counts: Minimum total counts per sample.
        
    Returns:
        Filtered BIOM Table.
    """
    # Convert to DataFrame for vectorized operations
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

def normalize(
    table: Union[dict, Table, pd.DataFrame], 
    axis: int = 1
) -> Table:
    """Normalize table to relative abundance with strict type enforcement.
    
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
    
    if axis == 1:  # Sample-wise normalization (convert to relative abundance)
        return biom_table.norm(axis='sample')
    else:  # Feature-wise normalization
        return biom_table.norm(axis='observation')
        

def clr(
    table: Union[dict, Table, pd.DataFrame], 
    pseudocount: float = constants.DEFAULT_PSEUDOCOUNT
) -> Table:
    """Apply centered log-ratio (CLR) transformation to table.
    
    Args:
        table:       Input table.
        pseudocount: Small value to add to avoid log(0).
        
    Returns:
        CLR-transformed BIOM Table.
    """
    biom_table = to_biom(table)
    
    # Convert to dense array (samples x features)
    if issparse(biom_table.matrix_data):
        dense_data = biom_table.matrix_data.toarray().T # type: ignore
    else:
        dense_data = np.asarray(biom_table.matrix_data).T
    
    # Apply CLR transformation
    clr_data = CLR(dense_data + pseudocount)
    
    # Transpose back to features x samples
    clr_data = clr_data.T
    
    # Create new BIOM Table with original metadata
    return Table(
        data=clr_data,
        observation_ids=biom_table.ids(axis='observation'),
        sample_ids=biom_table.ids(axis='sample'),
        observation_metadata=biom_table.metadata(axis='observation'),
        sample_metadata=biom_table.metadata(axis='sample')
    )


# TODO: Integrate into workflow
def classify_feature_format(
    cols: Iterable[str], 
    verbose: bool = False
) -> Dict[str, int]:
    """Classify feature IDs into taxonomic, hash, sequence, or unknown types.
    
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
            if not col:
                counts["unknown"] += 1
                continue
                
            matched = False
            for name, pattern in FEATURE_PATTERNS.items():
                if pattern.match(col):
                    counts[name] += 1
                    matched = True
                    break
                    
            if not matched:
                counts["unknown"] += 1
                
        # Print classification summary
        total = sum(counts.values())
        if total > 0:
            dominant = max(counts, key=lambda k: counts[k])
            confidence = counts[dominant] / total
            logger.info(
                f"Feature classification: {dominant} "
                f"({confidence:.0%} confidence)"
            )
    else: 
        with get_progress_bar() as progress:
            task_desc = "Classifying feature IDs..."
            cols_list = list(map(str, cols))
            task = progress.add_task(
                _format_task_desc(task_desc), 
                total=len(cols_list)
            )
            for col in cols_list:
                try:
                    col = col.strip()
                    if not col:
                        counts["unknown"] += 1
                        continue
                        
                    matched = False
                    for name, pattern in FEATURE_PATTERNS.items():
                        if pattern.match(col):
                            counts[name] += 1
                            matched = True
                            break
                            
                    if not matched:
                        counts["unknown"] += 1
                except Exception as e:
                    logger.error(f"Classification failed for feature ID {col}: {e!r}")
                finally:
                    progress.update(task, advance=1)
        
    return counts


# TODO: Integrate into workflow
def trim_and_merge_asvs(
    asv_table: pd.DataFrame,
    asv_seqs: List[str],
    trim_len: int = 250,
    n_workers: int = 8,
    verbose: bool = False,
    progress: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Trim ASV sequences and merge identical trimmed variants with Rich progress tracking.
    
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
        raise ValueError(
            f"ASV count mismatch: "
            f"{len(asv_seqs)} sequences vs {asv_table.shape[0]} features"
        )
    if trim_len < 1:
        raise ValueError(f"Invalid trim length: {trim_len}")

    # Initialize parallel processing WITHOUT pandarallel progress
    pandarallel.initialize(nb_workers=n_workers, progress_bar=False)
    
    # Get progress bar context manager (assume this is defined elsewhere)
    with get_progress_bar(disable=not progress) as progress_bar:
        # Task 1: Sequence trimming
        task_desc = "Trimming sequences"
        trim_task = progress_bar.add_task(
            _format_task_desc(task_desc), 
            total=len(asv_seqs),
            visible=progress
        )
        
        # Process in chunks to update progress
        chunk_size = max(100, len(asv_seqs) // 100)  # 100 chunks max
        trimmed_seqs = []
        
        for start_idx in range(0, len(asv_seqs), chunk_size):
            end_idx = start_idx + chunk_size
            chunk = asv_seqs[start_idx:end_idx]
            
            # Process chunk in parallel
            chunk_trimmed = (
                pd.Series(chunk)
                .apply(lambda x: x[:trim_len])
            )
            trimmed_seqs.extend(chunk_trimmed)
            
            # Update progress bar
            progress_bar.update(
                trim_task,
                advance=len(chunk),
                description=f"[white]{task_desc} ({start_idx+len(chunk)}/{len(asv_seqs)})"
            )
        
        trimmed_seqs = pd.Series(trimmed_seqs)
        
        # Task 2: BIOM table creation
        task_desc = "Creating BIOM table"
        table_task = progress_bar.add_task(
            _format_task_desc(task_desc), 
            total=1,
            visible=progress
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
        
        # Task 3: Collapsing features
        task_desc = "Collapsing sequences"
        collapse_task = progress_bar.add_task(
            _format_task_desc(task_desc), 
            total=1,
            visible=progress
        )
        
        # Collapse features by trimmed sequences
        merged_biom = biom_table.collapse(
            lambda id_, md: md['trimmed_seq'],
            axis='observation',
            norm=False,
            min_group_size=1,
            include_collapsed_metadata=False
        )
        progress_bar.update(collapse_task, advance=1)

    # Convert to DataFrame
    merged_df = merged_biom.to_dataframe(dense=True).astype(np.uint32)
    
    # Log reduction statistics
    orig = len(asv_seqs)
    new = merged_df.shape[0]
    if verbose:
        logger.info(
            f"Feature reduction: {orig} → {new} "
            f"({new/orig:.1%}) after {trim_len}bp trim"
        )
    
    return merged_df, trimmed_seqs, obs_ids


def collapse_taxa(
    table: Union[pd.DataFrame, Table], 
    target_level: str, 
    progress=None, 
    task_id=None,
    verbose: bool = False
) -> Table:
    """Collapse feature table to specified taxonomic level.
    
    Args:
        table:        Input BIOM Table or DataFrame.
        target_level: Taxonomic level to collapse to (phylum/class/order/family).
        output_dir:   Directory to save collapsed table.
        verbose:      Verbosity flag.
    
    Returns:
        Collapsed BIOM Table.
    
    Raises:
        ValueError: For invalid target_level.
    """
    
    table = table.copy()
    table = to_biom(table)
        
    if target_level not in constants.levels:
        raise ValueError(
            f"Invalid `target_level`: {target_level}. "
            f"Expected one of {list(constants.levels.keys())}")

    level_idx = constants.levels[target_level]

    # Create taxonomy mapping
    id_map = {}
    sub_desc = "Feature:"
    sub_task = progress.add_task(
        _format_task_desc(sub_desc),
        parent=task_id,
        total=len(table.ids(axis='observation').astype(str))
    )
    for taxon in table.ids(axis='observation').astype(str):
        try:
            new_desc = f"Feature: {taxon}"
            if len(new_desc) > constants.DEFAULT_N:
                new_desc = f"{new_desc[:constants.DEFAULT_N-3]}..."
            progress.update(sub_task, description=new_desc)
            parts = taxon.split(';')
            truncated = ';'.join(
                parts[:level_idx + 1]
            ) if len(parts) >= level_idx + 1 else 'Unclassified'
            id_map[taxon] = truncated
        except Exception as e:
            logger.error(f"Mapping failed for taxon {taxon}: {e!r}")
        finally:
            progress.update(sub_task, advance=1)

    # Collapse table
    collapsed_table = table.collapse(
        lambda id, _: id_map.get(id, 'Unclassified'),
        norm=False,
        axis='observation',
        include_collapsed_metadata=False
    ).remove_empty()
    progress.remove_task(sub_task)
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
        pa_df_filtered.values,
        pa_df_filtered.index,
        pa_df_filtered.columns,
        table_id='Presence Absence BIOM Table'
    )


def filter_presence_absence(
    table: Table, 
    metadata: pd.DataFrame, 
    col: str = constants.DEFAULT_GROUP_COLUMN, 
    prevalence_threshold: float = constants.DEFAULT_PREVALENCE_THRESHOLD, 
    group_threshold: float = constants.DEFAULT_GROUP_THRESHOLD
) -> Table:
    """Filter presence/absence table based on prevalence and group differences.
    
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
        selected_species = mask[mask].index
        df_with_meta = df_with_meta[selected_species.union(pd.Index([col]))]

    return Table(
        df_with_meta.drop(columns=[col]).values.T,
        df_with_meta.columns.tolist(),
        df_with_meta.index.tolist(),
        table_id='Filtered Presence/Absence Table'
    )
  
###########################################################


from typing import Optional, Tuple, Set

def merge_data(
    table: pd.DataFrame,
    meta: pd.DataFrame,
    group_col: str = constants.DEFAULT_GROUP_COLUMN,
    meta_id_col: Optional[str] = constants.DEFAULT_META_ID_COLUMN,
    verbose: bool = False
) -> pd.DataFrame:
    """Merge feature table with metadata using intelligent ID matching.
    
    Automatically detects table orientation and handles common ID formatting issues
    including case sensitivity and whitespace.
    
    Args:
        table: Feature table, either (samples × features) or (features × samples)
        meta: Metadata DataFrame with sample information
        group_col: Name of metadata column to merge into table
        meta_id_col: Column in metadata containing sample IDs. 
                    If None, uses metadata index
        verbose: Enable detailed logging
        
    Returns:
        Feature table with group column added (samples × features+1)
        
    Raises:
        ValueError: Invalid inputs, duplicate IDs, or no matching samples
        KeyError: Missing required columns
    """
    # Input validation
    logger.info("Converting to DataFrame")
    table = table_to_df(table)
    logger.info("Validating inputs")
    _validate_inputs(table, meta, group_col, meta_id_col)
    logger.info("Aligning table and metadata")
    aligned_biom_table, meta = align_table_and_metadata(to_biom(table), meta)
    table = table_to_df(aligned_biom_table)
    
    # Extract and normalize metadata IDs
    logger.info("Normalizing metadata IDs")
    meta_ids_norm, meta_ids_orig = _extract_metadata_ids(
        meta, '#sampleid', verbose
    )
    logger.info(f"Original: {meta_ids_orig}")
    logger.info(f"Normalized: {meta_ids_norm}")
    logger.info("Checking for duplicate IDs")
    
    # Check for duplicates in metadata
    _check_duplicate_ids(meta_ids_norm, meta_ids_orig, meta_id_col)
    logger.info("Orienting table and finding normalized IDs")
    # Find correct table orientation and get normalized table IDs
    table_oriented, table_ids_norm = _orient_table_for_matching(
        table, meta_ids_norm, verbose
    )
    logger.info(f"Normalized: {table_ids_norm}")
    logger.info("Merging table with metadata")
    # Create mapping and merge
    merged_table = _merge_with_metadata(
        table_oriented, table_ids_norm, meta, meta_ids_norm, 
        group_col, meta_id_col, verbose
    )
    
    return merged_table


def _validate_inputs(
    table: pd.DataFrame, 
    meta: pd.DataFrame, 
    group_col: str, 
    meta_id_col: Optional[str]
) -> None:
    """Validate input parameters."""
    if table.empty:
        raise ValueError("Feature table is empty")
    
    if meta.empty:
        raise ValueError("Metadata table is empty")
    
    if group_col not in meta.columns:
        raise KeyError(f"Group column '{group_col}' not found in metadata")
    
    if meta_id_col and meta_id_col not in meta.columns:
        raise KeyError(f"Metadata ID column '{meta_id_col}' not found")


def _normalize_ids(ids: pd.Series) -> pd.Series:
    """Normalize IDs by converting to string, stripping whitespace, and lowercasing."""
    return ids.astype(str).str.strip().str.lower()


def _extract_metadata_ids(
    meta: pd.DataFrame, 
    meta_id_col: Optional[str], 
    verbose: bool = True
) -> Tuple[pd.Series, pd.Series]:
    """Extract and normalize metadata sample IDs."""
    if meta_id_col:
        logger.info(f"Using metadata column '{meta_id_col}' for sample IDs")
        ids_original = meta[meta_id_col]
    else:
        logger.info("Using metadata index for sample IDs")
        ids_original = meta.index.to_series()
    logger.info(ids_original)
    ids_normalized = _normalize_ids(ids_original)
    return ids_normalized, ids_original


def _check_duplicate_ids(
    meta_ids_norm: pd.Series, 
    meta_ids_orig: pd.Series, 
    meta_id_col: Optional[str]
) -> None:
    """Check for and report duplicate IDs in metadata."""
    duplicates = meta_ids_norm[meta_ids_norm.duplicated(keep=False)]
    
    if not duplicates.empty:
        unique_duplicates = duplicates.unique()
        n_duplicates = len(unique_duplicates)
        
        # Get original values for the duplicates
        duplicate_mask = meta_ids_norm.isin(unique_duplicates)
        original_duplicates = meta_ids_orig[duplicate_mask].unique()
        
        # Show examples (limit to 5 for readability)
        example_norm = unique_duplicates[:5].tolist()
        example_orig = original_duplicates[:5].tolist()
        
        source = f"column '{meta_id_col}'" if meta_id_col else "index"
        
        raise ValueError(
            f"Found {n_duplicates} duplicate sample IDs in metadata {source}\n"
            f"Normalized duplicates: {example_norm}\n"
            f"Original values: {example_orig}"
        )


def _calculate_id_overlap(table_ids: pd.Series, meta_ids: pd.Series) -> Set[str]:
    """Calculate overlap between table and metadata IDs."""
    return set(table_ids) & set(meta_ids)


def _orient_table_for_matching(
    table: pd.DataFrame, 
    meta_ids_norm: pd.Series, 
    verbose: bool
) -> Tuple[pd.DataFrame, pd.Series]:
    """Determine correct table orientation and return oriented table with normalized IDs."""
    # Try samples-as-rows orientation first
    table = table_to_df(table)
    logger.info(table)
    logger.info(meta_ids_norm)
    logger.info(table.index.to_series())
    table_ids_norm = _normalize_ids(table.index.to_series())
    shared_ids = _calculate_id_overlap(table_ids_norm, meta_ids_norm)
    
    if shared_ids:
        if verbose:
            logger.info(f"Using samples-as-rows orientation ({len(shared_ids)} matches)")
        return table, table_ids_norm
    
    # Try transposed orientation (features-as-rows)
    if verbose:
        logger.info("No matches with samples-as-rows, trying transposed orientation")
    
    table_transposed = table.T
    table_ids_norm_t = _normalize_ids(table_transposed.index.to_series())
    shared_ids_t = _calculate_id_overlap(table_ids_norm_t, meta_ids_norm)
    
    if shared_ids_t:
        if verbose:
            logger.info(f"Using features-as-rows orientation ({len(shared_ids_t)} matches)")
        return table_transposed, table_ids_norm_t
    
    # No matches found in either orientation
    _raise_no_matches_error(table_ids_norm, table_ids_norm_t, meta_ids_norm)


def _raise_no_matches_error(
    table_ids_rows: pd.Series, 
    table_ids_cols: pd.Series, 
    meta_ids: pd.Series
) -> None:
    """Raise informative error when no ID matches are found."""
    table_examples_rows = sorted(table_ids_rows.unique())[:5]
    table_examples_cols = sorted(table_ids_cols.unique())[:5]
    meta_examples = sorted(meta_ids.unique())[:5]
    
    raise ValueError(
        "No matching sample IDs found in either table orientation\n"
        f"Table IDs (samples-as-rows): {table_examples_rows}\n"
        f"Table IDs (features-as-rows): {table_examples_cols}\n"
        f"Metadata IDs: {meta_examples}\n"
        "Check for ID format differences or ensure IDs are present in both datasets"
    )


def _merge_with_metadata(
    table: pd.DataFrame,
    table_ids_norm: pd.Series,
    meta: pd.DataFrame,
    meta_ids_norm: pd.Series,
    group_col: str,
    meta_id_col: Optional[str],
    verbose: bool
) -> pd.DataFrame:
    """Perform the actual merge operation."""
    # Create mapping from normalized IDs to group values
    meta_with_norm_ids = meta.copy()
    meta_with_norm_ids['_norm_id'] = meta_ids_norm
    
    # Handle potential duplicate normalized IDs (though we checked earlier)
    group_mapping = meta_with_norm_ids.drop_duplicates('_norm_id').set_index('_norm_id')[group_col]
    
    # Create result table
    result = table.copy()
    result[group_col] = table_ids_norm.map(group_mapping)
    
    # Validate merge results
    _validate_merge_results(result, group_col, verbose)
    
    return result


def _validate_merge_results(
    result: pd.DataFrame, 
    group_col: str, 
    verbose: bool
) -> None:
    """Validate that merge was successful."""
    missing_mask = result[group_col].isna()
    
    if missing_mask.any():
        missing_count = missing_mask.sum()
        missing_samples = result.index[missing_mask][:5].tolist()
        
        raise ValueError(
            f"{missing_count} samples failed to merge with metadata\n"
            f"Missing '{group_col}' for samples: {missing_samples}\n"
            "This indicates samples present in feature table but not in metadata"
        )
    
    if verbose:
        unique_groups = result[group_col].nunique()
        total_samples = len(result)
        logger.info(f"Successfully merged {total_samples} samples with {unique_groups} unique groups")



