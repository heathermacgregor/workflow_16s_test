# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import numpy as np
import pandas as pd
from biom import Table as BiomTable
from scipy import stats
from skbio.stats.composition import clr

# Local Imports
from workflow_16s import constants
from workflow_16s.utils.data import merge_table_with_meta, table_to_df

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ================================ TABLE CONVERSION ================================== #

def validate_inputs(
    table: Union[Dict, Any, pd.DataFrame],
    metadata: Optional[pd.DataFrame] = None,
    group_column: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Validate and standardize inputs for analysis functions."""
    df = table_to_df(table)
    
    if df.empty:
        raise ValueError("Input table is empty")
    
    if df.isnull().all().all():
        raise ValueError("Input table contains only null values")
    
    if metadata is not None:
        if not isinstance(metadata, pd.DataFrame):
            raise ValueError("Metadata must be a pandas DataFrame")
        
        if group_column and group_column not in metadata.columns:
            raise ValueError(f"Group column '{group_column}' not found in metadata")
        
        common_samples = df.index.intersection(metadata.index) + df.columns.intersection(metadata.index)
        if len(common_samples) == 0:
            raise ValueError("No common samples between table and metadata")
        
        if len(common_samples) < len(df.index) * 0.5:
            warnings.warn(
                f"Only {len(common_samples)}/{len(df.index)} samples have metadata. "
                "Consider checking sample ID matching."
            )
    
    return df, metadata

def table_to_dataframe(
    table: Union[Dict, BiomTable, pd.DataFrame]
) -> pd.DataFrame:
    """Convert various table formats to samples × features DataFrame.
    
    Handles:
    - Pandas DataFrame (returns unchanged)
    - BIOM Table (transposes to samples × features)
    - Dictionary (converts to DataFrame)
    
    Args:
        table: Input table in various formats
        
    Returns:
        DataFrame in samples × features orientation
        
    Raises:
        TypeError: For unsupported input types
    """
    if isinstance(table, pd.DataFrame):  # samples × features
        return table
    if isinstance(table, BiomTable):     # features × samples
        return table.to_dataframe(dense=True).T
    if isinstance(table, dict):          # samples × features
        return pd.DataFrame(table)
    raise TypeError("Input must be BIOM Table, dict, or DataFrame.")


def to_biom_table(
    table: Union[dict, BiomTable, pd.DataFrame]
) -> BiomTable:
    """Convert various table formats to BIOM Table with features × samples orientation.
    
    Args:
        table: Input table in various formats
        
    Returns:
        BIOM Table in features × samples orientation
        
    Raises:
        ValueError: For unsupported input types
    """
    if isinstance(table, BiomTable):
        return table
    if isinstance(table, dict):
        return BiomTable.from_json(table)
    if isinstance(table, pd.DataFrame):
        # Ensure features x samples orientation
        return BiomTable(
            table.values,
            observation_ids=table.index.tolist(),
            sample_ids=table.columns.tolist(),
            observation_metadata=None,
            sample_metadata=None
        )
    raise ValueError(f"Unsupported table type: {type(table)}")


# ================================ TABLE OPERATIONS ================================== #

def merge_table_with_metadata(
    table: pd.DataFrame,
    metadata: pd.DataFrame,
    group_column: str,
    metadata_id_column: Optional[str] = '#sampleid',
    verbose: bool = False
) -> pd.DataFrame:
    """Merge feature table with metadata column using direct ID matching.
    
    Features:
    - Automatic orientation detection
    - Duplicate ID detection
    - Case normalization
    - Transposition when needed
    
    Args:
        table:              Feature table (samples × features) or (features × samples).
        metadata:           Metadata table.
        group_column:       Metadata column to add.
        metadata_id_column: Column in metadata containing sample IDs.
        verbose:            Enable debug output.
        
    Returns:
        Table with added group_column (samples × features+1).
        
    Raises:
        ValueError: For duplicate IDs or mismatched samples.
    """
    # Identify sample IDs in metadata
    if metadata_id_column:
        if verbose:
            print(f"Using metadata column '{metadata_id_column}' for sample IDs")
        
        if metadata_id_column not in metadata.columns:
            raise ValueError(f"Column '{metadata_id_column}' not found in metadata")
        
        # Extract and normalize metadata sample IDs
        meta_ids = metadata[metadata_id_column].astype(str).str.strip().str.lower()
    else:
        if verbose:
            print("Using metadata index for sample IDs")
        meta_ids = metadata.index.astype(str).str.strip().str.lower()

    # Check for duplicates in normalized metadata IDs
    duplicate_mask = meta_ids.duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = meta_ids[duplicate_mask].unique()
        n_duplicates = len(duplicates)
        example_duplicates = duplicates[:5]
        
        # Find original values for duplicates
        if metadata_id_column:
            original_values = metadata.loc[duplicate_mask, metadata_id_column].unique()
        else:
            original_values = metadata.index[duplicate_mask].unique()
        
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
    if metadata_id_column:
        group_map = (
            metadata
            .assign(norm_id=meta_ids)
            .set_index("norm_id")[group_column]
        )
    else:
        # Use normalized index directly
        group_map = metadata.set_index(meta_ids)[group_column]
    
    # Create normalized table index
    table_normalized_index = table.index.astype(str).str.strip().str.lower()
    # Map group values using normalized IDs
    table[group_column] = table_normalized_index.map(group_map)
    
    # Validate mapping
    if table[group_column].isna().any():
        missing_count = table[group_column].isna().sum()
        missing_samples = table.index[table[group_column].isna()][:5].tolist()
        raise ValueError(
            f"{missing_count} samples missing '{group_column}' values\n"
            f"First 5: {missing_samples}"
        )
    
    return table


# ================================ TABLE FILTERING =================================== #

def filter_table(
    table: Union[dict, BiomTable, pd.DataFrame],
    min_rel_abundance: float = constants.DEFAULT_MIN_REL_ABUNDANCE,
    min_samples: int = constants.DEFAULT_MIN_SAMPLES,
    min_counts: int = constants.DEFAULT_MIN_COUNTS,
) -> BiomTable:
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
    biom_table = to_biom_table(table)
    biom_table = filter_features(biom_table, min_rel_abundance, min_samples)
    biom_table = filter_samples(biom_table, min_counts)
    return biom_table
    

def filter_features(
    table: BiomTable, 
    min_rel_abundance: float = constants.DEFAULT_MIN_REL_ABUNDANCE, 
    min_samples: int = constants.DEFAULT_MIN_SAMPLES
) -> BiomTable:
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
    table: BiomTable, 
    min_counts: int = constants.DEFAULT_MIN_COUNTS
) -> BiomTable:
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

def normalize_table(
    table: Union[dict, BiomTable, pd.DataFrame], 
    axis: int = 1
) -> BiomTable:
    """Normalize table to relative abundance with strict type enforcement.
    
    Args:
        table: Input table.
        axis:  Normalization axis (0=features, 1=samples).
        
    Returns:
        Normalized BIOM Table.
        
    Raises:
        ValueError: For invalid axis values.
    """
    biom_table = to_biom_table(table)
    
    if axis == 1:  # Sample-wise normalization (convert to relative abundance)
        return biom_table.norm(axis='sample')
    elif axis == 0:  # Feature-wise normalization
        return biom_table.norm(axis='observation')
    else:
        raise ValueError("axis must be 0 (features) or 1 (samples)")
        

def clr_transform_table(
    table: Union[dict, BiomTable, pd.DataFrame], 
    pseudocount: float = constants.DEFAULT_PSEUDOCOUNT
) -> BiomTable:
    """Apply centered log-ratio (CLR) transformation to table.
    
    Args:
        table:       Input table.
        pseudocount: Small value to add to avoid log(0).
        
    Returns:
        CLR-transformed BIOM Table.
    """
    biom_table = to_biom_table(table)
    
    # Convert to dense array (samples x features)
    dense_data = biom_table.matrix_data.toarray().T
    
    # Apply CLR transformation
    clr_data = clr(dense_data + pseudocount)
    
    # Transpose back to features x samples
    clr_data = clr_data.T
    
    # Create new BIOM Table with original metadata
    return BiomTable(
        data=clr_data,
        observation_ids=biom_table.ids(axis='observation'),
        sample_ids=biom_table.ids(axis='sample'),
        observation_metadata=biom_table.metadata(axis='observation'),
        sample_metadata=biom_table.metadata(axis='sample')
    )
