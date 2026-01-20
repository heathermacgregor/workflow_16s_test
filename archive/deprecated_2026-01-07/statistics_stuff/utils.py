# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from typing import Optional

# Third-Party Imports
import pandas as pd


# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ================================ TABLE CONVERSION ================================== #


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