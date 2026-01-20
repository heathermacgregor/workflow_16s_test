# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from functools import reduce
from pathlib import Path
from typing import Dict, List, Union

# Third-Party Imports
import h5py
import numpy as np
import pandas as pd
from biom import load_table
from biom.table import Table

# Local Imports
from workflow_16s import constants
from workflow_16s.constants import (
    DEFAULT_GROUP_COLUMN, GROUP_THRESHOLD, PREVALENCE_THRESHOLD, 
    TAXONOMIC_LEVELS_MAPPING
)
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ==================================== FUNCTIONS ===================================== #

def import_biom(biom_path: Union[str, Path]) -> Table:
    """Load a BIOM table from file.
    
    Args:
        biom_path: Path to .biom file.
    
    Returns:
        BIOM Table object or pandas DataFrame.
    """
    try:
        with h5py.File(biom_path) as f:
            return Table.from_hdf5(f)
    except:
        return load_table(biom_path)


def import_merged_biom_table(biom_paths: List[Union[str, Path]]) -> Table:
    """Merge multiple BIOM tables into a single unified table.
    
    Args:
        biom_paths: List of paths to .biom files.
    
    Returns:
        Merged BIOM Table or DataFrame.
    
    Raises:
        ValueError: If no valid tables are loaded.
    """
    tables: List[Table] = []
    with get_progress_bar() as progress:
        task = progress.add_task(
            _format_task_desc("Loading feature tables"), 
            total=len(biom_paths)
        )
        for path in biom_paths:
            try:
                tables.append(import_biom(path))
            except Exception as e:
                logger.error(f"BIOM load failed for {path}: {e}")
            finally:
                progress.update(task, advance=1)

    if not tables:
        raise ValueError("No valid BIOM tables loaded")

    # Merge the tables using reduce
    return reduce(lambda t1, t2: t1.merge(t2), tables)
        

def export_h5py(
    table: Union[pd.DataFrame, Table],
    output_path: Union[str, Path]
) -> None:
    """Export BIOM Table with h5py.

    Args:
        table:       BIOM Table.
        output_path: File path to save output.
    """
    table = table.copy()
    table = df_to_biom(table)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        table.to_hdf5(f, generated_by=f"Table")
        

def df_to_biom(table: Union[pd.DataFrame, Table]) -> Table:
    """Convert pandas DataFrame to BIOM Table.
    
    Args:
        table: Input DataFrame containing feature counts.
    
    Returns:
        BIOM Table representation of the DataFrame.
    """
    if isinstance(table, Table):
        return table
    
    # Ensure we have a DataFrame with features as rows and samples as columns
    if table.shape[0] < table.shape[1]:
        # This might be transposed - transpose back
        table = table.T
    
    # Convert all values to numeric, coercing errors to NaN
    table = table.apply(pd.to_numeric, errors='coerce')
    
    # Fill NaN values with 0 and convert to integers
    table = table.fillna(0).astype(int)
    
    return Table(
        data=table.values,
        observation_ids=table.index.astype(str).tolist(),
        sample_ids=table.columns.astype(str).tolist(),
        type="OTU table"
    )
  

def collapse_taxa(
    table: Union[pd.DataFrame, Table], 
    target_level: str,
    levels: Dict = TAXONOMIC_LEVELS_MAPPING
) -> Table:
    """Collapse feature table to specified taxonomic level.
    
    Args:
        table:        Input BIOM Table or DataFrame.
        target_level: Taxonomic level to collapse to (phylum/class/order/family).
        levels:       Taxonomic levels mapping dictionary.
    
    Returns:
        Collapsed BIOM Table.
    
    Raises:
        ValueError: For invalid target_level.
    """
    table = table.copy()
    table = df_to_biom(table)
        
    if target_level not in levels:
        raise ValueError(
            f"Invalid `target_level`: {target_level}. "
            f"Expected one of {list(levels.keys())}"
        )

    i = levels[target_level]

    # Create taxonomy mapping
    id_map = {}
    for taxon in table.ids(axis='observation').astype(str):
        parts = taxon.split(';')
        truncated = ';'.join(parts[:i + 1]) if len(parts) >= i + 1 else 'Unclassified'
        id_map[taxon] = truncated

    # Collapse table
    return table.collapse(
        lambda id, _: id_map.get(id, 'Unclassified'),
        norm=False,
        axis='observation',
        include_collapsed_metadata=False
    ).remove_empty()
  

def presence_absence(
    table: Union[Table, pd.DataFrame], 
    target_level: str, 
) -> Table:
    """Convert table to presence/absence format and filter by abundance.
    
    Args:
        table:        Input BIOM Table or DataFrame.
        target_level: Taxonomic level.
        output_dir:   Directory to save output.
    
    Returns:
        Presence/absence BIOM Table filtered by abundance.
    """
    table = table.copy()
    table = df_to_biom(table)
    
    # Filter by abundance
    feature_sums = np.array(table.sum(axis='observation')).flatten()
    sorted_idx = np.argsort(feature_sums)[::-1]
    cumulative = np.cumsum(feature_sums[sorted_idx]) / feature_sums.sum()
    stop_idx = np.searchsorted(cumulative, 0.99) + 1
    keep_ids = [table.ids(axis='observation')[i] for i in sorted_idx[:stop_idx]]
    
    # Convert to presence/absence
    pa_table = table.pa(inplace=False)
    pa_table_fltr = pa_table.filter(keep_ids, axis='observation')
    pa_df_fltr = pa_table_fltr.to_dataframe(dense=True)
    
    return Table(
        data=pa_df_fltr.values,
        observation_ids=pa_df_fltr.index.tolist(),
        sample_ids=pa_df_fltr.columns.tolist(),
        table_id='Presence Absence BIOM Table'
    )


def filter_presence_absence(
    table: Table, 
    metadata: pd.DataFrame, 
    group_column: str = DEFAULT_GROUP_COLUMN, 
    prevalence_threshold: float = PREVALENCE_THRESHOLD, 
    group_threshold: float = GROUP_THRESHOLD
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
    df_with_meta = df.join(metadata[[group_column]], how='inner')

    # Apply prevalence filter
    if prevalence_threshold:
        species_data = df_with_meta.drop(columns=[group_column])
        prev = species_data.mean(axis=0)
        filtered_species = prev[prev >= prevalence_threshold].index
        df_with_meta = df_with_meta[filtered_species.union(pd.Index([col]))]

    # Apply group filter
    if group_threshold:
        #group_column_type = df_with_meta[group_column].dtype
        groups = df_with_meta.groupby(group_column)
        if True not in groups.groups or False not in groups.groups:
            error_msg = f"Metadata column '{group_column}' must have True/False groups"
            raise ValueError(error_msg)
        sum_per_group = groups.sum(numeric_only=True)
        n_samples = groups.size()
        percentages = sum_per_group.div(n_samples, axis=0)
        mask = (percentages.loc[True] >= group_threshold) & (percentages.loc[False] >= group_threshold)
        selected_species = mask[mask].index
        df_with_meta = df_with_meta[selected_species.union(pd.Index([group_column]))]

    return Table(
        data=df_with_meta.drop(columns=[group_column]).values.T,
        observation_ids=df_with_meta.columns.tolist(),
        sample_ids=df_with_meta.index.tolist(),
        table_id='Filtered Presence/Absence Table'
    )


def sample_id_map(table: Table) -> Dict[str, str]:
    """Create lowercase to original-case ID mapping for BIOM table samples."""
    table = df_to_biom(table)
    if table.is_empty(): # Handle empty table
        return {}
    
    mapping: Dict[str, str] = {}
    for orig_id in table.ids(axis='sample'):
        lower_id = orig_id.lower()
        if lower_id in mapping:
            raise ValueError(f"Duplicate lowercase sample ID: '{lower_id}' "
                             f"(from '{orig_id}' and '{mapping[lower_id]}')")
        mapping[lower_id] = orig_id
    return mapping
    
