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
from biom.table import Table
from biom import load_table

# Local Imports
from workflow_16s.core.progress import get_progress_bar

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ==================================== FUNCTIONS ===================================== #

def import_feature_table(biom_path: Union[str, Path]) -> Table:
    """Load a BIOM table from file."""
    biom_path = Path(biom_path)
    if not biom_path.exists():
        raise FileNotFoundError(f"Feature table file not found: {biom_path}")
    try:
        with h5py.File(biom_path) as f: return Table.from_hdf5(f)
    except: return load_table(biom_path)


def import_merged_feature_table(biom_paths: List[Union[str, Path]]) -> Table:
    """
    Merge multiple BIOM tables into a single unified table.
    
    Args:
        biom_paths: List of paths to .biom files.
    
    Returns:
        Merged BIOM Table or DataFrame.
    
    Raises:
        ValueError: If no valid tables are loaded.
    """
    tables: List[Table] = []
    with get_progress_bar() as progress:
        task = progress.add_task("Loading feature tables", total=len(biom_paths))
        for path in biom_paths:
            try: tables.append(import_feature_table(path))
            except Exception as e: logger.error(f"BIOM load failed for {path}: {e}")
            finally: progress.update(task, advance=1)

    if not tables: raise ValueError("No valid BIOM tables loaded")

    # Merge the tables using reduce
    return reduce(lambda t1, t2: t1.merge(t2), tables)
        

def to_biom(table: Union[dict, pd.DataFrame, Table]) -> Table:
    """
    Convert pandas DataFrame or dict to BIOM Table.
    
    Args:
        table: Input DataFrame or dict containing feature counts.
    
    Returns:
        BIOM Table representation of the DataFrame.
    """
    if isinstance(table, Table): return table

    # Convert dict to DataFrame if necessary
    if isinstance(table, dict): table = pd.DataFrame.from_dict(table)

    # Ensure we have a DataFrame with features as rows and samples as columns
    if isinstance(table, pd.DataFrame):
        if table.shape[0] < table.shape[1]: table = table.T

        # Convert all values to numeric, coercing errors to NaN
        table = table.apply(pd.to_numeric, errors='coerce')

        # Fill NaN values with 0 and convert to integers
        table = table.fillna(0).astype(int)
    else:
        raise ValueError("Input must be a pandas DataFrame, dict, or BIOM Table.")

    return Table(
        data=table.values, observation_ids=table.index.astype(str).tolist(),
        sample_ids=table.columns.astype(str).tolist(), type="OTU table"
    )


def to_df(table: Union[Dict, Table, pd.DataFrame]) -> pd.DataFrame:
    """
    Convert various table formats to samples × features DataFrame.
    
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
    if isinstance(table, pd.DataFrame): return table
    if isinstance(table, Table): return table.to_dataframe(dense=True).T
    if isinstance(table, dict): return pd.DataFrame(table)
    raise TypeError("Input must be BIOM Table, dict, or DataFrame.")


def export_h5py(
    table: Union[pd.DataFrame, Table], output_path: Union[str, Path]
) -> None:
    """
    Export BIOM Table with h5py.

    Args:
        table:       BIOM Table.
        output_path: File path to save output.
    """
    table = table.copy()
    table = to_biom(table)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, 'w') as f: table.to_hdf5(f, generated_by=f"Table")
