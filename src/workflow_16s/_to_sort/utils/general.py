# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Third-Party Imports
import pandas as pd
from biom import Table 

# Local Imports
from workflow_16s.utils.data import table_to_df

# ================================== LOGGER SETUP ==================================== #

logger = logging.getLogger('workflow_16s')

# =================================== DATA UTILS ===================================== #

def print_data_dicts(data):
    for attr_name, attr_value in data.__dict__.items():
        if not attr_name.startswith('__'):
            logger.info(f"{attr_name}: {type(attr_value)}")
            if isinstance(attr_value, dict):
                print_dict_structure(attr_value, attr_name)
                

def print_dict_structure(d, parent_key):
    for k, v in d.items():
        new_key = f"{parent_key}.{k}"
        if isinstance(v, dict):
            print_dict_structure(v, new_key)
        else:
            logger.info(f"{new_key}: {type(v)}")

# TODO: Delete if unused
def get_first_existing_col(
    df: pd.DataFrame, 
    columns: List[str]
) -> Optional[pd.Series]:
    """
    Retrieve the first existing column from a list of candidates.
    
    Args:
        df:      DataFrame to search.
        columns: Ordered list of column names to try.
        
    Returns:
        Series from first existing column, or None if none found.
    """
    for col in columns:
        if col in df.columns:
            return df[col]
    return None
  

# TODO: Delete if unused
def match_indices_or_transpose(
    df1: pd.DataFrame, 
    df2: Union[pd.DataFrame, Table]
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Align DataFrames by index or transpose to find matches.
    
    Checks for index matches between:
    - df1.index and df2.index
    - df1.index and df2.columns (after transpose)
    
    Args:
        df1: Primary DataFrame with index to match.
        df2: Secondary DataFrame or BIOM table.
        
    Returns:
        Tuple: 
            - df1 (unchanged)
            - Aligned df2 (possibly transposed)
            - Boolean indicating if transpose occurred
    """
    # Convert BIOM tables to DataFrame
    if not isinstance(df2, pd.DataFrame):
        df2 = table_to_df(df2)
        
    # Ensure df1 has sample IDs in index
    if '#sampleid' in df1.columns:
        df1 = df1.set_index('#sampleid')
    
    # Check direct index match
    if df1.index.intersection(df2.index).any():
        return df1, df2, False

    # Try transposing df2
    df2_t = df2.T
    if df1.index.intersection(df2_t.index).any():
        return df1, df2_t, True

    # No matches found
    return df1, df2, False


# TODO: Delete if unused
def check_matching_index(
    metadata: pd.DataFrame, 
    features: pd.DataFrame
) -> bool:
    """
    Check for overlapping indices between metadata and features.
    
    Args:
        metadata: DataFrame with sample metadata.
        features: Feature table (samples x features).
        
    Returns:
        True if any matching indices found, False otherwise.
    """
    samples = set(metadata.index)
    return samples.intersection(features.index) or samples.intersection(features.columns)


# TODO: Delete if unused
def match_samples(
    metadata: pd.DataFrame, 
    features: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter and align metadata and features to common samples.
    
    Args:
        metadata: Sample metadata.
        features: Feature table (samples x features).
        
    Returns:
        Tuple of aligned (metadata, features) DataFrames.
    """
    common = metadata.index.intersection(features.columns)
    return metadata.loc[common], features[common]
  
