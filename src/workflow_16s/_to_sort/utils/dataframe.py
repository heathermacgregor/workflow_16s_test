# ===================================== IMPORTS ====================================== #

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Bio import SeqIO
from biom import load_table, Table as BiomTable
from pandarallel import pandarallel
from scipy import sparse
from scipy.spatial.distance import cdist
from tabulate import tabulate

import logging
logger = logging.getLogger('workflow_16s')


def table_to_df(table: Union[Dict, BiomTable]) -> pd.DataFrame:
    """Convert a BIOM table to pandas DataFrame with samples as rows.
    
    Args:
        table: Input table as either a dictionary or BIOM Table object.
    
    Returns:
        DataFrame with samples as rows and features as columns.
    """
    if isinstance(table, BiomTable):
        df = table.to_dataframe(dense=True)  # features x samples
        return df.T                          # samples  x features
    if isinstance(table, Dict):
        return pd.DataFrame(table)           # samples  x features
    raise TypeError("Input must be BIOM Table or dictionary!")
