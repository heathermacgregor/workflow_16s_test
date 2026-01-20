# ===================================== IMPORTS ====================================== #

# Standard Library Imports
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
from biom import Table
import logging
import numpy as np
import pandas as pd
from rich.progress import (
    Progress, 
    BarColumn, 
    TextColumn, 
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    TaskID
)
from scipy import stats
from scipy.spatial.distance import braycurtis, pdist, squareform
from scipy.stats import kruskal, mannwhitneyu, spearmanr, ttest_ind
from skbio.stats.composition import clr
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import pcoa as PCoA
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.data import merge_table_with_meta, table_to_df
from workflow_16s.stats.utils import (
    create_progress
)

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ================================= DEFAULT VALUES =================================== #

DEFAULT_MIN_REL_ABUNDANCE = 1
DEFAULT_MIN_SAMPLES = 10
DEFAULT_MIN_COUNTS = 1000
DEFAULT_PA_THRESHOLD = 0.99
DEFAULT_N_CLUSTERS = 10
DEFAULT_RANDOM_STATE = 0
DEFAULT_GROUP_COLUMN = 'nuclear_contamination_status'
DEFAULT_GROUP_COLUMN_VALUES = [True, False]
DEFAULT_PSEUDOCOUNT = 1e-5

DEFAULT_PROGRESS_TEXT_N = 50

# ================================ CORE FUNCTIONALITY ================================ #

DEFAULT_GROUP_COLUMN = 'nuclear_contamination_status'
DEFAULT_GROUP_COLUMN_VALUES = [True, False]

def t_test(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_col: str = DEFAULT_GROUP_COLUMN,
    group_col_values: List[Union[bool, int, str]] = DEFAULT_GROUP_COLUMN_VALUES,
    progress: Optional[Progress] = None,
    parent_task_id: Optional[TaskID] = None,
    level: Optional[str] = None,
) -> pd.DataFrame:
    """
    Performs independent t-tests between groups for all features.
    
    Args:
        table:      Input abundance table (samples x features).
        metadata:   Sample metadata DataFrame (must contain the same samples as the table).
        group_col:        Metadata column containing group labels.
        group_col_values: Two group identifiers to compare.
        
    Returns:
        results:    Results sorted by p-value with test statistics, excluding features with p=0 or NaN.
    """
    table = table_to_df(table)
    table_with_col = merge_table_with_meta(table, metadata, group_col)
    features = list(table_with_col.columns.drop(group_col))

    task_desc = f"[white]T-Test[/] ({level or 'all features'})".ljust(DEFAULT_PROGRESS_TEXT_N)
    task_id = progress.add_task(
        description=task_desc,
        total=len(features),
        parent=parent_task_id
    )
  
    results = []
    for feature in features:
        mask_group1 = (table_with_col[group_col] == group_col_values[0])
        mask_group2 = (table_with_col[group_col] == group_col_values[1])
        
        group1_values = table_with_col.loc[mask_group1, feature].dropna()
        group2_values = table_with_col.loc[mask_group2, feature].dropna()
        if len(group1_values) < 1 or len(group2_values) < 1:
            continue
            
        t_stat, p_val = ttest_ind(group1_values, group2_values, equal_var=False)
        results.append({'feature': feature, 't_statistic': t_stat, 'p_value': p_val})
        progress.update(task_id, advance=1)

    progress.stop_task(task_id)
    progress.update(task_id, visible=False)
    # Results processing
  
    results_df = pd.DataFrame(results)
    if results_df.empty:
        logger.error(f"No features passed t-test for {col_values} in '{col}'")
        return pd.DataFrame(columns=['feature', 't_statistic', 'p_value'])

    results_df = results_df[results_df['p_value'].notna()]
    results_df = results_df[results_df['p_value'] > 0]
    results_df = results_df.sort_values('p_value')
    logger.info(results_df.head())
    return results_df
    
