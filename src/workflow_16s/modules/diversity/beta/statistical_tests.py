"""
Statistical tests for beta diversity analysis.

Provides parallel implementations of PERMANOVA and Mantel tests.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix, MissingIDError, mantel, permanova

from workflow_16s.utils.logger import get_logger


def run_permanova_parallel(
    col: str,
    metadata_df: pd.DataFrame,
    dist_matrix: DistanceMatrix
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Run PERMANOVA test for a single categorical variable in parallel.

    PERMANOVA (Permutational Multivariate Analysis of Variance) tests whether
    groups defined by a categorical variable have significantly different
    community compositions based on a distance matrix.

    Parameters
    ----------
    col : str
        Column name in metadata to test
    metadata_df : pd.DataFrame
        Sample metadata containing the grouping variable
    dist_matrix : DistanceMatrix
        Pairwise distance matrix between samples

    Returns
    -------
    Optional[Tuple[str, Dict[str, Any]]]
        Tuple of (column_name, results_dict) containing test statistic and p-value,
        or None if test fails or requirements not met

    Notes
    -----
    - Requires at least 2 groups with 2+ samples each
    - Uses 999 permutations for p-value calculation
    - Unknown/missing values are grouped as 'Unknown'
    """
    logger = get_logger("workflow_16s")
    try:
        if col not in metadata_df.columns:
            return None

        grouping_series = metadata_df[col].copy()

        # Handle categorical dtype
        if isinstance(grouping_series.dtype, pd.CategoricalDtype):
            if 'Unknown' not in grouping_series.cat.categories:
                try:
                    grouping_series = grouping_series.cat.add_categories('Unknown')
                except Exception:
                    grouping_series = grouping_series.astype(str)

        # Fill missing values and get valid groups
        grouping = grouping_series.astype(str).fillna('Unknown')
        group_counts = grouping.value_counts()
        valid_groups = group_counts[group_counts >= 2].index

        if len(valid_groups) < 2:
            return None

        # Subset to valid samples
        keep_mask = grouping.isin(valid_groups)
        keep_ids = metadata_df.index[keep_mask].tolist()

        if len(keep_ids) < 3:
            return None

        # Filter distance matrix and grouping
        dm_subset = dist_matrix.filter(ids=keep_ids)
        grouping_subset = grouping[keep_mask]

        if dm_subset.shape[0] < 2 or grouping_subset.nunique() < 2:
            return None

        # Run PERMANOVA
        perm_res = permanova(dm_subset, grouping=grouping_subset, permutations=999)

        if pd.isna(perm_res['p-value']):
            return None

        return (col, perm_res)

    except Exception as e:
        logger.error(f"PERMANOVA '{col}' failed: {e}")
        return None


def run_mantel_parallel(
    col: str,
    metadata_df: pd.DataFrame,
    dist_matrix: DistanceMatrix
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Run Mantel test for a single numeric variable in parallel.

    The Mantel test evaluates the correlation between two distance matrices.
    Here, it tests whether a numeric metadata variable is associated with
    community dissimilarity patterns.

    Parameters
    ----------
    col : str
        Column name in metadata to test (must be numeric)
    metadata_df : pd.DataFrame
        Sample metadata containing the numeric variable
    dist_matrix : DistanceMatrix
        Pairwise beta diversity distance matrix

    Returns
    -------
    Optional[Tuple[str, Dict[str, Any]]]
        Tuple of (column_name, results_dict) containing Mantel r statistic
        and p-value, or None if test fails

    Notes
    -----
    - Requires at least 3 samples with non-missing values
    - Uses 999 permutations for p-value calculation
    - Euclidean distance is computed for the numeric variable
    - Returns None if all pairwise distances are zero (no variation)
    """
    logger = get_logger("workflow_16s")
    try:
        if col not in metadata_df.columns:
            return None

        numeric_vector = metadata_df[col].copy()
        valid_mask = numeric_vector.notna()
        valid_ids = metadata_df.index[valid_mask].tolist()

        if len(valid_ids) < 3:
            return None

        # Filter distance matrix
        try:
            dist_subset = dist_matrix.filter(ids=valid_ids)
        except MissingIDError:
            return None

        if dist_subset.shape[0] < 2:
            return None

        # Create distance matrix from numeric variable
        numeric_vec_sub = np.asarray(numeric_vector[valid_mask]).reshape(-1, 1)

        try:
            num_dist_cond = pdist(numeric_vec_sub, 'euclidean')
        except ValueError:
            return None

        # Check for zero variance
        if np.all(num_dist_cond == 0):
            return None

        numeric_dm = DistanceMatrix(squareform(num_dist_cond), ids=valid_ids)

        # Run Mantel test
        r, p, _ = mantel(dist_subset, numeric_dm, permutations=999)

        if pd.isna(p):
            return None

        return (col, {'r': r, 'p-value': p})

    except Exception as e:
        logger.error(f"Mantel '{col}' failed: {e}")
        return None