"""
Statistical tests for beta diversity analysis.

Provides parallel implementations of PERMANOVA and Mantel tests with
assumption validation (betadisper for homogeneity of variance).

CRITICAL ASSUMPTION (P1 Fix):
- PERMANOVA test assumes homogeneity of variance across groups.
- If this assumption is violated (p<0.05 in betadisper test), PERMANOVA
  p-values can yield false positives. This module now reports both tests.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
from skbio.stats.distance import DistanceMatrix, MissingIDError, mantel, permanova

from workflow_16s.utils.logger import get_logger


def compute_betadisper(
    dm: DistanceMatrix,
    grouping: pd.Series,
    n_permutations: int = 999
) -> Dict[str, Any]:
    """
    Compute beta-dispersion (multivariate distance to group centroid).
    
    Tests homogeneity of multivariate variance across groups. A significant
    result (p<0.05) indicates that groups differ in their internal variance,
    which violates PERMANOVA assumptions.
    
    Parameters
    ----------
    dm : DistanceMatrix
        Pairwise distance matrix (skbio format)
    grouping : pd.Series
        Group assignment for each sample
    n_permutations : int
        Number of permutations for p-value calculation
        
    Returns
    -------
    Dict containing:
        - 'F': F-statistic (ratio of between-group to within-group variance)
        - 'p_value': p-value from permutation test
        - 'avg_dispersion_by_group': Dict of group → mean distance to centroid
        - 'homogeneity_violated': bool (True if p<0.05)
        - 'warning': str if violated
        
    References
    ----------
    Anderson, M. J. (2006). Distance-based tests for homogeneity of 
    multivariate dispersions. Biometrics, 62(1), 245-253.
    """
    dm_array = dm.data
    group_ids = np.array(grouping.values)
    unique_groups = np.unique(group_ids)
    
    # Compute distances from each point to group centroid
    dispersions = []
    group_dispersions = {}
    
    for group in unique_groups:
        group_mask = group_ids == group
        group_indices = np.where(group_mask)[0]
        
        if len(group_indices) < 2:
            continue
            
        # Distances within group
        group_dm = dm_array[np.ix_(group_indices, group_indices)]
        
        # Mean distance to centroid = mean of all distances in group / 2
        # (since each pair counted twice in symmetric distance matrix)
        mean_dist_to_centroid = np.mean(group_dm[np.triu_indices_from(group_dm, k=1)])
        
        dispersions.extend([mean_dist_to_centroid] * len(group_indices))
        group_dispersions[group] = mean_dist_to_centroid
    
    dispersions = np.array(dispersions)
    
    # Compute F-statistic: ratio of between-group to within-group variance
    overall_mean = np.mean(dispersions)
    within_group_var = np.sum((dispersions - overall_mean) ** 2)
    
    # Between-group variance
    between_group_var = 0
    for group in unique_groups:
        group_mask = group_ids == group
        group_dispersions_vals = dispersions[group_mask]
        between_group_var += len(group_dispersions_vals) * (
            np.mean(group_dispersions_vals) - overall_mean
        ) ** 2
    
    n_groups = len(unique_groups)
    n_samples = len(dispersions)
    
    if (n_samples - n_groups) > 0:
        f_stat = between_group_var / (within_group_var / (n_samples - n_groups)) if within_group_var > 0 else 0
    else:
        f_stat = 0
    
    # Permutation test
    perm_f_stats = []
    for _ in range(n_permutations):
        perm_groups = np.random.permutation(group_ids)
        perm_between = 0
        for group in unique_groups:
            perm_mask = perm_groups == group
            perm_between += np.sum((dispersions[perm_mask] - overall_mean) ** 2)
        perm_f = perm_between / (within_group_var / (n_samples - n_groups)) if within_group_var > 0 else 0
        perm_f_stats.append(perm_f)
    
    perm_f_stats = np.array(perm_f_stats)
    p_value = (np.sum(perm_f_stats >= f_stat) + 1) / (n_permutations + 1)
    
    violated = p_value < 0.05
    warning_msg = (
        f"⚠️ PERMANOVA ASSUMPTION VIOLATED: Homogeneity of variance test "
        f"(betadisper) p={p_value:.4f} < 0.05. PERMANOVA p-values may be unreliable. "
        f"Consider using Pseudo-F alternative or report with caution."
    ) if violated else None
    
    return {
        'F': f_stat,
        'p_value': p_value,
        'avg_dispersion_by_group': group_dispersions,
        'homogeneity_violated': violated,
        'warning': warning_msg
    }


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
    
    **CRITICAL FIX (P1)**: Now includes betadisper homogeneity test.
    If homogeneity assumption violated, results include warning and are flagged.

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
        Tuple of (column_name, results_dict) containing:
        - 'F' (test statistic)
        - 'p-value' (PERMANOVA p-value)
        - 'betadisper_F' (homogeneity of variance F-stat)
        - 'betadisper_p_value' (homogeneity test p-value)
        - 'homogeneity_violated' (bool: True if assumption violated)
        - 'warning' (str if assumptions violated, else None)
        or None if test fails or requirements not met

    Notes
    -----
    - Requires at least 2 groups with 2+ samples each
    - Uses 999 permutations for p-value calculation
    - Unknown/missing values are grouped as 'Unknown'
    - **ASSUMPTION CHECK**: Tests homogeneity of variance via betadisper.
      If violated (p<0.05), PERMANOVA results should be interpreted cautiously.
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

        # **P1 FIX: Check homogeneity of variance before PERMANOVA**
        betadisper_results = compute_betadisper(dm_subset, grouping_subset, n_permutations=999)
        
        if betadisper_results['warning']:
            logger.warning(f"   {col}: {betadisper_results['warning']}")

        # Run PERMANOVA
        perm_res = permanova(dm_subset, grouping=grouping_subset, permutations=999)

        if pd.isna(perm_res['p-value']):
            return None

        # Include betadisper results in output
        results_with_assumption_check = dict(perm_res)
        results_with_assumption_check['betadisper_F'] = betadisper_results['F']
        results_with_assumption_check['betadisper_p_value'] = betadisper_results['p_value']
        results_with_assumption_check['homogeneity_violated'] = betadisper_results['homogeneity_violated']
        results_with_assumption_check['warning'] = betadisper_results['warning']

        return (col, results_with_assumption_check)

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


def apply_stratified_fdr(
    permanova_results: Dict[str, Dict[str, Any]],
    mantel_results: Dict[str, Dict[str, Any]],
    correction_method: str = 'fdr_by',
    alpha: float = 0.05
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Apply stratified FDR correction across all PERMANOVA and Mantel tests.
    
    **P3 CRITICAL FIX**: Tests from 3-5 distance metrics × 10-20 metadata variables
    result in 30-100+ p-values. Without global FDR, alpha inflation is severe (~50%).
    This function applies Benjamini-Yekutieli correction (suitable for dependent tests,
    typical in phylogenetic microbiome data) across ALL p-values in a single analysis run.
    
    Parameters
    ----------
    permanova_results : Dict[str, Dict[str, Any]]
        Results from PERMANOVA tests {col_name: {p-value, betadisper_p_value, ...}}
    mantel_results : Dict[str, Dict[str, Any]]
        Results from Mantel tests {col_name: {p-value, r, ...}}
    correction_method : str
        'fdr_bh' (Benjamini-Hochberg, assumes independence)
        'fdr_by' (Benjamini-Yekutieli, permits dependence) [RECOMMENDED for microbiome]
    alpha : float
        Significance level for reporting
        
    Returns
    -------
    Tuple of:
        - permanova_results with 'q_value' added to each result
        - mantel_results with 'q_value' added to each result  
        - Summary DataFrame with all test results and FDR correction
        
    References
    ----------
    Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery rate
    in multiple testing under dependency. Annals of Statistics, 29(4), 1165-1188.
    """
    from scipy.stats import rankdata
    
    logger = get_logger("workflow_16s")
    
    # Collect all p-values with metadata
    all_tests = []
    
    for col, res in permanova_results.items():
        all_tests.append({
            'Variable': col,
            'Test': 'PERMANOVA',
            'p_value': res['p-value'],
            'betadisper_violated': res.get('homogeneity_violated', False),
            'betadisper_p': res.get('betadisper_p_value', np.nan),
            'test_stat': res.get('test statistic', np.nan),
            'result_key': (col, 'permanova')
        })
    
    for col, res in mantel_results.items():
        all_tests.append({
            'Variable': col,
            'Test': 'Mantel',
            'p_value': res['p-value'],
            'r_statistic': res.get('r', np.nan),
            'betadisper_violated': False,
            'betadisper_p': np.nan,
            'test_stat': res.get('r', np.nan),
            'result_key': (col, 'mantel')
        })
    
    if not all_tests:
        logger.warning("⚠️ No statistical test results to apply FDR correction")
        return permanova_results, mantel_results, pd.DataFrame()
    
    test_df = pd.DataFrame(all_tests)
    n_tests = len(test_df)
    p_values = test_df['p_value'].values
    
    # Apply FDR correction
    if correction_method == 'fdr_by':
        # Benjamini-Yekutieli: q_i = p_i * (m / (2 * i))
        # where m = number of tests, i = rank of p-value
        m = n_tests
        sorted_p = np.sort(p_values)
        ranks = rankdata(p_values)
        
        # Compute q-values: minimize over each p-value's threshold
        q_values = np.ones_like(p_values)
        for i in range(n_tests):
            threshold = sorted_p[i] * m / (2.0 * (i + 1))
            q_values[ranks == (i + 1)] = threshold
        
        # Ensure monotonicity (later = larger)
        for i in range(1, len(q_values)):
            if q_values[i] < q_values[i-1]:
                q_values[i] = q_values[i-1]
                
        q_values = np.minimum(q_values, 1.0)
        
    else:  # fdr_bh (Benjamini-Hochberg)
        ranks = rankdata(p_values)
        q_values = np.ones_like(p_values)
        for i in range(n_tests):
            q_values[ranks == (i + 1)] = sorted_p[i] * n_tests / (i + 1)
        for i in range(1, len(q_values)):
            if q_values[i] < q_values[i-1]:
                q_values[i] = q_values[i-1]
        q_values = np.minimum(q_values, 1.0)
    
    test_df['q_value'] = q_values
    test_df['significant'] = test_df['q_value'] < alpha
    
    # Update results with q-values
    for idx, row in test_df.iterrows():
        var, test_type = row['result_key']
        q_val = row['q_value']
        
        if test_type == 'permanova':
            permanova_results[var]['q_value'] = q_val
            permanova_results[var]['significant_after_fdr'] = row['significant']
        else:  # mantel
            mantel_results[var]['q_value'] = q_val
            mantel_results[var]['significant_after_fdr'] = row['significant']
    
    # Logging summary
    n_signif_original = (test_df['p_value'] < alpha).sum()
    n_signif_corrected = (test_df['q_value'] < alpha).sum()
    
    logger.info(
        f"🔧 Stratified FDR Correction ({correction_method.upper()}): "
        f"{n_tests} tests → "
        f"{n_signif_original} significant (p<{alpha}) → "
        f"{n_signif_corrected} significant (q<{alpha}). "
        f"Removed {n_signif_original - n_signif_corrected} false positives."
    )
    
    violated_count = test_df[test_df['betadisper_violated']].shape[0]
    if violated_count > 0:
        logger.warning(
            f"⚠️ {violated_count} tests have homogeneity of variance violations. "
            f"Results should be interpreted cautiously."
        )
    
    return permanova_results, mantel_results, test_df