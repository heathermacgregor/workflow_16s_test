# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import numpy as np
import pandas as pd
from biom import Table
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import (
    fisher_exact, f_oneway, kruskal, mannwhitneyu, spearmanr, ttest_ind,
    shapiro, levene
)
from skbio.diversity import alpha
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import pcoa as PCoA
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# Local Imports
from workflow_16s.constants import DEFAULT_GROUP_COLUMN, DEFAULT_N_CLUSTERS, DEFAULT_RANDOM_STATE, DEFAULT_GROUP_COLUMN_VALUES
from workflow_16s.utils.biom_utils import to_df
from workflow_16s.utils.data import merge_dataframes_on_sample_id as merge_table_with_metadata
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ==================================== FUNCTIONS ===================================== #

# CLUSTERING
def k_means(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = DEFAULT_GROUP_COLUMN,
    sample_id_column: str = "#sampleid",
    n_clusters: int = DEFAULT_N_CLUSTERS, 
    random_state: int = DEFAULT_RANDOM_STATE,
    verbose: bool = False
) -> pd.Series:
    """Apply K-means clustering and return cluster labels.

    Args:
        table: Input table data
        metadata: Sample metadata
        group_column: Column name for grouping in metadata
        sample_id_column: Column name for sample IDs
        n_clusters: Number of clusters
        random_state: Random state for reproducibility
        verbose: Whether to print verbose output

    Returns:
        Cluster labels as a pandas Series
    """
    # Merge table and metadata
    merged = merge_table_with_metadata(
        features_df=to_df(table), 
        metadata_df=metadata
    )
    
    if merged.empty:
        return pd.Series()
    
    kmeans = KMeans(
        n_clusters, 
        random_state=random_state
    ).fit(merged.drop(group_column, axis=1))

    results = pd.Series(
        kmeans.labels_, 
        index=merged.index, 
        name='kmeans_cluster'
    )
    return results
    

# TWO-GROUP STATISTICAL TESTS
def ttest(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = DEFAULT_GROUP_COLUMN_VALUES,
    sample_id_column: str = "#sampleid",
    equal_var: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """Performs independent t-tests for two groups using standardized merging."""
    if len(group_column_values) != 2:
        logger.warning(f"T-test requires exactly 2 groups, but {len(group_column_values)} were provided. Skipping.")
        return pd.DataFrame()
    # Merge table and metadata
    merged = merge_table_with_metadata(
        features_df=to_df(table), 
        metadata_df=metadata
    )
    
    if merged.empty:
        return pd.DataFrame()

    # Ensure group column is of string type to avoid mixed-type comparison errors
    merged[group_column] = merged[group_column].astype(str)
    
    results = []
    for feature in merged.columns.drop(group_column):
        # Subset groups
        mask_group1 = (merged[group_column] == group_column_values[0])
        mask_group2 = (merged[group_column] == group_column_values[1])
        
        group1_values = merged.loc[mask_group1, feature].dropna()
        group2_values = merged.loc[mask_group2, feature].dropna()
        
        # Skip features with < 2 samples in either group
        if len(group1_values) < 2 or len(group2_values) < 2:
            continue
            
        try:
            t_stat, p_val = ttest_ind(group1_values, group2_values, equal_var=equal_var)
        except ValueError:
            continue  # Handle cases with invalid variance calculations
            
        # Calculate effect size (Cohen's d)
        n1, n2 = len(group1_values), len(group2_values)
        mean_diff = group1_values.mean() - group2_values.mean()
        std1 = group1_values.std(ddof=1)
        std2 = group2_values.std(ddof=1)
        
        # Pooled standard deviation for Cohen's d
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
        cohen_d = mean_diff / pooled_std if pooled_std != 0 else 0.0
            
        results.append({
            'feature': feature,
            't_statistic': t_stat,
            'p_value': max(p_val, 1e-10),  # Prevent zero p-values
            'mean_difference': mean_diff,
            'cohens_d': cohen_d
        })
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        if verbose:
            logger.error(
                f"No features passed for groups: {group_column_values} "
                f"in column '{group_column}'"
            )
        return pd.DataFrame(columns=['feature', 't_statistic', 'p_value'])

    # Filter invalid p-values and sort
    results_df = results_df[(results_df['p_value'] != 0) & (
        results_df['p_value'].notna()
    )]
    results_df = results_df[results_df['p_value'] <= 0.05]
    results_df = results_df.sort_values('p_value', ascending=True)
    return results_df


def mwu_bonferroni(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = DEFAULT_GROUP_COLUMN_VALUES,
    sample_id_column: str = "#sampleid",
    verbose: bool = False
) -> pd.DataFrame:
    """Performs Mann-Whitney U tests with Bonferroni correction for two groups."""
    if len(group_column_values) != 2:
        logger.warning(f"Mann-Whitney U test requires exactly 2 groups, but {len(group_column_values)} were provided. Skipping.")
        return pd.DataFrame()
    # Merge table and metadata
    merged = merge_table_with_metadata(
        features_df=to_df(table), 
        metadata_df=metadata
    )
    
    if merged.empty:
        return pd.DataFrame()

    # Ensure group column is of string type to avoid mixed-type comparison errors
    merged[group_column] = merged[group_column].astype(str)
    
    # Total features tested (for Bonferroni)
    total_features = len(merged.columns.drop(group_column))
    threshold = 0.01 / total_features
    
    results = []
    for feature in merged.columns.drop(group_column):
        # Subset groups safely
        mask_group1 = (merged[group_column] == group_column_values[0])
        mask_group2 = (merged[group_column] == group_column_values[1])
        
        group1_values = merged.loc[mask_group1, feature].dropna()
        group2_values = merged.loc[mask_group2, feature].dropna()
        
        # Skip features with empty groups
        if len(group1_values) < 1 or len(group2_values) < 1:
            continue
        
        # Perform MWU test
        u_stat, p_val = mannwhitneyu(
            group1_values, 
            group2_values, 
            alternative='two-sided'
        )
        
        # Effect size and median difference
        n1, n2 = len(group1_values), len(group2_values)
        r = 1 - (2 * u_stat) / (n1 * n2)
        median_diff = group1_values.median() - group2_values.median()
            
        results.append({
            'feature': feature,
            'u_statistic': u_stat,
            'p_value': max(p_val, 1e-10),  # Cap p-values
            'median_difference': median_diff,
            'effect_size_r': r
        })
        
    results_df = pd.DataFrame(results)
    if results_df.empty:
        if verbose:
            logger.error(
                f"No features passed Mann-Whitney U tests with Bonferroni correction "
                f"for groups: {group_column_values} "
                f"in column '{group_column}'"
            )
        return pd.DataFrame(columns=['feature', 'u_statistic', 'p_value'])

    # Filter invalid p-values and sort
    results_df = results_df[(results_df['p_value'] != 0) & (
        results_df['p_value'].notna()
    )]
    results_df = results_df[results_df['p_value'] <= 0.05]
    results_df = results_df.sort_values('p_value', ascending=True)
    
    # Apply Bonferroni threshold
    results_df = results_df[results_df['p_value'] <= threshold]
    return results_df


def fisher_exact_bonferroni(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str,
    group_column_values: List[Union[bool, int, str]],
    sample_id_column: str = "#sampleid",
    alpha: float = 0.01,
    min_samples: int = 5,
    verbose: bool = False
) -> pd.DataFrame:
    """Performs Fisher's Exact Tests with Bonferroni correction for presence-absence data."""
    if len(group_column_values) != 2:
        logger.warning(f"Fisher's Exact test requires exactly 2 groups, but {len(group_column_values)} were provided. Skipping.")
        return pd.DataFrame()
    # Convert to DataFrame and merge with metadata
    merged = merge_table_with_metadata(
        features_df=to_df(table), 
        metadata_df=metadata
    )
    
    if merged.empty:
        return pd.DataFrame()

    # Ensure group column is of string type to avoid mixed-type comparison errors
    merged[group_column] = merged[group_column].astype(str)
    
    # Total features for Bonferroni correction
    total_features = len(merged.columns) - 1  # Exclude group column
    threshold = alpha / total_features
    
    results = []
    for feature in merged.columns.drop(group_column):
        # Subset groups
        mask_group1 = (merged[group_column] == group_column_values[0])
        mask_group2 = (merged[group_column] == group_column_values[1])
        
        group1 = merged.loc[mask_group1, feature].dropna()
        group2 = merged.loc[mask_group2, feature].dropna()
        
        # Skip small groups
        if len(group1) < min_samples or len(group2) < min_samples:
            continue
            
        # Build 2x2 contingency table
        a = (group1 == 1).sum()  # Group1 present
        b = (group2 == 1).sum()  # Group2 present
        c = (group1 == 0).sum()  # Group1 absent
        d = (group2 == 0).sum()  # Group2 absent
        
        # Skip invariant features
        if (a + b == 0) or (c + d == 0):
            continue
            
        # Perform Fisher's Exact Test
        try:
            odds_ratio, p_val = fisher_exact(
                [[a, b], [c, d]], alternative='two-sided'
            )
        except ValueError:
            continue  # Skip invalid tables
            
        # Calculate proportions
        prop1 = a / (a + c) if (a + c) > 0 else 0
        prop2 = b / (b + d) if (b + d) > 0 else 0
        prop_diff = prop1 - prop2

        results.append({
            'feature': feature,
            'p_value': max(p_val, 1e-10),
            'odds_ratio': odds_ratio,
            'proportion_diff': prop_diff,
            f'prop_{group_column_values[0]}': prop1,
            f'prop_{group_column_values[1]}': prop2,
            f'present_{group_column_values[0]}': a,
            f'absent_{group_column_values[0]}': c,
            f'present_{group_column_values[1]}': b,
            f'absent_{group_column_values[1]}': d
        })
        
            
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    if results_df.empty:
        if verbose:
            logger.warning(
                "No significant features found after Fisher's Exact "
                f"Tests with Bonferroni correction"
            )
        return pd.DataFrame()
    
    # Apply Bonferroni correction
    results_df = results_df.sort_values('p_value', ascending=True)
    results_df['p_adj'] = results_df['p_value'] * total_features
    results_df['p_adj'] = results_df['p_adj'].clip(upper=1.0)  # Cap at 1.0
    
    # Filter significant results
    results_df = results_df[results_df['p_value'] <= threshold]
    return results_df


# MULTI-GROUP STATISTICAL TESTS
def kruskal_bonferroni(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = None,
    sample_id_column: str = "#sampleid",
    verbose: bool = False
) -> pd.DataFrame:
    """Performs Kruskal-Wallis H-test with Bonferroni correction for ≥3 groups."""
    # Merge table and metadata
    merged = merge_table_with_metadata(
        features_df=to_df(table), 
        metadata_df=metadata
    )
    
    if merged.empty:
        return pd.DataFrame()

    # Ensure group column is of string type to avoid mixed-type comparison errors
    merged[group_column] = merged[group_column].astype(str)
    
    # Get unique groups if group_column_values not specified
    if group_column_values is None:
        group_column_values = merged[group_column].unique().tolist()
    
    # Pre-calculate Bonferroni threshold
    total_features = len(merged.columns.drop(group_column))
    threshold = 0.01 / total_features
    
    results = []
    for feature in merged.columns.drop(group_column):
        # Collect data for all groups
        groups = []
        for group_value in group_column_values:
            mask = (merged[group_column] == group_value)
            group_data = merged.loc[mask, feature].dropna()
            if len(group_data) > 0:  # Skip empty groups
                groups.append(group_data)
        
        # Skip feature if < 2 groups have data
        if len(groups) < 2:
            continue
        
        try:
            h_stat, p_val = kruskal(*groups)
        except ValueError:
            continue  # Handle identical values in all groups
            
        # Calculate effect size (epsilon squared)
        n_total = sum(len(g) for g in groups)
        epsilon_sq = h_stat / (n_total - 1)
            
        results.append({
            'feature': feature,
            'h_statistic': h_stat,
            'p_value': max(p_val, 1e-10),
            'epsilon_squared': epsilon_sq,
            'groups_tested': len(groups)
        })
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        if verbose:
            logger.error(
                f"No features passed Kruskal-Wallis H-test with Bonferroni correction "
                f"for groups: {group_column_values} "
                f"in column '{group_column}'"
            )
        return pd.DataFrame(columns=['feature', 't_statistic', 'p_value'])

    # Filter invalid p-values and sort
    results_df = results_df[(results_df['p_value'] != 0) & (
        results_df['p_value'].notna()
    )]
    results_df = results_df[results_df['p_value'] <= 0.05]
    results_df = results_df.sort_values('p_value', ascending=True)
    
    # Apply Bonferroni correction
    results_df = results_df[results_df['p_value'] <= threshold]
    return results_df


def anova(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = None,
    sample_id_column: str = "#sampleid",
    verbose: bool = False
) -> pd.DataFrame:
    """Performs one-way ANOVA for ≥3 groups."""
    # Merge table and metadata
    merged = merge_table_with_metadata(
        features_df=to_df(table), 
        metadata_df=metadata
    )
    
    if merged.empty:
        return pd.DataFrame()

    # Ensure group column is of string type to avoid mixed-type comparison errors
    merged[group_column] = merged[group_column].astype(str)
    
    # Get unique groups if group_column_values not specified
    if group_column_values is None:
        group_column_values = merged[group_column].unique().tolist()
    
    results = []
    for feature in merged.columns.drop(group_column):
        # Collect data for all groups
        groups = []
        for group_value in group_column_values:
            mask = (merged[group_column] == group_value)
            group_data = merged.loc[mask, feature].dropna()
            if len(group_data) >= 2:  # Require ≥ 2 samples per group
                groups.append(group_data.values)
        
        # Skip feature if < 2 groups have sufficient data
        if len(groups) < 2:
            continue
            
        try:
            # Perform one-way ANOVA
            f_stat, p_val = f_oneway(*groups)
            
            # Calculate effect size (eta squared)
            all_data = np.concatenate(groups)
            ss_between = sum([len(g) * (np.mean(g) - np.mean(all_data))**2 
                              for g in groups])
            ss_total = sum((x - np.mean(all_data))**2 for x in all_data)
            eta_sq = ss_between / ss_total if ss_total != 0 else 0.0
            
        except (ValueError, ZeroDivisionError):
            continue  # Handle degenerate cases
            
        results.append({
            'feature': feature,
            'f_statistic': f_stat,
            'p_value': max(p_val, 1e-10),
            'eta_squared': eta_sq,
            'groups_tested': len(groups)
        })
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        if verbose:
            logger.error(
                f"No features passed one-way ANOVA for groups: {group_column_values} "
                f"in column '{group_column}'"
            )
        return pd.DataFrame(columns=['feature', 'f_statistic', 'p_value'])

    # Filter and sort results
    results_df = results_df[(results_df['p_value'] != 0) & (
        results_df['p_value'].notna()
    )]
    results_df = results_df[results_df['p_value'] <= 0.05]
    results_df = results_df.sort_values('p_value', ascending=True)
    return results_df


# CORRELATION ANALYSIS
def spearman_correlation(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    continuous_column: str,
    sample_id_column: str = "#sampleid",
    alpha: float = 0.01,
    min_samples: int = 5,
    progress: Any = None,
    task_id: Any = None
) -> pd.DataFrame:
    """Spearman correlations using the standardized merging function"""
    # Validate continuous column
    if continuous_column not in metadata.columns:
        logger.error(f"Column '{continuous_column}' not found in metadata")
        return pd.DataFrame()
    
    # Check sufficient non-missing values
    non_missing_count = metadata[continuous_column].notna().sum()
    if non_missing_count < min_samples:
        logger.error(f"Only {non_missing_count} valid values for '{continuous_column}'")
        return pd.DataFrame()
    
    # Merge table and metadata
    merged = merge_table_with_metadata(
        features_df=to_df(table), 
        metadata_df=metadata
    )
    
    if merged.empty:
        return pd.DataFrame()
    
    # Convert continuous variable to numeric
    merged[continuous_column] = pd.to_numeric(
        merged[continuous_column], errors='coerce'
    )
    
    # Drop rows with missing continuous values
    merged = merged.dropna(subset=[continuous_column])
    
    logger.debug(
        f"Correlation analysis for '{continuous_column}': "
        f"{len(merged)} samples with valid data"
    )
    
    results = []
    total = len(merged.columns.drop(continuous_column))
    
    for i, feature in enumerate(merged.columns.drop(continuous_column)):
        # Convert feature to numeric
        merged[feature] = pd.to_numeric(merged[feature], errors='coerce')
        
        # Skip non-numeric features
        if not pd.api.types.is_numeric_dtype(merged[feature]):
            logger.warning(f"Skipping non-numeric feature: {feature}")
            continue
            
        # Pairwise complete observations
        valid_idx = merged[[feature, continuous_column]].dropna().index
        n_valid = len(valid_idx)
        
        if n_valid < min_samples:
            continue
            
        try:
            rho, p_val = spearmanr(
                merged.loc[valid_idx, feature], 
                merged.loc[valid_idx, continuous_column]
            )
            results.append({
                'feature': feature,
                'rho': rho,
                'p_value': p_val,
                'n_samples': n_valid
            })
        except Exception as e:
            logger.warning(f"Correlation failed for {feature}: {str(e)}")
        finally:
            if progress and task_id:
                progress.update(task_id, description=f"Correlation of {feature}: {i} / {total}")
    
    if not results:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(results)
    result_df['p_adj'] = result_df['p_value'] * len(result_df)
    result_df['p_adj'] = result_df['p_adj'].clip(upper=1.0)
    sig_df = result_df[result_df['p_adj'] <= alpha].sort_values(
        'rho', key=abs, ascending=False
    )
    
    logger.debug(f"Found {len(sig_df)} significant correlations")
    return sig_df


# ENHANCED STATISTICAL TESTS
def enhanced_statistical_tests(
    table: Union[Dict, Table, pd.DataFrame],
    metadata: pd.DataFrame,
    group_column: str,
    sample_id_column: str = "#sampleid",
    test_type: str = 'auto',
    correction_method: str = 'fdr_bh',
    alpha: float = 0.05,
    effect_size_threshold: float = 0.5,
    progress: Any = None,
    task_id: Any = None
) -> pd.DataFrame:
    """Enhanced statistical testing with automatic test selection and effect sizes."""
    # Merge table and metadata
    merged = merge_table_with_metadata(
        features_df=to_df(table), 
        metadata_df=metadata
    )
    
    if merged.empty:
        return pd.DataFrame()

    # Ensure group column is of string type to avoid mixed-type comparison errors
    merged[group_column] = merged[group_column].astype(str)
    
    groups = merged[group_column].unique()
    n_groups = len(groups)
    
    if n_groups < 2:
        raise ValueError("Need at least 2 groups for comparison")
    
    results = []

    total = len(merged.columns.drop(group_column))
    for i, feature in enumerate(merged.columns.drop(group_column)):
        try:
            group_data = []
            for group in groups:
                data = merged[merged[group_column] == group][feature].dropna()
                if len(data) >= 3:
                    group_data.append(data)
            
            if len(group_data) < 2:
                continue
            
            # Test for normality and equal variances if auto mode
            normality_ok = True
            equal_var_ok = True
            
            if test_type == 'auto':
                for data in group_data:
                    if len(data) < 50:
                        _, p_norm = shapiro(data)
                        if p_norm < 0.05:
                            normality_ok = False
                            break
                
                if normality_ok:
                    _, p_levene = levene(*group_data)
                    if p_levene < 0.05:
                        equal_var_ok = False
            
            # Choose appropriate test
            if n_groups == 2:
                if test_type == 'parametric' or (test_type == 'auto' and normality_ok):
                    stat, p_val = ttest_ind(*group_data, equal_var=equal_var_ok)
                    test_name = "Welch's t-test" if not equal_var_ok else "Student's t-test"
                    
                    pooled_std = np.sqrt((np.var(group_data[0], ddof=1) + 
                                        np.var(group_data[1], ddof=1)) / 2)
                    effect_size = (np.mean(group_data[0]) - np.mean(group_data[1])) / pooled_std
                    
                else:
                    stat, p_val = mannwhitneyu(*group_data, alternative='two-sided')
                    test_name = "Mann-Whitney U"
                    
                    n1, n2 = len(group_data[0]), len(group_data[1])
                    effect_size = 1 - (2 * stat) / (n1 * n2)
            
            else:
                if test_type == 'parametric' or (test_type == 'auto' and normality_ok):
                    stat, p_val = f_oneway(*group_data)
                    test_name = "One-way ANOVA"
                    
                    all_data = np.concatenate(group_data)
                    grand_mean = np.mean(all_data)
                    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in group_data)
                    ss_total = sum((x - grand_mean)**2 for x in all_data)
                    effect_size = ss_between / ss_total if ss_total > 0 else 0
                    
                else:
                    stat, p_val = kruskal(*group_data)
                    test_name = "Kruskal-Wallis"
                    
                    n_total = sum(len(g) for g in group_data)
                    effect_size = stat / (n_total - 1) if n_total > 1 else 0
            
            means = [np.mean(g) for g in group_data]
            medians = [np.median(g) for g in group_data]
            
            results.append({
                'feature': feature,
                'test': test_name,
                'statistic': stat,
                'p_value': p_val,
                'effect_size': effect_size,
                'mean_values': means,
                'median_values': medians,
                'n_groups': len(group_data),
                'total_samples': sum(len(g) for g in group_data)
            })
        except Exception as e:
            logger.error(f"Error with statistical test of {feature}: {e}")
        finally:
            if progress and task_id:
                progress.update(task_id, description=f"Enhanced statistical tests: {i} / {total}")
    
    if not results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    _, p_adj, _, _ = multipletests(results_df['p_value'], method=correction_method)
    results_df['p_adj'] = p_adj
    
    significant = results_df[
        (results_df['p_adj'] <= alpha) & 
        (np.abs(results_df['effect_size']) >= effect_size_threshold)
    ]
    
    return significant.sort_values('p_adj')


