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
    fisher_exact, f_oneway, kruskal, mannwhitneyu, spearmanr, ttest_ind
)
from skbio.diversity import alpha
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import pcoa as PCoA
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tqdm import tqdm

# Local Imports
from workflow_16s import constants
from workflow_16s.utils.data import merge_table_with_meta, table_to_df

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ==================================== FUNCTIONS ===================================== #

def alpha_diversity(
    table: Union[Dict, Any, pd.DataFrame],
    metrics: List[str] = constants.DEFAULT_ALPHA_METRICS,
    tree: Optional[Any] = None,
    pseudo_count: float = 1e-12
) -> pd.DataFrame:
    """Calculate alpha diversity metrics for each sample.
    
    Args:
        table:        Input abundance table (samples x features).
        metrics:      List of alpha diversity metrics to compute.
        tree:         Phylogenetic tree (required for phylogenetic metrics).
        pseudo_count: Small value to avoid log(0) (: 1e-12).
        
    Returns:
        DataFrame with alpha diversity values (samples x metrics).
    """
    df = table_to_df(table)
    results = pd.DataFrame(index=df.index)
    
    # Precompute common statistics vectorially
    totals = df.sum(axis=1)
    non_zeros = (df > 0).sum(axis=1)
    proportions = df.div(totals, axis=0).fillna(0)
    
    # Track which metrics we've warned about non-integer values
    warned_metrics = set()
    
    def calculate_metric(
        metric: str, 
        values: np.ndarray, 
        total: float, 
        non_zero: int, 
        proportions: np.ndarray
    ) -> float:
        """Helper function to compute a single metric for a sample"""
        nonlocal warned_metrics
        
        try:
            # Phylogenetic metrics
            if metric in ['faith_pd', 'pd_whole_tree']:
                return alpha.faith_pd(values, ids=df.columns, tree=tree)
            
            # Richness metrics
            elif metric == 'observed_features':
                return non_zero
            elif metric == 'chao1':
                # Check and convert to integers for chao1
                if np.allclose(values, np.round(values), atol=1e-5):
                    int_vals = np.round(values).astype(int)
                    return alpha.chao1(int_vals)
                else:
                    if metric not in warned_metrics:
                        logger.warning(
                            f"Non-integer values detected for {metric}. "
                            "Requires integer counts. Returning NaN."
                        )
                        warned_metrics.add(metric)
                    return np.nan
            
            # Diversity indices
            elif metric == 'shannon':
                return alpha.shannon(proportions, base=np.e)
            elif metric == 'simpson':
                return alpha.simpson(proportions)
            
            # Evenness metrics
            elif metric == 'pielou_evenness':
                shannon_val = alpha.shannon(proportions, base=np.e)
                return shannon_val / np.log(non_zero) if non_zero > 0 else 0.0
            elif metric == 'heip_evenness':
                simpson_val = alpha.simpson(proportions)
                return (1 - simpson_val) / (1 - 1/non_zero) if non_zero > 1 else 0.0
            
            # Dominance metrics
            elif metric == 'berger_parker_dominance':
                return np.max(values) / total if total > 0 else 0.0
            elif metric == 'mcintosh_dominance':
                return total / np.sqrt(np.sum(values**2)) if total > 0 else 0.0
            elif metric == 'dominance':
                # Simpson's dominance index (1 - Simpson's evenness)
                return 1 - alpha.simpson(proportions)
            
            # Rarefaction metrics
            elif metric == 'ace':
                # Check and convert to integers for ace
                if np.allclose(values, np.round(values), atol=1e-5):
                    int_vals = np.round(values).astype(int)
                    return alpha.ace(int_vals)
                else:
                    if metric not in warned_metrics:
                        logger.warning(
                            f"Non-integer values detected for {metric}. "
                            "Requires integer counts. Returning NaN."
                        )
                        warned_metrics.add(metric)
                    return np.nan
                    
            elif metric == 'goods_coverage':
                # Check and convert to integers for goods_coverage
                if np.allclose(values, np.round(values), atol=1e-5):
                    int_vals = np.round(values).astype(int)
                    total_int = int_vals.sum()
                    singletons = (int_vals == 1).sum()
                    return 1 - singletons/total_int if total_int > 0 else 0.0
                else:
                    if metric not in warned_metrics:
                        logger.warning(
                            f"Non-integer values detected for {metric}. "
                            "Requires integer counts. Returning NaN."
                        )
                        warned_metrics.add(metric)
                    return np.nan
            
            # Gini index
            elif metric == 'gini_index':
                sorted_vals = np.sort(values)
                n = len(values)
                cum_sum = np.cumsum(sorted_vals)
                return 1 - (2 * np.sum(cum_sum)) / (n * total) if total > 0 else 0.0
            
            # Fallback to skbio's alpha functions
            else:
                # Check if metric exists in skbio's alpha module
                if hasattr(alpha, metric):
                    func = getattr(alpha, metric)
                    # For metrics that require proportions
                    if metric in ['shannon', 'simpson', 'pielou_e', 'heip_e']:
                        return func(proportions)
                    # For metrics that require counts
                    else:
                        return func(values)
                else:
                    logger.warning(f"Unsupported alpha diversity metric: {metric}")
                    return np.nan
                
        except Exception as e:
            logger.warning(f"Error calculating {metric}: {str(e)}")
            return np.nan

    # Compute each metric for all samples
    for metric in metrics:
        if metric in constants.PHYLO_METRICS and tree is None:
            logger.warning(f"Skipping {metric} - phylogenetic tree not provided")
            continue
            
        metric_values = []
        for i, sample in enumerate(df.index):
            vals = df.loc[sample].values
            total = totals.iloc[i]
            non_zero = non_zeros.iloc[i]
            props = proportions.iloc[i].values
            
            metric_values.append(
                calculate_metric(metric, vals, total, non_zero, props)
            )
            
        results[metric] = metric_values

    return results
    

def analyze_alpha_diversity(
    alpha_diversity_df: pd.DataFrame,
    metadata: pd.DataFrame,
    group_column: str = 'nuclear_contamination_status',
    parametric: bool = False
) -> pd.DataFrame:
    """Analyze relationship between alpha diversity metrics and a grouping variable.
    
    Args:
        alpha_diversity_df: DataFrame from alpha_diversity() (samples x metrics).
        metadata:           Metadata DataFrame (must include group_col).
        group_col:          Metadata column containing group labels.
        parametric:         Use parametric tests (False for non-parametric).
        
    Returns:
        DataFrame with statistical results (metric, test, p-value, effect_size).
    """
    merged = merge_table_with_meta(alpha_diversity_df, metadata, group_column)
    
    # Check group validity
    groups = merged[group_column].dropna().unique()
    if len(groups) < 2:
        raise ValueError(
            f"Grouping column '{group_column}' must contain at least 2 groups"
        )
    
    results = []
    
    for metric in alpha_diversity_df.columns:
        # Prepare data: list of values per group
        group_data = []
        for group in groups:
            group_vals = merged.loc[merged[group_column] == group, metric].dropna()
            if len(group_vals) == 0:
                logger.warning(f"No data for {metric} in group '{group}'")
                continue
            group_data.append(group_vals)
        
        # Skip metric if <2 groups have data
        if len(group_data) < 2:
            logger.warning(f"Insufficient groups for {metric} - skipping")
            continue
        
        # Statistical testing
        test_name = ""
        test_result = None
        effect_size = np.nan
        
        try:
            if parametric:
                # Parametric tests
                if len(group_data) == 2:
                    # T-test for two groups
                    t_stat, p_val = ttest_ind(*group_data, equal_var=False)
                    test_name = "Welch's t-test"
                    test_result = (t_stat, p_val)
                    
                    # Cohen's d effect size
                    n1 = len(group_data[0])
                    n2 = len(group_data[1])
                    pooled_std = np.sqrt(
                        ((n1-1)*np.var(group_data[0], ddof=1) + 
                         (n2-1)*np.var(group_data[1], ddof=1)) 
                        / (n1+n2-2)
                    )
                    effect_size = (np.mean(group_data[0]) - np.mean(group_data[1])) / pooled_std
                else:
                    # One-way ANOVA for >2 groups
                    f_stat, p_val = f_oneway(*group_data)
                    test_name = "ANOVA"
                    test_result = (f_stat, p_val)
                    
                    # Eta squared effect size
                    all_values = np.concatenate(group_data)
                    grand_mean = np.mean(all_values)
                    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in group_data)
                    ss_total = sum((x - grand_mean)**2 for x in all_values)
                    effect_size = ss_between / ss_total if ss_total != 0 else 0.0
            else:
                # Non-parametric tests
                if len(group_data) == 2:
                    # Mann-Whitney U test for two groups
                    u_stat, p_val = mannwhitneyu(*group_data)
                    test_name = "Mann-Whitney U"
                    test_result = (u_stat, p_val)
                    
                    # Rank-biserial correlation effect size
                    n1 = len(group_data[0])
                    n2 = len(group_data[1])
                    effect_size = 1 - (2 * u_stat) / (n1 * n2)
                else:
                    # Kruskal-Wallis for >2 groups
                    h_stat, p_val = kruskal(*group_data)
                    test_name = "Kruskal-Wallis"
                    test_result = (h_stat, p_val)
                    
                    # Epsilon squared effect size
                    n_total = sum(len(g) for g in group_data)
                    effect_size = h_stat / ((n_total**2 - 1) / (n_total + 1))
            
            # Store results
            results.append({
                'metric': metric,
                'test': test_name,
                'statistic': test_result[0],
                'p_value': test_result[1],
                'effect_size': effect_size,
                'groups': len(group_data)
            })
            
        except Exception as e:
            logger.error(f"Error analyzing {metric}: {str(e)}")
    
    return pd.DataFrame(results)


def analyze_alpha_correlations(
    alpha_df: pd.DataFrame,
    metadata: pd.DataFrame,
    max_categories: int = 20,
    min_samples: int = 5
) -> Dict[str, pd.DataFrame]:
    """Analyze relationships between alpha diversity metrics and metadata columns.
    
    Args:
        alpha_df:       DataFrame of alpha diversity metrics.
        metadata:       Sample metadata DataFrame.
        max_categories: Maximum unique values for categorical variables.
        min_samples:    Minimum samples per group for valid comparison.
        
    Returns:
        Dictionary of DataFrames with correlation results per metric.
    """
    results = {}
    
    # Align indices
    common_idx = alpha_df.index.intersection(metadata.index)
    alpha_df = alpha_df.loc[common_idx]
    meta = metadata.loc[common_idx]
    
    for metric in alpha_df.columns:
        metric_results = []
        y = alpha_df[metric]
        
        for col in meta.columns:
            # Skip columns with too many missing values
            if meta[col].isna().mean() > 0.5:
                continue
                
            col_data = meta[col].dropna()
            common_idx = y.index.intersection(col_data.index)
            x = col_data.loc[common_idx]
            y_vals = y.loc[common_idx]
            
            # Skip if insufficient data
            if len(x) < 10:
                continue
                
            # Handle numerical columns
            if pd.api.types.is_numeric_dtype(x):
                # Pearson correlation
                r_pearson, p_pearson = stats.pearsonr(x, y_vals)
                # Spearman correlation
                r_spearman, p_spearman = stats.spearmanr(x, y_vals)
                
                metric_results.append({
                    'metadata_column': col,
                    'type': 'numerical',
                    'pearson_r': r_pearson,
                    'pearson_p': p_pearson,
                    'spearman_rho': r_spearman,
                    'spearman_p': p_spearman,
                    'n_samples': len(x)
                })
                
            # Handle categorical columns
            elif pd.api.types.is_categorical_dtype(x) or x.nunique() <= max_categories:
                # ANOVA for parametric
                groups = x.unique()
                group_data = [y_vals[x == g] for g in groups]
                
                # Skip small groups
                if any(len(g) < min_samples for g in group_data):
                    continue
                    
                # Kruskal-Wallis for non-parametric
                h_stat, p_kruskal = stats.kruskal(*group_data)
                
                # Calculate eta squared (effect size)
                ss_between = h_stat
                ss_total = len(y_vals) - 1
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                # Pairwise comparisons
                pairwise = []
                if len(groups) > 2 and p_kruskal < 0.05:
                    tukey = pairwise_tukeyhsd(
                        y_vals.values, 
                        x.values
                    )
                    pairwise = [
                        f"{groups[i]} vs {groups[j]}: {p:.4f}" 
                        for i, j, p in zip(
                            tukey._results[0], tukey._results[1], tukey._results[4]
                        )
                    ]
                
                metric_results.append({
                    'metadata_column': col,
                    'type': 'categorical',
                    'n_categories': len(groups),
                    'kruskal_h': h_stat,
                    'kruskal_p': p_kruskal,
                    'eta_squared': eta_squared,
                    'pairwise_comparisons': "; ".join(pairwise),
                    'n_samples': len(x)
                })
        
        # Create sorted DataFrame for this metric
        df = pd.DataFrame(metric_results)
        
        # Add ranking columns
        if not df.empty:
            # Numerical columns - rank by absolute Spearman correlation
            num_df = df[df['type'] == 'numerical'].copy()
            num_df['strength_rank'] = num_df['spearman_rho'].abs().rank(ascending=False)
            
            # Categorical columns - rank by eta squared
            cat_df = df[df['type'] == 'categorical'].copy()
            cat_df['strength_rank'] = cat_df['eta_squared'].rank(ascending=False)
            
            # Combine and sort
            df = pd.concat([num_df, cat_df]).sort_values('strength_rank')
            df = df.reset_index(drop=True)
        
        results[metric] = df
    
    return results
    

def k_means(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = constants.DEFAULT_GROUP_COLUMN,
    n_clusters: int = constants.DEFAULT_N_CLUSTERS, 
    random_state: int = constants.DEFAULT_RANDOM_STATE,
    verbose: bool = False
) -> pd.Series:
    """Apply K-means clustering and return cluster labels.

    Args:
        table:
        metadata:
        group_column:
        n_clusters:
        random_state:
        verbose:

    Returns:
    """
    table = table_to_df(table)
    table_with_column = merge_table_with_meta(table, metadata, group_column)
    
    kmeans = KMeans(
        n_clusters, 
        random_state=random_state
    ).fit(table_with_column.drop(group_column, axis=1))

    results = pd.Series(
        kmeans.labels_, 
        index=table_with_column.index, 
        name='kmeans_cluster'
    )
    return results
    

def ttest(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = constants.DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = constants.DEFAULT_GROUP_COLUMN_VALUES,
    equal_var: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """Performs independent t-tests for two groups.
    
    Args:
        table:               Input abundance table (samples x features).
        metadata:            Sample metadata DataFrame.
        group_column:        Metadata column containing group labels.
        group_column_values: Two group identifiers to compare.
        equal_var:           Whether to assume equal population variances (default: False).
        verbose:
        
    Returns:
        DataFrame with significant features (p < Bonferroni-corrected threshold).
    """
    table = table_to_df(table)
    table_with_column = merge_table_with_meta(table, metadata, group_column)
    
    results = []
    for feature in table_with_column.columns.drop(group_column):
        # Subset groups
        mask_group1 = (table_with_column[group_column] == group_column_values[0])
        mask_group2 = (table_with_column[group_column] == group_column_values[1])
        
        group1_values = table_with_column.loc[mask_group1, feature].dropna()
        group2_values = table_with_column.loc[mask_group2, feature].dropna()
        
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

        if constants.debug_mode:
            print(f"{feature}: {t_stat}, {p_val}, {mean_diff}, {cohen_d}")
            
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
                    f"{table.shape} {table_with_column.shape} "
                    f"{table.index} {table_with_column.index} "
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
    group_column: str = constants.DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = constants.DEFAULT_GROUP_COLUMN_VALUES,
    verbose: bool = False
) -> pd.DataFrame:
    """Performs Mann-Whitney U tests with Bonferroni correction for two groups.
    
    Args:
        table:               Input abundance table (samples x features).
        metadata:            Sample metadata DataFrame.
        group_column:        Metadata column containing group labels.
        group_column_values: Two group identifiers to compare.
        verbose:
        
    Returns:
        Results with p-values below Bonferroni-corrected threshold.
    """
    table = table_to_df(table)
    table_with_column = merge_table_with_meta(table, metadata, group_column)
    
    # Total features tested (for Bonferroni)
    total_features = len(table_with_column.columns.drop(group_column))
    threshold = 0.01 / total_features
    
    results = []
    for feature in table_with_column.columns.drop(group_column):
        # Subset groups safely
        mask_group1 = (table_with_column[group_column] == group_column_values[0])
        mask_group2 = (table_with_column[group_column] == group_column_values[1])
        
        group1_values = table_with_column.loc[mask_group1, feature].dropna()
        group2_values = table_with_column.loc[mask_group2, feature].dropna()
        
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

        if constants.debug_mode:
            print(f"{feature}: {u_stat}, {p_val}, {median_diff}, {r}")
            
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


def kruskal_bonferroni(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = constants.DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """Performs Kruskal-Wallis H-test with Bonferroni correction for ≥3 groups.
    
    Args:
        table:               Input abundance table (samples x features).
        metadata:            Sample metadata DataFrame.
        group_column:        Metadata column containing group labels.
        group_column_values: List of group identifiers to compare 
                             (None = use all groups).
        verbose:
        
    Returns:
        DataFrame with significant features after Bonferroni correction.
    """
    table = table_to_df(table)
    table_with_column = merge_table_with_meta(table, metadata, group_column)
    
    # Get unique groups if group_column_values not specified
    if group_column_values is None:
        group_column_values = table_with_column[group_column].unique().tolist()
    
    # Pre-calculate Bonferroni threshold
    total_features = len(table_with_column.columns.drop(group_column))
    threshold = 0.01 / total_features
    
    results = []
    for feature in table_with_column.columns.drop(group_column):
        # Collect data for all groups
        groups = []
        for group_value in group_column_values:
            mask = (table_with_column[group_column] == group_value)
            group_data = table_with_column.loc[mask, feature].dropna()
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
        
        if constants.debug_mode:
            print(f"{feature}: {h_stat}, {p_val}, {epsilon_sq}")
            
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
    group_column: str = constants.DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """Performs one-way ANOVA for ≥3 groups.
    
    Args:
        table:               Input abundance table (samples x features).
        metadata:            Sample metadata DataFrame.
        group_column:        Metadata column containing group labels.
        group_column_values: List of group identifiers to compare 
                             (None = use all groups).
        verbose:
        
    Returns:
        DataFrame with significant features after Bonferroni correction.
    
    Note: 
        - Effect size (eta squared) represents the proportion of variance 
          explained by groups. Values range from 0 to 1, with higher values 
          indicating stronger group separation.
    """
    table = table_to_df(table)
    table_with_column = merge_table_with_meta(table, metadata, group_column)
    
    # Get unique groups if group_column_values not specified
    if group_column_values is None:
        group_column_values = table_with_column[group_column].unique().tolist()
    
    results = []
    for feature in table_with_column.columns.drop(group_column):
        # Collect data for all groups
        groups = []
        for group_value in group_column_values:
            mask = (table_with_column[group_column] == group_value)
            group_data = table_with_column.loc[mask, feature].dropna()
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


def fisher_exact_bonferroni(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str,
    group_column_values: List[Union[bool, int, str]],
    alpha: float = 0.01,
    min_samples: int = 5,
    debug_mode: bool = constants.debug_mode,
    verbose: bool = False
) -> pd.DataFrame:
    """Performs Fisher's Exact Tests with Bonferroni correction for 
    presence-absence data.
    
    Args:
        table:               Input presence-absence table (samples x 
                             features, binary 0/1).
        metadata:            Sample metadata DataFrame.
        group_column:        Metadata column containing group labels.
        group_column_values: Two group identifiers to compare.
        alpha:               Significance level before correction 
                             (default: 0.01).
        min_samples:         Minimum samples required per group 
                             (default: 5).
        debug_mode:          Print debug information if True.
        verbose:
        
    Returns:
        DataFrame with significant results (p-value ≤ Bonferroni-
        corrected threshold)
    """
    # Convert to DataFrame and merge with metadata
    table_df = table_to_df(table)
    merged_df = merge_table_with_meta(table_df, metadata, group_column)
    
    # Total features for Bonferroni correction
    total_features = len(merged_df.columns) - 1  # Exclude group column
    threshold = alpha / total_features
    
    results = []
    for feature in merged_df.columns.drop(group_column):
        # Subset groups
        mask_group1 = (merged_df[group_column] == group_column_values[0])
        mask_group2 = (merged_df[group_column] == group_column_values[1])
        
        group1 = merged_df.loc[mask_group1, feature].dropna()
        group2 = merged_df.loc[mask_group2, feature].dropna()
        
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
        
        if constants.debug_mode:
            print(
                f"{feature}: OR={odds_ratio:.3f}, "
                f"p={p_val:.4f}, "
                f"diff={prop_diff:.2f}"
            )
            
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


def spearman_correlation(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    continuous_column: str,
    alpha: float = 0.01
) -> pd.DataFrame:
    """Calculate Spearman correlations between features and a 
    continuous metadata variable.
    
    Args:
        table:             Input abundance table.
        metadata:          Sample metadata.
        continuous_column: Metadata column with continuous values.
        alpha:             Significance threshold.
        
    Returns:
        DataFrame with correlation results.
    """
    df = table_to_df(table)
    merged = merge_table_with_meta(df, metadata, continuous_column)
    
    results = []
    for feature in tqdm(
        merged.columns.drop(continuous_column), 
        desc="Calculating correlations"
    ):
        # Remove NA values pairwise
        valid_idx = merged[[feature, continuous_column]].dropna().index
        if len(valid_idx) < 3:
            continue
            
        subset = merged.loc[valid_idx]
        rho, p_val = spearmanr(subset[feature], subset[continuous_column])
        
        results.append({
            'feature': feature,
            'rho': rho,
            'p_value': p_val,
            'n_samples': len(valid_idx)
        })
    
    result_df = pd.DataFrame(results)
    result_df['p_adj'] = result_df['p_value'] * len(result_df)
    results_df = result_df[result_df['p_adj'] <= alpha].sort_values(
        'rho', key=abs, ascending=False
    )
    return results_df


def calculate_distance_matrix(
    table: Union[Dict, Table, pd.DataFrame],
    metric: str = 'braycurtis'
) -> DistanceMatrix:
    """Calculate distance matrix from abundance table.
    
    Args:
        table:  Input abundance table.
        metric: Distance metric (default: braycurtis).
        
    Returns:
        skbio DistanceMatrix object.
    """
    df = table_to_df(table)
    ids = df.index.tolist()
    dist_array = pdist(df.values, metric=metric)
    return DistanceMatrix(squareform(dist_array), ids)
    

def run_ordination(
    table: Union[Dict, Table, pd.DataFrame],
    method: str = 'pca',
    n_components: int = 2,
    random_state: int = constants.DEFAULT_RANDOM_STATE
) -> pd.DataFrame:
    """Perform dimensionality reduction.
    
    Args:
        table:        Input abundance table.
        method:       Ordination method to use: 'pca', 'pcoa', 'tsne', 
                      or 'umap'.
        n_components: Number of dimensions to keep.
        random_state: Random seed.
        
    Returns:
        DataFrame with ordination coordinates
    """
    df = table_to_df(table)
    scaled = StandardScaler().fit_transform(df)
    
    if method == 'pca':
        model = PCA(n_components=n_components, random_state=random_state)
        results = model.fit_transform(scaled)
        print('results')
    elif method == 'pcoa':
        dm = calculate_distance_matrix(df)
        results = PCoA(dm).scores(scores_df).samples.values[:, :n_components]
    elif method == 'tsne':
        model = TSNE(n_components=n_components, random_state=random_state)
        results = model.fit_transform(scaled)
    elif method == 'umap':
        model = UMAP(n_components=n_components, random_state=random_state)
        results = model.fit_transform(scaled)
    else:
        raise ValueError(f"Unknown method: {method}")
        
    df = pd.DataFrame(
        results, 
        index=df.index, 
        columns=[f"{method.upper()}{i+1}" for i in range(n_components)]
    )
    return df
