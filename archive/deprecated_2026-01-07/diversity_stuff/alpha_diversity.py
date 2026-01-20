# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from typing import Any, Dict, List, Optional, Union

# Third-Party Imports
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, kruskal, mannwhitneyu, ttest_ind
from skbio.diversity import alpha
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Local Imports
from workflow_16s.constants import DEFAULT_ALPHA_METRICS, PHYLO_METRICS
from workflow_16s.utils.data import add_metadata_column
from workflow_16s.utils.biom_utils import to_df

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ==================================== FUNCTIONS ===================================== #

def alpha_diversity(
    table: Union[Dict, Any, pd.DataFrame],
    metrics: List[str] = DEFAULT_ALPHA_METRICS,
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
    df = to_df(table)
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
                return alpha.faith_pd(values, otu_ids=df.columns, tree=tree)
            
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
        if metric in PHYLO_METRICS and tree is None:
            logger.warning(f"Skipping {metric} - phylogenetic tree not provided")
            continue
            
        metric_values = []
        for i, sample in enumerate(df.index):
            vals = df.loc[sample].values
            total = totals.iloc[i]
            non_zero = non_zeros.iloc[i]
            props = proportions.iloc[i].to_numpy()
            
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
    merged = add_metadata_column(alpha_diversity_df, metadata, group_column)
    
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
            group_vals = merged.loc[merged[group_column] == group, [metric]].dropna()[metric]
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
            common_idx = y.index.intersection(col_data.index.to_list())
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
            elif (hasattr(x.dtype, 'name') and x.dtype.name == 'category') or \
                 (not pd.api.types.is_numeric_dtype(x) and x.nunique() <= max_categories):
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
                        endog=y_vals, 
                        groups=x,
                        alpha=0.05
                    )
                    df_tukey = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
                    for _, row in df_tukey.iterrows():
                        if row['reject']:
                            pairwise.append(
                                f"{row['group1']} vs {row['group2']}: {row['p-adj']:.4f}"
                            )
                
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