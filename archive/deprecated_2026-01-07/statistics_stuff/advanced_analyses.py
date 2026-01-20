from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
from biom.table import Table
from workflow_16s.utils.biom_utils import to_df
from workflow_16s.utils.data import merge_table_with_metadata
# DIFFERENTIAL ABUNDANCE ANALYSIS
def _deseq2_like_test(group1: pd.Series, group2: pd.Series) -> Tuple[float, float]:
    """Simplified DESeq2-like differential expression test."""
    def variance_stabilize(x):
        return np.arcsinh(x)
    
    vs1 = variance_stabilize(group1)
    vs2 = variance_stabilize(group2)
    
    return ttest_ind(vs1, vs2)


def differential_abundance_analysis(
    table: Union[Dict, Any, pd.DataFrame],
    metadata: pd.DataFrame,
    group_column: str,
    sample_id_column: str = "sample_id",
    method: str = 'deseq2_like',
    alpha: float = 0.05,
    fold_change_threshold: float = 1.5,
    min_prevalence: float = 0.1,
    progress: Any = None,
    task_id: Any = None
) -> pd.DataFrame:
    """Comprehensive differential abundance analysis with multiple methods."""
    # Convert table to DataFrame
    df = to_df(table)
    
    # Filter by prevalence
    prevalence = (df > 0).mean()
    df_filt = df.loc[:, prevalence >= min_prevalence]
    
    # Merge filtered table and metadata
    merged = merge_table_with_metadata(
        df_filt, 
        metadata, 
        sample_id_column=sample_id_column,
        additional_columns=[group_column]
    )
    
    if merged.empty:
        return pd.DataFrame()
    
    groups = merged[group_column].unique()
    if len(groups) != 2:
        raise ValueError("Differential abundance analysis requires exactly 2 groups")
    
    results = []

    total = len(merged.columns.drop(group_column))
    for i, feature in enumerate(merged.columns.drop(group_column)):
        try:
            group1_data = merged[merged[group_column] == groups[0]][feature]
            group2_data = merged[merged[group_column] == groups[1]][feature]
            
            if len(group1_data) < 3 or len(group2_data) < 3:
                continue
            
            # Calculate fold change
            mean1 = group1_data.mean()
            mean2 = group2_data.mean()
            fold_change = (mean1 + 1e-8) / (mean2 + 1e-8)
            log2_fc = np.log2(fold_change)
            
            if abs(log2_fc) < np.log2(fold_change_threshold):
                continue
            
            # Statistical testing
            if method == 'wilcoxon':
                statistic, p_value = mannwhitneyu(group1_data, group2_data)
            elif method == 'ttest':
                statistic, p_value = ttest_ind(group1_data, group2_data)
            elif method == 'deseq2_like':
                statistic, p_value = _deseq2_like_test(group1_data, group2_data)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            results.append({
                'feature': feature,
                'log2_fold_change': log2_fc,
                'fold_change': fold_change,
                'mean_group1': mean1,
                'mean_group2': mean2,
                'statistic': statistic,
                'p_value': p_value,
                'prevalence': prevalence[feature]
            })
        except Exception as e:
            logger.error(f"Error with DA analysis of {feature}: {e}")
        finally:
            if progress and task_id:
                progress.update(task_id, description=_format_task_desc(f"{i} / {total}"))
    
    if not results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    _, p_adj, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
    results_df['p_adj'] = p_adj
    
    significant = results_df[results_df['p_adj'] <= alpha]
    significant = significant.sort_values('p_adj')
    
    return significant


# CORE MICROBIOME ANALYSIS
def core_microbiome(
    table: Union[Dict, pd.DataFrame, Table],
    metadata: pd.DataFrame,
    group_column: str,
    sample_id_column: str = "sample_id",
    prevalence_threshold: float = 0.8,
    abundance_threshold: float = 0.01
) -> Dict[str, pd.DataFrame]:
    """Identify core microbiome for each group using standardized merging"""
    # Merge table and metadata
    merged = merge_table_with_metadata(
        table, 
        metadata, 
        sample_id_column=sample_id_column,
        additional_columns=[group_column]
    )
    
    if merged.empty:
        return {}
    
    # Calculate relative abundance
    feature_columns = [col for col in merged.columns if col != group_column]
    rel_abundance = merged[feature_columns].div(
        merged[feature_columns].sum(axis=1), axis=0
    )
    
    # Add group column back
    rel_abundance[group_column] = merged[group_column]
    
    core_features = {}
    groups = rel_abundance[group_column].unique()
    logger.info(f"Groups found: {groups}")
    
    for group_val in groups:
        # Filter data for current group
        group_data = rel_abundance[rel_abundance[group_column] == group_val]
        group_features = group_data.drop(columns=[group_column])
        
        if group_features.empty:
            logger.warning(f"No data for group: {group_val}")
            continue
        
        # Calculate prevalence and mean abundance
        prevalence = (group_features > 0).mean()
        mean_abundance = group_features.mean()
        
        # Identify core features
        core_mask = (
            (prevalence >= prevalence_threshold) & 
            (mean_abundance >= abundance_threshold)
        )
        
        if not core_mask.any():
            logger.warning(f"No core features found for group {group_val} with thresholds: "
                          f"prevalence={prevalence_threshold}, abundance={abundance_threshold}")
            continue
        
        # Create results DataFrame
        core_df = pd.DataFrame({
            'feature': core_mask.index[core_mask],
            'prevalence': prevalence[core_mask],
            'mean_abundance': mean_abundance[core_mask],
            'group': group_val
        })
        
        core_features[group_val] = core_df.sort_values('mean_abundance', ascending=False)
        logger.info(f"Found {len(core_df)} core features for group {group_val}")
    
    return core_features


# NETWORK ANALYSIS
def microbial_network_analysis(
    table: Union[Dict, Table, pd.DataFrame],
    method: str = 'sparcc',
    threshold: float = 0.3,
    min_prevalence: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construct microbial co-occurrence networks."""
    df = table_to_df(table)
    prevalence = (df > 0).mean()
    df_filt = df.loc[:, prevalence >= min_prevalence]
    
    logger.debug(f"Network analysis: {df_filt.shape[1]} features after prevalence filtering")
    
    if method == 'sparcc':
        corr_matrix = df_filt.corr(method='spearman')
    elif method in ['spearman', 'pearson']:
        corr_matrix = df_filt.corr(method=method)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    edges = []
    n_features = len(corr_matrix)
    
    for i in range(n_features):
        for j in range(i + 1, n_features):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                edges.append({
                    'source': corr_matrix.index[i],
                    'target': corr_matrix.index[j],
                    'correlation': corr_val,
                    'abs_correlation': abs(corr_val),
                    'edge_type': 'positive' if corr_val > 0 else 'negative'
                })
    
    edges_df = pd.DataFrame(edges).sort_values('abs_correlation', ascending=False)
    
    return corr_matrix, edges_df
    