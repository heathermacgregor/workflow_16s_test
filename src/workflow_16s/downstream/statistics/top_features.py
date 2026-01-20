"""
Top Features Analysis and Summary

Aggregates results across multiple statistical tests to identify
the most consistently important features.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger('workflow_16s')


def create_top_features_table(
    stats_results: Dict,
    n_top: int = 20,
    taxonomy_df: Optional[pd.DataFrame] = None,
    sort_by: str = 'frequency'
) -> pd.DataFrame:
    """
    Create summary table of top features across all statistical tests.
    
    Args:
        stats_results: Dictionary of {test_name: results_df}
        n_top: Number of top features to include
        taxonomy_df: DataFrame with taxonomy info (optional)
        sort_by: 'frequency' (# tests significant) or 'effect_size' (mean effect)
        
    Returns:
        Summary DataFrame with top features
    """
    logger.info("Creating top features summary table")
    
    # Collect all significant features across tests
    feature_records = []
    
    for test_name, results_df in stats_results.items():
        if not isinstance(results_df, pd.DataFrame):
            continue
        
        if 'p_value' not in results_df.columns and 'p_adj' not in results_df.columns:
            continue
        
        # Get p-value column
        p_col = 'p_adj' if 'p_adj' in results_df.columns else 'p_value'
        
        # Filter significant
        sig_mask = results_df[p_col] < 0.05
        
        for feature_id in results_df[sig_mask].index:
            record = {
                'feature_id': feature_id,
                'test': test_name,
                'p_value': results_df.loc[feature_id, p_col]
            }
            
            # Add effect size if available
            if 'effect_size' in results_df.columns:
                record['effect_size'] = results_df.loc[feature_id, 'effect_size']
            
            # Add fold change if available
            if 'log2_fold_change' in results_df.columns:
                record['log2FC'] = results_df.loc[feature_id, 'log2_fold_change']
            
            feature_records.append(record)
    
    if not feature_records:
        logger.warning("No significant features found across tests")
        return pd.DataFrame()
    
    # Convert to DataFrame
    all_results = pd.DataFrame(feature_records)
    
    # Aggregate by feature
    summary_data = []
    
    for feature_id in all_results['feature_id'].unique():
        feature_data = all_results[all_results['feature_id'] == feature_id]
        
        # Count tests where significant
        n_tests = len(feature_data)
        
        # Get test names
        tests = ', '.join(feature_data['test'].tolist())
        
        # Mean p-value and effect size
        mean_p = feature_data['p_value'].mean()
        min_p = feature_data['p_value'].min()
        
        record = {
            'feature_id': feature_id,
            'n_significant_tests': n_tests,
            'tests': tests,
            'mean_p_value': mean_p,
            'min_p_value': min_p
        }
        
        # Add mean effect size if available
        if 'effect_size' in feature_data.columns:
            mean_effect = feature_data['effect_size'].abs().mean()
            record['mean_effect_size'] = mean_effect
        
        # Add taxonomy if available
        if taxonomy_df is not None and feature_id in taxonomy_df.index:
            tax_cols = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
            tax_str = '; '.join([
                f"{col[0]}__{taxonomy_df.loc[feature_id, col]}"
                for col in tax_cols
                if col in taxonomy_df.columns and pd.notna(taxonomy_df.loc[feature_id, col])
            ])
            record['taxonomy'] = tax_str
        
        summary_data.append(record)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Sort
    if sort_by == 'frequency':
        summary_df = summary_df.sort_values(
            ['n_significant_tests', 'mean_p_value'],
            ascending=[False, True]
        )
    elif sort_by == 'effect_size' and 'mean_effect_size' in summary_df.columns:
        summary_df = summary_df.sort_values(
            'mean_effect_size',
            ascending=False
        )
    
    # Take top N
    summary_df = summary_df.head(n_top)
    
    logger.info(f"Top features table created: {len(summary_df)} features")
    
    return summary_df


def plot_top_features_heatmap(
    top_features_df: pd.DataFrame,
    stats_results: Dict,
    output_path: Optional[Path] = None
) -> go.Figure:
    """
    Create heatmap showing top features across tests.
    
    Args:
        top_features_df: DataFrame from create_top_features_table()
        stats_results: Original statistics results
        output_path: Where to save plot
        
    Returns:
        Plotly figure
    """
    if top_features_df.empty:
        logger.warning("No features to plot")
        return None
    
    # Get list of features and tests
    features = top_features_df['feature_id'].tolist()
    tests = list(stats_results.keys())
    
    # Create matrix: features × tests
    matrix = np.zeros((len(features), len(tests)))
    
    for i, feature_id in enumerate(features):
        for j, test_name in enumerate(tests):
            results_df = stats_results[test_name]
            
            if not isinstance(results_df, pd.DataFrame):
                continue
            
            if feature_id in results_df.index:
                # Get -log10(p-value)
                p_col = 'p_adj' if 'p_adj' in results_df.columns else 'p_value'
                p_val = results_df.loc[feature_id, p_col]
                
                if p_val > 0:
                    matrix[i, j] = -np.log10(p_val)
    
    # Get taxonomy labels if available
    if 'taxonomy' in top_features_df.columns:
        feature_labels = top_features_df['taxonomy'].tolist()
    else:
        feature_labels = features
    
    # Truncate long labels
    feature_labels = [label[:60] + '...' if len(label) > 60 else label 
                     for label in feature_labels]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=tests,
        y=feature_labels,
        colorscale='Reds',
        colorbar=dict(title="-log10(p)"),
        hoverongaps=False,
        hovertemplate='Test: %{x}<br>Feature: %{y}<br>-log10(p): %{z:.2f}<extra></extra>'
    ))
    
    # Add significance threshold line
    fig.add_shape(
        type="line",
        x0=-0.5, x1=len(tests)-0.5,
        y0=-0.5, y1=-0.5,
        line=dict(color="blue", width=2, dash="dash")
    )
    
    fig.update_layout(
        title="Top Features Across Statistical Tests",
        xaxis_title="Statistical Test",
        yaxis_title="Feature",
        height=400 + len(features) * 20,
        width=600 + len(tests) * 80,
        template='plotly_white'
    )
    
    if output_path:
        fig.write_html(str(output_path))
        logger.info(f"Top features heatmap saved to: {output_path}")
    
    return fig


def create_feature_consistency_plot(
    top_features_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> go.Figure:
    """
    Plot showing consistency of features across tests.
    
    Args:
        top_features_df: DataFrame from create_top_features_table()
        output_path: Where to save
        
    Returns:
        Plotly figure
    """
    if top_features_df.empty:
        return None
    
    # Create bar plot
    fig = go.Figure()
    
    # Get top 20 most consistent features
    top_20 = top_features_df.head(20).copy()
    
    # Reverse order for plotting (most significant at top)
    top_20 = top_20.iloc[::-1]
    
    # Get labels
    if 'taxonomy' in top_20.columns:
        labels = top_20['taxonomy'].tolist()
    else:
        labels = top_20['feature_id'].tolist()
    
    # Truncate
    labels = [label[:50] + '...' if len(label) > 50 else label for label in labels]
    
    # Color by mean effect size if available
    if 'mean_effect_size' in top_20.columns:
        colors = top_20['mean_effect_size'].tolist()
        colorscale = 'RdYlGn'
    else:
        colors = top_20['n_significant_tests'].tolist()
        colorscale = 'Blues'
    
    fig.add_trace(go.Bar(
        x=top_20['n_significant_tests'],
        y=labels,
        orientation='h',
        marker=dict(
            color=colors,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title="Mean Effect Size" if 'mean_effect_size' in top_20.columns else "# Tests"
            )
        ),
        text=top_20['n_significant_tests'],
        textposition='auto',
        hovertemplate='%{y}<br>Significant in %{x} tests<extra></extra>'
    ))
    
    fig.update_layout(
        title="Feature Consistency Across Tests<br><sub>Top 20 Most Consistently Significant Features</sub>",
        xaxis_title="Number of Tests Where Significant",
        yaxis_title="Feature",
        height=500 + len(top_20) * 15,
        width=900,
        template='plotly_white',
        showlegend=False
    )
    
    if output_path:
        fig.write_html(str(output_path))
        logger.info(f"Feature consistency plot saved to: {output_path}")
    
    return fig


def export_top_features_summary(
    top_features_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Export top features to CSV with annotations.
    
    Args:
        top_features_df: DataFrame from create_top_features_table()
        output_path: Where to save CSV
    """
    if top_features_df.empty:
        logger.warning("No top features to export")
        return
    
    # Round numeric columns
    for col in top_features_df.columns:
        if top_features_df[col].dtype in [np.float64, np.float32]:
            top_features_df[col] = top_features_df[col].round(4)
    
    # Save
    top_features_df.to_csv(output_path, index=False)
    logger.info(f"Top features summary exported to: {output_path}")
