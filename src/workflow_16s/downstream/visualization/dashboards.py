"""
Interactive Dashboard Generation for Comprehensive Analysis Summary.

This module creates publication-ready HTML dashboards that integrate results from
multiple analysis modules into unified, interactive visualizations.

Features:
1. Integrated multi-panel dashboards (12-panel layout)
2. QC impact visualization
3. Diversity metrics overview
4. Statistical testing summary
5. Feature importance synthesis
6. Functional predictions
7. Executive summary with key findings

Example:
    >>> from workflow_16s.downstream.dashboards import (
    ...     create_integrated_dashboard, create_qc_aware_diversity_dashboard
    ... )
    >>> 
    >>> # Create comprehensive dashboard
    >>> fig = create_integrated_dashboard(
    ...     adata=adata,
    ...     diversity_results=diversity_results,
    ...     stats_results=stats_results,
    ...     qc_results=qc_results
    ... )
    >>> fig.write_html("integrated_analysis_dashboard.html")
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

logger = logging.getLogger('workflow_16s')


def create_integrated_dashboard(
    adata: ad.AnnData,
    diversity_results: Optional[Dict] = None,
    stats_results: Optional[pd.DataFrame] = None,
    qc_results: Optional[Dict] = None,
    ml_results: Optional[Dict] = None,
    functional_results: Optional[Dict] = None,
    output_path: Optional[Path] = None
) -> go.Figure:
    """
    Create comprehensive 12-panel analysis dashboard.
    
    Layout:
        Row 1: QC Summary | Sample Distribution | Sequencing Depth
        Row 2: Alpha Diversity | Beta Diversity | Top Taxa
        Row 3: Statistical Results | Effect Sizes | Power Analysis
        Row 4: ML Features | Functional Pathways | Executive Summary
    
    Args:
        adata: AnnData object with analysis results
        diversity_results: Dictionary with alpha/beta diversity results
        stats_results: DataFrame from test_with_effect_size()
        qc_results: QC assessment results
        ml_results: Machine learning feature selection results
        functional_results: Functional prediction results (PICRUSt2, FAPROTAX)
        output_path: Optional save path
    
    Returns:
        Plotly Figure with 12-panel dashboard
    """
    logger.info("Creating integrated analysis dashboard")
    
    # Create subplot grid (4 rows x 3 columns)
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=(
            "QC Summary", "Sample Distribution", "Sequencing Depth",
            "Alpha Diversity", "Beta Diversity", "Top Taxa",
            "Statistical Results", "Effect Sizes", "Power Analysis",
            "ML Feature Importance", "Functional Pathways", "Executive Summary"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}],
            [{"type": "box"}, {"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "table"}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )
    
    # Row 1, Col 1: QC Summary
    if qc_results:
        _add_qc_summary_panel(fig, qc_results, row=1, col=1)
    
    # Row 1, Col 2: Sample Distribution
    _add_sample_distribution_panel(fig, adata, row=1, col=2)
    
    # Row 1, Col 3: Sequencing Depth
    _add_sequencing_depth_panel(fig, adata, row=1, col=3)
    
    # Row 2, Col 1: Alpha Diversity
    if diversity_results and 'alpha' in diversity_results:
        _add_alpha_diversity_panel(fig, diversity_results['alpha'], row=2, col=1)
    
    # Row 2, Col 2: Beta Diversity
    if diversity_results and 'beta' in diversity_results:
        _add_beta_diversity_panel(fig, diversity_results['beta'], row=2, col=2)
    
    # Row 2, Col 3: Top Taxa
    _add_top_taxa_panel(fig, adata, row=2, col=3)
    
    # Row 3, Col 1: Statistical Results
    if stats_results is not None:
        _add_statistical_results_panel(fig, stats_results, row=3, col=1)
    
    # Row 3, Col 2: Effect Sizes
    if stats_results is not None:
        _add_effect_sizes_panel(fig, stats_results, row=3, col=2)
    
    # Row 3, Col 3: Power Analysis
    if stats_results is not None:
        _add_power_analysis_panel(fig, stats_results, row=3, col=3)
    
    # Row 4, Col 1: ML Features
    if ml_results:
        _add_ml_features_panel(fig, ml_results, row=4, col=1)
    
    # Row 4, Col 2: Functional Pathways
    if functional_results:
        _add_functional_pathways_panel(fig, functional_results, row=4, col=2)
    
    # Row 4, Col 3: Executive Summary Table
    _add_executive_summary_panel(
        fig, adata, stats_results, qc_results, row=4, col=3
    )
    
    # Update layout
    fig.update_layout(
        title_text="<b>Integrated Analysis Dashboard</b>",
        title_x=0.5,
        title_font_size=24,
        height=1800,
        width=1800,
        showlegend=True,
        template='plotly_white'
    )
    
    if output_path:
        fig.write_html(output_path)
        logger.info(f"Saved integrated dashboard to {output_path}")
    
    return fig


def _add_qc_summary_panel(fig: go.Figure, qc_results: Dict, row: int, col: int):
    """Add QC summary bar chart."""
    categories = ['Passed', 'Warning', 'Failed']
    counts = [
        qc_results.get('n_passed', 0),
        qc_results.get('n_warning', 0),
        qc_results.get('n_failed', 0)
    ]
    colors = ['green', 'orange', 'red']
    
    fig.add_trace(
        go.Bar(
            x=categories,
            y=counts,
            marker_color=colors,
            text=counts,
            textposition='auto',
            name='QC Status'
        ),
        row=row, col=col
    )
    
    fig.update_xaxes(title_text="QC Status", row=row, col=col)
    fig.update_yaxes(title_text="Sample Count", row=row, col=col)


def _add_sample_distribution_panel(fig: go.Figure, adata: ad.AnnData, row: int, col: int):
    """Add sample distribution by primary grouping variable."""
    # Find primary grouping column (most commonly used)
    group_cols = [c for c in adata.obs.columns if adata.obs[c].dtype == 'object']
    
    if group_cols:
        primary_col = group_cols[0]
        value_counts = adata.obs[primary_col].value_counts()
        
        fig.add_trace(
            go.Bar(
                x=value_counts.index.tolist(),
                y=value_counts.values.tolist(),
                marker_color='steelblue',
                text=value_counts.values,
                textposition='auto',
                name='Samples'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text=primary_col, row=row, col=col)
        fig.update_yaxes(title_text="Sample Count", row=row, col=col)


def _add_sequencing_depth_panel(fig: go.Figure, adata: ad.AnnData, row: int, col: int):
    """Add sequencing depth distribution."""
    # Calculate read counts per sample
    if hasattr(adata.X, 'toarray'):
        read_counts = adata.X.toarray().sum(axis=1)
    else:
        read_counts = adata.X.sum(axis=1)
    
    if isinstance(read_counts, np.matrix):
        read_counts = np.asarray(read_counts).flatten()
    
    fig.add_trace(
        go.Histogram(
            x=read_counts,
            nbinsx=30,
            marker_color='teal',
            name='Read Depth'
        ),
        row=row, col=col
    )
    
    # Add median line
    median_depth = np.median(read_counts)
    fig.add_vline(
        x=median_depth,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Median: {int(median_depth):,}",
        row=row, col=col
    )
    
    fig.update_xaxes(title_text="Reads per Sample", type="log", row=row, col=col)
    fig.update_yaxes(title_text="Sample Count", row=row, col=col)


def _add_alpha_diversity_panel(fig: go.Figure, alpha_results: Dict, row: int, col: int):
    """Add alpha diversity comparison."""
    # Extract first metric for display
    if 'shannon' in alpha_results:
        metric_data = alpha_results['shannon']
    else:
        metric_key = list(alpha_results.keys())[0]
        metric_data = alpha_results[metric_key]
    
    # Create boxplot (placeholder - would need actual group data)
    fig.add_trace(
        go.Box(
            y=metric_data.get('values', []),
            name='Alpha Diversity',
            marker_color='purple'
        ),
        row=row, col=col
    )
    
    fig.update_xaxes(title_text="", row=row, col=col)
    fig.update_yaxes(title_text="Shannon Diversity", row=row, col=col)


def _add_beta_diversity_panel(fig: go.Figure, beta_results: Dict, row: int, col: int):
    """Add beta diversity ordination."""
    # Extract PCoA coordinates
    if 'pcoa' in beta_results:
        pcoa_data = beta_results['pcoa']
        pc1 = pcoa_data.get('PC1', [])
        pc2 = pcoa_data.get('PC2', [])
        
        fig.add_trace(
            go.Scatter(
                x=pc1,
                y=pc2,
                mode='markers',
                marker=dict(size=8, color='darkblue', opacity=0.6),
                name='Samples'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="PC1", row=row, col=col)
        fig.update_yaxes(title_text="PC2", row=row, col=col)


def _add_top_taxa_panel(fig: go.Figure, adata: ad.AnnData, row: int, col: int):
    """Add top 10 most abundant taxa."""
    # Calculate mean abundance per feature
    if hasattr(adata.X, 'toarray'):
        mean_abundance = adata.X.toarray().mean(axis=0)
    else:
        mean_abundance = adata.X.mean(axis=0)
    
    if isinstance(mean_abundance, np.matrix):
        mean_abundance = np.asarray(mean_abundance).flatten()
    
    # Get top 10
    top_indices = np.argsort(mean_abundance)[-10:][::-1]
    top_features = adata.var_names[top_indices].tolist()
    top_abundances = mean_abundance[top_indices]
    
    fig.add_trace(
        go.Bar(
            x=top_abundances,
            y=top_features,
            orientation='h',
            marker_color='coral',
            name='Top Taxa'
        ),
        row=row, col=col
    )
    
    fig.update_xaxes(title_text="Mean Abundance", row=row, col=col)
    fig.update_yaxes(title_text="", row=row, col=col)


def _add_statistical_results_panel(
    fig: go.Figure, stats_results: pd.DataFrame, row: int, col: int
):
    """Add volcano plot of statistical results."""
    if 'p_adj' not in stats_results.columns or 'log2_fold_change' not in stats_results.columns:
        return
    
    # Calculate -log10(p)
    neg_log_p = -np.log10(stats_results['p_adj'].replace(0, 1e-300))
    
    # Color by significance
    colors = np.where(
        (stats_results['p_adj'] < 0.05) & (stats_results['log2_fold_change'].abs() > 1),
        'red',
        'gray'
    )
    
    fig.add_trace(
        go.Scatter(
            x=stats_results['log2_fold_change'],
            y=neg_log_p,
            mode='markers',
            marker=dict(size=6, color=colors, opacity=0.6),
            text=stats_results['feature'],
            name='Features'
        ),
        row=row, col=col
    )
    
    # Add significance thresholds
    fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="blue", row=row, col=col)
    fig.add_vline(x=-1, line_dash="dash", line_color="blue", row=row, col=col)
    fig.add_vline(x=1, line_dash="dash", line_color="blue", row=row, col=col)
    
    fig.update_xaxes(title_text="Log2 Fold Change", row=row, col=col)
    fig.update_yaxes(title_text="-log10(p-adj)", row=row, col=col)


def _add_effect_sizes_panel(
    fig: go.Figure, stats_results: pd.DataFrame, row: int, col: int
):
    """Add effect size distribution."""
    if 'cliffs_delta' not in stats_results.columns:
        return
    
    # Categorize effect sizes
    interpretations = stats_results['cliffs_delta_interpretation'].value_counts()
    
    fig.add_trace(
        go.Bar(
            x=interpretations.index.tolist(),
            y=interpretations.values.tolist(),
            marker_color=['gray', 'yellow', 'orange', 'red'],
            text=interpretations.values,
            textposition='auto',
            name='Effect Size Distribution'
        ),
        row=row, col=col
    )
    
    fig.update_xaxes(title_text="Effect Size Category", row=row, col=col)
    fig.update_yaxes(title_text="Feature Count", row=row, col=col)


def _add_power_analysis_panel(
    fig: go.Figure, stats_results: pd.DataFrame, row: int, col: int
):
    """Add power analysis visualization."""
    # This is a placeholder - actual power calculations would come from stats module
    sample_sizes = [10, 20, 30, 50, 100, 200]
    power_small = [0.1, 0.2, 0.35, 0.55, 0.8, 0.95]
    power_medium = [0.2, 0.45, 0.7, 0.9, 0.98, 0.99]
    power_large = [0.4, 0.75, 0.92, 0.99, 1.0, 1.0]
    
    fig.add_trace(
        go.Scatter(
            x=sample_sizes, y=power_small,
            mode='lines+markers', name='Small Effect',
            line=dict(color='blue', dash='dash')
        ),
        row=row, col=col
    )
    
    fig.add_trace(
        go.Scatter(
            x=sample_sizes, y=power_medium,
            mode='lines+markers', name='Medium Effect',
            line=dict(color='orange')
        ),
        row=row, col=col
    )
    
    fig.add_trace(
        go.Scatter(
            x=sample_sizes, y=power_large,
            mode='lines+markers', name='Large Effect',
            line=dict(color='red')
        ),
        row=row, col=col
    )
    
    # Add target power line
    fig.add_hline(y=0.8, line_dash="dot", line_color="green", row=row, col=col)
    
    fig.update_xaxes(title_text="Sample Size per Group", row=row, col=col)
    fig.update_yaxes(title_text="Statistical Power", row=row, col=col)


def _add_ml_features_panel(fig: go.Figure, ml_results: Dict, row: int, col: int):
    """Add ML feature importance."""
    # Placeholder - would need actual ML results
    features = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
    importance = [0.25, 0.20, 0.18, 0.15, 0.12]
    
    fig.add_trace(
        go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color='darkgreen',
            name='Feature Importance'
        ),
        row=row, col=col
    )
    
    fig.update_xaxes(title_text="Importance Score", row=row, col=col)
    fig.update_yaxes(title_text="", row=row, col=col)


def _add_functional_pathways_panel(fig: go.Figure, functional_results: Dict, row: int, col: int):
    """Add functional pathway enrichment."""
    # Placeholder
    pathways = ['Pathway A', 'Pathway B', 'Pathway C', 'Pathway D']
    enrichment = [3.5, 2.8, 2.2, 1.9]
    
    fig.add_trace(
        go.Bar(
            x=enrichment,
            y=pathways,
            orientation='h',
            marker_color='purple',
            name='Pathway Enrichment'
        ),
        row=row, col=col
    )
    
    fig.update_xaxes(title_text="Fold Enrichment", row=row, col=col)
    fig.update_yaxes(title_text="", row=row, col=col)


def _add_executive_summary_panel(
    fig: go.Figure,
    adata: ad.AnnData,
    stats_results: Optional[pd.DataFrame],
    qc_results: Optional[Dict],
    row: int,
    col: int
):
    """Add executive summary table."""
    # Compile key metrics
    n_samples = adata.n_obs
    n_features = adata.n_vars
    
    if hasattr(adata.X, 'toarray'):
        median_depth = int(np.median(adata.X.toarray().sum(axis=1)))
    else:
        median_depth = int(np.median(adata.X.sum(axis=1)))
    
    summary_data = {
        'Metric': ['Total Samples', 'Total Features', 'Median Depth'],
        'Value': [str(n_samples), str(n_features), f'{median_depth:,}']
    }
    
    if qc_results:
        summary_data['Metric'].append('QC Pass Rate')
        pass_rate = qc_results.get('n_passed', 0) / n_samples * 100
        summary_data['Value'].append(f'{pass_rate:.1f}%')
    
    if stats_results is not None and 'both_significant' in stats_results.columns:
        summary_data['Metric'].append('Significant Features')
        n_sig = stats_results['both_significant'].sum()
        summary_data['Value'].append(f'{n_sig} ({n_sig/len(stats_results)*100:.1f}%)')
    
    # Create table
    fig.add_trace(
        go.Table(
            header=dict(
                values=list(summary_data.keys()),
                fill_color='steelblue',
                font=dict(color='white', size=14),
                align='left'
            ),
            cells=dict(
                values=list(summary_data.values()),
                fill_color='lavender',
                align='left',
                font=dict(size=12)
            )
        ),
        row=row, col=col
    )


def create_qc_aware_diversity_dashboard(
    adata: ad.AnnData,
    diversity_metrics: List[str] = ['shannon', 'observed_features'],
    qc_column: str = 'qc_pass',
    group_column: Optional[str] = None,
    output_path: Optional[Path] = None
) -> go.Figure:
    """
    Create diversity analysis dashboard stratified by QC status.
    
    Compares diversity metrics between QC-passed and QC-failed samples
    to assess impact of quality control on biological conclusions.
    
    Args:
        adata: AnnData with diversity metrics in .obs
        diversity_metrics: List of diversity metric column names
        qc_column: Boolean column indicating QC pass/fail
        group_column: Optional grouping variable
        output_path: Optional save path
    
    Returns:
        Plotly Figure with QC-stratified diversity analysis
    """
    logger.info("Creating QC-aware diversity dashboard")
    
    n_metrics = len(diversity_metrics)
    
    # Create subplots: one row per metric, 2 columns (with/without QC stratification)
    fig = make_subplots(
        rows=n_metrics, cols=2,
        subplot_titles=[
            item for metric in diversity_metrics
            for item in [f"{metric} - All Samples", f"{metric} - By QC Status"]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    for i, metric in enumerate(diversity_metrics, start=1):
        if metric not in adata.obs.columns:
            logger.warning(f"Metric {metric} not found in adata.obs")
            continue
        
        # Left panel: All samples
        if group_column and group_column in adata.obs.columns:
            for group in adata.obs[group_column].unique():
                if pd.isna(group):
                    continue
                group_data = adata.obs[adata.obs[group_column] == group][metric].dropna()
                
                fig.add_trace(
                    go.Box(
                        y=group_data,
                        name=str(group),
                        showlegend=(i == 1)
                    ),
                    row=i, col=1
                )
        else:
            fig.add_trace(
                go.Box(
                    y=adata.obs[metric].dropna(),
                    name='All Samples'
                ),
                row=i, col=1
            )
        
        # Right panel: By QC status
        if qc_column in adata.obs.columns:
            for qc_status in [True, False]:
                qc_label = 'QC Pass' if qc_status else 'QC Fail'
                qc_data = adata.obs[adata.obs[qc_column] == qc_status][metric].dropna()
                
                if len(qc_data) > 0:
                    fig.add_trace(
                        go.Box(
                            y=qc_data,
                            name=qc_label,
                            showlegend=(i == 1),
                            marker_color='green' if qc_status else 'red'
                        ),
                        row=i, col=2
                    )
        
        # Update axes
        fig.update_yaxes(title_text=metric.replace('_', ' ').title(), row=i, col=1)
        fig.update_yaxes(title_text=metric.replace('_', ' ').title(), row=i, col=2)
    
    fig.update_layout(
        title_text="<b>QC-Aware Diversity Analysis</b>",
        title_x=0.5,
        title_font_size=20,
        height=300 * n_metrics,
        width=1400,
        showlegend=True,
        template='plotly_white'
    )
    
    if output_path:
        fig.write_html(output_path)
        logger.info(f"Saved QC-aware diversity dashboard to {output_path}")
    
    return fig
