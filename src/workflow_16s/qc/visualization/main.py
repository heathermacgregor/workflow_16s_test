"""
QC Visualization and Interpretation Module

Creates publication-ready visualizations that integrate QC results
with downstream analysis to show QC impact and data quality.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

try:
    import anndata as ad
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False

logger = logging.getLogger('workflow_16s')


def create_qc_impact_dashboard(
    adata_before: Optional[ad.AnnData],
    adata_after: ad.AnnData,
    qc_results: Dict,
    output_path: Union[str, Path]
) -> go.Figure:
    """
    Create comprehensive dashboard showing QC impact on analysis.
    
    Args:
        adata_before: AnnData before QC (optional)
        adata_after: AnnData after QC
        qc_results: Results from ComprehensiveQC
        output_path: Where to save HTML dashboard
        
    Returns:
        Plotly figure object
    """
    # Create 2x3 subplot grid
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '① Sample QC Flags Distribution',
            '② Contamination Detection',
            '③ Metadata Quality Improvement',
            '④ Alpha Diversity by QC Status',
            '⑤ PCA Colored by QC Flags',
            '⑥ Feature-Level QC Summary'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'scatter'}],
            [{'type': 'bar'}, {'type': 'box'}],
            [{'type': 'scatter'}, {'type': 'bar'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )
    
    # ===== Plot 1: QC Flags Distribution =====
    if 'qc_overall_flag' in adata_after.obs.columns:
        flag_counts = adata_after.obs['qc_overall_flag'].value_counts()
        
        colors = {'PASS': '#2ecc71', 'WARNING': '#f39c12', 'FAIL': '#e74c3c'}
        
        fig.add_trace(
            go.Bar(
                x=flag_counts.index,
                y=flag_counts.values,
                marker=dict(color=[colors.get(f, '#95a5a6') for f in flag_counts.index]),
                text=flag_counts.values,
                textposition='auto',
                name='QC Flags'
            ),
            row=1, col=1
        )
    
    # ===== Plot 2: Contamination Scores =====
    if 'is_contaminant' in adata_after.var.columns:
        if 'contamination_score' in adata_after.var.columns:
            contam_scores = adata_after.var['contamination_score']
        else:
            contam_scores = adata_after.var['is_contaminant'].astype(float)
        
        # Histogram of contamination scores
        fig.add_trace(
            go.Histogram(
                x=contam_scores,
                nbinsx=50,
                marker=dict(color='#e74c3c', opacity=0.7),
                name='Contamination Score',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add threshold line if available
        threshold = qc_results.get('contamination_threshold', 0.5)
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Threshold: {threshold}",
            row=1, col=2
        )
    
    # ===== Plot 3: Metadata Quality =====
    if 'report' in qc_results.get('metadata', {}):
        report = qc_results['metadata']['report']
        
        if not report.empty and 'level' in report.columns:
            level_counts = report['level'].value_counts()
            
            fig.add_trace(
                go.Bar(
                    x=level_counts.index,
                    y=level_counts.values,
                    marker=dict(color=['#2ecc71', '#f39c12', '#e74c3c']),
                    text=level_counts.values,
                    textposition='auto',
                    name='Validation Issues'
                ),
                row=2, col=1
            )
    
    # ===== Plot 4: Alpha Diversity by QC Status =====
    if 'shannon' in adata_after.obs.columns and 'qc_overall_flag' in adata_after.obs.columns:
        for flag in ['PASS', 'WARNING', 'FAIL']:
            mask = adata_after.obs['qc_overall_flag'] == flag
            if mask.sum() > 0:
                values = adata_after.obs.loc[mask, 'shannon'].dropna()
                
                fig.add_trace(
                    go.Box(
                        y=values,
                        name=flag,
                        marker=dict(color=colors.get(flag, '#95a5a6')),
                        boxmean='sd'
                    ),
                    row=2, col=2
                )
    
    # ===== Plot 5: PCA colored by QC =====
    if 'X_pca' in adata_after.obsm.keys() and 'qc_overall_flag' in adata_after.obs.columns:
        pca_coords = adata_after.obsm['X_pca'][:, :2]
        qc_flags = adata_after.obs['qc_overall_flag']
        
        for flag in ['PASS', 'WARNING', 'FAIL']:
            mask = qc_flags == flag
            if mask.sum() > 0:
                fig.add_trace(
                    go.Scatter(
                        x=pca_coords[mask, 0],
                        y=pca_coords[mask, 1],
                        mode='markers',
                        name=flag,
                        marker=dict(
                            size=8,
                            color=colors.get(flag, '#95a5a6'),
                            opacity=0.7,
                            line=dict(width=0.5, color='white')
                        )
                    ),
                    row=3, col=1
                )
    
    # ===== Plot 6: Feature Summary =====
    feature_summary = []
    
    total_features = adata_after.n_vars
    feature_summary.append(('Total Features', total_features))
    
    if 'is_contaminant' in adata_after.var.columns:
        n_contam = adata_after.var['is_contaminant'].sum()
        feature_summary.append(('Contaminants', n_contam))
        feature_summary.append(('Clean Features', total_features - n_contam))
    
    if feature_summary:
        labels, values = zip(*feature_summary)
        
        fig.add_trace(
            go.Bar(
                x=list(labels),
                y=list(values),
                marker=dict(color=['#3498db', '#e74c3c', '#2ecc71']),
                text=list(values),
                textposition='auto'
            ),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>QC Impact Dashboard</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        height=1200,
        width=1400,
        showlegend=True,
        template='plotly_white'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="QC Flag", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    
    fig.update_xaxes(title_text="Contamination Score", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    fig.update_xaxes(title_text="Validation Level", row=2, col=1)
    fig.update_yaxes(title_text="Issues Found", row=2, col=1)
    
    fig.update_yaxes(title_text="Shannon Diversity", row=2, col=2)
    
    fig.update_xaxes(title_text="PC1", row=3, col=1)
    fig.update_yaxes(title_text="PC2", row=3, col=1)
    
    # Save
    output_path = Path(output_path)
    fig.write_html(str(output_path))
    logger.info(f"QC dashboard saved to: {output_path}")
    
    return fig


def create_qc_interpretation_report(
    adata: ad.AnnData,
    qc_results: Dict,
    output_path: Union[str, Path]
) -> str:
    """
    Generate automated interpretation of QC results.
    
    Args:
        adata: AnnData with QC annotations
        qc_results: Results from ComprehensiveQC
        output_path: Where to save report
        
    Returns:
        Markdown-formatted interpretation text
    """
    lines = [
        "# QC Results Interpretation Report",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Executive Summary",
        ""
    ]
    
    # Sample-level summary
    if 'qc_overall_flag' in adata.obs.columns:
        flag_counts = adata.obs['qc_overall_flag'].value_counts()
        total = len(adata)
        pass_pct = (flag_counts.get('PASS', 0) / total) * 100
        warn_pct = (flag_counts.get('WARNING', 0) / total) * 100
        fail_pct = (flag_counts.get('FAIL', 0) / total) * 100
        
        lines.extend([
            f"**Total Samples:** {total}",
            f"- ✅ PASS: {flag_counts.get('PASS', 0)} ({pass_pct:.1f}%)",
            f"- ⚠️  WARNING: {flag_counts.get('WARNING', 0)} ({warn_pct:.1f}%)",
            f"- ❌ FAIL: {flag_counts.get('FAIL', 0)} ({fail_pct:.1f}%)",
            ""
        ])
        
        # Interpretation
        if pass_pct >= 90:
            lines.append("**Interpretation:** ✅ Excellent data quality. Most samples passed QC.")
        elif pass_pct >= 75:
            lines.append("**Interpretation:** ✓ Good data quality. Some samples flagged for review.")
        elif pass_pct >= 50:
            lines.append("**Interpretation:** ⚠️ Moderate data quality. Significant QC issues detected.")
        else:
            lines.append("**Interpretation:** ❌ Poor data quality. Major QC concerns - review experimental protocol.")
        
        lines.append("")
    
    # Feature-level summary
    if 'is_contaminant' in adata.var.columns:
        n_contam = adata.var['is_contaminant'].sum()
        total_features = adata.n_vars
        contam_pct = (n_contam / total_features) * 100
        
        lines.extend([
            "## Contamination Analysis",
            "",
            f"**Total Features:** {total_features}",
            f"- 🦠 Potential Contaminants: {n_contam} ({contam_pct:.1f}%)",
            f"- ✓ Clean Features: {total_features - n_contam} ({100-contam_pct:.1f}%)",
            ""
        ])
        
        if contam_pct < 5:
            lines.append("**Interpretation:** ✅ Low contamination levels. Data appears clean.")
        elif contam_pct < 15:
            lines.append("**Interpretation:** ⚠️ Moderate contamination detected. Review flagged features.")
        else:
            lines.append("**Interpretation:** ❌ High contamination levels. Consider re-sequencing or stricter filtering.")
        
        lines.append("")
    
    # Metadata quality
    if 'metadata' in qc_results and 'n_removed_columns' in qc_results['metadata']:
        n_removed = qc_results['metadata']['n_removed_columns']
        
        lines.extend([
            "## Metadata Quality",
            "",
            f"**Redundant Columns Removed:** {n_removed}",
            ""
        ])
        
        if n_removed == 0:
            lines.append("**Interpretation:** ✅ Metadata is clean with no redundant information.")
        elif n_removed < 5:
            lines.append("**Interpretation:** ✓ Minor metadata cleanup performed.")
        else:
            lines.append(f"**Interpretation:** ⚠️ Significant redundancy removed ({n_removed} columns). Review metadata collection protocol.")
        
        lines.append("")
    
    # ENVO categorization
    if 'env_category_type' in adata.obs.columns:
        categories = adata.obs['env_category_type'].value_counts()
        
        lines.extend([
            "## Environmental Categorization",
            "",
            "**Sample Distribution by Environment:**"
        ])
        
        for cat, count in categories.items():
            pct = (count / len(adata)) * 100
            lines.append(f"- {cat}: {count} ({pct:.1f}%)")
        
        lines.extend([
            "",
            f"**Interpretation:** Samples span {len(categories)} distinct environment types. ",
            "This diversity enables robust comparative analyses.",
            ""
        ])
    
    # Recommendations
    lines.extend([
        "## Recommendations",
        ""
    ])
    
    if 'qc_overall_flag' in adata.obs.columns:
        fail_count = (adata.obs['qc_overall_flag'] == 'FAIL').sum()
        if fail_count > 0:
            lines.append(f"1. **Remove Failed Samples:** Consider excluding {fail_count} samples marked as FAIL from downstream analysis.")
    
    if 'is_contaminant' in adata.var.columns:
        contam_count = adata.var['is_contaminant'].sum()
        if contam_count > 0:
            lines.append(f"2. **Filter Contaminants:** Remove {contam_count} flagged contaminant features before diversity analysis.")
    
    if 'env_category_type' in adata.obs.columns:
        lines.append("3. **Stratified Analysis:** Perform separate analyses for each environment type to account for ecological differences.")
    
    lines.extend([
        "",
        "## Next Steps",
        "",
        "1. Review flagged samples/features in detail",
        "2. Apply recommended filtering",
        "3. Proceed with downstream analysis on cleaned data",
        "4. Document QC decisions for reproducibility",
        ""
    ])
    
    # Write report
    report_text = '\n'.join(lines)
    output_path = Path(output_path)
    output_path.write_text(report_text)
    logger.info(f"QC interpretation report saved to: {output_path}")
    
    return report_text


def plot_qc_metrics_over_sequencing_depth(
    adata: ad.AnnData,
    output_path: Union[str, Path]
) -> go.Figure:
    """
    Plot how QC metrics relate to sequencing depth.
    
    Shows whether QC failures are associated with low coverage.
    
    Args:
        adata: AnnData with QC annotations
        output_path: Where to save plot
        
    Returns:
        Plotly figure
    """
    # Calculate read depth
    if hasattr(adata.X, 'toarray'):
        read_depth = np.array(adata.X.sum(axis=1)).flatten()
    else:
        read_depth = adata.X.sum(axis=1)
    
    adata.obs['read_depth'] = read_depth
    
    fig = go.Figure()
    
    if 'qc_overall_flag' in adata.obs.columns:
        colors = {'PASS': '#2ecc71', 'WARNING': '#f39c12', 'FAIL': '#e74c3c'}
        
        for flag in ['PASS', 'WARNING', 'FAIL']:
            mask = adata.obs['qc_overall_flag'] == flag
            if mask.sum() > 0:
                depths = adata.obs.loc[mask, 'read_depth']
                
                fig.add_trace(go.Histogram(
                    x=depths,
                    name=flag,
                    marker=dict(color=colors[flag], opacity=0.7),
                    nbinsx=50
                ))
    
    fig.update_layout(
        title="Sequencing Depth Distribution by QC Status",
        xaxis_title="Total Reads per Sample (log10 scale)",
        xaxis_type="log",
        yaxis_title="Number of Samples",
        barmode='overlay',
        template='plotly_white',
        height=500,
        width=800
    )
    
    # Save
    output_path = Path(output_path)
    fig.write_html(str(output_path))
    logger.info(f"QC depth plot saved to: {output_path}")
    
    return fig


def create_sample_qc_heatmap(
    adata: ad.AnnData,
    qc_columns: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> go.Figure:
    """
    Create heatmap showing QC status across multiple metrics.
    
    Args:
        adata: AnnData with QC columns in obs
        qc_columns: List of QC column names to include
        output_path: Where to save
        
    Returns:
        Plotly figure
    """
    if qc_columns is None:
        # Auto-detect QC columns
        qc_columns = [col for col in adata.obs.columns 
                     if 'qc_' in col.lower() or col in ['is_outlier', 'is_human_contamination']]
    
    if not qc_columns:
        logger.warning("No QC columns found for heatmap")
        return None
    
    # Create binary matrix
    qc_matrix = adata.obs[qc_columns].copy()
    
    # Convert to numeric (1 = flagged, 0 = passed)
    for col in qc_columns:
        if qc_matrix[col].dtype == 'object':
            # Map PASS/FAIL to 0/1
            qc_matrix[col] = qc_matrix[col].map({
                'PASS': 0, 'WARNING': 0.5, 'FAIL': 1
            }).fillna(0)
        elif qc_matrix[col].dtype == 'bool':
            qc_matrix[col] = qc_matrix[col].astype(int)
    
    # Sort samples by number of failures
    qc_matrix['n_flags'] = qc_matrix.sum(axis=1)
    qc_matrix = qc_matrix.sort_values('n_flags', ascending=False)
    qc_matrix = qc_matrix.drop('n_flags', axis=1)
    
    fig = go.Figure(data=go.Heatmap(
        z=qc_matrix.T.values,
        x=qc_matrix.index,
        y=qc_matrix.columns,
        colorscale=[[0, '#2ecc71'], [0.5, '#f39c12'], [1, '#e74c3c']],
        colorbar=dict(title="QC Flag"),
        hoverongaps=False,
        hovertemplate='Sample: %{x}<br>Metric: %{y}<br>Status: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Sample-Level QC Heatmap",
        xaxis_title="Samples (sorted by QC flags)",
        yaxis_title="QC Metrics",
        height=400 + len(qc_columns) * 20,
        width=1000,
        template='plotly_white'
    )
    
    if output_path:
        output_path = Path(output_path)
        fig.write_html(str(output_path))
        logger.info(f"QC heatmap saved to: {output_path}")
    
    return fig
