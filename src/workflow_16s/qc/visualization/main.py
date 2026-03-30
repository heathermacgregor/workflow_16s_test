"""
QC Visualization and Interpretation Module

Creates publication-ready visualizations that integrate QC results
with downstream analysis to show QC impact and data quality.

Uses a custom 'heather' Plotly template for consistent, professional aesthetic.
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
import plotly.io as pio

try:
    import anndata as ad
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False

logger = logging.getLogger('workflow_16s')


# ==================================================================================== #
# CUSTOM HEATHER TEMPLATE - Soft, professional aesthetic for QC visualizations
# ==================================================================================== #

def register_heather_template():
    """Register the custom 'heather' Plotly template."""
    heather = go.layout.Template(
        layout=go.Layout(
            # Color scheme: Soft purples, greys, and earth tones
            colorway=[
                '#8B7BA4',  # Soft purple (primary)
                '#A89CC4',  # Light purple
                '#C4B8D1',  # Pale heather
                '#6B6B7A',  # Steel grey
                '#9A8E99',  # Taupe
                '#7A9E8F',  # Soft sage
                '#B8A89C',  # Warm grey
            ],
            # Background and font
            paper_bgcolor='#FAFAF9',  # Off-white
            plot_bgcolor='#FFFFFF',   # White plotting area
            font=dict(family="Arial, sans-serif", size=12, color='#3D3D3D'),
            title=dict(font=dict(size=18, color='#2D2D2D', family='Arial, sans-serif')),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='#E8E8E6',
                showline=True,
                linewidth=1,
                linecolor='#9A9A99',
                title=dict(font=dict(size=13, color='#3D3D3D'))
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='#E8E8E6',
                showline=True,
                linewidth=1,
                linecolor='#9A9A99',
                title=dict(font=dict(size=13, color='#3D3D3D'))
            ),
            legend=dict(
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='#9A9A99',
                borderwidth=1,
                font=dict(size=11, color='#3D3D3D')
            ),
            hoverlabel=dict(
                bgcolor='#FAFAF9',
                font=dict(size=12, family='Arial, sans-serif', color='#3D3D3D'),
                bordercolor='#8B7BA4'
            )
        )
    )
    pio.templates['heather'] = heather
    return heather


# Register template on module load
try:
    register_heather_template()
except Exception as e:
    logger.warning(f"Failed to register heather template: {e}")


def create_qc_impact_dashboard(
    adata_before: Optional[ad.AnnData],
    adata_after: ad.AnnData,
    qc_results: Dict,
    output_path: Union[str, Path]
) -> go.Figure:
    """
    Create comprehensive dashboard showing QC impact on analysis.
    
    Uses the 'heather' template for professional, publication-ready aesthetic.
    
    Args:
        adata_before: AnnData before QC (optional)
        adata_after: AnnData after QC
        qc_results: Results from ComprehensiveQC
        output_path: Where to save HTML dashboard
        
    Returns:
        Plotly figure object
    """
    # Define QC status colors (heather-friendly palette)
    qc_colors = {
        'PASS': '#7A9E8F',      # Soft sage (pass)
        'WARNING': '#C9A876',   # Warm amber (warning)
        'FAIL': '#A85A5A'       # Soft red (failure)
    }
    
    # Create 3x2 subplot grid
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '<b>① QC Flags Distribution</b>',
            '<b>② Contamination Detection</b>',
            '<b>③ Metadata Quality</b>',
            '<b>④ Alpha Diversity by QC</b>',
            '<b>⑤ PCA Colored by QC</b>',
            '<b>⑥ Feature Summary</b>'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'scatter'}],
            [{'type': 'bar'}, {'type': 'box'}],
            [{'type': 'scatter'}, {'type': 'bar'}]
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.14
    )
    
    # ===== Plot 1: QC Flags Distribution =====
    if 'qc_overall_flag' in adata_after.obs.columns:
        flag_counts = adata_after.obs['qc_overall_flag'].value_counts()
        
        fig.add_trace(
            go.Bar(
                x=flag_counts.index,
                y=flag_counts.values,
                marker=dict(
                    color=[qc_colors.get(f, '#9A9A99') for f in flag_counts.index],
                    line=dict(color='#6B6B7A', width=1.5)
                ),
                text=flag_counts.values,
                textposition='outside',
                name='QC Status',
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # ===== Plot 2: Contamination Scores =====
    if 'is_contaminant' in adata_after.var.columns:
        if 'contamination_score' in adata_after.var.columns:
            contam_scores = adata_after.var['contamination_score']
        else:
            contam_scores = adata_after.var['is_contaminant'].astype(float)
        
        fig.add_trace(
            go.Histogram(
                x=contam_scores,
                nbinsx=50,
                marker=dict(color='#8B7BA4', line=dict(color='#6B6B7A', width=0.5)),
                name='Contamination Score',
                hovertemplate='Score: %{x:.2f}<br>Frequency: %{y}<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add threshold line
        threshold = qc_results.get('contamination_threshold', 0.5)
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color='#A85A5A',
            line_width=2,
            annotation_text=f"<b>Threshold: {threshold:.2f}</b>",
            annotation_position="top right",
            row=1, col=2
        )
    
    # ===== Plot 3: Metadata Quality =====
    if 'report' in qc_results.get('metadata', {}):
        report = qc_results['metadata']['report']
        
        if not report.empty and 'level' in report.columns:
            level_counts = report['level'].value_counts()
            qual_colors = {'info': '#7A9E8F', 'warning': '#C9A876', 'error': '#A85A5A'}
            
            fig.add_trace(
                go.Bar(
                    x=level_counts.index,
                    y=level_counts.values,
                    marker=dict(
                        color=[qual_colors.get(str(idx).lower(), '#9A9A99') for idx in level_counts.index],
                        line=dict(color='#6B6B7A', width=1.5)
                    ),
                    text=level_counts.values,
                    textposition='outside',
                    name='Quality Issues',
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',
                    showlegend=False
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
                        marker=dict(color=qc_colors.get(flag, '#9A9A99')),
                        boxmean='sd',
                        hovertemplate='<b>%{fullData.name}</b><br>Shannon: %{y:.2f}<extra></extra>'
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
                            size=7,
                            color=qc_colors.get(flag, '#9A9A99'),
                            opacity=0.75,
                            line=dict(width=0.5, color='white')
                        ),
                        hovertemplate='<b>%{fullData.name}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
                    ),
                    row=3, col=1
                )
    
    # ===== Plot 6: Feature Summary =====
    feature_summary = []
    total_features = adata_after.n_vars
    feature_summary.append(('Total', total_features))
    
    if 'is_contaminant' in adata_after.var.columns:
        n_contam = adata_after.var['is_contaminant'].sum()
        feature_summary.append(('Contaminants', n_contam))
        feature_summary.append(('Clean', total_features - n_contam))
    
    if feature_summary:
        labels, values = zip(*feature_summary)
        feat_colors = ['#8B7BA4', '#A85A5A', '#7A9E8F']
        
        fig.add_trace(
            go.Bar(
                x=list(labels),
                y=list(values),
                marker=dict(
                    color=feat_colors[:len(labels)],
                    line=dict(color='#6B6B7A', width=1.5)
                ),
                text=list(values),
                textposition='outside',
                name='Features',
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',
                showlegend=False
            ),
            row=3, col=2
        )
    
    # Update layout with heather template
    fig.update_layout(
        title=dict(
            text="<b>QC Impact Dashboard</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=22, color='#2D2D2D', family='Arial, sans-serif')
        ),
        height=1300,
        width=1500,
        showlegend=True,
        template='heather',
        hovermode='closest',
        margin=dict(l=80, r=80, t=120, b=80)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="<b>QC Flag</b>", row=1, col=1)
    fig.update_yaxes(title_text="<b>Count</b>", row=1, col=1)
    
    fig.update_xaxes(title_text="<b>Contamination Score</b>", row=1, col=2)
    fig.update_yaxes(title_text="<b>Frequency</b>", row=1, col=2)
    
    fig.update_xaxes(title_text="<b>Quality Level</b>", row=2, col=1)
    fig.update_yaxes(title_text="<b>Count</b>", row=2, col=1)
    
    fig.update_yaxes(title_text="<b>Shannon Index</b>", row=2, col=2)
    
    fig.update_xaxes(title_text="<b>PC1</b>", row=3, col=1)
    fig.update_yaxes(title_text="<b>PC2</b>", row=3, col=1)
    
    fig.update_xaxes(title_text="<b>Feature Category</b>", row=3, col=2)
    fig.update_yaxes(title_text="<b>Count</b>", row=3, col=2)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    logger.info(f"✅ QC dashboard saved to: {output_path}")
    
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
    Uses the 'heather' template for consistent styling.
    
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
    
    # QC status colors (heather palette)
    qc_colors = {
        'PASS': '#7A9E8F',
        'WARNING': '#C9A876',
        'FAIL': '#A85A5A'
    }
    
    if 'qc_overall_flag' in adata.obs.columns:
        for flag in ['PASS', 'WARNING', 'FAIL']:
            mask = adata.obs['qc_overall_flag'] == flag
            if mask.sum() > 0:
                depths = adata.obs.loc[mask, 'read_depth']
                
                fig.add_trace(go.Histogram(
                    x=depths,
                    name=flag,
                    marker=dict(
                        color=qc_colors[flag],
                        opacity=0.7,
                        line=dict(color='#6B6B7A', width=0.5)
                    ),
                    nbinsx=50,
                    hovertemplate='<b>%{fullData.name}</b><br>Reads: %{x:.0f}<br>Count: %{y}<extra></extra>'
                ))
    
    fig.update_layout(
        title=dict(
            text="<b>Sequencing Depth Distribution by QC Status</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#2D2D2D')
        ),
        xaxis_title="<b>Total Reads per Sample</b>",
        xaxis_type="log",
        yaxis_title="<b>Number of Samples</b>",
        barmode='overlay',
        template='heather',
        height=600,
        width=900,
        hovermode='x unified',
        margin=dict(l=80, r=60, t=100, b=80)
    )
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    logger.info(f"✅ QC depth plot saved to: {output_path}")
    
    return fig


def create_sample_qc_heatmap(
    adata: ad.AnnData,
    qc_columns: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> go.Figure:
    """
    Create heatmap showing QC status across multiple metrics.
    
    Uses the 'heather' template for professional visualization.
    
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
    
    # Create heatmap with heather color scale
    fig = go.Figure(data=go.Heatmap(
        z=qc_matrix.T.values,
        x=qc_matrix.index,
        y=qc_matrix.columns,
        colorscale=[
            [0.0, '#7A9E8F'],    # Pass: soft sage
            [0.5, '#C9A876'],    # Warning: warm amber
            [1.0, '#A85A5A']     # Fail: soft red
        ],
        colorbar=dict(
            title="<b>QC Status</b>",
            thickness=15,
            len=0.7,
            x=1.02,
            tickvals=[0, 0.5, 1.0],
            ticktext=['Pass', 'Warning', 'Fail']
        ),
        hoverongaps=False,
        hovertemplate='<b>Sample:</b> %{x}<br><b>Metric:</b> %{y}<br><b>Status:</b> %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Sample-Level QC Heatmap</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#2D2D2D')
        ),
        xaxis_title="<b>Samples (sorted by failure count)</b>",
        yaxis_title="<b>QC Metrics</b>",
        height=400 + len(qc_columns) * 25,
        width=1200,
        template='heather',
        hovermode='closest',
        margin=dict(l=120, r=100, t=100, b=100)
    )
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ QC heatmap saved to: {output_path}")
    
    return fig
