"""
Quality Control Visualization Suite

Orchestrates comprehensive QC and sample metadata visualizations
for the downstream workflow. Generates publication-ready plots showing:

1. Sample QC Status (flags, contamination, outliers)
2. Sequencing Depth Analysis (depth vs QC status)
3. Sample Metadata Distributions (categorical & numeric)
4. Metadata Correlation Heatmap
5. Geographic Distribution (if coordinates available)
6. Data Quality Summary Table
7. Sample Composition Overview (taxa distribution)

Integration Points:
- Called after preprocessing, before analysis
- Generates both HTML and PNG outputs
- Uses ReportOrchestrator for consistent reporting
- Respects heather template for consistent aesthetics

Example:
    >>> from workflow_16s.downstream.visualization.quality_control_suite import run_qc_suite
    >>> run_qc_suite(workflow)
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import anndata as ad
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from workflow_16s.qc.visualization.main import (
    create_qc_impact_dashboard,
    plot_qc_metrics_over_sequencing_depth,
    create_sample_qc_heatmap
)
from workflow_16s.downstream.visualization.sample_metadata import (
    plot_sample_distribution,
    plot_metadata_heatmap,
    plot_metadata_summary_table,
    create_geographic_map
)

logger = logging.getLogger('workflow_16s')


def create_taxa_abundance_overview(
    adata: ad.AnnData,
    top_n: int = 15,
    output_path: Optional[Path] = None
) -> go.Figure:
    """
    Create overview of top taxa abundance distribution across samples.
    
    Shows which taxa dominate the community structure.
    
    Args:
        adata: AnnData object with taxonomy in var metadata
        top_n: Number of top taxa to display
        output_path: Optional save path for HTML
        
    Returns:
        Plotly figure
    """
    # Calculate mean abundance per feature
    abundance = pd.Series(
        np.asarray(adata.X.mean(axis=0)).flatten(),
        index=adata.var_names
    ).sort_values(ascending=False).head(top_n)
    
    # Get taxonomy names
    if 'Genus' in adata.var.columns:
        taxa_names = adata.var.loc[abundance.index, 'Genus'].values
        taxa_names = [str(t) if pd.notna(t) else f"OTU_{i}" for i, t in enumerate(taxa_names)]
    else:
        taxa_names = [str(name) for name in abundance.index[:top_n]]
    
    fig = go.Figure(data=[go.Bar(
        x=taxa_names,
        y=abundance.values,
        marker=dict(
            color='#8B7BA4',
            line=dict(color='#6B6B7A', width=1.5)
        ),
        hovertemplate='<b>%{x}</b><br>Mean Abundance: %{y:.4f}<extra></extra>',
        showlegend=False
    )])
    
    fig.update_layout(
        title=dict(
            text="<b>Top Taxa Abundance Overview</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#2D2D2D')
        ),
        xaxis_title="<b>Taxonomy</b>",
        yaxis_title="<b>Mean Abundance (CPM)</b>",
        height=500,
        width=1200,
        template='heather',
        hovermode='x unified',
        margin=dict(l=80, r=80, t=100, b=120),
        xaxis_tickangle=-45
    )
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Taxa abundance overview saved: {output_path}")
    
    return fig


def create_feature_sparsity_plot(
    adata: ad.AnnData,
    output_path: Optional[Path] = None
) -> go.Figure:
    """
    Create visualization of feature sparsity distribution.
    
    Shows prevalence of features (how many samples each feature appears in).
    
    Args:
        adata: AnnData object
        output_path: Optional save path for HTML
        
    Returns:
        Plotly figure
    """
    # Calculate prevalence (% samples feature present in)
    if hasattr(adata.X, 'toarray'):
        prevalence = (adata.X > 0).sum(axis=0) / adata.n_obs * 100
    else:
        prevalence = (adata.X > 0).sum(axis=0) / adata.n_obs * 100
    prevalence = np.asarray(prevalence).flatten()
    
    fig = go.Figure()
    
    # Histogram of prevalence
    fig.add_trace(go.Histogram(
        x=prevalence,
        nbinsx=50,
        marker=dict(
            color='#A89CC4',
            line=dict(color='#6B6B7A', width=0.5)
        ),
        hovertemplate='Prevalence: %{x:.1f}%<br>Count: %{y}<extra></extra>',
        name='Feature Prevalence'
    ))
    
    # Add median line
    median_prev = np.median(prevalence)
    fig.add_vline(
        x=median_prev,
        line_dash="dash",
        line_color="#C9A876",
        annotation_text=f"Median: {median_prev:.1f}%",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=dict(
            text="<b>Feature Sparsity Distribution</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#2D2D2D')
        ),
        xaxis_title="<b>Prevalence (% Samples Features Present)</b>",
        yaxis_title="<b>Number of Features</b>",
        height=500,
        width=1200,
        template='heather',
        hovermode='x',
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Feature sparsity plot saved: {output_path}")
    
    return fig


def create_sample_sequencing_depth_histogram(
    adata: ad.AnnData,
    output_path: Optional[Path] = None
) -> go.Figure:
    """
    Create histogram of sequencing depth distribution.
    
    Shows read depth distribution across samples with statistics.
    
    Args:
        adata: AnnData object
        output_path: Optional save path for HTML
        
    Returns:
        Plotly figure
    """
    # Calculate read depths
    if hasattr(adata.X, 'sum'):
        read_depths = np.asarray(adata.X.sum(axis=1)).flatten()
    else:
        read_depths = np.asarray(adata.X.sum(axis=1)).flatten()
    
    # Use existing read_depth column if available
    if 'read_depth' in adata.obs.columns:
        read_depths = adata.obs['read_depth'].values
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=read_depths,
        nbinsx=50,
        marker=dict(
            color='#7A9E8F',
            line=dict(color='#6B6B7A', width=0.5)
        ),
        hovertemplate='Depth: %{x:.0f}<br>Count: %{y}<extra></extra>',
        name='Samples'
    ))
    
    # Add statistical lines
    mean_depth = np.mean(read_depths)
    median_depth = np.median(read_depths)
    
    fig.add_vline(x=mean_depth, line_dash="dash", line_color="#C9A876",
                  annotation_text=f"Mean: {mean_depth:.0f}")
    fig.add_vline(x=median_depth, line_dash="dot", line_color="#A85A5A",
                  annotation_text=f"Median: {median_depth:.0f}")
    
    fig.update_layout(
        title=dict(
            text="<b>Sequencing Depth Distribution</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#2D2D2D')
        ),
        xaxis_title="<b>Read Depth (counts per sample)</b>",
        yaxis_title="<b>Number of Samples</b>",
        height=500,
        width=1200,
        template='heather',
        hovermode='x',
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"✅ Sequencing depth histogram saved: {output_path}")
    
    return fig


def run_qc_suite(workflow: Any) -> bool:
    """
    Execute comprehensive QC and sample metadata visualization suite.
    
    Generates multiple plots covering data quality, sample characteristics,
    and metadata distributions. Saves both HTML and PNG versions.
    
    Plots Generated:
    ================
    1. QC Impact Dashboard (6-panel: flags, contamination, metadata quality, alpha, PCA, features)
    2. QC Depth Plot (sequencing depth vs QC status)
    3. QC Heatmap (sample-level QC metrics)
    4. Metadata Distributions (categorical & numeric)
    5. Metadata Correlation Heatmap (numeric metadata correlations)
    6. Geographic Map (lat/lon distribution, if available)
    7. Data Quality Summary Table (descriptive statistics)
    8. Taxa Abundance Overview (top taxa by abundance)
    9. Feature Sparsity Distribution (prevalence across samples)
    10. Sequencing Depth Histogram (depth distribution)
    
    Args:
        workflow: DownstreamWorkflow instance with adata and output_dir
        
    Returns:
        True if successful, False if disabled or failed
    """
    if workflow.adata is None:
        logger.warning("Cannot run QC suite: AnnData object is None")
        return False
    
    # Check if QC visualizations are enabled in config
    viz_cfg = getattr(workflow.config, 'visualization', None)
    if viz_cfg and not getattr(viz_cfg, 'generate_qc_plots', True):
        logger.info("QC visualizations disabled in config")
        return False
    
    logger.info("🎨 Starting Comprehensive QC & Sample Metadata Visualization Suite...")
    
    # Create output directory
    qc_output_dir = workflow.output_dir / 'qc_visualizations'
    qc_output_dir.mkdir(parents=True, exist_ok=True)
    
    adata = workflow.adata
    results = {'generated': [], 'failed': []}
    
    try:
        # ===== PLOT 1: QC Impact Dashboard =====
        logger.info("📊 Generating QC Impact Dashboard...")
        try:
            fig = create_qc_impact_dashboard(
                adata,
                output_path=str(qc_output_dir / 'qc_impact_dashboard.html')
            )
            results['generated'].append('qc_impact_dashboard')
            # Also save PNG
            try:
                fig.write_image(
                    str(qc_output_dir / 'qc_impact_dashboard.png'),
                    width=1400, height=900
                )
                logger.info("✅ QC Impact Dashboard (HTML + PNG)")
            except Exception as e:
                logger.warning(f"Could not save PNG: {e}")
                logger.info("✅ QC Impact Dashboard (HTML only)")
        except Exception as e:
            logger.error(f"Failed to generate QC Impact Dashboard: {e}")
            results['failed'].append(('qc_impact_dashboard', str(e)))
        
        # ===== PLOT 2: QC Depth Plot =====
        logger.info("📊 Generating QC Depth Plot...")
        try:
            fig = plot_qc_metrics_over_sequencing_depth(
                adata,
                output_path=str(qc_output_dir / 'qc_depth_plot.html')
            )
            results['generated'].append('qc_depth_plot')
            try:
                fig.write_image(
                    str(qc_output_dir / 'qc_depth_plot.png'),
                    width=1200, height=600
                )
            except:
                pass
            logger.info("✅ QC Depth Plot")
        except Exception as e:
            logger.error(f"Failed to generate QC Depth Plot: {e}")
            results['failed'].append(('qc_depth_plot', str(e)))
        
        # ===== PLOT 3: QC Heatmap =====
        logger.info("📊 Generating QC Heatmap...")
        try:
            fig = create_sample_qc_heatmap(
                adata,
                output_path=str(qc_output_dir / 'qc_heatmap.html')
            )
            results['generated'].append('qc_heatmap')
            try:
                fig.write_image(
                    str(qc_output_dir / 'qc_heatmap.png'),
                    width=1200, height=700
                )
            except:
                pass
            logger.info("✅ QC Heatmap")
        except Exception as e:
            logger.error(f"Failed to generate QC Heatmap: {e}")
            results['failed'].append(('qc_heatmap', str(e)))
        
        # ===== PLOT 4: Metadata Distributions =====
        logger.info("📊 Generating Metadata Distributions...")
        try:
            fig = plot_sample_distribution(
                adata,
                categorical_cols=workflow.priority_categorical[:5],
                numeric_cols=workflow.priority_numeric[:5],
                output_path=str(qc_output_dir / 'metadata_distributions.html')
            )
            results['generated'].append('metadata_distributions')
            try:
                if fig:
                    fig.write_image(
                        str(qc_output_dir / 'metadata_distributions.png'),
                        width=1400, height=1000
                    )
            except:
                pass
            logger.info("✅ Metadata Distributions")
        except Exception as e:
            logger.error(f"Failed to generate Metadata Distributions: {e}")
            results['failed'].append(('metadata_distributions', str(e)))
        
        # ===== PLOT 5: Metadata Correlation Heatmap =====
        logger.info("📊 Generating Metadata Correlation Heatmap...")
        try:
            fig = plot_metadata_heatmap(
                adata,
                numeric_cols=workflow.priority_numeric[:10],
                output_path=str(qc_output_dir / 'metadata_correlation.html')
            )
            results['generated'].append('metadata_correlation')
            try:
                if fig:
                    fig.write_image(
                        str(qc_output_dir / 'metadata_correlation.png'),
                        width=900, height=800
                    )
            except:
                pass
            logger.info("✅ Metadata Correlation Heatmap")
        except Exception as e:
            logger.error(f"Failed to generate Metadata Correlation Heatmap: {e}")
            results['failed'].append(('metadata_correlation', str(e)))
        
        # ===== PLOT 6: Geographic Map (if coordinates available) =====
        has_coords = ('latitude' in adata.obs.columns and 'longitude' in adata.obs.columns) or \
                     ('lat' in adata.obs.columns and 'lon' in adata.obs.columns)
        
        if has_coords:
            logger.info("📊 Generating Geographic Map...")
            try:
                lat_col = 'latitude' if 'latitude' in adata.obs.columns else 'lat'
                lon_col = 'longitude' if 'longitude' in adata.obs.columns else 'lon'
                
                fig = create_geographic_map(
                    adata,
                    lat_col=lat_col,
                    lon_col=lon_col,
                    color_by='env_category_type' if 'env_category_type' in adata.obs.columns else None,
                    output_path=str(qc_output_dir / 'geographic_map.html')
                )
                results['generated'].append('geographic_map')
                try:
                    if fig:
                        fig.write_image(
                            str(qc_output_dir / 'geographic_map.png'),
                            width=1200, height=700
                        )
                except:
                    pass
                logger.info("✅ Geographic Map")
            except Exception as e:
                logger.error(f"Failed to generate Geographic Map: {e}")
                results['failed'].append(('geographic_map', str(e)))
        
        # ===== PLOT 7: Data Quality Summary Table =====
        logger.info("📊 Generating Data Quality Summary Table...")
        try:
            fig = plot_metadata_summary_table(
                adata,
                output_path=str(qc_output_dir / 'metadata_summary_table.html')
            )
            results['generated'].append('metadata_summary_table')
            try:
                if fig:
                    fig.write_image(
                        str(qc_output_dir / 'metadata_summary_table.png'),
                        width=1200, height=600
                    )
            except:
                pass
            logger.info("✅ Data Quality Summary Table")
        except Exception as e:
            logger.error(f"Failed to generate Summary Table: {e}")
            results['failed'].append(('metadata_summary_table', str(e)))
        
        # ===== PLOT 8: Taxa Abundance Overview =====
        logger.info("📊 Generating Taxa Abundance Overview...")
        try:
            fig = create_taxa_abundance_overview(
                adata,
                top_n=15,
                output_path=str(qc_output_dir / 'taxa_abundance_overview.html')
            )
            results['generated'].append('taxa_abundance_overview')
            try:
                fig.write_image(
                    str(qc_output_dir / 'taxa_abundance_overview.png'),
                    width=1200, height=500
                )
            except:
                pass
            logger.info("✅ Taxa Abundance Overview")
        except Exception as e:
            logger.error(f"Failed to generate Taxa Abundance Overview: {e}")
            results['failed'].append(('taxa_abundance_overview', str(e)))
        
        # ===== PLOT 9: Feature Sparsity Distribution =====
        logger.info("📊 Generating Feature Sparsity Distribution...")
        try:
            fig = create_feature_sparsity_plot(
                adata,
                output_path=str(qc_output_dir / 'feature_sparsity.html')
            )
            results['generated'].append('feature_sparsity')
            try:
                fig.write_image(
                    str(qc_output_dir / 'feature_sparsity.png'),
                    width=1200, height=500
                )
            except:
                pass
            logger.info("✅ Feature Sparsity Distribution")
        except Exception as e:
            logger.error(f"Failed to generate Feature Sparsity: {e}")
            results['failed'].append(('feature_sparsity', str(e)))
        
        # ===== PLOT 10: Sequencing Depth Histogram =====
        logger.info("📊 Generating Sequencing Depth Histogram...")
        try:
            fig = create_sample_sequencing_depth_histogram(
                adata,
                output_path=str(qc_output_dir / 'sequencing_depth_histogram.html')
            )
            results['generated'].append('sequencing_depth_histogram')
            try:
                fig.write_image(
                    str(qc_output_dir / 'sequencing_depth_histogram.png'),
                    width=1200, height=500
                )
            except:
                pass
            logger.info("✅ Sequencing Depth Histogram")
        except Exception as e:
            logger.error(f"Failed to generate Sequencing Depth Histogram: {e}")
            results['failed'].append(('sequencing_depth_histogram', str(e)))
    
    except Exception as e:
        logger.error(f"QC Suite execution failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False
    
    # --- Summary Report ---
    logger.info("\n" + "="*70)
    logger.info("QC VISUALIZATION SUITE SUMMARY")
    logger.info("="*70)
    logger.info(f"✅ Generated: {len(results['generated'])} plots")
    for plot_name in results['generated']:
        logger.info(f"  • {plot_name}")
    
    if results['failed']:
        logger.warning(f"⚠️  Failed: {len(results['failed'])} plots")
        for plot_name, error in results['failed']:
            logger.warning(f"  • {plot_name}: {error[:80]}")
    
    logger.info(f"📁 Output saved to: {qc_output_dir}")
    logger.info("="*70)
    
    return True


# Type hints helper (avoid circular import)
from typing import Any
