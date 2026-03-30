"""
Mock Examples for QC Visualization Module

Demonstrates how to use the QC visualization functions with the 'heather' 
Plotly template for publication-ready plots.

Examples:
    >>> from workflow_16s.qc.visualization.examples import (
    ...     create_mock_qc_data, 
    ...     example_qc_impact_dashboard,
    ...     example_qc_depth_plot,
    ...     example_qc_heatmap
    ... )
    >>> 
    >>> # Create mock data
    >>> adata = create_mock_qc_data(n_samples=200, n_features=1000)
    >>> 
    >>> # Generate QC impact dashboard
    >>> fig = example_qc_impact_dashboard(adata)
    >>> fig.show()
    >>> 
    >>> # Generate QC depth plot
    >>> fig = example_qc_depth_plot(adata)
    >>> fig.show()
"""

import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from typing import Optional
import plotly.graph_objects as go
from typing import Tuple

from workflow_16s.qc.visualization.main import (
    create_qc_impact_dashboard,
    plot_qc_metrics_over_sequencing_depth,
    create_sample_qc_heatmap
)


def save_figure_as_png(fig: go.Figure, png_path: Path, width: int = 1200, height: int = 800) -> bool:
    """
    Save a Plotly figure as PNG using kaleido.
    
    Args:
        fig: Plotly figure
        png_path: Path to save PNG
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        True if successful, False otherwise
    """
    try:
        fig.write_image(str(png_path), width=width, height=height)
        return True
    except Exception as e:
        print(f"⚠️  Could not save PNG {png_path}: {e}")
        return False


def create_mock_qc_data(
    n_samples: int = 200,
    n_features: int = 1000,
    contamination_rate: float = 0.15,
    fail_rate: float = 0.10
) -> ad.AnnData:
    """
    Create synthetic AnnData object with realistic QC metrics.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features (taxa)
        contamination_rate: Fraction of features to mark as contaminants
        fail_rate: Fraction of samples to mark as QC failed
        
    Returns:
        AnnData object with QC annotations
        
    Example:
        >>> adata = create_mock_qc_data(n_samples=100)
        >>> print(f"Created {adata.n_obs} samples × {adata.n_vars} features")
        >>> print(f"QC Flags: {adata.obs['qc_overall_flag'].value_counts()}")
    """
    # Create random count matrix (log-normal distribution for realistic microbiome data)
    np.random.seed(42)
    counts = np.random.lognormal(
        mean=2.0, 
        sigma=1.5, 
        size=(n_samples, n_features)
    ).astype(int)
    
    # Create AnnData object
    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=np.arange(n_samples)),
        var=pd.DataFrame(index=[f"OTU_{i}" for i in range(n_features)])
    )
    
    # ===== Add Metadata =====
    adata.obs['sample_id'] = [f"Sample_{i:03d}" for i in range(n_samples)]
    
    # Environmental categories
    env_categories = ['Soil', 'Ocean', 'Gut', 'Oral', 'Environmental']
    adata.obs['env_category_type'] = np.random.choice(env_categories, n_samples)
    
    # Location coordinates
    adata.obs['latitude'] = np.random.uniform(-90, 90, n_samples)
    adata.obs['longitude'] = np.random.uniform(-180, 180, n_samples)
    
    # Study/project assignment
    adata.obs['project_accession'] = np.random.choice(
        ['PRJNA001', 'PRJNA002', 'PRJNA003', 'PRJNA004'],
        n_samples
    )
    
    # ===== Add Read Depth =====
    read_depth = np.random.lognormal(mean=10.5, sigma=0.8, size=n_samples)
    adata.obs['read_depth'] = read_depth.astype(int)
    
    # ===== Add QC Flags =====
    # Create mixed QC flags
    n_fail = int(n_samples * fail_rate)
    n_warn = int(n_samples * 0.15)  # 15% warnings
    
    flags = ['PASS'] * (n_samples - n_fail - n_warn) + ['WARNING'] * n_warn + ['FAIL'] * n_fail
    np.random.shuffle(flags)
    adata.obs['qc_overall_flag'] = flags
    
    # Sub-metrics contributing to QC
    adata.obs['qc_env_match'] = np.random.choice(
        ['PASS', 'WARNING', 'FAIL'],
        n_samples,
        p=[0.85, 0.10, 0.05]
    )
    adata.obs['qc_metadata_outlier'] = np.random.choice(
        ['PASS', 'WARNING'],
        n_samples,
        p=[0.92, 0.08]
    )
    adata.obs['qc_primer_match'] = np.random.choice(
        ['PASS', 'FAIL'],
        n_samples,
        p=[0.95, 0.05]
    )
    
    # ===== Add Contamination Detection =====
    n_contam = int(n_features * contamination_rate)
    is_contaminant = np.zeros(n_features, dtype=bool)
    is_contaminant[np.random.choice(n_features, n_contam, replace=False)] = True
    adata.var['is_contaminant'] = is_contaminant
    
    # Contamination scores
    contam_scores = np.zeros(n_features)
    contam_scores[is_contaminant] = np.random.uniform(0.3, 1.0, n_contam)
    contam_scores[~is_contaminant] = np.random.uniform(0.0, 0.3, n_features - n_contam)
    adata.var['contamination_score'] = contam_scores
    
    # ===== Add Diversity Metrics =====
    # Shannon diversity (higher = more diverse)
    adata.obs['shannon'] = np.random.normal(loc=5.5, scale=0.8, size=n_samples)
    adata.obs['shannon'] = np.clip(adata.obs['shannon'], 2, 8)
    
    # Observed features
    adata.obs['observed_features'] = (counts > 0).sum(axis=1)
    
    # ===== Add Mock PCA =====
    pca_coords = np.random.normal(0, 1, (n_samples, 10))
    adata.obsm['X_pca'] = pca_coords
    
    return adata


def example_qc_impact_dashboard(
    adata: ad.AnnData = None,
    output_path: str = None
) -> 'go.Figure':
    """
    Example: Create QC impact dashboard.
    
    Args:
        adata: AnnData object (or creates mock if None)
        output_path: Optional save path for HTML
        
    Returns:
        Plotly figure
        
    Example:
        >>> fig = example_qc_impact_dashboard()
        >>> fig.show()
        >>> # Save to file
        >>> fig.write_html("qc_impact_dashboard.html")
    """
    if adata is None:
        adata = create_mock_qc_data(n_samples=150, n_features=800)
    
    # Create mock QC results dict
    qc_results = {
        'n_passed': (adata.obs['qc_overall_flag'] == 'PASS').sum(),
        'n_warning': (adata.obs['qc_overall_flag'] == 'WARNING').sum(),
        'n_failed': (adata.obs['qc_overall_flag'] == 'FAIL').sum(),
        'contamination_threshold': 0.5,
        'metadata': {
            'report': pd.DataFrame({
                'level': np.random.choice(['info', 'warning', 'error'], 10),
                'message': [f'QC Check {i}' for i in range(10)]
            })
        }
    }
    
    fig = create_qc_impact_dashboard(
        adata_before=None,
        adata_after=adata,
        qc_results=qc_results,
        output_path=output_path or 'mock_qc_impact_dashboard.html'
    )
    
    return fig


def example_qc_depth_plot(
    adata: ad.AnnData = None,
    output_path: str = None
) -> 'go.Figure':
    """
    Example: Create sequencing depth distribution plot.
    
    Demonstrates how QC status correlates with sequencing depth.
    
    Args:
        adata: AnnData object (or creates mock if None)
        output_path: Optional save path for HTML
        
    Returns:
        Plotly figure
        
    Example:
        >>> fig = example_qc_depth_plot()
        >>> fig.show()
    """
    if adata is None:
        adata = create_mock_qc_data(n_samples=200, n_features=1000)
    
    fig = plot_qc_metrics_over_sequencing_depth(
        adata=adata,
        output_path=output_path or 'mock_qc_depth_plot.html'
    )
    
    return fig


def example_qc_heatmap(
    adata: ad.AnnData = None,
    output_path: str = None
) -> 'go.Figure':
    """
    Example: Create sample-level QC heatmap.
    
    Shows QC status across multiple metrics for each sample.
    
    Args:
        adata: AnnData object (or creates mock if None)
        output_path: Optional save path for HTML
        
    Returns:
        Plotly figure
        
    Example:
        >>> fig = example_qc_heatmap()
        >>> fig.show()
    """
    if adata is None:
        adata = create_mock_qc_data(n_samples=100, n_features=500)
    
    # Define QC columns to include
    qc_columns = [
        'qc_overall_flag',
        'qc_env_match',
        'qc_metadata_outlier',
        'qc_primer_match'
    ]
    
    # Only use columns that exist
    qc_columns = [col for col in qc_columns if col in adata.obs.columns]
    
    fig = create_sample_qc_heatmap(
        adata=adata,
        qc_columns=qc_columns,
        output_path=output_path or 'mock_qc_heatmap.html'
    )
    
    return fig


def run_all_examples(output_dir: str = '.') -> dict:
    """
    Run all example plots and save to directory.
    
    Args:
        output_dir: Directory to save HTML outputs
        
    Returns:
        Dictionary with results
        
    Example:
        >>> results = run_all_examples(output_dir='./qc_examples')
        >>> print(f"Generated {len(results)} plots")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock data once
    adata = create_mock_qc_data(n_samples=200, n_features=1000)
    
    results = {}
    
    # Generate all plots
    try:
        print("📊 Generating QC Impact Dashboard...")
        fig1 = example_qc_impact_dashboard(
            adata=adata,
            output_path=str(output_dir / 'qc_impact_dashboard.html')
        )
        results['dashboard'] = '✅ QC Impact Dashboard'
    except Exception as e:
        results['dashboard'] = f'❌ Error: {e}'
    
    try:
        print("📊 Generating Sequencing Depth Plot...")
        fig2 = example_qc_depth_plot(
            adata=adata,
            output_path=str(output_dir / 'qc_depth_plot.html')
        )
        results['depth'] = '✅ QC Depth Plot'
    except Exception as e:
        results['depth'] = f'❌ Error: {e}'
    
    try:
        print("📊 Generating QC Heatmap...")
        fig3 = example_qc_heatmap(
            adata=adata,
            output_path=str(output_dir / 'qc_heatmap.html')
        )
        results['heatmap'] = '✅ QC Heatmap'
    except Exception as e:
        results['heatmap'] = f'❌ Error: {e}'
    
    # Print summary
    print("\n" + "="*60)
    print("QC EXAMPLE GENERATION SUMMARY")
    print("="*60)
    for name, status in results.items():
        print(f"{name:20s}: {status}")
    print(f"\nOutput directory: {output_dir}")
    print("="*60)
    
    return results


def example_metadata_distributions(
    adata: ad.AnnData = None,
    output_path: str = None
) -> go.Figure:
    """
    Example: Plot sample metadata distributions.
    
    Args:
        adata: Optional AnnData object (generates mock data if None)
        output_path: Optional path to save HTML output
        
    Returns:
        Plotly figure
    """
    if adata is None:
        adata = create_mock_qc_data(n_samples=150, n_features=800)
    
    from workflow_16s.downstream.visualization.sample_metadata import plot_sample_distribution
    
    fig = plot_sample_distribution(
        adata,
        categorical_cols=['env_category_type', 'project_accession'],
        numeric_cols=['read_depth', 'latitude', 'longitude'],
        output_path=output_path or 'mock_metadata_distributions.html'
    )
    
    return fig


def example_metadata_correlation(
    adata: ad.AnnData = None,
    output_path: str = None
) -> go.Figure:
    """
    Example: Plot metadata correlation heatmap.
    
    Args:
        adata: Optional AnnData object (generates mock data if None)
        output_path: Optional path to save HTML output
        
    Returns:
        Plotly figure
    """
    if adata is None:
        adata = create_mock_qc_data(n_samples=150, n_features=800)
    
    from workflow_16s.downstream.visualization.sample_metadata import plot_metadata_heatmap
    
    fig = plot_metadata_heatmap(
        adata,
        numeric_cols=['read_depth', 'latitude', 'longitude', 'shannon'],
        output_path=output_path or 'mock_metadata_correlation.html'
    )
    
    return fig


def example_geographic_map(
    adata: ad.AnnData = None,
    output_path: str = None
) -> go.Figure:
    """
    Example: Plot geographic distribution of samples.
    
    Args:
        adata: Optional AnnData object (generates mock data if None)
        output_path: Optional path to save HTML output
        
    Returns:
        Plotly figure
    """
    if adata is None:
        adata = create_mock_qc_data(n_samples=150, n_features=800)
    
    from workflow_16s.downstream.visualization.sample_metadata import create_geographic_map
    
    fig = create_geographic_map(
        adata,
        lat_col='latitude',
        lon_col='longitude',
        color_by='env_category_type',
        output_path=output_path or 'mock_geographic_map.html'
    )
    
    return fig


def run_all_examples(
    output_dir: str = '.',
    include_qc: bool = True,
    include_metadata: bool = True,
    save_png: bool = True
) -> dict:
    """
    Run all example plots and save to directory.
    
    Args:
        output_dir: Directory to save HTML and PNG outputs
        include_qc: Whether to generate QC examples
        include_metadata: Whether to generate metadata examples
        save_png: Whether to save PNG versions (requires kaleido)
        
    Returns:
        Dictionary with results
        
    Example:
        >>> results = run_all_examples(output_dir='./examples', save_png=True)
        >>> print(f"Generated {len(results)} plots")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock data once (will be reused in all examples)
    adata = create_mock_qc_data(n_samples=200, n_features=1000)
    
    results = {}
    png_success_count = 0
    png_fail_count = 0
    
    if include_qc:
        print("\n" + "="*60)
        print("QUALITY CONTROL EXAMPLES")
        print("="*60)
        
        # Generate QC plots
        try:
            print("📊 Generating QC Impact Dashboard...")
            fig1 = example_qc_impact_dashboard(
                adata=adata,
                output_path=str(output_dir / 'qc_impact_dashboard.html')
            )
            results['qc_dashboard'] = '✅ QC Impact Dashboard'
            if save_png and fig1 is not None:
                if save_figure_as_png(fig1, output_dir / 'qc_impact_dashboard.png', width=1400, height=900):
                    png_success_count += 1
                else:
                    png_fail_count += 1
        except Exception as e:
            results['qc_dashboard'] = f'❌ Error: {e}'
        
        try:
            print("📊 Generating Sequencing Depth Plot...")
            fig2 = example_qc_depth_plot(
                adata=adata,
                output_path=str(output_dir / 'qc_depth_plot.html')
            )
            results['qc_depth'] = '✅ QC Depth Plot'
            if save_png and fig2 is not None:
                if save_figure_as_png(fig2, output_dir / 'qc_depth_plot.png', width=1200, height=600):
                    png_success_count += 1
                else:
                    png_fail_count += 1
        except Exception as e:
            results['qc_depth'] = f'❌ Error: {e}'
        
        try:
            print("📊 Generating QC Heatmap...")
            fig3 = example_qc_heatmap(
                adata=adata,
                output_path=str(output_dir / 'qc_heatmap.html')
            )
            results['qc_heatmap'] = '✅ QC Heatmap'
            if save_png and fig3 is not None:
                if save_figure_as_png(fig3, output_dir / 'qc_heatmap.png', width=1200, height=700):
                    png_success_count += 1
                else:
                    png_fail_count += 1
        except Exception as e:
            results['qc_heatmap'] = f'❌ Error: {e}'
    
    if include_metadata:
        print("\n" + "="*60)
        print("SAMPLE METADATA EXAMPLES")
        print("="*60)
        
        # Generate metadata plots
        try:
            print("📊 Generating Metadata Distributions...")
            fig4 = example_metadata_distributions(
                adata=adata,
                output_path=str(output_dir / 'metadata_distributions.html')
            )
            results['metadata_dist'] = '✅ Metadata Distributions'
            if save_png and fig4 is not None:
                if save_figure_as_png(fig4, output_dir / 'metadata_distributions.png', width=1400, height=1000):
                    png_success_count += 1
                else:
                    png_fail_count += 1
        except Exception as e:
            results['metadata_dist'] = f'❌ Error: {e}'
        
        try:
            print("📊 Generating Metadata Correlation Heatmap...")
            fig5 = example_metadata_correlation(
                adata=adata,
                output_path=str(output_dir / 'metadata_correlation.html')
            )
            results['metadata_corr'] = '✅ Metadata Correlation'
            if save_png and fig5 is not None:
                if save_figure_as_png(fig5, output_dir / 'metadata_correlation.png', width=900, height=800):
                    png_success_count += 1
                else:
                    png_fail_count += 1
        except Exception as e:
            results['metadata_corr'] = f'❌ Error: {e}'
        
        try:
            print("📊 Generating Geographic Map...")
            fig6 = example_geographic_map(
                adata=adata,
                output_path=str(output_dir / 'geographic_map.html')
            )
            results['geographic'] = '✅ Geographic Map'
            if save_png and fig6 is not None:
                if save_figure_as_png(fig6, output_dir / 'geographic_map.png', width=1200, height=700):
                    png_success_count += 1
                else:
                    png_fail_count += 1
        except Exception as e:
            results['geographic'] = f'❌ Error: {e}'
    
    # Print final summary
    print("\n" + "="*60)
    print("EXAMPLE GENERATION SUMMARY")
    print("="*60)
    for name, status in results.items():
        print(f"{name:20s}: {status}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Total plots: {len(results)}")
    if save_png:
        print(f"PNG exports: {png_success_count} ✅ | {png_fail_count} ❌")
    print("="*60 + "\n")
    
    return results


if __name__ == '__main__':
    # Run all examples (QC + metadata)
    run_all_examples(output_dir='./visualization_examples')
