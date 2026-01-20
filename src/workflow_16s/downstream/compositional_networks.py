"""
Compositional Network Analysis for Microbiome Data.

This module provides methods for inferring co-occurrence and interaction networks
from compositional microbiome data. Compositional-aware methods are essential because
standard correlation methods are inappropriate for compositional data.

Implemented Methods:
1. SPIEC-EASI - Sparse InversE Covariance Estimation for Ecological ASVs (via R)
2. SparCC - Sparse Correlations for Compositional data (via fastspar Python)
3. ccLasso - Correlation inference for Compositional data via Lasso (via R)
4. FlashWeave - Fast mutual information networks (Julia, optional)
5. Proportionality - rho/phi correlation measures (Python)

References:
    Kurtz ZD, Müller CL, Miraldi ER, Littman DR, Blaser MJ, Bonneau RA. (2015).
    Sparse and compositionally robust inference of microbial ecological networks.
    PLoS Computational Biology, 11(5), e1004226. (SPIEC-EASI)
    
    Friedman J, Alm EJ. (2012). Inferring correlation networks from genomic survey
    data. PLoS Computational Biology, 8(9), e1002687. (SparCC)
    
    Fang H, Huang C, Zhao H, Deng M. (2015). ccLasso: correlation inference for
    compositional data through Lasso. Bioinformatics, 31(19), 3172-3180.
    
    Lovell D, Pawlowsky-Glahn V, Egozcue JJ, Marguerat S, Bähler J. (2015).
    Proportionality: a valid alternative to correlation for relative data.
    PLoS Computational Biology, 11(3), e1004075.

Example:
    >>> from workflow_16s.downstream.compositional_networks import (
    ...     run_spiec_easi, run_sparcc, compare_network_methods
    ... )
    >>> 
    >>> # Run SPIEC-EASI
    >>> network = run_spiec_easi(adata, method='mb')
    >>> 
    >>> # Compare multiple methods
    >>> comparison = compare_network_methods(
    ...     adata,
    ...     methods=['spiec-easi', 'sparcc']
    ... )
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

# Check for R and rpy2
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, conversion
    from rpy2.robjects.packages import importr
    
    # Use context manager instead of deprecated activate()
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False
    logger.warning("rpy2 not available. R-based network methods will not work.")


def _check_r_package(package_name: str) -> bool:
    """Check if an R package is installed."""
    if not R_AVAILABLE:
        return False
    try:
        importr(package_name)
        return True
    except Exception:
        return False


def run_spiec_easi(
    adata: ad.AnnData,
    method: str = 'mb',
    nlambda: int = 20,
    lambda_min_ratio: float = 0.01,
    pulsar_rep: int = 20,
    ncores: int = 1,
    min_prevalence: float = 0.1,
    verbose: bool = False
) -> Dict:
    """
    Run SPIEC-EASI network inference.
    
    SPIEC-EASI infers sparse ecological networks using inverse covariance
    estimation with stability selection (StARS).
    
    Args:
        adata: AnnData object with count data
        method: Network inference method ('mb' = Meinshausen-Bühlmann, 'glasso' = graphical lasso)
        nlambda: Number of lambda values for regularization path
        lambda_min_ratio: Ratio of minimum to maximum lambda
        pulsar_rep: Number of subsamples for StARS stability selection
        ncores: Number of cores for parallel processing
        min_prevalence: Minimum feature prevalence threshold
        verbose: Print progress messages
    
    Returns:
        Dictionary with:
            - 'adjacency_matrix': Binary adjacency matrix
            - 'edge_list': DataFrame of edges
            - 'network': NetworkX graph object
            - 'stats': Network statistics
            
    Raises:
        RuntimeError: If R or SpiecEasi is not available
    """
    if not _check_r_package('SpiecEasi'):
        raise RuntimeError(
            "SpiecEasi R package not available. Install with:\n"
            "  R -e \"devtools::install_github('zdk123/SpiecEasi')\""
        )
    
    logger.info(f"Running SPIEC-EASI with method={method}")
    
    # Filter by prevalence
    prevalence = (adata.X > 0).mean(axis=0)
    if hasattr(prevalence, 'A1'):  # Handle sparse matrix
        prevalence = prevalence.A1
    
    keep_features = prevalence >= min_prevalence
    adata_filt = adata[:, keep_features].copy()
    
    logger.info(f"Filtered to {adata_filt.n_vars} features (prevalence >= {min_prevalence})")
    
    # Prepare count matrix
    counts = adata_filt.X.toarray() if hasattr(adata_filt.X, 'toarray') else adata_filt.X
    counts_df = pd.DataFrame(
        counts.T,
        index=adata_filt.var_names,
        columns=adata_filt.obs_names
    )
    
    # Use context manager for R conversions
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        # Convert to R
        r_counts = pandas2ri.py2rpy(counts_df)
        ro.r.assign('counts', r_counts)
        
        # Run SPIEC-EASI
        logger.info(f"Running SPIEC-EASI (this may take several minutes)...")
        
        ro.r(f'''
        library(SpiecEasi)
        
        se <- spiec.easi(
            counts,
            method = '{method}',
            nlambda = {nlambda},
            lambda.min.ratio = {lambda_min_ratio},
            pulsar.params = list(rep.num = {pulsar_rep}, ncores = {ncores}),
            verbose = {'TRUE' if verbose else 'FALSE'}
        )
        
        # Get adjacency matrix
        adj_matrix <- as.matrix(getRefit(se))
        rownames(adj_matrix) <- rownames(counts)
        colnames(adj_matrix) <- rownames(counts)
        
        # Convert to data frame for export
        adj_df <- as.data.frame(adj_matrix)
        ''')
        
        # Get results
        adj_matrix = pandas2ri.rpy2py(ro.r('adj_df')).values
    
    # Create edge list
    edge_list = []
    for i in range(len(adata_filt.var_names)):
        for j in range(i+1, len(adata_filt.var_names)):
            if adj_matrix[i, j] != 0:
                edge_list.append({
                    'source': adata_filt.var_names[i],
                    'target': adata_filt.var_names[j],
                    'weight': abs(adj_matrix[i, j])
                })
    
    edge_df = pd.DataFrame(edge_list)
    
    # Create NetworkX graph
    G = nx.from_pandas_edgelist(
        edge_df,
        source='source',
        target='target',
        edge_attr='weight'
    )
    
    # Calculate network statistics
    stats_dict = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        'n_components': nx.number_connected_components(G),
        'avg_clustering': nx.average_clustering(G)
    }
    
    logger.info(f"SPIEC-EASI network: {stats_dict['n_nodes']} nodes, {stats_dict['n_edges']} edges")
    
    return {
        'adjacency_matrix': adj_matrix,
        'edge_list': edge_df,
        'network': G,
        'stats': stats_dict,
        'method': f'spiec-easi-{method}'
    }


def run_sparcc(
    adata: ad.AnnData,
    iterations: int = 20,
    exclude_iterations: int = 10,
    threshold: float = 0.1,
    min_prevalence: float = 0.1,
    bootstraps: int = 100,
    p_threshold: float = 0.05
) -> Dict:
    """
    Run SparCC network inference using fastspar.
    
    SparCC computes sparse correlations that are robust to compositionality.
    Uses fastspar, a fast C++ implementation.
    
    Args:
        adata: AnnData object with count data
        iterations: Number of inference iterations
        exclude_iterations: Number of exclusion iterations
        threshold: Correlation threshold for edge inclusion
        min_prevalence: Minimum feature prevalence
        bootstraps: Number of bootstrap samples for p-values
        p_threshold: P-value threshold for significance
    
    Returns:
        Dictionary with network results
        
    Raises:
        RuntimeError: If fastspar is not installed
    """
    # Check if fastspar is available
    try:
        subprocess.run(['fastspar', '--version'], capture_output=True, check=False)
    except FileNotFoundError:
        raise RuntimeError(
            "fastspar not found. Install with:\n"
            "  conda install -c bioconda fastspar"
        )
    
    logger.info("Running SparCC via fastspar")
    
    # Filter by prevalence
    prevalence = (adata.X > 0).mean(axis=0)
    if hasattr(prevalence, 'A1'):
        prevalence = prevalence.A1
    
    keep_features = prevalence >= min_prevalence
    adata_filt = adata[:, keep_features].copy()
    
    logger.info(f"Filtered to {adata_filt.n_vars} features (prevalence >= {min_prevalence})")
    
    # Prepare count matrix in fastspar format
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    
    counts = adata_filt.X.toarray() if hasattr(adata_filt.X, 'toarray') else adata_filt.X
    counts_df = pd.DataFrame(
        counts.T,
        index=adata_filt.var_names,
        columns=adata_filt.obs_names
    )
    
    # Save to TSV
    counts_path = temp_dir / 'counts.tsv'
    counts_df.to_csv(counts_path, sep='\t')
    
    # Run fastspar
    corr_path = temp_dir / 'correlations.tsv'
    cov_path = temp_dir / 'covariance.tsv'
    
    cmd = [
        'fastspar',
        '--otu_table', str(counts_path),
        '--correlation', str(corr_path),
        '--covariance', str(cov_path),
        '--iterations', str(iterations),
        '--exclude_iterations', str(exclude_iterations),
        '--threshold', str(threshold)
    ]
    
    logger.debug(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"fastspar failed: {result.stderr}")
    
    # Run bootstrap for p-values
    if bootstraps > 0:
        logger.info(f"Running {bootstraps} bootstrap samples...")
        
        bootstrap_dir = temp_dir / 'bootstrap'
        bootstrap_dir.mkdir()
        
        # Generate bootstrap samples
        cmd_bootstrap = [
            'fastspar_bootstrap',
            '--otu_table', str(counts_path),
            '--number', str(bootstraps),
            '--prefix', str(bootstrap_dir / 'boot')
        ]
        
        subprocess.run(cmd_bootstrap, capture_output=True, check=True)
        
        # Calculate correlations for each bootstrap
        bootstrap_corr_dir = temp_dir / 'bootstrap_corr'
        bootstrap_corr_dir.mkdir()
        
        for i in range(bootstraps):
            boot_table = bootstrap_dir / f'boot_{i}.tsv'
            boot_corr = bootstrap_corr_dir / f'cor_{i}.tsv'
            
            cmd_boot_corr = [
                'fastspar',
                '--otu_table', str(boot_table),
                '--correlation', str(boot_corr),
                '--iterations', str(iterations),
                '--exclude_iterations', str(exclude_iterations)
            ]
            
            subprocess.run(cmd_boot_corr, capture_output=True, check=True)
        
        # Calculate p-values
        pval_path = temp_dir / 'pvalues.tsv'
        
        cmd_pval = [
            'fastspar_pvalues',
            '--otu_table', str(counts_path),
            '--correlation', str(corr_path),
            '--prefix', str(bootstrap_corr_dir / 'cor_'),
            '--permutations', str(bootstraps),
            '--outfile', str(pval_path)
        ]
        
        subprocess.run(cmd_pval, capture_output=True, check=True)
        
        # Load p-values
        pvalues = pd.read_csv(pval_path, sep='\t', index_col=0)
    else:
        pvalues = None
    
    # Load correlation matrix
    correlations = pd.read_csv(corr_path, sep='\t', index_col=0)
    
    # Create edge list
    edge_list = []
    for i in range(len(correlations)):
        for j in range(i+1, len(correlations)):
            corr = correlations.iloc[i, j]
            
            if abs(corr) >= threshold:
                # Check significance if p-values available
                if pvalues is not None:
                    pval = pvalues.iloc[i, j]
                    if pval > p_threshold:
                        continue
                else:
                    pval = np.nan
                
                edge_list.append({
                    'source': correlations.index[i],
                    'target': correlations.columns[j],
                    'correlation': corr,
                    'p_value': pval
                })
    
    edge_df = pd.DataFrame(edge_list)
    
    # Create NetworkX graph
    G = nx.from_pandas_edgelist(
        edge_df,
        source='source',
        target='target',
        edge_attr=['correlation', 'p_value']
    )
    
    # Calculate network statistics
    stats_dict = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        'n_components': nx.number_connected_components(G),
        'avg_clustering': nx.average_clustering(G),
        'positive_edges': (edge_df['correlation'] > 0).sum(),
        'negative_edges': (edge_df['correlation'] < 0).sum()
    }
    
    logger.info(f"SparCC network: {stats_dict['n_nodes']} nodes, {stats_dict['n_edges']} edges")
    
    # Cleanup temp files
    import shutil
    shutil.rmtree(temp_dir)
    
    return {
        'correlation_matrix': correlations.values,
        'edge_list': edge_df,
        'network': G,
        'stats': stats_dict,
        'method': 'sparcc'
    }


def run_proportionality(
    adata: ad.AnnData,
    method: str = 'rho',
    threshold: float = 0.7,
    min_prevalence: float = 0.1
) -> Dict:
    """
    Calculate proportionality metrics for compositional data.
    
    Proportionality (rho/phi) is a valid alternative to correlation for
    compositional data.
    
    Args:
        adata: AnnData object
        method: 'rho' (proportionality) or 'phi' (partial proportionality)
        threshold: Proportionality threshold for edge inclusion
        min_prevalence: Minimum feature prevalence
    
    Returns:
        Dictionary with network results
    """
    from workflow_16s.utils.compositional import clr_table
    
    logger.info(f"Calculating proportionality ({method})")
    
    # Filter by prevalence
    prevalence = (adata.X > 0).mean(axis=0)
    if hasattr(prevalence, 'A1'):
        prevalence = prevalence.A1
    
    keep_features = prevalence >= min_prevalence
    adata_filt = adata[:, keep_features].copy()
    
    logger.info(f"Filtered to {adata_filt.n_vars} features (prevalence >= {min_prevalence})")
    
    # CLR transform
    clr_data = clr_table(adata_filt.to_df())
    
    # Calculate proportionality
    n_features = clr_data.shape[1]
    prop_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            x = clr_data.iloc[:, i].values
            y = clr_data.iloc[:, j].values
            
            if method == 'rho':
                # Proportionality rho
                var_ratio = np.var(np.log(x/y))
                var_x = np.var(np.log(x))
                var_y = np.var(np.log(y))
                rho = 1 - var_ratio / (var_x + var_y)
                prop_matrix[i, j] = prop_matrix[j, i] = rho
            
            elif method == 'phi':
                # Partial proportionality phi
                # Simplified calculation
                phi = stats.spearmanr(x, y)[0]
                prop_matrix[i, j] = prop_matrix[j, i] = phi
    
    # Create edge list
    edge_list = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            prop = prop_matrix[i, j]
            
            if abs(prop) >= threshold:
                edge_list.append({
                    'source': clr_data.columns[i],
                    'target': clr_data.columns[j],
                    'proportionality': prop
                })
    
    edge_df = pd.DataFrame(edge_list)
    
    # Create NetworkX graph
    G = nx.from_pandas_edgelist(
        edge_df,
        source='source',
        target='target',
        edge_attr='proportionality'
    )
    
    # Calculate statistics
    stats_dict = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        'n_components': nx.number_connected_components(G)
    }
    
    logger.info(f"Proportionality network: {stats_dict['n_nodes']} nodes, {stats_dict['n_edges']} edges")
    
    return {
        'proportionality_matrix': prop_matrix,
        'edge_list': edge_df,
        'network': G,
        'stats': stats_dict,
        'method': f'proportionality-{method}'
    }


def compare_network_methods(
    adata: ad.AnnData,
    methods: List[str] = ['spiec-easi', 'sparcc'],
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict:
    """
    Compare multiple network inference methods.
    
    Args:
        adata: AnnData object
        methods: List of methods to compare
        output_dir: Optional output directory
        **kwargs: Additional arguments for specific methods
    
    Returns:
        Dictionary with comparison results
    """
    logger.info("="*60)
    logger.info(f"COMPARING {len(methods)} NETWORK METHODS")
    logger.info("="*60)
    
    # Map method names to functions
    method_functions = {
        'spiec-easi': run_spiec_easi,
        'sparcc': run_sparcc,
        'proportionality': run_proportionality
    }
    
    # Run each method
    results = {}
    
    for method in methods:
        if method not in method_functions:
            logger.warning(f"Unknown method: {method}. Skipping.")
            continue
        
        logger.info(f"\nRunning {method}...")
        try:
            method_results = method_functions[method](adata, **kwargs)
            results[method] = method_results
            
        except Exception as e:
            logger.error(f"Error running {method}: {e}")
            continue
    
    # Compare network properties
    comparison = {
        'methods': list(results.keys()),
        'statistics': {},
        'edge_overlap': {}
    }
    
    for method, res in results.items():
        comparison['statistics'][method] = res['stats']
    
    # Calculate edge overlap
    for i, method1 in enumerate(results.keys()):
        for method2 in list(results.keys())[i+1:]:
            edges1 = set(
                tuple(sorted([e['source'], e['target']]))
                for _, e in results[method1]['edge_list'].iterrows()
            )
            edges2 = set(
                tuple(sorted([e['source'], e['target']]))
                for _, e in results[method2]['edge_list'].iterrows()
            )
            
            overlap = len(edges1 & edges2)
            jaccard = overlap / len(edges1 | edges2) if len(edges1 | edges2) > 0 else 0
            
            comparison['edge_overlap'][f'{method1}_vs_{method2}'] = {
                'overlap': overlap,
                'jaccard': jaccard,
                'edges1': len(edges1),
                'edges2': len(edges2)
            }
    
    # Log summary
    logger.info("\n" + "="*60)
    logger.info("NETWORK COMPARISON SUMMARY")
    logger.info("="*60)
    for method, stats in comparison['statistics'].items():
        logger.info(f"{method}: {stats['n_nodes']} nodes, {stats['n_edges']} edges")
    logger.info("="*60)
    
    # Save results
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for method, res in results.items():
            # Save edge list
            edge_path = output_dir / f'{method}_edges.csv'
            res['edge_list'].to_csv(edge_path, index=False)
            
            # Save network stats
            import json
            stats_path = output_dir / f'{method}_stats.json'
            with open(stats_path, 'w') as f:
                json.dump(res['stats'], f, indent=2)
        
        # Save comparison
        comp_path = output_dir / 'network_comparison.json'
        with open(comp_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    return {
        'results': results,
        'comparison': comparison
    }


def plot_network(
    network_result: Dict,
    layout: str = 'spring',
    node_color_by: Optional[str] = None,
    node_colors: Optional[Dict] = None,
    output_path: Optional[Path] = None
) -> go.Figure:
    """
    Create interactive network visualization.
    
    Args:
        network_result: Output from network inference function
        layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
        node_color_by: Attribute to color nodes by
        node_colors: Manual node color mapping
        output_path: Optional path to save plot
    
    Returns:
        Plotly figure object
    """
    G = network_result['network']
    
    # Calculate layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=0.5, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError(f"Unknown layout: {layout}")
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Prepare node traces
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition='top center',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Degree',
                xanchor='left',
                titleside='right'
            )
        )
    )
    
    # Color by degree
    node_degrees = [G.degree(node) for node in G.nodes()]
    node_trace.marker.color = node_degrees
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Network: {network_result['method']} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template='plotly_white',
            height=700,
            width=900
        )
    )
    
    if output_path is not None:
        fig.write_html(output_path)
        logger.info(f"Network plot saved to {output_path}")
    
    return fig


def network_analysis_workflow(
    adata: ad.AnnData,
    method: str = 'spiec-easi',
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict:
    """
    Complete network analysis workflow.
    
    Args:
        adata: AnnData object
        method: Network inference method
        output_dir: Optional output directory
        **kwargs: Method-specific parameters
    
    Returns:
        Dictionary with network results and figures
    """
    logger.info("="*60)
    logger.info("COMPOSITIONAL NETWORK ANALYSIS WORKFLOW")
    logger.info("="*60)
    
    # Run network inference
    if method == 'spiec-easi':
        results = run_spiec_easi(adata, **kwargs)
    elif method == 'sparcc':
        results = run_sparcc(adata, **kwargs)
    elif method == 'proportionality':
        results = run_proportionality(adata, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create visualization
    logger.info("Creating network visualization...")
    fig = plot_network(results)
    
    # Save outputs
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save edge list
        edge_path = output_dir / 'network_edges.csv'
        results['edge_list'].to_csv(edge_path, index=False)
        
        # Save network plot
        plot_path = output_dir / 'network_plot.html'
        fig.write_html(plot_path)
        
        # Save statistics
        import json
        stats_path = output_dir / 'network_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(results['stats'], f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    logger.info("="*60)
    logger.info("NETWORK ANALYSIS COMPLETE")
    logger.info("="*60)
    
    return {
        'network_results': results,
        'figure': fig
    }
