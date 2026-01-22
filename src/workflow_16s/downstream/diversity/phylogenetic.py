"""
Phylogenetic Diversity Metrics for 16S rRNA Analysis.

This module provides phylogenetically-informed diversity metrics that account
for the evolutionary relationships between sequences. These metrics are generally
more powerful than taxonomy-independent metrics because they leverage phylogenetic
information.

Key Metrics:
1. Faith's Phylogenetic Diversity (PD) - Alpha diversity
2. UniFrac distances - Beta diversity (weighted and unweighted)
3. Phylogenetic entropy

References:
    Faith DP. (1992). Conservation evaluation and phylogenetic diversity.
    Biological Conservation, 61(1), 1-10.
    
    Lozupone C, Knight R. (2005). UniFrac: a new phylogenetic method for comparing
    microbial communities. Applied and Environmental Microbiology, 71(12), 8228-8235.
    
    Lozupone CA, Hamady M, Kelley ST, Knight R. (2007). Quantitative and qualitative
    beta diversity measures lead to different insights into factors that structure
    microbial communities. Applied and Environmental Microbiology, 73(5), 1576-1585.

Example:
    >>> from workflow_16s.downstream.phylogenetic_diversity import (
    ...     calculate_faith_pd,
    ...     calculate_unifrac
    ... )
    >>> 
    >>> # Calculate Faith's PD
    >>> pd_values = calculate_faith_pd(adata, tree)
    >>> 
    >>> # Calculate UniFrac distances
    >>> unifrac_dm = calculate_unifrac(adata, tree, weighted=True)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import squareform
from skbio.tree import TreeNode
from skbio import DistanceMatrix
from skbio.diversity import alpha_diversity, beta_diversity

from workflow_16s.utils.logger import get_logger

logger = get_logger("workflow_16s")


def load_tree(tree_path: Union[str, Path]) -> TreeNode:
    """
    Load a phylogenetic tree from file.
    
    Supports Newick format (.nwk, .tree, .tre).
    
    Args:
        tree_path: Path to tree file
    
    Returns:
        TreeNode object from scikit-bio
    """
    tree_path = Path(tree_path)
    
    if not tree_path.exists():
        raise FileNotFoundError(f"Tree file not found: {tree_path}")
    
    logger.info(f"Loading phylogenetic tree from {tree_path}")
    tree = TreeNode.read(str(tree_path))
    
    # Validate tree
    n_tips = len([node for node in tree.tips()])
    logger.info(f"Tree loaded: {n_tips} tips")
    
    return tree


def calculate_faith_pd(
    adata: ad.AnnData,
    tree: Union[TreeNode, str, Path],
    feature_id_column: Optional[str] = None
) -> pd.Series:
    """
    Calculate Faith's Phylogenetic Diversity (PD) for each sample.
    
    Faith's PD is the sum of branch lengths for all branches in the tree
    that are ancestral to observed features in a sample. It measures the
    total evolutionary history represented in a community.
    
    Args:
        adata: AnnData object with feature table (samples × features)
        tree: TreeNode object or path to tree file
        feature_id_column: Column in adata.var with feature IDs matching tree tips
                          (if None, uses adata.var_names)
    
    Returns:
        Series with Faith's PD values for each sample
    """
    # Load tree if path provided
    if isinstance(tree, (str, Path)):
        tree = load_tree(tree)
    
    # Get feature IDs
    if feature_id_column is not None:
        feature_ids = adata.var[feature_id_column].values
    else:
        feature_ids = adata.var_names.values
    
    # Prepare counts table
    counts = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    counts_df = pd.DataFrame(
        counts.T,
        index=feature_ids,
        columns=adata.obs_names
    )
    
    # Calculate Faith's PD using scikit-bio
    logger.info("Calculating Faith's Phylogenetic Diversity...")
    try:
        pd_values = alpha_diversity('faith_pd', counts_df.T, tree=tree, ids=adata.obs_names)
    except Exception as e:
        logger.error(f"Error calculating Faith's PD: {e}")
        logger.info("Checking for feature ID mismatches between tree and table...")
        
        tree_tips = {tip.name for tip in tree.tips()}
        table_features = set(feature_ids)
        
        missing_in_tree = table_features - tree_tips
        missing_in_table = tree_tips - table_features
        
        if missing_in_tree:
            logger.warning(f"Features in table but not in tree: {len(missing_in_tree)}")
            logger.debug(f"Missing features: {list(missing_in_tree)[:10]}")
        if missing_in_table:
            logger.warning(f"Tips in tree but not in table: {len(missing_in_table)}")
        
        raise ValueError(
            f"Feature IDs don't match tree tips. "
            f"Missing in tree: {len(missing_in_tree)}, "
            f"Missing in table: {len(missing_in_table)}"
        )
    
    logger.info(f"Faith's PD calculated for {len(pd_values)} samples")
    logger.info(f"PD range: {pd_values.min():.2f} - {pd_values.max():.2f}")
    
    return pd_values


def calculate_unifrac(
    adata: ad.AnnData,
    tree: Union[TreeNode, str, Path],
    weighted: bool = True,
    normalized: bool = True,
    feature_id_column: Optional[str] = None,
    threads: int = 1
) -> DistanceMatrix:
    """
    Calculate UniFrac distances between samples.
    
    UniFrac quantifies the phylogenetic distance between communities by
    measuring the fraction of unique branch length.
    
    - Unweighted UniFrac: Considers only presence/absence
    - Weighted UniFrac: Considers relative abundances
    
    Args:
        adata: AnnData object with feature table
        tree: TreeNode object or path to tree file
        weighted: If True, use weighted UniFrac (considers abundances)
        normalized: If True, normalize distances to [0, 1]
        feature_id_column: Column in adata.var with feature IDs
        threads: Number of threads for calculation (scikit-bio >= 0.5.7)
    
    Returns:
        DistanceMatrix with pairwise UniFrac distances
    """
    # Load tree if path provided
    if isinstance(tree, (str, Path)):
        tree = load_tree(tree)
    
    # Get feature IDs
    if feature_id_column is not None:
        feature_ids = adata.var[feature_id_column].values
    else:
        feature_ids = adata.var_names.values
    
    # Prepare counts table
    counts = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    counts_df = pd.DataFrame(
        counts.T,
        index=feature_ids,
        columns=adata.obs_names
    )
    
    # Select metric
    metric = 'weighted_unifrac' if weighted else 'unweighted_unifrac'
    
    logger.info(f"Calculating {metric} (normalized={normalized})...")
    
    try:
        # Calculate UniFrac
        unifrac_dm = beta_diversity(
            metric,
            counts_df.T,
            tree=tree,
            ids=adata.obs_names,
            normalized=normalized
        )
    except Exception as e:
        logger.error(f"Error calculating UniFrac: {e}")
        
        # Debug information
        tree_tips = {tip.name for tip in tree.tips()}
        table_features = set(feature_ids)
        missing_in_tree = table_features - tree_tips
        
        if missing_in_tree:
            logger.warning(f"Features in table but not in tree: {len(missing_in_tree)}")
        
        raise ValueError(
            f"Failed to calculate UniFrac. "
            f"Check that feature IDs match tree tips."
        )
    
    logger.info(f"UniFrac calculated: {unifrac_dm.shape[0]} samples")
    logger.info(f"Distance range: {unifrac_dm.data.min():.4f} - {unifrac_dm.data.max():.4f}")
    
    return unifrac_dm


def calculate_phylogenetic_entropy(
    adata: ad.AnnData,
    tree: Union[TreeNode, str, Path],
    feature_id_column: Optional[str] = None
) -> pd.Series:
    """
    Calculate phylogenetic entropy for each sample.
    
    Phylogenetic entropy extends Shannon entropy by weighting each species
    by its distance from the root of the phylogenetic tree.
    
    Args:
        adata: AnnData object with feature table
        tree: TreeNode object or path to tree file
        feature_id_column: Column in adata.var with feature IDs
    
    Returns:
        Series with phylogenetic entropy values
    """
    # Load tree if path provided
    if isinstance(tree, (str, Path)):
        tree = load_tree(tree)
    
    # Get feature IDs
    if feature_id_column is not None:
        feature_ids = adata.var[feature_id_column].values
    else:
        feature_ids = adata.var_names.values
    
    # Get counts and convert to relative abundances
    counts = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    rel_abund = counts / counts.sum(axis=1, keepdims=True)
    
    # Calculate phylogenetic distances from root
    root = tree
    tip_distances = {}
    for tip in tree.tips():
        if tip.name in feature_ids:
            tip_distances[tip.name] = tip.distance(root)
    
    # Calculate phylogenetic entropy for each sample
    entropies = []
    for i, sample in enumerate(adata.obs_names):
        entropy = 0.0
        for j, feature in enumerate(feature_ids):
            if feature in tip_distances and rel_abund[i, j] > 0:
                p = rel_abund[i, j]
                d = tip_distances[feature]
                entropy += -p * np.log(p) * d
        entropies.append(entropy)
    
    entropy_series = pd.Series(entropies, index=adata.obs_names, name='phylogenetic_entropy')
    
    logger.info(f"Phylogenetic entropy calculated for {len(entropy_series)} samples")
    logger.info(f"Entropy range: {entropy_series.min():.4f} - {entropy_series.max():.4f}")
    
    return entropy_series


def prune_tree_to_features(
    tree: TreeNode,
    feature_ids: List[str]
) -> TreeNode:
    """
    Prune phylogenetic tree to only include specified features.
    
    Args:
        tree: Full phylogenetic tree
        feature_ids: List of feature IDs to keep
    
    Returns:
        Pruned tree containing only specified features
    """
    logger.info(f"Pruning tree to {len(feature_ids)} features...")
    
    # Get current tree tips
    tree_tips = {tip.name for tip in tree.tips()}
    
    # Find tips to keep
    tips_to_keep = set(feature_ids) & tree_tips
    
    if len(tips_to_keep) == 0:
        raise ValueError("No matching features between tree and feature list")
    
    logger.info(f"Keeping {len(tips_to_keep)}/{len(tree_tips)} tree tips")
    
    # Shear tree to keep only desired tips
    pruned_tree = tree.shear(tips_to_keep)
    
    return pruned_tree


def build_tree_fasttree(
    sequences_path: Union[str, Path],
    output_path: Union[str, Path],
    aligned: bool = False,
    model: str = 'gtr',
    threads: int = 1
) -> TreeNode:
    """
    Build phylogenetic tree using FastTree.
    
    Args:
        sequences_path: Path to FASTA file with sequences
        output_path: Path to save output tree
        aligned: Whether sequences are already aligned
        model: Nucleotide substitution model ('gtr' or 'jc')
        threads: Number of OpenMP threads
    
    Returns:
        TreeNode object
    
    Raises:
        RuntimeError: If FastTree is not installed
    """
    import subprocess
    
    sequences_path = Path(sequences_path)
    output_path = Path(output_path)
    
    # Check if FastTree is available
    try:
        subprocess.run(['FastTree', '-h'], capture_output=True, check=False)
    except FileNotFoundError:
        raise RuntimeError(
            "FastTree not found. Install with:\n"
            "  conda install -c bioconda fasttree"
        )
    
    logger.info(f"Building tree with FastTree (model={model}, threads={threads})")
    
    # Build command
    cmd = ['FastTree']
    
    if model == 'gtr':
        cmd.append('-gtr')
    
    if not aligned:
        logger.warning("Sequences not aligned. FastTree requires aligned sequences.")
        logger.info("Consider using MAFFT or MUSCLE for alignment first.")
    
    cmd.extend(['-nt', '-quiet'])
    
    # Set threads via environment variable
    import os
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(threads)
    
    # Run FastTree
    with open(sequences_path, 'r') as infile, open(output_path, 'w') as outfile:
        result = subprocess.run(
            cmd,
            stdin=infile,
            stdout=outfile,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )
    
    if result.returncode != 0:
        raise RuntimeError(f"FastTree failed: {result.stderr}")
    
    logger.info(f"Tree saved to {output_path}")
    
    # Load and return tree
    tree = TreeNode.read(str(output_path))
    
    return tree


def insert_sequences_sepp(
    query_sequences_path: Union[str, Path],
    reference_tree: Union[str, Path],
    reference_alignment: Union[str, Path],
    output_dir: Union[str, Path],
    threads: int = 1
) -> TreeNode:
    """
    Insert query sequences into reference tree using SEPP.
    
    SEPP (SATé-enabled phylogenetic placement) is useful when you have
    a reference tree (e.g., SILVA) and want to place new sequences.
    
    Args:
        query_sequences_path: Path to query sequences (FASTA)
        reference_tree: Path to reference tree (Newick)
        reference_alignment: Path to reference alignment (FASTA)
        output_dir: Directory for SEPP output
        threads: Number of threads
    
    Returns:
        TreeNode with inserted sequences
    
    Raises:
        RuntimeError: If SEPP is not installed
    """
    import subprocess
    
    query_sequences_path = Path(query_sequences_path)
    reference_tree = Path(reference_tree)
    reference_alignment = Path(reference_alignment)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if SEPP is available
    try:
        subprocess.run(['run_sepp.py', '-h'], capture_output=True, check=False)
    except FileNotFoundError:
        raise RuntimeError(
            "SEPP not found. Install with:\n"
            "  conda install -c bioconda sepp"
        )
    
    logger.info("Inserting sequences with SEPP...")
    
    # Build command
    cmd = [
        'run_sepp.py',
        '-t', str(reference_tree),
        '-a', str(reference_alignment),
        '-f', str(query_sequences_path),
        '-o', 'sepp_output',
        '-d', str(output_dir),
        '-x', str(threads)
    ]
    
    # Run SEPP
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"SEPP failed: {result.stderr}")
    
    # Load output tree
    output_tree_path = output_dir / 'sepp_output_placement.tog.tre'
    if not output_tree_path.exists():
        raise RuntimeError(f"SEPP output tree not found: {output_tree_path}")
    
    logger.info(f"SEPP completed. Tree saved to {output_tree_path}")
    
    tree = TreeNode.read(str(output_tree_path))
    
    return tree


def plot_tree(
    tree: TreeNode,
    output_path: Optional[Path] = None,
    max_tips: int = 100
) -> go.Figure:
    """
    Create an interactive visualization of a phylogenetic tree.
    
    Args:
        tree: TreeNode object
        output_path: Optional path to save HTML plot
        max_tips: Maximum number of tips to display (for performance)
    
    Returns:
        Plotly figure object
    """
    from skbio.tree import TreeNode
    
    # Get tree tips
    tips = list(tree.tips())
    n_tips = len(tips)
    
    if n_tips > max_tips:
        logger.warning(
            f"Tree has {n_tips} tips, which may be slow to render. "
            f"Limiting to {max_tips} tips. Use max_tips parameter to change."
        )
        # Sample random tips
        import random
        random.seed(42)
        keep_tips = random.sample([tip.name for tip in tips], max_tips)
        tree = tree.shear(keep_tips)
        tips = list(tree.tips())
    
    # Convert tree to coordinates using rectangular layout
    def get_tree_coords(node, x=0, y=0, y_step=1):
        """Recursively compute x, y coordinates for tree nodes."""
        coords = {}
        edges = []
        
        if node.is_tip():
            coords[node.name or id(node)] = (x, y)
            return coords, edges, y + y_step
        
        # Process children
        child_y = y
        child_coords_list = []
        for child in node.children:
            child_coords, child_edges, child_y = get_tree_coords(
                child, x + (child.length or 1), child_y, y_step
            )
            coords.update(child_coords)
            edges.extend(child_edges)
            child_coords_list.append(child_coords)
        
        # Parent position is average of children
        child_y_vals = [list(c.values())[0][1] for c in child_coords_list]
        parent_y = np.mean(child_y_vals)
        parent_name = node.name or id(node)
        coords[parent_name] = (x, parent_y)
        
        # Add edges to children
        for child in node.children:
            child_name = child.name or id(child)
            edges.append((parent_name, child_name))
        
        return coords, edges, child_y
    
    coords, edges, _ = get_tree_coords(tree)
    
    # Prepare data for plotting
    edge_x = []
    edge_y = []
    for parent, child in edges:
        x0, y0 = coords[parent]
        x1, y1 = coords[child]
        # Rectangular tree: horizontal then vertical
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(color='black', width=1),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add tip labels
    tip_x = [coords[tip.name][0] for tip in tips]
    tip_y = [coords[tip.name][1] for tip in tips]
    tip_names = [tip.name for tip in tips]
    
    fig.add_trace(go.Scatter(
        x=tip_x,
        y=tip_y,
        mode='markers+text',
        marker=dict(size=5, color='blue'),
        text=tip_names,
        textposition='middle right',
        textfont=dict(size=8),
        hovertext=tip_names,
        showlegend=False
    ))
    
    fig.update_layout(
        title=f'Phylogenetic Tree ({len(tips)} tips)',
        xaxis=dict(title='Branch Length', showgrid=False),
        yaxis=dict(title='', showticklabels=False, showgrid=False),
        template='plotly_white',
        height=max(600, len(tips) * 10),
        width=1000
    )
    
    if output_path is not None:
        fig.write_html(output_path)
        logger.info(f"Tree visualization saved to {output_path}")
    
    return fig


def phylogenetic_diversity_workflow(
    adata: ad.AnnData,
    tree: Union[TreeNode, str, Path],
    calculate_pd: bool = True,
    calculate_wunifrac: bool = True,
    calculate_uunifrac: bool = False,
    feature_id_column: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Complete phylogenetic diversity analysis workflow.
    
    Args:
        adata: AnnData object with feature table
        tree: TreeNode or path to tree file
        calculate_pd: Calculate Faith's PD
        calculate_wunifrac: Calculate weighted UniFrac
        calculate_uunifrac: Calculate unweighted UniFrac
        feature_id_column: Column with feature IDs matching tree
        output_dir: Optional directory for outputs
    
    Returns:
        Dictionary with:
            - 'faith_pd': Faith's PD values (if requested)
            - 'weighted_unifrac': Weighted UniFrac distance matrix (if requested)
            - 'unweighted_unifrac': Unweighted UniFrac distance matrix (if requested)
            - 'tree': Loaded tree object
    """
    logger.info("="*60)
    logger.info("PHYLOGENETIC DIVERSITY WORKFLOW")
    logger.info("="*60)
    
    # Load tree
    if isinstance(tree, (str, Path)):
        tree = load_tree(tree)
    
    results = {'tree': tree}
    
    # Calculate Faith's PD
    if calculate_pd:
        logger.info("Step 1: Calculating Faith's Phylogenetic Diversity...")
        pd_values = calculate_faith_pd(adata, tree, feature_id_column)
        results['faith_pd'] = pd_values
        
        # Add to adata
        adata.obs['faith_pd'] = pd_values
        logger.info(f"Faith's PD added to adata.obs['faith_pd']")
    
    # Calculate weighted UniFrac
    if calculate_wunifrac:
        logger.info("Step 2: Calculating weighted UniFrac...")
        wunifrac_dm = calculate_unifrac(
            adata, tree, weighted=True, feature_id_column=feature_id_column
        )
        results['weighted_unifrac'] = wunifrac_dm
        
        # Add to adata
        adata.uns['weighted_unifrac'] = wunifrac_dm.data
        logger.info(f"Weighted UniFrac added to adata.uns['weighted_unifrac']")
    
    # Calculate unweighted UniFrac
    if calculate_uunifrac:
        logger.info("Step 3: Calculating unweighted UniFrac...")
        uunifrac_dm = calculate_unifrac(
            adata, tree, weighted=False, feature_id_column=feature_id_column
        )
        results['unweighted_unifrac'] = uunifrac_dm
        
        # Add to adata
        adata.uns['unweighted_unifrac'] = uunifrac_dm.data
        logger.info(f"Unweighted UniFrac added to adata.uns['unweighted_unifrac']")
    
    # Save outputs
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if 'faith_pd' in results:
            pd_path = output_dir / 'faith_pd.csv'
            results['faith_pd'].to_csv(pd_path)
            logger.info(f"Faith's PD saved to {pd_path}")
        
        if 'weighted_unifrac' in results:
            wunifrac_path = output_dir / 'weighted_unifrac.tsv'
            results['weighted_unifrac'].write(str(wunifrac_path))
            logger.info(f"Weighted UniFrac saved to {wunifrac_path}")
        
        if 'unweighted_unifrac' in results:
            uunifrac_path = output_dir / 'unweighted_unifrac.tsv'
            results['unweighted_unifrac'].write(str(uunifrac_path))
            logger.info(f"Unweighted UniFrac saved to {uunifrac_path}")
    
    logger.info("="*60)
    logger.info("PHYLOGENETIC DIVERSITY WORKFLOW COMPLETE")
    logger.info("="*60)
    
    return results
