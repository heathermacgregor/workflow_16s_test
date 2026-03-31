"""
Distance matrix computation and processing.

Handles calculation of beta diversity metrics and PCoA ordination.
"""

import multiprocessing
from functools import partial
from pathlib import Path
from typing import List

import anndata as ad
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import pcoa

from workflow_16s.downstream.diversity.beta.plotting import plot_ordination
from workflow_16s.downstream.diversity.beta.statistical_tests import (
    run_mantel_parallel,
    run_permanova_parallel,
    apply_stratified_fdr
)
from workflow_16s.utils.logger import get_logger

def process_distance_matrix(
    dist_matrix: DistanceMatrix,
    dist_name: str,
    level: str,
    adata_agg: ad.AnnData,
    plottable_categorical: List[str],
    plottable_numeric: List[str],
    plot_dir_beta: Path,
    n_cpus: int
) -> int:
    """
    Process a distance matrix through PCoA and statistical testing.

    Performs Principal Coordinates Analysis (PCoA) on the distance matrix,
    runs parallel PERMANOVA and Mantel tests for metadata associations,
    and generates ordination plots.

    Parameters
    ----------
    dist_matrix : DistanceMatrix
        Pairwise distance matrix between samples
    dist_name : str
        Name of the distance metric (e.g., 'braycurtis', 'weighted_unifrac')
    level : str
        Taxonomic level being analyzed
    adata_agg : ad.AnnData
        Aggregated AnnData object to store results
    plottable_categorical : List[str]
        Categorical metadata columns for PERMANOVA
    plottable_numeric : List[str]
        Numeric metadata columns for Mantel tests
    plot_dir_beta : Path
        Output directory for plots
    n_cpus : int
        Number of CPUs for parallel processing

    Returns
    -------
    int
        Number of plots successfully generated

    Notes
    -----
    - Stores PCoA results in adata_agg.obsm[f'X_pcoa_{dist_name}']
    - Stores variance explained in adata_agg.uns[f'pcoa_{dist_name}_variance']
    - Uses multiprocessing for parallel statistical tests
    """
    logger = get_logger("workflow_16s")
    # Perform PCoA
    try:
        pcoa_results = pcoa(dist_matrix)
    except Exception as e:
        logger.error(f"PCoA failed for {dist_name}: {e}")
        return 0

    # Store results in AnnData
    adata_agg.obsm[f'X_pcoa_{dist_name}'] = pcoa_results.samples
    adata_agg.uns[f'pcoa_{dist_name}_variance'] = pcoa_results.proportion_explained

    # Prepare axis labels with variance
    var = pcoa_results.proportion_explained
    axis_labels = {
        'x': f"PC1 ({var[0]:.2%})", # type: ignore
        'y': f"PC2 ({var[1]:.2%})" # type: ignore
    }

    # Run parallel statistical tests
    permanova_results = {}
    mantel_results = {}

    with multiprocessing.Pool(processes=n_cpus) as pool:
        # PERMANOVA for categorical variables
        permanova_partial = partial(
            run_permanova_parallel,
            metadata_df=adata_agg.obs,
            dist_matrix=dist_matrix
        )
        p_results = pool.imap_unordered(permanova_partial, plottable_categorical)

        # Mantel for numeric variables
        mantel_partial = partial(
            run_mantel_parallel,
            metadata_df=adata_agg.obs,
            dist_matrix=dist_matrix
        )
        m_results = pool.imap_unordered(mantel_partial, plottable_numeric)

        # Collect results
        for r in p_results:
            if r:
                permanova_results[r[0]] = r[1]

        for r in m_results:
            if r:
                mantel_results[r[0]] = r[1]

    # **P3 CRITICAL FIX: Apply stratified FDR correction across all tests**
    # This prevents alpha inflation when testing 3-5 distance metrics × 10-20 metadata variables
    permanova_results, mantel_results, stats_summary = apply_stratified_fdr(
        permanova_results, mantel_results, correction_method='fdr_by', alpha=0.05
    )

    # Generate ordination plots
    return plot_ordination(
        adata_agg,
        'PCoA',
        level,
        plottable_categorical,
        plottable_numeric,
        plot_dir_beta,
        dist_name,
        axis_labels,
        permanova_results,
        mantel_results
    )