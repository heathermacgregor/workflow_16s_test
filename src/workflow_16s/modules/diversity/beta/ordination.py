# ==================================================================================== #
#                    downstream/diversity/beta/ordination.py
# ==================================================================================== #

"""
High-level ordination analysis functions.

Provides main entry points for beta diversity, RDA, and trajectory analysis.
"""

import os
import re
from pathlib import Path
from typing import List, Optional

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import scanpy as sc
from scipy.sparse import issparse
from skbio.diversity import beta_diversity
from skbio.stats.ordination import rda
from skbio.tree import TreeNode
from sklearn.preprocessing import StandardScaler

from workflow_16s.downstream.utils import AnalysisUtils
from workflow_16s.visualization.utils import PlottingUtils
from workflow_16s.downstream.diversity.beta.distance_matrix import process_distance_matrix
from workflow_16s.downstream.diversity.beta.plotting import plot_ordination
from workflow_16s.utils.logger import get_logger


def run_beta_diversity_and_stats(
    adata: ad.AnnData,
    analysis_levels: List[str],
    plot_dir_beta: Path,
    tree_path: Optional[Path] = None,
    n_cpus: Optional[int] = None
):
    """
    Calculate beta diversity metrics and perform statistical testing.

    Computes distance matrices (Bray-Curtis, UniFrac), performs PCoA and UMAP,
    and tests associations with metadata using PERMANOVA and Mantel tests.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with microbial abundance data in layers['raw_counts']
    analysis_levels : List[str]
        Taxonomic levels to analyze (e.g., ['Genus', 'Family', 'ASV'])
    plot_dir_beta : Path
        Output directory for beta diversity plots
    tree_path : Optional[Path], optional
        Path to phylogenetic tree file for UniFrac metrics, by default None
    n_cpus : Optional[int], optional
        Number of CPUs for parallel processing, by default None (uses all available)

    Notes
    -----
    - Generates ordination plots colored by metadata variables
    - Performs PERMANOVA for categorical variables
    - Performs Mantel tests for numeric variables
    - Highlights significant results (p < 0.05) in plot filenames
    - UniFrac metrics only computed for ASV level with valid tree
    - Samples with zero total counts are removed before analysis
    """
    logger = get_logger("workflow_16s")
    plot_utils = PlottingUtils(logger)
    # Always use n_cpus if provided, otherwise default to 1 (not os.cpu_count())
    _CPU_COUNT = n_cpus if n_cpus is not None else 1
    logger.info(f"--- Starting Beta Diversity (Using {_CPU_COUNT} CPUs) ---")

    # Load tree if provided
    tree = TreeNode.read(str(tree_path)) if tree_path and tree_path.exists() else None

    # Find plottable metadata
    metadata = AnalysisUtils.find_plottable_metadata(
        adata,
        admin_noise_columns=None,
        fullness_threshold=0.25,
        max_categories=50
    )
    p_cat = [c for c in metadata['categorical'] if c in adata.obs.columns]
    p_num = [c for c in metadata['numeric'] if c in adata.obs.columns]

    total_plots = 0

    # Process each taxonomic level
    for level in analysis_levels:
        adata_agg = (AnalysisUtils.get_analysis_adata(adata, level=level)
                     if level != 'ASV'
                     else adata.copy())

        if adata_agg is None or adata_agg.n_obs < 3:
            continue

        # Get counts and remove zero-count samples
        counts = (adata_agg.layers['raw_counts'].toarray()
                 if issparse(adata_agg.layers['raw_counts'])
                 else adata_agg.layers['raw_counts'])

        zero_mask = counts.sum(axis=1) == 0
        if np.any(zero_mask):
            adata_agg = adata_agg[~zero_mask].copy()
            counts = counts[~zero_mask]

        if adata_agg.n_obs < 3:
            continue

        # Compute UMAP using CLR-transformed data
        # Use try-except for CLR transform in case the helper is missing/renamed
        try:
            adata_clr = adata_agg.copy()
            
            # Use local CLR if method is missing, otherwise use AnalysisUtils
            if hasattr(AnalysisUtils, '_clr_transform'):
                adata_clr.X = AnalysisUtils._clr_transform(adata_clr, pseudocount=1).values
            else:
                # Fallback local CLR
                X = adata_clr.X.toarray() if issparse(adata_clr.X) else adata_clr.X
                X = X + 1
                X_log = np.log(X)
                gm = np.mean(X_log, axis=1, keepdims=True)
                adata_clr.X = X_log - gm

            sc.pp.neighbors(adata_clr, n_neighbors=min(15, adata_clr.n_obs - 1))
            sc.tl.umap(adata_clr)
            adata_agg.obsm['X_umap'] = adata_clr.obsm['X_umap']
        except Exception as e:
            logger.warning(f"UMAP calculation failed for {level}: {e}")

        # Compute distance matrices and ordinations
        for metric in ['braycurtis', 'unweighted_unifrac', 'weighted_unifrac']:
            # Skip UniFrac if not at ASV level or no tree
            if 'unifrac' in metric and (level != 'ASV' or tree is None):
                continue

            try:
                if metric == 'braycurtis':
                    dm = beta_diversity(
                        "braycurtis",
                        counts.astype(int),
                        ids=adata_agg.obs_names.tolist()
                    )
                else:
                    # Filter features to tree tips
                    tips = {t.name for t in tree.tips()}
                    f_ids = [f for f in adata_agg.var_names if f in tips]
                    f_idx = [adata_agg.var_names.tolist().index(f) for f in f_ids]

                    dm = beta_diversity(
                        metric,
                        counts[:, f_idx].astype(int),
                        ids=adata_agg.obs_names.tolist(),
                        tree=tree,
                        otu_ids=f_ids,
                        validate=False
                    )

                total_plots += process_distance_matrix(
                    dm, metric, level, adata_agg,
                    p_cat, p_num, plot_dir_beta, _CPU_COUNT
                )

            except Exception as e:
                logger.error(f"{metric} failed for {level}: {e}")

        # Plot UMAP
        if 'X_umap' in adata_agg.obsm:
            total_plots += plot_ordination(
                adata_agg, 'UMAP', level, p_cat, p_num, plot_dir_beta
            )

    plot_utils.flush_plot_queue()
    logger.info(f"Beta diversity analysis complete. Generated {total_plots} plots.")


def run_constrained_ordination(
    adata: ad.AnnData,
    analysis_levels: List[str],
    plot_dir_beta: Path,
    priority_vars: List[str]
):
    """
    Perform Redundancy Analysis (RDA) to identify metadata-constrained variation.

    RDA is a constrained ordination method that identifies patterns in community
    composition that are explained by environmental/metadata variables. It performs
    multivariate linear regression followed by PCA on the fitted values.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with microbial abundance data
    analysis_levels : List[str]
        Taxonomic levels to analyze
    plot_dir_beta : Path
        Output directory for RDA plots
    priority_vars : List[str]
        Metadata variables to use as constraints (predictors)

    Notes
    -----
    - Uses CLR-transformed abundances as response matrix
    - Categorical variables are one-hot encoded (dropping first category)
    - Numeric variables are standardized (z-scored)
    - Requires at least 3 samples with complete metadata
    - Missing numeric values are imputed with column means
    """
    logger = get_logger("workflow_16s")
    plot_utils = PlottingUtils(logger)
    logger.info("--- Starting Redundancy Analysis (RDA) ---")

    # Filter to available environmental variables
    env_vars = [v for v in priority_vars if v in adata.obs.columns]
    if not env_vars:
        logger.warning("No priority variables found in metadata. Skipping RDA.")
        return

    env_df = adata.obs[env_vars].copy()
    
    # Remove columns that are entirely NaN (e.g., failed weather lookups)
    initial_cols = len(env_df.columns)
    env_df = env_df.dropna(axis=1, how='all')
    removed_cols = initial_cols - len(env_df.columns)
    if removed_cols > 0:
        logger.warning(f"Removed {removed_cols} all-NaN columns from environment matrix: {set(env_vars) - set(env_df.columns)}")
    
    if env_df.empty:
        logger.warning("All environmental variables are NaN. Skipping RDA.")
        return

    # One-hot encode categorical variables
    cat_vars = [v for v in env_vars
                if pd.api.types.is_categorical_dtype(env_df[v]) or 
                   pd.api.types.is_object_dtype(env_df[v])]

    if cat_vars:
        env_df = pd.get_dummies(
            env_df,
            columns=cat_vars,
            drop_first=True,
            dummy_na=True,
            dtype=float
        )

    # Standardize numeric variables
    # Filter to only columns that actually exist in the dataframe
    existing_vars = [v for v in env_vars if v in env_df.columns]
    # Exclude boolean columns to avoid dtype errors when filling with mean
    num_vars = [
        v for v in existing_vars 
        if pd.api.types.is_numeric_dtype(env_df[v]) and not pd.api.types.is_bool_dtype(env_df[v])
    ]
    if num_vars:
        # Handle nullable Int64 dtypes - convert to float before fillna
        filled_data = env_df[num_vars].copy()
        for col in num_vars:
            if pd.api.types.is_integer_dtype(filled_data[col]):
                # Convert Int64 to float64 to allow float means
                filled_data[col] = filled_data[col].astype('float64')
        
        # Now fillna with mean (which is float)
        filled_data = filled_data.fillna(filled_data.mean())
        
        # Standardize
        env_df[num_vars] = StandardScaler().fit_transform(filled_data)

    # Remove samples with missing data
    env_df_before = len(env_df)
    env_df = env_df.dropna()
    env_df_after = len(env_df)
    
    if env_df_before > env_df_after:
        logger.warning(f"Metadata completeness: Removed {env_df_before - env_df_after} samples with NaN→ {env_df_after} remaining")

    if env_df.shape[0] < 3:
        logger.warning("Insufficient samples with complete metadata. Skipping RDA.")
        return

    # Perform RDA for each taxonomic level
    for level in analysis_levels:
        adata_agg = AnalysisUtils.get_analysis_adata(adata, level=level)

        if adata_agg is None or adata_agg.n_obs < 5:
            continue

        # CLR transform
        # Use local fallback if needed
        if hasattr(AnalysisUtils, '_clr_transform'):
            clr_df = AnalysisUtils._clr_transform(adata_agg, pseudocount=1)
        else:
            X = adata_agg.X.toarray() if issparse(adata_agg.X) else adata_agg.X
            X = X + 1
            X_log = np.log(X)
            gm = np.mean(X_log, axis=1, keepdims=True)
            clr_df = pd.DataFrame(X_log - gm, index=adata_agg.obs_names, columns=adata_agg.var_names)

        # Find common samples
        common = clr_df.index.intersection(env_df.index)
        if len(common) < 3:
            logger.warning(f"Insufficient overlap for RDA at {level}: clr has {len(clr_df)} samples, env_df has {len(env_df)} samples, common={len(common)}. Skipping.")
            continue

        # Run RDA
        try:
            rda_res = rda(
                y=clr_df.loc[common],
                x=env_df.loc[common],
                scale_Y=False
            )
        except Exception as e:
            logger.error(f"RDA failed for {level}: {e}")
            continue

        # Prepare plotting dataframe
        plot_df = (pd.DataFrame(
            rda_res.samples,
            index=common,
            columns=['RDA1', 'RDA2']
        ).join(adata_agg.obs))

        # Generate plots for each priority variable
        for col in priority_vars:
            if col not in plot_df.columns:
                continue

            color_col = col if col in (cat_vars + num_vars) else None

            fig = px.scatter(
                plot_df,
                x='RDA1',
                y='RDA2',
                title=f"RDA ({level}) by {col}",
                color=color_col,
                opacity=0.7
            )

            plot_utils.save_plotly_fig(
                fig,
                plot_dir_beta / level / f"RDA_vs_{re.sub(r'[^A-Za-z0-9_]+', '', col)}",
                batch=True
            )

    plot_utils.flush_plot_queue()
    logger.info("RDA analysis complete.")


def run_trajectory_analysis(adata: ad.AnnData, plot_dir: Path):
    """
    Perform trajectory analysis using PAGA to map community transitions.

    Uses Partition-based Graph Abstraction (PAGA) to identify connectivity between
    microbial community states and visualize potential transition pathways along
    environmental gradients. PAGA constructs an abstracted graph where nodes represent
    community clusters and edges represent transition probabilities.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with microbial abundance data and computed neighbors graph
    plot_dir : Path
        Output directory for trajectory plots

    Notes
    -----
    - Uses Leiden clustering (resolution=0.5) to define community groups
    - PAGA threshold of 0.1 filters weak connections
    - Generates two visualizations:
        1. Abstract PAGA graph showing community connectivity
        2. PAGA overlaid on UMAP colored by facility_distance_km (if available)
    - Requires precomputed neighbors graph in adata
    """
    logger = get_logger("workflow_16s")
    plot_utils = PlottingUtils(logger)
    logger.info("--- Starting Trajectory Analysis (PAGA) ---")

    # Preprocessing for PAGA
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
    sc.tl.leiden(adata, resolution=0.5, key_added='groups')

    # Run PAGA
    sc.tl.paga(adata, groups='groups')

    # Generate abstract PAGA plot
    fig, ax = plt.subplots(figsize=(8, 8))
    sc.pl.paga(
        adata,
        threshold=0.1,
        show=False,
        ax=ax,
        title="Microbial Community Trajectory (PAGA)"
    )

    plt.savefig(
        plot_dir / "community_trajectory_paga.png",
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()

    # Color by distance gradient if available
    if 'facility_distance_km' in adata.obs.columns:
        sc.pl.paga_compare(
            adata,
            basis='umap',
            color='facility_distance_km',
            show=False
        )
        plt.savefig(
            plot_dir / "community_trajectory_paga_distance.png",
            dpi=150,
            bbox_inches='tight'
        )
        plt.close()
        logger.info("Generated PAGA plot colored by facility distance")

    logger.info("Trajectory Analysis (PAGA) complete.")