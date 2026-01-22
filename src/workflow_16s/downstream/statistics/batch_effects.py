"""
Batch Effect Diagnostics and Correction
========================================

Tools for identifying, visualizing, and correcting batch effects in
microbiome data. Batch effects can arise from different sequencing runs,
DNA extraction kits, storage conditions, or processing dates.

References:
    - Gibbons et al. (2018). Correcting for batch effects in microbiome data.
    - Jiang et al. (2020). ConQuR: batch effect removal for microbiome data.
"""

# ===================================== IMPORTS ====================================== #

import logging
from pathlib import Path
from typing import Union, Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import mannwhitneyu, kruskal
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from skbio.stats.distance import permanova, DistanceMatrix
import anndata as ad

from workflow_16s import constants
from workflow_16s.downstream.utils import safe_write_h5ad

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ============================== BATCH DIAGNOSTICS =================================== #

def detect_batch_effects(
    adata: ad.AnnData,
    batch_col: str = 'batch',
    biology_col: Optional[str] = None,
    distance_metric: str = 'braycurtis',
    n_permutations: int = 999
) -> Dict[str, any]:
    """
    Comprehensive batch effect detection using multiple methods.
    
    Tests:
        1. PERMANOVA: Variance explained by batch vs. biology
        2. Silhouette coefficient: Batch separation score
        3. PCA visualization: Visual batch clustering
        4. Entropy: Distribution uniformity across batches
    
    Args:
        adata: AnnData object with count data
        batch_col: Column name for batch variable
        biology_col: Column name for biological variable (optional)
        distance_metric: Distance metric ('braycurtis', 'euclidean', 'jaccard')
        n_permutations: Number of PERMANOVA permutations
    
    Returns:
        Dictionary with diagnostic results
    """
    if batch_col not in adata.obs.columns:
        raise ValueError(f"Batch column '{batch_col}' not found in adata.obs")
    
    results = {}
    
    # 1. PERMANOVA for batch effect
    logger.info(f"Running PERMANOVA to quantify batch variance...")
    try:
        from skbio.diversity import beta_diversity
        
        # Calculate distance matrix
        dm = beta_diversity(
            distance_metric,
            adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
            ids=adata.obs_names
        )
        
        # Test batch effect
        batch_permanova = permanova(
            dm,
            adata.obs[batch_col],
            permutations=n_permutations
        )
        results['batch_permanova'] = {
            'test_statistic': batch_permanova['test statistic'],
            'p_value': batch_permanova['p-value'],
            'r_squared': batch_permanova.get('R2', np.nan)
        }
        
        # Test biology effect if provided
        if biology_col and biology_col in adata.obs.columns:
            bio_permanova = permanova(
                dm,
                adata.obs[biology_col],
                permutations=n_permutations
            )
            results['biology_permanova'] = {
                'test_statistic': bio_permanova['test statistic'],
                'p_value': bio_permanova['p-value'],
                'r_squared': bio_permanova.get('R2', np.nan)
            }
            
            # Ratio of batch to biology signal
            batch_r2 = results['batch_permanova']['r_squared']
            bio_r2 = results['biology_permanova']['r_squared']
            results['batch_to_biology_ratio'] = batch_r2 / bio_r2 if bio_r2 > 0 else np.inf
            
    except Exception as e:
        logger.error(f"PERMANOVA failed: {e}")
        results['batch_permanova'] = None
    
    # 2. Silhouette coefficient
    logger.info("Calculating silhouette coefficients...")
    try:
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        labels = pd.Categorical(adata.obs[batch_col]).codes
        
        # Overall silhouette
        sil_score = silhouette_score(X, labels, metric='euclidean')
        results['silhouette_score'] = sil_score
        
        # Per-sample silhouette
        sil_samples = silhouette_samples(X, labels, metric='euclidean')
        results['silhouette_samples'] = sil_samples
        
    except Exception as e:
        logger.error(f"Silhouette calculation failed: {e}")
        results['silhouette_score'] = None
    
    # 3. PCA variance explained by batch
    logger.info("Running PCA for batch visualization...")
    try:
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        pca = PCA(n_components=min(10, X.shape[1]))
        pca_coords = pca.fit_transform(X)
        
        results['pca_coords'] = pca_coords
        results['pca_variance_ratio'] = pca.explained_variance_ratio_
        
        # Calculate R² for batch on each PC
        batch_r2_per_pc = []
        for pc_idx in range(min(5, pca_coords.shape[1])):
            pc_values = pca_coords[:, pc_idx]
            batches = adata.obs[batch_col]
            
            # ANOVA-like R²
            total_var = np.var(pc_values)
            batch_means = pd.DataFrame({
                'pc': pc_values,
                'batch': batches
            }).groupby('batch')['pc'].mean()
            
            between_var = np.sum([
                len(batches[batches == b]) * (batch_means[b] - np.mean(pc_values))**2
                for b in batch_means.index
            ]) / len(pc_values)
            
            r2 = between_var / total_var if total_var > 0 else 0
            batch_r2_per_pc.append(r2)
        
        results['batch_r2_per_pc'] = batch_r2_per_pc
        
    except Exception as e:
        logger.error(f"PCA calculation failed: {e}")
        results['pca_coords'] = None
    
    # 4. Entropy-based uniformity test
    logger.info("Calculating batch entropy...")
    try:
        batch_counts = adata.obs[batch_col].value_counts()
        proportions = batch_counts / len(adata)
        entropy = -np.sum(proportions * np.log(proportions + 1e-10))
        max_entropy = np.log(len(batch_counts))
        
        results['batch_entropy'] = entropy
        results['max_entropy'] = max_entropy
        results['entropy_ratio'] = entropy / max_entropy if max_entropy > 0 else 0
        
    except Exception as e:
        logger.error(f"Entropy calculation failed: {e}")
        results['batch_entropy'] = None
    
    # Interpretation
    results['interpretation'] = _interpret_batch_results(results)
    
    return results


def _interpret_batch_results(results: Dict) -> str:
    """Generate human-readable interpretation of batch effect tests."""
    
    messages = []
    
    # PERMANOVA interpretation
    if results.get('batch_permanova'):
        p_val = results['batch_permanova']['p_value']
        r2 = results['batch_permanova']['r_squared']
        
        if p_val < 0.001:
            messages.append(f"🔴 STRONG batch effect detected (p < 0.001, R² = {r2:.3f})")
        elif p_val < 0.05:
            messages.append(f"🟠 MODERATE batch effect detected (p = {p_val:.3f}, R² = {r2:.3f})")
        else:
            messages.append(f"🟢 No significant batch effect (p = {p_val:.3f})")
    
    # Batch vs biology comparison
    if 'batch_to_biology_ratio' in results:
        ratio = results['batch_to_biology_ratio']
        if ratio > 1.0:
            messages.append(f"⚠️  Batch signal STRONGER than biological signal (ratio = {ratio:.2f})")
        elif ratio > 0.5:
            messages.append(f"⚠️  Batch signal comparable to biological signal (ratio = {ratio:.2f})")
        else:
            messages.append(f"✓ Biological signal stronger than batch (ratio = {ratio:.2f})")
    
    # Silhouette interpretation
    if results.get('silhouette_score') is not None:
        sil = results['silhouette_score']
        if sil > 0.7:
            messages.append(f"🔴 Very high batch clustering (silhouette = {sil:.3f})")
        elif sil > 0.5:
            messages.append(f"🟠 Moderate batch clustering (silhouette = {sil:.3f})")
        elif sil > 0.25:
            messages.append(f"🟡 Weak batch clustering (silhouette = {sil:.3f})")
        else:
            messages.append(f"🟢 No batch clustering (silhouette = {sil:.3f})")
    
    # PCA interpretation
    if results.get('batch_r2_per_pc'):
        max_r2 = max(results['batch_r2_per_pc'][:3])  # First 3 PCs
        if max_r2 > 0.5:
            messages.append(f"🔴 Batch explains {max_r2:.1%} of variance in top PCs")
        elif max_r2 > 0.25:
            messages.append(f"🟠 Batch explains {max_r2:.1%} of variance in top PCs")
        else:
            messages.append(f"🟢 Batch explains only {max_r2:.1%} of variance in top PCs")
    
    return "\n".join(messages)


# ============================ BATCH VISUALIZATIONS ================================== #

def plot_batch_pca(
    adata: ad.AnnData,
    batch_col: str = 'batch',
    biology_col: Optional[str] = None,
    output_path: Optional[Path] = None,
    title: str = "PCA: Batch Effect Visualization"
) -> go.Figure:
    """
    Create interactive PCA plot colored by batch and biology.
    
    Args:
        adata: AnnData object
        batch_col: Column for batch labels
        biology_col: Column for biological labels
        output_path: Where to save HTML plot
        title: Plot title
    
    Returns:
        Plotly figure
    """
    # Run PCA
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    pca = PCA(n_components=min(10, X.shape[1]))
    pca_coords = pca.fit_transform(X)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'PC1': pca_coords[:, 0],
        'PC2': pca_coords[:, 1],
        'PC3': pca_coords[:, 2] if pca_coords.shape[1] > 2 else 0,
        'Batch': adata.obs[batch_col].astype(str),
        'Sample': adata.obs_names
    })
    
    if biology_col and biology_col in adata.obs.columns:
        plot_df['Biology'] = adata.obs[biology_col].astype(str)
    
    # Create figure
    var_explained = pca.explained_variance_ratio_
    
    fig = px.scatter(
        plot_df,
        x='PC1',
        y='PC2',
        color='Batch',
        symbol='Biology' if biology_col else None,
        hover_data=['Sample'],
        title=title,
        labels={
            'PC1': f'PC1 ({var_explained[0]:.1%} var)',
            'PC2': f'PC2 ({var_explained[1]:.1%} var)'
        }
    )
    
    fig.update_layout(
        width=900,
        height=700,
        font=dict(size=12)
    )
    
    if output_path:
        fig.write_html(output_path)
        logger.info(f"Saved batch PCA plot to {output_path}")
    
    return fig


def plot_silhouette_analysis(
    adata: ad.AnnData,
    batch_col: str = 'batch',
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create silhouette plot showing batch separation quality.
    
    Silhouette values near +1 indicate strong batch clustering (BAD).
    Values near 0 indicate overlapping batches (GOOD for batch correction).
    
    Args:
        adata: AnnData object
        batch_col: Column for batch labels
        output_path: Where to save figure
    
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import silhouette_samples, silhouette_score
    
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    batch_labels = pd.Categorical(adata.obs[batch_col])
    cluster_labels = batch_labels.codes
    
    # Compute silhouette scores
    silhouette_avg = silhouette_score(X, cluster_labels, metric='euclidean')
    sample_silhouette_values = silhouette_samples(X, cluster_labels, metric='euclidean')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    y_lower = 10
    n_clusters = len(batch_labels.categories)
    
    for i, batch_name in enumerate(batch_labels.categories):
        # Get silhouette values for this batch
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
            label=batch_name
        )
        
        # Label batch at center
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(batch_name))
        
        y_lower = y_upper + 10
    
    ax.set_title(f'Silhouette Plot for Batch Clustering\n(Average score: {silhouette_avg:.3f})')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Batch')
    
    # Vertical line for average score
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
               label=f'Average ({silhouette_avg:.3f})')
    
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved silhouette plot to {output_path}")
    
    return fig


def plot_batch_heatmap(
    adata: ad.AnnData,
    batch_col: str = 'batch',
    biology_col: Optional[str] = None,
    n_features: int = 50,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Hierarchical clustering heatmap colored by batch.
    
    Samples clustering by batch indicate strong batch effects.
    
    Args:
        adata: AnnData object
        batch_col: Batch annotation column
        biology_col: Biology annotation column
        n_features: Number of top variable features to show
        output_path: Where to save figure
    
    Returns:
        Matplotlib figure
    """
    import scanpy as sc
    
    # Select top variable features
    adata_subset = adata.copy()
    if adata_subset.n_vars > n_features:
        sc.pp.highly_variable_genes(adata_subset, n_top_genes=n_features)
        adata_subset = adata_subset[:, adata_subset.var.highly_variable]
    
    # Prepare annotations
    row_colors = pd.DataFrame(index=adata_subset.obs_names)
    
    # Batch colors
    batch_palette = sns.color_palette("Set2", len(adata_subset.obs[batch_col].unique()))
    batch_lut = dict(zip(adata_subset.obs[batch_col].unique(), batch_palette))
    row_colors['Batch'] = adata_subset.obs[batch_col].map(batch_lut)
    
    # Biology colors if provided
    if biology_col and biology_col in adata_subset.obs.columns:
        bio_palette = sns.color_palette("Set1", len(adata_subset.obs[biology_col].unique()))
        bio_lut = dict(zip(adata_subset.obs[biology_col].unique(), bio_palette))
        row_colors['Biology'] = adata_subset.obs[biology_col].map(bio_lut)
    
    # Create clustermap
    X = adata_subset.X.toarray() if hasattr(adata_subset.X, 'toarray') else adata_subset.X
    
    g = sns.clustermap(
        X.T,  # Features x Samples
        row_cluster=True,
        col_cluster=True,
        col_colors=row_colors,
        cmap='RdBu_r',
        center=0,
        figsize=(12, 10),
        cbar_kws={'label': 'Expression'},
        yticklabels=False  # Too many features to label
    )
    
    g.fig.suptitle('Hierarchical Clustering (colored by batch)', y=1.02)
    
    if output_path:
        g.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved batch heatmap to {output_path}")
    
    return g.fig


# ============================= BATCH CORRECTION ================================== #

def apply_conqur_correction(
    adata: ad.AnnData,
    batch_col: str = 'batch',
    covariate_cols: Optional[List[str]] = None,
    taxa_col: str = 'Taxon',
    output_dir: Optional[Path] = None
) -> ad.AnnData:
    """
    Apply ConQuR batch effect correction (via R).
    
    ConQuR is specifically designed for microbiome data and preserves
    the compositional structure while removing batch effects.
    
    Args:
        adata: AnnData object with raw counts
        batch_col: Column name for batch variable
        covariate_cols: Biological covariates to preserve
        taxa_col: Column in .var with taxonomy strings
        output_dir: Directory to save corrected data
    
    Returns:
        Corrected AnnData object
        
    Requires:
        R with ConQuR package installed:
        ```R
        install.packages("devtools")
        devtools::install_github("wdl2459/ConQuR")
        ```
    
    References:
        Jiang et al. (2020). ConQuR: batch correction for microbiome data.
        BMC Bioinformatics.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri, numpy2ri, conversion
        from rpy2.robjects.packages import importr
        
        # Use context manager instead of deprecated activate()
        
    except ImportError:
        raise ImportError(
            "ConQuR correction requires rpy2:\n"
            "  pip install rpy2\n"
            "And R with ConQuR package installed."
        )
    
    logger.info("Applying ConQuR batch correction...")
    
    try:
        # Import R packages
        base = importr('base')
        conqur = importr('ConQuR')
        
        # Prepare data
        counts = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        feature_table = pd.DataFrame(
            counts.T,  # Features x Samples
            index=adata.var_names,
            columns=adata.obs_names
        )
        
        # Prepare metadata
        meta = adata.obs[[batch_col] + (covariate_cols if covariate_cols else [])].copy()
        
        # Run ConQuR with context manager
        logger.info("Running ConQuR (this may take a while)...")
        with conversion.localconverter(ro.default_converter + pandas2ri.converter):
            # Convert to R objects
            r_counts = pandas2ri.py2rpy(feature_table)
            r_meta = pandas2ri.py2rpy(meta)
            r_batch = ro.StrVector(meta[batch_col].values)
            
            # Prepare covariates if provided
            if covariate_cols:
                r_covariates = pandas2ri.py2rpy(meta[covariate_cols])
            else:
                r_covariates = ro.NULL
            
            # Run ConQuR
            corrected = conqur.ConQuR(
                tax_tab=r_counts,
                batchid=r_batch,
                covariates=r_covariates,
                batch_ref=ro.r('NULL')  # Auto-select reference
            )
            
            # Convert back to Python
            corrected_df = pandas2ri.rpy2py(corrected)
        
        corrected_counts = corrected_df.T.values  # Samples x Features
        
        # Create new AnnData
        adata_corrected = ad.AnnData(
            X=corrected_counts,
            obs=adata.obs.copy(),
            var=adata.var.copy()
        )
        
        # Mark as corrected
        adata_corrected.uns['batch_corrected'] = True
        adata_corrected.uns['batch_correction_method'] = 'ConQuR'
        adata_corrected.uns['batch_col_used'] = batch_col
        
        logger.info("✓ ConQuR correction completed successfully")
        
        if output_dir:
            output_file = Path(output_dir) / "adata_conqur_corrected.h5ad"
            safe_write_h5ad(adata_corrected, output_file)
            logger.info(f"Saved corrected data to {output_file}")
        
        return adata_corrected
        
    except Exception as e:
        logger.error(f"ConQuR correction failed: {e}")
        logger.warning("Returning uncorrected data")
        return adata


def apply_combat_correction(
    adata: ad.AnnData,
    batch_col: str = 'batch',
    covariate_cols: Optional[List[str]] = None
) -> ad.AnnData:
    """
    Apply ComBat batch correction (simpler alternative to ConQuR).
    
    ComBat is a parametric empirical Bayes method originally developed
    for gene expression data. Works on log-transformed data.
    
    Args:
        adata: AnnData object
        batch_col: Column name for batch variable
        covariate_cols: Biological covariates to preserve
    
    Returns:
        Corrected AnnData object
        
    Note:
        Requires log-transformed data. Apply after CLR or log-normalization.
    """
    try:
        from combat.pycombat import pycombat
    except ImportError:
        raise ImportError("ComBat requires pycombat: pip install combat")
    
    logger.info("Applying ComBat correction...")
    
    # Prepare data
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    data_df = pd.DataFrame(
        X.T,  # Features x Samples
        index=adata.var_names,
        columns=adata.obs_names
    )
    
    # Prepare batch vector
    batch = adata.obs[batch_col].values
    
    # Prepare covariates
    if covariate_cols:
        covariate_df = adata.obs[covariate_cols]
    else:
        covariate_df = None
    
    # Run ComBat
    corrected_df = pycombat(
        data_df,
        batch,
        mod=covariate_df
    )
    
    # Create corrected AnnData
    adata_corrected = ad.AnnData(
        X=corrected_df.T.values,
        obs=adata.obs.copy(),
        var=adata.var.copy()
    )
    
    adata_corrected.uns['batch_corrected'] = True
    adata_corrected.uns['batch_correction_method'] = 'ComBat'
    adata_corrected.uns['batch_col_used'] = batch_col
    
    logger.info("✓ ComBat correction completed")
    
    return adata_corrected


# ========================== COMPREHENSIVE WORKFLOW ================================== #

def batch_effect_workflow(
    adata: ad.AnnData,
    batch_col: str = 'batch',
    biology_col: Optional[str] = None,
    output_dir: Optional[Path] = None,
    correct_method: Optional[str] = None
) -> Dict:
    """
    Complete batch effect analysis workflow.
    
    Workflow:
        1. Detect batch effects (PERMANOVA, silhouette, PCA)
        2. Visualize batch effects (plots)
        3. Optionally apply correction (ConQuR or ComBat)
        4. Re-assess after correction
    
    Args:
        adata: AnnData object
        batch_col: Batch column name
        biology_col: Biological variable column
        output_dir: Where to save results
        correct_method: Correction method ('conqur', 'combat', or None)
    
    Returns:
        Dictionary with results and corrected data (if requested)
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Detect batch effects
    logger.info("=" * 60)
    logger.info("STEP 1: Detecting batch effects...")
    logger.info("=" * 60)
    
    detection_results = detect_batch_effects(
        adata,
        batch_col=batch_col,
        biology_col=biology_col
    )
    results['before_correction'] = detection_results
    
    logger.info("\n" + detection_results['interpretation'])
    
    # 2. Visualize
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Creating visualizations...")
    logger.info("=" * 60)
    
    if output_dir:
        # PCA plot
        plot_batch_pca(
            adata,
            batch_col=batch_col,
            biology_col=biology_col,
            output_path=output_dir / "batch_pca.html"
        )
        
        # Silhouette plot
        plot_silhouette_analysis(
            adata,
            batch_col=batch_col,
            output_path=output_dir / "batch_silhouette.png"
        )
        
        # Heatmap
        plot_batch_heatmap(
            adata,
            batch_col=batch_col,
            biology_col=biology_col,
            output_path=output_dir / "batch_heatmap.png"
        )
    
    # 3. Correction (if requested)
    if correct_method:
        logger.info("\n" + "=" * 60)
        logger.info(f"STEP 3: Applying {correct_method.upper()} correction...")
        logger.info("=" * 60)
        
        if correct_method.lower() == 'conqur':
            adata_corrected = apply_conqur_correction(
                adata,
                batch_col=batch_col,
                covariate_cols=[biology_col] if biology_col else None,
                output_dir=output_dir
            )
        elif correct_method.lower() == 'combat':
            adata_corrected = apply_combat_correction(
                adata,
                batch_col=batch_col,
                covariate_cols=[biology_col] if biology_col else None
            )
        else:
            raise ValueError(f"Unknown correction method: {correct_method}")
        
        results['corrected_data'] = adata_corrected
        
        # 4. Re-assess after correction
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Re-assessing after correction...")
        logger.info("=" * 60)
        
        post_correction_results = detect_batch_effects(
            adata_corrected,
            batch_col=batch_col,
            biology_col=biology_col
        )
        results['after_correction'] = post_correction_results
        
        logger.info("\n" + post_correction_results['interpretation'])
        
        # Visualize corrected data
        if output_dir:
            plot_batch_pca(
                adata_corrected,
                batch_col=batch_col,
                biology_col=biology_col,
                output_path=output_dir / "batch_pca_corrected.html",
                title="PCA After Batch Correction"
            )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("BATCH EFFECT ANALYSIS COMPLETE")
    logger.info("=" * 60)
    
    return results
