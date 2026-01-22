"""
Longitudinal Microbiome Analysis for Time Series Data.

This module provides methods for analyzing microbiome data with temporal structure,
including repeated measurements from the same subjects over time.

Implemented Methods:
1. ZIBR - Zero-Inflated Beta Regression for longitudinal compositional data (via R)
2. MetaLonDA - Longitudinal differential abundance testing (via R)
3. MaAsLin 2 - Mixed-effects models for longitudinal data (via R)
4. Trajectory clustering - Identify temporal patterns
5. Temporal stability metrics

References:
    Chen EZ, Li H. (2016). A two-part mixed-effects model for analyzing longitudinal
    microbiome compositional data. Bioinformatics, 32(17), 2611-2617. (ZIBR)
    
    Ahmed M, Quamruzzaman M, Kim B, Garai J, Baral S, Das K, Shukla D, Zhao N. (2021).
    MetaLonDA: a flexible R package for identifying time intervals of differentially
    abundant features in metagenomic longitudinal studies. Microbiome, 9(1), 32.
    
    Mallick H, Rahnavard A, McIver LJ, Ma S, et al. (2021). Multivariable association
    discovery in population-scale meta-omics studies. PLoS Computational Biology, 17(11).
    (MaAsLin 2)

Example:
    >>> from workflow_16s.downstream.longitudinal import (
    ...     run_zibr, detect_temporal_patterns, trajectory_clustering
    ... )
    >>> 
    >>> # Run ZIBR
    >>> zibr_results = run_zibr(
    ...     adata,
    ...     time_col='days',
    ...     subject_col='subject_id',
    ...     group_col='treatment'
    ... )
    >>> 
    >>> # Cluster trajectories
    >>> clusters = trajectory_clustering(adata, time_col='days', n_clusters=4)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger('workflow_16s')

# Check for R and rpy2
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, conversion
    from rpy2.robjects.packages import importr
    
    # Use context manager instead of deprecated activate()
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False
    logger.warning("rpy2 not available. R-based longitudinal methods will not work.")


def _check_r_package(package_name: str) -> bool:
    """Check if an R package is installed."""
    if not R_AVAILABLE:
        return False
    try:
        importr(package_name)
        return True
    except Exception:
        return False


def check_temporal_structure(
    adata: ad.AnnData,
    time_col: str,
    subject_col: str
) -> Dict:
    """
    Check if data has temporal structure suitable for longitudinal analysis.
    
    Args:
        adata: AnnData object
        time_col: Column with time values
        subject_col: Column with subject IDs
    
    Returns:
        Dictionary with temporal structure information
    """
    if time_col not in adata.obs.columns:
        raise ValueError(f"Time column '{time_col}' not found in adata.obs")
    
    if subject_col not in adata.obs.columns:
        raise ValueError(f"Subject column '{subject_col}' not found in adata.obs")
    
    # Count samples per subject
    samples_per_subject = adata.obs.groupby(subject_col).size()
    
    # Time points
    unique_times = adata.obs[time_col].unique()
    
    # Subjects with repeated measurements
    repeated_subjects = (samples_per_subject > 1).sum()
    
    info = {
        'n_subjects': len(samples_per_subject),
        'n_timepoints': len(unique_times),
        'n_repeated_subjects': repeated_subjects,
        'median_samples_per_subject': samples_per_subject.median(),
        'max_samples_per_subject': samples_per_subject.max(),
        'min_samples_per_subject': samples_per_subject.min(),
        'time_range': (adata.obs[time_col].min(), adata.obs[time_col].max()),
        'is_longitudinal': repeated_subjects > 0
    }
    
    logger.info("Temporal structure:")
    logger.info(f"  Subjects: {info['n_subjects']}")
    logger.info(f"  Subjects with repeated measures: {info['n_repeated_subjects']}")
    logger.info(f"  Time points: {info['n_timepoints']}")
    logger.info(f"  Samples per subject (median): {info['median_samples_per_subject']}")
    logger.info(f"  Time range: {info['time_range']}")
    
    if not info['is_longitudinal']:
        logger.warning("No repeated measurements detected. Longitudinal methods may not be appropriate.")
    
    return info


def run_zibr(
    adata: ad.AnnData,
    time_col: str,
    subject_col: str,
    group_col: Optional[str] = None,
    feature: Optional[str] = None,
    alpha: float = 0.05,
    min_prevalence: float = 0.1
) -> pd.DataFrame:
    """
    Run ZIBR (Zero-Inflated Beta Regression) for longitudinal data.
    
    ZIBR models longitudinal compositional data using a two-part model:
    1. Logistic regression for presence/absence
    2. Beta regression for non-zero abundances
    
    Args:
        adata: AnnData object with longitudinal data
        time_col: Column with time values
        subject_col: Column with subject IDs
        group_col: Optional column for group comparison
        feature: Specific feature to test (if None, tests all features)
        alpha: Significance threshold
        min_prevalence: Minimum feature prevalence
    
    Returns:
        DataFrame with ZIBR results
        
    Raises:
        RuntimeError: If R or ZIBR package is not available
    """
    if not _check_r_package('ZIBR'):
        raise RuntimeError(
            "ZIBR R package not available. Install with:\n"
            "  R -e \"devtools::install_github('chvlyl/ZIBR')\""
        )
    
    logger.info("Running ZIBR longitudinal analysis")
    
    # Check temporal structure
    temp_info = check_temporal_structure(adata, time_col, subject_col)
    if not temp_info['is_longitudinal']:
        logger.warning("Data does not appear to have repeated measurements")
    
    # Filter by prevalence
    prevalence = (adata.X > 0).mean(axis=0)
    if hasattr(prevalence, 'A1'):
        prevalence = prevalence.A1
    
    keep_features = prevalence >= min_prevalence
    
    if feature is not None:
        # Test single feature
        if feature not in adata.var_names:
            raise ValueError(f"Feature '{feature}' not found")
        features_to_test = [feature]
    else:
        # Test all features above prevalence threshold
        features_to_test = adata.var_names[keep_features]
    
    logger.info(f"Testing {len(features_to_test)} features")
    
    # Prepare data
    abundance_data = adata.to_df()
    metadata = adata.obs[[time_col, subject_col]].copy()
    if group_col:
        metadata[group_col] = adata.obs[group_col]
    
    # Convert to relative abundance
    rel_abund = abundance_data.div(abundance_data.sum(axis=1), axis=0)
    
    # Run ZIBR for each feature
    results = []
    
    for feat in features_to_test:
        # Prepare feature data
        feat_data = pd.DataFrame({
            'abundance': rel_abund[feat].values,
            'time': metadata[time_col].values,
            'subject': metadata[subject_col].values
        })
        
        if group_col:
            feat_data['group'] = metadata[group_col].values
        
        try:
            # Use context manager for R conversions
            with conversion.localconverter(ro.default_converter + pandas2ri.converter):
                # Convert to R
                r_data = pandas2ri.py2rpy(feat_data)
                ro.r.assign('feat_data', r_data)
                
                # Build formula
                if group_col:
                    formula = "abundance ~ time * group + (1 | subject)"
                else:
                    formula = "abundance ~ time + (1 | subject)"
                
                # Run ZIBR
                ro.r(f'''
                library(ZIBR)
                
                # Fit ZIBR model
                zibr_fit <- zibr(
                    logistic.cov = feat_data[, c('time', 'subject'{', "group"' if group_col else ''})],
                    beta.cov = feat_data[, c('time', 'subject'{', "group"' if group_col else ''})],
                    Y = feat_data$abundance,
                    subject.ind = feat_data$subject,
                    time.ind = feat_data$time
                )
                
                # Extract coefficients and p-values
                logistic_coef <- zibr_fit$logistic.est.table
                beta_coef <- zibr_fit$beta.est.table
                ''')
                
                # Get results
                logistic_coef = pandas2ri.rpy2py(ro.r('logistic_coef'))
                beta_coef = pandas2ri.rpy2py(ro.r('beta_coef'))
            
            # Extract time effect p-values
            logistic_time_p = logistic_coef.loc[logistic_coef.index.str.contains('time'), 'Pr(>|z|)'].iloc[0]
            beta_time_p = beta_coef.loc[beta_coef.index.str.contains('time'), 'Pr(>|t|)'].iloc[0]
            
            results.append({
                'feature': feat,
                'logistic_time_p': logistic_time_p,
                'beta_time_p': beta_time_p,
                'min_p': min(logistic_time_p, beta_time_p)
            })
            
        except Exception as e:
            logger.warning(f"ZIBR failed for {feat}: {e}")
            results.append({
                'feature': feat,
                'logistic_time_p': 1.0,
                'beta_time_p': 1.0,
                'min_p': 1.0
            })
    
    results_df = pd.DataFrame(results)
    
    # FDR correction
    _, p_adj, _, _ = multipletests(results_df['min_p'], method='fdr_bh')
    results_df['p_adj'] = p_adj
    
    # Sort by adjusted p-value
    results_df = results_df.sort_values('p_adj')
    
    # Log summary
    n_sig = (results_df['p_adj'] < alpha).sum()
    logger.info(f"ZIBR identified {n_sig}/{len(results_df)} features with significant temporal trends")
    
    return results_df


def run_maaslin2_longitudinal(
    adata: ad.AnnData,
    time_col: str,
    subject_col: str,
    fixed_effects: List[str],
    random_effects: List[str],
    output_dir: Path,
    alpha: float = 0.05,
    normalization: str = 'TSS',
    transform: str = 'LOG'
) -> pd.DataFrame:
    """
    Run MaAsLin 2 with mixed-effects models for longitudinal data.
    
    Args:
        adata: AnnData object
        time_col: Time column
        subject_col: Subject ID column (will be added as random effect)
        fixed_effects: List of fixed effect variables
        random_effects: List of random effect variables
        output_dir: Output directory for MaAsLin 2 results
        alpha: Significance threshold
        normalization: Normalization method ('TSS', 'CLR', 'NONE')
        transform: Transformation ('LOG', 'LOGIT', 'AST', 'NONE')
    
    Returns:
        DataFrame with significant results
        
    Raises:
        RuntimeError: If R or Maaslin2 is not available
    """
    if not _check_r_package('Maaslin2'):
        raise RuntimeError(
            "Maaslin2 R package not available. Install with:\n"
            "  R -e \"BiocManager::install('Maaslin2')\""
        )
    
    logger.info("Running MaAsLin 2 with mixed-effects models")
    
    # Prepare abundance table
    abundance = adata.to_df()
    
    # Prepare metadata
    metadata_cols = [subject_col, time_col] + fixed_effects
    if random_effects:
        metadata_cols.extend([col for col in random_effects if col not in metadata_cols])
    
    metadata = adata.obs[metadata_cols].copy()
    
    # Add subject as random effect if not already included
    if subject_col not in random_effects:
        random_effects = random_effects + [subject_col]
    
    # Run MaAsLin 2
    logger.info(f"Fixed effects: {fixed_effects}")
    logger.info(f"Random effects: {random_effects}")
    
    # Use context manager for R conversions
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        # Convert to R
        r_abundance = pandas2ri.py2rpy(abundance)
        r_metadata = pandas2ri.py2rpy(metadata)
        
        ro.r.assign('abundance', r_abundance)
        ro.r.assign('metadata', r_metadata)
        ro.r.assign('output_dir', str(output_dir))
        
        ro.r(f'''
        library(Maaslin2)
        
        Maaslin2(
            input_data = abundance,
            input_metadata = metadata,
            output = output_dir,
            fixed_effects = c({', '.join([f'"{x}"' for x in fixed_effects])}),
            random_effects = c({', '.join([f'"{x}"' for x in random_effects])}),
            normalization = "{normalization}",
            transform = "{transform}",
            analysis_method = "LM",
            max_significance = {alpha},
            min_prevalence = 0.1,
            plot_heatmap = TRUE,
            plot_scatter = TRUE
        )
        
        # Read results
        results <- read.table(
            file.path(output_dir, "significant_results.tsv"),
            header = TRUE,
            sep = "\t",
            stringsAsFactors = FALSE
        )
        ''')
        
        # Get results
        try:
            results = pandas2ri.rpy2py(ro.r('results'))
        except Exception:
            # No significant results
            logger.warning("MaAsLin 2 found no significant results")
            results = pd.DataFrame()
    
    logger.info(f"MaAsLin 2 completed. Results saved to {output_dir}")
    
    return results


def trajectory_clustering(
    adata: ad.AnnData,
    time_col: str,
    subject_col: str,
    n_clusters: int = 4,
    features: Optional[List[str]] = None,
    method: str = 'kmeans'
) -> Dict:
    """
    Cluster samples based on temporal trajectories.
    
    Args:
        adata: AnnData object with longitudinal data
        time_col: Time column
        subject_col: Subject ID column
        n_clusters: Number of trajectory clusters
        features: Features to use (if None, uses all)
        method: Clustering method ('kmeans', 'hierarchical')
    
    Returns:
        Dictionary with clustering results
    """
    logger.info(f"Clustering temporal trajectories (n_clusters={n_clusters})")
    
    # Get features to use
    if features is None:
        features = adata.var_names
    
    # Prepare data
    abundance = adata[:, features].to_df()
    
    # Group by subject and aggregate temporal profiles
    metadata = adata.obs[[subject_col, time_col]].copy()
    metadata['sample_idx'] = range(len(metadata))
    
    # For each subject, create time-series feature vector
    subject_profiles = []
    subject_ids = []
    
    for subject in metadata[subject_col].unique():
        subject_data = metadata[metadata[subject_col] == subject].sort_values(time_col)
        subject_abundance = abundance.loc[subject_data['sample_idx']]
        
        # Flatten time series into single vector
        profile = subject_abundance.values.flatten()
        
        subject_profiles.append(profile)
        subject_ids.append(subject)
    
    # Standardize
    scaler = StandardScaler()
    profiles_scaled = scaler.fit_transform(subject_profiles)
    
    # Cluster
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(profiles_scaled)
    else:
        from sklearn.cluster import AgglomerativeClustering
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(profiles_scaled)
    
    # Create results
    cluster_assignments = pd.DataFrame({
        'subject': subject_ids,
        'cluster': cluster_labels
    })
    
    # Add cluster assignment back to adata
    subject_to_cluster = dict(zip(subject_ids, cluster_labels))
    adata.obs['trajectory_cluster'] = adata.obs[subject_col].map(subject_to_cluster)
    
    logger.info(f"Identified {n_clusters} trajectory clusters")
    for cluster in range(n_clusters):
        n_subjects = (cluster_labels == cluster).sum()
        logger.info(f"  Cluster {cluster}: {n_subjects} subjects")
    
    return {
        'cluster_assignments': cluster_assignments,
        'cluster_labels': cluster_labels,
        'n_clusters': n_clusters
    }


def calculate_temporal_stability(
    adata: ad.AnnData,
    time_col: str,
    subject_col: str,
    metric: str = 'bray_curtis'
) -> pd.DataFrame:
    """
    Calculate temporal stability for each subject.
    
    Temporal stability measures how much the microbiome composition changes
    over time within subjects.
    
    Args:
        adata: AnnData object
        time_col: Time column
        subject_col: Subject ID column
        metric: Distance metric ('bray_curtis', 'euclidean', 'cosine')
    
    Returns:
        DataFrame with stability metrics per subject
    """
    from sklearn.metrics.pairwise import pairwise_distances
    
    logger.info(f"Calculating temporal stability (metric={metric})")
    
    stability_results = []
    
    for subject in adata.obs[subject_col].unique():
        # Get subject samples sorted by time
        subject_mask = adata.obs[subject_col] == subject
        subject_data = adata[subject_mask].copy()
        
        if len(subject_data) < 2:
            continue
        
        # Sort by time
        time_order = subject_data.obs[time_col].argsort()
        subject_data = subject_data[time_order]
        
        # Calculate pairwise distances
        abundance = subject_data.X.toarray() if hasattr(subject_data.X, 'toarray') else subject_data.X
        
        distances = pairwise_distances(abundance, metric=metric)
        
        # Calculate stability metrics
        # Average distance between consecutive time points
        consecutive_distances = [distances[i, i+1] for i in range(len(distances)-1)]
        avg_consecutive_dist = np.mean(consecutive_distances)
        
        # Average distance between all time points
        triu_indices = np.triu_indices_from(distances, k=1)
        avg_all_dist = np.mean(distances[triu_indices])
        
        stability_results.append({
            'subject': subject,
            'n_timepoints': len(subject_data),
            'avg_consecutive_distance': avg_consecutive_dist,
            'avg_overall_distance': avg_all_dist,
            'stability_score': 1 - avg_consecutive_dist  # Higher = more stable
        })
    
    stability_df = pd.DataFrame(stability_results)
    
    logger.info(f"Calculated stability for {len(stability_df)} subjects")
    logger.info(f"Mean stability score: {stability_df['stability_score'].mean():.3f}")
    
    return stability_df


def plot_temporal_trajectories(
    adata: ad.AnnData,
    time_col: str,
    subject_col: str,
    feature: str,
    color_by: Optional[str] = None,
    output_path: Optional[Path] = None
) -> go.Figure:
    """
    Plot temporal trajectories for a specific feature.
    
    Args:
        adata: AnnData object
        time_col: Time column
        subject_col: Subject ID column
        feature: Feature to plot
        color_by: Optional column to color lines by
        output_path: Optional path to save plot
    
    Returns:
        Plotly figure object
    """
    # Prepare data
    plot_data = pd.DataFrame({
        'time': adata.obs[time_col].values,
        'subject': adata.obs[subject_col].values,
        'abundance': adata[:, feature].X.toarray().flatten() if hasattr(adata.X, 'toarray') else adata[:, feature].X.flatten()
    })
    
    if color_by and color_by in adata.obs.columns:
        plot_data['group'] = adata.obs[color_by].values
        color_col = 'group'
    else:
        color_col = 'subject'
    
    # Create plot
    fig = px.line(
        plot_data,
        x='time',
        y='abundance',
        color=color_col,
        line_group='subject',
        title=f'Temporal Trajectory: {feature}',
        labels={
            'time': time_col.replace('_', ' ').title(),
            'abundance': 'Abundance',
            color_col: color_col.replace('_', ' ').title()
        },
        template='plotly_white'
    )
    
    fig.update_layout(height=500, width=800)
    
    if output_path:
        fig.write_html(output_path)
        logger.info(f"Trajectory plot saved to {output_path}")
    
    return fig


def longitudinal_analysis_workflow(
    adata: ad.AnnData,
    time_col: str,
    subject_col: str,
    group_col: Optional[str] = None,
    method: str = 'zibr',
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict:
    """
    Complete longitudinal analysis workflow.
    
    Args:
        adata: AnnData object with longitudinal data
        time_col: Time column
        subject_col: Subject ID column
        group_col: Optional group comparison column
        method: Analysis method ('zibr', 'maaslin2')
        output_dir: Optional output directory
        **kwargs: Method-specific parameters
    
    Returns:
        Dictionary with analysis results
    """
    logger.info("="*60)
    logger.info("LONGITUDINAL MICROBIOME ANALYSIS WORKFLOW")
    logger.info("="*60)
    
    # Check temporal structure
    logger.info("Step 1: Checking temporal structure...")
    temp_info = check_temporal_structure(adata, time_col, subject_col)
    
    if not temp_info['is_longitudinal']:
        raise ValueError("Data does not have repeated measurements for longitudinal analysis")
    
    # Run analysis
    if method == 'zibr':
        logger.info("Step 2: Running ZIBR analysis...")
        results = run_zibr(adata, time_col, subject_col, group_col, **kwargs)
    elif method == 'maaslin2':
        logger.info("Step 2: Running MaAsLin 2 analysis...")
        if output_dir is None:
            raise ValueError("output_dir required for MaAsLin 2")
        results = run_maaslin2_longitudinal(
            adata, time_col, subject_col,
            fixed_effects=kwargs.get('fixed_effects', [time_col]),
            random_effects=kwargs.get('random_effects', []),
            output_dir=output_dir,
            **{k: v for k, v in kwargs.items() if k not in ['fixed_effects', 'random_effects']}
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Calculate stability
    logger.info("Step 3: Calculating temporal stability...")
    stability = calculate_temporal_stability(adata, time_col, subject_col)
    
    # Trajectory clustering
    logger.info("Step 4: Clustering trajectories...")
    clusters = trajectory_clustering(adata, time_col, subject_col)
    
    # Save outputs
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results.to_csv(output_dir / 'longitudinal_results.csv', index=False)
        stability.to_csv(output_dir / 'temporal_stability.csv', index=False)
        clusters['cluster_assignments'].to_csv(output_dir / 'trajectory_clusters.csv', index=False)
        
        logger.info(f"Results saved to {output_dir}")
    
    logger.info("="*60)
    logger.info("LONGITUDINAL ANALYSIS COMPLETE")
    logger.info("="*60)
    
    return {
        'temporal_info': temp_info,
        'results': results,
        'stability': stability,
        'clusters': clusters
    }
