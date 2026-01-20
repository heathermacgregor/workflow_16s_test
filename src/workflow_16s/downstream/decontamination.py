"""
Contamination Detection and Removal using decontam R package.

This module provides tools for identifying and removing contaminant sequences
from microbiome data using the decontam package (Davis et al. 2018).

Decontam implements two statistical methods:
1. Frequency-based: Uses DNA quantification to identify contaminants
   (contaminants are inversely correlated with DNA concentration)
2. Prevalence-based: Uses negative control samples to identify contaminants
   (contaminants are more prevalent in controls than true samples)

References:
    Davis NM, Proctor DM, Holmes SP, Relman DA, Callahan BJ. (2018).
    Simple statistical identification and removal of contaminant sequences
    in marker-gene and metagenomics data. Microbiome, 6(1), 226.
    https://doi.org/10.1186/s40168-018-0605-2

Example:
    >>> from workflow_16s.downstream.decontamination import identify_contaminants
    >>> 
    >>> # Frequency-based method
    >>> contaminants = identify_contaminants(
    ...     adata,
    ...     method='frequency',
    ...     dna_conc_column='quant_reading',
    ...     threshold=0.1
    ... )
    >>> 
    >>> # Prevalence-based method
    >>> contaminants = identify_contaminants(
    ...     adata,
    ...     method='prevalence',
    ...     control_column='sample_type',
    ...     control_value='negative_control',
    ...     threshold=0.5
    ... )
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Check for R and rpy2
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects import conversion
    
    # Use context manager instead of deprecated activate()
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False
    logger.warning("rpy2 not available. Decontam functions will not work.")


def _check_decontam_available() -> bool:
    """
    Check if R decontam package is available.
    
    Returns:
        True if decontam is installed, False otherwise
    """
    if not R_AVAILABLE:
        return False
    
    try:
        ro.r('library(decontam)')
        return True
    except Exception:
        return False


def identify_contaminants(
    adata: ad.AnnData,
    method: str = 'combined',
    dna_conc_column: Optional[str] = None,
    control_column: Optional[str] = None,
    control_value: Optional[Union[str, List[str]]] = None,
    threshold: float = 0.1,
    batch_column: Optional[str] = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Identify contaminant ASVs using decontam methods.
    
    Args:
        adata: AnnData object with feature table and metadata
        method: Detection method - 'frequency', 'prevalence', or 'combined'
        dna_conc_column: Column name with DNA concentration values (for frequency method)
        control_column: Column name indicating sample type (for prevalence method)
        control_value: Value(s) in control_column that indicate negative controls
        threshold: Contamination probability threshold (0-1, lower = more stringent)
        batch_column: Optional column for batch-specific contamination detection
        normalize: Whether to normalize by library size (default: True)
    
    Returns:
        DataFrame with contaminant scores and classifications
        
    Raises:
        RuntimeError: If R or decontam package is not available
        ValueError: If required parameters for method are missing
    """
    if not _check_decontam_available():
        raise RuntimeError(
            "R decontam package not available. Install with:\n"
            "  R -e \"install.packages('BiocManager'); BiocManager::install('decontam')\""
        )
    
    # Validate inputs
    if method not in ['frequency', 'prevalence', 'combined']:
        raise ValueError(f"Method must be 'frequency', 'prevalence', or 'combined', got: {method}")
    
    if method in ['frequency', 'combined'] and dna_conc_column is None:
        raise ValueError("dna_conc_column required for frequency-based detection")
    
    if method in ['prevalence', 'combined'] and (control_column is None or control_value is None):
        raise ValueError("control_column and control_value required for prevalence-based detection")
    
    logger.info(f"Running decontam with method={method}, threshold={threshold}")
    
    # Prepare data
    feature_table = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    feature_table = feature_table.T  # decontam expects features × samples
    
    # Convert to R objects
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        r_feature_table = ro.r.matrix(
            ro.FloatVector(feature_table.flatten()),
            nrow=feature_table.shape[0],
            ncol=feature_table.shape[1]
        )
        ro.r.assign('feature_table', r_feature_table)
    
    # Run appropriate method
    if method == 'frequency':
        results = _run_frequency_method(
            adata, dna_conc_column, threshold, batch_column, normalize
        )
    elif method == 'prevalence':
        results = _run_prevalence_method(
            adata, control_column, control_value, threshold, batch_column, normalize
        )
    else:  # combined
        freq_results = _run_frequency_method(
            adata, dna_conc_column, threshold, batch_column, normalize
        )
        prev_results = _run_prevalence_method(
            adata, control_column, control_value, threshold, batch_column, normalize
        )
        results = _combine_results(freq_results, prev_results, threshold)
    
    # Add feature metadata
    results.index = adata.var_names
    if 'taxonomy' in adata.var.columns:
        results['taxonomy'] = adata.var['taxonomy']
    
    # Log summary
    n_contaminants = results['contaminant'].sum()
    logger.info(
        f"Identified {n_contaminants}/{len(results)} ({n_contaminants/len(results)*100:.1f}%) "
        f"contaminant features at threshold={threshold}"
    )
    
    return results


def _run_frequency_method(
    adata: ad.AnnData,
    dna_conc_column: str,
    threshold: float,
    batch_column: Optional[str],
    normalize: bool
) -> pd.DataFrame:
    """Run frequency-based decontam method."""
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        # Get DNA concentrations
        dna_conc = adata.obs[dna_conc_column].values
        r_dna_conc = ro.FloatVector(dna_conc)
        ro.r.assign('dna_conc', r_dna_conc)
        
        # Build R command
        cmd = f"isContaminant(feature_table, conc=dna_conc, method='frequency', threshold={threshold}"
        if batch_column is not None:
            batch = adata.obs[batch_column].values
            r_batch = ro.StrVector(batch)
            ro.r.assign('batch', r_batch)
            cmd += ", batch=batch"
        if normalize:
            cmd += ", normalize=TRUE"
        cmd += ")"
        
        # Run decontam
        logger.debug(f"Running R command: {cmd}")
        r_results = ro.r(cmd)
        
        # Convert to pandas
        results = pd.DataFrame({
            'freq': np.array(r_results.rx2('freq')),
            'p_freq': np.array(r_results.rx2('p.freq')),
            'p_freq_adj': np.array(r_results.rx2('p')),
            'contaminant': np.array(r_results.rx2('contaminant')).astype(bool)
        })
        
        return results


def _run_prevalence_method(
    adata: ad.AnnData,
    control_column: str,
    control_value: Union[str, List[str]],
    threshold: float,
    batch_column: Optional[str],
    normalize: bool
) -> pd.DataFrame:
    """Run prevalence-based decontam method."""
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        # Create negative indicator
        if isinstance(control_value, str):
            control_value = [control_value]
        
        neg = adata.obs[control_column].isin(control_value).values
        r_neg = ro.BoolVector(neg)
        ro.r.assign('neg', r_neg)
        
        # Build R command
        cmd = f"isContaminant(feature_table, neg=neg, method='prevalence', threshold={threshold}"
        if batch_column is not None:
            batch = adata.obs[batch_column].values
            r_batch = ro.StrVector(batch)
            ro.r.assign('batch', r_batch)
            cmd += ", batch=batch"
        if normalize:
            cmd += ", normalize=TRUE"
        cmd += ")"
        
        # Run decontam
        logger.debug(f"Running R command: {cmd}")
        r_results = ro.r(cmd)
        
        # Convert to pandas
        results = pd.DataFrame({
            'prev': np.array(r_results.rx2('prev')),
            'p_prev': np.array(r_results.rx2('p.prev')),
            'p_prev_adj': np.array(r_results.rx2('p')),
            'contaminant': np.array(r_results.rx2('contaminant')).astype(bool)
        })
        
        return results


def _combine_results(
    freq_results: pd.DataFrame,
    prev_results: pd.DataFrame,
    threshold: float
) -> pd.DataFrame:
    """Combine frequency and prevalence results."""
    # Merge dataframes
    results = freq_results.join(prev_results, rsuffix='_prev')
    
    # Combined score: minimum p-value (most conservative)
    results['p_combined'] = np.minimum(
        results['p_freq_adj'],
        results['p_prev_adj']
    )
    
    # Feature is contaminant if EITHER method identifies it
    results['contaminant'] = (
        results['contaminant'] | results['contaminant_prev']
    )
    
    # Remove duplicate column
    results = results.drop(columns=['contaminant_prev'])
    
    return results


def remove_contaminants(
    adata: ad.AnnData,
    contaminants: Union[pd.DataFrame, List[str], Set[str]],
    inplace: bool = False
) -> ad.AnnData:
    """
    Remove contaminant features from AnnData object.
    
    Args:
        adata: AnnData object with feature table
        contaminants: Either DataFrame from identify_contaminants() or list/set of feature names
        inplace: Whether to modify adata in place (default: False)
    
    Returns:
        AnnData object with contaminants removed
    """
    if not inplace:
        adata = adata.copy()
    
    # Get contaminant feature names
    if isinstance(contaminants, pd.DataFrame):
        if 'contaminant' not in contaminants.columns:
            raise ValueError("DataFrame must have 'contaminant' column")
        contaminant_names = set(contaminants[contaminants['contaminant']].index)
    else:
        contaminant_names = set(contaminants)
    
    # Filter features
    keep_features = [f for f in adata.var_names if f not in contaminant_names]
    adata = adata[:, keep_features]
    
    logger.info(f"Removed {len(contaminant_names)} contaminant features")
    
    return adata


def plot_decontam_scores(
    results: pd.DataFrame,
    method: str = 'combined',
    threshold: float = 0.1,
    top_n: int = 20,
    output_path: Optional[Path] = None
) -> go.Figure:
    """
    Visualize decontam contamination scores.
    
    Args:
        results: DataFrame from identify_contaminants()
        method: Which scores to plot - 'frequency', 'prevalence', or 'combined'
        threshold: Contamination threshold to show on plot
        top_n: Number of top contaminants to label
        output_path: Optional path to save plot
    
    Returns:
        Plotly figure object
    """
    # Select appropriate p-value column
    if method == 'frequency':
        p_col = 'p_freq_adj'
        score_col = 'freq'
        title = 'Frequency-Based Contamination Scores'
        xaxis = 'Frequency (correlation with DNA concentration)'
    elif method == 'prevalence':
        p_col = 'p_prev_adj'
        score_col = 'prev'
        title = 'Prevalence-Based Contamination Scores'
        xaxis = 'Prevalence Ratio (controls/true samples)'
    else:  # combined
        p_col = 'p_combined'
        score_col = 'freq'  # Default to frequency for x-axis
        title = 'Combined Contamination Scores'
        xaxis = 'Frequency Score'
    
    if p_col not in results.columns:
        raise ValueError(f"Column {p_col} not found. Run identify_contaminants with method='{method}'")
    
    # Prepare data
    plot_data = results.copy()
    plot_data['log_p'] = -np.log10(plot_data[p_col] + 1e-300)  # Avoid log(0)
    plot_data['is_contaminant'] = plot_data['contaminant'].map({True: 'Contaminant', False: 'True Feature'})
    
    # Create scatter plot
    fig = px.scatter(
        plot_data,
        x=score_col,
        y='log_p',
        color='is_contaminant',
        color_discrete_map={'Contaminant': 'red', 'True Feature': 'blue'},
        hover_data=['taxonomy'] if 'taxonomy' in plot_data.columns else None,
        title=title,
        labels={
            score_col: xaxis,
            'log_p': '-log10(adjusted p-value)',
            'is_contaminant': 'Classification'
        },
        template='plotly_white'
    )
    
    # Add threshold line
    threshold_y = -np.log10(threshold)
    fig.add_hline(
        y=threshold_y,
        line_dash='dash',
        line_color='gray',
        annotation_text=f'Threshold (p={threshold})',
        annotation_position='right'
    )
    
    # Label top contaminants
    if top_n > 0:
        top_contaminants = plot_data[plot_data['contaminant']].nlargest(top_n, 'log_p')
        for idx, row in top_contaminants.iterrows():
            fig.add_annotation(
                x=row[score_col],
                y=row['log_p'],
                text=idx[:20],  # Truncate long names
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                ax=20,
                ay=-20,
                font=dict(size=8)
            )
    
    fig.update_layout(height=600, width=800)
    
    if output_path is not None:
        fig.write_html(output_path)
        logger.info(f"Saved decontam plot to {output_path}")
    
    return fig


def plot_prevalence_comparison(
    adata: ad.AnnData,
    results: pd.DataFrame,
    control_column: str,
    control_value: Union[str, List[str]],
    top_n: int = 20,
    output_path: Optional[Path] = None
) -> go.Figure:
    """
    Plot prevalence comparison between controls and true samples.
    
    Shows the prevalence (fraction of samples present) for each feature
    in control samples vs. true samples.
    
    Args:
        adata: AnnData object with feature table
        results: DataFrame from identify_contaminants()
        control_column: Column name indicating sample type
        control_value: Value(s) indicating negative controls
        top_n: Number of top contaminants to label
        output_path: Optional path to save plot
    
    Returns:
        Plotly figure object
    """
    if isinstance(control_value, str):
        control_value = [control_value]
    
    # Calculate prevalence
    is_control = adata.obs[control_column].isin(control_value).values
    is_true = ~is_control
    
    feature_matrix = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    prev_control = (feature_matrix[is_control] > 0).mean(axis=0)
    prev_true = (feature_matrix[is_true] > 0).mean(axis=0)
    
    # Prepare data
    plot_data = pd.DataFrame({
        'feature': adata.var_names,
        'prev_control': prev_control,
        'prev_true': prev_true,
        'contaminant': results['contaminant'].values,
        'p_value': results['p_prev_adj'].values if 'p_prev_adj' in results.columns else np.nan
    })
    
    if 'taxonomy' in adata.var.columns:
        plot_data['taxonomy'] = adata.var['taxonomy'].values
    
    plot_data['is_contaminant'] = plot_data['contaminant'].map({True: 'Contaminant', False: 'True Feature'})
    
    # Create scatter plot
    fig = px.scatter(
        plot_data,
        x='prev_true',
        y='prev_control',
        color='is_contaminant',
        color_discrete_map={'Contaminant': 'red', 'True Feature': 'blue'},
        hover_data=['taxonomy', 'p_value'] if 'taxonomy' in plot_data.columns else ['p_value'],
        title='Prevalence: Controls vs. True Samples',
        labels={
            'prev_true': 'Prevalence in True Samples',
            'prev_control': 'Prevalence in Control Samples',
            'is_contaminant': 'Classification'
        },
        template='plotly_white'
    )
    
    # Add diagonal line (equal prevalence)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        )
    )
    
    # Label top contaminants
    if top_n > 0:
        top_contaminants = plot_data[plot_data['contaminant']].nlargest(
            top_n, 'prev_control'
        )
        for _, row in top_contaminants.iterrows():
            fig.add_annotation(
                x=row['prev_true'],
                y=row['prev_control'],
                text=row['feature'][:20],
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-20,
                font=dict(size=8)
            )
    
    fig.update_layout(height=600, width=800)
    
    if output_path is not None:
        fig.write_html(output_path)
        logger.info(f"Saved prevalence comparison plot to {output_path}")
    
    return fig


def decontamination_workflow(
    adata: ad.AnnData,
    method: str = 'combined',
    dna_conc_column: Optional[str] = None,
    control_column: Optional[str] = None,
    control_value: Optional[Union[str, List[str]]] = None,
    threshold: float = 0.1,
    batch_column: Optional[str] = None,
    output_dir: Optional[Path] = None,
    remove: bool = False
) -> Dict:
    """
    Complete decontamination workflow: detect, visualize, and optionally remove.
    
    Args:
        adata: AnnData object with feature table
        method: Detection method - 'frequency', 'prevalence', or 'combined'
        dna_conc_column: DNA concentration column (for frequency method)
        control_column: Sample type column (for prevalence method)
        control_value: Control sample value(s) (for prevalence method)
        threshold: Contamination probability threshold
        batch_column: Optional batch column
        output_dir: Directory for output plots (if None, plots not saved)
        remove: Whether to remove contaminants from adata
    
    Returns:
        Dictionary with:
            - 'results': Contamination scores DataFrame
            - 'n_contaminants': Number of contaminants identified
            - 'contaminant_features': List of contaminant feature names
            - 'cleaned_data': AnnData with contaminants removed (if remove=True)
            - 'figures': Dictionary of plotly figures
    """
    logger.info("="*60)
    logger.info("DECONTAMINATION WORKFLOW")
    logger.info("="*60)
    
    # Identify contaminants
    logger.info("Step 1: Identifying contaminants...")
    results = identify_contaminants(
        adata,
        method=method,
        dna_conc_column=dna_conc_column,
        control_column=control_column,
        control_value=control_value,
        threshold=threshold,
        batch_column=batch_column
    )
    
    contaminant_features = results[results['contaminant']].index.tolist()
    n_contaminants = len(contaminant_features)
    
    # Create visualizations
    logger.info("Step 2: Creating visualizations...")
    figures = {}
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Decontam scores plot
    score_path = output_dir / 'decontam_scores.html' if output_dir else None
    figures['scores'] = plot_decontam_scores(
        results,
        method=method,
        threshold=threshold,
        output_path=score_path
    )
    
    # Prevalence comparison (if prevalence method used)
    if method in ['prevalence', 'combined'] and control_column is not None:
        prev_path = output_dir / 'prevalence_comparison.html' if output_dir else None
        figures['prevalence'] = plot_prevalence_comparison(
            adata,
            results,
            control_column,
            control_value,
            output_path=prev_path
        )
    
    # Remove contaminants if requested
    cleaned_data = None
    if remove:
        logger.info("Step 3: Removing contaminants...")
        cleaned_data = remove_contaminants(adata, results, inplace=False)
    
    # Summary
    logger.info("="*60)
    logger.info("DECONTAMINATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total features: {len(results)}")
    logger.info(f"Contaminants identified: {n_contaminants} ({n_contaminants/len(results)*100:.1f}%)")
    logger.info(f"Threshold: {threshold}")
    logger.info(f"Method: {method}")
    if remove:
        logger.info(f"Remaining features: {cleaned_data.n_vars}")
    logger.info("="*60)
    
    # Save results table
    if output_dir is not None:
        results_path = output_dir / 'decontam_results.csv'
        results.to_csv(results_path)
        logger.info(f"Results saved to {results_path}")
    
    return {
        'results': results,
        'n_contaminants': n_contaminants,
        'contaminant_features': contaminant_features,
        'cleaned_data': cleaned_data,
        'figures': figures
    }
