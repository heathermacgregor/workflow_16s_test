"""
Decontam - Contaminant Identification for Microbiome Data

This module integrates the R package 'decontam' for identifying contaminant features
in microbiome data using negative control samples (extraction blanks, reagent controls, etc.).

WHY THIS IS CRITICAL:
- Reagent contamination is pervasive in low-biomass samples
- Contaminants can dominate signals and lead to false biological conclusions
- Standard filtering (prevalence, abundance) doesn't distinguish contaminants from real taxa
- Negative controls are the ONLY way to identify laboratory contamination

METHODS:
1. Frequency-based: Contaminants are inversely correlated with DNA concentration
   - Real taxa: higher abundance in high-biomass samples
   - Contaminants: higher relative abundance in low-biomass samples
   
2. Prevalence-based: Contaminants are more prevalent in negative controls
   - Real taxa: low prevalence in blanks
   - Contaminants: high prevalence in blanks
   
3. Combined: Uses both methods for maximum sensitivity

REFERENCES:
- Davis et al. (2018). Simple statistical identification and removal of contaminant 
  sequences in marker-gene and metagenomics data. Microbiome.
  https://doi.org/10.1186/s40168-018-0605-2

Author: GitHub Copilot (AI Assistant)
Date: 2026-01-08
"""

# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Third-Party Imports
import anndata as ad
import numpy as np
import pandas as pd

# R Integration
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
    numpy2ri.activate()
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')


# ==================================== FUNCTIONS ===================================== #

def _check_r_dependencies() -> None:
    """Check if R and decontam package are available."""
    if not R_AVAILABLE:
        raise ImportError(
            "rpy2 is not installed. Install with: conda install -c conda-forge rpy2"
        )
    
    try:
        # Try to import decontam
        ro.r('library(decontam)')
    except Exception:
        raise ImportError(
            "R package 'decontam' is not installed.\n"
            "Install in R with:\n"
            "  if (!requireNamespace('BiocManager', quietly = TRUE))\n"
            "      install.packages('BiocManager')\n"
            "  BiocManager::install('decontam')"
        )


def identify_contaminants_frequency(
    adata: ad.AnnData,
    concentration_col: str,
    threshold: float = 0.1,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Identify contaminants based on inverse correlation with DNA concentration.
    
    PRINCIPLE:
    Real biological taxa should be more abundant in high-biomass samples.
    Contaminants (from reagents) are diluted in high-biomass samples, so they
    show HIGHER relative abundance in low-biomass samples.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with count data
    concentration_col : str
        Column in adata.obs with DNA concentration (ng/µL)
    threshold : float, default=0.1
        P-value threshold for calling contaminants (0.1 = 10% FDR)
    normalize : bool, default=True
        Whether to normalize counts to relative abundance
    
    Returns
    -------
    pd.DataFrame
        Columns:
        - feature: Feature name
        - freq: Regression coefficient (negative = potential contaminant)
        - p_freq: P-value for frequency test
        - contaminant: Boolean, True if p < threshold
    
    Notes
    -----
    - Requires DNA concentration measurements for all samples
    - Works best for low-biomass samples where contamination is significant
    - Use threshold=0.1 for conservative detection, 0.5 for aggressive
    
    Examples
    --------
    >>> # Identify contaminants using DNA concentration
    >>> contam_freq = identify_contaminants_frequency(
    ...     adata, concentration_col='dna_conc_ng_ul', threshold=0.1
    ... )
    >>> # Remove contaminants
    >>> clean_adata = adata[:, ~contam_freq['contaminant']]
    """
    _check_r_dependencies()
    
    logger.info("Running decontam frequency-based contaminant identification...")
    
    # Validate inputs
    if concentration_col not in adata.obs.columns:
        raise ValueError(f"Concentration column '{concentration_col}' not found in adata.obs")
    
    # Check for missing concentrations
    missing = adata.obs[concentration_col].isna().sum()
    if missing > 0:
        logger.warning(
            f"{missing} samples have missing DNA concentration. "
            "These will be excluded from frequency-based detection."
        )
    
    # Get counts matrix (convert to dense if sparse)
    if hasattr(adata.X, 'toarray'):
        counts = adata.X.toarray()
    else:
        counts = adata.X
    
    # Convert to DataFrame for R
    count_df = pd.DataFrame(
        counts,
        index=adata.obs_names,
        columns=adata.var_names
    )
    
    # Get DNA concentrations
    concentrations = adata.obs[concentration_col].values
    
    # Transfer to R
    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        r_counts = ro.conversion.py2rpy(count_df.T)  # R expects features x samples
        r_conc = ro.FloatVector(concentrations)
    
    # Run decontam in R
    logger.debug("Calling decontam::isContaminant() with method='frequency'")
    
    ro.r.assign('seqtab', r_counts)
    ro.r.assign('conc', r_conc)
    ro.r.assign('threshold', threshold)
    ro.r.assign('normalize', 'TRUE' if normalize else 'FALSE')
    
    ro.r('''
        contamdf <- isContaminant(
            seqtab, 
            conc = conc, 
            method = "frequency",
            threshold = threshold,
            normalize = normalize
        )
    ''')
    
    # Get results back from R
    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        result_df = ro.conversion.rpy2py(ro.r('contamdf'))
    
    # Add feature names
    result_df.insert(0, 'feature', adata.var_names.tolist())
    
    # Log summary
    n_contaminants = result_df['contaminant'].sum()
    logger.info(
        f"Frequency-based detection: {n_contaminants} contaminants identified "
        f"({n_contaminants/len(result_df)*100:.1f}%)"
    )
    
    return result_df


def identify_contaminants_prevalence(
    adata: ad.AnnData,
    neg_control_col: str,
    neg_control_value: Union[str, bool] = True,
    threshold: float = 0.1
) -> pd.DataFrame:
    """
    Identify contaminants based on higher prevalence in negative controls.
    
    PRINCIPLE:
    Real biological taxa should be absent (or rare) in negative controls.
    Contaminants (from extraction kits, reagents) appear frequently in blanks.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with count data
    neg_control_col : str
        Column in adata.obs identifying negative controls
    neg_control_value : str or bool, default=True
        Value in neg_control_col indicating negative control samples
    threshold : float, default=0.1
        P-value threshold for calling contaminants
    
    Returns
    -------
    pd.DataFrame
        Columns:
        - feature: Feature name
        - prev: Prevalence statistic
        - p_prev: P-value for prevalence test
        - contaminant: Boolean, True if p < threshold
    
    Notes
    -----
    - Requires at least 2 negative control samples
    - More negative controls = more statistical power
    - Use threshold=0.1 for conservative, 0.5 for aggressive detection
    
    Examples
    --------
    >>> # Identify contaminants using extraction blanks
    >>> contam_prev = identify_contaminants_prevalence(
    ...     adata, neg_control_col='sample_type', neg_control_value='blank'
    ... )
    >>> # See most prevalent contaminants
    >>> top_contam = contam_prev[contam_prev['contaminant']].sort_values('p_prev')
    """
    _check_r_dependencies()
    
    logger.info("Running decontam prevalence-based contaminant identification...")
    
    # Validate inputs
    if neg_control_col not in adata.obs.columns:
        raise ValueError(f"Negative control column '{neg_control_col}' not found in adata.obs")
    
    # Get negative control mask
    is_neg = adata.obs[neg_control_col] == neg_control_value
    n_neg = is_neg.sum()
    n_pos = (~is_neg).sum()
    
    if n_neg < 2:
        raise ValueError(
            f"Need at least 2 negative control samples, found {n_neg}. "
            "Prevalence-based detection requires multiple blanks for statistics."
        )
    
    logger.info(f"Found {n_neg} negative controls and {n_pos} true samples")
    
    # Get counts matrix
    if hasattr(adata.X, 'toarray'):
        counts = adata.X.toarray()
    else:
        counts = adata.X
    
    # Convert to DataFrame for R
    count_df = pd.DataFrame(
        counts,
        index=adata.obs_names,
        columns=adata.var_names
    )
    
    # Transfer to R
    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        r_counts = ro.conversion.py2rpy(count_df.T)  # Features x samples
        r_neg = ro.BoolVector(is_neg.values)
    
    # Run decontam in R
    logger.debug("Calling decontam::isContaminant() with method='prevalence'")
    
    ro.r.assign('seqtab', r_counts)
    ro.r.assign('neg', r_neg)
    ro.r.assign('threshold', threshold)
    
    ro.r('''
        contamdf <- isContaminant(
            seqtab,
            neg = neg,
            method = "prevalence",
            threshold = threshold
        )
    ''')
    
    # Get results
    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        result_df = ro.conversion.rpy2py(ro.r('contamdf'))
    
    # Add feature names
    result_df.insert(0, 'feature', adata.var_names.tolist())
    
    # Log summary
    n_contaminants = result_df['contaminant'].sum()
    logger.info(
        f"Prevalence-based detection: {n_contaminants} contaminants identified "
        f"({n_contaminants/len(result_df)*100:.1f}%)"
    )
    
    return result_df


def identify_contaminants_combined(
    adata: ad.AnnData,
    concentration_col: str,
    neg_control_col: str,
    neg_control_value: Union[str, bool] = True,
    threshold: float = 0.1,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Identify contaminants using BOTH frequency and prevalence methods.
    
    PRINCIPLE:
    Maximum sensitivity - a feature is a contaminant if EITHER:
    1. Inversely correlated with DNA concentration (frequency), OR
    2. More prevalent in negative controls (prevalence)
    
    This combined approach catches:
    - Reagent contaminants (high in blanks)
    - Extraction contaminants (inversely correlated with biomass)
    - Cross-contamination between samples
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with count data
    concentration_col : str
        Column in adata.obs with DNA concentration
    neg_control_col : str
        Column identifying negative controls
    neg_control_value : str or bool, default=True
        Value indicating negative control samples
    threshold : float, default=0.1
        P-value threshold for both tests
    normalize : bool, default=True
        Whether to normalize for frequency test
    
    Returns
    -------
    pd.DataFrame
        Columns:
        - feature: Feature name
        - freq, p_freq: Frequency test results
        - prev, p_prev: Prevalence test results
        - p_combined: Combined p-value (Fisher's method)
        - contaminant: Boolean, True if EITHER test significant
    
    Notes
    -----
    - Most sensitive method (catches more contaminants)
    - Requires both DNA concentrations AND negative controls
    - Recommended for low-biomass studies (soil, skin, environmental)
    
    Examples
    --------
    >>> # Full contaminant detection
    >>> contam = identify_contaminants_combined(
    ...     adata,
    ...     concentration_col='dna_conc',
    ...     neg_control_col='sample_type',
    ...     neg_control_value='blank'
    ... )
    >>> # Remove all identified contaminants
    >>> clean = adata[:, ~contam['contaminant']]
    >>> logger.info(f"Removed {contam['contaminant'].sum()} contaminants")
    """
    _check_r_dependencies()
    
    logger.info("Running decontam combined (frequency + prevalence) detection...")
    
    # Validate inputs
    if concentration_col not in adata.obs.columns:
        raise ValueError(f"Concentration column '{concentration_col}' not found")
    if neg_control_col not in adata.obs.columns:
        raise ValueError(f"Negative control column '{neg_control_col}' not found")
    
    # Get negative control mask
    is_neg = adata.obs[neg_control_col] == neg_control_value
    n_neg = is_neg.sum()
    
    if n_neg < 2:
        raise ValueError(f"Need at least 2 negative controls, found {n_neg}")
    
    # Get counts matrix
    if hasattr(adata.X, 'toarray'):
        counts = adata.X.toarray()
    else:
        counts = adata.X
    
    count_df = pd.DataFrame(counts, index=adata.obs_names, columns=adata.var_names)
    
    # Get metadata
    concentrations = adata.obs[concentration_col].values
    
    # Transfer to R
    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        r_counts = ro.conversion.py2rpy(count_df.T)
        r_conc = ro.FloatVector(concentrations)
        r_neg = ro.BoolVector(is_neg.values)
    
    # Run combined decontam
    logger.debug("Calling decontam::isContaminant() with method='combined'")
    
    ro.r.assign('seqtab', r_counts)
    ro.r.assign('conc', r_conc)
    ro.r.assign('neg', r_neg)
    ro.r.assign('threshold', threshold)
    ro.r.assign('normalize', 'TRUE' if normalize else 'FALSE')
    
    ro.r('''
        contamdf <- isContaminant(
            seqtab,
            conc = conc,
            neg = neg,
            method = "combined",
            threshold = threshold,
            normalize = normalize
        )
    ''')
    
    # Get results
    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        result_df = ro.conversion.rpy2py(ro.r('contamdf'))
    
    result_df.insert(0, 'feature', adata.var_names.tolist())
    
    # Log summary
    n_contaminants = result_df['contaminant'].sum()
    freq_only = (result_df['p.freq'] < threshold).sum()
    prev_only = (result_df['p.prev'] < threshold).sum()
    
    logger.info(
        f"Combined detection: {n_contaminants} contaminants identified "
        f"({n_contaminants/len(result_df)*100:.1f}%)"
    )
    logger.info(
        f"  Frequency-based: {freq_only}, "
        f"Prevalence-based: {prev_only}, "
        f"Combined: {n_contaminants}"
    )
    
    return result_df


def remove_contaminants(
    adata: ad.AnnData,
    contaminant_df: pd.DataFrame,
    contaminant_col: str = 'contaminant'
) -> ad.AnnData:
    """
    Remove identified contaminants from AnnData object.
    
    Parameters
    ----------
    adata : ad.AnnData
        Original AnnData object
    contaminant_df : pd.DataFrame
        Results from identify_contaminants_*() function
    contaminant_col : str, default='contaminant'
        Column in contaminant_df marking contaminants
    
    Returns
    -------
    ad.AnnData
        Filtered AnnData with contaminants removed
    
    Examples
    --------
    >>> contam = identify_contaminants_prevalence(adata, 'sample_type', 'blank')
    >>> clean = remove_contaminants(adata, contam)
    >>> logger.info(f"Before: {adata.n_vars} features, After: {clean.n_vars} features")
    """
    # Get non-contaminant features
    non_contam_features = contaminant_df[~contaminant_df[contaminant_col]]['feature'].tolist()
    
    # Filter AnnData
    clean_adata = adata[:, non_contam_features].copy()
    
    # Store contaminant info in uns
    clean_adata.uns['contaminants_removed'] = {
        'n_contaminants': contaminant_df[contaminant_col].sum(),
        'n_remaining': len(non_contam_features),
        'pct_removed': contaminant_df[contaminant_col].sum() / len(contaminant_df) * 100,
        'contaminant_list': contaminant_df[contaminant_df[contaminant_col]]['feature'].tolist()
    }
    
    n_removed = contaminant_df[contaminant_col].sum()
    logger.info(
        f"Removed {n_removed} contaminant features "
        f"({n_removed/len(contaminant_df)*100:.1f}%)"
    )
    
    return clean_adata


def plot_contaminant_distribution(
    adata: ad.AnnData,
    contaminant_df: pd.DataFrame,
    neg_control_col: str,
    neg_control_value: Union[str, bool],
    output_path: Optional[Path] = None,
    top_n: int = 20
) -> None:
    """
    Plot the distribution of top contaminants in negative controls vs true samples.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with count data
    contaminant_df : pd.DataFrame
        Results from contaminant identification
    neg_control_col : str
        Column identifying negative controls
    neg_control_value : str or bool
        Value indicating negative controls
    output_path : Path, optional
        Where to save the plot
    top_n : int, default=20
        Number of top contaminants to show
    
    Notes
    -----
    Creates a heatmap showing:
    - Rows: Top N contaminants (by p-value)
    - Columns: Samples (controls vs true samples)
    - Color: Relative abundance
    
    Examples
    --------
    >>> contam = identify_contaminants_prevalence(adata, 'sample_type', 'blank')
    >>> plot_contaminant_distribution(
    ...     adata, contam, 'sample_type', 'blank',
    ...     output_path=Path('qc/contaminants.png')
    ... )
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping plot")
        return
    
    # Get top contaminants
    contaminants = contaminant_df[contaminant_df['contaminant']].copy()
    
    if len(contaminants) == 0:
        logger.info("No contaminants identified, skipping plot")
        return
    
    # Sort by p-value
    if 'p_prev' in contaminants.columns:
        contaminants = contaminants.sort_values('p_prev').head(top_n)
    elif 'p_freq' in contaminants.columns:
        contaminants = contaminants.sort_values('p_freq').head(top_n)
    else:
        contaminants = contaminants.head(top_n)
    
    top_features = contaminants['feature'].tolist()
    
    # Get data for these features
    feature_idx = [adata.var_names.tolist().index(f) for f in top_features if f in adata.var_names]
    
    if hasattr(adata.X, 'toarray'):
        data = adata.X[:, feature_idx].toarray()
    else:
        data = adata.X[:, feature_idx]
    
    # Convert to relative abundance
    data_rel = data / data.sum(axis=1, keepdims=True) * 100
    
    # Sort samples: negatives first, then true samples
    is_neg = adata.obs[neg_control_col] == neg_control_value
    sample_order = np.concatenate([
        np.where(is_neg)[0],
        np.where(~is_neg)[0]
    ])
    
    data_rel = data_rel[sample_order, :]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(
        data_rel.T,
        cmap='YlOrRd',
        yticklabels=top_features,
        xticklabels=False,
        cbar_kws={'label': 'Relative Abundance (%)'},
        ax=ax
    )
    
    # Add separator between controls and samples
    n_neg = is_neg.sum()
    ax.axvline(x=n_neg, color='blue', linestyle='--', linewidth=2)
    
    ax.set_xlabel(f'Samples (Negative Controls | True Samples)', fontsize=12)
    ax.set_ylabel('Contaminant Features', fontsize=12)
    ax.set_title(
        f'Top {len(top_features)} Contaminants: Distribution Across Samples',
        fontsize=14, fontweight='bold'
    )
    
    # Add text annotation
    ax.text(
        n_neg/2, -1, 'Negative\nControls',
        ha='center', va='top', fontsize=10, fontweight='bold', color='blue'
    )
    ax.text(
        n_neg + (len(sample_order)-n_neg)/2, -1, 'True\nSamples',
        ha='center', va='top', fontsize=10, fontweight='bold', color='green'
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved contaminant distribution plot: {output_path}")
    else:
        plt.show()
    
    plt.close()


def decontam_workflow(
    adata: ad.AnnData,
    method: str = 'combined',
    concentration_col: Optional[str] = None,
    neg_control_col: Optional[str] = None,
    neg_control_value: Union[str, bool] = True,
    threshold: float = 0.1,
    remove_contam: bool = True,
    output_dir: Optional[Path] = None
) -> Tuple[ad.AnnData, pd.DataFrame]:
    """
    Complete decontam workflow: identify and optionally remove contaminants.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with count data
    method : str, default='combined'
        Detection method: 'frequency', 'prevalence', or 'combined'
    concentration_col : str, optional
        Column with DNA concentration (required for frequency/combined)
    neg_control_col : str, optional
        Column identifying negative controls (required for prevalence/combined)
    neg_control_value : str or bool, default=True
        Value indicating negative controls
    threshold : float, default=0.1
        P-value threshold for detection
    remove_contam : bool, default=True
        Whether to remove contaminants from returned AnnData
    output_dir : Path, optional
        Directory to save results and plots
    
    Returns
    -------
    tuple
        (cleaned_adata, contaminant_df)
        
        If remove_contam=False, returns original adata
    
    Examples
    --------
    >>> # Full workflow with both methods
    >>> clean, contam = decontam_workflow(
    ...     adata,
    ...     method='combined',
    ...     concentration_col='dna_conc',
    ...     neg_control_col='sample_type',
    ...     neg_control_value='blank',
    ...     output_dir=Path('qc/decontam')
    ... )
    >>> 
    >>> # Just prevalence (if no DNA conc available)
    >>> clean, contam = decontam_workflow(
    ...     adata,
    ...     method='prevalence',
    ...     neg_control_col='sample_type',
    ...     neg_control_value='blank'
    ... )
    """
    logger.info("=" * 70)
    logger.info("DECONTAM WORKFLOW: Identifying Laboratory Contaminants")
    logger.info("=" * 70)
    
    # 1. Run appropriate detection method
    if method == 'frequency':
        if not concentration_col:
            raise ValueError("concentration_col required for frequency method")
        
        contam_df = identify_contaminants_frequency(
            adata, concentration_col, threshold=threshold
        )
    
    elif method == 'prevalence':
        if not neg_control_col:
            raise ValueError("neg_control_col required for prevalence method")
        
        contam_df = identify_contaminants_prevalence(
            adata, neg_control_col, neg_control_value, threshold=threshold
        )
    
    elif method == 'combined':
        if not concentration_col or not neg_control_col:
            raise ValueError(
                "Both concentration_col and neg_control_col required for combined method"
            )
        
        contam_df = identify_contaminants_combined(
            adata, concentration_col, neg_control_col, 
            neg_control_value, threshold=threshold
        )
    
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose 'frequency', 'prevalence', or 'combined'"
        )
    
    # 2. Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save contaminant table
        csv_path = output_dir / 'identified_contaminants.csv'
        contam_df.to_csv(csv_path, index=False)
        logger.info(f"Saved contaminant list: {csv_path}")
        
        # Plot distribution (if negative controls available)
        if neg_control_col:
            plot_path = output_dir / 'contaminant_distribution.png'
            plot_contaminant_distribution(
                adata, contam_df, neg_control_col, neg_control_value,
                output_path=plot_path
            )
    
    # 3. Remove contaminants if requested
    if remove_contam:
        clean_adata = remove_contaminants(adata, contam_df)
        
        logger.info(
            f"Contaminant removal complete: "
            f"{adata.n_vars} → {clean_adata.n_vars} features"
        )
        
        return clean_adata, contam_df
    else:
        return adata, contam_df
