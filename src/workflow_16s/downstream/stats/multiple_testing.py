# ==================================================================================== #
# statistics/multiple_testing.py
# Multiple Testing Correction Methods
# ==================================================================================== #

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from workflow_16s.utils.logger import get_logger


# ==================================================================================== #

def apply_multiple_testing_correction(
    p_values: np.ndarray,
    method: str = 'fdr_bh',
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply multiple testing correction to p-values.
    
    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values
    method : str, optional
        Correction method:
        - 'bonferroni': Bonferroni correction (most conservative)
        - 'sidak': Sidak correction
        - 'fdr_bh': Benjamini-Hochberg FDR (recommended for microbiome)
        - 'fdr_by': Benjamini-Yekutieli FDR (for dependent tests)
        - 'fdr_tsbh': Two-stage Benjamini-Hochberg
        - 'fdr_tsbky': Two-stage Benjamini-Krieger-Yekutieli
        - 'holm': Holm-Bonferroni
        - 'hommel': Hommel
        - 'simes-hochberg': Simes-Hochberg
        Default is 'fdr_bh'
    alpha : float, optional
        Family-wise error rate or FDR level, by default 0.05
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - reject: Boolean array indicating which tests reject null hypothesis
        - pvals_corrected: Adjusted p-values
        - alphacBonf: Bonferroni corrected alpha (for some methods)
        
    Notes
    -----
    For microbiome differential abundance testing:
    - Use 'fdr_bh' (Benjamini-Hochberg) for independent tests
    - Use 'fdr_by' (Benjamini-Yekutieli) for dependent tests (e.g., taxonomic levels)
    - Bonferroni is too conservative for high-dimensional microbiome data
    
    References
    ----------
    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate.
    Journal of the Royal Statistical Society: Series B, 57(1), 289-300.
    """
    logger = get_logger("workflow_16s")
    # Handle NaN values
    valid_mask = ~np.isnan(p_values)
    n_valid = np.sum(valid_mask)
    
    if n_valid == 0:
        logger.warning("No valid p-values for correction")
        return (
            np.zeros_like(p_values, dtype=bool),
            np.full_like(p_values, np.nan),
            np.array([alpha])
        )
    
    # Initialize output arrays
    reject = np.zeros_like(p_values, dtype=bool)
    pvals_corrected = np.full_like(p_values, np.nan, dtype=float)
    
    # Apply correction only to valid p-values
    valid_pvals = p_values[valid_mask]
    
    try:
        reject_valid, pvals_valid, alphacSidak, alphacBonf = multipletests(
            valid_pvals, 
            alpha=alpha, 
            method=method, 
            is_sorted=False, 
            returnsorted=False
        )
        
        # Map back to original array
        reject[valid_mask] = reject_valid
        pvals_corrected[valid_mask] = pvals_valid
        
        # Log summary
        n_significant = np.sum(reject_valid)
        pct_significant = (n_significant / n_valid) * 100
        
        logger.info(
            f"Multiple testing correction: {method} | "
            f"α={alpha} | "
            f"{n_significant}/{n_valid} significant ({pct_significant:.1f}%)"
        )
        
        return reject, pvals_corrected, np.array([alphacBonf])
        
    except Exception as e:
        logger.error(f"Multiple testing correction failed: {e}")
        return (
            np.zeros_like(p_values, dtype=bool),
            np.full_like(p_values, np.nan),
            np.array([alpha])
        )


def compare_correction_methods(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compare multiple correction methods side-by-side.
    
    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values
    alpha : float, optional
        Significance threshold, by default 0.05
        
    Returns
    -------
    pd.DataFrame
        Comparison table with number of discoveries per method
    """
    logger = get_logger("workflow_16s")
    methods = [
        'bonferroni',
        'sidak', 
        'fdr_bh',
        'fdr_by',
        'holm',
        'hommel'
    ]
    
    results = []
    
    for method in methods:
        reject, pvals_corrected, _ = apply_multiple_testing_correction(
            p_values, 
            method=method, 
            alpha=alpha
        )
        
        n_significant = np.sum(reject)
        pct_significant = (n_significant / len(p_values)) * 100
        
        results.append({
            'method': method,
            'n_significant': n_significant,
            'pct_significant': pct_significant,
            'mean_adjusted_p': np.nanmean(pvals_corrected)
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('n_significant', ascending=False)
    
    logger.info("\n=== Multiple Testing Correction Comparison ===")
    logger.info(f"\n{df.to_string(index=False)}")
    
    return df


def stratified_fdr_correction(
    p_values: np.ndarray,
    strata: np.ndarray,
    method: str = 'fdr_bh',
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply FDR correction within strata (e.g., per taxonomic level).
    
    When testing multiple taxonomic levels separately, apply FDR correction
    within each level to avoid losing power from cross-level multiplicity.
    
    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values
    strata : np.ndarray
        Stratum labels (e.g., taxonomic level for each test)
    method : str, optional
        Correction method, by default 'fdr_bh'
    alpha : float, optional
        FDR level, by default 0.05
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - reject: Boolean array of rejections
        - pvals_corrected: Adjusted p-values
        
    Example
    -------
    >>> p_vals = np.array([0.001, 0.05, 0.001, 0.06])
    >>> levels = np.array(['genus', 'genus', 'family', 'family'])
    >>> reject, pvals_adj = stratified_fdr_correction(p_vals, levels)
    """
    logger = get_logger("workflow_16s")
    unique_strata = np.unique(strata)
    
    reject = np.zeros_like(p_values, dtype=bool)
    pvals_corrected = np.full_like(p_values, np.nan, dtype=float)
    
    for stratum in unique_strata:
        mask = strata == stratum
        
        if np.sum(mask) == 0:
            continue
        
        stratum_pvals = p_values[mask]
        
        stratum_reject, stratum_corrected, _ = apply_multiple_testing_correction(
            stratum_pvals, 
            method=method, 
            alpha=alpha
        )
        
        reject[mask] = stratum_reject
        pvals_corrected[mask] = stratum_corrected
        
        n_sig = np.sum(stratum_reject)
        logger.info(f"Stratum '{stratum}': {n_sig}/{np.sum(mask)} significant after {method}")
    
    return reject, pvals_corrected


def effective_number_of_tests(correlation_matrix: np.ndarray) -> float:
    """
    Estimate effective number of independent tests accounting for correlation.
    
    When features are correlated (as in microbiome data), the effective number
    of tests is less than the nominal number. This can be used to adjust
    Bonferroni or FDR thresholds.
    
    Parameters
    ----------
    correlation_matrix : np.ndarray
        Feature correlation matrix
        
    Returns
    -------
    float
        Effective number of independent tests
        
    Notes
    -----
    Based on eigenvalue decomposition. More sophisticated than simple Bonferroni.
    
    References
    ----------
    Nyholt, D. R. (2004). A simple correction for multiple testing for 
    single-nucleotide polymorphisms in linkage disequilibrium with each other.
    The American Journal of Human Genetics, 74(4), 765-769.
    """
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(correlation_matrix)
    eigenvalues = eigenvalues[eigenvalues > 0]  # Keep positive eigenvalues
    
    # Effective number of tests (variance explained)
    M_eff = 1 + (len(eigenvalues) - 1) * (1 - np.var(eigenvalues) / len(eigenvalues))
    
    get_logger("workflow_16s").info(
        f"Nominal tests: {correlation_matrix.shape[0]} | "
        f"Effective tests: {M_eff:.1f}"
    )
    
    return float(M_eff)


import numpy as np
from statsmodels.stats.multitest import multipletests
from typing import Dict, List

def hierarchical_fdr(
    p_values: np.ndarray,
    children_map: Dict[int, List[int]],
    root_indices: List[int],
    alpha: float = 0.05
) -> np.ndarray:
    """
    Hierarchical FDR correction for nested hypotheses (Yekutieli, 2008).
    
    For taxonomic data with nested structure (Kingdom > Phylum > Class > ...),
    apply hierarchical FDR that respects the tree structure. A child hypothesis 
    is only tested (and potentially rejected) if its direct parent is rejected.
    
    Parameters
    ----------
    p_values : np.ndarray
        1D array of P-values for all tests.
    children_map : Dict[int, List[int]]
        Dictionary mapping a parent index to a list of its children's indices.
    root_indices : List[int]
        List of indices representing the root nodes (e.g., Kingdom level).
    alpha : float, optional
        FDR level, by default 0.05.
        
    Returns
    -------
    np.ndarray
        Boolean array of rejections respecting the hierarchy.
        
    References
    ----------
    Yekutieli, D. (2008). Hierarchical false discovery rate-controlling methodology.
    Journal of the American Statistical Association, 103(481), 309-316.
    """
    reject = np.zeros_like(p_values, dtype=bool)
    
    # We use a queue to test families iteratively. Start with the root family.
    families_to_test = [root_indices]
    
    while families_to_test:
        current_family = families_to_test.pop(0)
        
        if not current_family:
            continue
            
        # Extract p-values for the current family
        family_pvals = p_values[current_family]
        
        # Apply standard Benjamini-Hochberg FDR to this specific family
        family_reject, _, _, _ = multipletests(
            family_pvals, 
            alpha=alpha, 
            method='fdr_bh'
        )
        
        # Record rejections and queue children of the REJECTED hypotheses
        for idx, is_rejected in zip(current_family, family_reject):
            reject[idx] = is_rejected
            
            # If rejected, its children form a new family to be tested
            if is_rejected and idx in children_map:
                families_to_test.append(children_map[idx])
                
    return reject


def export_fdr_results(
    feature_names: List[str],
    p_values: np.ndarray,
    method: str = 'fdr_bh',
    alpha: float = 0.05,
    additional_data: Optional[Dict[str, np.ndarray]] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Export FDR-corrected results to a formatted table.
    
    Parameters
    ----------
    feature_names : List[str]
        Names of features tested
    p_values : np.ndarray
        Raw p-values
    method : str, optional
        Correction method, by default 'fdr_bh'
    alpha : float, optional
        Significance threshold, by default 0.05
    additional_data : Optional[Dict[str, np.ndarray]], optional
        Additional columns (e.g., fold-change, effect size)
    output_path : Optional[str], optional
        Path to save CSV, by default None
        
    Returns
    -------
    pd.DataFrame
        Results table with corrected p-values and significance flags
    """
    reject, pvals_corrected, _ = apply_multiple_testing_correction(
        p_values, method=method, alpha=alpha
    )
    
    results_df = pd.DataFrame({
        'feature': feature_names,
        'p_value_raw': p_values,
        'p_value_adjusted': pvals_corrected,
        'significant': reject
    })
    
    # Add additional data
    if additional_data:
        for col_name, col_data in additional_data.items():
            results_df[col_name] = col_data
    
    # Sort by adjusted p-value
    results_df = results_df.sort_values('p_value_adjusted')
    
    # Add rank
    results_df.insert(0, 'rank', range(1, len(results_df) + 1))
    
    if output_path:
        results_df.to_csv(output_path, index=False)
        get_logger("workflow_16s").info(f"FDR results exported to: {output_path}")
    
    # Log summary
    n_sig = np.sum(reject)
    get_logger("workflow_16s").info(
        f"\n=== FDR-Corrected Results ({method}) ===\n"
        f"Significant features: {n_sig}/{len(feature_names)} ({n_sig/len(feature_names)*100:.1f}%)\n"
        f"Top 5 significant:\n{results_df.head().to_string(index=False)}"
    )
    
    return results_df
