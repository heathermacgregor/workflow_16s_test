# ==================================================================================== #
# diversity/beta/dispersion.py
# Homogeneity of Dispersion Testing (PERMDISP)
# ==================================================================================== #

from typing import Dict, Any, Optional
import pandas as pd
from skbio.stats.distance import permdisp, DistanceMatrix
from workflow_16s.utils.logger import get_logger

# ==================================================================================== #

def run_permdisp(
    distance_matrix: DistanceMatrix,
    grouping: pd.Series,
    permutations: int = 999
) -> Optional[Dict[str, Any]]:
    """
    Test for homogeneity of multivariate dispersions across groups.
    
    PERMDISP tests whether the dispersion (variance) of samples within groups
    is homogeneous. This is critical to run BEFORE interpreting PERMANOVA results,
    as PERMANOVA can be confounded by differences in group dispersions rather than
    true location effects.
    
    Parameters
    ----------
    distance_matrix : DistanceMatrix
        Distance matrix (e.g., Bray-Curtis, Euclidean)
    grouping : pd.Series
        Categorical grouping variable aligned with distance matrix IDs
    permutations : int, optional
        Number of permutations for statistical test, by default 999
        
    Returns
    -------
    Dict[str, Any] or None
        Results dictionary containing:
        - test_statistic: F-statistic for dispersion differences
        - p_value: Permutational p-value
        - interpretation: Guidance on PERMANOVA validity
        - warning: Flag if dispersion differences detected
        
    Notes
    -----
    - If p < 0.05, groups have significantly different dispersions
    - Significant PERMDISP invalidates PERMANOVA location interpretation
    - Consider using alternative tests (e.g., ANOSIM) if dispersions differ
    
    References
    ----------
    Anderson, M.J. (2006). Distance-based tests for homogeneity of multivariate
    dispersions. Biometrics 62(1):245-253.
    """
    logger = get_logger("workflow_16s")
    try:
        # Align distance matrix and grouping
        common_ids = distance_matrix.ids.intersection(grouping.index) # type: ignore
        if len(common_ids) < 3:
            logger.warning("PERMDISP requires ≥3 samples. Skipping.")
            return None
            
        # Filter to common samples
        dm_subset = distance_matrix.filter(common_ids, strict=False)
        grouping_subset = grouping.loc[common_ids]
        
        # Remove groups with < 2 samples
        group_counts = grouping_subset.value_counts()
        valid_groups = group_counts[group_counts >= 2].index
        if len(valid_groups) < 2:
            logger.warning("PERMDISP requires ≥2 groups with ≥2 samples each. Skipping.")
            return None
            
        mask = grouping_subset.isin(valid_groups)
        dm_subset = dm_subset.filter(grouping_subset[mask].index, strict=False)
        grouping_subset = grouping_subset[mask]
        
        # Run PERMDISP
        result = permdisp(dm_subset, grouping_subset, permutations=permutations)
        
        p_val = result['p-value']
        f_stat = result['test statistic']
        
        # Interpretation
        if p_val < 0.05:
            interpretation = (
                "⚠️ Significant dispersion differences detected (p < 0.05). "
                "PERMANOVA results may be confounded by dispersion rather than location effects. "
                "Consider using ANOSIM or checking PERMANOVA results cautiously."
            )
            warning = True
        else:
            interpretation = (
                "✓ No significant dispersion differences (p ≥ 0.05). "
                "PERMANOVA results can be interpreted as true location effects."
            )
            warning = False
            
        logger.info(
            f"PERMDISP: F={f_stat:.4f}, p={p_val:.4e} | "
            f"{'WARNING: Dispersion differs' if warning else 'OK: Homogeneous dispersion'}"
        )
        
        return {
            'test_statistic': f_stat,
            'p_value': p_val,
            'interpretation': interpretation,
            'warning': warning,
            'n_groups': len(valid_groups),
            'n_samples': len(grouping_subset)
        }
        
    except Exception as e:
        logger.error(f"PERMDISP failed: {e}")
        return None


def check_permanova_validity(
    permanova_result: Dict[str, Any],
    permdisp_result: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Cross-validate PERMANOVA results with PERMDISP dispersion test.
    
    Parameters
    ----------
    permanova_result : Dict[str, Any]
        PERMANOVA results with 'p-value' key
    permdisp_result : Optional[Dict[str, Any]]
        PERMDISP results from run_permdisp()
        
    Returns
    -------
    Dict[str, Any]
        Validation results with interpretation guidance
    """
    if permdisp_result is None:
        return {
            'valid': 'unknown',
            'message': 'PERMDISP not run - cannot validate PERMANOVA'
        }
    
    permanova_sig = permanova_result.get('p-value', 1.0) < 0.05
    dispersion_sig = permdisp_result.get('warning', False)
    
    if permanova_sig and dispersion_sig:
        return {
            'valid': False,
            'message': (
                '⚠️ PERMANOVA significant BUT dispersion differs between groups. '
                'Results may reflect variance differences rather than compositional differences. '
                'Recommend: (1) Check ANOSIM, (2) Use pairwise tests, (3) Transform data differently.'
            )
        }
    elif permanova_sig and not dispersion_sig:
        return {
            'valid': True,
            'message': (
                '✓ PERMANOVA significant AND dispersion is homogeneous. '
                'Results reflect true compositional differences between groups.'
            )
        }
    elif not permanova_sig and dispersion_sig:
        return {
            'valid': 'inconclusive',
            'message': (
                'PERMANOVA not significant but dispersion differs. '
                'Groups may have different variances but similar centroids.'
            )
        }
    else:
        return {
            'valid': True,
            'message': 'No significant differences detected in composition or dispersion.'
        }
