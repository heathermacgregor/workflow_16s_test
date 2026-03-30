"""Variance Partitioning Analysis

Wraps R vegan::varpart() for community variation partitioning.
Decomposes multivariate variance into fractions attributed to different factor groups.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import anndata
from dataclasses import dataclass

logger = logging.getLogger("workflow_16s")

try:
    import rpy2
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri, numpy2ri
    import rpy2.robjects as ro
    RLIBS_AVAILABLE = True
except ImportError:
    RLIBS_AVAILABLE = False


@dataclass
class VarpartResult:
    """Results from variance partitioning"""
    fractions: pd.DataFrame  # Variance fractions for each combination
    total_var: float  # Total explained variance
    parts: Dict[str, float]  # Named parts (a, b, c, d, etc.)
    raw_result: Optional[object] = None  # Raw R object for inspection


class VariancePartitioningAnalyzer:
    """Wrapper for R vegan variance partitioning"""
    
    def __init__(self, logger_obj=None):
        """
        Initialize analyzer.
        
        Args:
            logger_obj: Logger instance
            
        Raises:
            ImportError: If rpy2 or vegan not available
        """
        self.logger_obj = logger_obj or logger
        
        if not RLIBS_AVAILABLE:
            raise ImportError(
                "rpy2 required. Install with: pip install rpy2\n"
                "Also requires R package: install.packages('vegan')"
            )
        
        pandas2ri.activate()
        numpy2ri.activate()
        
        try:
            self.vegan = importr('vegan')
            self.logger_obj.info("✓ Loaded R vegan package")
        except Exception as e:
            raise ImportError(f"Failed to load vegan: {e}")
    
    def partition_variance(
        self,
        X: np.ndarray,
        group_matrices: Dict[str, np.ndarray],
        distance_metric: str = 'euclidean',
        method: str = 'capscale',
        logger_obj=None,
    ) -> VarpartResult:
        """
        Partition variance in X into groups.
        
        Uses Redundancy Analysis (RDA) via vegan to partition variance
        explained by multiple groups of explanatory variables.
        
        Args:
            X: Response matrix (samples × features)
            group_matrices: Dict mapping group_name → predictor matrix (samples × predictors)
            distance_metric: Distance metric for RDA ("euclidean", "manhattan")
            method: Ordination method ("rda", "capscale")
            logger_obj: Logger instance
            
        Returns:
            VarpartResult with variance fractions
            
        Example:
            >>> analyzer = VariancePartitioningAnalyzer()
            result = analyzer.partition_variance(
                ko_matrix,
                {
                    'taxonomy': tax_features,
                    'environment': env_features,
                    'functional': func_features,
                }
            )
        """
        logger_obj = logger_obj or self.logger_obj
        
        group_names = list(group_matrices.keys())
        logger_obj.info(
            f"🔧 Partitioning variance into {len(group_names)} groups: {group_names}"
        )
        
        # Convert to R objects
        ro.globalenv['response'] = pandas2ri.py2r(pd.DataFrame(X))
        for name, matrix in group_matrices.items():
            ro.globalenv[f'group_{name}'] = pandas2ri.py2r(pd.DataFrame(matrix))
        
        # Build R formula
        formula_parts = [f'group_{name}' for name in group_names]
        formula_str = ' + '.join(formula_parts)
        
        try:
            # Perform ordination
            ro.r(f"library(vegan)")
            ro.r(f"ord <- rda(response ~ {formula_str}, scale=TRUE)")
            ro.r("summary_ord <- summary(ord)")
            
            # Extract variance explained
            summary_r = ro.r('summary_ord')
            total_inertia = float(ro.r('summary_ord$cont$inertia["Total", "Variance"]')[0])
            
            # Variance per group
            constrainted_inertia = float(
                ro.r('summary_ord$cont$inertia["Conditional", "Variance"]')[0]
                if 'Conditional' in ro.r('rownames(summary_ord$cont$inertia)').rx2(True)
                else 0
            )
            
            # Build results
            parts = {}
            for name in group_names:
                ro.r(f"var_{name} <- RsquareAdj(rda(response ~ group_{name}), zero.missing=TRUE)$r.squared")
                var_val = float(ro.r(f'var_{name}')[0])
                parts[name] = var_val * 100  # Convert to percentage
            
            logger_obj.info(
                f"✅ Variance partitioning complete:\n" +
                "\n".join([f"  {name}: {var:.2f}%" for name, var in parts.items()])
            )
            
            return VarpartResult(
                fractions=pd.DataFrame(parts, index=['variance_%']),
                total_var=total_inertia,
                parts=parts,
            )
            
        except Exception as e:
            logger_obj.error(f"❌ Variance partitioning failed: {e}")
            raise


def perform_variance_partitioning(
    adata: anndata.AnnData,
    outcome_col: str,
    grouping_cols: List[str],
    ko_matrix: Optional[np.ndarray] = None,
    logger_obj=None,
) -> Dict:
    """
    High-level API for variance partitioning.
    
    Args:
        adata: AnnData object
        outcome_col: Column in .obs used as response
        grouping_cols: Columns in .obs to partition by
        ko_matrix: Optional pre-computed KO matrix (uses adata.obsm['KO_counts'] if None)
        logger_obj: Logger instance
        
    Returns:
        Dictionary with results and metadata
    """
    logger_obj = logger_obj or logger
    
    if ko_matrix is None:
        if 'KO_counts' in adata.obsm:
            ko_matrix = adata.obsm['KO_counts']
        else:
            raise ValueError("KO matrix not found in adata.obsm['KO_counts']")
    
    analyzer = VariancePartitioningAnalyzer(logger_obj)
    
    # Extract predictor matrices
    group_matrices = {}
    for col in grouping_cols:
        if col not in adata.obs.columns:
            logger_obj.warning(f"Column {col} not in adata.obs, skipping")
            continue
        
        vals = adata.obs[col].values
        
        # Encode categorical
        if vals.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            vals_encoded = encoder.fit_transform(vals)
            group_matrices[col] = np.column_stack([vals_encoded] * len(encoder.classes_))
        else:
            group_matrices[col] = vals.reshape(-1, 1)
    
    result = analyzer.partition_variance(ko_matrix, group_matrices)
    
    return {
        'result': result,
        'groups': grouping_cols,
    }
