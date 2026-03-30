# src/workflow_16s/downstream/machine_learning/batch_control/confounding.py

import anndata as ad
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from scipy.stats import spearmanr, chi2_contingency
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from workflow_16s.visualization.machine_learning.batch_dependency import plot_confounding_heatmap
from workflow_16s.utils.logger import get_logger


def detect_confounding(
    adata: ad.AnnData,
    batch_columns: List[str],
    target_col: str,
    sample_indices: pd.Index,
    threshold: float = 0.7,
    plot_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Detect confounding between batch variables and target variable.
    
    METHODOLOGY:
    - Spearman ρ: For Numeric-Numeric (Monotonic associations)
    - Bias-Corrected Cramér's V: For Categorical-Categorical
    - Eta-Squared: For Mixed types (ANOVA effect size)
    
    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix with metadata in .obs.
    batch_columns : List[str]
        List of batch variable column names in adata.obs.
    target_col : str
        Target variable column name in adata.obs.
    sample_indices : pd.Index
        Indices of samples to consider for confounding analysis.
    threshold : float
        Threshold above which confounding is considered high.
    plot_dir : Optional[Path]
        Directory to save confounding diagnostic plots.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing confounding results and statistics.
    """
    logger = get_logger("workflow_16s")
    plot_utils = PlottingUtils(logger)
    confounding_results = {
        'target': target_col,
        'high_confounding': [],
        'moderate_confounding': [],
        'low_confounding': [],
        'statistics': {}
    }
    
    target_data = adata.obs.loc[sample_indices, target_col].dropna()
    if target_data.empty:
        logger.warning(f"Target column '{target_col}' has no valid data for the selected indices.")
        return confounding_results

    target_is_numeric = pd.api.types.is_numeric_dtype(target_data)
    
    for batch_col in batch_columns:
        if batch_col not in adata.obs.columns or batch_col == target_col:
            continue
        
        batch_data = adata.obs.loc[sample_indices, batch_col]
        # Align target and batch indices
        common_idx = target_data.index.intersection(batch_data.dropna().index)
        
        if len(common_idx) < 10:
            continue
        
        target_common = target_data.loc[common_idx]
        batch_common = batch_data.loc[common_idx]
        batch_is_numeric = pd.api.types.is_numeric_dtype(batch_common)
        
        association_value = None
        association_type = None
        pval = None
        
        try:
            # 1. Numeric-Numeric: Spearman correlation
            if target_is_numeric and batch_is_numeric:
                corr, pval = spearmanr(target_common, batch_common, nan_policy='omit')
                # Critical Fix: Handle scalar vs array return from scipy
                if isinstance(corr, (np.ndarray, list)):
                    corr = np.atleast_1d(corr)[0]
                # Ensure corr is a scalar convertible to float
                corr_scalar = np.asarray(corr).item() if hasattr(corr, 'item') else corr
                if isinstance(corr_scalar, (int, float, np.integer, np.floating)):
                    association_value = abs(float(corr_scalar)) if pd.notnull(corr_scalar) else 0
                else:
                    association_value = 0
                association_type = 'spearman'

            # 2. Categorical-Categorical: Cramér's V
            elif not target_is_numeric and not batch_is_numeric:
                contingency = pd.crosstab(target_common, batch_common)
                if contingency.size > 1:
                    chi2, pval, _, _ = chi2_contingency(contingency)
                    n = contingency.sum().sum()
                    
                    # Bias-corrected Cramér's V logic
                    phi2 = chi2 / n
                    r, k = contingency.shape
                    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
                    r_corr = r - ((r-1)**2)/(n-1)
                    k_corr = k - ((k-1)**2)/(n-1)
                    
                    association_value = np.sqrt(phi2_corr / min((k_corr-1), (r_corr-1)))
                    association_type = 'cramers_v'

            # 3. Mixed: Eta-squared (ANOVA effect size)
            else:
                num_var = target_common if target_is_numeric else batch_common
                cat_var = batch_common.astype(str) if target_is_numeric else target_common.astype(str)
                
                groups = [num_var[cat_var == cat].values for cat in cat_var.unique()]
                groups = [g for g in groups if len(g) > 1]
                
                if len(groups) >= 2:
                    grand_mean = num_var.mean()
                    ss_total = np.sum((num_var - grand_mean)**2)
                    ss_between = np.sum([len(g) * (np.mean(g) - grand_mean)**2 for g in groups])
                    
                    association_value = ss_between / ss_total if ss_total > 0 else 0
                    association_type = 'eta_squared'
                    pval = None # ANOVA p-value would require scipy.stats.f_oneway
        
        except Exception as e:
            logger.debug(f"Association failed for {batch_col}: {e}")
            continue
        
        if association_value is not None:
            confounding_results['statistics'][batch_col] = {
                'value': float(association_value),
                'type': association_type,
                'p_value': pval
            }
            
            # Categorize Confounding Level
            if association_value >= threshold:
                confounding_results['high_confounding'].append(batch_col)
                logger.warning(f"🔴 HIGH CONFOUNDING: '{batch_col}' -> '{target_col}' ({association_type}={association_value:.3f})")
            elif association_value >= (threshold * 0.7):
                confounding_results['moderate_confounding'].append(batch_col)
                logger.info(f"🟡 MODERATE CONFOUNDING: '{batch_col}' -> '{target_col}' ({association_value:.3f})")
            else:
                confounding_results['low_confounding'].append(batch_col)
    
    # 4. Visualization
    if plot_dir and confounding_results['statistics']:
        plot_confounding_heatmap(confounding_results, plot_dir, target_col, threshold)
    
    return confounding_results

