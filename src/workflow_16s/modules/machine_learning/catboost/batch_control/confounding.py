# workflow_16s/modules/machine_learning/catboost/batch_control/confounding.py

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, spearmanr

from workflow_16s.visualization.machine_learning.batch_dependency import (
    plot_confounding_heatmap
)
from workflow_16s.utils.logger import get_logger, with_logger
logger = get_logger("workflow_16s")

def _is_effectively_categorical(series: pd.Series, unique_threshold: int = 15) -> bool:
    """
    Determines if a series should be treated as categorical for statistical testing.
    Catches integer-encoded categories (e.g., [0, 1, 2]) that Pandas thinks are numeric.
    """
    if series.dtype == 'object' or isinstance(series.dtype, pd.CategoricalDtype):
        return True
    # If it's numeric but has very few unique values, treat it as categorical
    if pd.api.types.is_integer_dtype(series) and series.nunique() <= unique_threshold:
        return True
    return False

@with_logger
def detect_confounding(
    adata: ad.AnnData, batch_columns: List[str], target_col: str,
    sample_indices: pd.Index, threshold: float = 0.7, plot_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Detect confounding between batch variables and target variable.
    
    METHODOLOGY:
    - Spearman ρ: For Numeric-Numeric
    - Bias-Corrected Cramér's V: For Categorical-Categorical
    - Eta-Squared (η²): For Mixed types
    """
    confounding_results = {
        'target': target_col,
        'high_confounding': [],
        'moderate_confounding': [],
        'low_confounding': [],
        'statistics': {}
    }
    
    target_data = adata.obs.loc[sample_indices, target_col].dropna()
    if target_data.empty:
        return confounding_results

    target_is_cat = _is_effectively_categorical(target_data)
    
    for batch_col in batch_columns:
        if batch_col not in adata.obs.columns or batch_col == target_col:
            continue
        
        batch_data = adata.obs.loc[sample_indices, batch_col]
        common_idx = target_data.index.intersection(batch_data.dropna().index)
        
        if len(common_idx) < 10:
            continue
        
        target_common = target_data.loc[common_idx]
        batch_common = batch_data.loc[common_idx]
        batch_is_cat = _is_effectively_categorical(batch_common)
        
        association_value, association_type, pval = None, None, None
        
        try:
            # 1. Categorical-Categorical: Bias-Corrected Cramér's V
            if target_is_cat and batch_is_cat:
                contingency = pd.crosstab(target_common, batch_common)
                if contingency.size > 1 and contingency.shape[0] > 1 and contingency.shape[1] > 1:
                    chi2, pval, _, _ = chi2_contingency(contingency)
                    n = contingency.sum().sum()
                    phi2 = chi2 / n
                    r, k = contingency.shape
                    
                    # Bias correction
                    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
                    r_corr = r - ((r-1)**2)/(n-1)
                    k_corr = k - ((k-1)**2)/(n-1)
                    
                    denom = min((k_corr-1), (r_corr-1))
                    association_value = np.sqrt(phi2_corr / denom) if denom > 0 else 0.0
                    association_type = 'cramers_v'

            # 2. Numeric-Numeric: Spearman Correlation
            elif not target_is_cat and not batch_is_cat:
                corr, pval = spearmanr(target_common, batch_common, nan_policy='omit')
                if isinstance(corr, (np.ndarray, list)):
                    corr = np.atleast_1d(corr)[0]
                corr_scalar = np.asarray(corr).item() if hasattr(corr, 'item') else corr
                
                if pd.notnull(corr_scalar):
                    association_value = abs(float(corr_scalar))
                    association_type = 'spearman'

            # 3. Mixed Types: Eta-Squared (η²)
            else:
                num_var = target_common if not target_is_cat else batch_common
                cat_var = batch_common.astype(str) if not target_is_cat else target_common.astype(str)
                
                groups = [num_var[cat_var == cat].values for cat in cat_var.unique()]
                groups = [g for g in groups if len(g) > 1]
                
                if len(groups) >= 2:
                    grand_mean = num_var.mean()
                    ss_total = np.sum((num_var - grand_mean)**2)
                    ss_between = np.sum([len(g) * (np.mean(g) - grand_mean)**2 for g in groups])
                    
                    association_value = ss_between / ss_total if ss_total > 0 else 0.0
                    association_type = 'eta_squared'
                    
        except Exception as e:
            # Silently skip if a specific statistical test fails due to edge-case data shapes
            continue
        
        if association_value is not None:
            confounding_results['statistics'][batch_col] = {
                'value': float(association_value),
                'type': association_type,
                'p_value': pval
            }
            
            # Routing into severity buckets
            if association_value >= threshold:
                confounding_results['high_confounding'].append(batch_col)
            elif association_value >= (threshold * 0.7):
                confounding_results['moderate_confounding'].append(batch_col)
            else:
                confounding_results['low_confounding'].append(batch_col)
    
    # Visualization trigger
    if plot_dir and confounding_results['statistics']:
        plot_confounding_heatmap(confounding_results, plot_dir, target_col, threshold)
        
    return confounding_results
