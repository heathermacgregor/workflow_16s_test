# feature_selection/reporting.py

import logging
import pandas as pd
import numpy as np
import shap
from typing import Tuple, Union
from pathlib import Path
from scipy.stats import spearmanr
from workflow_16s.visualization._machine_learning import (
    plot_shap,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_precision_recall_curve
)

logger = logging.getLogger('workflow_16s')

def _check_shap_installed():
    """Safety check for library availability."""
    if shap is None: raise ImportError("SHAP not found.")

def generate_shap_report(model, X: pd.DataFrame, K: int = 10) -> Tuple[str, pd.DataFrame]:
    """Generates detailed text and data reports on microbial predictors."""
    _check_shap_installed()
    try: 
        expl = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        sv = expl.shap_values(X) 
        if isinstance(sv, list): sv = sv[1] if len(sv) == 2 else np.mean([np.abs(s) for s in sv], axis=0)
        if sv.ndim == 3: sv = sv.mean(axis=2) 
        
        mean_abs = np.abs(sv).mean(axis=0)
        idx = np.argsort(mean_abs)[::-1][:K]
        top_f = list(X.columns[idx]); top_m = mean_abs[idx]

        lines = [f"Impact Report (Top {K} Taxa):"]
        for f, m in zip(top_f, top_m):
            vals = X[f].values
            corr_result = spearmanr(vals, sv[:, X.columns.get_loc(f)], nan_policy='omit')
            rho = corr_result.correlation if hasattr(corr_result, 'correlation') else corr_result[0]  # type: ignore
            dir_txt = "positive" if rho > 0 else "negative"  # type: ignore
            lines.append(f" • {f}: impact={m:.3f}, correlation={dir_txt} (ρ={rho:.2f})")
            
        return "\n".join(lines), pd.DataFrame({'feature': top_f, 'impact': top_m})
    
    except Exception as e: logger.error(f"SHAP Report Failure: {e}"); return "Error", pd.DataFrame()

def save_feature_importances(model, X_train: pd.DataFrame, output_dir: Path):
    """Persists model importances to CSV."""
    imp = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.zeros(X_train.shape[1])
    pd.Series(imp, index=X_train.columns).sort_values(ascending=False).to_csv(output_dir / "feat_imp.csv")