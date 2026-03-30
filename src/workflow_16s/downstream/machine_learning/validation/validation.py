# src/workflow_16s/downstream/machine_learning/validation.py

import json
import numpy as np
import pandas as pd
import scipy.sparse
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import matthews_corrcoef, r2_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, CatBoostRegressor

from workflow_16s.utils.logger import get_logger
from workflow_16s.downstream.utils import AnalysisUtils
from .utils import resolve_feature_names, clean_feature_names


# ============================================================================
# LAYER 1: CHANCE VALIDATION (SHUFFLE TEST)
# ============================================================================

def run_shuffle_baseline(
    adata, 
    target_col: str, 
    output_dir: Path, 
    real_score: float, 
    n_permutations: int = 20, 
    level: str = 'Genus'
) -> float:
    """
    Calculates the 'Microbial Significance p-value'.
    
    METHODOLOGY:
    Randomly shuffles labels (Target) and refits a model N times. 
    $p = (N_{shuffled >= real} + 1) / (N_{permutations} + 1)$
    
    If p > 0.05, the model results are statistically indistinguishable from noise.
    """
    logger = get_logger("workflow_16s")
    logger.info(f" 🎲 Significance Audit: Shuffling {target_col} ({n_permutations} iterations)...")
    
    # 1. Prepare Data
    adata_agg = AnalysisUtils.get_analysis_adata(adata, level=level)
    raw = adata_agg.X.toarray() if scipy.sparse.issparse(adata_agg.X) else adata_agg.X # type: ignore
    feature_names = resolve_feature_names(adata_agg, level)
    
    X = pd.DataFrame(raw, index=adata_agg.obs_names, columns=feature_names) # type: ignore
    X = clean_feature_names(X)
    X.index = X.index.astype(str)
    X = AnalysisUtils.clr_transform_from_df(X, pseudocount=1.0)

    y = adata.obs[target_col].replace(['', 'nan', 'NaN', 'None', '<NA>'], np.nan).dropna()
    common = X.index.intersection(y.index)
    if len(common) < 10: return 1.0

    X, y = X.loc[common], y.loc[common]
    
    # 2. Task Detection
    is_numeric = pd.api.types.is_numeric_dtype(y)
    is_regression = is_numeric and (y.nunique() > 10)
    
    if is_regression:
        y_safe = y.values
    else:
        le = LabelEncoder()
        y_safe = le.fit_transform(y.astype(str))
    
    # 3. Permutation Loop
    shuffled_scores = []
    for i in range(n_permutations):
        y_shuffled = np.random.permutation(y_safe)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X.values, 
            y_shuffled, 
            test_size=0.3, 
            random_state=i
        )
        
        # We use slightly faster parameters for the null distribution checks
        params = {'iterations': 100, 'depth': 4, 'verbose': False, 'allow_writing_files': False}
        
        if is_regression:
            model = CatBoostRegressor(**params).fit(X_tr, y_tr)
            score = model.score(X_te, y_te) # R²
        else:
            model = CatBoostClassifier(**params).fit(X_tr, y_tr)
            preds = model.predict(X_te)
            score = matthews_corrcoef(y_te, preds)
        shuffled_scores.append(score)

    # 4. Results Aggregation
    shuffled_scores = np.array(shuffled_scores)
    p_value = float((np.sum(shuffled_scores >= real_score) + 1) / (n_permutations + 1))
    
    results = {
        'target': target_col, 'real_score': real_score, 
        'mean_shuffled_score': np.mean(shuffled_scores),
        'p_value': p_value, 'shuffled_scores': shuffled_scores.tolist(),
        'metric': 'r2' if is_regression else 'mcc'
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "shuffle_stats.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"🎯 Shuffle Baseline p-value: {p_value:.4f}")
    return p_value

# ============================================================================
# LAYER 2: CROSS-STUDY GENERALIZATION (LOPOCV)
# ============================================================================

def validate_consensus_panel(
    X_final: pd.DataFrame, 
    y_final: pd.Series, 
    batches: np.ndarray, 
    consensus_features: List[str], 
    output_dir: Path
) -> float:
    """
    Validates the 'Universal' Biomarker Panel via Leave-One-Project-Out CV.
    
    RATIONALE:
    A forensic biomarker is only useful if it predicts the target in a 
    geographic region or lab that was NOT part of the training set.
    """
    logger = get_logger("workflow_16s")
    logger.info(f" 🧪 Stress-Testing Panel ({len(consensus_features)} features) via LOPOCV...")
    
    valid_feats = [f for f in consensus_features if f in X_final.columns]
    if len(valid_feats) < 1: return 0.0
    X_panel = X_final[valid_feats]
    
    # 1. Encoding
    is_numeric = pd.api.types.is_numeric_dtype(y_final)
    is_regression = is_numeric and (y_final.nunique() > 10)
    y_encoded = y_final if is_regression else (y_final.astype(str).str.lower() == 'true').astype(int)
    
    logo = LeaveOneGroupOut()
    test_scores = []
    groups_arr = np.array([str(x) for x in batches])
    
    if logo.get_n_splits(X_panel, y_encoded, groups_arr) < 2:
        logger.warning("Not enough studies for LOPOCV validation.")
        return 0.0

    # 2. LOPOCV Loop
    for train_idx, test_idx in logo.split(X_panel, y_encoded, groups_arr):
        X_tr, X_te = X_panel.iloc[train_idx], X_panel.iloc[test_idx]
        y_tr, y_te = y_encoded.iloc[train_idx], y_encoded.iloc[test_idx]
        
        if not is_regression and len(np.unique(y_te)) < 2: continue
            
        try:
            params = {'iterations': 200, 'depth': 4, 'verbose': False, 'allow_writing_files': False}
            if is_regression:
                model = CatBoostRegressor(**params).fit(X_tr, y_tr)
                test_scores.append(r2_score(y_te, model.predict(X_te)))
            else:
                model = CatBoostClassifier(**params, auto_class_weights='Balanced').fit(X_tr, y_tr)
                test_scores.append(matthews_corrcoef(y_te, model.predict(X_te)))
        except Exception: continue
            
    if not test_scores: return 0.0
        
    avg_test = float(np.mean(test_scores))
    logger.info(f"🏁 LOPOCV Consensus Score ({'R2' if is_regression else 'MCC'}): {avg_test:.3f}")
    
    results = {
        "avg_test_score": avg_test, "metric": 'r2' if is_regression else 'mcc',
        "n_features": len(valid_feats), "features": valid_feats
    }
    
    with open(output_dir / "consensus_validation_score.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    return avg_test