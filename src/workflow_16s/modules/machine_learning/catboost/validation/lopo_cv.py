# workflow_16s/modules/machine_learning/catboost/validation/lopo_cv.py

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import matthews_corrcoef, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder

from workflow_16s.utils.logger import get_logger, with_logger
logger = get_logger("workflow_16s")
@with_logger
def validate_consensus_panel(
    X_final: pd.DataFrame, y_final: pd.Series, batches: np.ndarray, 
    consensus_features: List[str], output_dir: Path
) -> float:
    """
    Validates the 'Universal' Biomarker Panel via Leave-One-Project-Out CV.
    
    RATIONALE:
    A forensic biomarker is only useful if it predicts the target in a 
    geographic region or lab that was NOT part of the training set.
    """
    logger.info(f" 🧪 Stress-Testing Panel ({len(consensus_features)} features) via LOPOCV...")
    
    valid_feats = [f for f in consensus_features if f in X_final.columns]
    if len(valid_feats) < 1: 
        logger.warning("No consensus features found in the final dataset. Skipping LOPOCV.")
        return 0.0
        
    X_panel = X_final[valid_feats]
    
    # 1. Safe Target Encoding
    is_numeric = pd.api.types.is_numeric_dtype(y_final)
    is_regression = is_numeric and (y_final.nunique() > 10)
    
    if is_regression:
        y_encoded = y_final
    else:
        # 💡 FIX: Safely encode arbitrary strings ("Control", "Disease") to integers
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y_final.astype(str)), index=y_final.index)
    
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
        
        # Safety net: If the held-out study only has one class, MCC calculation will crash/be 0
        if not is_regression and len(np.unique(y_te)) < 2: 
            logger.debug(f"Skipping fold: Held-out study only contains 1 class.")
            continue
            
        try:
            params = {'iterations': 200, 'depth': 4, 'verbose': False, 'allow_writing_files': False}
            if is_regression:
                model = CatBoostRegressor(**params).fit(X_tr, y_tr)
                test_scores.append(r2_score(y_te, model.predict(X_te)))
            else:
                # auto_class_weights='Balanced' handles imbalanced batches beautifully
                model = CatBoostClassifier(**params, auto_class_weights='Balanced').fit(X_tr, y_tr)
                test_scores.append(matthews_corrcoef(y_te, model.predict(X_te)))
        except Exception as e: 
            logger.error(f"LOPOCV Fold Failed: {e}")
            continue
            
    if not test_scores: 
        logger.warning("LOPOCV completed but no valid test scores were generated.")
        return 0.0
        
    avg_test = float(np.mean(test_scores))
    logger.info(f"🏁 LOPOCV Consensus Score ({'R2' if is_regression else 'MCC'}): {avg_test:.3f}")
    
    results = {
        "avg_test_score": avg_test, "metric": 'r2' if is_regression else 'mcc',
        "n_features": len(valid_feats), "features": valid_feats
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "consensus_validation_score.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    return avg_test
