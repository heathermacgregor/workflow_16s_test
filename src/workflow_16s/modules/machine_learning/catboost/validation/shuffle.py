# workflow_16s/modules/machine_learning/catboost/validation/shuffle.py

import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import matthews_corrcoef, r2_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder

from workflow_16s.modules.machine_learning.catboost.feature_selection.validation import (
    check_compositionality_transformation
)
from workflow_16s.modules.machine_learning.catboost.utils import (
    clean_feature_names, resolve_feature_names
)
from workflow_16s.utils.analysis import AnalysisUtils
from workflow_16s.utils.logger import get_logger, with_logger
logger = get_logger("workflow_16s")

@with_logger
def run_shuffle_baseline(
    adata: ad.AnnData, target_col: str, output_dir: Path, real_score: float, 
    n_permutations: int = 20, level: str = 'Genus'
) -> float:
    """
    Calculates the 'Microbial Significance p-value'.
    """
    logger.info(f" 🎲 Significance Audit: Shuffling {target_col} ({n_permutations} iterations)...")
    
    # 1. Prepare Data
    adata_agg = AnalysisUtils.get_analysis_adata(adata, level=level)
    raw = adata_agg.X.toarray() if scipy.sparse.issparse(adata_agg.X) else adata_agg.X # type: ignore
    feature_names = resolve_feature_names(adata_agg, level)
    
    X = pd.DataFrame(raw, index=adata_agg.obs_names, columns=feature_names) # type: ignore
    X = clean_feature_names(X)
    X.index = X.index.astype(str)
    
    # 🚨 REMOVED: X = AnalysisUtils.clr_transform_from_df(X, pseudocount=1.0)
    
    # 👉 NEW: Warn the user if the data coming from AnnData doesn't look CLR-transformed
    check_compositionality_transformation(X)

    y = adata.obs[target_col].replace(['', 'nan', 'NaN', 'None', '<NA>'], np.nan).dropna()
    common = X.index.intersection(y.index)
    if len(common) < 10: return 1.0

    X, y = X.loc[common], y.loc[common]
    
    # 2. Task Detection & Safe Encoding
    is_numeric = pd.api.types.is_numeric_dtype(y)
    is_regression = is_numeric and (y.nunique() > 10)
    
    if is_regression:
        y_safe = y.values
    else:
        # Safe string encoding for classification
        le = LabelEncoder()
        y_safe = le.fit_transform(y.astype(str))
    
    # 3. Permutation Loop
    shuffled_scores = []
    
    # 💡 UPGRADE: Using a fast 3-fold CV instead of a single split for a more stable null distribution
    cv = KFold(n_splits=3, shuffle=True) if is_regression else StratifiedKFold(n_splits=3, shuffle=True)
    params = {'iterations': 100, 'depth': 4, 'verbose': False, 'allow_writing_files': False}
    
    model = CatBoostRegressor(**params) if is_regression else CatBoostClassifier(**params)
    scoring = 'r2' if is_regression else 'matthews_corrcoef' # Note: make sure you use 'make_scorer(matthews_corrcoef)' if passing to cross_val_score natively

    from sklearn.metrics import make_scorer
    scorer = make_scorer(r2_score) if is_regression else make_scorer(matthews_corrcoef)

    for i in range(n_permutations):
        # Shuffle the target labels
        np.random.seed(i)
        y_shuffled = np.random.permutation(y_safe)
        
        # Calculate the score across folds to reduce split-variance
        scores = cross_val_score(model, X.values, y_shuffled, cv=cv, scoring=scorer, n_jobs=-1)
        shuffled_scores.append(np.mean(scores))

    # 4. Results Aggregation
    shuffled_scores = np.array(shuffled_scores)
    
    # Calculate permutation p-value
    p_value = float((np.sum(shuffled_scores >= real_score) + 1) / (n_permutations + 1))
    
    results = {
        'target': target_col, 'real_score': real_score, 
        'mean_shuffled_score': float(np.mean(shuffled_scores)),
        'p_value': p_value, 'shuffled_scores': shuffled_scores.tolist(),
        'metric': 'r2' if is_regression else 'mcc'
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "shuffle_stats.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"🎯 Shuffle Baseline p-value: {p_value:.4f}")
    return p_value
