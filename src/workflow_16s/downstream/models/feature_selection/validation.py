# feature_selection/validation.py

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

logger = logging.getLogger('workflow_16s')

def filter_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    metadata: pd.DataFrame, 
    group_col: str, 
    test_size: float = 0.3,  
    random_state: int = 42, 
    min_samples_per_class: int = 2,
    task_type: str = 'Classification',
    cv_groups: Optional[np.ndarray] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Filters data ensuring Group-Level consistency (Batches isolated in Train/Test).
    Prevents 'memorization' of batch effects.
    """
    if not X.index.isin(metadata.index).all(): metadata = metadata.loc[X.index]
    y_labels = metadata.loc[X.index, group_col]
    stratify_target = None 

    if task_type == 'Classification':
        counts = y_labels.value_counts(); keep_idx = counts[counts >= min_samples_per_class].index
        if len(keep_idx) < len(counts):
            logger.info(f"Filtering {len(counts) - len(keep_idx)} rare classes.")
            mask = y_labels.isin(keep_idx); X, y, y_labels = X.loc[mask], y.loc[mask], y_labels.loc[mask]
            if cv_groups is not None: cv_groups = cv_groups[mask.values]  # type: ignore
        if y_labels.nunique() < 2: return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series(), None, None
        stratify_target = y_labels 
    else: 
        mask = y_labels.notna()
        if mask.sum() < len(y_labels):
            X, y = X.loc[mask], y.loc[mask]
            if cv_groups is not None: cv_groups = cv_groups[mask.values]  # type: ignore
        if X.empty: return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series(), None, None

    try:
        if cv_groups is not None:
            # GroupKFold style splitting: isolate unique batches
            unique_groups = np.unique(cv_groups)
            tr_g, te_g = train_test_split(unique_groups, test_size=test_size, random_state=random_state)
            tr_m, te_m = np.isin(cv_groups, tr_g), np.isin(cv_groups, te_g)
            return X[tr_m], X[te_m], y[tr_m], y[te_m], cv_groups[tr_m], cv_groups[te_m]
        
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_target)
        return X_tr, X_te, y_tr, y_te, None, None
    except Exception as e:
        logger.warning(f"Split failed ({e}). Falling back to simple random split.")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_tr, X_te, y_tr, y_te, None, None

def check_for_data_leakage(X_train: pd.DataFrame, X_test: pd.DataFrame, threshold: float = 0.99):
    """Detects accidental feature contamination between training and test sets."""
    for col in X_train.columns:
        if X_train[col].nunique() < 2: continue
        corr, _ = spearmanr(X_train[col].sample(min(100, len(X_train))), 
                            X_test[col].sample(min(100, len(X_test))), nan_policy='omit')
        if abs(corr) > threshold: logger.warning(f"LEAKAGE WARNING: Feature '{col}' has {corr:.2f} correlation across split.")  # type: ignore