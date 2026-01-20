# feature_selection/methods.py

import pandas as pd
import numpy as np
import shap
import logging
from typing import List, Tuple, Dict, Optional, Literal, Callable, Union
from sklearn.feature_selection import RFE, SelectKBest, SelectFromModel, chi2, f_classif, f_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier, CatBoostRegressor

from .reporting import _check_shap_installed

logger = logging.getLogger('workflow_16s')

def rfe_feature_selection(X_train, y_train, X_test, y_test, num_features, step_size, threads, random_state, **kwargs):
    model_cls = CatBoostClassifier if kwargs.get('task_type', 'Classification') == 'Classification' else CatBoostRegressor
    m = model_cls(iterations=500, thread_count=threads, random_state=random_state, verbose=False)
    rfe = RFE(estimator=m, n_features_to_select=min(num_features, X_train.shape[1]), step=step_size, verbose=1)  # type: ignore
    rfe.fit(X_train, y_train); cols = X_train.columns[rfe.support_]
    return pd.DataFrame(rfe.transform(X_train), columns=cols, index=X_train.index), pd.DataFrame(rfe.transform(X_test), columns=cols, index=X_test.index), cols.tolist()  # type: ignore

def select_k_best_feature_selection(X_train, y_train, X_test, y_test, num_features, **kwargs):
    score_func = f_classif if kwargs.get('task_type') == 'Classification' else f_regression
    skb = SelectKBest(score_func=score_func, k=min(num_features, X_train.shape[1])).fit(X_train, y_train)
    cols = X_train.columns[skb.get_support()]
    return pd.DataFrame(skb.transform(X_train), columns=cols, index=X_train.index), pd.DataFrame(skb.transform(X_test), columns=cols, index=X_test.index), cols.tolist()  # type: ignore

def chi_squared_feature_selection(X_train, y_train, X_test, y_test, num_features, **kwargs):
    if (X_train < 0).any().any(): raise ValueError("Chi2 requires non-negative count data.")
    skb = SelectKBest(score_func=chi2, k=min(num_features, X_train.shape[1])).fit(X_train, y_train)
    cols = X_train.columns[skb.get_support()]
    return pd.DataFrame(skb.transform(X_train), columns=cols, index=X_train.index), pd.DataFrame(skb.transform(X_test), columns=cols, index=X_test.index), cols.tolist()  # type: ignore

def lasso_feature_selection(X_train, y_train, X_test, y_test, num_features, random_state=42, **kwargs):
    l = LogisticRegression(penalty='l1', solver='liblinear', random_state=random_state).fit(X_train, y_train)
    m = SelectFromModel(l, max_features=num_features, prefit=True); cols = X_train.columns[m.get_support()]
    return pd.DataFrame(m.transform(X_train), columns=cols, index=X_train.index), pd.DataFrame(m.transform(X_test), columns=cols, index=X_test.index), cols.tolist()  # type: ignore

def shap_feature_selection(X_train, y_train, X_test, y_test, num_features, threads, compute_interactions=True, **kwargs):
    """
    SHAP-based feature selection with optional interaction value computation.
    
    Args:
        compute_interactions: If True, compute SHAP interaction values (slower but more informative)
                             Default is True for publication-quality analysis.
    
    Returns:
        Tuple of (X_train_selected, X_test_selected, selected_cols, shap_interaction_values)
    """
    _check_shap_installed()
    cls = CatBoostClassifier if kwargs.get('task_type') == 'Classification' else CatBoostRegressor
    m = cls(iterations=1000, thread_count=threads, random_state=42, verbose=False).fit(X_train, y_train)
    
    # Compute regular SHAP values
    explainer = shap.TreeExplainer(m)
    sample_X = X_train.sample(min(1000, len(X_train)))
    v = explainer.shap_values(sample_X)
    
    if isinstance(v, list): v = v[-1]
    elif v.ndim == 3: v = v[:, :, 1]
    imp = np.abs(v).mean(axis=0); idx = np.argsort(imp)[-num_features:]; cols = X_train.columns[idx].tolist()
    
    # Optionally compute interaction values
    interaction_values = None
    if compute_interactions:
        try:
            logger.info("Computing SHAP interaction values (this may take a while)...")
            # Use smaller sample for interactions (more expensive)
            interaction_sample = X_train.sample(min(500, len(X_train)))
            interaction_values = explainer.shap_interaction_values(interaction_sample)
            
            # Handle multiclass case
            if isinstance(interaction_values, list):
                interaction_values = interaction_values[-1]  # Use last class
            
            logger.info(f"SHAP interaction values computed: shape {interaction_values.shape}")
        except Exception as e:
            logger.warning(f"Failed to compute SHAP interactions: {e}")
            interaction_values = None
    
    return X_train[cols], X_test[cols], cols, interaction_values

def perform_feature_selection(X_train, y_train, X_test, y_test, feature_selection='rfe', **kwargs):
    """
    Router for all implemented selection methods.
    Returns:
        X_train_sel, X_test_sel, selected_cols, y_train_norm, y_test_norm
        (y_train_norm/y_test_norm are the possibly normalized binary labels for downstream use)
    """
    # --- Robust binary label normalization for classification tasks ---
    task_type = kwargs.get('task_type', 'Classification')
    y_train_norm, y_test_norm = y_train, y_test
    if task_type == 'Classification':
        # Only relabel if binary and not already {0,1} or {-1,1}
        unique_vals = np.unique(y_train)
        if len(unique_vals) == 2:
            if not (set(unique_vals) == {0, 1} or set(unique_vals) == {-1, 1}):
                logger.info(f"Normalizing binary labels from {set(unique_vals)} to {{0,1}} for sklearn compatibility.")
                min_val, max_val = unique_vals.min(), unique_vals.max()
                y_train_norm = (y_train == max_val).astype(int)
                y_test_norm = (y_test == max_val).astype(int)
    res = None
    if feature_selection == 'rfe':
        res = rfe_feature_selection(X_train, y_train_norm, X_test, y_test_norm, **kwargs)
    elif feature_selection == 'select_k_best':
        res = select_k_best_feature_selection(X_train, y_train_norm, X_test, y_test_norm, **kwargs)
    elif feature_selection == 'chi_squared':
        res = chi_squared_feature_selection(X_train, y_train_norm, X_test, y_test_norm, **kwargs)
    elif feature_selection == 'lasso':
        res = lasso_feature_selection(X_train, y_train_norm, X_test, y_test_norm, **kwargs)
    elif feature_selection == 'shap':
        # Extract and remove explicit args for shap_feature_selection
        threads = kwargs.pop('thread_count', kwargs.pop('threads', 4))
        num_features = kwargs.pop('num_features', 50)
        compute_interactions = kwargs.pop('compute_shap_interactions', False)
        return_interactions = kwargs.pop('return_interactions', False)
        res = shap_feature_selection(X_train, y_train_norm, X_test, y_test_norm, num_features, threads, compute_interactions=compute_interactions, **kwargs)
        # Always return only 3 values unless explicitly requested
        if len(res) == 4:
            if return_interactions:
                return (*res, y_train_norm, y_test_norm)  # (X_train_sel, X_test_sel, cols, interaction_values, y_train_norm, y_test_norm)
            else:
                return (*res[:3], y_train_norm, y_test_norm)  # (X_train_sel, X_test_sel, cols, y_train_norm, y_test_norm)
        else:
            return (*res, y_train_norm, y_test_norm)  # Already 3-tuple
    else:
        res = (X_train, X_test, X_train.columns.tolist())
        return (*res, y_train_norm, y_test_norm)
    
    # Prune features that fail permutation tests across batches
    # Only applies if not already returned above (for non-shap, non-early return)
    if feature_selection not in ['shap']:
        X_tr_s, X_te_s, sel = res
        if kwargs.get('use_permutation_importance', True) and len(sel) > 1:
            logger.info("Computing Permutation Importance across isolated batches...")
            m_cls = CatBoostClassifier if kwargs.get('task_type') == 'Classification' else CatBoostRegressor
            pm = m_cls(iterations=500, verbose=False).fit(X_tr_s, y_train_norm, eval_set=(X_te_s, y_test_norm))
            p_i = permutation_importance(pm, X_te_s.values, y_test_norm, n_repeats=10, random_state=42)
            f_sel = pd.Series(p_i['importances_mean'], index=sel)[lambda x: x > 0].index.tolist()
            if f_sel:
                return X_tr_s[f_sel], X_te_s[f_sel], f_sel, y_train_norm, y_test_norm
        return X_tr_s, X_te_s, sel, y_train_norm, y_test_norm