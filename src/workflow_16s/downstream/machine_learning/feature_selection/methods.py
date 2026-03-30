import pandas as pd
import numpy as np
import shap
import logging
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Literal, Any, Union
from sklearn.feature_selection import (
    RFE, SelectKBest, SelectFromModel, chi2, 
    f_classif, f_regression
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier, CatBoostRegressor

logger = logging.getLogger('workflow_16s')


# RECOVERY & PROXY MAPPING 

def filter_multicollinear_features(
    X: pd.DataFrame, 
    y: pd.Series, 
    threshold: float = 0.95, 
    task_type: str = 'Classification',
    recover_best: bool = True,
    out_dir: Optional[Path] = None
) -> List[str]:
    """
    Groups correlated features, picks the best univariate representative, 
    and saves a proxy mapping.
    """
    logger.info(f"Filtering multicollinearity (Threshold > {threshold})...")
    # Only correlate numeric columns (Ignore String/Batch IDs)
    X_numeric = X.select_dtypes(include=['number'])
    
    # Safety check: if no numeric data remains, return original features
    if X_numeric.empty:
        logger.warning(f"⚠️ No numeric features found. Returning all {len(X.columns)} features.")
        return X.columns.tolist()
        
    corr_matrix = X_numeric.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    clusters: Dict[str, List[str]] = {}
    to_drop = set()

    for col in upper.columns:
        correlated = upper.index[upper[col] > threshold].tolist()
        if correlated:
            clusters[col] = correlated
            for c in correlated: to_drop.add(c)

    final_keep = [c for c in X.columns if c not in to_drop]

    if recover_best and clusters:
        score_func = f_classif if task_type == 'Classification' else f_regression
        recovered_map = {}
        
        for rep, members in clusters.items():
            full_group = [rep] + members
            scores = score_func(X[full_group].fillna(0), y)[0]
            best_feature = full_group[np.argmax(scores)]
            
            recovered_map[best_feature] = [f for f in full_group if f != best_feature]
            if best_feature not in final_keep:
                final_keep.append(best_feature)

        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "cluster_mapping.json", "w") as f:
                json.dump(recovered_map, f, indent=4)
    
    # 🚨 SAFETY: Never return 0 features
    if not final_keep:
        logger.warning(f"⚠️ Multicollinearity filter would remove all features! Returning all features.")
        return X.columns.tolist()
    
    logger.debug(f"Multicollinearity filter: {len(X.columns)} → {len(final_keep)} features")
    return list(set(final_keep))


def annotate_proxies(top_features_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Reads cluster_mapping.json and adds a 'Proxy_For' column to results.
    """
    map_path = out_dir / "cluster_mapping.json"
    if not map_path.exists(): return top_features_df
    
    with open(map_path, "r") as f: proxy_map = json.load(f)
    
    top_features_df['proxy_for'] = top_features_df['feature'].apply(
        lambda x: ", ".join(proxy_map[x]) if x in proxy_map else ""
    )
    top_features_df.columns = top_features_df.columns.str.lower()
    return top_features_df


# --- SECTION 2: UNIVERSAL ROUTER ---

def perform_feature_selection(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    feature_selection: Literal['rfe', 'shap', 'lasso', 'chi_squared', 'select_k_best'] = 'rfe', 
    **kwargs: Any
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.Series, pd.Series]:
    """
    Router for all implemented selection methods.
    """
    task_type = kwargs.get('task_type', 'Classification')
    y_tr_n, y_te_n = y_train, y_test
    
    # 🚨 VIP PASS: Extract Categorical Columns upfront so we can protect them
    cat_features_names = []
    if hasattr(X_train, 'columns'):
        cat_features_names = X_train.select_dtypes(exclude=['number']).columns.tolist()
    
    # Label Normalization
    if task_type == 'Classification':
        u = np.unique(y_train)
        if len(u) == 2 and not (set(u) == {0, 1}):
            y_tr_n = (y_train == u.max()).astype(int)
            y_te_n = (y_test == u.max()).astype(int)

    # 1. Multicollinearity Filter (Pre-selection)
    out_dir = Path(kwargs.get('output_dir', '.'))
    if kwargs.get('filter_correlation', True):
        keep = filter_multicollinear_features(
            X_train, y_tr_n, threshold=kwargs.get('correlation_threshold', 0.95),
            task_type=task_type, recover_best=kwargs.get('recover_features', True), out_dir=out_dir
        )
        # Ensure categorical columns survive correlation filtering
        for c in cat_features_names:
            if c not in keep:
                keep.append(c)
        X_tr_p, X_te_p = X_train[keep], X_test[keep]
    else:
        X_tr_p, X_te_p = X_train, X_test

    # Calculate CatBoost Indices
    cat_features_indices = []
    if hasattr(X_tr_p, 'columns'):
        cat_features_indices = [X_tr_p.columns.get_loc(c) for c in cat_features_names]

    # 2. Algorithm Routing
    num_f = kwargs.get('num_features', 50)

    if feature_selection == 'rfe':
        if task_type == 'Classification':
            m = CatBoostClassifier(iterations=500, verbose=False, cat_features=cat_features_indices)
        else:
            m = CatBoostRegressor(iterations=500, verbose=False, cat_features=cat_features_indices)
        sel_proc = RFE(estimator=m, n_features_to_select=min(num_f, X_tr_p.shape[1]), step=5).fit(X_tr_p, y_tr_n) # type: ignore
        sel = X_tr_p.columns[sel_proc.support_].tolist()

    elif feature_selection == 'select_k_best':
        func = f_classif if task_type == 'Classification' else f_regression
        num_cols = [c for c in X_tr_p.columns if c not in cat_features_names]
        sel_proc = SelectKBest(score_func=func, k=min(num_f, len(num_cols))).fit(X_tr_p[num_cols], y_tr_n)
        sel = X_tr_p[num_cols].columns[sel_proc.get_support()].tolist()

    elif feature_selection == 'chi_squared':
        num_cols = [c for c in X_tr_p.columns if c not in cat_features_names]
        if (X_tr_p[num_cols] < 0).any().any(): raise ValueError("Chi2 requires non-negative count data.")
        sel_proc = SelectKBest(score_func=chi2, k=min(num_f, len(num_cols))).fit(X_tr_p[num_cols], y_tr_n)
        sel = X_tr_p[num_cols].columns[sel_proc.get_support()].tolist()

    elif feature_selection == 'lasso':
        num_cols = [c for c in X_tr_p.columns if c not in cat_features_names]
        l = LogisticRegression(penalty='l1', solver='liblinear', random_state=42).fit(X_tr_p[num_cols], y_tr_n)
        sel_proc = SelectFromModel(l, max_features=num_f, prefit=True)
        sel = X_tr_p[num_cols].columns[sel_proc.get_support()].tolist()

    elif feature_selection == 'shap':
        m = CatBoostClassifier(iterations=500, verbose=False, cat_features=cat_features_indices).fit(X_tr_p, y_tr_n)
        explainer = shap.TreeExplainer(m)
        v = explainer.shap_values(X_tr_p.sample(min(1000, len(X_tr_p))))
        if isinstance(v, list): v = v[-1]
        elif v.ndim == 3: v = v[:, :, 1]
        idx = np.argsort(np.abs(v).mean(axis=0))[-num_f:]
        sel = X_tr_p.columns[idx].tolist()
    else:
        sel = X_tr_p.columns.tolist()

    # 🚨 VIP PASS: Re-inject categorical targets if the selector deleted them
    for cat_col in cat_features_names:
        if cat_col not in sel:
            sel.append(cat_col)

    # 3. Final Permutation Pruning (Safety Valve)
    if kwargs.get('use_permutation_importance', True) and len(sel) > 1:
        X_tr_sel = X_tr_p[sel]
        X_te_sel = X_te_p[sel]
        
        final_cat_indices = [X_tr_sel.columns.get_loc(c) for c in cat_features_names if c in X_tr_sel.columns]

        pm = CatBoostClassifier(iterations=500, verbose=False, cat_features=final_cat_indices).fit(X_tr_sel, y_tr_n)
        p_i = permutation_importance(pm, X_te_sel, y_te_n, n_repeats=5, random_state=42)
        
        # Keep features with positive importance OR if they are our categorical batch ID
        sel = [f for f, imp in zip(sel, p_i['importances_mean']) if imp > 0 or f in cat_features_names]

    # Final ultimate safety check before returning
    for cat_col in cat_features_names:
        if cat_col not in sel:
            sel.append(cat_col)

    return X_train[sel], X_test[sel], sel, y_tr_n, y_te_n