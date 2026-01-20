# ==================================================================================== #
# ===================================== IMPORTS ====================================== #
# ==================================================================================== #

# Standard Library Imports
import itertools
import gc 
import logging
import os
import time
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

# Third‑Party Imports
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, CatBoostRegressor, cv, Pool
from scipy.stats import spearmanr
from sklearn.feature_selection import (
    RFE, SelectFromModel, SelectKBest, VarianceThreshold, chi2, f_classif, f_regression
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, confusion_matrix, f1_score,
    get_scorer, matthews_corrcoef, make_scorer, precision_recall_curve,
    roc_auc_score, roc_curve, r2_score, mean_squared_error
)
# GroupKFold is the structural fix for 300+ batches
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold, GroupKFold

# Local Workflow Integrations
from workflow_16s.visualization._machine_learning import (
    plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve, plot_shap
)
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ==================================================================================== #
# ========================== INITIALISATION & CONFIGURATION ========================== #
# ==================================================================================== #

logger = logging.getLogger('workflow_16s')

# Configuration Defaults
DEFAULT_TEST_SIZE = 0.3
DEFAULT_RANDOM_STATE = 42
DEFAULT_METHOD = 'rfe'
DEFAULT_USE_PERMUTATION_IMPORTANCE = True
DEFAULT_THREAD_COUNT = 4
DEFAULT_STEP_SIZE = 1000
DEFAULT_NUM_FEATURES = 500
DEFAULT_ITERATIONS_RFE = 500
DEFAULT_LEARNING_RATE_RFE = 0.1
DEFAULT_DEPTH_RFE = 4
DEFAULT_ITERATIONS_SHAP = 1000
DEFAULT_LEARNING_RATE_SHAP = 0.1
DEFAULT_DEPTH_SHAP = 4
DEFAULT_LOSS_FUNCTION = 'Logloss'
DEFAULT_EVAL_METRIC = 'MCC'

# Comprehensive Grid Search Space
DEFAULT_PARAM_GRID = {
    'iterations': [500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5],
    'border_count': [32, 64],
    'random_strength': [1, 2]
}

# ==================================================================================== #
# ================================= UTILITY FUNCTIONS ================================ #
# ==================================================================================== #

def _check_shap_installed():
    """Verify SHAP library is available for interpretation logic."""
    if shap is None: 
        raise ImportError("SHAP not installed. Required for biomarker interpretation.")

def _validate_inputs(X_train, y_train, X_test, y_test):
    """Enforce data alignment and integrity checks across batches."""
    if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]: 
        raise ValueError("Training/Testing set label alignment error.")
    if X_train.empty or X_test.empty: 
        raise ValueError("Input dataset is empty.")

def _save_dataframe(df: pd.DataFrame, output_path: Union[str, Path]):
    """Standardized result persistence."""
    p = Path(output_path); p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)

def check_for_data_leakage(X_train, X_test, threshold=0.99):
    """Detects features that are too similar between train/test, signaling batch contamination."""
    for col in X_train.columns:
        if X_train[col].nunique() < 2: continue
        corr, _ = spearmanr(X_train[col].sample(min(100, len(X_train))), 
                            X_test[col].sample(min(100, len(X_test))), nan_policy='omit')
        if abs(corr) > threshold: 
            logger.warning(f"BATCH LEAKAGE ALERT: Feature '{col}' has {corr:.2f} similarity.")

# ==================================================================================== #
# ================================= SHAP REPORTING =================================== #
# ==================================================================================== #

def generate_shap_report(model, X: pd.DataFrame, K: int = 10) -> Tuple[str, pd.DataFrame]:
    """Generates an exhaustive summary of microbial feature impact on the model."""
    _check_shap_installed()
    try: 
        expl = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        sv_exp = expl(X); sv = sv_exp.values 
        
        # Consolidation for multiclass or regression tasks
        if isinstance(sv, list): sv = sv[1] if len(sv) == 2 else np.mean([np.abs(s) for s in sv], axis=0)
        if sv.ndim == 3: sv = sv.mean(axis=2) 
        
        mean_abs = np.abs(sv).mean(axis=0)
        feat_idx = np.argsort(mean_abs)[::-1][:K]
        top_feats = list(X.columns[feat_idx]); top_means = mean_abs[feat_idx]

        features, mean_abs_shap, beeswarm_corr, beeswarm_direction = [], [], [], []
        interaction_partner, interaction_strength, relationship_type = [], [], []

        lines = [f"Summary of Top {K} Microbial Impactors:"]
        for feat, m in zip(top_feats, top_means):
            lines.append(f" • {feat} (mean impact = {m:.3f})")
            features.append(feat); mean_abs_shap.append(m)
        lines.append("")

        lines.append("Trend and Directionality:")
        for feat in top_feats:
            vals = X[feat].values; shap_vals = sv[:, X.columns.get_loc(feat)]
            rho, _ = spearmanr(vals, shap_vals, nan_policy='omit')
            dir_txt = "increasing value → higher prediction" if rho > 0 else "increasing value → lower prediction"
            lines.append(f" • {feat}: {dir_txt} (ρ = {rho:.2f})")
            beeswarm_corr.append(rho); beeswarm_direction.append("pos" if rho > 0 else "neg") 

        # Interaction Analysis: Identify co-dependent microbial signatures
        mean_abs_int = None
        try:
            int_expl = shap.TreeExplainer(model)
            int_v = int_expl.shap_interaction_values(X) 
            if isinstance(int_v, list): int_v = int_v[1] if len(int_v) == 2 else np.mean([np.abs(iv) for iv in int_v], axis=0)
            if int_v.ndim == 4: int_v = int_v.mean(axis=3)
            mean_abs_int = np.abs(int_v).mean(axis=0)
        except: pass

        for i, feat in enumerate(top_feats):
            i_idx = X.columns.get_loc(feat)
            if mean_abs_int is not None:
                i_str = mean_abs_int[i_idx].copy(); i_str[i_idx] = -np.inf 
                j_idx = i_str.argmax(); partner = X.columns[j_idx]; score = i_str[j_idx]
                if i_idx != j_idx:
                    r_i, _ = spearmanr(X[partner].values, sv[:, i_idx], nan_policy='omit')
                    r_j, _ = spearmanr(X[feat].values, sv[:, j_idx], nan_policy='omit')
                    rel = "reinforcing" if r_i > 0 and r_j > 0 else "diminishing" if r_i < 0 and r_j < 0 else "opposing"
                    lines.append(f" • Synergy: {feat} & {partner} (score={score:.3f}): {rel}")
                    interaction_partner.append(partner); interaction_strength.append(score); relationship_type.append(rel)
            else:
                interaction_partner.append(None); interaction_strength.append(None); relationship_type.append(None)

        return "\n".join(lines), pd.DataFrame({'feature': features, 'shap_impact': mean_abs_shap, 'rho': beeswarm_corr, 'interaction': interaction_partner})
    except Exception as e:
        logger.error(f"SHAP Failure: {e}"); return "Error", pd.DataFrame()

# ==================================================================================== #
# ============================== GROUP-AWARE PIPELINE ================================ #
# ==================================================================================== #

def filter_data(X, y, metadata, group_col, test_size=0.3, random_state=42, min_samples=2, task_type='Classification', cv_groups=None):
    """Filters and splits while strictly isolating 300+ batches."""
    if not X.index.isin(metadata.index).all(): metadata = metadata.loc[X.index]
    y_labs = metadata.loc[X.index, group_col]

    if task_type == 'Classification':
        c = y_labs.value_counts(); k = c[c >= min_samples].index
        if len(k) < len(c):
            m = y_labs.isin(k); X, y, y_labs = X.loc[m], y.loc[m], y_labs.loc[m]
            if cv_groups is not None: cv_groups = cv_groups[m.values]
        if y_labs.nunique() < 2: return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series(), None, None
    else: 
        m = y_labs.notna()
        if m.sum() < len(y_labs): 
            X, y = X.loc[m], y.loc[m]
            if cv_groups is not None: cv_groups = cv_groups[m.values]
    
    try:
        # Crucial for 300 batches: We split on unique Batch IDs to avoid leakage
        if cv_groups is not None:
            u_b = np.unique(cv_groups)
            tr_b, te_b = train_test_split(u_b, test_size=test_size, random_state=random_state)
            tr_m, te_m = np.isin(cv_groups, tr_b), np.isin(cv_groups, te_b)
            return X[tr_m], X[te_m], y[tr_m], y[te_m], cv_groups[tr_m], cv_groups[te_m]
        return (*train_test_split(X, y, test_size=test_size, random_state=random_state), None, None)
    except: return (*train_test_split(X, y, test_size=test_size, random_state=random_state), None, None)

# ==================================================================================== #
# ========================= FULL FEATURE SELECTION ROUTERS =========================== #
# ==================================================================================== #

def rfe_feature_selection(X_train, y_train, X_test, y_test, num_features, step_size, threads, random_state, **kwargs):
    """CatBoost-based Recursive Feature Elimination."""
    cls = CatBoostClassifier if kwargs.get('task_type') == 'Classification' else CatBoostRegressor
    m = cls(iterations=DEFAULT_ITERATIONS_RFE, thread_count=threads, random_state=random_state, verbose=False)
    rfe = RFE(estimator=m, n_features_to_select=min(num_features, X_train.shape[1]), step=step_size, verbose=1)
    rfe.fit(X_train, y_train); c = X_train.columns[rfe.support_]
    return pd.DataFrame(rfe.transform(X_train), columns=c, index=X_train.index), pd.DataFrame(rfe.transform(X_test), columns=c, index=X_test.index), c.tolist()

def select_k_best_feature_selection(X_train, y_train, X_test, y_test, num_features, **kwargs):
    """F-score univariate selection."""
    f = f_classif if len(np.unique(y_train)) < 10 else f_regression
    s = SelectKBest(score_func=f, k=min(num_features, X_train.shape[1])).fit(X_train, y_train)
    c = X_train.columns[s.get_support()]
    return pd.DataFrame(s.transform(X_train), columns=c, index=X_train.index), pd.DataFrame(s.transform(X_test), columns=c, index=X_test.index), c.tolist()

def chi_squared_feature_selection(X_train, y_train, X_test, y_test, num_features, **kwargs):
    """Restored: Chi-Squared selection pathway."""
    if (X_train < 0).any().any(): raise ValueError("Chi2 requires absolute count data.")
    s = SelectKBest(score_func=chi2, k=min(num_features, X_train.shape[1])).fit(X_train, y_train)
    c = X_train.columns[s.get_support()]
    return pd.DataFrame(s.transform(X_train), columns=c, index=X_train.index), pd.DataFrame(s.transform(X_test), columns=c, index=X_test.index), c.tolist()

def lasso_feature_selection(X_train, y_train, X_test, y_test, num_features, random_state=42, **kwargs):
    """Restored: L1-Regularized Lasso Selection pathway."""
    l = LogisticRegression(penalty='l1', solver='liblinear', random_state=random_state).fit(X_train, y_train)
    m = SelectFromModel(l, max_features=num_features, prefit=True); c = X_train.columns[m.get_support()]
    return pd.DataFrame(m.transform(X_train), columns=c, index=X_train.index), pd.DataFrame(m.transform(X_test), columns=c, index=X_test.index), c.tolist()

def shap_feature_selection(X_train, y_train, X_test, y_test, num_features, threads, task_type='Classification', **kwargs):
    """SHAP Attribution-based selection."""
    _check_shap_installed()
    cls = CatBoostClassifier if task_type == 'Classification' else CatBoostRegressor
    m = cls(iterations=DEFAULT_ITERATIONS_SHAP, thread_count=threads, random_state=42, verbose=False).fit(X_train, y_train)
    v = shap.TreeExplainer(m).shap_values(X_train.sample(min(1000, len(X_train))))
    if isinstance(v, list): v = v[-1]
    elif v.ndim == 3: v = v[:, :, 1]
    i = np.abs(v).mean(axis=0); idx = np.argsort(i)[-num_features:]; c = X_train.columns[idx].tolist()
    return X_train[c], X_test[c], c

def perform_feature_selection(X_train, y_train, X_test, y_test, feature_selection=DEFAULT_METHOD, **kwargs):
    """Master Router: Supports RFE, Stats, Lasso, and SHAP logic paths."""
    if feature_selection == 'rfe': res = rfe_feature_selection(X_train, y_train, X_test, y_test, **kwargs)
    elif feature_selection == 'select_k_best': res = select_k_best_feature_selection(X_train, y_train, X_test, y_test, **kwargs)
    elif feature_selection == 'chi_squared': res = chi_squared_feature_selection(X_train, y_train, X_test, y_test, **kwargs)
    elif feature_selection == 'lasso': res = lasso_feature_selection(X_train, y_train, X_test, y_test, **kwargs)
    elif feature_selection == 'shap': res = shap_feature_selection(X_train, y_train, X_test, y_test, **kwargs)
    
    # Batch Stability check via Permutation Importance
    X_tr_s, X_te_s, sel = res
    if kwargs.get('use_permutation_importance', True) and len(sel) > 1:
        logger.info("Verifying features via Permutation across all batches...")
        m_cls = CatBoostClassifier if kwargs.get('task_type') == 'Classification' else CatBoostRegressor
        pm = m_cls(iterations=500, verbose=False).fit(X_tr_s, y_train, eval_set=(X_te_s, y_test))
        p_i = permutation_importance(pm, X_te_s.values, y_test, n_repeats=10, random_state=42)
        f_sel = pd.Series(p_i['importances_mean'], index=sel)[lambda x: x > 0].index.tolist()
        if f_sel: return X_tr_s[f_sel], X_te_s[f_sel], f_sel
    return X_tr_s, X_te_s, sel

# ==================================================================================== #
# ============================== GRID SEARCH ENGINE ================================== #
# ==================================================================================== #

def grid_search(X_train, y_train, X_test, y_test, groups_train=None, param_grid=DEFAULT_PARAM_GRID, output_dir=None, n_splits=5, refit='mcc', verbose=1, fixed_params=None, task_type='Classification', progress=None, task_id=None):
    """Comprehensive Optimization that strictly isolates 300+ batches."""
    out = Path(output_dir or "gs_out"); out.mkdir(parents=True, exist_ok=True); best_s = -np.inf if refit.upper() != 'RMSE' else np.inf
    model_cls = CatBoostClassifier if task_type == 'Classification' else CatBoostRegressor
    
    # Select CV Strategy: GroupKFold ensures model is tested on unseen sequencing batches
    cv_strat = GroupKFold(n_splits=n_splits) if groups_train is not None else (StratifiedKFold(n_splits=n_splits, shuffle=True) if task_type == 'Classification' else KFold(n_splits=n_splits, shuffle=True))
    
    param_list = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    best_m, best_p = None, None
    for i, p in enumerate(param_list, 1):
        if fixed_params: p.update(fixed_params)
        fold_scores = []
        for tr_idx, val_idx in list(cv_strat.split(X_train, y_train, groups=groups_train)):
            m = model_cls(**p, verbose=False).fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx], eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]), early_stopping_rounds=50)
            y_p = m.predict(X_train.iloc[val_idx])
            s_val = r2_score(y_train.iloc[val_idx], y_p) if task_type == 'Regression' else matthews_corrcoef(y_train.iloc[val_idx], y_p)
            fold_scores.append(s_val)
        
        avg_score = np.mean(fold_scores)
        if (refit.upper() == 'RMSE' and avg_score < best_score) or (refit.upper() != 'RMSE' and avg_score > best_score): 
            best_score, best_params = avg_score, p

    # Training final robust model
    best_model = model_cls(**best_params, verbose=False).fit(X_train, y_train, eval_set=(X_test, y_test))
    t_scores = {'r2' if task_type == 'Regression' else 'mcc': best_model.score(X_test, y_test)}
    
    # Integration with local visualization scripts
    figs = []
    if task_type == 'Classification' and len(np.unique(y_test)) == 2:
        y_prob = best_model.predict_proba(X_test)[:, 1]
        figs.append(plot_roc_curve(*roc_curve(y_test, y_prob)[:2], roc_auc_score(y_test, y_prob), str(out / "roc")))
        figs.append(plot_precision_recall_curve(*precision_recall_curve(y_test, y_prob)[:2], average_precision_score(y_test, y_prob), str(out / "prc")))
        figs.append(plot_confusion_matrix(confusion_matrix(y_test, best_model.predict(X_test)), str(out / "cm"), class_names=[str(c) for c in best_model.classes_]))
        
    return best_model, best_params, float(best_score), test_scores, figs

# ==================================================================================== #
# ============================== CORE ORCHESTRATOR =================================== #
# ==================================================================================== #

def catboost_feature_selection(metadata, features, output_dir, group_col, cv_groups=None, method='rfe', n_top_features=100, task_type='Classification', loss_function=DEFAULT_LOSS_FUNCTION, eval_metric=DEFAULT_EVAL_METRIC, test_size=DEFAULT_TEST_SIZE, progress=None, task_id=None, verbose=True, **kwargs):
    """The master orchestrator for identifying robust indicators across 300+ batches."""
    final_out = Path(output_dir) / method; final_out.mkdir(parents=True, exist_ok=True)
    common_idx = features.index.intersection(metadata.index); X, y, meta = features.loc[common_idx], metadata.loc[common_idx, group_col], metadata.loc[common_idx]
    
    # Map cv_groups for consistent splitting
    g_arr = meta[cv_groups].values if cv_groups and cv_groups in meta.columns else None
    X_tr, X_te, y_tr, y_te, g_tr, g_te = filter_data(X, y, meta, group_col, test_size=test_size, task_type=task_type, cv_groups=g_arr)
    
    # execute selection router
    X_tr_s, X_te_s, sel_feat = perform_feature_selection(X_tr, y_tr, X_te, y_te, feature_selection=method, task_type=task_type, **kwargs)
    
    # Inject Batch Covariate as Categorical for the final model run
    cat_f = []
    if 'batch_original' in meta.columns:
        X_tr_s['batch_original'] = meta.loc[X_tr_s.index, 'batch_original'].astype(str)
        X_te_s['batch_original'] = meta.loc[X_te_s.index, 'batch_original'].astype(str); cat_f = ['batch_original']

    # Optimization using out-of-batch validation
    m, p, s, t_scores, figs = grid_search(X_tr_s, y_tr, X_te_s, y_te, groups_train=g_tr, task_type=task_type, refit=eval_metric, fixed_params={'cat_features': cat_f}, progress=progress, task_id=task_id, verbose=verbose)
    
    # Ranking Taxa while ignoring technical covariate (batch)
    pd.Series(m.feature_importances_, index=X_tr_s.columns).sort_values(ascending=False).to_csv(final_out / "feat_importances.csv")
    top_taxa = pd.Series(m.feature_importances_, index=X_tr_s.columns).drop('batch_original', errors='ignore').head(n_top_features).index.tolist()
    
    # Full Integrated SHAP Synthesis
    shap_df, shap_figs = pd.DataFrame(), {}
    if y_tr.nunique() <= 2 or task_type == 'Regression':
        rep, shap_df = generate_shap_report(m, X_tr_s.sample(n=min(1000, len(X_tr_s))), K=20)
        expl = shap.TreeExplainer(m, feature_perturbation="tree_path_dependent")
        sv = expl.shap_values(X_tr_s.sample(n=min(1000, len(X_tr_s))))
        if isinstance(sv, list): sv = sv[1]
        shap_figs = plot_shap(expl.expected_value[1] if isinstance(expl.expected_value, list) else expl.expected_value, sv, X_tr_s.sample(n=min(1000, len(X_tr_s))).values, X_tr_s.columns.tolist(), output_dir=final_out)

    return {'method': method, 'top_features': top_taxa, 'best_cv_score': s, 'test_scores': t_scores, 'shap_report_df': shap_df, 'figures': {**{f'fig_{i}': f for i, f in enumerate(figs)}, 'shap': shap_figs}}

# ==================================================================================== #
# MEMORY AND LOGISTICS HELPERS
# ==================================================================================== #

def optimize_feature_memory(df):
    """Downcasts floats to preserve RAM during 300+ batch merge loops."""
    floats = df.select_dtypes(include=['float64']).columns
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df

def cleanup_stale_cache():
    """Manual garbage collection for high-cardinality loops."""
    gc.collect()