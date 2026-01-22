# feature_selection/core.py

import numpy as np
import pandas as pd
import itertools
import shap
from pathlib import Path
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import matthews_corrcoef, r2_score, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix

from .validation import filter_data
from .methods import perform_feature_selection
from .reporting import (
    generate_shap_report,
    save_feature_importances,
    plot_roc_curve,
    plot_confusion_matrix
)
from workflow_16s.visualization._machine_learning import plot_shap
from workflow_16s.utils.logger import get_logger

logger = get_logger("workflow_16s")

def grid_search(X_train, y_train, X_test, y_test, groups_train=None, param_grid=None, output_dir=None, n_splits=5, refit='mcc', task_type='Classification', enable_overfitting_detector=True, **kwargs):
    """
    Grid search with GroupKFold to strictly isolate sampling batches and detect overfitting.
    
    Parameters
    ----------
    enable_overfitting_detector : bool
        Enable CatBoost's built-in overfitting detector (default: True)
        Monitors validation score and stops if no improvement for many iterations
    """
    out = Path(output_dir or "gs_out"); out.mkdir(parents=True, exist_ok=True)
    model_cls = CatBoostClassifier if task_type == 'Classification' else CatBoostRegressor
    
    # Structural Batch Fix: ensures batches never split across folds
    cv = GroupKFold(n_splits=n_splits) if groups_train is not None else KFold(n_splits=n_splits, shuffle=True)
    
    best_s = -np.inf if refit.upper() != 'RMSE' else np.inf; best_m, best_p = None, None
    param_grid = param_grid or {}
    combs = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())] if param_grid else [{}]
    
    # Track overfitting across folds
    fold_train_scores = []
    fold_val_scores = []
    
    for p in combs:
        scores = []
        train_scores = []
        
        for tr_i, val_i in cv.split(X_train, y_train, groups=groups_train):
            # Enhanced early stopping and overfitting detection
            m = model_cls(
                **p, 
                verbose=False,
                od_type='Iter' if enable_overfitting_detector else None,  # Overfitting detector
                od_wait=50 if enable_overfitting_detector else None  # Stop if no improvement for 50 iterations
            ).fit(
                X_train.iloc[tr_i], 
                y_train.iloc[tr_i], 
                eval_set=(X_train.iloc[val_i], y_train.iloc[val_i]), 
                early_stopping_rounds=50,
                use_best_model=True  # Use best iteration, not final
            )
            
            # Track both training and validation performance
            y_pred_train = m.predict(X_train.iloc[tr_i])
            y_pred_val = m.predict(X_train.iloc[val_i])
            
            train_sc = r2_score(y_train.iloc[tr_i], y_pred_train) if task_type == 'Regression' else matthews_corrcoef(y_train.iloc[tr_i], y_pred_train)
            val_sc = r2_score(y_train.iloc[val_i], y_pred_val) if task_type == 'Regression' else matthews_corrcoef(y_train.iloc[val_i], y_pred_val)
            
            scores.append(val_sc)
            train_scores.append(train_sc)
            
            # Check for overfitting in this fold
            fold_gap = train_sc - val_sc
            if fold_gap > 0.15:
                logger.warning(f"  Fold overfitting detected: Train={train_sc:.3f}, Val={val_sc:.3f}, Gap={fold_gap:.3f}")
        
        mean_val = np.mean(scores)
        mean_train = np.mean(train_scores)
        fold_train_scores.append(mean_train)
        fold_val_scores.append(mean_val)
        
        # Log cross-fold overfitting gap
        cv_gap = mean_train - mean_val
        logger.info(f"  Params {p}: Train={mean_train:.3f}, Val={mean_val:.3f}, Gap={cv_gap:.3f}")
        
        if cv_gap > 0.15:
            logger.warning(f"  ⚠️  HIGH OVERFITTING: Gap={cv_gap:.3f} - Consider increasing regularization")
        
        if np.mean(scores) > best_s: 
            best_s, best_p = np.mean(scores), p

    best_p = best_p or {}
    
    # Train final model with overfitting detection
    best_m = model_cls(
        **best_p, 
        verbose=False,
        od_type='Iter' if enable_overfitting_detector else None,
        od_wait=50 if enable_overfitting_detector else None
    ).fit(
        X_train, 
        y_train, 
        eval_set=(X_test, y_test),
        early_stopping_rounds=50,
        use_best_model=True
    )
    
    # Final overfitting check on test set
    y_pred_train_final = best_m.predict(X_train)
    y_pred_test_final = best_m.predict(X_test)
    
    final_train_score = r2_score(y_train, y_pred_train_final) if task_type == 'Regression' else matthews_corrcoef(y_train, y_pred_train_final)
    final_test_score = r2_score(y_test, y_pred_test_final) if task_type == 'Regression' else matthews_corrcoef(y_test, y_pred_test_final)
    final_gap = final_train_score - final_test_score
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Final Model Performance:")
    logger.info(f"  Training Score: {final_train_score:.3f}")
    logger.info(f"  Test Score: {final_test_score:.3f}")
    logger.info(f"  Overfitting Gap: {final_gap:.3f}")
    
    if final_gap > 0.15:
        logger.warning(f"  ⚠️  HIGH OVERFITTING DETECTED")
        logger.warning(f"  Model may not generalize to new data!")
    elif final_gap > 0.10:
        logger.warning(f"  ⚠️  MODERATE OVERFITTING")
    else:
        logger.info(f"  ✓ Good generalization (gap < 0.10)")
    
    if hasattr(best_m, 'get_best_iteration'):
        best_iter = best_m.get_best_iteration()
        total_iter = best_m.tree_count_
        logger.info(f"  Best Iteration: {best_iter} / {total_iter}")
        if best_iter < total_iter * 0.5:
            logger.info(f"  ✓ Early stopping worked well (stopped at {best_iter/total_iter:.1%})")
    logger.info(f"{'='*60}\n")
    
    # Custom eval visualization
    figs = []
    if task_type == 'Classification' and len(np.unique(y_test)) == 2 and hasattr(best_m, 'predict_proba'):
        y_prob = best_m.predict_proba(X_test)[:, 1]  # type: ignore
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        figs.append(plot_roc_curve(fpr, tpr, auc_score, str(out / "roc")))
        class_names = [str(c) for c in best_m.classes_] if hasattr(best_m, 'classes_') and best_m.classes_ is not None else ['0', '1']
        figs.append(plot_confusion_matrix(confusion_matrix(y_test, best_m.predict(X_test)), str(out / "cm"), class_names=class_names))
        
    return best_m, best_p, float(best_s), {'mcc': best_s}, figs

def catboost_feature_selection(metadata, features, output_dir, group_col, cv_groups=None, method='rfe', **kwargs):
    """Package Entry Point: Orchestrates robust biomarker identification."""
    out = Path(output_dir) / method; out.mkdir(parents=True, exist_ok=True)
    common = features.index.intersection(metadata.index); X, y, meta = features.loc[common], metadata.loc[common, group_col], metadata.loc[common]
    
    # Pre-map batches
    g_arr = meta[cv_groups].values if cv_groups and cv_groups in meta.columns else None
    # Extract only the parameters that filter_data accepts
    filter_kwargs = {k: v for k, v in kwargs.items() if k in ['test_size', 'random_state', 'min_samples_per_class', 'task_type']}
    X_tr, X_te, y_tr, y_te, g_tr, g_te = filter_data(X, y, meta, group_col, cv_groups=g_arr, **filter_kwargs)

    # Unpack normalized labels from perform_feature_selection
    X_tr_s, X_te_s, sel, y_tr_norm, y_te_norm = perform_feature_selection(X_tr, y_tr, X_te, y_te, feature_selection=method, **kwargs)

    # Identify categorical features for batch correction, excluding unique identifiers
    cat_f = []
    if 'batch_original' in meta.columns:
        batch_col = meta.loc[X_tr_s.index, 'batch_original']
        n_unique = batch_col.nunique()
        n_samples = len(batch_col)
        uniqueness_ratio = n_unique / n_samples

        # Only use as categorical if groups are large enough (not a unique ID)
        # Keep if <70% unique (e.g., biosample accessions with multiple samples OK)
        # Exclude if >70% unique (e.g., run_accession where each is unique)
        if uniqueness_ratio < 0.7:
            cat_f.append('batch_original')
            logger.info(f"Using 'batch_original' as categorical: {n_unique} groups for {n_samples} samples ({uniqueness_ratio:.1%} unique)")
        else:
            logger.warning(f"Excluding 'batch_original' (too unique): {n_unique} groups for {n_samples} samples ({uniqueness_ratio:.1%} unique)")

    if cat_f:
        X_tr_s['batch_original'] = meta.loc[X_tr_s.index, 'batch_original'].astype(str)
        X_te_s['batch_original'] = meta.loc[X_te_s.index, 'batch_original'].astype(str)

    # Use normalized labels for all downstream model training and evaluation
    m, p, s, t_scores, figs = grid_search(X_tr_s, y_tr_norm, X_te_s, y_te_norm, groups_train=g_tr, fixed_params={'cat_features': cat_f}, **kwargs)
    save_feature_importances(m, X_tr_s, out)

    top = pd.Series(m.feature_importances_, index=X_tr_s.columns).drop('batch_original', errors='ignore').head(kwargs.get('n_top_features', 20)).index.tolist()

    # Final Visual Explanations
    if y_tr_norm.nunique() <= 2:
        rep, shap_df = generate_shap_report(m, X_tr_s.sample(min(1000, len(X_tr_s))))
        expl = shap.TreeExplainer(m, feature_perturbation="tree_path_dependent")
        sample_X = X_tr_s.sample(min(1000, len(X_tr_s)))
        shap_vals = expl.shap_values(sample_X)

        # Compute SHAP interaction values if enabled (default: True for publication quality)
        shap_interaction_vals = None
        if kwargs.get('compute_shap_interactions', True):
            try:
                logger.info("Computing SHAP interaction values for visualization...")
                interaction_sample = X_tr_s.sample(min(500, len(X_tr_s)))
                shap_interaction_vals = expl.shap_interaction_values(interaction_sample)
                if isinstance(shap_interaction_vals, list):
                    shap_interaction_vals = shap_interaction_vals[1]  # Binary classification
                logger.info(f"SHAP interactions computed: {shap_interaction_vals.shape}")
            except Exception as e:
                logger.warning(f"Failed to compute SHAP interactions: {e}")

        plot_shap(
            expl.expected_value[1] if isinstance(expl.expected_value, list) else expl.expected_value,  # type: ignore
            shap_vals, 
            sample_X.values, 
            X_tr_s.columns.tolist(), 
            output_dir=out,
            shap_interaction_values=shap_interaction_vals
        )

    return {'method': method, 'top_features': top, 'best_cv_score': s, 'test_scores': t_scores}