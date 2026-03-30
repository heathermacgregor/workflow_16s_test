# src/workflow_16s/downstream/machine_learning/feature_selection/core.py

import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import traceback
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

# Sklearn & CatBoost
from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit, train_test_split, StratifiedKFold
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    matthews_corrcoef, r2_score, roc_curve, roc_auc_score, 
    precision_recall_curve, average_precision_score, confusion_matrix,
    mean_squared_error
)

# Optuna
import optuna
from optuna.samplers import TPESampler
from optuna.exceptions import TrialPruned

# Local Workflow Imports
from .validation import filter_data
from .methods import perform_feature_selection, annotate_proxies
from .reporting import generate_shap_report, save_feature_importances

# Visualization Helpers
from workflow_16s.downstream.machine_learning.visualization import (
    plot_shap, 
    plot_roc_curve, 
    plot_precision_recall_curve, 
    plot_confusion_matrix, 
    plot_predicted_vs_actual,
    plot_residuals
)
from workflow_16s.utils.logger import get_logger
from workflow_16s.downstream.machine_learning.optuna.realtime_monitor import OptunaRealtimeCallback

# Type Aliases
PathLike = Union[str, Path]

# HELPER FUNCTION FOR FALLBACK TRAINING

def _train_fallback_model(
    X_train, y_train, X_test, y_test,
    groups_train, cv, task_type, enable_overfitting_detector,
    fixed_params, logger, n_splits
):
    """Train a single model with fixed parameters when Optuna has no successful trials."""
    model_cls = CatBoostClassifier if task_type == 'Classification' else CatBoostRegressor
    
    X_train_df = pd.DataFrame(X_train) if isinstance(X_train, np.ndarray) else X_train
    y_train_ser = pd.Series(y_train) if isinstance(y_train, np.ndarray) else y_train
    
    if groups_train is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        idx_train, idx_val = next(gss.split(X_train_df, y_train_ser, groups=groups_train))
    else:
        if task_type == 'Classification':
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
            idx_train, idx_val = next(skf.split(X_train_df, y_train_ser))
        else:
            # Regression: use standard KFold
            kf = KFold(n_splits=2, shuffle=True, random_state=42)
            idx_train, idx_val = next(kf.split(X_train_df, y_train_ser))
    
    X_fin_tr, X_fin_val = X_train_df.iloc[idx_train], X_train_df.iloc[idx_val]
    y_fin_tr, y_fin_val = y_train_ser.iloc[idx_train], y_train_ser.iloc[idx_val]
    
    best_m = model_cls(**fixed_params, verbose=False).fit(
        X_fin_tr, y_fin_tr,
        eval_set=(X_fin_val, y_fin_val),
        early_stopping_rounds=50,
        use_best_model=True
    )
    
    return None, None, 0.0, best_m

# HYPERPARAMETER OPTIMIZATION WITH OPTUNA

def grid_search(
    X_train: pd.DataFrame, 
    y_train: Union[pd.Series, np.ndarray], 
    X_test: pd.DataFrame, 
    y_test: Union[pd.Series, np.ndarray], 
    groups_train: Optional[Union[pd.Series, np.ndarray]] = None, 
    param_grid: Optional[Dict[str, List[Any]]] = None, 
    output_dir: Optional[PathLike] = None, 
    n_splits: int = 5, 
    task_type: Literal['Classification', 'Regression'] = 'Classification', 
    enable_overfitting_detector: bool = True, 
    fixed_params: Optional[Dict[str, Any]] = None, 
    n_trials: int = 20,
    telemetry: Optional[Any] = None,
    **kwargs: Any
) -> Tuple[Any, Dict[str, Any], float, Dict[str, float], List[Any]]:
    """
    Performs Bayesian hyperparameter optimization using Optuna with real-time progress tracking.
    
    Args:
        X_train, y_train, X_test, y_test: Training/test data
        groups_train: Group labels for group-aware CV (e.g., study/batch)
        param_grid: Parameter ranges (optuna will sample from these)
        output_dir: Output directory for results
        n_splits: CV folds
        task_type: 'Classification' or 'Regression'
        enable_overfitting_detector: Use CatBoost overfitting detection
        fixed_params: Fixed parameters (not tuned)
        n_trials: Number of Optuna trials (default 20)
        telemetry: Optional telemetry collector for dashboard
        
    Returns:
        (best_model, best_params, best_score, test_scores, figures)
    """
    logger = get_logger("workflow_16s")
    out = Path(output_dir or "gs_out")
    out.mkdir(parents=True, exist_ok=True)
    
    model_cls = CatBoostClassifier if task_type == 'Classification' else CatBoostRegressor
    fixed_params = fixed_params or {}
    
    # 1. CROSS-VALIDATION STRATEGY
    cv = GroupKFold(n_splits=n_splits) if groups_train is not None else KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 2. PREPARE PARAMETER GRID FOR OPTUNA SAMPLING
    # Convert param_grid lists to ranges for Optuna
    if param_grid is None:
        param_grid = {
            'depth': [4, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'l2_leaf_reg': [3, 7]
        }
    
    # Check if we have enough samples for group-aware CV
    enable_group_cv = groups_train is not None and len(np.unique(groups_train)) > 1
    
    if enable_group_cv and n_splits > len(np.unique(groups_train)) // 2:
        # Not enough groups for this many splits, reduce splits
        n_splits = max(2, len(np.unique(groups_train)) // 2)
        logger.info(f"⚠️ Adjusted n_splits to {n_splits} based on number of groups")
    
    # Use stratified K-fold if we don't have groups (safer for imbalanced classes)
    if enable_group_cv:
        cv = GroupKFold(n_splits=n_splits)
    else:
        if task_type == 'Classification':
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            # Regression: use standard KFold (StratifiedKFold doesn't work with continuous targets)
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Define parameter sampling ranges for Optuna
    param_ranges = {}
    for param, values in param_grid.items():
        if isinstance(values, list):
            if all(isinstance(v, (int, np.integer)) for v in values):
                # Integer parameter
                param_ranges[param] = ('int', min(values), max(values))
            elif all(isinstance(v, (float, np.floating)) for v in values):
                # Float parameter
                param_ranges[param] = ('float', min(values), max(values))
            else:
                # Categorical parameter
                param_ranges[param] = ('categorical', values)
    
    logger.info(f"🔍 Optuna Bayesian Optimization starting...")
    logger.info(f"   Parameters: {list(param_ranges.keys())}")
    logger.info(f"   Trials: {n_trials}")
    logger.info(f"   CV Strategy: {'GroupKFold' if groups_train is not None else 'KFold'}")
    
    # 3. DEFINE OBJECTIVE FUNCTION FOR OPTUNA
    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        
        # Sample hyperparameters
        params = {}
        for param, (ptype, *prange) in param_ranges.items():
            if ptype == 'int':
                params[param] = trial.suggest_int(param, prange[0], prange[1])
            elif ptype == 'float':
                params[param] = trial.suggest_float(param, prange[0], prange[1])
            elif ptype == 'categorical':
                params[param] = trial.suggest_categorical(param, prange[0])
        
        # Merge with fixed parameters
        current_params = {**params, **fixed_params}
        
        # Perform cross-validation
        scores, train_scores = [], []
        
        # Convert to pandas if needed for .iloc
        X_train_df = pd.DataFrame(X_train) if isinstance(X_train, np.ndarray) else X_train
        y_train_ser = pd.Series(y_train) if isinstance(y_train, np.ndarray) else y_train
        
        try:
            for fold_idx, (tr_i, val_i) in enumerate(cv.split(X_train_df, y_train_ser, groups=groups_train)):
                try:
                    # Check for class imbalance in this fold (prevent "unseen class" errors)
                    y_tr_fold = y_train_ser.iloc[tr_i]
                    y_val_fold = y_train_ser.iloc[val_i]
                    
                    # Filter validation set to only include classes seen in training
                    if task_type == 'Classification':
                        train_classes = set(y_tr_fold.unique())
                        val_classes = set(y_val_fold.unique())
                        unseen_classes = val_classes - train_classes
                        
                        if unseen_classes:
                            # Mask for samples with seen classes only
                            valid_mask = ~y_val_fold.isin(unseen_classes)
                            if valid_mask.sum() == 0:
                                # All validation samples have unseen classes, skip this fold
                                logger.debug(f"      Fold {fold_idx}: Skipping (all validation samples have unseen classes)")
                                continue
                            
                            # Filter to only valid samples using numpy boolean indexing
                            tr_i_eff = tr_i
                            # Convert mask to numpy array if it's a series
                            if isinstance(valid_mask, pd.Series):
                                valid_mask_np = valid_mask.values
                            else:
                                valid_mask_np = valid_mask
                            val_i_eff = val_i[valid_mask_np]
                        else:
                            tr_i_eff = tr_i
                            val_i_eff = val_i
                    else:
                        tr_i_eff = tr_i
                        val_i_eff = val_i
                    
                    m = model_cls(
                        **current_params, 
                        verbose=False,
                        od_type='Iter' if enable_overfitting_detector else None,
                        od_wait=50 if enable_overfitting_detector else None,
                        allow_writing_files=False
                    ).fit(
                        X_train_df.iloc[tr_i_eff], y_train_ser.iloc[tr_i_eff], 
                        eval_set=(X_train_df.iloc[val_i_eff], y_train_ser.iloc[val_i_eff]), 
                        early_stopping_rounds=50,
                        use_best_model=True
                    )
                    
                    y_pred_val = m.predict(X_train_df.iloc[val_i_eff])
                    y_pred_tr = m.predict(X_train_df.iloc[tr_i_eff])
                    
                    if task_type == 'Regression':
                        val_sc = r2_score(y_train_ser.iloc[val_i_eff], y_pred_val)
                        tr_sc = r2_score(y_train_ser.iloc[tr_i_eff], y_pred_tr)
                    else:
                        val_sc = matthews_corrcoef(y_train_ser.iloc[val_i_eff], y_pred_val)
                        tr_sc = matthews_corrcoef(y_train_ser.iloc[tr_i_eff], y_pred_tr)
                    
                    scores.append(val_sc)
                    train_scores.append(tr_sc)
                
                except Exception as fold_error:
                    # Skip problematic fold but continue with others
                    logger.debug(f"      Fold {fold_idx} failed: {str(fold_error)[:80]}")
                    continue
        
        except Exception as e:
            logger.warning(f"   ⚠️ Trial failed during CV: {str(e)[:100]}")
            raise TrialPruned()  # Tell Optuna the trial failed
        
        # Return mean validation score
        if scores:
            mean_val = np.mean(scores)
            return mean_val
        else:
            logger.warning(f"   ⚠️ Trial had no valid folds")
            raise TrialPruned()
    
    # 4. CREATE OPTUNA STUDY WITH TPE SAMPLER (BAYESIAN OPTIMIZATION)
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        sampler=sampler,
        direction='maximize'
    )
    
    # 5. SETUP REAL-TIME MONITORING CALLBACK
    optuna_callback = OptunaRealtimeCallback(
        output_dir=out,
        telemetry=telemetry,
        enable_json_stream=True
    )
    
    # 6. RUN OPTIMIZATION
    logger.info(f"🚀 Starting {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, callbacks=[optuna_callback])
    
    # 7. GET BEST TRIAL RESULTS (with fallback)
    if len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE])) == 0:
        logger.warning(f"⚠️ No completed trials. Using fixed parameters as fallback.")
        best_params = fixed_params
        best_score = 0.0
        
        # Train a single model with fixed params for final output
        m, p, s, best_m = _train_fallback_model(
            X_train, y_train, X_test, y_test,
            groups_train, cv, task_type, enable_overfitting_detector,
            fixed_params, logger, n_splits
        )
    else:
        best_trial = study.best_trial
        best_params = dict(best_trial.params)
        best_score = best_trial.value
        
        logger.info(f"\n✅ Optuna Optimization Complete!")
        logger.info(f"   Best Trial #{best_trial.number}: Score = {best_score:.4f}")
        logger.info(f"   Best Parameters: {best_params}")
    
    # 8. FINAL REFIT WITH BEST PARAMETERS
    best_p = {**best_params, **fixed_params}
    
    X_train_df = pd.DataFrame(X_train) if isinstance(X_train, np.ndarray) else X_train
    y_train_ser = pd.Series(y_train) if isinstance(y_train, np.ndarray) else y_train
    
    if enable_group_cv:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        idx_train, idx_val = next(gss.split(X_train_df, y_train_ser, groups=groups_train))
        X_fin_tr, X_fin_val = X_train_df.iloc[idx_train], X_train_df.iloc[idx_val]
        y_fin_tr, y_fin_val = y_train_ser.iloc[idx_train], y_train_ser.iloc[idx_val]
    else:
        X_fin_tr, X_fin_val, y_fin_tr, y_fin_val = train_test_split(
            X_train_df, y_train_ser, test_size=0.15, random_state=42,
            stratify=y_train_ser if task_type == 'Classification' else None
        )

    best_m = model_cls(**best_p, verbose=False).fit(
        X_fin_tr, y_fin_tr, 
        eval_set=(X_fin_val, y_fin_val), 
        early_stopping_rounds=50, 
        use_best_model=True
    )
    
    # 9. FINAL EVALUATION & LOGGING
    y_pred_train_final = best_m.predict(X_fin_tr) 
    y_pred_test_final = best_m.predict(X_test)

    _log_diagnostic_report(y_fin_tr, y_pred_train_final, y_test, y_pred_test_final, task_type)
            
    return best_m, best_p, float(best_score), {"best_val_score": float(best_score)}, []

def _log_diagnostic_report(y_tr, p_tr, y_te, p_te, task_type):
    """Internal helper to print detailed performance and interpretation guides."""
    logger = get_logger("workflow_16s")
    logger.info(f"\n{'='*20} FINAL PERFORMANCE REPORT {'='*20}")
    
    if task_type == 'Regression':
        r2_tr, r2_te = r2_score(y_tr, p_tr), r2_score(y_te, p_te)
        rmse_tr, rmse_te = np.sqrt(mean_squared_error(y_tr, p_tr)), np.sqrt(mean_squared_error(y_te, p_te))
        
        logger.info(f"R2 SCORE  | Train: {r2_tr:.3f} | Test: {r2_te:.3f} | Gap: {r2_tr-r2_te:.3f}")
        logger.info(f"RMSE      | Train: {rmse_tr:.3f} | Test: {rmse_te:.3f} | Gap: {rmse_te-rmse_tr:.3f}")
        
        logger.info("\n[INTERPRETATION GUIDE]")
        logger.info("- Large R2 Gap: High variance; the model is over-tuning to training patterns.")
        logger.info("- Large RMSE Gap vs Small R2 Gap: The model generalizes the 'trend' well, but fails "
                    "\n  on magnitude (outliers) in the test set. Check for group-specific distribution shifts.")
    else:
        mcc_tr, mcc_te = matthews_corrcoef(y_tr, p_tr), matthews_corrcoef(y_te, p_te)
        logger.info(f"MCC SCORE | Train: {mcc_tr:.3f} | Test: {mcc_te:.3f} | Gap: {mcc_tr-mcc_te:.3f}")
        
    logger.info(f"{'='*60}\n")


# FEATURE SELECTION ORCHESTRATOR

def catboost_feature_selection(
    metadata: pd.DataFrame, 
    features: pd.DataFrame, 
    output_dir: PathLike, 
    group_col: str, 
    cv_groups: Optional[Union[str, np.ndarray, pd.Series]] = None, 
    cv_strategy: Optional[Any] = None, 
    method: Literal['rfe', 'shap', 'lasso', 'chi_squared', 'select_k_best'] = 'rfe', 
    auto_fix_compositionality: bool = False,
    test_indices: Optional[list] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Orchestrates biomarker identification with technical noise diagnostics and proxy-aware recovery.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Sample metadata
    features : pd.DataFrame
        Feature matrix (CLR-transformed)
    output_dir : PathLike
        Output directory for results
    group_col : str
        Target column name in metadata
    cv_groups : Optional[Union[str, np.ndarray, pd.Series]]
        Group labels for cross-validation
    cv_strategy : Optional[Any]
        Cross-validation strategy object
    method : Literal
        Feature selection method ('rfe', 'shap', etc.)
    auto_fix_compositionality : bool
        If True, detect and fix CLR compositionality issues before processing
    **kwargs : Additional parameters
    
    Returns
    -------
    Dict[str, Any]
        Results dictionary with top_features, importances, etc.
    """
    logger = get_logger("workflow_16s")
    # 1. Setup Versioned Output Directory
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out: Path = Path(output_dir) / f"{method}_{timestamp}"
    out.mkdir(parents=True, exist_ok=True)
    
    # 2. Index Synchronization & Splitting
    common = features.index.intersection(metadata.index)
    X = features.loc[common]
    y = metadata.loc[common, group_col]
    # Ensure y is a Series (handle edge case where single column access might return scalar)
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=common, name=group_col)
    meta = metadata.loc[common]
    g_arr = meta[cv_groups].values if isinstance(cv_groups, str) else cv_groups
    
    if isinstance(g_arr, (pd.Series, pd.Index)):
        g_arr = g_arr.to_numpy()
    elif g_arr is not None and not isinstance(g_arr, np.ndarray):
        g_arr = np.array(g_arr)
    
    # ✅ AUTO-FIX COMPOSITIONALITY: Check and fix CLR if needed
    if auto_fix_compositionality:
        logger.info(" 🔧 Checking for CLR compositionality issues...")
        from .validation import fix_compositionality_if_needed
        X = fix_compositionality_if_needed(X, tolerance=1e-3, auto_fix=True)
        logger.info(" ✅ Compositionality check complete")
    
    X_tr, X_te, y_tr, y_te, g_tr, g_te = filter_data(
        X, y, meta, group_col, cv_groups=g_arr, test_indices=test_indices, **kwargs
    )

    # 🚨 SAFETY CHECK: Abort if filtering removed all samples
    if X_tr.empty or y_tr.empty or len(X_tr) < 2:
        logger.error(f"❌ Data filtering resulted in insufficient samples ({len(X_tr)} remaining). Aborting this strategy.")
        return {
            'strategy': kwargs.get('cv_strategy', 'unknown'),
            'top_features': [],
            'model_score': 0.0,
            'error': 'Insufficient samples after filtering'
        }

    # 3. Categorical Handling (Batch Effects)
    batch_cols: List[str] = ['batch_original', 'study_accession']
    cat_f: List[str] = [c for c in batch_cols if c in X_tr.columns]
    for col in cat_f:
        X_tr[col], X_te[col] = X_tr[col].astype(str), X_te[col].astype(str)

    # 4. Dynamic Parameter Sanitization
    task_type: str = kwargs.get('task_type', 'Classification')
    model_cls = CatBoostClassifier if task_type == 'Classification' else CatBoostRegressor
    valid_params = model_cls().get_params().keys()
    
    # FIX: Extract param_grid BEFORE sanitization so it doesn't get stripped out
    param_grid = kwargs.get('param_grid')
    
    # FIX: Provide default grid if none is passed to ensure optimization happens
    if param_grid is None:
        param_grid = {
            'depth': [4, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'l2_leaf_reg': [3, 7]
        }
        logger.info(f"⚙️ No param_grid provided. Using default grid: {param_grid}")

    safe_kwargs: Dict[str, Any] = {k: v for k, v in kwargs.items() if k in valid_params}
    
    # Block list for orchestration-only parameters (not meant for CatBoost)
    blocklist = ['cv_strategy', 'n_top_features', 'n_top_final', 'test_size', 'num_features', 'output_dir', 'param_grid', 'n_trials', 'telemetry']
    for k in blocklist:
        safe_kwargs.pop(k, None)
    
    if cat_f:
        safe_kwargs['cat_features'] = cat_f

    # 5. Selection & Optimization (Pass 'out' to allow for proxy map logging)
    X_tr_s, X_te_s, sel, y_tr_n, y_te_n = perform_feature_selection(
        X_tr, y_tr, X_te, y_te, feature_selection=method, output_dir=out, **kwargs
    )

    literal_task_type: Literal['Classification', 'Regression'] = (
        'Classification' if str(task_type) == 'Classification' else 'Regression'
    )
    
    # Extract optimization parameters
    n_trials = kwargs.get('n_trials', 20)  # Number of Optuna trials
    telemetry = kwargs.get('telemetry', None)  # Optional telemetry for dashboard
    
    m, p, s, t_scores, figs = grid_search(
        X_tr_s, y_tr_n, X_te_s, y_te_n, 
        groups_train=g_tr, 
        task_type=literal_task_type, 
        param_grid=param_grid,  
        fixed_params={'cat_features': cat_f}, 
        output_dir=out,
        n_trials=n_trials,
        telemetry=telemetry,
        **safe_kwargs
    )

    # 6. Batch Diagnostic & Feature Importance
    raw_importances: pd.Series = pd.Series(m.get_feature_importance(), index=X_tr_s.columns).sort_values(ascending=False)
    
    if len(raw_importances) > 0 and raw_importances.index[0] in batch_cols:
        logger.warning(f"⚠️ HIGH BIAS ALERT: '{raw_importances.index[0]}' is the dominant feature.")
    
    _plot_batch_diagnostic(raw_importances, batch_cols, out)

    # Cleanup Biomarker list & Annotate Proxies
    clean_importances = raw_importances.drop(batch_cols, errors='ignore')
    n_top: int = kwargs.get('n_top_features', 20)
    
    top_sorted = clean_importances.head(n_top).to_frame(name='importance').reset_index()
    top_sorted.rename(columns={'index': 'feature'}, inplace=True) 
    
    # Annotate with proxies from cluster_mapping.json
    top_annotated = annotate_proxies(top_sorted, out)
    
    # Use the standardized reporting utility
    save_feature_importances(top_annotated, out)

    # 7. SHAP Visual Interpretation
    _run_shap_analysis(m, X_tr_s, out, kwargs.get('n_top_final', 20), task_type)

    # 8. ZIP Archive Creation
    archive_path: str = shutil.make_archive(str(out), 'zip', out)
    logger.info(f"✓ Results archived to {archive_path}")

    return {
        'method': method,
        'model': m,  
        'top_features': top_annotated['feature'].tolist(), 
        'feature_importances': top_annotated['importance'].values,
        'best_score': s,
        'test_scores': t_scores,
        'archive': archive_path
    }

# --- SECTION 3: PRIVATE DIAGNOSTIC HELPERS ---

def _plot_batch_diagnostic(
    importances: pd.Series, 
    batch_cols: List[str], 
    out_dir: Path
) -> None:
    """
    Generates a bar plot comparing technical vs biological importance.
    """
    df = importances.reset_index(name='importance').rename(columns={'index': 'feature'})
    
    # 2. Safety check: ensure 'feature' column exists even if index had no name
    if 'feature' not in df.columns:
        df.rename(columns={df.columns[0]: 'feature'}, inplace=True)

    # Now this line will work safely
    df['type'] = df['feature'].apply(lambda x: 'Technical (Batch)' if x in batch_cols else 'Biological (Biomarker)')
    
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=df.head(30), x='importance', y='feature', hue='type', 
        palette={
            'Technical (Batch)': '#e74c3c', 
            'Biological (Biomarker)': '#3498db'
        }
    )
    plt.title("Technical Noise vs. Biological Signal")
    plt.savefig(out_dir / "batch_diagnostic_plot.png", bbox_inches='tight')
    plt.close()

def _analyze_shap_results(
    shap_values: Union[np.ndarray, List[np.ndarray]],
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    expected_values: Optional[Union[float, List[float]]] = None,
    output_dir: Optional[Path] = None,
    feature_values: Optional[pd.DataFrame] = None,
    model: Optional[Any] = None,
    y_true: Optional[Union[pd.Series, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Intelligently analyzes SHAP values and generates human-readable summaries.
    
    Args:
        shap_values: SHAP values (2D for binary, list of 2D for multiclass)
        feature_names: List of feature names
        class_names: List of class names (optional)
        expected_values: Base values for each class
        output_dir: Output directory for summary reports
    
    Returns:
        Dictionary with analysis results and summaries
    """
    logger = get_logger("workflow_16s")
    
    # Determine if multiclass
    is_multiclass = isinstance(shap_values, list)
    num_classes = len(shap_values) if is_multiclass else 1
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    logger.info("\n" + "="*80)
    logger.info("SHAP INTERPRETABILITY ANALYSIS")
    logger.info("="*80)
    
    analysis_results = {
        'feature_class_relationships': {},
        'top_features_by_class': {},
        'feature_contrasts': {},
        'summary_text': []
    }
    
    # Handle different SHAP value formats
    # Binary or True multiclass list: sv_list = list of 2D arrays (one per class)
    # CatBoost multiclass: 3D array (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        # True multiclass from sklearn or similar
        sv_list = shap_values
        is_multiclass = True
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # CatBoost multiclass: reshape from (n_samples, n_features, n_classes) to list
        sv_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        is_multiclass = True
    else:
        # Binary classification: 2D array
        sv_list = [shap_values]
        is_multiclass = False
    
    # 1. ANALYZE TOP FEATURES FOR EACH CLASS
    logger.info("\n[1] TOP FEATURES BY CLASS (Mean |SHAP|)")
    logger.info("-" * 80)
    
    for class_idx, sv in enumerate(sv_list):
        class_name = class_names[class_idx]
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(mean_shap)[::-1][:10]  # Top 10
        top_idx = [int(i) for i in top_idx]  # Convert numpy scalars to int
        
        top_features = [(feature_names[i], mean_shap[i]) for i in top_idx]
        analysis_results['top_features_by_class'][class_name] = top_features
        
        logger.info(f"\n{class_name}:")
        for rank, (feat, importance) in enumerate(top_features, 1):
            # Analyze direction (positive/negative impact)
            feat_idx = feature_names.index(feat)
            mean_direction = sv[:, feat_idx].mean()
            direction = "↑ increases" if mean_direction > 0 else "↓ decreases"
            logger.info(f"  {rank:2d}. {feat:30s} | Impact: {importance:.4f} ({direction} prediction)")
    
    # 2. FEATURE-CLASS RELATIONSHIPS
    logger.info("\n[2] FEATURE-CLASS RELATIONSHIPS (Direction & Magnitude)")
    logger.info("-" * 80)
    
    feature_relationships = {}
    
    for feat_idx, feat_name in enumerate(feature_names):
        feature_relationships[feat_name] = {}
        
        for class_idx, sv in enumerate(sv_list):
            class_name = class_names[class_idx]
            
            # Mean SHAP value (direction & magnitude)
            mean_impact = sv[:, feat_idx].mean()
            std_impact = sv[:, feat_idx].std()
            
            # Percentage of samples where this feature increases prediction
            n_increase = np.sum(sv[:, feat_idx] > 0) / len(sv) * 100
            
            feature_relationships[feat_name][class_name] = {
                'mean_impact': mean_impact,
                'std_impact': std_impact,
                'pct_increase': n_increase
            }
        
        analysis_results['feature_class_relationships'][feat_name] = feature_relationships[feat_name]
    
    # Log top 15 features with their class relationships
    # Compute mean importance across all classes
    all_class_means = []
    for sv in sv_list:
        all_class_means.append(np.abs(sv).mean(axis=0))
    
    # Stack and average across classes
    stacked_means = np.stack(all_class_means, axis=0)  # Shape: (num_classes, n_features)
    mean_importance_all = stacked_means.mean(axis=0)  # Shape: (n_features,)
    
    top_feat_idx = np.argsort(mean_importance_all)[::-1][:15]
    top_feat_idx = [int(i) for i in top_feat_idx]  # Convert numpy scalars to int
    
    for feat_idx in top_feat_idx:
        if feat_idx < len(feature_names):
            feat_name = feature_names[feat_idx]
            logger.info(f"\n{feat_name}:")
            
            for class_idx, class_name in enumerate(class_names):
                rel = feature_relationships[feat_name][class_name]
                direction = "↑" if rel['mean_impact'] > 0 else "↓"
                logger.info(
                    f"  {class_name:20s}: {direction} {abs(rel['mean_impact']):.4f} "
                    f"(±{rel['std_impact']:.4f}) | Increases in {rel['pct_increase']:.1f}% of samples"
                )
    
    # 3. FEATURE CONTRASTS (What distinguishes classes)
    if is_multiclass and num_classes > 1:
        logger.info("\n[3] CLASS-DISTINGUISHING FEATURES (Largest Contrasts)")
        logger.info("-" * 80)
        
        # For each pair of classes, find features that differ most
        for i in range(min(num_classes, 3)):  # Top 3 classes
            for j in range(i + 1, min(num_classes, 3)):
                class_i, class_j = class_names[i], class_names[j]
                sv_i, sv_j = sv_list[i], sv_list[j]
                
                # Mean SHAP difference
                mean_diff = np.abs(sv_i.mean(axis=0) - sv_j.mean(axis=0))
                top_contrast_idx = np.argsort(mean_diff)[::-1][:5]
                top_contrast_idx = [int(i) for i in top_contrast_idx]  # Convert numpy scalars to int
                
                analysis_results['feature_contrasts'][f"{class_i}_vs_{class_j}"] = [
                    (feature_names[idx], mean_diff[idx]) for idx in top_contrast_idx
                ]
                
                logger.info(f"\n{class_i} vs {class_j}:")
                for feat_idx in top_contrast_idx:
                    feat = feature_names[feat_idx]
                    contrast = mean_diff[feat_idx]
                    imp_i = sv_i[:, feat_idx].mean()
                    imp_j = sv_j[:, feat_idx].mean()
                    
                    winner = class_i if abs(imp_i) > abs(imp_j) else class_j
                    logger.info(
                        f"  {feat:30s} | Contrast: {contrast:.4f} "
                        f"(favors {winner})"
                    )
    
    # 4. INTERACTION INSIGHTS (Feature pairs that co-occur)
    logger.info("\n[4] FEATURE INTERACTION PATTERNS")
    logger.info("-" * 80)
    
    # Calculate correlation between top features' SHAP values
    top_n_features = 20
    all_shap = np.concatenate(sv_list, axis=0) if is_multiclass else sv_list[0]
    
    mean_importance = np.abs(all_shap).mean(axis=0)
    top_indices = np.argsort(mean_importance)[::-1][:top_n_features]
    top_indices = [int(i) for i in top_indices]  # Convert numpy scalars to int
    
    # Correlation of SHAP values (proxy for interaction)
    top_shap = all_shap[:, top_indices]
    shap_corr = np.corrcoef(top_shap.T)
    
    # Find strongest correlations (excluding diagonal)
    corr_pairs = []
    for i in range(len(top_indices)):
        for j in range(i + 1, len(top_indices)):
            corr = shap_corr[i, j]
            if abs(corr) > 0.3:  # Only significant correlations
                corr_pairs.append((
                    feature_names[top_indices[i]],
                    feature_names[top_indices[j]],
                    corr
                ))
    
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    if corr_pairs:
        logger.info("\nStrongest Feature Correlations (SHAP-based):")
        for feat_a, feat_b, corr in corr_pairs[:10]:
            direction = "coordinated" if corr > 0 else "opposing"
            logger.info(f"  {feat_a:30s} ↔ {feat_b:30s} | Corr: {corr:+.3f} ({direction})")
    else:
        logger.info("  No strong feature interactions detected (|corr| > 0.3)")
    
    # 5. SUMMARY STATISTICS
    logger.info("\n[5] SUMMARY STATISTICS")
    logger.info("-" * 80)
    
    for class_idx, class_name in enumerate(class_names):
        sv = sv_list[class_idx]
        mean_abs_shap = np.abs(sv).mean()
        max_impact_feat = feature_names[np.argmax(np.abs(sv).mean(axis=0))]
        
        logger.info(
            f"\n{class_name}:"
            f"\n  Average |SHAP|: {mean_abs_shap:.4f}"
            f"\n  Top Feature: {max_impact_feat}"
            f"\n  Num Features: {len(feature_names)}"
        )
    
    # 6. FEATURE VALUE STATISTICS (SUMMARY ONLY)
    if feature_values is not None:
        logger.info("\n[6] FEATURE VALUE STATISTICS SUMMARY")
        logger.info("-" * 80)
        
        feature_stats = {}
        for feat_idx, feat_name in enumerate(feature_names[:10]):  # Top 10 features only
            values = feature_values.iloc[:, feat_idx].values
            values_clean = values[~np.isnan(values)]
            
            if len(values_clean) > 0:
                stat_dict = {
                    'q1': float(np.percentile(values_clean, 25)),
                    'median': float(np.median(values_clean)),
                    'q3': float(np.percentile(values_clean, 75)),
                }
                feature_stats[feat_name] = stat_dict
                
                # Compact single-line summary
                logger.debug(
                    f"{feat_name:30s} | IQR: [{stat_dict['q1']:.2f}, {stat_dict['q3']:.2f}]"
                )
        
        analysis_results['feature_value_statistics'] = feature_stats
    
    # 7. SUMMARY STATUS (BRIEF)
    logger.info("\n[7] ANALYSIS COMPLETE")
    logger.info("-" * 80)
    logger.info(f"Analyzed {len(feature_names)} features across {num_classes} class(es)")
    logger.info(f"Samples analyzed: {len(all_shap)}")
    logger.info("Top features identified per class, feature interactions mapped")
            
    logger.info("  ✅ MOST STABLE FEATURES (Consistent SHAP values):")
    for rank, feat_idx in enumerate(reliable_idx, 1):
        if feat_idx < len(feature_names):
            logger.info(
                f"    {rank}. {feature_names[feat_idx]:30s} | "
                f"Consistency: {1 - shap_cv[feat_idx]:.3f} | "
                f"Std: {shap_std[feat_idx]:.4f}"
            )
    
    # 8. SIMPLE DECISION RULES (NEW ENHANCEMENT)
    if feature_values is not None:
        logger.info("\n[8] INTERPRETABLE DECISION RULES (If-Then Logic)")
        logger.info("-" * 80)
        
        for class_idx, class_name in enumerate(class_names):
            sv = sv_list[class_idx]
            
            # Get top 3 features for this class
            top_feat_idx = np.argsort(np.abs(sv).mean(axis=0))[::-1][:3]
            top_feat_idx = [int(i) for i in top_feat_idx]  # Convert numpy scalars to int
            
            logger.info(f"\n{class_name}:")
            rule_num = 1
            for feat_idx in top_feat_idx:
                if feat_idx < len(feature_names):
                    feat_name = feature_names[feat_idx]
                    values = feature_values.iloc[:, feat_idx].values
                    values_clean = values[~np.isnan(values)]
                    
                    if len(values_clean) > 0:
                        q75 = np.percentile(values_clean, 75)
                        mean_direction = sv[:, feat_idx].mean()
                        direction_text = "increases" if mean_direction > 0 else "decreases"
                        threshold_label = "HIGH" if mean_direction > 0 else "LOW"
                        threshold = q75 if mean_direction > 0 else np.percentile(values_clean, 25)
                        
                        logger.info(
                            f"  Rule {rule_num}: If {feat_name} is {threshold_label} "
                            f"(> {threshold:.2f}), then prediction {direction_text} for {class_name}"
                        )
                        rule_num += 1
    
    # 9. BORDERLINE SAMPLE IDENTIFICATION (NEW ENHANCEMENT)
    if model is not None and hasattr(model, 'predict_proba'):
        logger.info("\n[9] BORDERLINE SAMPLE IDENTIFICATION (Uncertain Predictions)")
        logger.info("-" * 80)
        
        try:
            proba = model.predict_proba(feature_values)
            
            # For multiclass, find samples near decision boundary
            max_proba = np.max(proba, axis=1)
            sorted_idx = np.argsort(max_proba)
            
            n_borderline = min(10, len(sorted_idx))
            borderline_idx = sorted_idx[:n_borderline]
            
            logger.info(f"\n  🎯 TOP {n_borderline} BORDERLINE SAMPLES (Most Uncertain):")
            for rank, idx in enumerate(borderline_idx, 1):
                pred_class = np.argmax(proba[idx])
                pred_prob = max_proba[idx]
                logger.info(
                    f"    {rank}. Sample {idx}: Pred={class_names[pred_class] if pred_class < len(class_names) else f'Class_{pred_class}'} "
                    f"(Confidence: {pred_prob:.1%}) | Second choice: {max_proba[idx]:.1%}"
                )
        except Exception as e:
            logger.debug(f"  ℹ️ Borderline analysis skipped: {e}")
    
    # 10. FEATURE MONOTONICITY ANALYSIS (NEW ENHANCEMENT)
    if feature_values is not None:
        logger.info("\n[10] FEATURE MONOTONICITY (Consistency of Feature-Prediction Relationship)")
        logger.info("-" * 80)
        
        for class_idx, class_name in enumerate(class_names):
            sv = sv_list[class_idx]
            
            # For each top feature, check if higher value consistently leads to higher SHAP
            top_feat_idx = np.argsort(np.abs(sv).mean(axis=0))[::-1][:10]
            top_feat_idx = [int(i) for i in top_feat_idx]  # Convert numpy scalars to int
            
            logger.info(f"\n{class_name}:")
            monotonic_count = 0
            
            for feat_idx in top_feat_idx:
                if feat_idx < len(feature_names):
                    feat_name = feature_names[feat_idx]
                    values = feature_values.iloc[:, feat_idx].values
                    values_clean = values[~np.isnan(values)]
                    
                    if len(values_clean) > 10:  # Only if enough samples
                        # Split by median and check consistency
                        median_val = np.median(values_clean)
                        mask = values >= median_val
                        
                        if np.sum(mask) > 5 and np.sum(~mask) > 5:
                            high_mean_shap = sv[mask, feat_idx].mean()
                            low_mean_shap = sv[~mask, feat_idx].mean()
                            is_monotonic = (high_mean_shap * low_mean_shap >= 0) or (high_mean_shap == low_mean_shap)
                            
                            if is_monotonic:
                                monotonic_count += 1
                                consistency = "✅ MONOTONIC"
                            else:
                                consistency = "⚠️ NON-MONOTONIC"
                            
                            logger.info(
                                f"  {feat_name:30s} {consistency}: "
                                f"High={high_mean_shap:+.4f}, Low={low_mean_shap:+.4f}"
                            )
            
            logger.info(f"  → {monotonic_count}/10 top features show monotonic relationships")
    
    # 11. CLASS CONFUSION ATTRIBUTION (NEW ENHANCEMENT)
    if model is not None and y_true is not None:
        logger.info("\n[11] CLASS CONFUSION ATTRIBUTION (Features Driving Misclassifications)")
        logger.info("-" * 80)
        
        try:
            # Get model predictions
            y_pred = model.predict(feature_values)
            
            # Find misclassified samples
            misclass_mask = y_pred != y_true
            misclass_count = np.sum(misclass_mask)
            
            if misclass_count > 0:
                logger.info(f"\n  Found {misclass_count} misclassifications ({misclass_count/len(y_true)*100:.1f}%)")
                
                # For each class confusion pair, identify contributing features
                if is_multiclass:
                    for i, class_i in enumerate(class_names[:3]):  # Top 3 classes
                        for j, class_j in enumerate(class_names[:3]):
                            if i < j:
                                # Samples confused as class_j but actually class_i
                                confusion_mask = (y_true == i) & (y_pred == j)
                                if np.sum(confusion_mask) > 0:
                                    # Features that pushed towards class_j
                                    sv_i = sv_list[i]
                                    sv_j = sv_list[j]
                                    
                                    # Average SHAP difference for confused samples
                                    confused_diff = sv_j[confusion_mask].mean(axis=0) - sv_i[confusion_mask].mean(axis=0)
                                    top_confused_idx = np.argsort(confused_diff)[::-1][:3]
                                    top_confused_idx = [int(i) for i in top_confused_idx]  # Convert numpy scalars to int
                                    
                                    logger.info(f"\n  {class_i} → {class_j} ({np.sum(confusion_mask)} cases):")
                                    for rank, feat_idx in enumerate(top_confused_idx, 1):
                                        if feat_idx < len(feature_names):
                                            logger.info(
                                                f"    {rank}. {feature_names[feat_idx]:30s} | "
                                                f"Confusion Driver: {confused_diff[feat_idx]:+.4f}"
                                            )
                else:
                    # Binary classification confusion
                    logger.info(f"\n  Misclassified: {misclass_count} samples")
                    sv = sv_list[0]
                    confused_shap = sv[misclass_mask].mean(axis=0)
                    correct_shap = sv[~misclass_mask].mean(axis=0)
                    
                    confusion_diff = confused_shap - correct_shap
                    top_confused_idx = np.argsort(np.abs(confusion_diff))[::-1][:5]
                    top_confused_idx = [int(i) for i in top_confused_idx]  # Convert numpy scalars to int
                    
                    logger.info(f"  Top features driving misclassification:")
                    for rank, feat_idx in enumerate(top_confused_idx, 1):
                        if feat_idx < len(feature_names):
                            logger.info(
                                f"    {rank}. {feature_names[feat_idx]:30s} | "
                                f"Error Driver: {confusion_diff[feat_idx]:+.4f}"
                            )
            else:
                logger.info(f"\n  ✅ Perfect classification! No misclassifications detected.")
        except Exception as e:
            logger.debug(f"  ℹ️ Confusion analysis skipped: {e}")
    
    logger.info("\n" + "="*80 + "\n")
    
    return analysis_results


def _run_shap_analysis(
    model: Any, 
    X: pd.DataFrame, 
    output_path: Path, 
    n_top: int, 
    task_type: str
) -> None:
    """
    Calculates SHAP values and triggers the modularized plotting suite.
    For multiclass: generates per-class SHAP plots showing feature importance for each class.
    For binary: generates standard SHAP plots for the positive class.
    Also generates intelligent analysis summaries.
    """
    logger = get_logger("workflow_16s")
    try:
        # 1. Initialize Explainer
        # TreeExplainer is optimized for CatBoost/XGBoost/RF
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        
        # Sample for performance if dataset is massive
        sample_size = min(1000, len(X))
        sample_X = X.sample(sample_size, random_state=42)
        shap_values = explainer.shap_values(sample_X)
        
        # 2. Detect Task Type & Handle Accordingly
        # Check for multiclass: either list of arrays OR 3D numpy array
        is_multiclass = isinstance(shap_values, list) or (isinstance(shap_values, np.ndarray) and shap_values.ndim == 3)
        
        if is_multiclass:
            # Multiclass: convert to list format if it's a 3D array
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # CatBoost format: (n_samples, n_features, n_classes)
                shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
            
            num_classes = len(shap_values)
            
            logger.info(f"📊 Multiclass detection: {num_classes} classes")
            logger.info(f"🎨 Generating per-class SHAP plots ({sample_size} samples each)...")
            
            # Get class labels from model if available, otherwise use numeric indices
            try:
                class_names = list(model.classes_)
            except AttributeError:
                class_names = [f"Class_{i}" for i in range(num_classes)]
            
            # Generate SHAP plots for each class (with rate limiting to prevent "too many open files")
            import time
            import gc
            
            for class_idx, (shap_vals, class_name) in enumerate(zip(shap_values, class_names)):
                # Base value for this class
                ev = explainer.expected_value
                if isinstance(ev, (list, np.ndarray)) and len(ev) > class_idx:
                    bv = ev[class_idx]
                else:
                    bv = np.mean(ev) if isinstance(ev, (list, np.ndarray)) else ev
                
                # Create class-specific output directory
                class_output = Path(output_path) / f"class_{class_name}"
                class_output.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"   ├─ Generating plots for {class_name}... ({class_idx+1}/{num_classes})")
                
                plot_shap(
                    base_value=float(bv[0]) if isinstance(bv, (list, np.ndarray)) else float(bv),
                    shap_values=shap_vals,
                    feature_values=sample_X.values,
                    feature_names=sample_X.columns.tolist(),
                    n_features=n_top,
                    output_dir=class_output,
                    interaction_feature='auto',
                    is_multiclass_avg=False  # Each class gets its own full analysis
                )
                logger.info(f"   └─ ✅ {class_name} SHAP plots saved")
                
                # Rate limiting: prevent "too many open files" error by closing resources between classes
                # Kaleido PNG rendering can keep file handles open, so add explicit cleanup
                gc.collect()
                if class_idx < num_classes - 1:  # Don't delay after last class
                    time.sleep(0.5)  # 500ms between class plots for file handle cleanup
            
            # Run intelligent analysis on multiclass SHAP values
            analysis = _analyze_shap_results(
                shap_values=shap_values,
                feature_names=sample_X.columns.tolist(),
                class_names=class_names,
                expected_values=explainer.expected_value,
                output_dir=output_path,
                feature_values=sample_X,
                model=model,
                y_true=None  # Would need to pass actual labels if available
            )
            
        else:
            # Binary classification: single 2D array
            logger.info(f"📊 Binary classification detected (2 classes)")
            logger.info(f"🎨 Generating SHAP plots for {sample_size} samples...")
            
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)) and len(ev) > 1:
                bv = ev[1]  # Use positive class
                class_names = [model.classes_[0] if hasattr(model, 'classes_') else 'Negative',
                              model.classes_[1] if hasattr(model, 'classes_') else 'Positive']
            else:
                bv = ev
                class_names = ['Positive']
            
            plot_shap(
                base_value=float(bv[0]) if isinstance(bv, (list, np.ndarray)) else float(bv),
                shap_values=shap_values,
                feature_values=sample_X.values,
                feature_names=sample_X.columns.tolist(),
                n_features=n_top,
                output_dir=output_path,
                interaction_feature='auto',
                is_multiclass_avg=False
            )
            
            # Run intelligent analysis on binary SHAP values
            analysis = _analyze_shap_results(
                shap_values=shap_values,
                feature_names=sample_X.columns.tolist(),
                class_names=class_names,
                expected_values=ev,
                output_dir=output_path,
                feature_values=sample_X,
                model=model,
                y_true=None  # Would need to pass actual labels if available
            )
            
    except Exception as e:
        logger.warning(f"⚠️ SHAP diagnostics skipped: {e}")
        logger.debug(traceback.format_exc())
        
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit

def perform_spatial_cv_split(X, y, coordinates, n_folds=5, test_size=0.2, random_state=42):
    """
    Performs Spatial Cross-Validation by clustering coordinates into spatial blocks.
    
    Args:
        X: Feature matrix
        y: Target array
        coordinates (pd.DataFrame): DataFrame with 'latitude' and 'longitude' columns.
        n_folds (int): Number of spatial clusters (blocks) to create.
        test_size (float): Fraction of blocks to hold out for testing.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger = get_logger("workflow_16s")
    # Remove samples with missing coordinates
    valid_coords = coordinates.dropna(subset=['latitude', 'longitude'])
    X_valid = X.loc[valid_coords.index]
    y_valid = y.loc[valid_coords.index]
    coords_valid = valid_coords[['latitude', 'longitude']]
    # 1. Create Spatial Blocks (Clusters)
    # We use KMeans to group samples into 'n_folds' geographic regions
    kmeans = KMeans(n_clusters=n_folds, random_state=random_state, n_init=10)
    spatial_folds = kmeans.fit_predict(coords_valid)
    
    # 2. Split based on these Spatial Blocks
    # This ensures that an entire geographic region is either in Train or Test, never both.
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X_valid, y_valid, groups=spatial_folds))
    
    # 3. Create the sets
    X_train, X_test = X_valid.iloc[train_idx], X_valid.iloc[test_idx]
    y_train, y_test = y_valid.iloc[train_idx], y_valid.iloc[test_idx]
    
    # Optional: Log the split visualization
    logger.info(f"   [Spatial CV] Created {n_folds} spatial blocks.")
    logger.info(f"   [Spatial CV] Holding out {len(set(spatial_folds[test_idx]))} blocks for testing.")
    
    return X_train, X_test, y_train, y_test