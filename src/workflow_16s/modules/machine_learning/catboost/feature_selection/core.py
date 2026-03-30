# workflow_16s/modules/machine_learning/catboost/feature_selection/core.py

import itertools
import logging
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, Literal

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostClassifier, CatBoostRegressor
from optuna.samplers import TPESampler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    matthews_corrcoef, r2_score, roc_curve, roc_auc_score, 
    precision_recall_curve, average_precision_score, confusion_matrix,
    mean_squared_error
)
from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit, train_test_split

from .methods import annotate_proxies, perform_feature_selection
from .reporting import generate_shap_report, save_feature_importances
from .validation import filter_data

from workflow_16s.utils.logger import get_logger, with_logger
logger = get_logger("workflow_16s")
from workflow_16s.visualization.machine_learning import (
    plot_confusion_matrix, plot_precision_recall_curve, plot_predicted_vs_actual,
    plot_residuals, plot_roc_curve, plot_shap, 
)

PathLike = Union[str, Path]


@with_logger
def grid_search(
    X_train: pd.DataFrame,  y_train: Union[pd.Series, np.ndarray], 
    X_test: pd.DataFrame, y_test: Union[pd.Series, np.ndarray], 
    groups_train: Optional[Union[pd.Series, np.ndarray]] = None, 
    param_grid: Optional[Dict[str, List[Any]]] = None, 
    output_dir: Optional[PathLike] = None, n_splits: int = 5, n_trials: int = 20,
    task_type: Literal['Classification', 'Regression'] = 'Classification', 
    enable_overfitting_detector: bool = True, fixed_params: Optional[Dict[str, Any]] = None, 
    telemetry: Optional[Any] = None,
    **kwargs: Any
) -> Tuple[Any, Dict[str, Any], float, Dict[str, float], List[Any]]:
    """
    Performs a group-aware hyperparameter search using Optuna (Bayesian Optimization) 
    with real-time progress monitoring integrated into the EnhancedDashboardMonitor.
    
    Args:
        telemetry: Optional TelemetryCollector for live dashboard updates
        ... (other args as before)
    """
    out = Path(output_dir or "gs_out")
    out.mkdir(parents=True, exist_ok=True)
    
    # Supress optuna's default noisy logging so it doesn't clutter your custom logger
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    model_cls = CatBoostClassifier if task_type == 'Classification' else CatBoostRegressor
    fixed_params = fixed_params or {}
    param_grid = param_grid or {}
    
    # 1. CROSS-VALIDATION STRATEGY
    cv = GroupKFold(n_splits=n_splits) if groups_train is not None else KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Convert to pandas if needed for .iloc
    X_train_df = pd.DataFrame(X_train) if isinstance(X_train, np.ndarray) else X_train
    y_train_ser = pd.Series(y_train) if isinstance(y_train, np.ndarray) else y_train

    logger.info(f"🚀 Starting Optuna Bayesian Search ({n_trials} trials)...")
    
    # Set up real-time monitoring callback if telemetry available
    optuna_callback = None
    if telemetry is not None:
        try:
            from workflow_16s.downstream.machine_learning.optuna.realtime_monitor import create_optuna_callback
            optuna_callback = create_optuna_callback(
                output_dir=Path(output_dir) / "optuna_trials",
                telemetry=telemetry,
                enable_json_stream=True
            )
            logger.info("   ✅ Real-time Optuna monitoring enabled (dashboard integration active)")
        except Exception as e:
            logger.debug(f"   ⚠️ Could not enable real-time Optuna monitoring: {e}")

    def objective(trial: optuna.Trial) -> float:
        # Dynamically sample from the provided param_grid
        current_params = fixed_params.copy()
        for key, values in param_grid.items():
            # We use suggest_categorical to mimic GridSearch behavior on a discrete list of options
            current_params[key] = trial.suggest_categorical(key, values)
            
        scores = []
        
        for tr_i, val_i in cv.split(X_train_df, y_train_ser, groups=groups_train):
            m = model_cls(
                **current_params, 
                verbose=False,
                od_type='Iter' if enable_overfitting_detector else None,
                od_wait=50 if enable_overfitting_detector else None,
                allow_writing_files=False
            ).fit(
                X_train_df.iloc[tr_i], y_train_ser.iloc[tr_i], 
                eval_set=(X_train_df.iloc[val_i], y_train_ser.iloc[val_i]), 
                early_stopping_rounds=50,
                use_best_model=True
            )
            
            y_pred_val = m.predict(X_train_df.iloc[val_i])

            if task_type == 'Regression':
                scores.append(r2_score(y_train_ser.iloc[val_i], y_pred_val))
            else:
                scores.append(matthews_corrcoef(y_train_ser.iloc[val_i], y_pred_val))
                
        mean_val = np.mean(scores)
        return float(mean_val)

    # 2. RUN OPTIMIZATION
    # We want to maximize R2 or MCC
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    
    # Build callbacks list
    callbacks = []
    if optuna_callback is not None:
        callbacks.append(optuna_callback)
    
    study.optimize(objective, n_trials=n_trials, callbacks=callbacks if callbacks else None)

    best_p = {**study.best_params, **fixed_params}
    best_s = study.best_value
    
    logger.info(f"🏆 Best Params Found: {study.best_params} | Val Score: {best_s:.4f}")

    # 3. FINAL REFIT (Group-Aware Validation Split)
    if groups_train is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        idx_train, idx_val = next(gss.split(X_train, y_train, groups=groups_train))
        X_fin_tr, X_fin_val = X_train.iloc[idx_train], X_train.iloc[idx_val]
        y_fin_tr, y_fin_val = y_train_ser.iloc[idx_train], y_train_ser.iloc[idx_val]
    else:
        X_fin_tr, X_fin_val, y_fin_tr, y_fin_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42,
            stratify=y_train if task_type == 'Classification' else None
        )

    best_m = model_cls(**best_p, verbose=False).fit(
        X_fin_tr, y_fin_tr, 
        eval_set=(X_fin_val, y_fin_val), 
        early_stopping_rounds=50, 
        use_best_model=True
    )
    
    # 4. FINAL EVALUATION & LOGGING
    y_pred_train_final = best_m.predict(X_fin_tr) 
    y_pred_test_final = best_m.predict(X_test)

    _log_diagnostic_report(y_fin_tr, y_pred_train_final, y_test, y_pred_test_final, task_type)
            
    return best_m, best_p, float(best_s), {"best_val_score": float(best_s)}, []

@with_logger
def _log_diagnostic_report(y_tr, p_tr, y_te, p_te, task_type):
    """Internal helper to print detailed performance and interpretation guides."""
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


@with_logger
def catboost_feature_selection(
    metadata: pd.DataFrame, features: pd.DataFrame, output_dir: PathLike, group_col: str, 
    cv_groups: Optional[Union[str, np.ndarray, pd.Series]] = None, 
    cv_strategy: Optional[Any] = None, 
    method: Literal['rfe', 'shap', 'lasso', 'chi_squared', 'select_k_best'] = 'rfe', 
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Orchestrates biomarker identification with technical noise diagnostics and proxy-aware recovery.
    """
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
    
    X_tr, X_te, y_tr, y_te, g_tr, g_te = filter_data(
        X, y, meta, group_col, cv_groups=g_arr, **kwargs
    )

    # 2b. AUTO-FIX Compositionality Issues (if configured)
    # This re-applies CLR transformation if row sums deviate too much
    auto_fix_compositionality = kwargs.get('auto_fix_compositionality', False)
    if auto_fix_compositionality:
        logger.info("🔧 Checking compositionality and auto-fixing if needed...")
        from .validation import fix_compositionality_if_needed
        X_tr = fix_compositionality_if_needed(X_tr, auto_fix=True)
        X_te = fix_compositionality_if_needed(X_te, auto_fix=True)
        logger.info("✅ Compositionality check complete")

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
    
    # Block list for orchestration-only parameters
    blocklist = ['cv_strategy', 'n_top_features', 'n_top_final', 'test_size', 'num_features', 'output_dir', 'param_grid']
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
    
    m, p, s, t_scores, figs = grid_search(
        X_tr_s, y_tr_n, X_te_s, y_te_n, 
        groups_train=g_tr, 
        task_type=literal_task_type, 
        param_grid=param_grid,  
        fixed_params={'cat_features': cat_f}, 
        output_dir=out, 
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

@with_logger
def _run_shap_analysis(
    model: Any, X: pd.DataFrame, output_path: Path, n_top: int, task_type: str
) -> None:
    """
    Calculates SHAP values and triggers the modularized plotting suite.
    """
    try:
        # 1. Initialize Explainer
        # TreeExplainer is optimized for CatBoost/XGBoost/RF
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        
        # Sample for performance if dataset is massive
        sample_size = min(1000, len(X))
        sample_X = X.sample(sample_size, random_state=42)
        shap_values = explainer.shap_values(sample_X)
        
        # 2. Handle Task-Specific Base Values
        # CatBoost binary classification often returns a list or a 3D array for shap_values
        if isinstance(shap_values, list):
            # For Binary: index 1 is the 'Positive' class impact
            sv_to_plot = shap_values[1]
            # Fix: Only subscript if expected_value is a list or array
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)) and len(ev) > 1:
                bv = ev[1]
            else:
                bv = ev
        else:
            sv_to_plot = shap_values
            bv = explainer.expected_value

        # 👉 NEW: Clean the feature names for the y-axis
        cleaned_feature_names = []
        for col in sample_X.columns.tolist():
            # If it's a batch column or metadata, leave it alone
            if col in ['batch_original', 'study_accession'] or '__' not in col:
                cleaned_feature_names.append(col)
                continue
                
            # Split by standard delimiters
            parts = [p for p in re.split(r'[;|]', col) if p.strip()]
            
            # Grab the last valid part (most specific rank)
            last_valid = parts[-1] if parts else col
            
            # Strip prefixes like 'g__' or 's__' and clean underscores
            clean_name = re.sub(r'^[a-zA-Z0-9_]+__', '', last_valid)
            clean_name = clean_name.replace('_', ' ').strip()
            
            # If it's unclassified, maybe grab the next level up + "sp."
            if clean_name.lower() in ['unclassified', 'uncultured', '']:
                if len(parts) > 1:
                    parent = re.sub(r'^[a-zA-Z0-9_]+__', '', parts[-2]).replace('_', ' ').strip()
                    clean_name = f"{parent} sp."
                else:
                    clean_name = "Unclassified Taxon"
                    
            cleaned_feature_names.append(clean_name)

        logger.info(f"🎨 Generating Interpretability Suite (SHAP) for {sample_size} samples...")

        # 3. Call the standardized visualization orchestrator
        # This will generate Bar, Beeswarm, Heatmap, Waterfall, and Dependency plots
        plot_shap(
            base_value=float(bv[0]) if isinstance(bv, (list, np.ndarray)) else float(bv), # type: ignore
            shap_values=sv_to_plot,
            feature_values=sample_X.values,
            feature_names=cleaned_feature_names,
            n_features=n_top,
            output_dir=output_path,
            interaction_feature='auto'
        )
            
    except Exception as e:
        logger.warning(f"⚠️ SHAP diagnostics skipped: {e}")
        logger.debug(traceback.format_exc())
        
@with_logger
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
