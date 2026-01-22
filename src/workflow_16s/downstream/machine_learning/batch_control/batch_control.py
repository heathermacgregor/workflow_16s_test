import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRegressor  # <--- IMPORT CATBOOST

from workflow_16s.utils.logger import get_logger

logger = get_logger("workflow_16s")

def get_model_class(task_type: str, algorithm: str):
    """Factory to return the correct model class based on task and algorithm."""
    if algorithm == 'catboost':
        if task_type == 'regression':
            return CatBoostRegressor
        return CatBoostClassifier
    else:
        # Default to Random Forest
        if task_type == 'regression':
            return RandomForestRegressor
        return RandomForestClassifier

def run_ml_with_batch_control(
    X_taxa: pd.DataFrame,
    y: pd.Series,
    batch_covariates: pd.DataFrame,
    target_col: str,
    task_type: str,
    plot_dir: Path,
    level: str,
    confounding_info: Dict[str, Any],
    batch_config: Dict[str, Any],
    model_algorithm: str = 'rf',   # <--- NEW ARGUMENT
    model_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Run ML models with three approaches: baseline, covariate-adjusted, stratified.
    Now supports both Random Forest and CatBoost.
    """
    results = {
        'target': target_col,
        'task_type': task_type,
        'level': level,
        'algorithm': model_algorithm, # Record which model was used
        'confounding': confounding_info,
        'models': {}
    }
    
    # 1. Setup Model Class and Params
    ModelClass = get_model_class(task_type, model_algorithm)
    
    # Define metric names
    if task_type == 'regression':
        metric_name = "R²"
        eval_func = r2_score
        stratify_opt = None
    else:
        metric_name = "Accuracy"
        eval_func = accuracy_score
        stratify_opt = y
    
    # Prepare params based on algorithm
    final_params = model_params.copy() if model_params else {}
    
    if model_algorithm == 'catboost':
        # CatBoost specific defaults
        final_params.setdefault('verbose', False)
        final_params.setdefault('allow_writing_files', False)
        final_params.setdefault('thread_count', 4)
    else:
        # RF specific defaults
        final_params.setdefault('n_estimators', 100)
        final_params.setdefault('max_depth', 15)
        final_params.setdefault('n_jobs', -1)
        final_params.setdefault('oob_score', True)  # Only for RF

    # ===================================================================================
    # APPROACH 1: BASELINE (No batch control)
    # ===================================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"BASELINE MODEL ({model_algorithm.upper()}) - {target_col}")
    logger.info(f"{'='*80}")
    
    X_baseline = X_taxa.fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(
        X_baseline, y, test_size=0.3, random_state=42, stratify=stratify_opt
    )
    
    model_baseline = ModelClass(**final_params)
    model_baseline.fit(X_train, y_train)
    y_pred_baseline = model_baseline.predict(X_test)
    
    baseline_score = eval_func(y_test, y_pred_baseline)
    
    # Handle OOB (RF only) vs just Test Score (CatBoost)
    baseline_oob = getattr(model_baseline, 'oob_score_', None)
    
    logger.info(f"✓ Test {metric_name}: {baseline_score:.3f}")
    if baseline_oob: logger.info(f"✓ OOB Score: {baseline_oob:.3f}")
    
    results['models']['baseline'] = {
        'test_score': baseline_score,
        'oob_score': baseline_oob,
        'feature_importances': getattr(model_baseline, 'feature_importances_', []).tolist()
    }
    
    # ===================================================================================
    # APPROACH 2: COVARIATE ADJUSTMENT
    # ===================================================================================
    if batch_config.get('covariate_adjustment', {}).get('enabled', False):
        logger.info(f"\n--- Covariate Adjusted ({model_algorithm.upper()}) ---")
        X_adjusted = pd.concat([X_taxa, batch_covariates], axis=1).fillna(0)
        
        X_train_adj, X_test_adj, y_train_adj, y_test_adj = train_test_split(
            X_adjusted, y, test_size=0.3, random_state=42, stratify=stratify_opt
        )
        
        model_adjusted = ModelClass(**final_params)
        model_adjusted.fit(X_train_adj, y_train_adj)
        y_pred_adj = model_adjusted.predict(X_test_adj)
        adjusted_score = eval_func(y_test_adj, y_pred_adj)
        
        # Calculate feature importance ratio (Batch vs Taxa)
        importances = getattr(model_adjusted, 'feature_importances_', np.zeros(X_adjusted.shape[1]))
        taxa_cols = set(X_taxa.columns)
        
        # Create DataFrame to sum importances
        feat_df = pd.DataFrame({'feat': X_adjusted.columns, 'imp': importances})
        taxa_imp = feat_df[feat_df['feat'].isin(taxa_cols)]['imp'].sum()
        batch_imp = feat_df[~feat_df['feat'].isin(taxa_cols)]['imp'].sum()
        total_imp = taxa_imp + batch_imp
        
        batch_frac = batch_imp / total_imp if total_imp > 0 else 0
        
        results['models']['covariate_adjusted'] = {
            'test_score': adjusted_score,
            'batch_importance_fraction': batch_frac
        }
        logger.info(f"✓ Adjusted {metric_name}: {adjusted_score:.3f}")
        logger.info(f"  Batch Importance: {batch_frac:.1%}")

    # ===================================================================================
    # APPROACH 3: STRATIFIED PREDICTION
    # ===================================================================================
    if batch_config.get('stratified_prediction', {}).get('enabled', False):
        logger.info(f"\n--- Stratified/Residual ({model_algorithm.upper()}) ---")
        
        # 1. Train Batch Model (Target ~ Batch)
        X_batch = batch_covariates.fillna(0)
        X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(
             X_batch, y, test_size=0.3, random_state=42, stratify=stratify_opt
        )
        
        # Use simpler params for batch model to avoid overfitting technical noise
        batch_model = ModelClass(**{**final_params, 'n_estimators': 50, 'max_depth': 6})
        batch_model.fit(X_b_train, y_b_train)
        
        y_pred_batch_all = batch_model.predict(X_batch)
        batch_score = eval_func(y_b_test, batch_model.predict(X_b_test))
        
        # 2. Calculate Residuals
        if task_type == 'regression':
            residuals = y - y_pred_batch_all
        else:
            # Classification residuals: 1 if wrong, 0 if right (Simple error modeling)
            # OR for probability: residuals = y_binary - y_prob
            residuals = (y != y_pred_batch_all).astype(float)
            
        # 3. Train Residual Model (Residuals ~ Taxa)
        # Residuals are always continuous-ish, so use Regressor usually
        # But if classification residuals are binary (0/1 error), we can use Classifier or Regressor
        # Here we default to Regressor to predict "Probability of Error" or "Magnitude of Residual"
        ResidClass = CatBoostRegressor if model_algorithm == 'catboost' else RandomForestRegressor
        
        X_r_train, X_r_test, r_train, r_test = train_test_split(
            X_taxa.fillna(0), residuals, test_size=0.3, random_state=42
        )
        
        model_resid = ResidClass(**final_params)
        model_resid.fit(X_r_train, r_train)
        resid_score = r2_score(r_test, model_resid.predict(X_r_test))
        
        results['models']['stratified'] = {
            'batch_model_score': batch_score,
            'residual_model_score': resid_score
        }
        logger.info(f"✓ Batch Model Score: {batch_score:.3f}")
        logger.info(f"✓ Residual Explained (R2): {resid_score:.3f}")

    # Save results to specific algo folder
    algo_dir = plot_dir / model_algorithm
    algo_dir.mkdir(exist_ok=True, parents=True)
    
    safe_target = re.sub(r'[^A-Za-z0-9_]+', '', target_col)
    with open(algo_dir / f"results_{safe_target}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results