# ==================================================================================== #
# machine_learning/nested_cv.py
# Nested Cross-Validation for Unbiased Performance Estimation
# ==================================================================================== #

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV, cross_val_score, StratifiedKFold, KFold
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, make_scorer
)
from catboost import CatBoostClassifier, CatBoostRegressor
from workflow_16s.utils.logger import get_logger

logger = get_logger("workflow_16s")

# ==================================================================================== #

def get_default_param_grid(task_type: str = 'classification') -> Dict[str, List]:
    """
    Get default hyperparameter grid for CatBoost models.
    
    Parameters
    ----------
    task_type : str, optional
        'classification' or 'regression', by default 'classification'
        
    Returns
    -------
    Dict[str, List]
        Parameter grid for GridSearchCV
    """
    if task_type == 'classification':
        return {
            'iterations': [100, 200, 300],
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5]
        }
    else:  # regression
        return {
            'iterations': [100, 200, 300],
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5]
        }


def nested_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str = 'classification',
    outer_cv: int = 5,
    inner_cv: int = 3,
    param_grid: Optional[Dict[str, List]] = None,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Perform nested cross-validation for unbiased performance estimation.
    
    Nested CV uses two loops:
    - Outer loop: Estimates model performance on unseen data
    - Inner loop: Optimizes hyperparameters
    
    This prevents data leakage and provides unbiased performance estimates,
    which is critical for microbiome ML where sample sizes are often limited.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (samples × features)
    y : np.ndarray
        Target variable
    task_type : str, optional
        'classification' or 'regression', by default 'classification'
    outer_cv : int, optional
        Number of outer CV folds, by default 5
    inner_cv : int, optional
        Number of inner CV folds, by default 3
    param_grid : Optional[Dict[str, List]], optional
        Hyperparameter grid, by default None (uses default grid)
    random_state : int, optional
        Random seed, by default 42
    n_jobs : int, optional
        Number of parallel jobs, by default -1 (all cores)
    verbose : int, optional
        Verbosity level, by default 1
        
    Returns
    -------
    Dict[str, Any]
        Results containing:
        - outer_scores: Performance on each outer fold
        - mean_score: Average performance across outer folds
        - std_score: Standard deviation of outer scores
        - best_params_per_fold: Optimal hyperparameters from each outer fold
        - feature_importances: Feature importance aggregated across folds
        
    Notes
    -----
    - Computationally intensive (outer_cv × inner_cv × n_params models)
    - Provides unbiased estimate of model generalization
    - Required for publication-quality ML results
    
    References
    ----------
    Varma, S., & Simon, R. (2006). Bias in error estimation when using 
    cross-validation for model selection. BMC Bioinformatics, 7(1), 91.
    """
    logger.info("=== Nested Cross-Validation ===")
    logger.info(f"Task: {task_type}")
    logger.info(f"Outer CV: {outer_cv} folds | Inner CV: {inner_cv} folds")
    logger.info(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")
    
    # Determine model and metrics
    if task_type == 'classification':
        n_classes = len(np.unique(y))
        
        if n_classes == 2:
            base_model = CatBoostClassifier(
                random_state=random_state,
                verbose=0,
                thread_count=1  # Parallelism handled by GridSearchCV
            )
            scoring = 'roc_auc'
            metric_name = 'ROC-AUC'
        else:
            base_model = CatBoostClassifier(
                random_state=random_state,
                verbose=0,
                thread_count=1
            )
            scoring = 'accuracy'
            metric_name = 'Accuracy'
            
        outer_cv_splitter = StratifiedKFold(
            n_splits=outer_cv, shuffle=True, random_state=random_state
        )
        inner_cv_splitter = StratifiedKFold(
            n_splits=inner_cv, shuffle=True, random_state=random_state
        )
        
    else:  # regression
        base_model = CatBoostRegressor(
            random_state=random_state,
            verbose=0,
            thread_count=1
        )
        scoring = 'r2'
        metric_name = 'R²'
        
        outer_cv_splitter = KFold(
            n_splits=outer_cv, shuffle=True, random_state=random_state
        )
        inner_cv_splitter = KFold(
            n_splits=inner_cv, shuffle=True, random_state=random_state
        )
    
    # Use default param grid if not provided
    if param_grid is None:
        param_grid = get_default_param_grid(task_type)
        logger.info(f"Using default parameter grid: {len(param_grid)} parameters")
    
    # Nested CV loop
    outer_scores = []
    best_params_per_fold = []
    feature_importances_per_fold = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv_splitter.split(X, y)):
        logger.info(f"\n--- Outer Fold {fold_idx + 1}/{outer_cv} ---")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Inner CV: Hyperparameter optimization
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=inner_cv_splitter,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=0,
            refit=True
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model from inner CV
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_params_per_fold.append(best_params)
        
        logger.info(f"Best params (inner CV): {best_params}")
        logger.info(f"Best inner score: {grid_search.best_score_:.4f}")
        
        # Evaluate on outer test fold
        y_pred = best_model.predict(X_test)
        
        if task_type == 'classification':
            if n_classes == 2:
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, y_pred_proba)
            else:
                score = accuracy_score(y_test, y_pred)
        else:
            score = r2_score(y_test, y_pred)
        
        outer_scores.append(score)
        logger.info(f"Outer test {metric_name}: {score:.4f}")
        
        # Store feature importances
        if hasattr(best_model, 'feature_importances_'):
            feature_importances_per_fold.append(best_model.feature_importances_)
    
    # Aggregate results
    mean_score = np.mean(outer_scores)
    std_score = np.std(outer_scores)
    
    logger.info("\n=== Nested CV Results ===")
    logger.info(f"{metric_name}: {mean_score:.4f} ± {std_score:.4f}")
    logger.info(f"Outer fold scores: {[f'{s:.4f}' for s in outer_scores]}")
    
    # Aggregate feature importances
    mean_importances = None
    if feature_importances_per_fold:
        mean_importances = np.mean(feature_importances_per_fold, axis=0)
        std_importances = np.std(feature_importances_per_fold, axis=0)
    
    results = {
        'task_type': task_type,
        'metric_name': metric_name,
        'outer_scores': outer_scores,
        'mean_score': mean_score,
        'std_score': std_score,
        'best_params_per_fold': best_params_per_fold,
        'feature_importances_mean': mean_importances,
        'feature_importances_std': std_importances if feature_importances_per_fold else None,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'outer_cv_folds': outer_cv,
        'inner_cv_folds': inner_cv
    }
    
    return results


def compare_with_simple_cv(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str = 'classification',
    cv_folds: int = 5,
    param_grid: Optional[Dict[str, List]] = None,
    random_state: int = 42,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Compare nested CV with simple CV to demonstrate bias.
    
    Simple CV optimistically estimates performance because hyperparameters
    are tuned on the same folds used for evaluation. Nested CV provides
    unbiased estimates by keeping test data completely separate.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target variable
    task_type : str, optional
        'classification' or 'regression'
    cv_folds : int, optional
        Number of CV folds, by default 5
    param_grid : Optional[Dict[str, List]], optional
        Hyperparameter grid
    random_state : int, optional
        Random seed
    n_jobs : int, optional
        Parallel jobs
        
    Returns
    -------
    Dict[str, Any]
        Comparison results showing bias in simple CV
    """
    logger.info("=== Comparing Simple CV vs Nested CV ===")
    
    # Simple CV (biased)
    if task_type == 'classification':
        base_model = CatBoostClassifier(random_state=random_state, verbose=0)
        cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scoring = 'roc_auc' if len(np.unique(y)) == 2 else 'accuracy'
    else:
        base_model = CatBoostRegressor(random_state=random_state, verbose=0)
        cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scoring = 'r2'
    
    if param_grid is None:
        param_grid = get_default_param_grid(task_type)
    
    # Simple CV: tune on entire dataset
    grid_search = GridSearchCV(
        base_model, param_grid, cv=cv_splitter, scoring=scoring, n_jobs=n_jobs
    )
    grid_search.fit(X, y)
    simple_cv_score = grid_search.best_score_
    
    logger.info(f"Simple CV score: {simple_cv_score:.4f} (BIASED - hyperparams tuned on test data)")
    
    # Nested CV (unbiased)
    nested_results = nested_cross_validation(
        X, y, task_type, outer_cv=cv_folds, inner_cv=3, 
        param_grid=param_grid, random_state=random_state, n_jobs=n_jobs
    )
    nested_cv_score = nested_results['mean_score']
    
    logger.info(f"Nested CV score: {nested_cv_score:.4f} ± {nested_results['std_score']:.4f} (UNBIASED)")
    
    bias = simple_cv_score - nested_cv_score
    logger.info(f"Optimistic bias: {bias:.4f} ({bias/simple_cv_score*100:.1f}%)")
    
    if bias > 0.05:
        logger.warning(
            "⚠️ Substantial optimistic bias detected! "
            "Simple CV overestimates performance. Use nested CV for publications."
        )
    
    return {
        'simple_cv_score': simple_cv_score,
        'nested_cv_score': nested_cv_score,
        'nested_cv_std': nested_results['std_score'],
        'bias': bias,
        'bias_percent': (bias / simple_cv_score) * 100,
        'nested_cv_results': nested_results
    }
