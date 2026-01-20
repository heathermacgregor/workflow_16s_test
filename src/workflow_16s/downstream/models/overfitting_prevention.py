"""
Comprehensive overfitting prevention and model validation methods.

This module provides robust validation techniques to ensure models generalize well:
1. Nested cross-validation for unbiased performance estimates
2. Learning curves to detect overfitting
3. Permutation tests for feature importance validation
4. Stability selection across bootstrap samples
5. Out-of-fold prediction tracking
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import cross_val_score, learning_curve, permutation_test_score, StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import matthews_corrcoef, r2_score, make_scorer, balanced_accuracy_score, mean_squared_error
from sklearn.base import clone
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger('workflow_16s')


def nested_cross_validation(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[np.ndarray] = None,
    outer_cv: int = 5,
    inner_cv: int = 3,
    task_type: str = 'Classification',
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Nested cross-validation for unbiased performance estimation.
    
    Outer loop: Performance estimation (never touches test data)
    Inner loop: Hyperparameter tuning (on training fold only)
    
    This prevents overfitting to the validation set during hyperparameter search.
    
    Parameters
    ----------
    model : sklearn estimator
        Model to validate (will be cloned for each fold)
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    groups : np.ndarray, optional
        Group labels for GroupKFold (e.g., batch identifiers)
    outer_cv : int
        Number of outer cross-validation folds
    inner_cv : int
        Number of inner cross-validation folds for hyperparameter tuning
    task_type : str
        'Classification' or 'Regression'
    random_state : int
        Random seed
        
    Returns
    -------
    dict
        - 'outer_scores': List of performance scores from outer loop
        - 'mean_score': Mean performance across outer folds
        - 'std_score': Standard deviation of performance
        - 'inner_scores': List of best inner CV scores per outer fold
        - 'overfitting_gap': Difference between inner and outer performance
    """
    logger.info(f"Running nested {outer_cv}-fold CV (inner: {inner_cv} folds)...")
    
    # Setup CV strategies
    if groups is not None:
        outer_splitter = GroupKFold(n_splits=outer_cv)
        inner_splitter = GroupKFold(n_splits=inner_cv)
    else:
        if task_type == 'Classification':
            outer_splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
            inner_splitter = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=random_state)
        else:
            outer_splitter = KFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
            inner_splitter = KFold(n_splits=inner_cv, shuffle=True, random_state=random_state)
    
    # Scoring metrics
    if task_type == 'Classification':
        scorer = make_scorer(matthews_corrcoef)
        metric_name = 'MCC'
    else:
        scorer = make_scorer(r2_score)
        metric_name = 'R²'
    
    outer_scores = []
    inner_best_scores = []
    
    # Outer loop: Unbiased performance estimation
    for fold_idx, (train_idx, test_idx) in enumerate(outer_splitter.split(X, y, groups)):
        logger.info(f"  Outer fold {fold_idx + 1}/{outer_cv}")
        
        # Split data
        X_train = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
        y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
        X_test = X.iloc[test_idx] if isinstance(X, pd.DataFrame) else X[test_idx]
        y_test = y.iloc[test_idx] if isinstance(y, pd.Series) else y[test_idx]
        
        groups_train = groups[train_idx] if groups is not None else None
        
        # Inner loop: Hyperparameter tuning (on training fold only)
        inner_scores = cross_val_score(
            clone(model),
            X_train,
            y_train,
            cv=inner_splitter,
            groups=groups_train,
            scoring=scorer,
            n_jobs=-1
        )
        inner_best_score = inner_scores.mean()
        inner_best_scores.append(inner_best_score)
        
        # Train final model on full training fold
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        
        # Evaluate on held-out test fold
        y_pred = model_clone.predict(X_test)
        if task_type == 'Classification':
            outer_score = matthews_corrcoef(y_test, y_pred)
        else:
            outer_score = r2_score(y_test, y_pred)
        
        outer_scores.append(outer_score)
        logger.info(f"    Inner CV {metric_name}: {inner_best_score:.3f}, Outer {metric_name}: {outer_score:.3f}")
    
    # Calculate statistics
    mean_outer = np.mean(outer_scores)
    std_outer = np.std(outer_scores)
    mean_inner = np.mean(inner_best_scores)
    overfitting_gap = mean_inner - mean_outer
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Nested CV Results ({metric_name}):")
    logger.info(f"  Outer CV (Unbiased): {mean_outer:.3f} ± {std_outer:.3f}")
    logger.info(f"  Inner CV (Optimistic): {mean_inner:.3f}")
    logger.info(f"  Overfitting Gap: {overfitting_gap:.3f}")
    
    if overfitting_gap > 0.15:
        logger.warning(f"⚠️  OVERFITTING DETECTED: Gap = {overfitting_gap:.3f} (threshold: 0.15)")
        logger.warning("  Model may not generalize well to new data!")
    elif overfitting_gap > 0.10:
        logger.warning(f"⚠️  MODERATE OVERFITTING: Gap = {overfitting_gap:.3f} (threshold: 0.10)")
    else:
        logger.info(f"✓ Model generalization appears good (gap < 0.10)")
    logger.info(f"{'='*60}\n")
    
    return {
        'outer_scores': outer_scores,
        'mean_score': mean_outer,
        'std_score': std_outer,
        'inner_scores': inner_best_scores,
        'overfitting_gap': overfitting_gap,
        'metric': metric_name
    }


def plot_learning_curves(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
    groups: Optional[np.ndarray] = None,
    cv: int = 5,
    task_type: str = 'Classification',
    n_jobs: int = -1,
    train_sizes: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Generate learning curves to diagnose overfitting.
    
    Learning curves show how model performance changes with training set size.
    - High training score + low validation score = overfitting
    - Both scores low = underfitting
    - Both scores high and close = good fit
    
    Parameters
    ----------
    model : sklearn estimator
        Model to evaluate
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    output_path : Path
        Path to save plot
    groups : np.ndarray, optional
        Group labels for GroupKFold
    cv : int
        Number of cross-validation folds
    task_type : str
        'Classification' or 'Regression'
    n_jobs : int
        Number of parallel jobs
    train_sizes : np.ndarray, optional
        Fractions of training set to use
        
    Returns
    -------
    dict
        Learning curve data (train_sizes, train_scores, validation_scores)
    """
    logger.info("Generating learning curves...")
    
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Setup CV and scoring
    if groups is not None:
        cv_splitter = GroupKFold(n_splits=cv)
    else:
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42) if task_type == 'Classification' else KFold(n_splits=cv, shuffle=True, random_state=42)
    
    if task_type == 'Classification':
        scorer = make_scorer(matthews_corrcoef)
        metric_name = 'MCC'
    else:
        scorer = make_scorer(r2_score)
        metric_name = 'R²'
    
    # Compute learning curves
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=cv_splitter,
        groups=groups,
        scoring=scorer,
        train_sizes=train_sizes,
        n_jobs=n_jobs,
        random_state=42
    )
    
    # Calculate means and std
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # Detect overfitting
    final_gap = train_mean[-1] - val_mean[-1]
    if final_gap > 0.15:
        logger.warning(f"⚠️  OVERFITTING: Training {metric_name} ({train_mean[-1]:.3f}) >> Validation {metric_name} ({val_mean[-1]:.3f})")
        logger.warning(f"    Gap = {final_gap:.3f} (threshold: 0.15)")
    elif final_gap > 0.10:
        logger.warning(f"⚠️  MODERATE OVERFITTING: Gap = {final_gap:.3f}")
    else:
        logger.info(f"✓ Good fit: Training-Validation gap = {final_gap:.3f} < 0.10")
    
    # Create plot
    fig = go.Figure()
    
    # Training score
    fig.add_trace(go.Scatter(
        x=train_sizes_abs,
        y=train_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes_abs, train_sizes_abs[::-1]]),
        y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Validation score
    fig.add_trace(go.Scatter(
        x=train_sizes_abs,
        y=val_mean,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes_abs, train_sizes_abs[::-1]]),
        y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=f'Learning Curves - {metric_name}<br><sub>Final Gap: {final_gap:.3f} (Train: {train_mean[-1]:.3f}, Val: {val_mean[-1]:.3f})</sub>',
        xaxis_title='Training Set Size',
        yaxis_title=metric_name,
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    logger.info(f"Learning curves saved to {output_path}")
    
    return {
        'train_sizes': train_sizes_abs,
        'train_scores': train_scores,
        'val_scores': val_scores,
        'overfitting_gap': final_gap
    }


def permutation_importance_test(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_permutations: int = 100,
    task_type: str = 'Classification',
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Permutation test to validate that model performance is not due to chance.
    
    Randomly shuffles target labels and refits model. If original model is not
    significantly better than permuted models, features may not be predictive.
    
    Parameters
    ----------
    model : sklearn estimator
        Fitted model to test
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    n_permutations : int
        Number of permutation iterations
    task_type : str
        'Classification' or 'Regression'
    random_state : int
        Random seed
        
    Returns
    -------
    dict
        - 'score': Actual model score
        - 'permutation_scores': Scores from permuted data
        - 'p_value': Probability that result is due to chance
        - 'is_significant': Whether p < 0.05
    """
    logger.info(f"Running permutation test ({n_permutations} permutations)...")
    
    if task_type == 'Classification':
        scorer = make_scorer(matthews_corrcoef)
        metric_name = 'MCC'
    else:
        scorer = make_scorer(r2_score)
        metric_name = 'R²'
    
    # Perform permutation test
    score, perm_scores, p_value = permutation_test_score(
        model,
        X,
        y,
        scoring=scorer,
        cv=5,
        n_permutations=n_permutations,
        n_jobs=-1,
        random_state=random_state
    )
    
    # Report results
    logger.info(f"\nPermutation Test Results:")
    logger.info(f"  Actual {metric_name}: {score:.3f}")
    logger.info(f"  Permuted {metric_name}: {perm_scores.mean():.3f} ± {perm_scores.std():.3f}")
    logger.info(f"  p-value: {p_value:.4f}")
    
    if p_value < 0.001:
        logger.info(f"  ✓✓✓ HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_value < 0.01:
        logger.info(f"  ✓✓ VERY SIGNIFICANT (p < 0.01)")
    elif p_value < 0.05:
        logger.info(f"  ✓ SIGNIFICANT (p < 0.05)")
    else:
        logger.warning(f"  ⚠️  NOT SIGNIFICANT (p >= 0.05)")
        logger.warning(f"  Model may not be better than random!")
    
    return {
        'score': score,
        'permutation_scores': perm_scores,
        'p_value': p_value,
        'is_significant': p_value < 0.05,
        'metric': metric_name
    }


def stability_selection(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_bootstrap: int = 100,
    sample_fraction: float = 0.8,
    threshold: float = 0.7,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Stability selection to identify robust features across bootstrap samples.
    
    Features selected consistently across many bootstrap samples are more likely
    to be truly predictive rather than overfitting artifacts.
    
    Parameters
    ----------
    model : sklearn estimator
        Model with feature_importances_ or coef_ attribute
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    n_bootstrap : int
        Number of bootstrap samples
    sample_fraction : float
        Fraction of data to sample in each bootstrap
    threshold : float
        Minimum selection frequency to consider feature stable
    random_state : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Features with selection frequencies, importance stats
    """
    logger.info(f"Running stability selection ({n_bootstrap} bootstraps, threshold={threshold})...")
    
    np.random.seed(random_state)
    feature_selection_freq = pd.Series(0, index=X.columns)
    feature_importance_vals = {col: [] for col in X.columns}
    
    n_samples = int(len(X) * sample_fraction)
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        boot_idx = np.random.choice(len(X), size=n_samples, replace=True)
        X_boot = X.iloc[boot_idx]
        y_boot = y.iloc[boot_idx]
        
        # Fit model
        model_clone = clone(model)
        model_clone.fit(X_boot, y_boot)
        
        # Extract feature importances
        if hasattr(model_clone, 'feature_importances_'):
            importances = model_clone.feature_importances_
        elif hasattr(model_clone, 'coef_'):
            importances = np.abs(model_clone.coef_)
        else:
            logger.error("Model has no feature_importances_ or coef_ attribute")
            return pd.DataFrame()
        
        # Record importances
        for idx, col in enumerate(X.columns):
            feature_importance_vals[col].append(importances[idx])
        
        # Select top 20% of features
        top_k = max(1, int(0.2 * len(X.columns)))
        top_features = X.columns[np.argsort(importances)[-top_k:]]
        feature_selection_freq[top_features] += 1
    
    # Calculate statistics
    selection_freq = feature_selection_freq / n_bootstrap
    stable_features = selection_freq[selection_freq >= threshold].sort_values(ascending=False)
    
    # Build results dataframe
    results = pd.DataFrame({
        'Feature': X.columns,
        'Selection_Frequency': selection_freq,
        'Mean_Importance': [np.mean(feature_importance_vals[col]) for col in X.columns],
        'Std_Importance': [np.std(feature_importance_vals[col]) for col in X.columns],
        'Is_Stable': selection_freq >= threshold
    }).sort_values('Selection_Frequency', ascending=False)
    
    logger.info(f"\nStability Selection Results:")
    logger.info(f"  Total features: {len(X.columns)}")
    logger.info(f"  Stable features (freq >= {threshold}): {len(stable_features)}")
    if len(stable_features) > 0:
        logger.info(f"  Top 5 stable features:")
        for feat, freq in stable_features.head(5).items():
            logger.info(f"    - {feat}: {freq:.2%}")
    else:
        logger.warning(f"  ⚠️  No features meet stability threshold!")
        logger.warning(f"  Model may be overfitting or features are unstable.")
    
    return results


def run_comprehensive_validation(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    target_name: str,
    groups: Optional[np.ndarray] = None,
    task_type: str = 'Classification',
    n_permutations: int = 100,
    n_bootstrap: int = 50,
    quick_mode: bool = False
) -> Dict[str, Any]:
    """
    Run all validation methods to comprehensively assess overfitting.
    
    Parameters
    ----------
    model : sklearn estimator
        Model to validate
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    output_dir : Path
        Directory to save validation results
    target_name : str
        Name of target variable (for labeling)
    groups : np.ndarray, optional
        Group labels for batch-aware validation
    task_type : str
        'Classification' or 'Regression'
    n_permutations : int
        Permutation test iterations
    n_bootstrap : int
        Bootstrap samples for stability selection
    quick_mode : bool
        If True, skip expensive computations
        
    Returns
    -------
    dict
        Validation results from all methods
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"COMPREHENSIVE OVERFITTING VALIDATION: {target_name}")
    logger.info(f"{'='*70}")
    
    output_dir = Path(output_dir) / "overfitting_validation" / target_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Nested Cross-Validation
    results['nested_cv'] = nested_cross_validation(
        model, X, y, groups=groups, task_type=task_type,
        outer_cv=5 if not quick_mode else 3,
        inner_cv=3
    )
    
    # 2. Learning Curves
    results['learning_curves'] = plot_learning_curves(
        model, X, y,
        output_path=output_dir / "learning_curves.html",
        groups=groups,
        cv=5 if not quick_mode else 3,
        task_type=task_type
    )
    
    # 3. Permutation Test
    if not quick_mode:
        results['permutation_test'] = permutation_importance_test(
            model, X, y,
            n_permutations=n_permutations,
            task_type=task_type
        )
    
    # 4. Stability Selection
    if not quick_mode and hasattr(model, 'feature_importances_'):
        results['stability_selection'] = stability_selection(
            model, X, y,
            n_bootstrap=n_bootstrap
        )
        
        # Save stability results
        stability_df = results['stability_selection']
        stability_df.to_csv(output_dir / "stability_selection.csv", index=False)
    
    # Summary report
    logger.info(f"\n{'='*70}")
    logger.info(f"VALIDATION SUMMARY: {target_name}")
    logger.info(f"{'='*70}")
    
    if 'nested_cv' in results:
        cv_gap = results['nested_cv']['overfitting_gap']
        logger.info(f"Nested CV Overfitting Gap: {cv_gap:.3f}")
    
    if 'learning_curves' in results:
        lc_gap = results['learning_curves']['overfitting_gap']
        logger.info(f"Learning Curve Gap: {lc_gap:.3f}")
    
    if 'permutation_test' in results:
        perm_p = results['permutation_test']['p_value']
        logger.info(f"Permutation Test p-value: {perm_p:.4f}")
    
    if 'stability_selection' in results:
        n_stable = results['stability_selection']['Is_Stable'].sum()
        logger.info(f"Stable Features: {n_stable} / {len(X.columns)}")
    
    # Overall assessment
    warnings = []
    if results.get('nested_cv', {}).get('overfitting_gap', 0) > 0.15:
        warnings.append("HIGH overfitting gap in nested CV")
    if results.get('learning_curves', {}).get('overfitting_gap', 0) > 0.15:
        warnings.append("HIGH overfitting gap in learning curves")
    if results.get('permutation_test', {}).get('p_value', 0) > 0.05:
        warnings.append("NOT significant in permutation test")
    if results.get('stability_selection', pd.DataFrame()).get('Is_Stable', pd.Series()).sum() < 5:
        warnings.append("FEW stable features")
    
    if warnings:
        logger.warning(f"\n⚠️  OVERFITTING CONCERNS:")
        for w in warnings:
            logger.warning(f"  - {w}")
        logger.warning(f"\nRecommendations:")
        logger.warning(f"  1. Increase regularization")
        logger.warning(f"  2. Reduce model complexity")
        logger.warning(f"  3. Collect more diverse training data")
        logger.warning(f"  4. Use feature selection to reduce dimensionality")
    else:
        logger.info(f"\n✓ Model validation passed all checks")
    
    logger.info(f"{'='*70}\n")
    
    return results
