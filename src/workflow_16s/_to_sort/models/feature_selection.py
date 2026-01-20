# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import itertools
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, cv, Pool
from scipy.stats import spearmanr
from sklearn.feature_selection import (
    chi2, f_classif, RFE, SelectFromModel, SelectKBest, VarianceThreshold
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, confusion_matrix, f1_score, 
    get_scorer, matthews_corrcoef, make_scorer, precision_recall_curve,
    roc_auc_score, roc_curve 
)
from sklearn.model_selection import StratifiedKFold, train_test_split

# Local Imports
from workflow_16s import constants
from workflow_16s.figures.figures import combine_figures_as_subplots
from workflow_16s.figures.models import (
    plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve, plot_shap
)
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ==================================== FUNCTIONS ===================================== #

def generate_shap_report(model, X: pd.DataFrame, K: int = 10) -> Tuple[str, pd.DataFrame]:
    """Generates a SHAP report and corresponding DataFrame with feature insights.
    
    Args:
        model: Trained model compatible with SHAP.
        X: Feature matrix used for explanation.
        K: Number of top features to include.
        
    Returns:
        Tuple containing the report string and a DataFrame with detailed metrics.
    """
    expl = shap.Explainer(model, X)
    sv_exp = expl(X)
    sv = sv_exp.values  # SHAP values: (n_samples, n_features)

    mean_abs = np.abs(sv).mean(axis=0)
    feat_idx = np.argsort(mean_abs)[::-1][:K]
    top_feats = list(X.columns[feat_idx])
    top_means = mean_abs[feat_idx]

    # Initialize lists to store DataFrame data
    features = []
    mean_abs_shap = []
    beeswarm_corr = []
    beeswarm_direction = []
    dependency_strength = []
    dependency_trend = []
    interaction_partner = []
    interaction_strength = []
    relationship_type = []

    lines = [f"**Top {K} features by average impact:**"]
    for feat, m in zip(top_feats, top_means):
        lines.append(f" • `{feat}` (mean |SHAP| = {m:.3f})")
        features.append(feat)
        mean_abs_shap.append(m)
    lines.append("")

    # Beeswarm interpretation
    lines.append("**Beeswarm interpretation:**")
    for feat in top_feats:
        vals = X[feat].values
        shap_vals = sv[:, X.columns.get_loc(feat)]
        rho, _ = spearmanr(vals, shap_vals)
        dir_text = "higher → higher prediction" if rho > 0 else "higher → lower prediction"
        lines.append(f" • `{feat}`: {dir_text} (ρ = {rho:.2f})")
        beeswarm_corr.append(rho)
        beeswarm_direction.append("positive" if rho > 0 else "negative")
    lines.append("")

    # Dependency interpretation
    lines.append("**Dependency‐plot interpretations:**")
    for feat in top_feats:
        vals = X[feat].values
        shap_vals = sv[:, X.columns.get_loc(feat)]
        rho, _ = spearmanr(vals, shap_vals)
        strength = "strong" if abs(rho) > 0.5 else "moderate" if abs(rho) > 0.3 else "weak"
        trend = "positive" if rho > 0 else "negative"
        lines.append(
            f" • `{feat}` shows a {strength} {trend} monotonic relationship (ρ = {rho:.2f})"
        )
        dependency_strength.append(strength)
        dependency_trend.append(trend)
    lines.append("")

    # Initialize interaction values as None
    mean_abs_int = None
    try:
        int_explainer = shap.TreeExplainer(model)
        int_vals = int_explainer.shap_interaction_values(X)
        mean_abs_int = np.abs(int_vals).mean(axis=0)
    except Exception:
        lines.append("Could not compute SHAP interaction values.")

    # Interactions with relationships
    lines.append("**Interaction summaries among top features:**")
    for i, feat in enumerate(top_feats):
        i_idx = X.columns.get_loc(feat)
        
        if mean_abs_int is not None:
            inter_strengths = mean_abs_int[i_idx].copy()
            inter_strengths[i_idx] = 0  # exclude self-interaction
            j_idx = inter_strengths.argmax()
            partner = X.columns[j_idx]
            score = inter_strengths[j_idx]

            # Characterize relationship
            xi = X[feat].values
            xj = X[partner].values
            shap_i = sv[:, i_idx]
            shap_j = sv[:, j_idx]

            rho_i = spearmanr(xj, shap_i)[0]
            rho_j = spearmanr(xi, shap_j)[0]

            if rho_i > 0 and rho_j > 0:
                relation = "mutually reinforcing"
            elif rho_i < 0 and rho_j < 0:
                relation = "mutually diminishing"
            elif rho_i * rho_j < 0:
                relation = "opposing or balancing"
            else:
                relation = "complex or weak"

            lines.append(
                f" • `{feat}` interacts most with `{partner}` "
                f"(mean |interaction SHAP| = {score:.3f}); relationship: {relation} "
                f"(ρ_feat→SHAP = {rho_j:.2f}, ρ_partner→SHAP = {rho_i:.2f})"
            )
            interaction_partner.append(partner)
            interaction_strength.append(score)
            relationship_type.append(relation)
        else:
            lines.append(f" • `{feat}`: No interaction data available")
            interaction_partner.append(None)
            interaction_strength.append(None)
            relationship_type.append(None)

    # Create DataFrame
    report_df = pd.DataFrame({
        'feature': features,
        'mean_abs_shap': mean_abs_shap,
        'beeswarm_correlation': beeswarm_corr,
        'beeswarm_direction': beeswarm_direction,
        'dependency_strength': dependency_strength,
        'dependency_trend': dependency_trend,
        'interaction_partner': interaction_partner,
        'interaction_strength': interaction_strength,
        'relationship_type': relationship_type
    })

    return "\n".join(lines), report_df


def _validate_inputs(X_train, y_train, X_test, y_test):
    """Validate input alignment and data integrity.
    
    Args:
        X_train:
        y_train:
        X_test:
        y_test:
    """
    if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"X_train [{X_train.shape[0]}], y_train [{y_train.shape[0]}], "
            f"X_test [{X_test.shape[0]}], and y_test [{y_test.shape[0]}] "
            "must have the same number of samples"
        )
    
    if X_train.empty or X_test.empty:
        raise ValueError("Input dataframes cannot be empty")
    
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_train and X_test must be pandas DataFrames")
    
    if not isinstance(y_train, pd.Series) or not isinstance(y_test, pd.Series):
        raise TypeError("y_train and y_test must be pandas Series")
        

def _save_dataframe(
    df: pd.DataFrame, 
    output_path: Union[str, Path], 
    file_format: str = 'csv'
):
    """Save DataFrame to specified file format.
    
    Args:
        df:
        output_path:
        file_format:
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if file_format == 'csv':
        df.to_csv(output_path, index=False)
    elif file_format == 'excel':
        df.to_excel(output_path, index=False)
    else:
        raise ValueError("file_format must be 'csv' or 'excel'")
        

def _check_shap_installed():
    """Verify SHAP library installation."""
    try:
        import shap  
    except ImportError:
        raise ImportError(
            "SHAP library is not installed. Please install with: "
            "pip install shap"
        )


def filter_data(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    group_col: str = constants.DEFAULT_GROUP_COLUMN,
    test_size: float = constants.DEFAULT_TEST_SIZE,
    random_state: int = constants.DEFAULT_RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and testing sets while maintaining proportion of classes.
    
    Args:
        X:            Feature matrix.
        y:            Target vector.
        metadata:     Metadata dataframe.
        group_col:    Column name for stratification.
        test_size:    Proportion of data for testing.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    # Input validation
    if not all(X.index == y.index) or not all(X.index == metadata.index):
        raise ValueError("X, y, and metadata must have the same index.")
    
    if group_col not in metadata.columns:
        raise ValueError(f"'{group_col}' not found in metadata.")
    
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    
    # Split data while maintaining class proportions
    stratify = metadata[group_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test
    

def rfe_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_features: int,
    step_size: int,
    threads: int,
    random_state: int,
    iterations: int = constants.DEFAULT_ITERATIONS_RFE,
    learning_rate: float = constants.DEFAULT_LEARNING_RATE_RFE,
    depth: int = constants.DEFAULT_DEPTH_RFE,
    verbose: int = 1,
    catboost_params: Optional[Dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Perform Recursive Feature Elimination (RFE) using CatBoostClassifier.
    
    Args:
        X_train:         Training feature matrix.
        y_train:         Training target vector.
        X_test:          Testing feature matrix.
        y_test:          Testing target vector.
        num_features:    Number of features to select.
        step_size:       Features to remove at each iteration.
        threads:         Number of threads to use.
        random_state:    Random seed.
        iterations:      CatBoost iterations.
        learning_rate:   CatBoost learning rate.
        depth:           CatBoost tree depth.
        verbose:         Verbosity level.
        catboost_params: Additional CatBoost parameters.
        
    Returns:
        Tuple of (X_train_selected, X_test_selected, selected_features).
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)
    
    if num_features <= 0 or step_size <= 0:
        raise ValueError("num_features and step_size must be positive integers.")
    
    if num_features > X_train.shape[1]:
        raise ValueError("num_features cannot be greater than the total number of features.")
    
    if threads <= 0:
        raise ValueError("threads must be a positive integer.")
    
    if iterations <= 0 or depth <= 0:
        raise ValueError("iterations and depth must be positive integers.")
    
    if learning_rate <= 0:
        raise ValueError("learning_rate must be a positive float.")
    
    # Initialize CatBoostClassifier
    catboost_params = catboost_params or {}
    rfe_model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        thread_count=threads,
        random_state=random_state,
        verbose=verbose > 1,
        **catboost_params
    )
    
    # Perform RFE
    rfe = RFE(
        estimator=rfe_model,
        n_features_to_select=num_features,
        step=step_size,
        verbose=verbose
    )
    
    rfe.fit(X_train, y_train)
    
    # Transform datasets
    X_train_selected = pd.DataFrame(
        rfe.transform(X_train), 
        columns=X_train.columns[rfe.support_],
        index=X_train.index
    )
    X_test_selected = pd.DataFrame(
        rfe.transform(X_test),
        columns=X_train.columns[rfe.support_],
        index=X_test.index
    )
    
    # Get selected features
    selected_features = X_train.columns[rfe.support_].tolist()
    
    if verbose:
        logger.debug(f"Selected {len(selected_features)} features using RFE")
    
    return X_train_selected, X_test_selected, selected_features
    

def select_k_best_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_features: int,
    score_func: Callable = f_classif,
    verbose: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Perform feature selection using SelectKBest.
    
    Args:
        X_train:      Training feature matrix.
        y_train:      Training target vector.
        X_test:       Testing feature matrix.
        y_test:       Testing target vector.
        num_features: Number of features to select.
        score_func:   Scoring function for feature selection.
        verbose:      Verbosity level.
        
    Returns:
        Tuple of (X_train_selected, X_test_selected, selected_features).
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)
    
    if num_features <= 0:
        raise ValueError("num_features must be a positive integer.")
    
    if num_features > X_train.shape[1]:
        raise ValueError("num_features cannot be greater than the total number of features.")
    
    # Perform SelectKBest
    skb = SelectKBest(score_func=score_func, k=num_features)
    skb.fit(X_train, y_train)
    
    # Transform datasets
    X_train_selected = pd.DataFrame(
        skb.transform(X_train),
        columns=X_train.columns[skb.get_support()],
        index=X_train.index
    )
    X_test_selected = pd.DataFrame(
        skb.transform(X_test),
        columns=X_train.columns[skb.get_support()],
        index=X_test.index
    )
    
    # Get selected features
    selected_features = X_train.columns[skb.get_support()].tolist()
    
    if verbose:
        logger.debug(f"Selected {num_features} features using {score_func.__name__}")
    
    return X_train_selected, X_test_selected, selected_features


def chi_squared_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_features: int,
    verbose: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Perform feature selection using Chi-Squared test.
    
    Args:
        X_train:      Training feature matrix.
        y_train:      Training target vector.
        X_test:       Testing feature matrix.
        y_test:       Testing target vector.
        num_features: Number of features to select.
        verbose:      Verbosity level.
        
    Returns:
        Tuple of (X_train_selected, X_test_selected, selected_features).
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)
    
    if num_features <= 0:
        raise ValueError("num_features must be a positive integer.")
    
    if num_features > X_train.shape[1]:
        raise ValueError("num_features cannot be greater than the total number of features.")
    
    if (X_train < 0).any().any() or (X_test < 0).any().any():
        raise ValueError("Chi-Squared test requires non-negative feature values.")
    
    # Perform Chi-Squared feature selection
    chi2_selector = SelectKBest(score_func=chi2, k=num_features)
    chi2_selector.fit(X_train, y_train)
    
    # Transform datasets
    X_train_selected = pd.DataFrame(
        chi2_selector.transform(X_train),
        columns=X_train.columns[chi2_selector.get_support()],
        index=X_train.index
    )
    X_test_selected = pd.DataFrame(
        chi2_selector.transform(X_test),
        columns=X_train.columns[chi2_selector.get_support()],
        index=X_test.index
    )
    
    # Get selected features
    selected_features = X_train.columns[chi2_selector.get_support()].tolist()
    
    if verbose:
        logger.debug(f"Selected {num_features} features using Chi-Squared Test")
    
    return X_train_selected, X_test_selected, selected_features
    

def lasso_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_features: int,
    penalty: str = constants.DEFAULT_PENALTY_LASSO,
    solver: str = constants.DEFAULT_SOLVER_LASSO,
    max_iter: int = 1000,
    random_state: int = constants.DEFAULT_RANDOM_STATE,
    verbose: int = 1,
    lasso_params: Optional[Dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Perform feature selection using Lasso (L1-regularized Logistic Regression).
    
    Args:
        X_train:      Training feature matrix.
        y_train:      Training target vector.
        X_test:       Testing feature matrix.
        y_test:       Testing target vector.
        num_features: Number of features to select.
        penalty:      Regularization penalty type.
        solver:       Optimization solver.
        max_iter:     Maximum iterations.
        random_state: Random seed.
        verbose:      Verbosity level.
        lasso_params: Additional Lasso parameters.
        
    Returns:
        Tuple of (X_train_selected, X_test_selected, selected_features).
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)
    
    if num_features <= 0:
        raise ValueError("num_features must be a positive integer.")
    
    if num_features > X_train.shape[1]:
        raise ValueError("num_features cannot be greater than the total number of features.")
    
    if penalty not in ['l1', 'l2']:
        raise ValueError("penalty must be either 'l1' or 'l2'.")
    
    if solver not in ['liblinear', 'saga'] and penalty == 'l1':
        raise ValueError("For L1 penalty, solver must be 'liblinear' or 'saga'.")
    
    # Initialize LogisticRegression
    lasso_params = lasso_params or {}
    lasso = LogisticRegression(
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        **lasso_params
    )
    
    # Fit the model
    lasso.fit(X_train, y_train)
    
    # Perform feature selection
    model = SelectFromModel(
        lasso,
        max_features=num_features,
        prefit=True
    )
    
    # Transform datasets
    X_train_selected = pd.DataFrame(
        model.transform(X_train),
        columns=X_train.columns[model.get_support()],
        index=X_train.index
    )
    X_test_selected = pd.DataFrame(
        model.transform(X_test),
        columns=X_train.columns[model.get_support()],
        index=X_test.index
    )
    
    # Get selected features
    selected_features = X_train.columns[model.get_support()].tolist()
    
    if verbose:
        logger.debug(f"Selected {len(selected_features)} features using Lasso")
    
    return X_train_selected, X_test_selected, selected_features
    

def shap_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_features: int,
    threads: int,
    iterations: int = constants.DEFAULT_ITERATIONS_SHAP,
    learning_rate: float = constants.DEFAULT_LEARNING_RATE_SHAP,
    depth: int = constants.DEFAULT_DEPTH_SHAP,
    verbose: int = 1,
    catboost_params: Optional[Dict] = None,
    sample_size: int = 1000
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Perform feature selection using SHAP values.
    
    Args:
        X_train:         Training feature matrix.
        y_train:         Training target vector.
        X_test:          Testing feature matrix.
        y_test:          Testing target vector.
        num_features:    Number of features to select.
        threads:         Number of threads to use.
        iterations:      CatBoost iterations.
        learning_rate:   CatBoost learning rate.
        depth:           CatBoost tree depth.
        verbose:         Verbosity level.
        catboost_params: Additional CatBoost parameters.
        sample_size:     Maximum samples to use for SHAP calculation.
        
    Returns:
        Tuple of (X_train_selected, X_test_selected, selected_features).
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)
    
    if num_features <= 0:
        raise ValueError("num_features must be a positive integer.")
    
    if num_features > X_train.shape[1]:
        raise ValueError(
            "num_features cannot be greater than the total number of features."
        )
    
    if threads <= 0:
        raise ValueError("threads must be a positive integer.")
    
    if iterations <= 0 or depth <= 0:
        raise ValueError("iterations and depth must be positive integers.")
    
    if learning_rate <= 0:
        raise ValueError("learning_rate must be a positive float.")
    
    # Check if SHAP is installed
    _check_shap_installed()
    
    # Initialize CatBoostClassifier
    catboost_params = catboost_params or {}
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        thread_count=threads,
        verbose=verbose > 1,
        **catboost_params
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Explain model with SHAP values
    if verbose:
        logger.debug("Calculating SHAP values...")
    
    # Subsample if dataset is large
    if len(X_train) > sample_size:
        X_train_sampled = X_train.sample(
            n=min(sample_size, len(X_train)), 
            random_state=constants.DEFAULT_RANDOM_STATE
        )
    else:
        X_train_sampled = X_train
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_sampled)
    
    # Handle multi-class output
    if isinstance(shap_values, list):
        # For multi-class: average absolute SHAP values across classes
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        # For binary classification
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Get top features
    top_indices = np.argsort(mean_abs_shap)[-num_features:]
    selected_features = X_train.columns[top_indices].tolist()
    
    # Select features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    if verbose:
        logger.debug(f"Selected {len(selected_features)} features using SHAP")
    
    return X_train_selected, X_test_selected, selected_features


def perform_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_selection: str = constants.DEFAULT_METHOD,
    use_permutation_importance: bool = constants.DEFAULT_USE_PERMUTATION_IMPORTANCE,
    thread_count: int = constants.DEFAULT_THREAD_COUNT,
    step_size: int = constants.DEFAULT_STEP_SIZE,
    num_features: int = constants.DEFAULT_NUM_FEATURES,
    random_state: int = constants.DEFAULT_RANDOM_STATE,
    verbose: int = 1,
    feature_selection_params: Optional[Dict] = None,
    perm_importance_scorer: Optional[Callable] = None,
    perm_importance_n_repeats: int = 10,
    catboost_params: Optional[Dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Perform feature selection using specified method with optional 
    permutation importance.
    
    Args:
        X_train:                    Training feature matrix.
        y_train:                    Training target vector.
        X_test:                     Testing feature matrix.
        y_test:                     Testing target vector.
        feature_selection:          Feature selection method.
        use_permutation_importance: Apply permutation importance.
        thread_count:               Number of threads.
        step_size:                  Step size for RFE.
        num_features:               Number of features to select.
        random_state:               Random seed.
        verbose:                    Verbosity level.
        feature_selection_params:   Parameters for feature selection method.
        perm_importance_scorer:     Scorer for permutation importance.
        perm_importance_n_repeats:  Repeats for permutation importance.
        catboost_params:            CatBoost parameters.
        
    Returns:
        Tuple of (X_train_selected, X_test_selected, selected_features).
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)
    
    valid_methods = ['rfe', 'select_k_best', 'chi_squared', 'lasso', 'shap']
    if feature_selection not in valid_methods:
        raise ValueError(
            f"feature_selection must be one of {valid_methods}"
        )
    
    if num_features <= 0 or step_size <= 0 or thread_count <= 0:
        raise ValueError(
            "num_features, step_size, and thread_count must be positive integers."
        )
    
    feature_selection_params = feature_selection_params or {}
    catboost_params = catboost_params or {}
    
    # Remove conflicting parameters from feature_selection_params to avoid "multiple values" errors
    # These are already provided as explicit parameters
    fs_params_copy = feature_selection_params.copy()
    fs_params_copy.pop('num_features', None)  # Already provided
    fs_params_copy.pop('threads', None)  # Already provided as thread_count
    fs_params_copy.pop('step_size', None)  # Already provided
    fs_params_copy.pop('random_state', None)  # Already provided
    
    # Perform feature selection
    if feature_selection == 'rfe':
        X_train_selected, X_test_selected, selected_features = rfe_feature_selection(
            X_train, y_train, X_test, y_test,
            num_features=num_features,
            step_size=step_size,
            threads=thread_count,
            random_state=random_state,
            **fs_params_copy
        )
    elif feature_selection == 'select_k_best':
        X_train_selected, X_test_selected, selected_features = select_k_best_feature_selection(
            X_train, y_train, X_test, y_test,
            num_features=num_features,
            **fs_params_copy
        )
    elif feature_selection == 'chi_squared':
        X_train_selected, X_test_selected, selected_features = chi_squared_feature_selection(
            X_train, y_train, X_test, y_test,
            num_features=num_features,
            **fs_params_copy
        )
    elif feature_selection == 'lasso':
        X_train_selected, X_test_selected, selected_features = lasso_feature_selection(
            X_train, y_train, X_test, y_test,
            num_features=num_features,
            **fs_params_copy
        )
    elif feature_selection == 'shap':
        X_train_selected, X_test_selected, selected_features = shap_feature_selection(
            X_train, y_train, X_test, y_test,
            num_features=num_features,
            threads=thread_count,
            **fs_params_copy
        )
    
    # Apply permutation importance if required
    if use_permutation_importance:
        if verbose:
            logger.debug("Computing permutation importances...")

        # Create default scorer if not provided
        if perm_importance_scorer is None:  # ADDED DEFAULT HANDLING
            perm_importance_scorer = make_scorer(matthews_corrcoef)
        if not callable(perm_importance_scorer):
            raise TypeError("perm_importance_scorer must be a callable function")
            
        perm_model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=4,
            verbose=verbose > 1,
            thread_count=thread_count,
            random_state=random_state,
            **catboost_params
        )
        perm_model.fit(
            X_train_selected,
            y_train,
            eval_set=(X_test_selected, y_test),
            verbose=verbose > 1,
            early_stopping_rounds=100
        )
        
        # Compute permutation importance
        perm_importance = permutation_importance(
            perm_model,
            X_test_selected.values,
            y_test,
            scoring=perm_importance_scorer,
            n_repeats=perm_importance_n_repeats,
            random_state=random_state,
            n_jobs=thread_count
        )
        
        importances = pd.Series(
            perm_importance.importances_mean, index=selected_features
        )
        # Select features with positive importance
        selected_features = importances[importances > 0].index.tolist()
        X_train_selected = X_train_selected[selected_features]
        X_test_selected = X_test_selected[selected_features]
    
    if verbose:
        logger.debug(f"Selected {len(selected_features)} features")
    
    return X_train_selected, X_test_selected, selected_features
    

def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict,
    output_dir: Union[str, Path],
    plot: bool = False,
    verbose: int = 1,
    early_stopping_rounds: int = 100,
    custom_metrics: Optional[Dict[str, Callable]] = None
) -> Tuple[CatBoostClassifier, float, float, float, pd.Series]:
    """Train and evaluate CatBoostClassifier.
    
    Args:
        X_train:               Training feature matrix.
        y_train:               Training target vector.
        X_test:                Testing feature matrix.
        y_test:                Testing target vector.
        params:                CatBoost parameters.
        output_dir:            Output directory.
        plot:                  Show training plots.
        verbose:               Verbosity level.
        early_stopping_rounds: Early stopping rounds.
        custom_metrics:        Additional evaluation metrics.
        
    Returns:
        Tuple of (model, accuracy, f1_score, mcc, predictions).
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)
    
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary.")
    
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CatBoostClassifier
    train_dir = str(output_dir / 'catboost_info')
    model = CatBoostClassifier(
        **params,
        train_dir=train_dir
    )
    
    if verbose:
        logger.debug(f"Training with parameters: {params}")
    
    # Train the model
    model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        plot=plot,
        verbose=verbose > 1,
        early_stopping_rounds=early_stopping_rounds
    )
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    if custom_metrics:
        for metric_name, metric_func in custom_metrics.items():
            metric_value = metric_func(y_test, y_pred)
            logger.debug(f"{metric_name}: {metric_value}")
    
    if verbose:
        logger.debug(f"Model evaluation: Accuracy={accuracy:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")
    
    # Save the model
    model.save_model(str(output_dir / 'model.cbm'))
    
    return model, accuracy, f1, mcc, y_pred
    

def grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param_grid: Dict[str, List] = constants.DEFAULT_PARAM_GRID,
    output_dir: Union[str, Path] = None,
    n_splits: int = 5,
    refit: str = 'mcc',
    progress: Any = None, 
    task_id: Any = None,
    verbose: int = 1,
    fixed_params: Optional[Dict] = None
) -> Tuple[CatBoostClassifier, Dict, float, Dict]:
    """Enhanced grid search with cross-validation and comprehensive evaluation.
    
    Args:
        X_train:      Training feature matrix.
        y_train:      Training target vector.
        X_test:       Testing feature matrix.
        y_test:       Testing target vector.
        param_grid:   Parameter grid for search.
        output_dir:   Output directory.
        n_splits:     Cross-validation splits.
        refit:        Metric to optimize.
        verbose:      Verbosity level.
        fixed_params: Fixed model parameters.
        progress:     Rich Progress instance for progress bar.
        task_id:      Parent task ID for nested progress bars.
        
    Returns:
        Tuple of (best_model, best_params, best_score, test_scores).
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results storage
    results = []
    best_score = -np.inf
    best_model = None
    best_params = None
    cv_model = None
    
    # Create parameter combinations
    param_combinations = list(itertools.product(*param_grid.values()))
    total_combinations = len(param_combinations)
    total_folds = total_combinations * n_splits
    
    # Setup progress bar - create a parent task if not provided
    if progress is not None and task_id is None:
        task_id = progress.add_task(description="Grid Search...", total=total_folds)
        
    child_desc = f"Grid Search..."
    child_task = progress.add_task(
        _format_task_desc(child_desc),
        parent=task_id,
        total=total_folds
    )
    logger.debug(
        f"Starting grid search with {total_combinations} parameter combinations\n"
        f"Using {n_splits}-fold cross-validation"
    )
    
    # Cross-validation setup
    cv = StratifiedKFold(
        n_splits=n_splits, 
        shuffle=True, 
        random_state=constants.DEFAULT_RANDOM_STATE
    )
    
    # Main grid search loop
    for i, params in enumerate(param_combinations, 1):
        # Combine grid parameters with fixed parameters
        current_params = dict(zip(param_grid.keys(), params))
        if fixed_params:
            current_params.update(fixed_params)
        
        # Initialize fold scores
        fold_scores = {
            'accuracy': [],
            'f1': [],
            'mcc': [],
            'roc_auc': [],
            'pr_auc': []
        }
        
        logger.debug(
            f"\n[{i}/{total_combinations}] Testing params: {current_params}"
        )
        
        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Initialize and train model
            model = CatBoostClassifier(
                **current_params,
                train_dir=str(output_dir / 'catboost_info'),
                verbose=False
            )
            
            model.fit(
                X_fold_train,
                y_fold_train,
                eval_set=(X_fold_val, y_fold_val),
                early_stopping_rounds=100,
                verbose=verbose > 1
            )
            
            # Evaluate on validation fold
            y_pred = model.predict(X_fold_val)
            y_proba = model.predict_proba(X_fold_val)[:, 1]  # Probability for positive class
            
            # Compute metrics
            scores = {
                'accuracy': accuracy_score(y_fold_val, y_pred),
                'f1': f1_score(y_fold_val, y_pred),
                'mcc': matthews_corrcoef(y_fold_val, y_pred),
                'roc_auc': roc_auc_score(y_fold_val, y_proba),
                'pr_auc': average_precision_score(y_fold_val, y_proba)
            }
            
            for metric_name, score_val in scores.items():
                fold_scores[metric_name].append(score_val)
            
            del model  # Clean up
            
            # Update progress bar after each fold
            if progress is not None:
                progress.update(
                    child_task,
                    advance=1,
                    description=f"Param {i}/{total_combinations} - Fold {fold}/{n_splits}"
                )
        progress.update(task_id, advance=1)
        # Calculate mean CV scores
        cv_means = {f"mean_{k}": np.mean(v) for k, v in fold_scores.items()}
        cv_stds = {f"std_{k}": np.std(v) for k, v in fold_scores.items()}
        current_result = {**current_params, **cv_means, **cv_stds}
        results.append(current_result)
        
        # Check if best model
        current_ref_score = cv_means[f"mean_{refit}"]
        if current_ref_score > best_score:
            best_score = current_ref_score
            best_params = current_params
            if verbose:
                logger.debug(
                    f"[!] New best {refit} (CV): {current_ref_score:.4f}"
                )
            
            # Train on full training set with best params
            cv_model = CatBoostClassifier(
                **best_params,
                train_dir=str(output_dir / 'catboost_info'),
                verbose=False
            )
            cv_model.fit(
                X_train,
                y_train,
                eval_set=(X_test, y_test),
                early_stopping_rounds=100,
                verbose=verbose > 1
            )
    
    # Final evaluation on test set
    test_scores = {}
    if cv_model:
        y_pred = cv_model.predict(X_test)
        y_proba = cv_model.predict_proba(X_test)[:, 1]
        
        test_scores = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'pr_auc': average_precision_score(y_test, y_proba)
        }
        
        # Save final model
        best_model = cv_model
        best_model.save_model(str(output_dir / "best_model.cbm"))
        
        # Generate evaluation plots
        cm_fig = plot_confusion_matrix(
            confusion_matrix(y_test, y_pred),
            str(output_dir / "best_confusion_matrix.png")
        )
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_fig = plot_roc_curve(
            fpr, 
            tpr, 
            roc_auc_score(y_test, y_proba),
            str(output_dir / "best_roc_curve.png")
        )
        
        prc_fig = plot_precision_recall_curve(
            *precision_recall_curve(y_test, y_proba)[:2],
            average_precision_score(y_test, y_proba),
            str(output_dir / "best_precision_recall_curve.png")
        )

        """
        eval_fig = combine_figures_as_subplots(
            figures=[roc_fig, prc_fig, cm_fig],
            figures_per_row=3,
            show=False,
            output_path=str(output_dir / "best_model_eval.png"),
            verbose=False
        )
        """
        
        # Add test scores to results
        for result in results:
            if result == best_params:  # Match by parameter set
                result.update({
                    f"test_{k}": v for k, v in test_scores.items()
                })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "grid_search_results.csv", index=False)
    
    logger.debug(
        f"Grid search completed\n"
        f"Best parameters: {best_params}\n"
        f"Best CV {refit}: {best_score:.4f}"
    )
    
    if test_scores:
        logger.debug("Test set performance:")
        for metric, score in test_scores.items():
            logger.debug(f"{metric}: {score:.4f}")
    
    return best_model, best_params, best_score, test_scores, [roc_fig, prc_fig, cm_fig]


def save_feature_importances(
    model: CatBoostClassifier, 
    X_train: pd.DataFrame, 
    output_dir: Union[str, Path]
) -> None:
    """Save feature importances from trained model.
    
    Args:
        model:      Trained model.
        X_train:    Training feature matrix.
        output_dir: Output directory.
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.zeros(X_train.shape[1])
    
    feat_imp = pd.Series(
        importances, 
        index=X_train.columns, 
        name='importance'
    ).sort_values(ascending=False)
    feat_imp.to_csv(output_dir / "feature_importances.csv")


def catboost_feature_selection(
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    output_dir: Union[str, Path],
    group_col: str,
    method: str = 'rfe',
    n_top_features: int = 100,
    filter_col: Optional[str] = None,
    filter_val: Optional[str] = None,
    progress: Any = None, 
    task_id: Any = None,
    verbose: bool = False,
    param_grid: dict = constants.DEFAULT_PARAM_GRID,
    **kwargs
) -> Dict:
    """Perform feature selection using CatBoost and save results.
    
    Args:
        metadata:       Sample metadata.
        features:       Feature matrix.
        output_dir:     Output directory.
        group_col:      Contamination status column.
        method:         Feature selection method.
        n_top_features: Number of top features to return.
        filter_col:     Column to filter on.
        filter_val:     Value to filter for.
        
    Returns:
        Dictionary with results
    """
    output_dir = Path(output_dir) / method
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply filtering if requested
    if filter_col and filter_val:
        metadata = metadata[metadata[filter_col].str.contains(
            filter_val, 
            case=False, 
            na=False
        )]
        output_dir = output_dir / f"{filter_col}-{filter_val}"
        os.makedirs(output_dir, exist_ok=True)
    
    # Filter data
    X, y = features, metadata[group_col]
    X_train, X_test, y_train, y_test = filter_data(
        X, y, metadata, group_col
    )
    
    # Determine number of features to select
    num_features = min(500, X_train.shape[1])
        
    # Perform feature selection
    X_train_selected, X_test_selected, final_selected_features = perform_feature_selection(
        X_train,
        y_train,
        X_test,
        y_test,
        feature_selection=method,
        num_features=num_features,
        **kwargs
    )
    
    if X_train_selected.empty:
        raise ValueError(
            "Feature selection returned no features. Check input data."
        )
    if verbose:
        logger.debug(
            f"Feature variance after selection: {X_train_selected.var().describe()}"
        ) 
    
    # Comprehensive parameter grid with fixed parameters
    fixed_params = {
        'loss_function': constants.DEFAULT_LOSS_FUNCTION,
        'thread_count': constants.DEFAULT_THREAD_COUNT,
        'random_state': constants.DEFAULT_RANDOM_STATE
    }
    
    # Run grid search
    best_model, best_params, best_score, test_scores, eval_fig = grid_search(
        X_train=X_train_selected, 
        y_train=y_train, 
        X_test=X_test_selected, 
        y_test=y_test, 
        param_grid=param_grid, 
        output_dir=output_dir,
        progress=progress, 
        task_id=task_id,
        fixed_params=fixed_params
    )
    
    # Save feature importances
    selected = pd.DataFrame(
        X_train_selected, 
        columns=final_selected_features
    )
    save_feature_importances(
        best_model, 
        selected, 
        output_dir
    )
    
    # Get feature importances
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    else:
        importances = np.zeros(len(final_selected_features))
    
    feat_imp = pd.Series(
        importances, 
        index=final_selected_features, 
        name='importance'
    ).sort_values(ascending=False)
    
    # Get top N features
    top_features = feat_imp.head(n_top_features).index.tolist()
    
    try:
        _check_shap_installed()
        explainer = shap.TreeExplainer(best_model)
        
        # Use subset for large datasets
        if len(X_train_selected) > 1000:
            X_sample = X_train_selected.sample(
                n=1000, 
                random_state=constants.DEFAULT_RANDOM_STATE
            )
        else:
            X_sample = X_train_selected
        
        # NOTE: Assumes binary classification
        shap_vals = explainer.shap_values(X_sample)
        base_value = explainer.expected_value
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]  # Use class 1 (positive)
        
        # Convert to numpy arrays
        feature_vals = X_sample.values
        feature_names = X_sample.columns.tolist()
        
        # Generate SHAP figures
        shap_figs = plot_shap(
            base_value,
            shap_vals, 
            feature_vals, 
            feature_names, 
            n_features=n_top_features, 
            interaction_feature='auto',
            output_dir=output_dir, 
            show=False,
            verbose=verbose
        )

        shap_report, shap_report_df = generate_shap_report(model=best_model, X=X_sample, K=20)
        logger.info(shap_report)
        
    except Exception as e:
        logger.error(f"SHAP plot generation failed: {e}")
        shap_figs = {}
    
    # Return comprehensive results
    logger.info(best_model.classes_)
    return {
        'method': method,
        'model': best_model,
        'feature_importances': feat_imp.to_dict(),
        'top_features': top_features,
        'best_params': best_params,
        'test_scores': test_scores,
        'shap_report': shap_report_df,
        'figures': {
            #'eval_plots': eval_fig,
            'roc': eval_fig[0],
            'prc': eval_fig[1],
            'confusion_matrix': eval_fig[2],
            'shap_summary_bar': shap_figs['bar_fig'], 
            'shap_summary_beeswarm': shap_figs['beeswarm_fig'], 
            'shap_summary_heatmap': shap_figs['heatmap_fig'],
            'shap_summary_force': shap_figs['force_fig'],
            'shap_dependency': shap_figs['dependency_figs']
        }
    }
