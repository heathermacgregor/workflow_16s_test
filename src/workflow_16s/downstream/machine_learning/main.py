# ==================================================================================== #
#                       downstream/machine_learning.py
# ==================================================================================== #

import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

import anndata as ad
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scanpy as sc
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, mean_squared_error, r2_score, 
    matthews_corrcoef, balanced_accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr, chi2_contingency
from scipy.stats.contingency import association

from workflow_16s.downstream.utils import AnalysisUtils
from workflow_16s.downstream.visualization import PlottingUtils, DEFAULT_HEIGHT
from workflow_16s.downstream.machine_learning.feature_selection import (
    catboost_feature_selection, filter_data
)
from workflow_16s.downstream.machine_learning.overfitting_prevention import run_comprehensive_validation
from workflow_16s.downstream.machine_learning.batch_control import (
    run_ml_with_batch_control, create_comparison_plots
)
from workflow_16s.downstream.machine_learning.visualization import generate_comprehensive_ml_report
from workflow_16s.utils.logger import get_logger
from catboost import CatBoostClassifier, CatBoostRegressor
from workflow_16s.config_schema import MLConfig
# ==================================================================================== #

logger = get_logger("workflow_16s")
plot_utils = PlottingUtils(logger)

EXPECTED_VAR_COLUMNS = {'Taxon', 'Confidence', 'sequence'}
EXPECTED_VAR_DTYPES = {'Taxon': 'string', 'Confidence': 'Float64', 'sequence': 'string'}
FACILITY_SHAPE_COLS = {'facility_capacity', 'facility_start_year', 'facility_end_year', 'facility_type', 'facility'}

sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white', frameon=False)
ad.settings.allow_write_nullable_strings = True

def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Removes leading/trailing whitespace from feature names."""
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    return df

# ==================================================================================== #
# BATCH COVARIATE HELPER FUNCTIONS
# ==================================================================================== #

def prepare_batch_covariates(
        logger.info(f"Available metadata columns in adata.obs: {list(adata.obs.columns)}")
    adata: ad.AnnData,
    batch_columns: List[str],
    sample_indices: pd.Index,
    one_hot_encode: bool = True,
    max_categories: int = 20
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Prepare batch/technical covariates for inclusion in ML models.
    """
    batch_df_list = []
    metadata = {'dropped': [], 'encoded': {}, 'numeric': [], 'categorical': []}
    
    for col in batch_columns:
        # Fallback: if 'batch_original' is missing, use 'study_accession' if present
        actual_col = col
        if col == 'batch_original' and col not in adata.obs.columns and 'study_accession' in adata.obs.columns:
            logger.info("'batch_original' not found, falling back to 'study_accession' as batch column.")
            actual_col = 'study_accession'
        if actual_col not in adata.obs.columns:
            logger.warning(f"Batch column '{actual_col}' not found in metadata, skipping")
            metadata['dropped'].append({'column': actual_col, 'reason': 'not_found'})
            continue
        col_data = adata.obs.loc[sample_indices, actual_col].copy()
        
        col_data = adata.obs.loc[sample_indices, col].copy()
        
        # Check for too many missing values
        missing_pct = col_data.isna().sum() / len(col_data)
        if missing_pct > 0.5:
            logger.warning(f"Batch column '{col}' has {missing_pct:.1%} missing values, skipping")
            metadata['dropped'].append({'column': col, 'reason': 'too_many_missing', 'missing_pct': missing_pct})
            continue
        
        # Detect column type
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        
        if is_numeric:
            # Numeric covariate - fill missing and normalize
            col_data_clean = col_data.fillna(col_data.median())
            batch_df_list.append(pd.DataFrame({col: col_data_clean}, index=sample_indices))
            metadata['numeric'].append(col)
        else:
            # Categorical covariate
            n_categories = col_data.nunique()
            if n_categories > max_categories:
                logger.warning(f"Batch column '{col}' has {n_categories} categories (max {max_categories}), skipping")
                metadata['dropped'].append({'column': col, 'reason': 'too_many_categories', 'n_categories': n_categories})
                continue
            
            # Fill missing with 'Unknown'
            col_data_clean = col_data.fillna('Unknown').astype(str)
            
            if one_hot_encode:
                # One-hot encode
                dummies = pd.get_dummies(col_data_clean, prefix=col, drop_first=True)
                batch_df_list.append(dummies)
                metadata['encoded'][col] = {'method': 'one_hot', 'n_categories': n_categories, 'n_features': len(dummies.columns)}
                metadata['categorical'].append(col)
            else:
                # Label encode
                le = LabelEncoder()
                encoded = le.fit_transform(col_data_clean)
                batch_df_list.append(pd.DataFrame({col: encoded}, index=sample_indices))
                metadata['encoded'][col] = {'method': 'label', 'n_categories': n_categories, 'classes': le.classes_.tolist()}
                metadata['categorical'].append(col)
    
    if not batch_df_list:
        logger.warning("No valid batch covariates prepared")
        return pd.DataFrame(index=sample_indices), metadata
    
    batch_df = pd.concat(batch_df_list, axis=1)
    logger.info(f"Prepared {len(batch_df.columns)} batch covariate features from {len(metadata['numeric']) + len(metadata['categorical'])} columns")
    
    return batch_df, metadata


def detect_confounding(
    adata: ad.AnnData,
    batch_columns: List[str],
    target_col: str,
    sample_indices: pd.Index,
    threshold: float = 0.7,
    plot_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Detect confounding between batch variables and target variable.
    """
    confounding_results = {
        'target': target_col,
        'high_confounding': [],
        'moderate_confounding': [],
        'low_confounding': [],
        'statistics': {}
    }
    
    target_data = adata.obs.loc[sample_indices, target_col].dropna()
    target_is_numeric = pd.api.types.is_numeric_dtype(target_data)
    
    for batch_col in batch_columns:
        if batch_col not in adata.obs.columns:
            continue
        
        batch_data = adata.obs.loc[sample_indices, batch_col]
        common_idx = target_data.index.intersection(batch_data.dropna().index)
        
        if len(common_idx) < 10:
            continue
        
        target_common = target_data.loc[common_idx]
        batch_common = batch_data.loc[common_idx]
        batch_is_numeric = pd.api.types.is_numeric_dtype(batch_common)
        
        # Calculate appropriate correlation/association measure
        association_value = None
        association_type = None
        pval = None
        
        try:
            if target_is_numeric and batch_is_numeric:
                # Numeric-numeric: Spearman correlation
                corr, pval = spearmanr(target_common, batch_common, nan_policy='omit')
                association_value = abs(corr)
                association_type = 'spearman'
            elif not target_is_numeric and not batch_is_numeric:
                # Categorical-categorical: Cramér's V
                contingency = pd.crosstab(target_common, batch_common)
                chi2, pval, dof, expected = chi2_contingency(contingency)
                n = contingency.sum().sum()
                cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
                association_value = cramers_v
                association_type = 'cramers_v'
            else:
                # Mixed: Eta-squared (ANOVA effect size)
                if target_is_numeric:
                    numeric_var = target_common
                    cat_var = batch_common.astype(str)
                else:
                    numeric_var = batch_common
                    cat_var = target_common.astype(str)
                
                groups = [numeric_var[cat_var == cat].values for cat in cat_var.unique()]
                groups = [g for g in groups if len(g) > 0]
                
                if len(groups) < 2:
                    continue
                
                # Calculate eta-squared
                grand_mean = numeric_var.mean()
                ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
                ss_total = sum((numeric_var - grand_mean)**2)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                association_value = eta_squared
                association_type = 'eta_squared'
                pval = None  # Not easily calculable
        
        except Exception as e:
            logger.debug(f"Could not calculate association between {batch_col} and {target_col}: {e}")
            continue
        
        if association_value is not None:
            confounding_results['statistics'][batch_col] = {
                'value': association_value,
                'type': association_type,
                'p_value': pval
            }
            
            # Categorize confounding level
            if association_value >= threshold:
                confounding_results['high_confounding'].append(batch_col)
                logger.warning(
                    f"⚠️  HIGH CONFOUNDING: '{batch_col}' strongly associated with '{target_col}' "
                    f"({association_type}={association_value:.3f}). Results may be unreliable."
                )
            elif association_value >= threshold * 0.7:
                confounding_results['moderate_confounding'].append(batch_col)
                logger.warning(
                    f"⚠️  MODERATE CONFOUNDING: '{batch_col}' moderately associated with '{target_col}' "
                    f"({association_type}={association_value:.3f}). Interpret with caution."
                )
            else:
                confounding_results['low_confounding'].append(batch_col)
    
    # Generate confounding plot if requested
    if plot_dir and confounding_results['statistics']:
        try:
            plot_confounding_heatmap(confounding_results, plot_dir, target_col)
        except Exception as e:
            logger.warning(f"Could not generate confounding plot: {e}")
    
    return confounding_results


def plot_confounding_heatmap(confounding_results: Dict[str, Any], plot_dir: Path, target_col: str):
    """Create heatmap visualization of batch-target confounding."""
    stats = confounding_results['statistics']
    if not stats:
        return
    
    batch_cols = list(stats.keys())
    values = [stats[col]['value'] for col in batch_cols]
    types = [stats[col]['type'] for col in batch_cols]
    
    # Color-code by confounding level
    colors = []
    for val in values:
        if val >= 0.7:
            colors.append('red')
        elif val >= 0.49:
            colors.append('orange')
        else:
            colors.append('green')
    
    fig = go.Figure(data=go.Bar(
        x=batch_cols,
        y=values,
        marker=dict(color=colors),
        text=[f"{v:.3f} ({t})" for v, t in zip(values, types)],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Batch Confounding with Target: {target_col}",
        xaxis_title="Batch Variables",
        yaxis_title="Association Strength",
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False
    )
    
    fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                  annotation_text="High Confounding")
    fig.add_hline(y=0.49, line_dash="dash", line_color="orange", 
                  annotation_text="Moderate Confounding")
    
    safe_target = re.sub(r'[^A-Za-z0-9_]+', '', target_col)
    plot_path = plot_dir / f"confounding_{safe_target}.html"
    plot_utils.save_plotly_fig(fig, plot_path)


def train_batch_residual_model(
    X_taxa: pd.DataFrame,
    batch_covariates: pd.DataFrame,
    y: pd.Series,
    test_indices: pd.Index,
    task_type: str
) -> Tuple[Any, pd.Series]:
    """
    Train model to predict target from batch covariates only,
    then return residuals for subsequent biological prediction.
    
    This implements the stratified/two-stage approach.
    """
    # Split batch covariates
    train_indices = X_taxa.index.difference(test_indices)
    batch_train = batch_covariates.loc[train_indices]
    batch_test = batch_covariates.loc[test_indices]
    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]
    
    # Train batch-only model
    if task_type == 'regression':
        batch_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    else:
        batch_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    
    batch_model.fit(batch_train.fillna(0), y_train)
    
    # Get predictions for all samples
    y_pred_batch = batch_model.predict(batch_covariates.fillna(0))
    
    # Calculate residuals
    if task_type == 'regression':
        residuals = y - y_pred_batch
    else:
        # For classification, use probability-based residuals
        # This is more complex - use indicator encoding
        residuals = (y != y_pred_batch).astype(float)
    
    return batch_model, residuals


# ==================================================================================== #

def plot_feature_importances(importance_df: pd.DataFrame, plot_dir_ml: Path, target_name: str, level: str, model_score: float, score_name: str):
    """Plots feature importances from ML model."""
    title = f"Top Predictive Taxa ({level}) for {target_name}<br>Model OOB {score_name}: {model_score:.3f}"
    try:
        fig = px.bar(importance_df.sort_values(by='Importance', ascending=True), x='Importance', y='Taxon', title=title, orientation='h')
        fig.update_layout(yaxis_title=f"Taxon ({level})", xaxis_title="Feature Importance (Gini)", height=max(400, len(importance_df) * 25))
        safe_target_name = re.sub(r'[^A-Za-z0-9_]+', '', target_name)
        plot_path = plot_dir_ml / f"feature_importance_{safe_target_name}_{level}.html"; plot_utils.save_plotly_fig(fig, plot_path)
    except Exception as e: logger.error(f"Failed feature importance plot for {target_name}: {e}")


def run_machine_learning_analysis(
    adata: ad.AnnData, 
    plot_dir_ml: Path, 
    level: str = 'Genus', 
    min_samples_per_group: int = 10, 
    max_classes: int = 10, 
    priority_targets: Optional[List[str]] = None, 
    strict_targets: bool = False, 
    validate_overfitting: bool = True, 
    quick_validation: bool = False,
    batch_config: Optional[Dict[str, Any]] = None,
    ml_config: Optional[MLConfig] = None,   # <--- NEW: Full Config Object
    X_custom: Optional[pd.DataFrame] = None # <--- NEW: Pre-transformed Data
):
    """
    Runs ML analysis with support for:
    1. Multiple Algorithms (RandomForest, CatBoost)
    2. Data Transformations (CLR, Binary, etc via X_custom)
    3. Batch Correction Strategies (Baseline, Adjusted, Stratified)
    4. Strict Target Scoping
    """
    logger.info(f"--- Starting Machine Learning Analysis ({level}) ---")
    
    # 1. Setup Data (Use X_custom if provided, else default CLR)
    if X_custom is not None:
        X_df = X_custom.copy()
        # Align indices with metadata
        common_samples = X_df.index.intersection(adata.obs.index)
        X_df = X_df.loc[common_samples]
        X_df = clean_feature_names(X_df)
    else:
        # Default Legacy Behavior
        adata_agg = AnalysisUtils.get_analysis_adata(adata, level=level)
        if adata_agg is None or adata_agg.n_obs < 20 or adata_agg.n_vars < 10: 
            logger.warning(f"Skipping ML: Insufficient data at {level}"); return
        X_df = AnalysisUtils.clr_transform(adata_agg, pseudocount=1)
        X_df = clean_feature_names(X_df)

    # 2. Target Selection Logic
    valid_targets = []
    if strict_targets:
        if priority_targets:
            logger.info(f"*** STRICT TARGETS ENABLED ***")
            valid_targets = [t for t in priority_targets if t in adata.obs.columns]
            if not valid_targets: 
                logger.warning("Strict mode: No matching targets found.")
                return
        else:
            logger.warning("Strict mode: No priority targets provided.")
            return
    else:
        # Discovery Mode: Use priority if exists, else scan obs
        candidate_cols = priority_targets if priority_targets else adata.obs.columns
        valid_targets = [t for t in candidate_cols 
                         if t in adata.obs.columns and t not in EXPECTED_VAR_COLUMNS]

    # 3. Determine Algorithms to Run
    algorithms_to_run = []
    if ml_config and ml_config.models:
        if ml_config.models.enable_random_forest: algorithms_to_run.append('rf')
        if ml_config.models.enable_catboost: algorithms_to_run.append('catboost')
    else:
        algorithms_to_run = ['rf'] # Default
    
    if not algorithms_to_run:
        logger.warning("No ML algorithms enabled in config.")
        return

    # 4. Prepare Batch Covariates (if enabled)
    batch_enabled = batch_config and batch_config.get('enabled', False)
    batch_covariates_df = None
    if batch_enabled and batch_config.get('covariate_columns'):
        try:
            # Use X_df.index to ensure alignment
            batch_covariates_df, _ = prepare_batch_covariates(
                adata=adata,
                batch_columns=batch_config['covariate_columns'],
                sample_indices=X_df.index,
                one_hot_encode=batch_config.get('covariate_adjustment', {}).get('one_hot_encode', True),
                max_categories=batch_config.get('covariate_adjustment', {}).get('max_categories', 20)
            )
        except Exception as e:
            logger.error(f"Failed to prepare batch covariates: {e}")
            batch_enabled = False

    # 5. Main Analysis Loop
    for target_col in valid_targets:
        logger.info(f"\n{'='*60}\nTARGET: {target_col}\n{'='*60}")
        
        y_series = adata.obs[target_col]
        common_idx = X_df.index.intersection(y_series.dropna().index)
        
        if len(common_idx) < 20: 
            logger.warning(f"Skipping {target_col}: < 20 samples"); continue
        
        X = X_df.loc[common_idx]
        y = y_series.loc[common_idx]
        
        # Detect Task Type
        is_numeric = pd.api.types.is_numeric_dtype(y)
        n_unique = y.nunique()
        
        # Granular Task Config Check
        model_settings = ml_config.models if ml_config else None
        
        if is_numeric and n_unique > max_classes: 
            if model_settings and not model_settings.enable_regression:
                logger.info(f"Skipping {target_col}: Regression disabled."); continue
            task_type = 'regression'
        elif not is_numeric or n_unique <= max_classes:
            if model_settings and not model_settings.enable_classification:
                logger.info(f"Skipping {target_col}: Classification disabled."); continue
            task_type = 'classification'
            # Check class balance
            if y.value_counts().min() < min_samples_per_group: 
                logger.warning(f"Skipping {target_col}: Rare classes detected."); continue
        else: 
            continue

        # Prepare Validation Config
        val_settings = ml_config.validation if ml_config else None
        test_size = val_settings.test_size if val_settings else 0.3
        
        # Detect Confounding (Once per target)
        confounding_info = {}
        if batch_enabled and batch_config.get('confounding_detection', {}).get('enabled', False):
            confounding_info = detect_confounding(
                adata=adata, batch_columns=batch_config['covariate_columns'],
                target_col=target_col, sample_indices=common_idx,
                threshold=batch_config.get('confounding_detection', {}).get('correlation_threshold', 0.7),
                plot_dir=plot_dir_ml if batch_config.get('confounding_detection', {}).get('plot_confounding', False) else None
            )

        # --- ALGORITHM LOOP ---
        for algo in algorithms_to_run:
            logger.info(f"👉 Running {algo.upper()} on {target_col} ({task_type})")
            
            # Prepare Model Params from Config
            model_params = {}
            if model_settings:
                model_params['n_estimators'] = model_settings.n_estimators
                model_params['max_depth'] = model_settings.max_depth
                if algo == 'catboost':
                    model_params['iterations'] = getattr(model_settings, 'catboost_iterations', 500)
                    model_params['learning_rate'] = getattr(model_settings, 'catboost_learning_rate', 0.03)

            # Path A: Batch Control Enabled (Deep Analysis)
            if batch_enabled and batch_covariates_df is not None:
                try:
                    run_ml_with_batch_control(
                        X_taxa=X,
                        y=y,
                        batch_covariates=batch_covariates_df.loc[common_idx],
                        target_col=target_col,
                        task_type=task_type,
                        plot_dir=plot_dir_ml,
                        level=level,
                        confounding_info=confounding_info,
                        batch_config=batch_config,
                        model_algorithm=algo,  # <--- Pass algo choice
                        model_params=model_params
                    )
                except Exception as e:
                    logger.error(f"Batch ML failed for {target_col} [{algo}]: {e}")

            # Path B: Standard/Baseline Analysis (No Batch Control)
            else:
                try:
                    # Instantiate correct model class
                    if algo == 'catboost':
                        ModelClass = CatBoostRegressor if task_type == 'regression' else CatBoostClassifier
                        # Clean params for CatBoost
                        cb_params = {k: v for k, v in model_params.items() if k in ['iterations', 'learning_rate', 'depth']}
                        if 'max_depth' in model_params: cb_params['depth'] = model_params['max_depth']
                        model = ModelClass(verbose=False, allow_writing_files=False, **cb_params)
                    else:
                        ModelClass = RandomForestRegressor if task_type == 'regression' else RandomForestClassifier
                        rf_params = {k: v for k, v in model_params.items() if k in ['n_estimators', 'max_depth']}
                        model = ModelClass(n_jobs=-1, oob_score=True, **rf_params)

                    # Split
                    stratify = y if task_type == 'classification' else None
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
                    
                    # Fit & Score
                    model.fit(X_tr, y_tr)
                    score = model.score(X_te, y_te) # R2 or Accuracy
                    
                    metric_name = "R²" if task_type == 'regression' else "Accuracy"
                    logger.info(f"   ✓ {algo.upper()} {metric_name}: {score:.3f}")
                    
                    # Overfitting Validation (Standard Path Only)
                    if validate_overfitting:
                        overfit_settings = ml_config.overfitting_prevention if ml_config else None
                        run_perm = overfit_settings.permutation_test if overfit_settings else True
                        
                        run_comprehensive_validation(
                            model=model, X=X, y=y, output_dir=plot_dir_ml / algo,
                            target_name=target_col, task_type=task_type.capitalize(),
                            n_permutations=50 if quick_validation else 100,
                            quick_mode=(not run_perm) # Skip expensive tests if config says False
                        )
                        
                except Exception as e:
                    logger.error(f"Standard ML failed for {target_col} [{algo}]: {e}")


def run_catboost_selection(
    adata: ad.AnnData, 
    catboost_output_dir: Path, 
    level: str = 'Genus', 
    priority_targets: Optional[List[str]] = None, 
    strict_targets: bool = False,
    strategies: Optional[List[str]] = None,  # <--- NEW: Strategy Toggle
    test_size: float = 0.3, 
    random_state: int = 42, 
    num_features: int = 50, 
    n_top_final: int = 20, 
    method: str = 'shap', 
    use_permutation: bool = False, 
    n_cpus: int = 4,
    # drop_batch and use_group_kfold removed from signature as they are determined by strategy
    batch_col: str = 'batch_original',
    run_context: str = None,
    X_custom: Optional[pd.DataFrame] = None, # <--- NEW ARGUMENT
):
    """Runs CatBoost feature selection for specified targets with batch control."""
    logger.info(f"--- Starting CatBoost Feature Selection ({level}, method={method}) ---")
    
    # 1. Setup Data
    adata_agg = AnalysisUtils.get_analysis_adata(adata, level=level)
    if adata_agg is None or adata_agg.n_obs < 20 or adata_agg.n_vars < 2: 
        logger.warning(f"Skipping CatBoost FS: Not enough samples/features at {level}."); return
        
    # [CHANGE] Use X_custom if provided, otherwise default to CLR (Legacy support)
    if X_custom is not None:
        X_df = X_custom.copy()
    else:
        # Default behavior for backward compatibility
        adata_agg = AnalysisUtils.get_analysis_adata(adata, level=level)
        X_df = AnalysisUtils.clr_transform(adata_agg, pseudocount=1)
        
    X_df = clean_feature_names(X_df)
    
    # 2. Target Selection Logic
    valid_targets = []
    if strict_targets:
        if priority_targets:
            logger.info(f"*** STRICT TARGETS ENABLED ***")
            valid_targets = [t for t in priority_targets if t in adata.obs.columns]
            if not valid_targets: logger.warning("Strict mode: No matching targets found."); return
        else:
            logger.warning("Strict mode: No priority targets provided."); return
    else:
        candidate_cols = priority_targets if priority_targets else adata.obs.columns
        valid_targets = [t for t in candidate_cols if t in adata.obs.columns]

    logger.info(f"Targets to process: {valid_targets}")

    # 3. Strategy Selection Logic
    available_strategies = {
        'baseline': '',                  
        'agnostic': '_agnostic',         
        'batch_adjusted': '_batch_adjusted', 
        'group_validated': '_group_validated', 
    }
    
    # Use provided strategies or default to all
    strategies_to_run = strategies if strategies else list(available_strategies.keys())
    # Validate strategies
    strategies_to_run = [s for s in strategies_to_run if s in available_strategies]
    
    if not strategies_to_run:
        logger.warning("No valid strategies selected. Exiting.")
        return

    # 4. Batch Column Fallback Logic
    actual_batch_col = batch_col
    if batch_col not in adata.obs.columns and 'study_accession' in adata.obs.columns:
        logger.info(f"Batch column '{batch_col}' missing. Falling back to 'study_accession'.")
        actual_batch_col = 'study_accession'
    
    if actual_batch_col not in adata.obs.columns:
        logger.warning(f"Batch column '{actual_batch_col}' not found. Disabling batch-dependent strategies.")
        # Remove strategies that require batch info if it's completely missing
        if 'batch_adjusted' in strategies_to_run: strategies_to_run.remove('batch_adjusted')
        if 'group_validated' in strategies_to_run: strategies_to_run.remove('group_validated')

    results_list = []

    # --- MAIN LOOP ---
    for target_col in valid_targets:
        for strategy in strategies_to_run:
            suffix = available_strategies.get(strategy, f"_{strategy}")
            logger.info(f"--- CatBoost FS: {target_col} [{strategy}] ---")
            
            target_output_dir = catboost_output_dir / f"{strategy}" / f"{level}_{target_col}"
            target_output_dir.mkdir(exist_ok=True, parents=True)

            # 5. Apply Strategy Logic
            current_drop_batch = True
            current_use_group_kfold = False
            
            if strategy == 'baseline':
                current_drop_batch = True
                current_use_group_kfold = False
            elif strategy == 'agnostic':
                current_drop_batch = True
                current_use_group_kfold = False
            elif strategy == 'batch_adjusted':
                current_drop_batch = False # Keep batch in X
                current_use_group_kfold = False
            elif strategy == 'group_validated':
                current_drop_batch = True
                current_use_group_kfold = True # Strict isolation

            # 6. Data Prep
            X_run = X_df.copy()
            
            # If batch adjusted, add batch column to features
            if not current_drop_batch and actual_batch_col in adata.obs.columns:
                batch_series = adata.obs.loc[X_run.index, actual_batch_col].astype(str)
                X_run[actual_batch_col] = batch_series

            # Prepare Y
            y_series = adata.obs[target_col].replace(r'^\s*$', np.nan, regex=True).dropna()
            common_idx = X_run.index.intersection(y_series.index)
            
            if len(common_idx) < 20:
                logger.warning(f"Skipping {target_col}: < 20 samples.")
                continue

            X_final = X_run.loc[common_idx]
            y_final = y_series.loc[common_idx]
            
            # Determine Task Type
            is_numeric = pd.api.types.is_numeric_dtype(y_final)
            n_unique = y_final.nunique()
            task_type = 'Regression' if is_numeric and n_unique > 10 else 'Classification'
            loss_function = 'RMSE' if task_type == 'Regression' else 'Logloss'
            eval_metric = 'R2' if task_type == 'Regression' else 'MCC'

            # Prepare Groups for GroupKFold
            cv_groups = None
            if current_use_group_kfold and actual_batch_col in adata.obs.columns:
                cv_groups = adata.obs.loc[common_idx, actual_batch_col].values
                logger.info(f"Grouping by {len(np.unique(cv_groups))} batches for validation.")

            try:
                # 7. Run Core Feature Selection
                fs_results = catboost_feature_selection(
                    metadata=adata.obs.loc[common_idx],
                    features=X_final,
                    output_dir=target_output_dir,
                    group_col=target_col,
                    cv_groups=cv_groups, # Passed correctly based on strategy
                    method=method,
                    n_top_features=n_top_final,
                    task_type=task_type,
                    loss_function=loss_function,
                    eval_metric=eval_metric,
                    test_size=test_size,
                    use_permutation_importance=use_permutation,
                    thread_count=n_cpus,
                    num_features=num_features,
                    random_state=random_state,
                    verbose=True
                )
                
                # Save results
                results_summary = {
                    'target': target_col,
                    'strategy': strategy,
                    'best_cv_score': fs_results.get('best_cv_score'),
                    'top_features': fs_results.get('top_features', []),
                    'settings': {'drop_batch': current_drop_batch, 'group_kfold': current_use_group_kfold}
                }
                with open(target_output_dir / "results_summary.json", 'w') as f:
                    json.dump(results_summary, f, indent=4)
                    
                results_list.append(fs_results)
                
            except Exception as e:
                logger.error(f"CatBoost FS failed for {target_col} [{strategy}]: {e}")

    # 8. Hook up Visualization (After all loops)
    if results_list:
        try:
            logger.info("Generating Comprehensive ML Report...")
            # Collect groupings for visualization (excluding targets)
            grouping_vars = []
            if priority_targets:
                grouping_vars = [p for p in priority_targets if p not in valid_targets]

            generate_comprehensive_ml_report(
                catboost_dir=catboost_output_dir,
                output_dir=catboost_output_dir.parent / "ml_visualizations", # Save parallel to catboost dir
                ml_targets=valid_targets,
                grouping_variables=grouping_vars,
                strategies=strategies_to_run
            )
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")

    return results_list