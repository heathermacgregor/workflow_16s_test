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
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr, chi2_contingency
from scipy.stats.contingency import association

from workflow_16s.downstream.helpers import AnalysisUtils
from workflow_16s.downstream.plotting import PlottingUtils, DEFAULT_HEIGHT
from workflow_16s.downstream.models.feature_selection import catboost_feature_selection, filter_data
from workflow_16s.downstream.models.overfitting_prevention import run_comprehensive_validation
from workflow_16s.downstream.batch_ml import run_ml_with_batch_control, create_comparison_plots
from workflow_16s.utils.logger import get_logger

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

# ==================================================================================== #
# BATCH COVARIATE HELPER FUNCTIONS
# ==================================================================================== #

def prepare_batch_covariates(
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
        if col not in adata.obs.columns:
            logger.warning(f"Batch column '{col}' not found in metadata, skipping")
            metadata['dropped'].append({'column': col, 'reason': 'not_found'})
            continue
        
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
    validate_overfitting: bool = True, 
    quick_validation: bool = False,
    batch_config: Optional[Dict[str, Any]] = None
):
    """
    Uses Random Forest (parallelized) to predict metadata with overfitting validation.
    Supports batch covariate adjustment and stratified prediction approaches.
    """
    logger.info(f"--- Starting Machine Learning Analysis ({level}) ---")
    
    # Check if batch covariates are enabled
    batch_enabled = batch_config and batch_config.get('enabled', False)
    run_baseline = True  # Always run baseline (no batch control)
    run_adjusted = batch_enabled and batch_config.get('covariate_adjustment', {}).get('enabled', False)
    run_stratified = batch_enabled and batch_config.get('stratified_prediction', {}).get('enabled', False)
    
    if batch_enabled:
        logger.info("=" * 80)
        logger.info("BATCH COVARIATE CONTROL ENABLED")
        logger.info("=" * 80)
        logger.info(f"  • Baseline (no batch control): {run_baseline}")
        logger.info(f"  • Covariate Adjustment: {run_adjusted}")
        logger.info(f"  • Stratified Prediction: {run_stratified}")
        logger.info("=" * 80)
    
    adata_agg = AnalysisUtils.get_analysis_adata(adata, level=level)
    if adata_agg is None or adata_agg.n_obs < 20 or adata_agg.n_vars < 10: 
        logger.warning(f"Skipping ML"); 
        return
    
    X_df = AnalysisUtils.clr_transform(adata_agg, pseudocount=1)
    ml_targets = AnalysisUtils.find_plottable_metadata(adata, fullness_threshold=0.5, max_categories=50)
    all_targets_auto = ml_targets['categorical'] + ml_targets['numeric']
    
    if priority_targets:
        final_targets = [var for var in priority_targets if var in adata.obs.columns]
        for var in all_targets_auto:
            if var not in final_targets: 
                final_targets.append(var)
        all_targets = final_targets
    else: 
        all_targets = all_targets_auto
    
    if not all_targets: 
        logger.warning("No suitable metadata targets found for ML."); 
        return
    
    # Pre-filter targets to avoid wasteful attempts and reduce log clutter
    filter_results = AnalysisUtils.filter_ml_targets(adata, all_targets, min_samples_per_group, max_classes)
    valid_targets = filter_results['valid']
    
    if not valid_targets: 
        logger.warning("No valid ML targets after pre-filtering."); 
        return
    
    # Initialize results storage for comparison
    all_results = {}
    
    # Prepare batch covariates if enabled
    batch_covariates_df = None
    if batch_enabled and batch_config.get('covariate_columns'):
        try:
            batch_covariates_df, batch_metadata = prepare_batch_covariates(
                adata=adata,
                batch_columns=batch_config['covariate_columns'],
                sample_indices=X_df.index,
                one_hot_encode=batch_config.get('covariate_adjustment', {}).get('one_hot_encode', True),
                max_categories=batch_config.get('covariate_adjustment', {}).get('max_categories', 20)
            )
            logger.info(f"✓ Batch covariates prepared: {len(batch_covariates_df.columns)} features")
        except Exception as e:
            logger.error(f"Failed to prepare batch covariates: {e}")
            batch_enabled = False
    
    for target_col in valid_targets:
        logger.info(f"\n{'='*80}")
        logger.info(f"TARGET: {target_col}")
        logger.info(f"{'='*80}")
        
        y_series = adata.obs[target_col]
        common_idx = X_df.index.intersection(y_series.dropna().index)
        if len(common_idx) < 20: 
            logger.warning(f"Skipping {target_col}: < 20 samples")
            continue
        
        X = X_df.loc[common_idx]
        y = y_series.loc[common_idx]
        is_numeric_dtype = pd.api.types.is_numeric_dtype(y)
        n_unique = y.nunique()
        
        # Determine task type
        task_type = 'regression' if (is_numeric_dtype and n_unique > max_classes) else 'classification'
        logger.info(f"Task: {task_type.capitalize()}")
        
        # Detect confounding if batch control is enabled
        confounding_info = {}
        if batch_enabled and batch_config.get('confounding_detection', {}).get('enabled', False):
            confounding_info = detect_confounding(
                adata=adata,
                batch_columns=batch_config['covariate_columns'],
                target_col=target_col,
                sample_indices=common_idx,
                threshold=batch_config.get('confounding_detection', {}).get('correlation_threshold', 0.7),
                plot_dir=plot_dir_ml if batch_config.get('confounding_detection', {}).get('plot_confounding', False) else None
            )
        
        # If batch control is enabled, use comprehensive batch-aware ML
        if batch_enabled and batch_covariates_df is not None and len(batch_covariates_df) > 0:
            batch_cov_aligned = batch_covariates_df.loc[common_idx]
            
            try:
                target_results = run_ml_with_batch_control(
                    X_taxa=X,
                    y=y,
                    batch_covariates=batch_cov_aligned,
                    target_col=target_col,
                    task_type=task_type,
                    plot_dir=plot_dir_ml,
                    level=level,
                    confounding_info=confounding_info,
                    batch_config=batch_config
                )
                all_results[target_col] = target_results
                continue  # Skip standard baseline below
            except Exception as e:
                logger.error(f"Batch-aware ML failed for {target_col}: {e}")
                logger.info("Falling back to standard baseline model...")
        
        # Standard baseline model (no batch control) 
        # This runs if batch control is disabled OR if batch-aware ML failed
        model = None
        eval_func = None
        stratify_opt = None
        if is_numeric_dtype and n_unique > max_classes: 
            logger.info(f"Task: Regression...")
            # Add regularization to prevent overfitting
            model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=15,  # Limit tree depth
                min_samples_split=10,  # Require more samples to split
                min_samples_leaf=5,  # Require more samples in leaf nodes
                max_features='sqrt',  # Only consider sqrt(n_features) at each split
                random_state=42, 
                oob_score=True, 
                n_jobs=-1
            )
            metric_name = "R-squared"; score_name = "R-squared"; eval_func = r2_score
        elif not is_numeric_dtype or n_unique <= max_classes:
            logger.info(f"Task: Classification..."); class_counts = y.value_counts()
            # These checks should pass after pre-filtering, but keep for safety
            if class_counts.min() < min_samples_per_group: continue
            if n_unique > max_classes: continue
            # Add regularization to prevent overfitting
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,  # Limit tree depth
                min_samples_split=10,  # Require more samples to split
                min_samples_leaf=5,  # Require more samples in leaf nodes
                max_features='sqrt',  # Only consider sqrt(n_features) at each split
                random_state=42, 
                oob_score=True, 
                n_jobs=-1
            )
            metric_name = "Accuracy"; score_name = "Accuracy"; eval_func = accuracy_score; stratify_opt = y
        else: continue
        
        # Validate data before train_test_split to prevent silent failures
        if X.shape[0] != y.shape[0]:
            logger.error(f"Skipping {target_col}: X and y shape mismatch ({X.shape[0]} vs {y.shape[0]})"); continue
        if X.isnull().any().any():
            logger.warning(f"Target {target_col}: X contains NaN values, filling with 0")
            X = X.fillna(0)
        if stratify_opt is not None and len(y.unique()) > len(y) * 0.3:
            logger.warning(f"Target {target_col}: Too many unique values for stratification, using random split")
            stratify_opt = None
        
        try: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=stratify_opt)
        except Exception as e: logger.error(f"Skipping {target_col}: train_test_split failed after validation. {e}"); continue
        model.fit(X_train, y_train); y_pred = model.predict(X_test); score = eval_func(y_test, y_pred); oob_score = model.oob_score_; logger.info(f"Model for {target_col} complete.\n -> Test Set {metric_name}: {score:.3f}\n -> OOB Score: {oob_score:.3f}")
        
        # Check for overfitting using OOB vs test score
        oob_test_gap = abs(oob_score - score)
        if oob_test_gap > 0.15:
            logger.warning(f" -> ⚠️  OVERFITTING WARNING: OOB-Test gap = {oob_test_gap:.3f} (threshold: 0.15)")
        elif oob_test_gap > 0.10:
            logger.warning(f" -> ⚠️  MODERATE OVERFITTING: OOB-Test gap = {oob_test_gap:.3f}")
        else:
            logger.info(f" -> ✓ Good generalization: OOB-Test gap = {oob_test_gap:.3f} < 0.10")
        
        if is_numeric_dtype and n_unique > max_classes: 
            rmse = np.sqrt(mean_squared_error(y_test, y_pred)); logger.info(f" -> Test Set RMSE: {rmse:.3f}")
        else: 
            # Calculate imbalance-robust metrics
            mcc = matthews_corrcoef(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            logger.info(f" -> Matthews Correlation Coefficient (MCC): {mcc:.3f}")
            logger.info(f" -> Balanced Accuracy: {balanced_acc:.3f}")
            report = str(classification_report(y_test, y_pred, zero_division=0, output_dict=False)); logger.info(" -> Classification Report (Test Set):\n" + report)
        
        # Comprehensive overfitting validation for priority targets or suspicious results
        run_validation = False
        if validate_overfitting and priority_targets and target_col in priority_targets:
            run_validation = True
            logger.info(f" -> Running validation (priority target)")
        elif validate_overfitting and (score > 0.9 or oob_score > 0.9):
            run_validation = True
            logger.info(f" -> Running validation (suspiciously high performance)")
        elif validate_overfitting and oob_test_gap > 0.10:
            run_validation = True
            logger.info(f" -> Running validation (overfitting detected)")
        
        if run_validation:
            try:
                task = 'Regression' if (is_numeric_dtype and n_unique > max_classes) else 'Classification'
                validation_results = run_comprehensive_validation(
                    model=model,
                    X=X,
                    y=y,
                    output_dir=plot_dir_ml,
                    target_name=target_col,
                    groups=None,  # Could pass batch info if available
                    task_type=task,
                    n_permutations=50 if quick_validation else 100,
                    n_bootstrap=25 if quick_validation else 50,
                    quick_mode=quick_validation
                )
                # Store validation flag in model name for reporting
                logger.info(f" -> ✓ Overfitting validation complete")
            except Exception as e:
                logger.warning(f" -> Validation failed: {e}")
        
        importances = model.feature_importances_
        feat_importance_df = pd.DataFrame({
            'Taxon': X.columns, 
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        top_features = feat_importance_df.head(20)
        text = [f" -> Top 10 predictive taxa:"]
        for _, row in top_features.head(10).iterrows():
            text.append(f"       - {row['Taxon']} ({row['Importance']:.4f})")
        logger.info("\n".join(text))
        plot_feature_importances(top_features, plot_dir_ml, target_col, level, oob_score, score_name)
    
    # Generate comparison plots if batch control was used
    if batch_enabled and len(all_results) > 0:
        try:
            create_comparison_plots(all_results, plot_dir_ml, level)
        except Exception as e:
            logger.error(f"Failed to create comparison plots: {e}")


def run_catboost_selection(
    adata: ad.AnnData, 
    catboost_output_dir: Path, 
    level: str = 'Genus', 
    priority_targets: Optional[List[str]] = None, 
    test_size: float = 0.3, 
    random_state: int = 42, 
    num_features: int = 50, 
    n_top_final: int = 20, 
    method: str = 'shap', 
    use_permutation: bool = False, 
    n_cpus: int = 4,
    drop_batch: bool = True,           
    use_group_kfold: bool = True,      
    batch_col: str = 'batch_original'
):
    """Runs CatBoost feature selection for specified targets with batch control."""
    logger.info(f"--- Starting CatBoost Feature Selection ({level}, method={method}) ---")
    logger.info(f"Control Strategy: drop_batch={drop_batch}, use_group_kfold={use_group_kfold}")
    
    if catboost_feature_selection is None or filter_data is None: logger.error("catboost_feature_selection functions not imported correctly. Skipping."); return
    
    adata_agg = AnalysisUtils.get_analysis_adata(adata, level=level)
    if adata_agg is None or adata_agg.n_obs < 20 or adata_agg.n_vars < 2: 
        logger.warning(f"Skipping CatBoost FS: Not enough samples/features at {level}."); return
        
    X_df = AnalysisUtils.clr_transform(adata_agg, pseudocount=1)
    
    # Logic for Strategy 1: Include batch as a covariate predictor
    if not drop_batch and batch_col in adata.obs.columns:
        logger.info(f"Adding '{batch_col}' as a covariate feature for Baseline run.")
        X_df[batch_col] = adata.obs.loc[X_df.index, batch_col].astype('category')
    
    ml_targets = AnalysisUtils.find_plottable_metadata(adata, fullness_threshold=0.5, max_categories=50); 
    all_targets_auto = ml_targets['categorical'] + ml_targets['numeric']
    
    if priority_targets: final_targets = [var for var in priority_targets if var in adata.obs.columns]; [final_targets.append(var) for var in all_targets_auto if var not in final_targets]; all_targets = final_targets
    else: all_targets = all_targets_auto
    if not all_targets: logger.warning("No suitable targets found for CatBoost FS."); return
    
    results_list = []

    for target_col in all_targets:
        logger.info(f"--- CatBoost FS for target: {target_col} ---")
        target_output_dir = catboost_output_dir / f"{level}_{target_col}"; target_output_dir.mkdir(exist_ok=True, parents=True)
        
        y_series_orig = adata.obs[target_col]
        y_series_cleaned = y_series_orig.replace(r'^\s*$', np.nan, regex=True).dropna()
        y_numeric = pd.to_numeric(y_series_cleaned, errors='coerce')
        if y_numeric.notna().sum() > (0.5 * len(y_series_cleaned)): y_series_cleaned = y_numeric[y_numeric.notna()]

        if y_series_cleaned.empty: logger.warning(f"Skipping {target_col}: No valid samples."); continue
        
        is_numeric_dtype = pd.api.types.is_numeric_dtype(y_series_cleaned); n_unique = y_series_cleaned.nunique(); max_classes = 10 
        task_type = 'Regression' if is_numeric_dtype and n_unique > max_classes else 'Classification'
        eval_metric = 'R2' if task_type == 'Regression' else 'MCC'
        loss_function = 'RMSE' if task_type == 'Regression' else 'Logloss'
        
        y_series = y_series_orig.replace(r'^\s*$', np.nan, regex=True)
        common_idx = X_df.index.intersection(y_series.dropna().index) 
        if len(common_idx) < 20: logger.warning(f"Skipping {target_col}: < 20 samples."); continue
        
        X = X_df.loc[common_idx]; y_series_common = y_series.loc[common_idx] 
        y = y_series_common.astype('category').cat.codes if task_type == 'Classification' else pd.to_numeric(y_series_common, errors='coerce')
        
        # Validate class balance for classification tasks
        if task_type == 'Classification':
            class_counts = pd.Series(y).value_counts()
            n_classes = len(class_counts)
            min_class_size = class_counts.min()
            max_class_size = class_counts.max()
            imbalance_ratio = max_class_size / min_class_size if min_class_size > 0 else float('inf')
            
            if min_class_size < 10:
                logger.warning(f"Skipping {target_col}: minority class has only {min_class_size} samples (need ≥10)")
                continue
            if imbalance_ratio > 100:
                logger.warning(f"Skipping {target_col}: severe class imbalance ({imbalance_ratio:.0f}:1 ratio, need <100:1)")
                continue
            if n_classes > max_classes:
                logger.warning(f"Skipping {target_col}: too many classes ({n_classes}, max {max_classes})")
                continue
        
        if y.isna().any():
            valid_y_idx = y.dropna().index
            if len(valid_y_idx) < 20: continue
            X = X.loc[valid_y_idx]; y = y.loc[valid_y_idx]

        metadata_subset = adata.obs.loc[X.index].copy()
        metadata_subset[target_col] = y 
        
        # Strategy 3: Identify Groups for GroupKFold
        cv_groups = None
        if use_group_kfold and batch_col in adata.obs.columns:
            cv_groups = adata.obs.loc[common_idx, batch_col].values
            logger.info(f"Grouping by {len(np.unique(cv_groups))} unique batches for out-of-batch validation.")
            
        try:
            fs_results = catboost_feature_selection(
                metadata=metadata_subset, features=X, output_dir=target_output_dir, group_col=target_col, 
                cv_groups=cv_groups, method=method, n_top_features=n_top_final, task_type=task_type,
                loss_function=loss_function, eval_metric=eval_metric, test_size=test_size, 
                use_permutation_importance=use_permutation, thread_count=n_cpus,
                num_features=num_features, random_state=random_state, verbose=True
            ); top_features = fs_results.get('top_features', [])
            
            best_score = fs_results.get('best_cv_score', 'N/A')
            best_score_str = f"{best_score:.4f}" if isinstance(best_score, (int, float)) else str(best_score)
            logger.info(f"CatBoost FS completed for {target_col}.\n   Best CV Score ({eval_metric}): {best_score_str}")
            
            results_summary = {
                'target': target_col, 'level': level, 'method': fs_results.get('method'), 'task_type': task_type,
                'n_initial_selected': num_features, 'n_final_reported': len(top_features), 
                'best_cv_score': fs_results.get('best_cv_score'), 'eval_metric': eval_metric,
                'test_scores': fs_results.get('test_scores'), 'top_features': top_features, 
                'best_params': fs_results.get('best_params'), 'drop_batch': drop_batch, 'use_group_kfold': use_group_kfold
            }
            summary_path = target_output_dir / "results_summary.json"
            with open(summary_path, 'w') as f: json.dump(results_summary, f, indent=4)
            
            shap_df = fs_results.get('shap_report_df')
            if shap_df is not None and not shap_df.empty: shap_df.to_csv(target_output_dir / "shap_report_details.csv", index=False)
            results_list.append(fs_results)
        
        except Exception as e: logger.error(f"CatBoost FS failed for {target_col}: {e}"); continue

    return results_list