"""
Batch-aware Machine Learning Module

This module extends the standard ML analysis with batch covariate control approaches:
1. Baseline: Standard ML without batch control
2. Covariate Adjustment: Include batch variables as features alongside taxa
3. Stratified Prediction: Two-stage residual analysis (predict batch, then biology)

Includes comprehensive confounding detection and comparison reporting.
"""

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
from catboost import CatBoostClassifier, CatBoostRegressor 

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
    model_algorithm: str = 'rf',   
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
        'algorithm': model_algorithm, 
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
    
    # Handle Feature Importances (CatBoost/RF differences)
    try:
        if hasattr(model_baseline, 'feature_importances_'):
            imps = model_baseline.feature_importances_
        elif hasattr(model_baseline, 'get_feature_importance'):
            imps = model_baseline.get_feature_importance()
        else:
            imps = []
    except Exception:
        imps = []

    results['models']['baseline'] = {
        'test_score': baseline_score,
        'oob_score': baseline_oob,
        'feature_importances': imps.tolist() if hasattr(imps, 'tolist') else list(imps)
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
        try:
            if hasattr(model_adjusted, 'feature_importances_'):
                importances = model_adjusted.feature_importances_
            elif hasattr(model_adjusted, 'get_feature_importance'):
                importances = model_adjusted.get_feature_importance()
            else:
                importances = np.zeros(X_adjusted.shape[1])
        except Exception:
            importances = np.zeros(X_adjusted.shape[1])
            
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
        batch_params = final_params.copy()
        if model_algorithm == 'catboost':
            batch_params['iterations'] = 50
            batch_params['depth'] = 6
        else:
            batch_params['n_estimators'] = 50
            batch_params['max_depth'] = 6
            
        batch_model = ModelClass(**batch_params)
        batch_model.fit(X_b_train, y_b_train)
        
        y_pred_batch_all = batch_model.predict(X_batch)
        batch_score = eval_func(y_b_test, batch_model.predict(X_b_test))
        
        # 2. Calculate Residuals
        if task_type == 'regression':
            residuals = y - y_pred_batch_all
        else:
            # Classification residuals: 1 if wrong, 0 if right (Simple error modeling)
            residuals = (y != y_pred_batch_all).astype(float)
            
        # 3. Train Residual Model (Residuals ~ Taxa)
        ResidClass = CatBoostRegressor if model_algorithm == 'catboost' else RandomForestRegressor
        
        X_r_train, X_r_test, r_train, r_test = train_test_split(
            X_taxa.fillna(0), residuals, test_size=0.3, random_state=42
        )
        
        # If residuals are constant (e.g. perfect prediction), skip training
        if residuals.nunique() <= 1:
             resid_score = 0.0
             logger.info("  Batch model predicted perfectly (residuals are 0). Skipping residual model.")
        else:
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

# ===================================================================================
#  VISUALIZATION FUNCTIONS (RESTORED)
# ===================================================================================

def create_comparison_plots(all_results: Dict[str, Dict], plot_dir: Path, level: str):
    """Create comparison visualizations across all targets and approaches."""
    if not all_results:
        return
    
    # We might have results from RF, CatBoost, or both. 
    # Current structure of all_results might be mixed.
    # This function expects standard structure, adapt if necessary.
    
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING BATCH CONTROL COMPARISON PLOTS")
    logger.info(f"{'='*80}")
    
    # Collect data for comparison
    targets = []
    baseline_scores = []
    adjusted_scores = []
    batch_fractions = []
    
    for target, result in all_results.items():
        if 'baseline' not in result.get('models', {}):
            continue
        
        targets.append(target)
        baseline_scores.append(result['models']['baseline']['test_score'])
        
        if 'covariate_adjusted' in result['models']:
            adjusted_scores.append(result['models']['covariate_adjusted']['test_score'])
            batch_fractions.append(result['models']['covariate_adjusted']['batch_importance_fraction'])
        else:
            adjusted_scores.append(None)
            batch_fractions.append(None)
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Model Performance Comparison", "Batch Variance Contribution"),
        vertical_spacing=0.15,
        specs=[[{"type": "bar"}], [{"type": "bar"}]]
    )
    
    # Plot 1: Performance comparison
    fig.add_trace(
        go.Bar(name='Baseline', x=targets, y=baseline_scores, marker_color='blue'),
        row=1, col=1
    )
    
    if any(s is not None for s in adjusted_scores):
        fig.add_trace(
            go.Bar(name='Covariate Adjusted', x=targets, y=[s for s in adjusted_scores if s is not None],
                   marker_color='orange'),
            row=1, col=1
        )
    
    # Plot 2: Batch contribution
    if any(f is not None for f in batch_fractions):
        fig.add_trace(
            go.Bar(name='Batch Importance Fraction', x=targets, 
                   y=[f*100 if f is not None else 0 for f in batch_fractions],
                   marker_color='red', showlegend=False),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Target Variable", row=2, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Batch Variance %", row=2, col=1)
    
    fig.update_layout(
        height=800,
        title_text=f"Batch Control Comparison Across Targets ({level})",
        showlegend=True
    )
    
    plot_path = plot_dir / f"batch_control_comparison_{level}.html"
    fig.write_html(str(plot_path))
    logger.info(f"✓ Comparison plot saved: {plot_path.name}")
    
    # Create summary markdown report
    create_summary_report(all_results, plot_dir, level)


def create_summary_report(all_results: Dict[str, Dict], plot_dir: Path, level: str):
    """Generate markdown summary report with interpretation guidance."""
    report_path = plot_dir / f"batch_control_summary_{level}.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# Batch Covariate Control - ML Analysis Summary\n\n")
        f.write(f"**Taxonomic Level:** {level}  \n")
        f.write(f"**Total Targets:** {len(all_results)}  \n\n")
        
        f.write("## Approaches Explained\n\n")
        f.write("### 1. Baseline (No Batch Control)\n")
        f.write("- **Method:** Train Model on taxa features only\n")
        f.write("- **Benefits:** Simple, interpretable, shows raw predictive power\n")
        f.write("- **Caveats:** May confound biological and technical variation\n")
        
        f.write("### 2. Covariate Adjustment\n")
        f.write("- **Method:** Train Model on [Taxa + Batch Variables] together\n")
        f.write("- **Benefits:** Taxa importances show biological signal AFTER controlling for batch\n")
        
        f.write("### 3. Stratified/Residual Prediction\n")
        f.write("- **Method:** Stage 1: Predict target from batch → Stage 2: Predict residuals from taxa\n")
        f.write("- **Benefits:** Explicitly quantifies batch vs biological contributions\n")
        
        f.write("## Results Summary\n\n")
        f.write("| Target | Baseline Score | Adjusted Score | Batch Variance % | Confounding |\n")
        f.write("|--------|---------------|----------------|------------------|-------------|\n")
        
        for target, result in sorted(all_results.items()):
            if 'baseline' not in result.get('models', {}):
                continue
            
            baseline = result['models']['baseline']['test_score']
            adjusted = result['models'].get('covariate_adjusted', {}).get('test_score', 'N/A')
            batch_frac = result['models'].get('covariate_adjusted', {}).get('batch_importance_fraction', 0)
            
            n_high = len(result.get('confounding', {}).get('high_confounding', []))
            n_mod = len(result.get('confounding', {}).get('moderate_confounding', []))
            
            if n_high > 0:
                confound_str = f"🔴 High ({n_high})"
            elif n_mod > 0:
                confound_str = f"🟡 Moderate ({n_mod})"
            else:
                confound_str = "🟢 Low"
            
            f.write(f"| {target} | {baseline:.3f} | {adjusted if isinstance(adjusted, str) else f'{adjusted:.3f}'} | ")
            f.write(f"{batch_frac*100:.1f}% | {confound_str} |\n")
        
        f.write("\n## Interpretation Guidelines\n\n")
        f.write("**High Confounding (🔴):**\n")
        f.write("- Batch variables are strongly associated with the target\n")
        f.write("- Results may be unreliable\n")
        
        f.write("**Batch Variance >50%:**\n")
        f.write("- Technical factors explain most of the predictive power\n")
        
    logger.info(f"✓ Summary report saved: {report_path.name}")