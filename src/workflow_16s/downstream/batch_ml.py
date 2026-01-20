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
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, matthews_corrcoef, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from workflow_16s.utils.logger import get_logger

logger = get_logger("workflow_16s")


def run_ml_with_batch_control(
    X_taxa: pd.DataFrame,
    y: pd.Series,
    batch_covariates: pd.DataFrame,
    target_col: str,
    task_type: str,
    plot_dir: Path,
    level: str,
    confounding_info: Dict[str, Any],
    batch_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run ML models with three approaches: baseline, covariate-adjusted, stratified.
    
    Returns comprehensive results for comparison.
    """
    results = {
        'target': target_col,
        'task_type': task_type,
        'level': level,
        'confounding': confounding_info,
        'models': {}
    }
    
    # Determine model type
    if task_type == 'regression':
        ModelClass = RandomForestRegressor
        metric_name = "R²"
        eval_func = r2_score
        stratify_opt = None
    else:
        ModelClass = RandomForestClassifier
        metric_name = "Accuracy"
        eval_func = accuracy_score
        stratify_opt = y
    
    # Common parameters
    model_params = {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'random_state': 42,
        'oob_score': True,
        'n_jobs': -1
    }
    
    # ===================================================================================
    # APPROACH 1: BASELINE (No batch control)
    # ===================================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"BASELINE MODEL (No Batch Control) - {target_col}")
    logger.info(f"{'='*80}")
    logger.info("Benefits: Simple, interpretable, shows raw predictive power")
    logger.info("Caveats: May confound biological and technical variation")
    
    X_baseline = X_taxa.fillna(0)
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_baseline, y, test_size=0.3, random_state=42, stratify=stratify_opt
        )
    except ValueError as e:
        logger.error(f"Train/test split failed: {e}")
        return results
    
    model_baseline = ModelClass(**model_params)
    model_baseline.fit(X_train, y_train)
    y_pred_baseline = model_baseline.predict(X_test)
    
    baseline_score = eval_func(y_test, y_pred_baseline)
    baseline_oob = model_baseline.oob_score_
    
    logger.info(f"✓ Test {metric_name}: {baseline_score:.3f}")
    logger.info(f"✓ OOB Score: {baseline_oob:.3f}")
    logger.info(f"✓ Generalization Gap: {abs(baseline_oob - baseline_score):.3f}")
    
    # Get feature importances
    baseline_importances = pd.DataFrame({
        'Feature': X_baseline.columns,
        'Importance': model_baseline.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    results['models']['baseline'] = {
        'test_score': baseline_score,
        'oob_score': baseline_oob,
        'generalization_gap': abs(baseline_oob - baseline_score),
        'top_features': baseline_importances.head(20).to_dict('records'),
        'n_features': len(X_baseline.columns)
    }
    
    # ===================================================================================
    # APPROACH 2: COVARIATE ADJUSTMENT (Include batch as features)
    # ===================================================================================
    if batch_config.get('covariate_adjustment', {}).get('enabled', False):
        logger.info(f"\n{'='*80}")
        logger.info(f"COVARIATE-ADJUSTED MODEL - {target_col}")
        logger.info(f"{'='*80}")
        logger.info("Benefits: Taxa importances show biological signal AFTER controlling for batch")
        logger.info("Caveats: Can't separate effects if batch perfectly confounds biology")
        logger.info("Approach: Train model on [Taxa + Batch Covariates] → Predict Target")
        
        # Combine taxa and batch features
        X_adjusted = pd.concat([X_taxa, batch_covariates], axis=1).fillna(0)
        logger.info(f"Combined features: {len(X_taxa.columns)} taxa + {len(batch_covariates.columns)} batch = {len(X_adjusted.columns)} total")
        
        X_train_adj, X_test_adj, y_train_adj, y_test_adj = train_test_split(
            X_adjusted, y, test_size=0.3, random_state=42, stratify=stratify_opt
        )
        
        model_adjusted = ModelClass(**model_params)
        model_adjusted.fit(X_train_adj, y_train_adj)
        y_pred_adjusted = model_adjusted.predict(X_test_adj)
        
        adjusted_score = eval_func(y_test_adj, y_pred_adjusted)
        adjusted_oob = model_adjusted.oob_score_
        
        logger.info(f"✓ Test {metric_name}: {adjusted_score:.3f}")
        logger.info(f"✓ OOB Score: {adjusted_oob:.3f}")
        logger.info(f"✓ Generalization Gap: {abs(adjusted_oob - adjusted_score):.3f}")
        
        # Separate taxa vs batch importances
        all_importances = pd.DataFrame({
            'Feature': X_adjusted.columns,
            'Importance': model_adjusted.feature_importances_
        })
        
        taxa_cols = set(X_taxa.columns)
        taxa_importances = all_importances[all_importances['Feature'].isin(taxa_cols)].sort_values('Importance', ascending=False)
        batch_importances = all_importances[~all_importances['Feature'].isin(taxa_cols)].sort_values('Importance', ascending=False)
        
        total_taxa_importance = taxa_importances['Importance'].sum()
        total_batch_importance = batch_importances['Importance'].sum()
        
        logger.info(f"\nFeature Importance Breakdown:")
        logger.info(f"  • Total Taxa Importance: {total_taxa_importance:.3f} ({total_taxa_importance/all_importances['Importance'].sum()*100:.1f}%)")
        logger.info(f"  • Total Batch Importance: {total_batch_importance:.3f} ({total_batch_importance/all_importances['Importance'].sum()*100:.1f}%)")
        
        if total_batch_importance > total_taxa_importance:
            logger.warning(f"⚠️  Batch effects explain MORE variance than taxa ({total_batch_importance:.2f} vs {total_taxa_importance:.2f})")
            logger.warning("    This suggests strong technical confounding - biological interpretation should be cautious")
        
        results['models']['covariate_adjusted'] = {
            'test_score': adjusted_score,
            'oob_score': adjusted_oob,
            'generalization_gap': abs(adjusted_oob - adjusted_score),
            'top_taxa_features': taxa_importances.head(20).to_dict('records'),
            'top_batch_features': batch_importances.head(10).to_dict('records'),
            'taxa_importance_fraction': total_taxa_importance / all_importances['Importance'].sum(),
            'batch_importance_fraction': total_batch_importance / all_importances['Importance'].sum(),
            'n_features': len(X_adjusted.columns)
        }
    
    # ===================================================================================
    # APPROACH 3: STRATIFIED PREDICTION (Two-stage residual analysis)
    # ===================================================================================
    if batch_config.get('stratified_prediction', {}).get('enabled', False):
        logger.info(f"\n{'='*80}")
        logger.info(f"STRATIFIED/RESIDUAL MODEL - {target_col}")
        logger.info(f"{'='*80}")
        logger.info("Benefits: Explicitly removes batch signal before biological prediction")
        logger.info("Caveats: May remove true biological variation if it correlates with batch")
        logger.info("Approach: Stage 1: Batch → Target, Stage 2: Taxa → Residuals")
        
        # Stage 1: Train batch-only model
        logger.info("\nStage 1: Predicting target from batch covariates only...")
        
        X_batch = batch_covariates.fillna(0)
        X_batch_train, X_batch_test, y_batch_train, y_batch_test = train_test_split(
            X_batch, y, test_size=0.3, random_state=42, stratify=stratify_opt
        )
        
        model_batch = ModelClass(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1, oob_score=True)
        model_batch.fit(X_batch_train, y_batch_train)
        
        # Get batch predictions for ALL samples
        y_pred_batch_all = model_batch.predict(X_batch.fillna(0))
        batch_only_score = eval_func(y_batch_test, model_batch.predict(X_batch_test))
        
        logger.info(f"  Batch-only model {metric_name}: {batch_only_score:.3f}")
        logger.info(f"  This represents the 'technical variance' explained by batch effects")
        
        # Calculate residuals
        if task_type == 'regression':
            residuals = y - y_pred_batch_all
            logger.info(f"  Calculated continuous residuals (target - batch_prediction)")
        else:
            # For classification, use indicator residuals (was prediction correct?)
            residuals = (y != y_pred_batch_all).astype(float)
            logger.info(f"  Calculated binary residuals (prediction errors)")
        
        # Stage 2: Train taxa model on residuals
        logger.info("\nStage 2: Predicting residuals from taxa...")
        
        X_taxa_clean = X_taxa.fillna(0)
        X_taxa_train, X_taxa_test, resid_train, resid_test = train_test_split(
            X_taxa_clean, residuals, test_size=0.3, random_state=42
        )
        
        if task_type == 'regression':
            model_residual = RandomForestRegressor(**model_params)
        else:
            # For classification residuals, use regression (predicting error probability)
            model_residual = RandomForestRegressor(**model_params)
        
        model_residual.fit(X_taxa_train, resid_train)
        resid_pred = model_residual.predict(X_taxa_test)
        
        if task_type == 'regression':
            residual_score = r2_score(resid_test, resid_pred)
        else:
            # For binary residuals, convert back and measure accuracy improvement
            residual_score = r2_score(resid_test, resid_pred)
        
        logger.info(f"  Residual model R²: {residual_score:.3f}")
        logger.info(f"  This represents the 'biological variance' explained by taxa after removing batch")
        
        residual_importances = pd.DataFrame({
            'Feature': X_taxa_clean.columns,
            'Importance': model_residual.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        results['models']['stratified'] = {
            'batch_model_score': batch_only_score,
            'residual_model_score': residual_score,
            'combined_interpretation': f"Batch explains {batch_only_score:.1%}, Taxa explain {residual_score:.1%} of remaining variance",
            'top_residual_features': residual_importances.head(20).to_dict('records'),
            'n_features': len(X_taxa_clean.columns)
        }
    
    # ===================================================================================
    # COMPARISON AND RECOMMENDATIONS
    # ===================================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"SUMMARY AND RECOMMENDATIONS - {target_col}")
    logger.info(f"{'='*80}")
    
    # Check for high confounding
    if confounding_info['high_confounding']:
        logger.warning(f"⚠️  HIGH CONFOUNDING DETECTED:")
        for batch_var in confounding_info['high_confounding']:
            stat = confounding_info['statistics'][batch_var]
            logger.warning(f"   • {batch_var}: {stat['type']}={stat['value']:.3f}")
        logger.warning("   RECOMMENDATION: Results may be unreliable. Consider:")
        logger.warning("   - Collecting more balanced data across batches")
        logger.warning("   - Stratified sampling in future experiments")
        logger.warning("   - Focus on stratified model results (less affected by confounding)")
    
    if 'covariate_adjusted' in results['models']:
        batch_frac = results['models']['covariate_adjusted']['batch_importance_fraction']
        if batch_frac > 0.5:
            logger.warning(f"⚠️  TECHNICAL VARIANCE DOMINATES:")
            logger.warning(f"   Batch effects explain {batch_frac:.1%} of total variance")
            logger.warning("   RECOMMENDATION: Biological signal is weak or heavily confounded")
    
    # Save detailed results
    safe_target = re.sub(r'[^A-Za-z0-9_]+', '', target_col)
    results_path = plot_dir / f"batch_ml_results_{safe_target}_{level}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✓ Detailed results saved to: {results_path.name}")
    
    return results


def create_comparison_plots(all_results: Dict[str, Dict], plot_dir: Path, level: str):
    """Create comparison visualizations across all targets and approaches."""
    if not all_results:
        return
    
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING BATCH CONTROL COMPARISON PLOTS")
    logger.info(f"{'='*80}")
    
    # Collect data for comparison
    targets = []
    baseline_scores = []
    adjusted_scores = []
    stratified_scores = []
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
        
        if 'stratified' in result['models']:
            # Use batch model score as proxy for comparison
            stratified_scores.append(result['models']['stratified']['batch_model_score'])
        else:
            stratified_scores.append(None)
    
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
        f.write("- **Method:** Train Random Forest on taxa features only\n")
        f.write("- **Benefits:** Simple, interpretable, shows raw predictive power\n")
        f.write("- **Caveats:** May confound biological and technical variation\n")
        f.write("- **Use When:** Batch effects are minimal or you want to assess total predictive power\n\n")
        
        f.write("### 2. Covariate Adjustment\n")
        f.write("- **Method:** Train Random Forest on [Taxa + Batch Variables] together\n")
        f.write("- **Benefits:** Taxa importances show biological signal AFTER controlling for batch\n")
        f.write("- **Caveats:** Can't separate effects if batch perfectly confounds biology\n")
        f.write("- **Use When:** You want to identify taxa that matter beyond batch effects\n\n")
        
        f.write("### 3. Stratified/Residual Prediction\n")
        f.write("- **Method:** Stage 1: Predict target from batch → Stage 2: Predict residuals from taxa\n")
        f.write("- **Benefits:** Explicitly quantifies batch vs biological contributions\n")
        f.write("- **Caveats:** May remove true biological variation if it correlates with batch\n")
        f.write("- **Use When:** You want to decompose technical vs biological variance\n\n")
        
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
        f.write("- Results may be unreliable - biological vs technical effects cannot be separated\n")
        f.write("- **Action:** Focus on stratified model, consider experimental redesign\n\n")
        
        f.write("**Batch Variance >50%:**\n")
        f.write("- Technical factors explain most of the predictive power\n")
        f.write("- Biological signal is weak or heavily confounded\n")
        f.write("- **Action:** Use covariate-adjusted feature importances, validate findings with independent data\n\n")
        
        f.write("**Low Confounding (🟢) + Low Batch Variance (<20%):**\n")
        f.write("- Batch effects are minimal\n")
        f.write("- Baseline and adjusted models should give similar results\n")
        f.write("- **Action:** Baseline results are reliable\n\n")
    
    logger.info(f"✓ Summary report saved: {report_path.name}")
