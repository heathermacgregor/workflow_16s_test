# workflow_16s/modules/machine_learning/catboost/batch_control/batch_control.py

import json
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from catboost import CatBoostClassifier, CatBoostRegressor 
from plotly.subplots import make_subplots
from scipy.stats import kruskal, spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from workflow_16s.utils.logger import get_logger, with_logger
logger = get_logger("workflow_16s")
from ..utils import get_model_class, sanitize_catboost_params
from workflow_16s.visualization.machine_learning.batch_dependency import (
    create_confounding_heatmap
)

def audit_biomarker_confidence(
    X_taxa: pd.DataFrame, batch_covs: pd.DataFrame, 
    top_taxa: List[str], effect_limit: float = 0.8
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Ranks biomarkers by how 'clean' they are from technical noise.
    Uses Spearman (ρ) for continuous batch vars and Eta-squared (η²) for categorical.
    Score starts at 100; -20 points for every significant technical correlation.
    """
    report, exclusions = [], []
    
    for taxon in top_taxa:
        score, links = 100, []
        is_leaky = False
        taxon_abundance = X_taxa[taxon]
        
        for var in batch_covs.columns:
            b_vals = batch_covs[var]
            is_categorical = b_vals.dtype == 'object' or isinstance(b_vals.dtype, pd.CategoricalDtype)
            
            # Drop NaNs to align the two vectors safely
            valid_idx = taxon_abundance.notna() & b_vals.notna()
            t_valid = taxon_abundance[valid_idx]
            b_valid = b_vals[valid_idx]
            
            if len(t_valid) < 10:
                continue
                
            p_val, effect_size = 1.0, 0.0
            stat_name = ""
            
            if not is_categorical and pd.api.types.is_numeric_dtype(b_valid):
                # Numeric batch variable
                rho, p_val = spearmanr(t_valid, b_valid)
                effect_size = abs(float(rho))
                stat_name = "ρ"
            else:
                # Categorical batch variable
                groups = [t_valid[b_valid == cat].values for cat in b_valid.unique()]
                groups = [g for g in groups if len(g) > 1]
                
                if len(groups) >= 2:
                    # Non-parametric p-value
                    stat, p_val = kruskal(*groups)
                    
                    # Compute Eta-squared (η²) for effect size against effect_limit
                    grand_mean = t_valid.mean()
                    ss_total = np.sum((t_valid - grand_mean)**2)
                    ss_between = np.sum([len(g) * (np.mean(g) - grand_mean)**2 for g in groups])
                    
                    effect_size = ss_between / ss_total if ss_total > 0 else 0.0
                    stat_name = "η²"
            
            # Score penalty logic
            if p_val < 0.05:
                score -= 20
                links.append(f"{var} ({stat_name}={effect_size:.2f})")
                
                if effect_size >= effect_limit:
                    is_leaky = True
                    exclusions.append({
                        'taxon': taxon, 'var': var, 
                        'effect_size': effect_size, 'metric': stat_name
                    })
        
        report.append({
            'taxon': taxon, 
            'score': max(0, score), 
            'technical_links': "; ".join(links),
            'status': 'REJECTED' if is_leaky else 'PASSED'
        })
        
    return pd.DataFrame(report), exclusions

@with_logger
def run_ml_with_batch_control(
    X_taxa: pd.DataFrame, y: pd.Series, batch_covariates: pd.DataFrame,
    target_col: str, task_type: str, plot_dir: Path, level: str,
    confounding_info: Dict[str, Any], batch_config: Dict[str, Any],
    model_algorithm: str = 'rf', model_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Full execution of the three batch-control ML strategies."""
    results = {
        'target': target_col, 'task_type': task_type, 'level': level,
        'algorithm': model_algorithm, 'confounding': confounding_info,
        'models': {}, 'confidence_report': []
    }
    
    ModelClass = get_model_class(task_type, model_algorithm)
    metric_name, eval_func, stratify_opt = ("R²", r2_score, None) if task_type.lower() == 'regression' else ("MCC", matthews_corrcoef, y)
    
    final_params = (model_params or {}).copy()
    if model_algorithm.lower() == 'catboost':
        final_params.update({'verbose': False, 'allow_writing_files': False, 'thread_count': 4})
        final_params = sanitize_catboost_params(final_params)
    else:
        final_params.setdefault('n_estimators', 100)
        final_params.setdefault('n_jobs', -1)

    # A1: Baseline
    X_train, X_test, y_train, y_test = train_test_split(X_taxa.fillna(0), y, test_size=0.3, random_state=42, stratify=stratify_opt)
    model_b = ModelClass(**final_params).fit(X_train, y_train)
    baseline_score = eval_func(y_test, model_b.predict(X_test))
    
    if model_algorithm.lower() == 'catboost':
        imp = model_b.get_feature_importance() # type: ignore
    else:
        imp = model_b.feature_importances_
    # Ensure imp is a 1D numpy array
    if isinstance(imp, pd.DataFrame):
        imp = imp.values.squeeze()
    elif isinstance(imp, pd.Series):
        imp = imp.values
    imp = np.asarray(imp).flatten()
    top_taxa = X_taxa.columns[np.argsort(imp)[::-1][:30]].tolist()
    results['models']['baseline'] = {'test_score': float(baseline_score)}

    # A2: Covariate Adjustment
    if batch_config.get('covariate_adjustment', {}).get('enabled', False):
        # Concatenate taxa (numeric) with batch covariates (mixed types)
        X_adj = pd.concat([X_taxa, batch_covariates], axis=1)
        
        # We cannot blanket fillna(0) anymore because 0 is numeric and categoricals need strings/categories.
        # X_taxa is numeric, so we can fill those safely. Batch covs were already cleaned in prepare_batch_covariates!
        X_taxa_filled = X_taxa.fillna(0)
        X_adj = pd.concat([X_taxa_filled, batch_covariates], axis=1)

        X_tr, X_te, y_tr, y_te = train_test_split(X_adj, y, test_size=0.3, random_state=42, stratify=stratify_opt)
        
        # ---------------------------------------------------------
        # CATBOOST CATEGORICAL INJECTION
        # ---------------------------------------------------------
        fit_params = {}
        if model_algorithm.lower() == 'catboost':
            # Dynamically find columns that are object or category dtypes
            cat_features = list(X_adj.select_dtypes(include=['object', 'category']).columns)
            fit_params['cat_features'] = cat_features
            
        # Unpack fit_params into the fit method. 
        # For RF, fit_params is empty. For CatBoost, it contains our categorical column names.
        model_adj = ModelClass(**final_params).fit(X_tr, y_tr, **fit_params)
        
        adj_imp = model_adj.feature_importances_ if hasattr(model_adj, 'feature_importances_') else model_adj.get_feature_importance() # type: ignore
        feat_df = pd.DataFrame({'feat': X_adj.columns, 'imp': adj_imp})
        batch_f = feat_df[feat_df['feat'].isin(batch_covariates.columns)]['imp'].sum() / feat_df['imp'].sum()
        
        results['models']['covariate_adjusted'] = {
            'test_score': float(eval_func(y_te, model_adj.predict(X_te))), 
            'batch_importance_fraction': float(batch_f)
        }

    # A3: Stratified Residuals
    if batch_config.get('stratified_prediction', {}).get('enabled', False):
        _, residuals = train_batch_residual_model(
            X_taxa, batch_covariates, y, X_test.index, task_type, model_algorithm=model_algorithm
        )
        ResidModel = CatBoostRegressor if model_algorithm.lower() == 'catboost' else RandomForestRegressor
        X_r_tr, X_r_te, r_tr, r_te = train_test_split(X_taxa.fillna(0), residuals, test_size=0.3, random_state=42)
        m_resid = ResidModel(n_estimators=50, max_depth=5, verbose=False).fit(X_r_tr, r_tr)
        results['models']['stratified'] = {'residual_model_score': float(r2_score(r_te, m_resid.predict(X_r_te)))}

    # Final Audit
    conf_df, exclusions = audit_biomarker_confidence(X_taxa, batch_covariates, top_taxa)
    results['confidence_report'] = conf_df.to_dict(orient='records')
    results['exclusions'] = exclusions

    create_confounding_heatmap(X_taxa, batch_covariates, top_taxa, plot_dir, target_col, level)
    
    algo_dir = plot_dir / model_algorithm
    algo_dir.mkdir(exist_ok=True, parents=True)
    import re
    safe_target_name = re.sub(r'\W+', '', target_col)
    filename = f"results_{safe_target_name}.json"

    with open(algo_dir / filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results

@with_logger
def create_summary_report(
    all_results: Dict[str, Dict], plot_dir: Path, level: str
):
    """
    Generates a Markdown summary report synthesizing results across all ML targets.
    This provides the final 'biological vs technical' verdict.
    """
    logger = get_logger("workflow_16s")
    report_path = plot_dir / f"batch_control_summary_{level}.md"
    with open(report_path, 'w') as f:
        f.write(f"# Batch Covariate Control - ML Discovery Summary\n\n")
        f.write(f"**Taxonomic Level:** {level}  \n**Audit Date:** {datetime.now().strftime('%Y-%m-%d')}  \n\n")
        
        f.write("## 1. Performance Summary\n")
        f.write("| Target | Baseline Score | Adj. Score | Batch Var % | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        for target, res in all_results.items():
            base = res['models']['baseline']['test_score']
            adj = res['models'].get('covariate_adjusted', {}).get('test_score', 'N/A')
            b_var = res['models'].get('covariate_adjusted', {}).get('batch_importance_fraction', 0)
            
            # Simple Status Logic
            status = "🟢 Robust" if b_var < 0.3 else "🟡 Contaminated" if b_var < 0.6 else "🔴 Technical Artifact"
            
            adj_str = f"{adj:.3f}" if isinstance(adj, float) else adj
            f.write(f"| {target} | {base:.3f} | {adj_str} | {b_var:.1%} | {status} |\n")

        f.write("\n## 2. Biomarker Confidence Audit (Top Taxa)\n")
        f.write("Identifies taxa that are potentially just proxies for technical variables.\n\n")
        
        for target, res in all_results.items():
            f.write(f"### Target: {target}\n")
            f.write("| Taxon | Confidence Score | Status | Technical Links |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for entry in res['confidence_report'][:10]: # Top 10 for brevity
                f.write(f"| {entry['taxon'].split('__')[-1]} | {entry['score']} | {entry['status']} | {entry['technical_links']} |\n")
            f.write("\n")
    logger.info(f"[✔] Summary report created at {report_path}")

def train_batch_residual_model(
    X_taxa: pd.DataFrame, batch_covs: pd.DataFrame, y: pd.Series, 
    test_idx: pd.Index, task: str, model_algorithm: str = 'rf'
) -> Tuple[Union[RandomForestClassifier, RandomForestRegressor, CatBoostClassifier, CatBoostRegressor], pd.Series]:
    """
    Trains a model on technical batch covariates and returns the residuals.
    For classification, returns probability residuals.
    Supports native categorical handling if model_algorithm is 'catboost'.
    """
    train_idx = X_taxa.index.difference(test_idx)
    
    # 1. Dynamically handle algorithm and categorical features
    fit_params = {}
    is_catboost = model_algorithm.lower() == 'catboost'
    
    if is_catboost:
        # Detect categorical columns
        cat_features = list(batch_covs.select_dtypes(include=['object', 'category']).columns)
        if cat_features:
            fit_params['cat_features'] = cat_features
            
        # Protect string/categorical columns from fillna(0)
        X_batch_train = batch_covs.loc[train_idx].copy()
        X_batch_all = batch_covs.copy()
        
        num_cols = X_batch_train.select_dtypes(include='number').columns
        X_batch_train[num_cols] = X_batch_train[num_cols].fillna(0)
        X_batch_all[num_cols] = X_batch_all[num_cols].fillna(0)
    else:
        # Standard Random Forest preparation
        X_batch_train = batch_covs.loc[train_idx].fillna(0)
        X_batch_all = batch_covs.fillna(0)

    # 2. Train Regression or Classification
    if task.lower() == 'regression':
        if is_catboost:
            model = CatBoostRegressor(iterations=50, depth=6, verbose=False)
        else:
            model = RandomForestRegressor(n_estimators=50, max_depth=8)
            
        model.fit(X_batch_train, y.loc[train_idx], **fit_params)
        
        # Standard regression residuals
        residuals = y - model.predict(X_batch_all)
        
    else:
        # Map labels to 0/1 to allow mathematical subtraction for residuals
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y), index=y.index)
        
        if is_catboost:
            model = CatBoostClassifier(iterations=50, depth=6, verbose=False)
        else:
            model = RandomForestClassifier(n_estimators=50, max_depth=8)
            
        model.fit(X_batch_train, y_encoded.loc[train_idx], **fit_params)
        
        # Extract probabilities for the positive class
        probas = model.predict_proba(X_batch_all)
        p_positive = probas[:, 1] if probas.shape[1] == 2 else probas[:, 0]
        
        # Probability residuals: True Label (0 or 1) - Predicted Probability (0.0 to 1.0)
        residuals = y_encoded - p_positive
        
    return model, residuals
