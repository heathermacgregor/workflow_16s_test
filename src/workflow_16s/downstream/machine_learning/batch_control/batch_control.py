# src/workflow_16s/downstream/machine_learning/batch_control/batch_control.py

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRegressor 
from datetime import datetime

from workflow_16s.utils.logger import get_logger

# --- SECTION 1: UTILITIES ---

def sanitize_catboost_params(
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Clean parameters to prevent 'n_estimators' vs 'iterations' conflicts."""
    clean = params.copy()
    aliases = {'n_estimators', 'num_boost_round', 'num_trees', 'iterations'}
    keys_present = [k for k in clean.keys() if k in aliases]
    if len(keys_present) > 1:
        primary = 'iterations' if 'iterations' in keys_present else keys_present[0]
        for k in keys_present:
            if k != primary: clean.pop(k, None)
    return clean

def get_model_class(
    task_type: str, 
    algorithm: str
) -> type:
    """Factory to return the correct model class based on task and algorithm."""
    if algorithm.lower() == 'catboost':
        return CatBoostRegressor if task_type.lower() == 'regression' else CatBoostClassifier
    return RandomForestRegressor if task_type.lower() == 'regression' else RandomForestClassifier

# --- SECTION 2: AUDIT & CONFIDENCE ENGINES ---

def audit_biomarker_confidence(
    X_taxa: pd.DataFrame, 
    batch_covs: pd.DataFrame, 
    top_taxa: List[str], 
    rho_limit: float = 0.8
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Ranks biomarkers by how 'clean' they are from technical noise.
    Score starts at 100; -20 points for every significant technical correlation.
    """
    report, exclusions = [], []
    for taxon in top_taxa:
        score, links = 100, []
        is_leaky = False
        for var in batch_covs.columns:
            # Handle encoding for categorical metadata
            b_vals = batch_covs[var].astype('category').cat.codes if batch_covs[var].dtype == 'object' else batch_covs[var]
            rho, p = spearmanr(X_taxa[taxon], b_vals, nan_policy='omit')
            
            if isinstance(p, float) and p < 0.05:
                score -= 20
                links.append(f"{var} (ρ={rho:.2f})")
                if isinstance(rho, float) and abs(rho) >= rho_limit:
                    is_leaky = True
                    exclusions.append({'taxon': taxon, 'var': var, 'rho': float(rho)})
        
        report.append({
            'taxon': taxon, 
            'score': max(0, score), 
            'technical_links': "; ".join(links),
            'status': 'REJECTED' if is_leaky else 'PASSED'
        })
    return pd.DataFrame(report), exclusions

# --- SECTION 3: CORE EXECUTION ---

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
    model_params: Optional[Dict[str, Any]] = None
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
        X_adj = pd.concat([X_taxa, batch_covariates], axis=1).fillna(0)
        X_tr, X_te, y_tr, y_te = train_test_split(X_adj, y, test_size=0.3, random_state=42, stratify=stratify_opt)
        model_adj = ModelClass(**final_params).fit(X_tr, y_tr)
        
        adj_imp = model_adj.feature_importances_ if hasattr(model_adj, 'feature_importances_') else model_adj.get_feature_importance() # type: ignore
        feat_df = pd.DataFrame({'feat': X_adj.columns, 'imp': adj_imp})
        batch_f = feat_df[feat_df['feat'].isin(batch_covariates.columns)]['imp'].sum() / feat_df['imp'].sum()
        results['models']['covariate_adjusted'] = {'test_score': float(eval_func(y_te, model_adj.predict(X_te))), 'batch_importance_fraction': float(batch_f)}

    # A3: Stratified Residuals
    if batch_config.get('stratified_prediction', {}).get('enabled', False):
        _, residuals = train_batch_residual_model(X_taxa, batch_covariates, y, X_test.index, task_type)
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

# --- SECTION 4: VISUALIZATION & REPORTING ---

def create_summary_report(
    all_results: Dict[str, Dict], 
    plot_dir: Path, 
    level: str
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

def create_confounding_heatmap(
    X_taxa, 
    batch_covariates, 
    top_taxa, 
    plot_dir, 
    target_name, 
    level
):
    """Significance-masked heatmap (p < 0.05)."""
    logger = get_logger("workflow_16s")
    taxa_subset = X_taxa[top_taxa]
    corr_matrix = pd.DataFrame(index=top_taxa, columns=batch_covariates.columns)
    p_matrix = pd.DataFrame(index=top_taxa, columns=batch_covariates.columns)

    for taxon in top_taxa:
        for b_var in batch_covariates.columns:
            b_vals = batch_covariates[b_var].astype('category').cat.codes if batch_covariates[b_var].dtype == 'object' else batch_covariates[b_var]
            rho, p = spearmanr(taxa_subset[taxon], b_vals, nan_policy='omit')
            corr_matrix.loc[taxon, b_var] = rho # type: ignore
            p_matrix.loc[taxon, b_var] = p # type: ignore

    masked_corr = corr_matrix.astype(float).where(p_matrix.astype(float) < 0.05, np.nan)
    fig = px.imshow(
        masked_corr,
        labels=dict(x="Batch Variable", y="Microbial Taxon", color="Significant ρ"),
        x=batch_covariates.columns, y=[t.split('__')[-1] for t in top_taxa],
        color_continuous_scale='RdBu_r', range_color=[-1, 1],
        title=f"Confounding Diagnostic (p < 0.05): {target_name}"
    )
    safe_target = re.sub(r'\W+', '', target_name)
    fig.write_html(str(plot_dir / f"confounding_heatmap_{safe_target}.html"))
    logger.info(f"[✔] Confounding heatmap saved for target '{target_name}' at {plot_dir / f'confounding_heatmap_{safe_target}.html'}")

def train_batch_residual_model(
    X_taxa, 
    batch_covs, 
    y, 
    test_idx, 
    task
):
    train_idx = X_taxa.index.difference(test_idx)
    M = RandomForestRegressor if task == 'regression' else RandomForestClassifier
    model = M(n_estimators=50, max_depth=8).fit(batch_covs.loc[train_idx].fillna(0), y.loc[train_idx])
    residuals = y - model.predict(batch_covs.fillna(0)) if task == 'regression' else (y != model.predict(batch_covs.fillna(0))).astype(float)
    return model, residuals

def create_comparison_plots(
    all_results: Dict[str, Dict], 
    plot_dir: Path, 
    level: str
):
    logger = get_logger("workflow_16s")
    targets = list(all_results.keys())
    baseline_s = [all_results[t]['models']['baseline']['test_score'] for t in targets]
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Model Accuracy", "Batch Contribution"))
    fig.add_trace(go.Bar(name='Baseline', x=targets, y=baseline_s), row=1, col=1)
    fig.write_html(str(plot_dir / f"batch_control_comparison_{level}.html"))
    logger.info(f"[✔] Comparison plots saved to {plot_dir / f'batch_control_comparison_{level}.html'}")