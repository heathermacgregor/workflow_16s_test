from workflow_16s.utils.analysis import AnalysisUtils
# workflow_16s/modules/machine_learning/catboost/workflows/feature_selection.py
"""
🔬 16S Machine Learning Discovery Architecture
Tiered Forensic Integrity Suite
"""

import json
import warnings
import time
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, ElasticNetCV
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

from workflow_16s.utils.analysis import AnalysisUtils
from workflow_16s.utils.logger import with_logger
from workflow_16s.utils.pandas import filter_by_prevalence
from workflow_16s.utils.logger import get_logger
logger = get_logger("workflow_16s")



from workflow_16s.modules.machine_learning.catboost.batch_control import (
    audit_biomarker_confidence, 
    create_confounding_heatmap,
    prepare_batch_covariates, 
    calculate_batch_importance
)
from workflow_16s.modules.machine_learning.catboost.constants import MANDATORY_METADATA
from workflow_16s.modules.machine_learning.catboost.feature_selection import catboost_feature_selection
from workflow_16s.modules.machine_learning.catboost.utils import (
    align_data_robust,
    apply_batch_centered_clr,
    clean_feature_names, 
    resolve_feature_names,  
    verify_model_outputs
)
from workflow_16s.modules.machine_learning.catboost.validation import (
    StudyEligibilityManager,
    run_comprehensive_validation,
    run_shuffle_baseline, 
    validate_consensus_panel,
    BiomarkerAuditor, 
    verify_run
)

from workflow_16s.visualization.machine_learning.batch_dependency import plot_batch_dependency
from workflow_16s.visualization.machine_learning.features import generate_comprehensive_ml_report

from workflow_16s.downstream.stats.batch_correction import conqur_batch_correction as apply_conqur_correction
from workflow_16s.downstream.utils.helpers import AnalysisUtils
from .meta_analysis import (
    perform_meta_analysis, 
    apply_meta_consensus_weighting
)

def _generate_optuna_reports(study: Any, out_dir: Path, prefix: str):
    """Generates visual forensic reports for the Optuna Bayesian Search."""
    if study is None:
        logger.debug(f"       ⚠️ [OPTUNA AUDIT] No study object provided for {prefix}. Skipping plots.")
        return
        
    try:
        import optuna.visualization as vis
        out_dir.mkdir(parents=True, exist_ok=True)
        
        fig1 = vis.plot_optimization_history(study)
        fig1.write_html(str(out_dir / f"{prefix}_optuna_history.html"), include_plotlyjs="cdn")
        
        fig2 = vis.plot_parallel_coordinate(study)
        fig2.write_html(str(out_dir / f"{prefix}_optuna_parallel.html"), include_plotlyjs="cdn")
        
        fig3 = vis.plot_param_importances(study)
        fig3.write_html(str(out_dir / f"{prefix}_optuna_param_importances.html"), include_plotlyjs="cdn")
        
        logger.info(f"       📊 [OPTUNA AUDIT] Generated Bayesian search plots in: {out_dir.name}/")
    except ImportError:
        logger.warning("       ⚠️ [OPTUNA AUDIT] 'optuna' or 'plotly' not installed. Cannot generate visualization.")
    except Exception as e:
        logger.warning(f"       ⚠️ [OPTUNA AUDIT] Failed to generate Optuna plots: {e}")

@with_logger
def run_catboost_selection(
    adata: Any, 
    catboost_output_dir: Union[str, Path], 
    level: str = 'Genus', 
    priority_targets: Optional[List[str]] = None, 
    strict_targets: bool = False,
    eligibility_mode: str = 'audit', 
    strategies: Optional[List[str]] = None, 
    test_size: float = 0.3, 
    random_state: int = 42, 
    num_features: int = 50,  
    n_top_final: int = 20,   
    method: Literal['rfe', 'shap', 'select_k_best'] = 'shap', 
    use_permutation: bool = False, 
    n_cpus: int = 4,
    batch_col: str = 'batch_original',
    meta_cols: Optional[List[str]] = None,  
    X_custom: Optional[pd.DataFrame] = None,
    param_grid: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    
    global_start_time = time.time()
    logger.info(f" 🎬 [AUDIT] Starting Forensic Discovery Engine ({level})")
    
    # --- ROUTE OPTUNA TO FILE LOGGER ---
    import optuna
    optuna.logging.set_verbosity(optuna.logging.INFO)
    optuna_logger = optuna.logging.get_logger("optuna")
    for handler in logger.handlers:
        if handler not in optuna_logger.handlers:
            optuna_logger.addHandler(handler)
    # -----------------------------------
    
    out_dir = Path(catboost_output_dir)
    import importlib
    utils_module = importlib.import_module('workflow_16s.utils.analysis')
    
    logger.info(f" 📊 [AUDIT] Initial AnnData check. Extracting level: {level}")
    adata_agg = utils_module.AnalysisUtils.get_analysis_adata(adata, level=level)

    if adata_agg is None or adata_agg.n_obs < 20: 
        logger.error(f"❌ [AUDIT] FATAL: Insufficient data for ML.")
        return []
    
    
    if adata_agg.n_obs > 15000:
        logger.warning(f" ⚠️ [AUDIT] CRITICAL: Matrix bypassed subsampling and has {adata_agg.n_obs} samples! Forcing downsample to 10k to prevent node death.")
        import numpy as np
        np.random.seed(42)
        keep_idx = np.random.choice(adata_agg.obs_names, 10000, replace=False)
        adata_agg = adata_agg[keep_idx].copy()

    logger.info(f" ✅ [AUDIT] Aggregation successful. Working with {adata_agg.n_obs} samples and {adata_agg.n_vars} features.")
        
    if X_custom is None:
        if scipy.sparse.issparse(adata_agg.X):
            raw_counts = np.asarray(adata_agg.X.toarray())
        elif hasattr(adata_agg.X, 'toarray'):
            raw_counts = np.asarray(adata_agg.X.toarray())
        else:
            raw_counts = np.asarray(adata_agg.X) if isinstance(adata_agg.X, np.ndarray) else np.array(adata_agg.X)
        
        feature_names = resolve_feature_names(adata_agg, level)
        raw_counts_df = pd.DataFrame(raw_counts, index=adata_agg.obs_names, columns=feature_names)
        
        # --- 1% PREVALENCE FILTER ---
        prevalence = (raw_counts_df > 0).mean(axis=0)
        keep_cols = prevalence[prevalence >= 0.01].index
        logger.info(f" ✂️ [AUDIT] Applying 1% prevalence filter: Dropping {len(prevalence) - len(keep_cols)} rare features.")
        raw_counts_df = raw_counts_df[keep_cols]
        
        # --- UNCLASSIFIED FEATURE BAN ---
        drop_keywords = ['Unclassified', 'uncultured', 'unknown', 'incertae_sedis', 'ambiguous']
        cols_to_drop = [col for col in raw_counts_df.columns if any(k.lower() in str(col).lower() for k in drop_keywords)]
        raw_counts_df = raw_counts_df.drop(columns=cols_to_drop)
        logger.info(f" 🛑 [AUDIT] Dropped {len(cols_to_drop)} unclassified features. Remaining: {raw_counts_df.shape[1]}")
            
        X_df = utils_module.AnalysisUtils.clr_transform_from_df(raw_counts_df, pseudocount=1.0)
        adata_agg = adata_agg[:, X_df.columns].copy()
        adata_agg.X = X_df.values
    else:
        X_df = X_custom.copy()
        adata_agg.X = X_df.values
        
    X_df = clean_feature_names(X_df)
    adata_agg.var_names = X_df.columns
    
    valid_targets = _generate_facility_targets(adata, priority_targets, strict_targets)
    strategies_to_run = ['baseline', 'lopocv']

    all_results = []
    
    for target_col in valid_targets:
        target_start_time = time.time()
        logger.info(f" {'='*50}")
        logger.info(f" 📍 [AUDIT] Commencing Pipeline for Target: '{target_col}'")
        
        eligibility = StudyEligibilityManager(adata_agg, target_col=target_col)
        eligibility.diagnose_studies()
        adata_working = adata_agg
        
        if eligibility_mode == 'filter':
            adata_working = eligibility.get_filtered_adata()
            if len(adata_working) < 20: continue

        meta_weight_map = {}
        modeling_strats = [s for s in strategies_to_run if s != 'meta_analysis']
        
        for strategy in modeling_strats:
            strat_start_time = time.time()
            logger.info(f"   ► [AUDIT] Initiating Strategy: {strategy.upper()}")
            target_out = out_dir / strategy / f"{level}_{target_col}"
            target_out.mkdir(exist_ok=True, parents=True)

            X_final = X_df.copy()
            X_final.index = adata_working.obs.index
            meta_final = adata_working.obs.copy()
            y_final = meta_final[target_col].copy()

            is_regression = any(k in target_col.lower() for k in ['ph', 'temp', 'altitude', 'depth', 'precip'])
            if not is_regression:
                counts = y_final.value_counts()
                valid_classes = counts[counts >= 10].index
                mask = y_final.isin(valid_classes)
                X_final, y_final, meta_final = X_final[mask], y_final[mask], meta_final[mask]
                
            X_final = X_final.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # --- DYNAMIC BATCH COLUMN SELECTION ---
            potential_batches = [batch_col, 'Project', 'study_accession', 'project', 'batch', 'Study']
            current_batch_col = 'unknown_batch'
            for col in potential_batches:
                if col in adata_working.obs.columns:
                    n_unique = adata_working.obs[col].nunique()
                    fill_rate = adata_working.obs[col].notna().mean()
                    if n_unique > 1 and fill_rate > 0.5:
                        current_batch_col = col
                        logger.info(f"     [AUDIT] Dynamic Batch Selector picked '{col}' (Fill rate: {fill_rate:.1%}, Groups: {n_unique})")
                        break
            
            if current_batch_col in meta_final.columns:
                meta_final[current_batch_col] = meta_final[current_batch_col].astype(object).fillna("unknown").astype(str)

            # CV SETUP (GroupKFold logic is inside this function now)
            cv_groups, cv_strat = _setup_cv_strategy(strategy, meta_final, current_batch_col)
            # Wiping out groups for baseline is REMOVED so it correctly uses GroupKFold!
            
            task_type = 'Regression' if is_regression else 'Classification'
            
            try:
                # =========================================================
                # PHASE 1: ENSEMBLE DISCOVERY
                # =========================================================
                logger.info(f"     🔍 [AUDIT] [Phase 01] Launching Triple Consensus Engine...")
                
                # ADDED 'verbose': [50] SO CATBOOST LOGS EVERY 50 TREES
                cb_grid = param_grid or {'depth': [4, 6], 'learning_rate': [0.05, 0.1], 'l2_leaf_reg': [3, 5]}
                
                logger.info("       > Triggering CatBoost & Optuna Bayesian Search...")
                cb_res = catboost_feature_selection(
                    metadata=meta_final, features=X_final, output_dir=target_out / "discovery_catboost",
                    group_col=target_col, cv_groups=cv_groups, cv_strategy=cv_strat,
                    method=method, num_features=num_features, n_top_final=num_features, 
                    task_type=task_type, random_state=42, thread_count=n_cpus, param_grid=cb_grid
                )
                
                cb_top = set(cb_res.get('top_features', []))
                logger.info(f"       ✅ [OPTUNA AUDIT] CatBoost Search Finished!")
                logger.info(f"       > CatBoost identified {len(cb_top)} features: {list(cb_top)}")
                
                if 'optuna_study' in cb_res:
                    _generate_optuna_reports(cb_res['optuna_study'], target_out / "discovery_catboost", "discovery")

                logger.info("       > Triggering Elastic Net (Linear) Scan...")
                linear_top = _run_linear_discovery(X_final, y_final, task_type, num_features)
                
                logger.info("       > Triggering Random Forest Scan...")
                rf_top = _run_rf_discovery(X_final, y_final, task_type, num_features)
                
                intersection = list(cb_top.intersection(linear_top).intersection(rf_top))
                if len(intersection) >= 5:
                    final_candidates = intersection[:n_top_final]
                    logger.info(f"     ✅ [AUDIT] STRICT CONSENSUS ACHIEVED! Features: {final_candidates}")
                else:
                    final_candidates = list(cb_top.intersection(rf_top))[:n_top_final] or list(cb_top)[:n_top_final]
                    logger.info(f"     ⚠️ [AUDIT] Relaxed Consensus Features: {final_candidates}")

                # =========================================================
                # PHASE 2: REFINEMENT
                # =========================================================
                logger.info(f"     🛠️ [AUDIT] [Phase 02] Training Final Refinement Model...")
                X_refined = X_final[final_candidates].copy()
                refinement_grid = {'depth': [4], 'learning_rate': [0.05], 'l2_leaf_reg': [3], 'iterations': [500]}
                
                final_res = catboost_feature_selection(
                    metadata=meta_final, features=X_refined, output_dir=target_out / "refined_model",
                    group_col=target_col, cv_groups=cv_groups, cv_strategy=cv_strat,
                    num_features=len(final_candidates), n_top_final=len(final_candidates),
                    method='shap', task_type=task_type, random_state=42, 
                    thread_count=n_cpus, param_grid=refinement_grid
                )
                
                logger.info(f"       ✅ [AUDIT] Refinement complete. Final best CV score: {final_res.get('best_score', 'UNKNOWN')}")
                
                # =========================================================
                # PHASE 3: FORENSIC AUDITS
                # =========================================================
                run_comprehensive_validation(
                    final_res['model'], X_refined, y_final, output_dir=target_out / "audit", 
                    target_name=target_col, groups=cv_groups.values if cv_groups is not None else None, # type: ignore
                    task_type=task_type
                )
                
                final_res['strategy'] = strategy
                all_results.append(final_res)

            except Exception as e:
                logger.error(f"     ❌ [AUDIT] CRITICAL PIPELINE FAILURE in Strategy '{strategy}': {e}", exc_info=True)

    return all_results

def _run_linear_discovery(X: pd.DataFrame, y: pd.Series, task_type: str, n_top: int = 50) -> set:
    try:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns).fillna(0)
        if task_type == 'Classification':
            model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.7, C=1.0, max_iter=200, class_weight='balanced', n_jobs=-1, random_state=42)
        else:
            model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99], cv=3, max_iter=200, n_jobs=-1, random_state=42)
        model.fit(X_scaled, y)
        coefs = np.mean(np.abs(model.coef_), axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_.flatten())
        top_idx = np.argsort(coefs)[::-1][:n_top]
        return set(X.columns[top_idx])
    except:
        return set()

def _run_rf_discovery(X: pd.DataFrame, y: pd.Series, task_type: str, n_top: int = 50) -> set:
    try:
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1) if task_type == 'Classification' else RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        return set(X.columns[np.argsort(model.feature_importances_)[::-1][:n_top]])
    except:
        return set()

def _generate_facility_targets(adata: Any, priority_targets: Optional[List[str]], strict_targets: bool = False) -> List[str]:
    targets = priority_targets if priority_targets else ['Env_Level_1', 'Env_Level_2']
    valid_targets = [t for t in targets if t in adata.obs.columns]
    return valid_targets

def _setup_cv_strategy(strategy: str, meta: pd.DataFrame, batch_col: str) -> tuple[Optional[pd.Series], Optional[Any]]:
    logger = get_logger("workflow_16s")
    cv_groups, cv_strategy_obj = None, None
    
    if batch_col in meta.columns and batch_col != 'unknown_batch':
        groups = meta[batch_col].astype(str)
        n_groups = groups.nunique()
        
        if n_groups >= 2:
            cv_groups = groups
            max_folds = 5
            if n_groups > max_folds:
                cv_strategy_obj = GroupKFold(n_splits=max_folds)
                logger.info(f" 🗒️ [AUDIT] CV Engine configured for GroupKFold using '{batch_col}' ({max_folds} folds across {n_groups} groups).")
            else:
                cv_strategy_obj = LeaveOneGroupOut()
                logger.info(f" 🗒️ [AUDIT] CV Engine configured for Leave-One-Group-Out using '{batch_col}'.")
                
    return cv_groups, cv_strategy_obj
