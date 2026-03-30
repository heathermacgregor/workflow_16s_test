# src/workflow_16s/downstream/machine_learning/workflows/standard_ml.py

from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from workflow_16s.config.config_schema import MLConfig
from workflow_16s.modules.machine_learning.catboost.batch_control import (
    prepare_batch_covariates, run_ml_with_batch_control
)
from workflow_16s.modules.machine_learning.catboost.constants import EXPECTED_VAR_COLUMNS
from workflow_16s.modules.machine_learning.catboost.utils import (
    align_data_robust, clean_feature_names, resolve_feature_names
)
from workflow_16s.modules.machine_learning.catboost.validation import run_comprehensive_validation
from workflow_16s.utils.analysis import AnalysisUtils
from workflow_16s.utils.logger import with_logger


@with_logger
def run_machine_learning_analysis(
    adata, 
    plot_dir_ml: Path, 
    level: str = 'Genus', 
    min_samples_per_group: int = 10, 
    max_classes: int = 10, 
    priority_targets: Optional[List[str]] = None, 
    strict_targets: bool = False, 
    validate_overfitting: bool = True, 
    quick_validation: bool = False,
    batch_config: Optional[Dict[str, Any]] = None,
    ml_config: Optional[MLConfig] = None, 
    X_custom: Optional[pd.DataFrame] = None
):
    logger.info(f"--- Starting Standard Machine Learning Analysis ({level}) ---")
    
    if X_custom is not None:
        X_df = X_custom.copy()
        X_df.index = X_df.index.astype(str)
        adata.obs.index = adata.obs.index.astype(str)
        X_df = clean_feature_names(X_df)
    else:
        adata_agg = AnalysisUtils.get_analysis_adata(adata, level=level)
        if adata_agg is None or adata_agg.n_obs < 20: 
            logger.warning(f"Skipping ML: Insufficient data at {level}"); return
        X_df = AnalysisUtils.clr_transform(adata_agg, pseudocount=1)
        
        new_names = resolve_feature_names(adata_agg, level)
        new_names = pd.io.common.dedup_names(new_names, is_potential_multiindex=False)
        X_df.columns = new_names
        
        X_df.index = X_df.index.astype(str)
        adata.obs.index = adata.obs.index.astype(str)
        X_df = clean_feature_names(X_df)

    if strict_targets:
        valid_targets = [t for t in (priority_targets or []) if t in adata.obs.columns]
    else:
        candidate_cols = priority_targets if priority_targets else adata.obs.columns
        valid_targets = [t for t in candidate_cols if t in adata.obs.columns and t not in EXPECTED_VAR_COLUMNS]

    algorithms_to_run = []
    if ml_config and ml_config.models:
        if ml_config.models.enable_random_forest: algorithms_to_run.append('rf')
        if ml_config.models.enable_catboost: algorithms_to_run.append('catboost')
    else: algorithms_to_run = ['rf']

    batch_enabled = batch_config and batch_config.get('enabled', False)
    batch_covariates_df = None
    if batch_enabled and batch_config.get('covariate_columns'):
        try:
            batch_covariates_df, _ = prepare_batch_covariates(
                adata=adata, batch_columns=batch_config['covariate_columns'],
                sample_indices=X_df.index, one_hot_encode=True
            )
        except Exception: batch_enabled = False

    for target_col in valid_targets:
        logger.info(f"TARGET: {target_col}")
        X, y, meta = align_data_robust(X_df, adata.obs, target_col)
        
        if len(y) < 20:
            logger.warning(f"Skipping {target_col}: Not enough samples after cleaning ({len(y)})")
            continue
        
        is_numeric = pd.api.types.is_numeric_dtype(y)
        task_type = 'regression' if is_numeric and y.nunique() > max_classes else 'classification'

        for algo in algorithms_to_run:
            if batch_enabled and batch_covariates_df is not None:
                try:
                    run_ml_with_batch_control(
                        X_taxa=X, y=y, batch_covariates=batch_covariates_df.loc[y.index],
                        target_col=target_col, task_type=task_type, 
                        level=level, batch_config=batch_config, model_algorithm=algo
                    )
                except Exception as e: logger.error(f"Batch ML failed for {target_col}: {e}")
            else:
                try:
                    if algo == 'catboost':
                        ModelClass = CatBoostRegressor if task_type == 'regression' else CatBoostClassifier
                        model = ModelClass(verbose=False, allow_writing_files=False)
                    else:
                        ModelClass = RandomForestRegressor if task_type == 'regression' else RandomForestClassifier
                        model = ModelClass(n_jobs=-1)

                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
                    model.fit(X_tr, y_tr)
                    score = model.score(X_te, y_te)
                    logger.info(f"   ✓ {algo.upper()} Score: {score:.3f}")

                    if validate_overfitting:
                        run_comprehensive_validation(
                            model=model, X=X, y=y, output_dir=plot_dir_ml / algo,
                            target_name=target_col, task_type=task_type.capitalize(),
                            n_permutations=50 if quick_validation else 100,
                            quick_mode=quick_validation
                        )
                except Exception as e: logger.error(f"Standard ML failed: {e}")