# ===================================== IMPORTS ====================================== #

import logging
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-Party Imports
import pandas as pd
import anndata as ad
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Local Imports
from workflow_16s.logger import get_logger
from workflow_16s.utils.progress import get_progress_bar

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = get_logger()

# =================================== CLASS ====================================== #
        
class FeatureSelection:
    """Performs machine learning feature selection using scikit-learn."""

    def __init__(self, config: Dict, adata: ad.AnnData):
        self.config = config
        self.adata = adata
        self.ml_config = self.config.get('ml', {})
        
        if not self.ml_config.get('enabled', True):
            self.target_columns, self.tasks = [], []
            return
            
        self.n_threads = self.config.get('threads', 8)
        self.verbose = self.config.get('verbose', False)
        
        primary_targets = self.ml_config.get('target_columns', [])
        auto_targets = self.adata.uns.get('analysis_columns', {}).get("group_comparison", []) if self.ml_config.get('analyze_all_valid_columns', True) else []
        self.target_columns = sorted(list(set(primary_targets + auto_targets)))
        
        self.tasks = self._get_enabled_tasks()
        if not self.tasks: logger.info("No ML feature selection tasks are enabled based on config.")

    def _get_enabled_tasks(self) -> List[Tuple[str, str]]:
        """Parses the config to determine which ML tasks to run."""
        all_tasks = []
        for layer_name, layer_conf in self.ml_config.get('tables', {}).items():
            if layer_conf.get('enabled', False) and layer_name in self.adata.layers:
                # Method is now handled internally, so we just need the layer
                all_tasks.append((layer_name, layer_conf.get('method', 'rfe')))
        return all_tasks
      
    def run(self, output_dir: Path) -> ad.AnnData:
        """Executes all feature selection tasks for each configured target column."""
        if not self.tasks or not self.target_columns:
            logger.warning("No tasks or target columns to run for feature selection.")
            return self.adata

        logger.info("STEP 7: Running machine learning feature selection...")
        for target_col in self.target_columns:
            if target_col not in self.adata.obs.columns or self.adata.obs[target_col].nunique() < 2:
                logger.warning(f"Target column '{target_col}' is invalid or has < 2 classes; skipping.")
                continue
            self._run_for_target(target_col, output_dir)
        return self.adata
            
    def _run_for_target(self, target_col: str, base_output_dir: Path):
        """Processes all tasks for a single target column."""
        logger.info(f"--- Running Feature Selection for Target: '{target_col}' ---")
        for task in self.tasks:
            self._process_task(task, target_col, base_output_dir)
            
    def _process_task(self, task: Tuple[str, str], target_col: str, output_dir: Path):
        """Processes a single feature selection task."""
        layer, method = task
        logger.info(f"Running {method.upper()} on '{layer}' layer for target '{target_col}'...")
        
        output_subdir = output_dir / 'ml' / target_col / layer / method
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        try:
            results = self._run_rfe_analysis(layer, target_col)
            
            if results:
                # Store serializable results in the AnnData object
                ml_results = self.adata.uns.setdefault('ml', {})
                target_results = ml_results.setdefault(target_col, {})
                layer_results = target_results.setdefault(layer, {})
                layer_results[method] = results

        except Exception as e:
            logger.error(f"ML task failed for {target_col} on {layer}: {e}", exc_info=self.verbose)

    def _run_rfe_analysis(self, layer: str, target_col: str) -> Optional[Dict[str, Any]]:
        """Runs a new RFE analysis using scikit-learn and returns the results."""
        X_matrix = self.adata.layers[layer]
        X_array = X_matrix.toarray() if hasattr(X_matrix, 'toarray') else X_matrix
        X = pd.DataFrame(X_array, index=self.adata.obs_names, columns=self.adata.var_names)
        y = self.adata.obs[target_col]
        
        # 1. Split data for training and final evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # 2. Set up the RFE with Cross-Validation
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=self.n_threads)
        cv = StratifiedKFold(3)
        
        # RFECV will find the optimal number of features
        selector = RFECV(estimator=estimator, step=0.10, cv=cv, scoring='accuracy', min_features_to_select=5)
        
        logger.info(f"  - Performing RFECV for '{target_col}'... (this may take a moment)")
        selector.fit(X_train, y_train)
        
        # 3. Get the results of the selection
        selected_features = X_train.columns[selector.support_].tolist()
        logger.info(f"  - RFECV selected {len(selected_features)} optimal features.")
        
        feature_rankings = pd.DataFrame({
            'feature': X_train.columns,
            'ranking': selector.ranking_
        }).sort_values('ranking')

        # 4. Train a final, more powerful model on only the selected features
        final_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=self.n_threads)
        final_model.fit(X_train[selected_features], y_train)
        
        # 5. Evaluate on the held-out test set
        y_pred = final_model.predict(X_test[selected_features])
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        logger.info(f"  - Final model accuracy on test set: {accuracy:.3f}")
        
        # 6. Return a dictionary of serializable results
        return {
            'accuracy': accuracy,
            'f1_score_weighted': f1,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'n_features_selected': len(selected_features),
            'selected_features': feature_rankings[feature_rankings['ranking'] == 1]
        }