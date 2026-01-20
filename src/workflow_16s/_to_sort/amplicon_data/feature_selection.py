# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import json
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
import catboost as cb
import pandas as pd
import numpy as np
from biom.table import Table

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s import constants
from workflow_16s.amplicon_data.helpers import _init_dict_level
from workflow_16s.models.feature_selection import (
    catboost_feature_selection,
)
from workflow_16s.utils.data import (
    table_to_df, update_table_and_meta
)
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")
debug_mode = False

# ================================= DEFAULT VALUES =================================== #

class FeatureSelection:
    """Performs feature selection"""
    def __init__(
        self, 
        config: Dict, 
        meta: pd.DataFrame,
        tables: Dict[str, Dict[str, Table]],
        verbose: bool = False
    ):
        self.config = config
        ml_config = self.config.get('ml', {})
        if not ml_config.get('enabled', False):
            logger.info("ML feature selection disabled")
            return
        self.group_column = config.get("group_column", constants.DEFAULT_GROUP_COLUMN)
        self.n_top_features = ml_config.get('num_features', 100)
        self.step_size = ml_config.get('step_size', 100)
        self.permutation_importance = ml_config.get('permutation_importance', {}).get('enabled', True)
        self.n_threads = ml_config.get('n_threads', 8)
        self.meta = meta
        self.tables = tables
        self.verbose = verbose
      
        self.models: Dict[str, Any] = {}
        
        if not ml_config.get('enabled', False):
            logger.info("ML feature selection disabled")
            self.tasks = []
            return
            
        self.tasks = self.get_enabled_tasks()  
        self.results = {}
        
        if len(self.tasks) == 0:
            logger.info("No methods for ML feature selection enabled")

    def get_enabled_tasks(self):        
        ml_config = self.config.get('ml', {})
        table_config = ml_config.get('tables', {})
        tasks = []
        
        for table_type, levels in self.tables.items():
            table_type_config = table_config.get(table_type, {})
            if not table_type_config.get('enabled', False):
                continue
                
            enabled_levels = [
                l for l in table_type_config.get('levels', levels.keys()) 
            ]
            enabled_methods = [
                m for m in table_type_config.get('methods', ["rfe"]) 
            ]
            
            for level in enabled_levels:
                for method in enabled_methods:  
                    tasks.append((table_type, level, method))  
        
        return tasks
      
    def run(
        self,
        output_dir: Optional[Path] = None,
    ) -> None:

        with get_progress_bar() as progress:
            cb_desc = "Running CatBoost feature selection"
            cb_task = progress.add_task(_format_task_desc(cb_desc), total=len(self.tasks))

            for table_type, level, method in self.tasks:
                method_desc = (
                    f"{table_type.replace('_', ' ').title()} ({level.title()})"
                    f" → {method.title()}"
                )
                progress.update(cb_task, description=_format_task_desc(method_desc))
    
                # Initialize data storage
                _init_dict_level(self.models, table_type, level)
                data_storage = self.models[table_type][level]
                # Initialize output directory and path
                tmp_output_dir = output_dir / 'ml' / self.group_column / table_type / level 
                tmp_output_dir.mkdir(parents=True, exist_ok=True)
    
                try:
                    if debug_mode:
                        time.sleep(3)
                        return
                    if table_type == "clr_transformed" and method == "chi_squared":
                        logger.warning(
                            "Skipping chi_squared feature selection for CLR data."
                        )
                        data_storage[method] = None
                    else:
                        # Define required files for this task
                        required_files = [
                            tmp_output_dir / method / "best_model.cbm",
                            tmp_output_dir / method / "feature_importances.csv",
                            tmp_output_dir / method / "grid_search_results.csv",
                            tmp_output_dir / method / "best_confusion_matrix.html",
                            tmp_output_dir / method / "best_roc_curve.html",
                            tmp_output_dir / method / "best_precision_recall_curve.html",
                            tmp_output_dir / method / "figs" / f"shap.summary.bar.{self.n_top_features}.html",
                            tmp_output_dir / method / "figs" / f"shap.summary.beeswarm.{self.n_top_features}.html",
                            tmp_output_dir / method / "figs" / f"shap.summary.heatmap.{self.n_top_features}.html",
                            tmp_output_dir / method / "figs" / f"shap.summary.force.{self.n_top_features}.html",
                        ]
                        
                        # Check if all files exist
                        all_files_exist = all(f.exists() for f in required_files)
                        
                        if all_files_exist:
                            logger.info(f"Loading existing results for {table_type}/{level}/{method}")
                            
                            # Load CatBoost model
                            model = cb.CatBoostClassifier()
                            model.load_model(str(tmp_output_dir / method / "best_model.cbm"))
                            
                            # Load feature importances
                            feature_importances = pd.read_csv(tmp_output_dir / method / "feature_importances.csv")
                            
                            # Load grid search results
                            grid_search_results = pd.read_csv(tmp_output_dir / method / "grid_search_results.csv")
                            
                            # Create figures dictionary with HTML content
                            figures = {
                                'confusion_matrix': (tmp_output_dir / method / "best_confusion_matrix.html").read_text(),
                                'roc': (tmp_output_dir / method / "best_roc_curve.html").read_text(),
                                'prc': (tmp_output_dir / method / "best_precision_recall_curve.html").read_text(),
                                'shap_summary_bar': (tmp_output_dir / method / "figs" / f"shap.summary.bar.{self.n_top_features}.html").read_text(),
                                'shap_summary_beeswarm': (tmp_output_dir / method / "figs" / f"shap.summary.beeswarm.{self.n_top_features}.html").read_text(),
                                'shap_summary_heatmap': (tmp_output_dir / method / "figs" / f"shap.summary.heatmap.{self.n_top_features}.html").read_text(),
                                'shap_summary_force': (tmp_output_dir / method / "figs" / f"shap.summary.force.{self.n_top_features}.html").read_text(),
                                'shap_dependency': None  # Placeholder
                            }
                            
                            # Create result dictionary
                            result = {
                                'model': model,
                                'feature_importances': feature_importances,
                                'grid_search_results': grid_search_results,
                                'figures': figures
                            }
                            
                            data_storage[method] = result
                        else:
                            logger.info(f"Running model for {table_type}/{level}/{method}")
                            # Run model normally if files are missing
                            table = self.tables[table_type][level]
                            X = table_to_df(table)
                            X.index = X.index.str.lower()
                            y = self.meta.set_index("#sampleid")[[self.group_column]]
                            y.index = y.index.astype(str).str.lower()
                            idx = X.index.intersection(y.index)
                            X, y = X.loc[idx], y.loc[idx]

                            use_permutation_importance = False if method == "select_k_best" else self.permutation_importance
                                        
                            model_result = catboost_feature_selection(
                                metadata=y,
                                features=X,
                                output_dir=tmp_output_dir / method,
                                group_col=self.group_column,
                                method=method,
                                n_top_features=self.n_top_features,
                                step_size=self.step_size,
                                use_permutation_importance=use_permutation_importance,
                                thread_count=self.n_threads,
                                progress=progress, 
                                task_id=cb_task,
                            )
                                    
                            # Log if no figures were generated
                            if not any(model_result['figures'].values()):
                                logger.warning(f"No figures generated for {table_type}/{level}/{method}")
                                    
                            data_storage[method] = model_result
                            
                            # Save lightweight results to JSON (without heavy objects)
                            result_to_save = model_result.copy()
                            result_to_save.pop('model', None)
                            result_to_save.pop('figures', None)
                            with open(tmp_output_dir / method / "results.json", 'w') as f:
                                json.dump(result_to_save, f, indent=4)
                                    
                except Exception as e:
                    logger.error(f"Model training failed for {table_type}/{level}/{method}: {e}")
                    data_storage = None
                                
                finally:
                    progress.update(cb_task, advance=1)
            progress.update(cb_task, description=_format_task_desc(cb_desc))
