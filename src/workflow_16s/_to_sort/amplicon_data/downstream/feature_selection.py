# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import json
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
import catboost as cb
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from biom.table import Table
from bs4 import BeautifulSoup

# Local Imports
from workflow_16s import constants
from workflow_16s.amplicon_data.helpers import _init_dict_level
from workflow_16s.models.feature_selection import catboost_feature_selection
from workflow_16s.utils.data import table_to_df, update_table_and_meta
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")
debug_mode = False

# =================================== FUNCTIONS ====================================== #

def html_to_plotly(html_content: str) -> Optional[go.Figure]:
    """Convert HTML content to Plotly Figure if it contains Plotly JSON data"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find the script tag containing Plotly JSON data
        script_tag = soup.find('script', type='application/json')
        if not script_tag:
            logger.warning("Plotly JSON data not found in HTML - may be static image")
            return None
            
        # Load JSON and create figure
        fig_json = json.loads(script_tag.string)
        return go.Figure(fig_json)
    except Exception as e:
        logger.error(f"Error converting HTML to Plotly figure: {e}")
        return None
    

class FeatureSelection:
    """Feature selection class"""
    def __init__(
        self, 
        config: Dict, 
        metadata: pd.DataFrame,
        tables: Dict[str, Dict[str, Table]],
        group_column: str = constants.DEFAULT_GROUP_COLUMN,
        verbose: bool = False
    ):
        self.config = config
        ml_config = self.config.get('ml', {})
            
        self.group_column = group_column
        self.n_top_features = ml_config.get('num_features', 100)
        self.step_size = ml_config.get('step_size', 100)
        self.permutation_importance = ml_config.get('permutation_importance', {}).get('enabled', True)
        self.n_threads = ml_config.get('n_threads', 8)
        self.metadata = metadata
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
                
            enabled_levels = [l for l in table_type_config.get('levels', levels.keys())]
            enabled_methods = [m for m in table_type_config.get('methods', ["rfe"])]
            
            tasks = [(table_type, level, method) for level, method in product(enabled_levels, enabled_methods)]
        
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
                output_subdir = output_dir / 'ml' / self.group_column / table_type / level 
                output_subdir.mkdir(parents=True, exist_ok=True)
    
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
                        # Define required files for loading existing results
                        required_files = [
                            output_subdir / method / "best_model.cbm",
                            output_subdir / method / "feature_importances.csv",
                            output_subdir / method / "grid_search_results.csv",
                        ]
                        
                        # Check if essential files exist
                        all_essential_files_exist = all(f.exists() for f in required_files)
                        try_to_load_old = self.config.get('ml', {}).get('load_old', True)
                        load_existing = self.config.get('ml', {}).get('load_existing', True)
                        
                        # MODIFIED: Check both load_existing flag and file existence
                        if all_essential_files_exist and try_to_load_old and load_existing:
                            logger.info(f"Loading existing results for {table_type}/{level}/{method}")
                            
                            # Load CatBoost model
                            model = cb.CatBoostClassifier()
                            model.load_model(str(output_subdir / method / "best_model.cbm"))
                            
                            # Load feature importances
                            feature_importances = pd.read_csv(output_subdir / method / "feature_importances.csv")
                            
                            # Load grid search results
                            grid_search_results = pd.read_csv(output_subdir / method / "grid_search_results.csv")
                            
                            # Create figure paths dictionary
                            figure_paths = {
                                'confusion_matrix': output_subdir / method / "best_confusion_matrix.html",
                                'roc': output_subdir / method / "best_roc_curve.html",
                                'prc': output_subdir / method / "best_precision_recall_curve.html",
                                'shap_summary_bar': output_subdir / method / "figs" / f"shap.summary.bar.{self.n_top_features}.html",
                                'shap_summary_beeswarm': output_subdir / method / "figs" / f"shap.summary.beeswarm.{self.n_top_features}.html",
                                'shap_summary_heatmap': output_subdir / method / "figs" / f"shap.summary.heatmap.{self.n_top_features}.html",
                                'shap_summary_force': output_subdir / method / "figs" / f"shap.summary.force.{self.n_top_features}.html",
                            }
                            
                            # Initialize figures dictionary
                            plotly_figures = {}
                            
                            # Load figures if files exist
                            for fig_name, fig_path in figure_paths.items():
                                if fig_path.exists():
                                    try:
                                        html_content = fig_path.read_text(encoding='utf-8')
                                        fig = html_to_plotly(html_content)
                                        if fig:
                                            plotly_figures[fig_name] = fig
                                        else:
                                            logger.warning(f"Plotly conversion failed for {fig_name} - may be static image")
                                    except Exception as e:
                                        logger.error(f"Error loading {fig_name}: {e}")
                                else:
                                    logger.warning(f"Figure file missing: {fig_path}")
                            
                            # Create result dictionary
                            model_result = {
                                'model': model,
                                'feature_importances': feature_importances,
                                'grid_search_results': grid_search_results,
                                'figures': plotly_figures
                            }
                            
                        else:
                            logger.info(f"Running model for {table_type}/{level}/{method}")
                            
                            table = self.tables[table_type][level]
                            metadata = self.metadata[table_type][level]
                            X = table_to_df(table)
                            X.index = X.index.str.lower()
                            y = metadata.set_index("#sampleid")[[self.group_column]]
                            y.index = y.index.astype(str).str.lower()
                            idx = X.index.intersection(y.index)
                            X, y = X.loc[idx], y.loc[idx]
    
                            use_permutation_importance = False if method == "select_k_best" else self.permutation_importance
                                        
                            model_result = catboost_feature_selection(
                                metadata=y,
                                features=X,
                                output_dir=output_subdir,
                                group_col=self.group_column,
                                method=method,
                                n_top_features=self.n_top_features,
                                step_size=self.step_size,
                                use_permutation_importance=use_permutation_importance,
                                thread_count=self.n_threads,
                                progress=progress, 
                                task_id=cb_task,
                            )
                        
                        data_storage[method] = model_result        
                        # Log if no figures were generated
                        if not any(model_result['figures'].values()):
                            logger.warning(f"No figures generated for {table_type}/{level}/{method}")
                                    
                except Exception as e:
                    logger.error(f"Model training failed for {table_type}/{level}/{method}: {e}")
                    data_storage = None
                                
                finally:
                    progress.update(cb_task, advance=1)
            progress.update(cb_task, description=_format_task_desc(cb_desc))
