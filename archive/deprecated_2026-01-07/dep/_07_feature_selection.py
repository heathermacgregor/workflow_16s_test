# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

# Third-Party Imports
import pandas as pd
import catboost as cb
from biom.table import Table

# Local Imports
# (Assuming these local imports are correct and available)
from workflow_16s.downstream import Data
from workflow_16s.utils.biom_utils import to_df
from workflow_16s.utils.progress import get_progress_bar
from workflow_16s.visualization.machine_learning import (
    plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve, plot_shap
)
from workflow_16s.constants import DEFAULT_GROUP_COLUMN
from workflow_16s.downstream.models.feature_selection import catboost_feature_selection

# ==================================================================================== #

logger = logging.getLogger("workflow_16s")

# =================================== CONSTANTS ====================================== #

BEST_MODEL_FILENAME = "best_model.cbm"
FEATURE_IMPORTANCES_FILENAME = "feature_importances.csv"
GRID_SEARCH_RESULTS_FILENAME = "grid_search_results.csv"

# ==================================================================================== #
        
class FeatureSelection:
    """A class to perform and manage machine learning feature selection tasks."""
    def __init__(
        self, 
        config: Dict, 
        data: Data,
        group_column: str = DEFAULT_GROUP_COLUMN,
    ):
        self.config = config
        ml_config = self.config.get('ml', {})
        self.data = data
        self.group_column = group_column
        
        # --- Configuration Attributes ---
        self.n_top_features = ml_config.get('num_features', 100)
        self.step_size = ml_config.get('step_size', 100)
        self.use_permutation_importance = ml_config.get('permutation_importance', {}).get('enabled', True)
        self.n_threads = ml_config.get('n_threads', 8)
        self.load_existing = ml_config.get('load_existing', True)
        
        # --- Data Attributes ---
        self.metadata = self.data.metadata
        self.tables = self.data.tables
      
        # --- Class State ---
        # Nested defaultdict for storing models: models[table_type][level][method]
        self.models: Dict[str, Any] = defaultdict(lambda: defaultdict(dict))
        
        if not ml_config.get('enabled', False):
            logger.info("ML feature selection is disabled in the configuration.")
            self.tasks = []
            return
            
        self.tasks = self.get_enabled_tasks()
        if not self.tasks:
            logger.info("No ML feature selection tasks are enabled.")

    def get_enabled_tasks(self) -> List[Tuple[str, str, str]]:
        """Parses the config to generate a list of ML tasks to run."""        
        ml_config = self.config.get('ml', {})
        table_config = ml_config.get('tables', {})
        all_tasks = []
        
        for table_type, levels in self.tables.items():
            table_type_config = table_config.get(table_type, {})
            if not table_type_config.get('enabled', False):
                continue
                
            enabled_levels = table_type_config.get('levels', list(levels.keys()))
            enabled_methods = table_type_config.get('methods', ["rfe"])
            
            # **FIXED**: Use extend to accumulate tasks from all table types
            tasks_for_table = list(product(enabled_levels, enabled_methods))
            all_tasks.extend([(table_type, level, method) for level, method in tasks_for_table])
        
        return all_tasks
      
    def run(self, output_dir: Optional[Path] = None) -> None:
        """Executes all enabled feature selection tasks."""
        if not self.tasks:
            logger.warning("No tasks to run for feature selection.")
            return

        if output_dir is None:
            output_dir = Path(self.config.get("project_dir", ".")) / "results"

        with get_progress_bar() as progress:
            task_id = progress.add_task(
                "Running CatBoost feature selection", total=len(self.tasks)
            )
            for task in self.tasks:
                self._process_task(task, output_dir, progress, task_id)
            progress.update(task_id, description="✅ CatBoost feature selection complete.")
            
    def _process_task(
        self, task: Tuple[str, str, str], output_dir: Path, progress, task_id
    ) -> None:
        """Processes a single feature selection task (load existing or run new)."""
        table_type, level, method = task
        
        method_desc = (
            f"{table_type.replace('_', ' ').title()} ({level.title()})"
            f" → {method.title()}"
        )
        progress.update(task_id, description=method_desc)
        
        output_subdir = output_dir / 'ml' / self.group_column / table_type / level
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        try:
            model_result = self._load_results(output_subdir, method)
            if model_result is None:
                logger.info(f"Running new analysis for {table_type}/{level}/{method}")
                model_result = self._run_new_analysis(
                    task, output_subdir, progress, task_id
                )
            
            self.models[table_type][level][method] = model_result

        except Exception as e:
            logger.error(f"Model training failed for {table_type}/{level}/{method}: {e}", exc_info=True)
            self.models[table_type][level][method] = None # Ensure failure is recorded
        finally:
            progress.update(task_id, advance=1)
            
    def _load_results(self, path: Path, method: str) -> Optional[Dict[str, Any]]:
        """Attempts to load pre-existing results for a given task."""
        if not self.load_existing:
            return None
            
        method_path = path / method
        required_files = [
            method_path / BEST_MODEL_FILENAME,
            method_path / FEATURE_IMPORTANCES_FILENAME,
            method_path / GRID_SEARCH_RESULTS_FILENAME,
        ]
        
        if not all(f.exists() for f in required_files):
            return None
            
        logger.info(f"Loading existing results from: {method_path}")
        model = cb.CatBoostClassifier()
        model.load_model(str(required_files[0]))
        
        return {
            'model': model,
            'feature_importances': pd.read_csv(required_files[1]),
            'grid_search_results': pd.read_csv(required_files[2]),
            'figures': {} # Figures are not loaded, but can be regenerated if needed
        }

    def _run_new_analysis(
        self, task: Tuple[str, str, str], output_subdir: Path, progress, task_id
    ) -> Optional[Dict[str, Any]]:
        """Runs a new feature selection analysis and returns the results."""
        table_type, level, method = task
        
        if table_type == "clr_transformed" and method == "chi_squared":
            logger.warning("Skipping chi_squared for CLR data as it's not applicable.")
            return None
            
        # --- Prepare Data ---
        table = self.tables[table_type][level]
        metadata = self.metadata
        if metadata is None or metadata.empty:
            raise ValueError("Metadata is required for ML feature selection but is missing or empty.")
        
        X = to_df(table)
        
        if metadata.index.name != '#sampleid':
            metadata = metadata.set_index('#sampleid')
        y = metadata[[self.group_column]]
        
        # Synchronize indices
        idx = X.index.intersection(y.index.astype(str))
        X, y = X.loc[idx], y.loc[idx]

        perm_importance_flag = False if method == "select_k_best" else self.use_permutation_importance
                                
        return catboost_feature_selection(
            metadata=y,
            features=X,
            output_dir=output_subdir,
            group_col=self.group_column,
            method=method,
            n_top_features=self.n_top_features,
            step_size=self.step_size,
            use_permutation_importance=perm_importance_flag,
            thread_count=self.n_threads,
            progress=progress, 
            task_id=task_id,
        )
        

if __name__ == "__main__":
    from pathlib import Path
    from workflow_16s.config import get_config # type: ignore
    from workflow_16s.downstream import _01_load_data as load_data_mod

    # --- Configuration and Setup ---
    config = get_config()
    project_dir = Path(config.get("project_dir", "."))
    results_dir = project_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # --- Step 1: Load Data ---
    logger.info("STEP 1: Loading and aligning data...")
    loader = load_data_mod.DataLoader(config)
    data_object = loader.run()
    logger.info("✅ STEP 1: Data loading complete.")

    # --- Step 7: Run Feature Selection ---
    logger.info("STEP 7: Running Machine Learning Feature Selection...")
    ml_selector = FeatureSelection(
        config=config, 
        data=data_object, 
        group_column='facility_match'
    )
    ml_selector.run(results_dir)
    logger.info("✅ STEP 7: Machine Learning Feature Selection complete.")