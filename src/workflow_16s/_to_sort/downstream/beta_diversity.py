# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import os
import re
import time
import threading
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
import json
import pandas as pd
import numpy as np
from biom.table import Table
import plotly.io as pio

# Local Imports\
from workflow_16s.constants import GROUP_COLUMNS, MODE, DEFAULT_DATASET_COLUMN, DEFAULT_GROUP_COLUMN
from workflow_16s.diversity import beta_diversity 
from workflow_16s.downstream.load_data import align_table_and_metadata
from workflow_16s.figures.downstream.beta_diversity import beta_diversity_plot
from workflow_16s.figures.tools import json_to_fig
from workflow_16s.utils.dataframe import table_to_df
from workflow_16s.utils.dir import Dir, ProjectDir
from workflow_16s.utils.dir_utils import SubDirs
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")
#umap_lock = threading.Lock() # Global lock for UMAP operations to prevent thread conflicts

# =================================== DATA CLASSES ================================== #

@dataclass(frozen=True)
class OrdinationTask:
    """Represents a single ordination task with all necessary parameters."""
    table_type: str
    level: str
    method: str
    
    def __str__(self):
        return f"{self.table_type}/{self.level}/{self.method}"

@dataclass  
class OrdinationConfig:
    """Configuration for ordination methods."""
    key: str
    func: Callable
    name: str
    plot_kwargs: Dict = None
    
    def __post_init__(self):
        if self.plot_kwargs is None:
            self.plot_kwargs = {}

# =================================== FUNCTIONS ====================================== #

class Ordination:
    """Performs ordination analyses (PCA, PCoA, t-SNE, UMAP) and stores figures."""
    
    # Class constants for better memory efficiency
    KnownMethods = frozenset(["pca", "pcoa", "tsne", "umap"])
    DefaultMethods = {
        "raw": ("pca",),
        "filtered": ("pca", "pcoa"),
        "normalized": ("pca", "pcoa", "tsne", "umap"),
        "clr_transformed": ("pca", "pcoa", "tsne", "umap"),
        "presence_absence": ("pcoa", "tsne", "umap")
    }
    
    DefaultColorCols = (
        DEFAULT_DATASET_COLUMN,
        DEFAULT_GROUP_COLUMN,
        "env_feature", 
        "env_material", 
        "country"
    )
    
    # Use dataclass for better structure
    TestConfig = {
        "pca": OrdinationConfig("pca", beta_diversity.pca, "PCA"),
        "pcoa": OrdinationConfig("pcoa", beta_diversity.pcoa, "PCoA"),
        "tsne": OrdinationConfig("tsne", beta_diversity.tsne, "t‑SNE", 
                                 {"mode": "TSNE"}),
        "umap": OrdinationConfig("umap", beta_diversity.umap, "UMAP", 
                                 {"mode": "UMAP"}),
    }
    
    def __init__(
        self, 
        config: Dict, 
        project_dir: Union[ProjectDir, SubDirs],
        metadata: pd.DataFrame,
        tables: Dict[str, Dict[str, Table]],
        group_columns: Optional[List],
    ):
        self.config = config
        # Check if ordination is enabled
        ordination_config = self.config.get('ordination', {})
        if not ordination_config.get('enabled', False):
            logger.info("Beta diversity analysis (ordination) disabled")
            self.tasks = ()
            return
      
        self.mode = self.config.get("target_subfragment_mode", MODE)
        self.verbose = True #self.config.get("verbose", False)
        self.project_dir = project_dir

        self.group_columns = group_columns or self.config.get("group_columns", GROUP_COLUMNS)
        self.symbol_col = 'nuclear_contamination_status'
        
        self.metadata = metadata
        self.tables = tables
     
        self.color_columns = tuple(set(self.config['maps'].get("color_columns", 
                                                               self.DefaultColorCols) 
                                       + self.group_columns))

        # Initialize results storage as defaultdict
        self.results = defaultdict(lambda: defaultdict(lambda: {'figures': {}}))
            
        # Check which ordination tasks are enabled    
        self.tasks = self._get_enabled_tasks(ordination_config)          
        if not self.tasks:
            self.log_ok("No tasks enabled for beta diversity analysis (ordination)")
        else:
            self.log_ok(f"Found {len(self.tasks)} beta diversity analysis (ordination) tasks to process")

    def log_ok(self, msg):
        return (lambda msg: logger.debug(msg)) if self.verbose else (lambda *_: None)

    def run(self) -> None:
        """Run ordination analysis with optimized parallel processing."""
        if not self.tasks:
            return
            
        with get_progress_bar() as progress:
            desc = "Running beta diversity module"
            parent_task_id = progress.add_task(_format_task_desc(desc), total=len(self.tasks))
            
            max_workers = self._calculate_optimal_workers()
            self.log_ok(f"Using {max_workers} worker threads")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks at once 
                future_to_task = {
                    executor.submit(self._run_single_ordination, 
                                    task, 
                                    progress,
                                    parent_task_id): task for task in self.tasks
                }
    
                # Process completed futures with timeout
                errors = []
                try:
                    for future in as_completed(future_to_task, timeout=2*3600):
                        task = future_to_task[future]
                        try:
                            result = future.result()
                            if result:  # Only store non-None results
                                self._store_results(*result)
                        except Exception as e:
                            errors.append(str(e))
                            errors.append(str(traceback.format_exc()))
                            self.log_ok(f"Ordination failed for {task}: {str(e)}; "
                                     f"Traceback: {traceback.format_exc()}")
                        finally:
                            progress.update(parent_task_id, advance=1)
                except TimeoutError:
                    logger.warning("Ordination timeout - proceeding with completed results")
                
                if errors: # Log summary of errors if any
                    logger.warning(f"Completed with {len(errors)} errors out of {len(self.tasks)} tasks\n"
                                  f"{', '.join(errors)}")
                
            progress.update(parent_task_id, description=_format_task_desc(desc))
        self.results = dict(self.results) # Convert to dict
        self.log_ok("Ordination completed")

    def _get_enabled_tasks(self, ordination_config) -> Tuple[OrdinationTask, ...]:
        """Get enabled tasks."""
        logger.debug("Retrieving enabled ordination tasks from the config file")

        tasks = []
        
        table_config = ordination_config.get('tables', {})        
        for table_type, levels in self.tables.items():
            table_type_config = table_config.get(table_type, {})
            if not table_type_config.get('enabled', False):
                self.log_ok(f"Skipping table type {table_type}: disabled in config")
                continue
                
            # Get valid levels   
            available_levels = set(levels.keys())
            enabled_levels = set(table_type_config.get('levels', available_levels))
            valid_levels = available_levels & enabled_levels

            # Get valid methods
            default_methods = set(self.DefaultMethods.get(table_type, ("pca",)))
            enabled_methods = set(table_type_config.get('methods', default_methods))
            valid_methods = self.KnownMethods & enabled_methods
            
            for level in valid_levels:
                for method in valid_methods:  
                    tasks.append(OrdinationTask(table_type, level, method))
                    self.log_ok(f"Added task: {table_type}/{level}/{method}")
        
        self.log_ok(f"Retrieved {len(tasks)} tasks")
        return tuple(tasks) 

    def _fetch_data(self, table_type: str, level: str) -> Tuple:
        """Fetch metadata and feature table for a specified table_type and level."""
        metadata = self.metadata.get(table_type, {}).get(level)
        table = self.tables.get(table_type, {}).get(level)
        if table is None or metadata is None:
            error_msg = f"Missing table or metadata for level '{level}' and table type '{table_type}'"
            raise ValueError(error_msg)
        return table, metadata
        
    @lru_cache(maxsize=32)
    def _should_skip_existing(self, task: OrdinationTask, output_dir: Path) -> Union[bool, Dict]:
        """Check if we should skip calculation due to existing figures (cached)."""
        if not self.config.get('ordination', {}).get('load_existing', False):
            self.log_ok(f"Skipping ordination {task}: disabled in config")
            return False

        figures = {}
        
        # Check if color columns exist in metadata
        metadata = self.metadata[task.table_type][task.level]
        required_color_columns = [col for col in self.color_columns if col in metadata.columns]
        if not required_color_columns:
            self.log_ok(f"Skipping ordination {task}: no valid color columns")
            return figures
      
        # Check if all required files exist, and load them
        for color_col in required_color_columns:
            file_stem = f"{task.method.lower()}.{task.table_type}.1-2.{color_col}"
            file_path = output_dir / f"{file_stem}.json"
            if not file_path.exists() or file_path.stat().st_size == 0:
                return False
            else:
                fig = json_to_fig(file_path)
                figures[color_col] = fig
                
        self.log_ok(f"Skipping ordination {task}: all figures exist")
        return figures

    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of worker threads.
        For I/O bound tasks with some CPU computation, use more threads,
        but cap at reasonable limit to avoid resource contention."""
        cpu_count = os.cpu_count() or 1
        return min(6, max(2, cpu_count // 2 + 1))

    def _store_results(self, table_type, level, method, result: Any, figures: Dict) -> None:
        """Store ordination results efficiently."""
        self.results[table_type][level][method] = result
        self.results[table_type][level]['figures'][method] = figures
 
    def _run_single_ordination(self, task: OrdinationTask, progress: Any, parent_task_id: int) -> Optional[Tuple]:
        """Run a single ordination task with optimized error handling."""
        method_desc = (
            f"{task.table_type.replace('_', ' ').title()} ({task.level.title()})"
            f" → {self.TestConfig[task.method].name}"  
        )
        # Prepare output directory
        ordination_dir = self.project_dir.final / 'ordination'
        output_dir = ordination_dir / task.table_type / task.level / task.method
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create progress task for this method
        method_task = progress.add_task(_format_task_desc(method_desc), total=1)

        # Get the method config
        method_config = self.TestConfig[task.method]
        
        try:
            # Check if we should skip and load existing figures
            existing_figures = self._should_skip_existing(task, output_dir)
            if existing_figures is not False:
                return task.table_type, task.level, task.method, None, existing_figures
            # If we can't load all the figures, continue to calculation
            self.log_ok(f"No figures loaded for {task}, proceeding with calculation")
            result, figures = self._run_test(task, method_config, output_dir)
        except Exception as e:
            logger.error(f"Ordination {task} failed: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            result, figures = None, None
        finally:
            progress.update(method_task, completed=1, visible=False)
            # Do NOT call progress.update with the task object!
            # progress.update(parent_task_id, advance=1)   # Already done in run() above
        
        return task.table_type, task.level, task.method, result, figures
            
    def _get_method_params(self, task: OrdinationTask) -> Dict:
        """Get method-specific parameters efficiently."""
        params = {}
        if task.method == "pcoa":
            table_config = self.config['ordination']['tables'].get(task.table_type, {})
            params["metric"] = table_config.get("pcoa_metric", "braycurtis")
        elif task.method in ("tsne", "umap"):
            params["n_jobs"] = 1  # Thread safety
        return params

    def _run_test(self, task: OrdinationTask, method_config: Any, output_dir: Union[str, Path]) -> Tuple:
        table, metadata = self._fetch_data(task.table_type, task.level)
        table, metadata = align_table_and_metadata(table, metadata)
        result = self._calculate(task, method_config, table)
        if not result:
            return None, None
        figures = self._plot(result, task, method_config, metadata, output_dir)
        return result, figures
      
    def _calculate(self, task: OrdinationTask, method_config: OrdinationConfig, table: Table):
        method_params = self._get_method_params(task)
        try:
            # NOTE: umap_lock is commented out. Add it back if needed.
            # if task.method in ("tsne", "umap"):
            #     self.log_ok(f"Acquiring lock for {task.method}")
            #     with umap_lock:
            #         os.environ['NUMBA_NUM_THREADS'] = '1'
            #         result = method_config.func(table=table, **method_params)
            # else:
            #     result = method_config.func(table=table, **method_params)
            result = method_config.func(table=table, **method_params)
        except Exception as e:
            logger.error(f"Failed {task}: {e}")
            self.log_ok(f"Traceback: {traceback.format_exc()}")
            return None
        return result

    def _plot(
        self, 
        result: Any, 
        task: OrdinationTask, 
        method_config: OrdinationConfig, 
        metadata: pd.DataFrame, 
        output_dir: Union[str, Path]
    ):
        """Generate figures."""
        figures = {}

        valid_color_cols = [col for col in self.color_columns if col in metadata.columns]
        if not valid_color_cols:
            logger.warning(f"No valid color columns found for {task}")
            return {}

        ordination_type = method_config.name
        transformation = task.table_type
        type = ordination_type.lower()
        x_dim, y_dim = 1, 2

        for color_col in valid_color_cols:
            file_stem = output_dir / f"{type}.{transformation}.{x_dim}-{y_dim}.{color_col}"

            plot_params = {
                "components": None,
                "metadata": metadata,
                "ordination_type": ordination_type,
                "proportion_explained": None,
                "color_col": color_col,
                "symbol_col": self.symbol_col,
                "dimensions": (x_dim, y_dim),
                "transformation": transformation,
                "output_path": file_stem
            }

            if task.method == "pca":
                plot_params.update({
                    "components": result["components"],
                    "proportion_explained": result["exp_var_ratio"],
                })
            elif task.method == "pcoa":
                plot_params.update({
                    "components": result.samples,
                    "proportion_explained": result.proportion_explained,
                })
            else:  # t-SNE/UMAP
                plot_params["components"] = result['components']
                plot_params["proportion_explained"] = None

            try:
                self.log_ok(f"Generating figure for {task} with color column: {color_col}")
                fig = beta_diversity_plot(**plot_params)
                figures[color_col] = fig
            except Exception as e:
                logger.warning(f"Failed to generate figure for {task} with color {color_col}: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                continue
        return figures

# ==================================================================================== #

def run_beta_diversity(
    config: Dict, 
    project_dir: Union[ProjectDir, SubDirs],
    metadata: pd.DataFrame,
    tables: Dict[str, Dict[str, Table]],
    group_columns: List[str]
):
    names = [group_column['name'] for group_column in group_columns]
    beta = Ordination(
        config=config, 
        project_dir=project_dir, 
        metadata=metadata, 
        tables=tables, 
        group_columns=names
    )
    beta.run()
    return beta.results
