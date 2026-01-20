# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
import pandas as pd
import numpy as np
from biom.table import Table

# Local Imports
from workflow_16s import constants
from workflow_16s.amplicon_data.helpers import _init_dict_level
from workflow_16s.figures.merged import (
    pca as plot_pca,
    pcoa as plot_pcoa,
    mds as plot_mds
)
from workflow_16s.stats.beta_diversity import (
    pca as calculate_pca,
    pcoa as calculate_pcoa,
    tsne as calculate_tsne,
    umap as calculate_umap,
)
from workflow_16s.utils.data import table_to_df, update_table_and_meta
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")
# Global lock for UMAP operations to prevent thread conflicts
umap_lock = threading.Lock()

# =================================== FUNCTIONS ====================================== #

class Ordination:
    """Performs ordination analyses (PCA, PCoA, t-SNE, UMAP) and stores figures."""
    
    TEST_CONFIG = {
        "pca": {
            "key": "pca", 
            "func": calculate_pca, 
            "plot_func": plot_pca, 
            "name": "PCA"
        },
        "pcoa": {
            "key": "pcoa", 
            "func": calculate_pcoa, 
            "plot_func": plot_pcoa, 
            "name": "PCoA"
        },
        "tsne": {
            "key": "tsne",
            "func": calculate_tsne,
            "plot_func": plot_mds,
            "name": "t‑SNE",
            "plot_kwargs": {"mode": "TSNE"},
        },
        "umap": {
            "key": "umap",
            "func": calculate_umap,
            "plot_func": plot_mds,
            "name": "UMAP",
            "plot_kwargs": {"mode": "UMAP"},
        },
    }

    def __init__(
        self, 
        config: Dict, 
        meta: pd.DataFrame,
        tables: Dict[str, Dict[str, Table]],
        verbose: bool = False
    ):
        self.config, self.verbose = config, verbose
        self.meta, self.tables = meta, tables
        self.color_columns = config['maps'].get(
            "color_columns",
            [
                constants.DEFAULT_DATASET_COLUMN, 
                constants.DEFAULT_GROUP_COLUMN,
                "env_feature", "env_material", "country"
            ],
        )
        self.group_column = config.get("group_column", constants.DEFAULT_GROUP_COLUMN)  
        self.results = {}
        
        ordination_config = self.config.get('ordination', {})
        if not ordination_config.get('enabled', False):
            logger.info("Beta diversity analysis (ordination) disabled")
            self.tasks = []
            return
            
        self.tasks = self.get_enabled_tasks()          
        if len(self.tasks) == 0:
            logger.info("No methods for beta diversity analysis (ordination) enabled")

    def get_enabled_tasks(self):  # Fixed: added self parameter
        KNOWN_METHODS = ["pca", "pcoa", "tsne", "umap"]
        DEFAULT_METHODS = {
            "raw": ["pca"],
            "filtered": ["pca", "pcoa"],
            "normalized": ["pca", "pcoa", "tsne", "umap"],
            "clr_transformed": ["pca", "pcoa", "tsne", "umap"],
            "presence_absence": ["pcoa", "tsne", "umap"]
        }
        
        ordination_config = self.config.get('ordination', {})
        table_config = ordination_config.get('tables', {})
        tasks = []
        
        for table_type, levels in self.tables.items():
            table_type_config = table_config.get(table_type, {})
            if not table_type_config.get('enabled', False):
                continue
                
            enabled_levels = [
                l for l in table_type_config.get('levels', levels.keys()) 
                if l in levels.keys()
            ]
            enabled_methods = [
                m for m in table_type_config.get('methods', DEFAULT_METHODS.get(table_type, ["pca"])) 
                if m in KNOWN_METHODS
            ]
            
            for level in enabled_levels:
                for method in enabled_methods:  
                    tasks.append((table_type, level, method))  
        
        return tasks

    def run(
        self,
        output_dir: Optional[Path] = None,
    ):
        # Initialize results storage
        for table_type, level, method in self.tasks:
            if table_type not in self.results:
                self.results[table_type] = {}
            if level not in self.results[table_type]:
                self.results[table_type][level] = {'figures': {}}
        
        # Handle no tasks case
        if not self.tasks:
            return
            
        # Set default output directory
        if output_dir is None:
            output_dir = Path(self.config['output_dir'])
            
        with get_progress_bar() as progress:
            stats_desc = f"Running beta diversity"
            stats_task = progress.add_task(  
                _format_task_desc(stats_desc),
                total=len(self.tasks)
            )
            
            # Calculate thread count safely
            cpu_count = os.cpu_count() or 1
            max_workers = min(4, max(1, cpu_count // 2))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                future_to_key = {}
    
                for table_type, level, method in self.tasks:
                    method_desc = (
                        f"{table_type.replace('_', ' ').title()} ({level.title()})"
                        f" → {self.TEST_CONFIG[method]['name']}"  
                    )
                    
                    # Create progress task for this method
                    method_task = progress.add_task(
                        _format_task_desc(method_desc),
                        total=1
                    )
                    
                    # Prepare output directory
                    table_output_dir = output_dir / 'ordination' / table_type / level
                    table_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Align table and metadata
                    table = self.tables[table_type][level]
                    table_aligned, meta_aligned = update_table_and_meta(table, self.meta)  
                        
                    future = executor.submit(
                        self._run_single_ordination,
                        table=table_aligned,
                        meta=meta_aligned,
                        symbol_col=self.group_column,  
                        table_type=table_type,
                        level=level,
                        method=method,
                        output_dir=table_output_dir,
                        progress=progress,
                        method_task=method_task,
                        method_desc=method_desc
                    )
                    futures.append(future)
                    future_to_key[future] = (table_type, level, method)
        
                errors = {}
                try:
                    for future in as_completed(futures, timeout=2*3600):
                        key = future_to_key[future]
                        try:
                            result = future.result()
                            # Store results
                            table_type, level, method, ord_result, figures = result
                            self.results[table_type][level][method] = ord_result
                            self.results[table_type][level]['figures'][method] = figures
                        except Exception as e:
                            errors[key] = str(e)
                            logger.error(f"Ordination failed for {key}: {str(e)}")
                        finally:
                            progress.update(stats_task, advance=1)  
                except TimeoutError:
                    logger.warning("Ordination timeout - proceeding with completed results")
                
            progress.update(stats_task, description=_format_task_desc(stats_desc))  

    def _run_single_ordination(
        self, table, meta, symbol_col, table_type, level, method, 
        output_dir, progress, method_task, method_desc
    ):
        try:
            progress.update(method_task, description=_format_task_desc(method_desc))
            result, figures = self._run_test(  
                table=table,
                metadata=meta,
                symbol_col=symbol_col,
                table_type=table_type,
                level=level,
                method=method,
                output_dir=output_dir
            )
            # Mark method task as complete
            progress.update(method_task, completed=1, visible=False)
            return table_type, level, method, result, figures
        except Exception as e:
            logger.error(f"Ordination {method} failed for {table_type}/{level}: {e}")
            progress.update(method_task, completed=1, visible=False)
            return table_type, level, method, None, None

    def _run_test(
        self, table, metadata, symbol_col, table_type, level, method, output_dir
    ):
        method_config = self.TEST_CONFIG[method]
        method_params = {}
        
        try:
            # Special handling for PCoA
            if method == "pcoa":
                method_params["metric"] = self.config['ordination']['tables'][table_type].get("pcoa_metric", "braycurtis")
                logger.debug(f"Using PCoA metric: {method_params['metric']}")  
            
            # Special handling for UMAP/TSNE thread safety
            if method in ["tsne", "umap"]:
                method_params["n_jobs"] = 1
                with umap_lock:
                    os.environ['NUMBA_NUM_THREADS'] = '1'
                    result = method_config["func"](table=table, **method_params)
            else:
                result = method_config["func"](table=table, **method_params)
                
        except Exception as e:
            logger.error(f"Failed {method_config['key']} for {table_type}: {e}")
            return None, {}

        try:
            figures = {}
            pkwargs = method_config.get("plot_kwargs", {})
            
            for color_col in self.color_columns:
                if color_col not in metadata.columns:
                    logger.warning(f"Color column '{color_col}' not found in metadata")
                    continue
    
                # Prepare plot parameters
                plot_params = {
                    "metadata": metadata,
                    "color_col": color_col,
                    "symbol_col": symbol_col,
                    "transformation": table_type,
                    "output_dir": output_dir,
                    **pkwargs
                }
                
                # Method-specific parameters
                if method == "pca":
                    plot_params.update({
                        "components": result["components"],
                        "proportion_explained": result["exp_var_ratio"],
                    })
                elif method == "pcoa":
                    plot_params.update({
                        "components": result.samples,
                        "proportion_explained": result.proportion_explained,
                    })
                else:  # t-SNE/UMAP
                    plot_params["df"] = result
    
                fig, _ = method_config["plot_func"](**plot_params)
                if fig:
                    figures[color_col] = fig
                    
            return result, figures

        except Exception as e:
            logger.error(f"Plotting failed for {method} ({table_type}/{level}): {e}")
            return result, {}
