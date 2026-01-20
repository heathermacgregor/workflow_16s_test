# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
import re
from pathlib import Path
from typing import Any, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-Party Imports
import pandas as pd
from biom.table import Table

# Assume these local imports exist in your project structure
from workflow_16s.constants import DEFAULT_ALPHA_METRICS, DEFAULT_GROUP_COLUMN
from workflow_16s.downstream import Data 
from workflow_16s.downstream.diversity.alpha_diversity import (
    alpha_diversity, analyze_alpha_diversity, analyze_alpha_correlations
)
from workflow_16s.visualization.alpha_diversity import (
    create_alpha_diversity_boxplot,
    create_alpha_diversity_stats_plot,
    plot_alpha_correlations
)
from workflow_16s.utils.biom_utils import to_df
from workflow_16s.utils.data import sync_samples
from workflow_16s.utils.progress import get_progress_bar


# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# =================================== CLASS ====================================== #

class AlphaDiversity:
    """Performs alpha diversity analysis, including statistical tests,
    correlation analysis, and plotting, with parallel execution of tasks.
    Results are saved directly to the provided Data object.
    
    Attributes:
        config:         Configuration dictionary.
        data:           Data object containing tables and metadata.
        metadata:       Metadata DataFrame.
        tables:         BIOM tables from the Data object.
        verbose:        Verbosity flag for logging.
        group_columns:  List of metadata columns to group by.
        metrics:        List of alpha diversity metrics to compute.
        parametric:     Whether to use parametric tests.
        corr_config:    Configuration for correlation analysis.
        plot_config:    Configuration for plotting.
        tasks:          List of (table_type, level) tuples to process.
    
    Methods:
        run:                Executes the alpha diversity analysis.
        _run_for_col:       Manages execution for a specific metadata column.
        _run_single_task:   Executes the analysis for a single table in parallel.
        _get_enabled_tasks: Determines which table/level combinations to analyze.
    """
    def __init__(
        self,
        config: Dict,
        data: Data,
        verbose: bool = False
    ):
        self.config = config
        self.data = data
        self.metadata = data.metadata
        self.tables = data.tables
        self.verbose = verbose
        
        alpha_config = self.config.get('alpha_diversity', {})
        if not alpha_config.get('enabled', False):
            logger.debug("Alpha diversity analysis disabled in config.")
            self.tasks = []
            return
        categorical_cols = self.data.analysis_columns.get("group_comparison", [])  
        continuous_cols = self.data.analysis_columns.get("correlation_gradient", [])  
        more_cols = categorical_cols + continuous_cols
        self.group_columns = self.config.get('group_columns', [])
        for col in more_cols:
            if col not in [g['name'] for g in self.group_columns]:
                self.group_columns.append({'name': col, 
                                           'values': self.metadata[col].dropna().unique().tolist() 
                                           if self.metadata is not None 
                                           and col in self.metadata.columns else []})
        
        self.metrics = alpha_config.get('metrics', DEFAULT_ALPHA_METRICS)
        self.parametric = alpha_config.get('parametric', False)
        self.corr_config = alpha_config.get("correlation_analysis", {})
        self.plot_config = alpha_config.get("plots", {})
        self.tasks = self._get_enabled_tasks()

    def run(self, output_dir: Path) -> None:
        """Executes the alpha diversity analysis for the primary group columns
        and for 'facility_match' if it exists. Modifies the Data object in place.
        """
        if not self.tasks:
            logger.warning("No enabled tasks for alpha diversity analysis.")
            return
        for col in self.group_columns:
            if col['name'] not in self.metadata.columns:
                logger.warning(f"Group column '{col['name']}' not found in metadata; skipping.")    
            self._run_for_col(col['name'], output_dir)

        if self.metadata is not None and 'facility_match' in self.metadata.columns:
            logger.info("Found 'facility_match' column, running secondary alpha diversity analysis.")
            self._run_for_col('facility_match', output_dir)
        
    def _run_for_col(self, group_column: str, base_output_dir: Path) -> None:
        """Manages the parallel execution of alpha diversity tasks for a given
        metadata group column.
        """
        # Ensure output directory exists
        base_output_dir.mkdir(parents=True, exist_ok=True)
        # Determine number of workers
        max_workers = self.config.get("threads", 4)
        
        with get_progress_bar() as progress:
            p_task = progress.add_task(f"Alpha diversity for '{group_column}'", 
                                       total=len(self.tasks))
            # Use ThreadPoolExecutor for parallel task execution
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._run_single_task, task, group_column, base_output_dir): task
                    for task in self.tasks
                }
                # Collect results as they complete
                for future in as_completed(futures):
                    task_id = futures[future]
                    try:
                        task_results = future.result()
                        if task_results:
                            table_type, level = task_id
                            # Assign results directly; defaultdict handles creation of nested keys
                            self.data.analysis_results[group_column][table_type][level] = task_results
                    except Exception as e:
                        logger.error(f"Task {task_id} failed: {e}", exc_info=self.verbose)
                    finally:
                        progress.update(p_task, advance=1)

    def _run_single_task(self, task: Tuple[str, str], group_column: str, base_output_dir: Path) -> Dict:
        """Executes the full alpha diversity pipeline for a single table. This function 
        is designed to be called in parallel.
        
        Args:
            task:            Tuple of (table_type, level) to process.
            group_column:   Metadata column to group by.
            base_output_dir: Base directory for output files.   
            
        Returns:
            Dictionary containing results, stats, correlations, and figures.
        """
        table_type, level = task
        task_output_dir = base_output_dir / 'alpha_diversity' / group_column / table_type / level
        task_output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Align the specific table with the master metadata
        table = self.tables[table_type][level]
        if self.metadata is None:
            logger.warning(f"Skipping {table_type}/{level} as metadata is not available.")
            return {}
        table, metadata = sync_samples(table, self.metadata)

        if table.is_empty():
            logger.warning(f"Skipping {table_type}/{level} as it is empty after alignment.")
            return {}
        
        # 2. Calculate Alpha Diversity
        table_df = to_df(table)
        alpha_df = alpha_diversity(table_df, metrics=self.metrics)
        alpha_df.to_csv(task_output_dir / 'alpha_diversity.tsv', sep='\t')

        # 3. Perform Statistical Analysis
        stats_df = analyze_alpha_diversity(
            alpha_diversity_df=alpha_df,
            metadata=metadata,
            group_column=group_column,
            parametric=self.parametric
        )
        # Save stats to file
        stats_df.to_csv(task_output_dir / 'stats.tsv', sep='\t')
        
        # 4. Collect all results
        task_data: Dict[str, Any] = {'results': alpha_df, 'stats': stats_df}

        # 5. Correlation Analysis (optional)
        if self.corr_config.get('enabled', False):
            corr_results = analyze_alpha_correlations(
                alpha_df, metadata,
                max_categories=self.corr_config.get("max_categories", 20),
                min_samples=self.corr_config.get("min_group_size", 5)
            )
            task_data['correlations'] = corr_results
            pd.DataFrame(corr_results).to_csv(task_output_dir / 'correlations.tsv', sep='\t')

        # 5. Generate Plots (optional)
        if self.plot_config.get('enabled', True):
            task_data['figures'] = {}
            for metric in self.metrics:
                if not alpha_df[metric].isnull().all():
                    create_alpha_diversity_boxplot(
                        alpha_df=alpha_df, metadata=metadata, group_column=group_column,
                        metric=metric, output_dir=task_output_dir
                    )
            create_alpha_diversity_stats_plot(
                stats_df=stats_df, output_dir=task_output_dir,
                effect_size_threshold=self.plot_config.get('effect_size_threshold', 0.5)
            )
            if 'correlations' in task_data:
                plot_alpha_correlations(
                    task_data['correlations'], output_dir=task_output_dir,
                    top_n=self.corr_config.get('top_n_correlations', 10)
                )
        return task_data

    def _get_enabled_tasks(self) -> list[Tuple[str, str]]:
        """Parses the config to determine which table/level combinations to analyze."""
        tasks = []
        table_config = self.config.get('alpha_diversity', {}).get('tables', {})
        for table_type, levels in self.tables.items():
            type_conf = table_config.get(table_type, {})
            if type_conf.get('enabled', False):
                enabled_levels = type_conf.get('levels', list(levels.keys()))
                tasks.extend([(table_type, lvl) for lvl in enabled_levels if lvl in levels])
        return tasks
    
if __name__ == "__main__":
    from workflow_16s.config import get_config
    from workflow_16s.downstream import _01_load_data as load_data_mod
    config = get_config()
    
    project_dir = Path(config.get("project_dir", "."))
    results_dir = project_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # --- Step 1: Load Data ---
    logger.info("STEP 1: Loading and aligning data...")
    loader = load_data_mod.DataLoader(config)
    data_object = loader.run()
    logger.info("STEP 1: Data loading complete.")
    
    # --- Step 3: Alpha Diversity Analysis ---
    logger.info("STEP 3: Running alpha diversity analysis...")
    alpha_diver = AlphaDiversity(config, data_object, verbose=True)
    alpha_diver.run(results_dir)
    logger.info("STEP 3: Alpha diversity analysis complete.")