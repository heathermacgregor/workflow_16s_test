# ===================================== IMPORTS ====================================== #

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# Assume these local imports exist in your project structure
from workflow_16s.downstream import Data
from workflow_16s.downstream.statistics import tests as stats_calcs 
from workflow_16s.utils.data import sync_samples as align_table_and_metadata
from workflow_16s.utils.progress import get_progress_bar

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# =================================== CLASS ====================================== #

class StatisticalTests:
    """
    Performs and visualizes core statistical tests in parallel, driven by a
    configuration file and a predefined test configuration map.
    """
    # Using the TestConfig dictionary you provided
    TEST_CONFIG = {
        "fisher": {"key": "fisher", "func": stats_calcs.fisher_exact_bonferroni},
        "ttest": {"key": "ttest", "func": stats_calcs.ttest},
        "mwu_bonferroni": {"key": "mwub", "func": stats_calcs.mwu_bonferroni},
        "kruskal_bonferroni": {"key": "kwb", "func": stats_calcs.kruskal_bonferroni},
        "enhanced_stats": {"key": "enhanced", "func": stats_calcs.enhanced_statistical_tests},
        "anova": {"key": "anova", "func": stats_calcs.anova},
    }
    
    TEST_DEFAULTS = {
        "raw": ["ttest"],
        "filtered": ['mwu_bonferroni', 'kruskal_bonferroni'],
        "normalized": ['ttest', 'mwu_bonferroni', 'kruskal_bonferroni'],
        "clr_transformed": ['ttest', 'mwu_bonferroni', 'kruskal_bonferroni'],
        "presence_absence": ["fisher"]
    }

    def __init__(self, config: Dict, data: Data, verbose: bool = False):
        self.config = config
        self.data = data
        self.metadata = data.metadata
        self.tables = data.tables
        self.verbose = verbose

        stats_config = self.config.get('stats', {})
        if not stats_config.get('enabled', False):
            self.tasks_by_group = {}
            return
        categorical_cols = self.data.analysis_columns.get("group_comparison", [])    
        self.group_columns = self.config.get('group_columns', [])
        for col in categorical_cols:
            if col not in [g['name'] for g in self.group_columns]:
                self.group_columns.append({'name': col, 'values': self.metadata[col].dropna().unique().tolist() if self.metadata is not None and col in self.metadata.columns else []})
        self.tasks_by_group = self._get_enabled_tasks(stats_config)

    def run(self, output_dir: Path) -> None:
        """Executes all configured statistical tests for each specified group column."""
        if not self.tasks_by_group:
            logger.warning("No enabled tasks for statistical tests.")
            return

        for group_info in self.group_columns:
            group_col = group_info['name']
            tasks = self.tasks_by_group.get(group_col, [])
            if tasks:
                self._run_for_group(group_col, group_info.get('values'), tasks, output_dir)

    def _run_for_group(self, group_col: str, groups: List[Any], tasks: list, output_dir: Path):
        """Manages the parallel execution of tests for a single group column."""
        max_workers = self.config.get("threads", 4)
        with get_progress_bar() as progress:
            task_desc = f"Statistical Tests for '{group_col}'"
            p_task = progress.add_task(task_desc, total=len(tasks))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._run_single_test, task, group_col, groups, output_dir): task
                    for task in tasks
                }
                for future in as_completed(futures):
                    task_id = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            table_type, level, test = task_id
                            self.data.analysis_results['statistical_tests'][group_col][table_type][level][test] = result
                    except Exception as e:
                        logger.error(f"Task {task_id} on group {group_col} failed: {e}", exc_info=self.verbose)
                    progress.update(p_task, advance=1)

    def _run_single_test(self, task: Tuple[str, str, str], group_col: str, groups: List[Any], base_output_dir: Path) -> pd.DataFrame:
        """Executes a single statistical test, including data alignment and plotting."""
        table_type, level, test_name = task
        
        table = self.tables[table_type][level]
        if self.metadata is None: return pd.DataFrame()
        
        table, metadata = align_table_and_metadata(table, self.metadata)
        if table.is_empty(): return pd.DataFrame()
        
        test_func = self.TEST_CONFIG[test_name]["func"]
        
        result_df = test_func(
            table=table, 
            metadata=metadata, 
            group_column=group_col, 
            group_column_values=groups
        )
        
        output_dir = base_output_dir / "statistical_tests" / group_col / table_type / level / test_name
        output_dir.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_dir / "statistics.tsv", sep="\t")
             
        return result_df

    def _get_enabled_tasks(self, stats_config: Dict) -> Dict[str, List]:
        """Parses the config to determine which tests to run for each group."""
        tasks_by_group = {}
        for group_info in self.group_columns:
            group_name = group_info['name']
            tasks = []
            table_config = stats_config.get('tables', {})
            for table_type, levels in self.tables.items():
                type_conf = table_config.get(table_type, {})
                if type_conf.get('enabled', False):
                    enabled_levels = set(type_conf.get('levels', levels.keys())) & set(levels.keys())
                    configured_tests = set(type_conf.get('tests', self.TEST_DEFAULTS.get(table_type, [])))
                    enabled_tests = configured_tests & set(self.TEST_CONFIG.keys())
                    
                    for level in enabled_levels:
                        tasks.extend([(table_type, level, test) for test in enabled_tests])
            tasks_by_group[group_name] = tasks
        return tasks_by_group
    
  
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
    
    # --- Step 5: Run Statistical Tests ---
    logger.info("STEP 5: Running statistical tests...")
    stats_analyzer = StatisticalTests(config, data_object)
    stats_analyzer.run(output_dir=results_dir)
    logger.info("STEP 5: Statistical tests complete.")