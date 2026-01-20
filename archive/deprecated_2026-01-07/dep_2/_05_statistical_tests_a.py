import logging
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import anndata as ad
import pandas as pd
from workflow_16s.downstream.statistics import tests as stats_calcs 
from workflow_16s.logger import get_logger
from workflow_16s.utils.progress import get_progress_bar

logger = get_logger()

class StatisticalTests:
    TEST_CONFIG = {
        "fisher": stats_calcs.fisher_exact_bonferroni,
        "ttest": stats_calcs.ttest,
        "mwu_bonferroni": stats_calcs.mwu_bonferroni,
        "kruskal_bonferroni": stats_calcs.kruskal_bonferroni,
        "anova": stats_calcs.anova,
    }

    def __init__(self, config: Dict, adata: ad.AnnData):
        self.config = config
        self.adata = adata
        self.stats_config = self.config.get('stats', {})
        if not self.stats_config.get('enabled', True):
            self.group_columns, self.tasks_by_group = [], {}
            return

        primary_cols = self.stats_config.get('group_columns', [])
        auto_cols = self.adata.uns.get('analysis_columns', {}).get("group_comparison", []) if self.stats_config.get('analyze_all_valid_columns', True) else []
        self.group_columns = sorted(list(set(primary_cols + auto_cols)))
        self.tasks_by_group = self._get_enabled_tasks()

    def run(self, output_dir: Path) -> ad.AnnData:
        if not self.tasks_by_group: return self.adata
        logger.info("STEP 5: Running statistical tests...")
        for group_col in self.group_columns:
            if group_col not in self.adata.obs.columns or self.adata.obs[group_col].nunique() < 2:
                logger.warning(f"Group column '{group_col}' is invalid or has < 2 groups; skipping stats.")
                continue
            if tasks := self.tasks_by_group.get(group_col, []):
                self._run_for_group(group_col, tasks, output_dir)
        return self.adata

    def _run_for_group(self, group_col: str, tasks: list, output_dir: Path):
        max_workers = self.config.get("threads", 4)
        with get_progress_bar() as progress:
            p_task = progress.add_task(f"[cyan]Stats for '{group_col}'[/]", total=len(tasks))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._run_single_test, task, group_col): task for task in tasks}
                for future in as_completed(futures):
                    task_id, result = futures[future], None
                    try:
                        result = future.result()
                        if result is not None and not result.empty:
                            layer, _, test = task_id
                            group_results = self.adata.uns.setdefault('statistical_tests', {}).setdefault(group_col, {})
                            layer_results = group_results.setdefault(layer, {})
                            layer_results[test] = result
                    except Exception as e:
                        logger.error(f"Task {task_id} on group '{group_col}' failed: {e}")
                    finally:
                        progress.update(p_task, advance=1)

    def _run_single_test(self, task: Tuple[str, str, str], group_col: str) -> pd.DataFrame:
        layer, _, test_name = task
        data_matrix = self.adata.layers[layer].toarray() if hasattr(self.adata.layers[layer], 'toarray') else self.adata.layers[layer]
        df = pd.DataFrame(data_matrix, index=self.adata.obs_names, columns=self.adata.var_names)
        
        metadata_copy = self.adata.obs.copy()
        metadata_copy.index.name = 'sample_id'
        metadata_copy.reset_index(inplace=True)
        
        group_values = metadata_copy[group_col].dropna().unique().tolist()
        test_func = self.TEST_CONFIG[test_name]
        
        # CORRECTED: Provide all four required positional arguments
        return test_func(df, metadata_copy, group_col, group_values)

    def _get_enabled_tasks(self) -> Dict[str, List]:
        tasks_by_group = {}
        for group_name in self.group_columns:
            tasks = []
            for layer_name, layer_conf in self.stats_config.get('tables', {}).items():
                if layer_conf.get('enabled', True) and layer_name in self.adata.layers:
                    for test in layer_conf.get('tests', []):
                        if test in self.TEST_CONFIG:
                            tasks.append((layer_name, 'asv', test))
            tasks_by_group[group_name] = tasks
        return tasks_by_group