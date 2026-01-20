# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-Party Imports
import pandas as pd
from biom.table import Table
import plotly.graph_objects as go

# Local Imports (ensure these paths are correct in your project)
from workflow_16s.downstream.diversity import beta_diversity as beta_calcs
from workflow_16s.logger import get_logger
from workflow_16s.visualization.beta_diversity import beta_diversity_plot
from workflow_16s.utils.data import sync_samples 
from workflow_16s.utils.biom_utils import to_df, to_biom
from workflow_16s.utils.progress import get_progress_bar
from workflow_16s.downstream import Data # Assuming Data class is in a local module

# ========================== INITIALISATION & CONFIGURATION ========================== #

#logger = logging.getLogger("workflow_16s")
logger = get_logger()

# =================================== CLASS ====================================== #

class BetaDiversity:
    """Performs and visualizes beta diversity ordination.
    
    Attributes:
        config:         Configuration dictionary.
        data:           Data object containing tables and metadata.
        metadata:       Metadata DataFrame.
        tables:         Dictionary of feature tables.
        verbose:        Enable verbose logging.
        color_columns:  List of metadata columns for coloring plots.
        symbol_column:  Metadata column for point symbols in plots.
        tasks:          List of ordination tasks to execute.
    """
    # Map method names from the config to your new calculation functions
    ORDINATION_METHODS = {
        "pca": beta_calcs.pca,
        "pcoa": beta_calcs.pcoa,
        "tsne": beta_calcs.tsne,
        "umap": beta_calcs.umap,
        "mds": beta_calcs.mds,
    }

    def __init__(self, config: Dict, data: Data, verbose: bool = False):
        """Initializes the BetaDiversity class with configuration and data.
        
        Args:
            config:  Configuration dictionary.
            data:    Data object containing tables and metadata.
            verbose: Enable verbose logging.
        """
        self.config = config
        self.data = data
        self.metadata = data.metadata
        self.tables = data.tables
        self.verbose = verbose

        beta_config = self.config.get('beta_diversity', {})
        if not beta_config.get('enabled', False):
            logger.debug("Beta diversity analysis disabled in config.")
            self.tasks = []
            return
        categorical_cols = self.data.analysis_columns.get("group_comparison", [])     
        self.color_columns = categorical_cols if categorical_cols else self.config.get('color_columns', [])
        self.symbol_column = beta_config.get('symbol_column', 'nuclear_contamination_status')
        self.tasks = self._get_enabled_tasks(beta_config)

    def run(self, output_dir: Path) -> None:
        """Executes the beta diversity ordination pipeline in parallel.
        Modifies the Data object in place by saving results to its
        'analysis_results' attribute.
        
        Args:
            output_dir: Directory to save results.
        """
        if not self.tasks:
            logger.warning("No enabled tasks for beta diversity analysis.")
            return

        # Safely initialize the nested dictionary for storing results
        self.data.analysis_results.setdefault('beta_diversity', {}) # type: ignore

        max_workers = self.config.get("threads", 4)
        with get_progress_bar() as progress:
            p_task = progress.add_task("Running Beta Diversity", total=len(self.tasks))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._run_single_task, task, output_dir): task
                    for task in self.tasks
                }

                for future in as_completed(futures):
                    task_id = futures[future]
                    try:
                        task_results = future.result()
                        if task_results:
                            table_type, level, method = task_id
                            # Safely create nested keys before assignment
                            self.data.analysis_results['beta_diversity'].setdefault(table_type, {})
                            self.data.analysis_results['beta_diversity'][table_type].setdefault(level, {})
                            self.data.analysis_results['beta_diversity'][table_type][level][method] = task_results
                    except Exception as e:
                        logger.error(f"Task {task_id} failed: {e}", exc_info=self.verbose)
                    progress.update(p_task, advance=1)

    def _run_single_task(self, task: Tuple[str, str, str], base_output_dir: Path) -> Dict[str, Any]:
        """Executes the full ordination pipeline for a single task with verbose logging.
        
         Args:
            task:            Tuple of (table_type, level, method).
            base_output_dir: Base directory to save results.
            
        Returns:
            Dictionary with ordination results and figures.
        """
        table_type, level, method = task
        task_name = f"'{table_type}/{level}/{method}'"
        logger.info(f"--- Starting debug for task: {task_name} ---")

        task_output_dir = base_output_dir / 'beta_diversity' / table_type / level / method
        task_output_dir.mkdir(parents=True, exist_ok=True)

        # 1. INITIAL STATE CHECK
        table = self.tables[table_type][level]
        if self.metadata is None:
            logger.warning(f"Task {task_name}: Skipping as metadata is not available.")
            return {}
        
        initial_table_ids = set(table.ids())
        initial_metadata_ids = set(self.metadata.index)
        logger.debug(f"Task {task_name}: Initial counts | Table samples: {len(initial_table_ids)}, "
                     f"Metadata samples: {len(initial_metadata_ids)}")
        # 2. ALIGNMENT AND DEBUGGING
        aligned_table, aligned_metadata = sync_samples(table, self.metadata)
        
        # Enforce lowercase on all sample IDs to prevent matching errors
        aligned_metadata.index = aligned_metadata.index.str.lower()
        df = to_df(aligned_table)
        df.index = df.index.str.lower()

        aligned_table_ids = set(df.index)
        aligned_metadata_ids = set(aligned_metadata.index)
        logger.debug(
            f"Task {task_name}: After alignment | Table samples: {len(aligned_table_ids)}, "
            f"Metadata samples: {len(aligned_metadata_ids)}"
        )

        # 3. ZERO-COUNT FILTERING
        non_zero_samples = set(df.index[df.sum(axis=1) > 0])
        logger.debug(f"Task {task_name}: Found {len(non_zero_samples)} samples with non-zero counts out of {len(aligned_table_ids)} aligned samples.")
        
        zero_count_dropped_ids = aligned_table_ids - non_zero_samples
        if zero_count_dropped_ids:
            logger.debug(f"Task {task_name}: Dropping {len(zero_count_dropped_ids)} samples because they have all-zero feature counts.")

        final_samples_to_keep = aligned_metadata_ids.intersection(non_zero_samples)
        
        # 4. FINAL STATE and EXECUTION
        logger.debug(f"Task {task_name}: Final sample count for analysis is {len(final_samples_to_keep)}.")
        
        if not final_samples_to_keep:
            logger.warning(f"Task {task_name}: Skipping as no samples remained after alignment and filtering.")
            return {}

        final_metadata = aligned_metadata.loc[list(final_samples_to_keep)]
        final_df = df.loc[list(final_samples_to_keep)]
        final_table = Table(final_df.values.T, observation_ids=final_df.columns, sample_ids=final_df.index)

        # 5. Calculate Ordination
        ordination_func = self.ORDINATION_METHODS[method]
        method_params = self._get_method_params(table_type, method)
        ordination_result = ordination_func(table=final_table, **method_params)

        # 6. Generate and save plots
        figures = self._plot_ordination(
            result=ordination_result,
            task=task,
            metadata=final_metadata,
            output_dir=task_output_dir
        )
        
        logger.debug(f"--- Finished task: {task_name} ---")
        return {'ordination': ordination_result, 'figures': figures}

    def _get_method_params(self, table_type: str, method: str) -> Dict[str, Any]:
        """Gets method-specific parameters from the config, like PCoA distance metric."""
        params = {}
        if method == "pcoa":
            table_config = self.config.get('beta_diversity', {}).get('tables', {}).get(table_type, {})
            params["metric"] = table_config.get("pcoa_metric", "braycurtis")
        return params

    def _plot_ordination(self, result: Any, task: Tuple[str, str, str], metadata: pd.DataFrame, output_dir: Path) -> Dict[str, go.Figure]:
        """Generates and saves ordination plots for a single result."""
        table_type, _, method = task
        figures = {}
        valid_color_cols = [col for col in self.color_columns if col in metadata.columns]
        
        for color_col in valid_color_cols:
            try:
                fig = beta_diversity_plot(
                    components=result['components'],
                    proportion_explained=result.get('proportion_explained'),
                    metadata=metadata,
                    color_col=color_col,
                    symbol_col=self.symbol_column,
                    ordination_type=method,
                    transformation=table_type,
                    output_path=output_dir / f"{method}_{color_col}"
                )
                figures[color_col] = fig
            except Exception as e:
                logger.warning(f"Failed to plot {task} for color column '{color_col}': {e}")
                
        return figures

    def _get_enabled_tasks(self, beta_config: Dict) -> List[Tuple[str, str, str]]:
        """Parses the config to determine which ordination tasks to run."""
        tasks = []
        table_config = beta_config.get('tables', {})
        for table_type, levels in self.tables.items():
            type_conf = table_config.get(table_type, {})
            if type_conf.get('enabled', False):
                enabled_levels = type_conf.get('levels', list(levels.keys()))
                enabled_methods = type_conf.get('methods', [])
                for level in enabled_levels:
                    if level in levels:
                        for method in enabled_methods:
                            if method in self.ORDINATION_METHODS:
                                tasks.append((table_type, level, method))
        return tasks
    
    
if __name__ == "__main__":
    from workflow_16s.config import get_config # type: ignore
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
    logger.info("STEP 4: Running beta diversity analysis...")
    beta_diver = BetaDiversity(config, data_object, verbose=True)
    beta_diver.run(results_dir)
    logger.info("STEP 4: Beta diversity analysis complete.")