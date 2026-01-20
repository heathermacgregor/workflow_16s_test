# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
import os
import time
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
    create_alpha_diversity_boxplot, 
    create_alpha_diversity_stats_plot, 
    plot_alpha_correlations
)
from workflow_16s.stats.tests import (
    alpha_diversity, analyze_alpha_diversity, analyze_alpha_correlations
)
from workflow_16s.utils.data import table_to_df, update_table_and_meta
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# =================================== FUNCTIONS ====================================== #

class AlphaDiversity:
    def __init__(
        self, 
        config: Dict,
        meta: pd.DataFrame,
        tables: Dict[str, Dict[str, Table]],
        verbose: bool = False
    ):
        self.config = config
        self.meta = meta
        self.tables = tables
        self.verbose = verbose
        self.results = None
        
        alpha_config = self.config.get('alpha_diversity', {})
        if not alpha_config.get('enabled', False):
            logger.debug("Alpha diversity analysis disabled.")
            return 

        self.group_column = self.config.get(
            'group_column', constants.DEFAULT_GROUP_COLUMN
        )
        self.group_column_values = self.config.get(
            'group_column_values', constants.DEFAULT_GROUP_COLUMN_VALUES
        )
        
        self.metrics = alpha_config.get('metrics', constants.DEFAULT_ALPHA_METRICS)
        self.parametric = alpha_config.get('parametric', False)
        
        self.corr_config = alpha_config.get("correlation_analysis", {})
        self.plot_config = alpha_config.get("plots", {})
        
        self.tasks = self.get_enabled_tasks() 
        if len(self.tasks) == 0:
            logger.debug("No enabled tasks for alpha diversity analysis.")
            return 

        self.results = {}

    def run(
        self,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.run_for_col(self.group_column, output_dir)
        if 'facility_match' in self.meta.columns:
            self.run_for_col('facility_match', output_dir)
            
    def run_for_col(
        self,
        group_column: str = constants.DEFAULT_GROUP_COLUMN,
        output_dir: Optional[Path] = None,
    ) -> None:
        
        with get_progress_bar() as progress:
            alpha_desc = f"Running alpha diversity for '{group_column}'"
            alpha_task = progress.add_task(
                _format_task_desc(alpha_desc), 
                total=len(self.tasks)
            )
            for table_type, level in self.tasks:
                level_desc = (
                    f"{table_type.replace('_', ' ').title()} ({level.title()})"
                )
                progress.update(
                    alpha_task, 
                    description=_format_task_desc(level_desc)
                )
                
                # Initialize data storage
                _init_dict_level(self.results, group_column, table_type, level)
                data_storage = self.results[group_column][table_type][level]
                if output_dir:
                    # Initialize output directory and path
                    output_dir = output_dir / 'alpha_diversity' / table_type / level
                    output_dir.mkdir(parents=True, exist_ok=True)

                try:
                    # Prepare data
                    table = self.tables[table_type][level]
                    table_df = table_to_df(table)
                        
                    alpha_df = alpha_diversity(
                        table_df, 
                        metrics=self.metrics
                    )
                    stats_df = analyze_alpha_diversity(
                        alpha_diversity_df=alpha_df,
                        metadata=self.meta,
                        group_column=group_column,
                        parametric=self.parametric
                    )
                        
                    # Store and save results
                    data_storage['results'] = alpha_df
                    data_storage['stats'] = stats_df
                    if output_dir:
                        alpha_df.to_csv(
                            output_dir / 'alpha_diversity.tsv', 
                            sep='\t', index=True
                        )
                        stats_df.to_csv(
                            output_dir / f'stats_{group_column}.tsv', 
                            sep='\t', index=True
                        )
                    
                    if self.corr_config.get('enabled', False):
                        corr_results = analyze_alpha_correlations(
                            alpha_df,
                            self.meta,
                            max_categories=self.corr_config.get("max_categories", 20),
                            min_samples=self.corr_config.get("min_group_size", 5)
                        )
                        # Store and save results
                        data_storage['correlations'] = corr_results
                        if output_dir:
                            pd.DataFrame.from_dict(
                                [corr_results], orient='index'
                            ).to_csv(
                                output_dir / f'correlations_{group_column}.tsv', 
                                sep='\t', index=True
                            )

                    if self.plot_config.get('enabled', True):
                        data_storage['figures'] = {}
                        fig_storage = data_storage['figures']
                            
                        for metric in self.metrics:
                            if alpha_df[metric].isnull().all():
                                logger.error(
                                    f"All values NaN for metric {metric} in "
                                    f"{table_type}/{level}"
                                )
                                    
                            fig = create_alpha_diversity_boxplot(
                                alpha_df=alpha_df,
                                metadata=self.meta,
                                group_column=group_column,
                                metric=metric,
                                output_dir=output_dir,
                                show=False,
                                verbose=self.verbose,
                                add_points=self.plot_config.get('add_points', True),
                                add_stat_annot=self.plot_config.get('add_stat_annot', True),
                                test_type='parametric' if self.parametric else 'nonparametric'
                            )
                            fig_storage[metric] = fig
                            
                        stats_fig = create_alpha_diversity_stats_plot(
                            stats_df=stats_df,
                            output_dir=output_dir,
                            verbose=self.verbose,
                            effect_size_threshold=self.plot_config.get('effect_size_threshold', 0.5)
                        )
                        fig_storage['summary'] = stats_fig
                            
                        if self.corr_config.get('enabled', False):
                            corr_figures = plot_alpha_correlations(
                                corr_results,
                                output_dir=output_dir,
                                top_n=self.corr_config.get('top_n_correlations', 10)
                            )
                            fig_storage['correlations'] = corr_figures
                            
                except Exception as e:
                    logger.error(
                        f"Alpha diversity analysis failed for {table_type}/{level}: {str(e)}"
                    )
                    data_storage['results'] = None
                    
                finally:
                    progress.update(alpha_task, advance=1)
        progress.update(
            alpha_task, 
            description=_format_task_desc(alpha_desc)
        )    
        return self.results
        
    def get_enabled_tasks(self):
        alpha_config = self.config.get('alpha_diversity', {})
        table_config = alpha_config.get('tables', {})

        tasks = []
        for table_type, levels in self.tables.items():
            table_type_config = table_config.get(table_type, {})
            if not table_type_config.get('enabled', False):
                continue
            enabled_levels = [
                l for l in table_type_config.get('levels', levels.keys()) 
                if l in levels.keys()
            ]
            for level in enabled_levels:
                tasks.append((table_type, level))
        return tasks
