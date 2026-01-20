# ===================================== IMPORTS ====================================== #
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from biom.table import Table

# Assume these local imports exist in your project structure
from .downstream import Data
from workflow_16s.downstream.statistics import advanced_analyses
from workflow_16s.utils.data import sync_samples 
from workflow_16s.visualization.statistics import core_microbiome_barplot, network_plot, correlation_heatmap

# ========================== INITIALISATION & CONFIGURATION ========================== #
logger = logging.getLogger("workflow_16s")

# =================================== CLASS ====================================== #

class AdvancedAnalyses:
    """
    Performs advanced analyses like core microbiome, network analysis,
    and batch correlations, driven by a configuration file.
    """
    def __init__(self, config: Dict, data: Data, verbose: bool = False):
        self.config = config
        self.data = data
        self.metadata = data.metadata
        self.tables = data.tables
        self.verbose = verbose
        self.advanced_config = self.config.get('advanced_analyses', {})

    def run(self, output_dir: Path):
        """Executes all enabled advanced analysis tasks sequentially."""
        if not self.advanced_config.get('enabled', False):
            logger.debug("Advanced statistical analyses disabled in config.")
            return

        if self.advanced_config.get('core_microbiome', {}).get('enabled', False):
            self._run_core_microbiome(self.advanced_config['core_microbiome'], output_dir)
            
        if self.advanced_config.get('network_analysis', {}).get('enabled', False):
            self._run_network_analysis(self.advanced_config['network_analysis'], output_dir)
        
        if self.advanced_config.get('batch_correlation', {}).get('enabled', False):
            self._run_batch_correlations(self.advanced_config['batch_correlation'], output_dir)

    def _run_core_microbiome(self, core_config: Dict, base_output_dir: Path):
        """Calculates the core microbiome for specified groups and data types."""
        logger.info("Running Core Microbiome Analysis...")
        group_columns = core_config.get('group_columns', [])
        prevalence = core_config.get('prevalence_threshold', 0.8)
        abundance = core_config.get('abundance_threshold', 0.01)
        output_dir = base_output_dir / "advanced" / "core_microbiome"

        for group_col in group_columns:
            for table_type, levels in self._get_enabled_tables(core_config):
                for level in levels:
                    if self.metadata is None: continue
                    table, metadata = sync_samples(self.tables[table_type][level], self.metadata)
                    if table.is_empty(): continue
                    
                    try:
                        result = advanced_analyses.core_microbiome(table, metadata, group_col, prevalence, abundance)
                        self.data.analysis_results['advanced_analyses']['core_microbiome'][group_col][table_type][level] = result
                        core_microbiome_barplot(result, output_dir=output_dir / group_col / table_type / level)
                    except Exception as e:
                        logger.error(f"Core microbiome for {group_col}/{table_type}/{level} failed: {e}")

    def _run_network_analysis(self, network_config: Dict, base_output_dir: Path):
        """Performs microbial network analysis."""
        logger.info("Running Network Analysis...")
        methods = network_config.get('methods', [])
        threshold = network_config.get('threshold', 0.3)
        output_dir = base_output_dir / "advanced" / "network_analysis"
        
        for method in methods:
            for table_type, levels in self._get_enabled_tables(network_config):
                for level in levels:
                    if self.metadata is None: continue
                    table, _ = sync_samples(self.tables[table_type][level], self.metadata)
                    if table.is_empty(): continue

                    try:
                        corr_matrix, edges_df = advanced_analyses.microbial_network_analysis(table, method, threshold)
                        network_stats = advanced_analyses.calculate_network_statistics(edges_df)
                        result = {'correlation_matrix': corr_matrix, 'edges': edges_df, 'statistics': network_stats}
                        self.data.analysis_results['advanced_analyses']['network_analysis'][method][table_type][level] = result
                        
                        task_dir = output_dir / method / table_type / level
                        task_dir.mkdir(parents=True, exist_ok=True)
                        edges_df.to_csv(task_dir / "network_edges.tsv", sep="\t")
                        network_stats.to_csv(task_dir / "network_statistics.tsv", sep="\t")
                        network_plot(edges_df, network_stats, output_dir=task_dir)
                    except Exception as e:
                        logger.error(f"Network analysis ({method}) for {table_type}/{level} failed: {e}")

    def _run_batch_correlations(self, corr_config: Dict, base_output_dir: Path):
        """Runs Spearman correlation against multiple continuous metadata variables."""
        logger.info("Running Batch Correlation Analysis...")
        variables = corr_config.get('continuous_variables', [])
        output_dir = base_output_dir / "advanced" / "batch_correlation"

        for var in variables:
            for table_type, levels in self._get_enabled_tables(corr_config):
                for level in levels:
                    if self.metadata is None or var not in self.metadata.columns: continue
                    
                    clean_metadata = self.metadata.dropna(subset=[var])
                    table, metadata = sync_samples(self.tables[table_type][level], clean_metadata)
                    if table.is_empty() or len(metadata) < 5: continue

                    try:
                        result = advanced_analyses.spearman_correlation(table, metadata, var)
                        self.data.analysis_results['advanced_analyses']['batch_correlation'][var][table_type][level] = result
                        
                        task_dir = output_dir / var / table_type / level
                        task_dir.mkdir(parents=True, exist_ok=True)
                        result.to_csv(task_dir / "correlations.tsv", sep="\t")
                        correlation_heatmap(result, output_dir=task_dir)
                    except Exception as e:
                        logger.error(f"Batch correlation for {var}/{table_type}/{level} failed: {e}")

    def _get_enabled_tables(self, analysis_config: Dict) -> List[Tuple[str, List[str]]]:
        """Helper to parse which tables and levels are enabled for an analysis."""
        enabled_tasks = []
        table_config = analysis_config.get('tables', {})
        for table_type, type_conf in table_config.items():
            if type_conf.get('enabled', False) and table_type in self.tables:
                enabled_levels = [lvl for lvl in type_conf.get('levels', []) if lvl in self.tables[table_type]]
                if enabled_levels:
                    enabled_tasks.append((table_type, enabled_levels))
        return enabled_tasks