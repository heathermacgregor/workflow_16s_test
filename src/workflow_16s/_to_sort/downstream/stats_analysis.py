# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import hashlib
import json
import logging
import multiprocessing as mp
import os
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple

# Third‑Party Imports
import pandas as pd
import numpy as np
from biom.table import Table

# Local Imports
from workflow_16s import constants
from workflow_16s.amplicon_data.downstream.input import update_table_and_metadata
from workflow_16s.stats.test import (
    anova, core_microbiome, differential_abundance_analysis, 
    enhanced_statistical_tests, fisher_exact_bonferroni, kruskal_bonferroni, 
    microbial_network_analysis, mwu_bonferroni, ttest, spearman_correlation
)
from workflow_16s.stats.utils import validate_inputs
from workflow_16s.utils.data import (
    clr, collapse_taxa, filter, normalize, presence_absence, table_to_df,
    merge_table_with_meta, merge_data
)
from workflow_16s.utils.io import export_h5py
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc
from workflow_16s.constants import GROUP_COLUMNS, MODE
from workflow_16s.downstream.load_data import align_table_and_metadata
from workflow_16s.downstream.stats_helpers import (
    DataCache, LocalResultLoader, TestConfig, calculate_config_hash, get_enabled_tasks, 
    run_single_statistical_test
)
from workflow_16s.figures.downstream.stats import (
    volcano_plot, core_microbiome_barplot, network_plot, 
    correlation_heatmap, statistical_results_table, create_statistical_summary_dashboard
)
from workflow_16s.utils.metadata import get_group_column_values
from workflow_16s.stats.test import microbial_network_analysis

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ==================================================================================== #

CORE_MICROBIOME_PREVALENCE_THRESHOLD: float = 0.8
CORE_MICROBIOME_ABUNDANCE_THRESHOLD: float = 0.01
NETWORK_ANALYSIS_METHODS: List[str] = ['sparcc', 'spearman']
NETWORK_ANALYSIS_THRESHOLD: float = 0.3
N_TOP_FEATURES: int = 30

# ==================================================================================== #

def _init_nested_dict(dictionary: Dict, keys: List[str]) -> None:
    """Initialize nested dictionary levels efficiently."""
    current = dictionary
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current.setdefault(keys[-1], {})

# ==================================================================================== #

class TaskProcessor:
    def __init__(
        self, 
        project_dir: Any, 
        _data_cache: Any, 
        use_process_pool: Any,
        max_workers: Any,
        name: str, 
        values: List[Any],
        tables: Dict,
        metadata: Dict
    ):
        self.project_dir = project_dir
        self._data_cache = _data_cache
        self.use_process_pool = use_process_pool
        self.max_workers = max_workers
        self.name, self.values = name, values
        self.tables = tables
        self.metadata = metadata

    def _parallel(
        self, 
        tasks: List[Tuple[str, str, str]], 
    ) -> Dict:
        """Process tasks in parallel."""
        results = defaultdict(lambda: defaultdict(dict))
        
        if not tasks:
            return results
        
        # Prepare task data
        task_data_list = []
        for table_type, level, test in tasks:
            output_dir = self.project_dir / 'stats' / self.name / table_type / level
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get cached data
            table, metadata = self._get_cached_data(table_type, level)
            
            task_data = (
                table_type, level, test, table, metadata,
                self.name, self.values, output_dir
            )
            task_data_list.append(task_data)
        
        # Execute in parallel
        executor_class = ProcessPoolExecutor if self.use_process_pool else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(run_single_statistical_test, task_data): task_data
                for task_data in task_data_list
            }
            # Collect results with progress tracking
            with get_progress_bar() as progress:
                task_desc = f"Parallel analysis for '{self.name}' ({len(tasks)} tasks)"
                task_desc_fmt = _format_task_desc(task_desc)
                task_id = progress.add_task(task_desc_fmt, total=len(future_to_task))
                
                for future in as_completed(future_to_task):
                    task_result = future.result()
                    
                    # Store result and timing
                    results[task_result.table_type][task_result.level][task_result.test] = task_result.result
                    
                    # Record processing time
                    task_key = f"{task_result.table_type}_{task_result.level}_{task_result.test}"
                    self.analyzer.load_statistics['task_times'][task_key] = task_result.processing_time
                    
                    if task_result.error:
                        logger.warning(f"Task failed: {task_result.error}")
                    
                    progress.update(task_id, advance=1)
        
        return results
    
    def _sequential(self, tasks: List[Tuple[str, str, str]]) -> Dict:
        """Process tasks sequentially."""
        results = defaultdict(lambda: defaultdict(dict))
        
        if not tasks:
            return results
        
        with get_progress_bar() as progress:
            task_desc = f"Sequential analysis for '{self.name}' ({len(tasks)} tasks)"
            task_desc_fmt = _format_task_desc(task_desc)
            task_id = progress.add_task(task_desc_fmt, total=len(tasks))
            
            for table_type, level, test in tasks:
                output_dir = self.project_dir / 'stats' / self.name / table_type / level
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Get cached data
                table, metadata = self._get_cached_data(table_type, level)
                
                task_data = (
                    table_type, level, test, table, metadata,
                    self.name, self.values, output_dir
                )
                
                start_time = time.time()
                task_result = run_single_statistical_test(task_data)
                processing_time = time.time() - start_time
                
                # Store result and timing
                results[task_result.table_type][task_result.level][task_result.test] = task_result.result
                
                # Record processing time
                task_key = f"{task_result.table_type}_{task_result.level}_{task_result.test}"
                self.analyzer.load_statistics['task_times'][task_key] = processing_time
                
                if task_result.error:
                    logger.warning(f"Task failed: {task_result.error}")
                
                progress.update(task_id, advance=1)
        
        return results

    def _get_cached_data(self, table_type: str, level: str) -> Tuple[Table, pd.DataFrame]:
        """Get cached aligned table and metadata with better error handling."""
        cache_key = f"{table_type}_{level}"
        cached = self._data_cache.get(cache_key)
        
        if cached is not None:
            logger.debug(f"Cache hit for {cache_key}")
            return cached
        
        try:
            # Get original data
            raw_table = self.tables[table_type][level]
            raw_metadata = self.metadata[table_type][level]
            
            # Align data
            table, metadata = align_table_and_metadata(raw_table, raw_metadata)
            
            # Validate alignment
            if len(table.ids()) != len(metadata.index):
                logger.warning(f"Alignment issue: {len(table.ids())} samples in table vs {len(metadata.index)} in metadata")
            
            cached_data = (table, metadata)
            self._data_cache.put(cache_key, cached_data)
            return cached_data
            
        except KeyError as e:
            logger.error(f"Missing data for {cache_key}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to align data for {cache_key}: {e}")
            raise

# ==================================================================================== #

class AdvancedTaskProcessor:
    def __init__(
        self, 
        project_dir: Any, 
        _data_cache: Any, 
        tables: Dict,
        metadata: Dict,
        group_columns: Any
    ):
        self.project_dir = project_dir
        self._data_cache = _data_cache
        self.tables = tables
        self.metadata = metadata
        self.group_columns = group_columns
        self.results = {}

    def run_core_microbiome_analysis(
        self, 
        prevalence_threshold: float = CORE_MICROBIOME_PREVALENCE_THRESHOLD, 
        abundance_threshold: float = CORE_MICROBIOME_ABUNDANCE_THRESHOLD
    ) -> Dict:
        """Run core microbiome analysis for all groups."""
        core_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for group_column in self.group_columns:
            name = group_column['name']
            with get_progress_bar() as progress:
                main_desc = f"Running core microbiome analysis for '{name}'"
                main_desc_fmt = _format_task_desc(main_desc)
                main_n = len(self.tables) * len(self.tables['raw'])
                main_task = progress.add_task(main_desc_fmt, total=main_n)
                for table_type in self.tables: 
                    if table_type == "clr_transformed":
                        logger.debug(
                            f"Skipping core microbiome analysis for table type '{table_type}'. "
                            f"Will error due to float division by zero."
                        )
                        continue
                    for level in self.tables[table_type]:
                        level_desc = f"{name} / {table_type.replace('_', ' ').title()} ({level.title()})"
                        level_desc_fmt = _format_task_desc(level_desc)
                        progress.update(main_task, description=level_desc_fmt)
                        table, metadata = self._get_cached_data(table_type, level)
                        logger.info(table.shape)
                        logger.info(type(table))
                        logger.info(metadata.shape)
                        logger.info(type(metadata))
                        try:
                            core_features = core_microbiome(
                                table=table,
                                metadata=metadata,
                                group_column=name,
                                prevalence_threshold=prevalence_threshold,
                                abundance_threshold=abundance_threshold
                            )
                            # Store results
                            core_results[name][table_type][level]['features'] = core_features
                            # Save results
                            output_dir = self.project_dir / 'core_microbiome' / name / table_type / level
                            output_dir.mkdir(parents=True, exist_ok=True)
                            for group, core_df in core_features.items():
                                output_path = output_dir / f'core_features_{group}.tsv'
                                core_df.to_csv(output_path, sep='\t', index=False)

                            # Plot
                            fig = core_microbiome_barplot(
                                core_results=core_features,
                                output_dir=output_dir
                            )
                            # Store plot
                            core_results[name][table_type][level]['barplot'] = fig
                        except Exception as e:
                            logger.error(f"Core microbiome analysis failed for {name}/{table_type}/{level}: {e}")
                        finally:
                            progress.update(main_task, advance=1)
                progress.update(main_task, description=main_desc_fmt)
        self.results['core_microbiome'] = core_results
        return core_results

    def run_network_analysis(
        self, 
        methods: List[str] = NETWORK_ANALYSIS_METHODS, 
        threshold: float = NETWORK_ANALYSIS_THRESHOLD
    ) -> Dict:
        """Run network analysis for multiple correlation methods."""
        results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for method in methods:
            with get_progress_bar() as progress:
                main_desc = f"Running network analysis with '{method}'"
                main_desc_fmt = _format_task_desc(main_desc)
                main_n = len(self.tables) * len(self.tables['raw'])
                main_task = progress.add_task(main_desc_fmt, total=main_n)
                
                for table_type in self.tables:
                    for level in self.tables[table_type]:
                        if table_type == "clr_transformed_presence_absence":
                            logger.debug(
                                f"Skipping network analysis for table type '{table_type}' with '{method}'. "
                                f"Will error due to `abs_correlation`."
                            )
                            continue
                        # Use cached data
                        table, _ = self._get_cached_data(table_type, level)
                        
                        try:
                            corr_matrix, edges_df = microbial_network_analysis(
                                table=table,
                                method=method,
                                threshold=threshold
                            )                          
                            # Generate network statistics
                            network_stats = pd.DataFrame([self._calculate_network_statistics(edges_df)])
                            # Store results
                            results[table_type][level][method] = {
                                'correlation_matrix': corr_matrix,
                                'edges': edges_df,
                                'network_stats': network_stats
                            }
                            # Save results
                            output_dir = self.project_dir / 'networks' / table_type / level / method 
                            output_dir.mkdir(parents=True, exist_ok=True)
                        
                            corr_matrix.to_csv(output_dir / 'correlation_matrix.tsv', sep='\t')
                            edges_df.to_csv(output_dir / 'network_edges.tsv', sep='\t', index=False)
                            network_stats.to_csv(output_dir / 'network_statistics.tsv', sep='\t', index=False)

                            # Plot
                            fig = network_plot(
                                edges_df=edges_df,
                                network_stats=network_stats,
                                output_dir=output_dir
                            )
                            # Store plot
                            results[table_type][level][method]['plot'] = fig                            
                        except Exception as e:
                            logger.error(f"Network analysis failed for {method}/{table_type}/{level}: {e}")
                        finally:
                            progress.update(main_task, advance=1)
        self.results['networks'] = results
        return results
    
    def _calculate_network_statistics(self, edges_df: pd.DataFrame) -> Dict:
        """Calculate basic network statistics from edge list."""
        if edges_df.empty:
            return {
                'total_edges': 0,
                'positive_edges': 0,
                'negative_edges': 0,
                'mean_correlation': 0,
                'unique_nodes': 0
            }
        
        total_edges = len(edges_df)
        positive_edges = (edges_df['correlation'] > 0).sum()
        negative_edges = (edges_df['correlation'] < 0).sum()
        mean_correlation = edges_df['correlation'].mean()
        
        # Count unique nodes
        unique_nodes = len(
            set(edges_df['source'].tolist() + edges_df['target'].tolist())
        )
        
        return {
            'total_edges': total_edges,
            'positive_edges': positive_edges,
            'negative_edges': negative_edges,
            'mean_correlation': mean_correlation,
            'unique_nodes': unique_nodes,
            'density': total_edges / (unique_nodes * (unique_nodes - 1) / 2) if unique_nodes > 1 else 0
        }

    def run_batch_correlation_analysis(self, continuous_variables: List[str]) -> Dict:
        """Run correlation analysis for multiple continuous variables."""
        results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for var in continuous_variables:
            with get_progress_bar() as progress:
                main_desc = f"Running batch correlation analysis for '{var}'"
                main_desc_fmt = _format_task_desc(main_desc)
                main_n = len(self.tables) * len(self.tables['raw'])
                main_task = progress.add_task(main_desc_fmt, total=main_n)
                
                for table_type in self.tables:                        
                    for level in self.tables[table_type]:
                        table, metadata = self._get_cached_data(table_type, level)
                        # Check if variable exists in metadata
                        if var not in metadata.columns:
                            continue
                        # Filter out samples with missing values
                        metadata = metadata.dropna(subset=[var])
                        if len(metadata) < 5:  # Require min samples
                            logger.warning(f"Skipping {var}/{table_type}/{level}: only {len(metadata)} valid samples")
                            continue
                        table, metadata = align_table_and_metadata(table, metadata)
                        
                        try:
                            result = spearman_correlation(
                                table=table,
                                metadata=metadata,
                                continuous_column=var,
                                progress=progress,
                                task_id=main_task
                            )
                            results[var][table_type][level] = result
                            
                            # Save results
                            output_dir = self.project_dir / 'correlations' / var / table_type / level
                            output_dir.mkdir(parents=True, exist_ok=True)
                            result.to_csv(output_dir / 'spearman_correlations.tsv', sep='\t', index=False)
                            
                        except Exception as e:
                            logger.error(f"Correlation analysis failed for {var}/{table_type}/{level}: {e}")

                        finally:
                            progress.update(main_task, advance=1)
        
        self.results['correlations'] = results
        return results
        
    def _get_cached_data(self, table_type: str, level: str) -> Tuple[Table, pd.DataFrame]:
        """Get cached aligned table and metadata with better error handling."""
        cache_key = f"{table_type}_{level}"
        cached = self._data_cache.get(cache_key)
        
        if cached is not None:
            logger.debug(f"Cache hit for {cache_key}")
            return cached
        
        try:
            # Get original data
            raw_table = self.tables[table_type][level]
            raw_metadata = self.metadata[table_type][level]
            
            # Align data
            table, metadata = align_table_and_metadata(raw_table, raw_metadata)
            
            # Validate alignment
            if len(table.ids()) != len(metadata.index):
                logger.warning(f"Alignment issue: {len(table.ids())} samples in table vs {len(metadata.index)} in metadata")
            
            cached_data = (table, metadata)
            self._data_cache.put(cache_key, cached_data)
            return cached_data
            
        except KeyError as e:
            logger.error(f"Missing data for {cache_key}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to align data for {cache_key}: {e}")
            raise

# ==================================================================================== #

class StatisticalAnalysis:
    """Statistical Analysis class with result loading."""
    TestConfig = TestConfig
    def __init__(
        self,
        config: Dict,
        metadata: Dict,
        tables: Dict,
        project_dir: Any,
        use_process_pool: bool = False,
    ) -> None:
        self.config = config
        self.metadata, self.tables = metadata, tables
        self.project_dir = Path(project_dir.final) if hasattr(project_dir, 'final') else Path(project_dir)
        
        # Settings
        stats_config = self.config.get("stats", {})
        self.load_existing = stats_config.get("load_existing", False)
        self.max_file_age_hours = stats_config.get("max_file_age_hours", None) if self.load_existing else None
        self.force_recalculate = False # set(force_recalculate or [])
        self.max_workers = stats_config["max_workers"] if "max_workers" in stats_config else min(mp.cpu_count(), 8)
        self.use_process_pool = use_process_pool
        
        # Initialize caching and result loader
        self._data_cache = DataCache()
        self.result_loader = LocalResultLoader(
            base_path=self.project_dir / 'stats', 
            max_age_hours=self.max_file_age_hours
        ) if self.load_existing else None

        self.mode = self.config.get("target_subfragment_mode", MODE)
      
        self.group_columns = self.config.get("group_columns", GROUP_COLUMNS)
        # Add NFC facilities if enabled
        nfc_facilities_enabled = self.config.get("nfc_facilities", {}).get('enabled', False)
        nfc_facilities_column = 'facility_match' in self.metadata["raw"]["genus"].columns
        if nfc_facilities_enabled and nfc_facilities_column:
            self.group_columns.append({
                'name': 'facility_match', 
                'type': 'bool', 
                'values': [True, False]
            })

        # Initialize result storage
        self.results: Dict = {}
        self.load_statistics = {
            'total_tasks': 0,
            'loaded_from_files': 0,
            'calculated_fresh': 0,
            'load_time_saved_seconds': 0,
            'task_times': {}
        }
        
        # Pre-validate configuration
        validation_issues = self._validate_configuration()
        if validation_issues['errors']:
            logger.error(f"Configuration errors: {validation_issues['errors']}")
            raise ValueError("Configuration validation failed")
        
    def run(self):
        self._run_basic_analysis()
        self._get_top_features_across_tests()
        advanced_analysis_config = self.config.get("stats", {}).get("advanced_analysis", {})
        if advanced_analysis_config.get("enabled", False):
            advanced_processor = AdvancedTaskProcessor(
                project_dir=self.project_dir,
                _data_cache=self._data_cache,
                tables=self.tables,
                metadata=self.metadata,
                group_columns=self.group_columns
            )
            core_microbiome_config = advanced_analysis_config.get("core_microbiome_analysis", {})
            if core_microbiome_config.get("enabled", False):
                advanced_processor.run_core_microbiome_analysis(
                    prevalence_threshold=core_microbiome_config.get("prevalence_threshold", CORE_MICROBIOME_PREVALENCE_THRESHOLD), 
                    abundance_threshold=core_microbiome_config.get("abundance_threshold", CORE_MICROBIOME_ABUNDANCE_THRESHOLD)
                )
            network_analysis_config = advanced_analysis_config.get("network_analysis", {})
            if network_analysis_config.get("enabled", False):
                advanced_processor.run_network_analysis(
                    methods=network_analysis_config.get("methods", NETWORK_ANALYSIS_METHODS), 
                    threshold=network_analysis_config.get("theshold", NETWORK_ANALYSIS_THRESHOLD)
                )
            batch_correlation_config = advanced_analysis_config.get("batch_correlation", {})
            if batch_correlation_config.get("enabled", False):
                advanced_processor.run_batch_correlation_analysis(
                    continuous_variables=batch_correlation_config.get("continuous_variables", ['ph', 'distance_from_facility_km']), 
                )
            self.results['advanced'] = advanced_processor.results
      
    def _get_cached_data(self, table_type: str, level: str) -> Tuple[Table, pd.DataFrame]:
        """Get cached aligned table and metadata with better error handling."""
        cache_key = f"{table_type}_{level}"
        cached = self._data_cache.get(cache_key)
        
        if cached is not None:
            logger.debug(f"Cache hit for {cache_key}")
            return cached
        
        try:
            # Get original data
            raw_table = self.tables[table_type][level]
            raw_metadata = self.metadata[table_type][level]
            
            # Align data
            table, metadata = align_table_and_metadata(raw_table, raw_metadata)
            
            # Validate alignment
            if len(table.ids()) != len(metadata.index):
                logger.warning(f"Alignment issue: {len(table.ids())} samples in table vs {len(metadata.index)} in metadata")
            
            cached_data = (table, metadata)
            self._data_cache.put(cache_key, cached_data)
            return cached_data
            
        except KeyError as e:
            logger.error(f"Missing data for {cache_key}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to align data for {cache_key}: {e}")
            raise
    
    def validate_table_retrieval(self):
        """Validate that all tables can be retrieved successfully."""
        logger.info("Validating table retrieval...")
        for table_type in self.tables:
            for level in self.tables[table_type]:
                try:
                    table, metadata = self._get_cached_data(table_type, level)
                    logger.info(f"Successfully retrieved {table_type}_{level}: "
                               f"{table.shape[0]} samples, {table.shape[1]} features")
                except Exception as e:
                    logger.error(f"Failed to retrieve {table_type}_{level}: {e}")
    
    def clear_cache(self):
        """Clear the data cache."""
        self._data_cache.clear()
        logger.info("Data cache cleared")
    
    def _task_should_force_recalculate(self, group_column: str, table_type: str, level: str, test: str) -> bool:
        """Check if a specific task should be force recalculated."""
        if not self.force_recalculate:
            return False
        
        # Check various patterns for force recalculation
        patterns = [
            f"{group_column}_{table_type}_{level}_{test}",  # Specific task
            f"{table_type}_{level}_{test}",                 # Specific test
            f"{group_column}_{test}",                       # Test for specific group
            test,                                           # All instances of test
            f"{table_type}_{level}",                        # All tests for table/level
            group_column,                                   # All tests for group
            table_type                                      # All tests for table type
        ]
        
        return any(pattern in self.force_recalculate for pattern in patterns)
    
    def _run_basic_analysis(self) -> None:
        """Run analysis with parallel processing and result loading."""
        start_time = time.time()
        self.results["basic"] = {}
        # Process each group column
        for group_column in self.group_columns:
            name = group_column['name'] if 'name' in group_column else group_column
            values = group_column['values'] if 'values' in group_column else get_group_column_values(group_column, self.metadata["raw"]["genus"])
            logger.info(f"Column: {name}; Values: {values}")
            self.results["basic"][name] = self._run_group_column_in_parallel(name, values)
        
        # Calculate time savings
        end_time = time.time()
        analysis_time = end_time - start_time
        self.load_statistics['analysis_time_seconds'] = analysis_time
        
        # Log loading statistics
        self._log_load_statistics()

    def _attempt_to_load_existing(self, tasks, name) -> Tuple:
        existing_results, remaining_tasks = {}, []
        # Load existing results if enabled
        if self.load_existing and self.result_loader:
            # Calculate configuration hash for validation
            config_hashes = {}
            for table_type, level, test in tasks:
                config_hashes[(table_type, level, test)] = calculate_config_hash(
                    self.config, name, table_type, level, test
                )
            
            existing_results = self.result_loader.get_existing_results(
                name, tasks, config_hash=config_hashes.get((table_type, level, test), None)
            )
            
            # Filter out tasks that have existing results and don't need to be recalculated
            for table_type, level, test in tasks:
                recalculate_task = self._task_should_force_recalculate(name, table_type, level, test)
                
                if (not recalculate_task and 
                    table_type in existing_results and 
                    level in existing_results[table_type] and 
                    test in existing_results[table_type][level]):
                    self.load_statistics['loaded_from_files'] += 1
                    logger.info(f"Loaded existing result: {name}/{table_type}/{level}/{test}")
                else:
                    remaining_tasks.append((table_type, level, test))
                    if recalculate_task:
                        logger.info(f"Force recalculating: {name}/{table_type}/{level}/{test}")
        else:
            remaining_tasks = tasks
        return existing_results, remaining_tasks
        
    def _run_group_column_in_parallel(self, name: str, values: List[Any]) -> Dict:
        """Run statistical analysis with parallel processing and result loading."""
        tasks = get_enabled_tasks(config=self.config, tables=self.tables)
        if not tasks:
            return {}
        self.load_statistics['total_tasks'] += len(tasks)
        existing_results, remaining_tasks = self._attempt_to_load_existing(tasks, name)
        self.load_statistics['calculated_fresh'] += len(remaining_tasks)
        # Separate parallel-safe and sequential tasks from remaining tasks
        parallel_tasks, sequential_tasks = [], []
        
        for table_type, level, test in remaining_tasks:
            if TestConfig[test].get('parallel_safe', True):
                parallel_tasks.append((table_type, level, test))
            else:
                sequential_tasks.append((table_type, level, test))
        
        results = existing_results.copy()  # Start with loaded results

        processor = TaskProcessor(
            project_dir=self.project_dir,
            _data_cache=self._data_cache,
            use_process_pool=self.use_process_pool,
            max_workers=self.max_workers,
            name=name, 
            values=values,
            tables=self.tables,
            metadata=self.metadata
        )
        # Process parallel tasks
        if parallel_tasks:
            calculated_results = processor._parallel(parallel_tasks)
            results = self._merge_results(results, calculated_results)
        
        # Process sequential tasks
        if sequential_tasks:
            calculated_results = processor._sequential(sequential_tasks)
            results = self._merge_results(results, calculated_results)
        
        return results

    def _get_top_features_across_tests(self, n_features: int = N_TOP_FEATURES) -> pd.DataFrame:
        """Get top features that appear consistently across multiple tests."""
        feature_counts, feature_effects = {}, {}
        for group_col, group_results in self.results["basic"].items():
            for table_type, levels in group_results.items():
                for level, tests in levels.items():
                    for test_name, result in tests.items():
                        if isinstance(result, pd.DataFrame) and not result.empty:
                            for _, row in result.iterrows():
                                feature = row.get('feature', '')
                                if feature:
                                    # Count occurrences
                                    if feature not in feature_counts:
                                        feature_counts[feature] = 0
                                        feature_effects[feature] = []
                                    
                                    feature_counts[feature] += 1
                                    
                                    # Store effect sizes
                                    effect_size = self.get_effect_size(test_name, row)
                                    if effect_size is not None:
                                        feature_effects[feature].append(abs(effect_size))
        
        # Create summary DataFrame
        summary_data = []
        for feature, count in feature_counts.items():
            effects = feature_effects[feature]
            summary_data.append({
                'feature': feature,
                'test_count': count,
                'mean_effect_size': np.mean(effects) if effects else 0,
                'max_effect_size': np.max(effects) if effects else 0,
                'effect_size_std': np.std(effects) if len(effects) > 1 else 0
            })
        
        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty:
            # Sort by test count and mean effect size
            summary_df = summary_df.sort_values(
                ['test_count', 'mean_effect_size'], 
                ascending=[False, False]
            ).head(n_features)
        self.results["basic"]["top_features"] = summary_df
        return summary_df

    def get_effect_size(self, test_name: str, row: pd.Series) -> Optional[float]:
        """Optimized effect size extraction."""
        test_config = self.TestConfig.get(test_name)
        if not test_config:
            return None
        
        # Check primary effect column first
        effect_col = test_config["effect_col"]
        if effect_col and effect_col in row and pd.notna(row[effect_col]):
            return float(row[effect_col])
        
        # Check alternative effect column
        alt_col = test_config["alt_effect_col"]
        if alt_col and alt_col in row and pd.notna(row[alt_col]):
            return float(row[alt_col])
        
        return None

    def _merge_results(self, existing: Dict, new: Dict) -> Dict:
        """Merge existing and newly calculated results."""
        merged = existing.copy()
        
        for table_type, levels in new.items():
            if table_type not in merged:
                merged[table_type] = {}
            
            for level, tests in levels.items():
                if level not in merged[table_type]:
                    merged[table_type][level] = {}
                
                merged[table_type][level].update(tests)
        
        return merged

    def _validate_configuration(self) -> Dict[str, List[str]]:
        """Optimized configuration validation."""
        issues = {'errors': [], 'warnings': [], 'info': []}
        
        # Batch validation of tables/metadata alignment
        alignment_tasks = [
            (table_type, level, self.tables[table_type][level], self.metadata[table_type][level])
            for table_type in self.tables
            for level in self.tables[table_type]
        ]
        
        for table_type, level, table, metadata in alignment_tasks:
            try:
                update_table_and_metadata(table, metadata)
            except Exception as e:
                issues['errors'].append(f"Alignment failed for {table_type}/{level}: {e}")
        
        # Vectorized group column validation
        for group_column in self.group_columns:
            col_name = group_column['name']
            found_locations = []
            
            for table_type in self.metadata:
                for level in self.metadata[table_type]:
                    metadata = self.metadata[table_type][level]
                    if col_name in metadata.columns:
                        found_locations.append((table_type, level))
                        
                        # Efficient group size checking
                        group_counts = metadata[col_name].value_counts()
                        small_groups = group_counts[group_counts < 3]
                        if not small_groups.empty:
                            issues['warnings'].append(
                                f"Small groups in '{col_name}' at {table_type}/{level}: {dict(small_groups)}"
                            )
            
            if not found_locations:
                issues['errors'].append(f"Group column '{col_name}' not found")
        
        return issues

    def _log_load_statistics(self) -> None:
        """Log statistics about loaded vs calculated results."""
        stats = self.load_statistics
        total = stats['total_tasks']
        loaded = stats['loaded_from_files']
        calculated = stats['calculated_fresh']
        
        if total > 0:
            load_percentage = (loaded / total) * 100
            logger.info(
                f"Analysis Statistics:\n"
                f"  Total tasks: {total}\n"
                f"  Loaded from files: {loaded} ({load_percentage:.1f}%)\n"
                f"  Calculated fresh: {calculated} ({100-load_percentage:.1f}%)\n"
                f"  Total analysis time: {stats.get('analysis_time_seconds', 0):.2f} seconds"
            )
            
            # Log timing statistics for calculated tasks
            if stats['task_times']:
                avg_time = sum(stats['task_times'].values()) / len(stats['task_times'])
                max_time = max(stats['task_times'].values())
                min_time = min(stats['task_times'].values())
                
                logger.info(
                    f"  Task timing (calculated tasks):\n"
                    f"    Average: {avg_time:.2f}s, Min: {min_time:.2f}s, Max: {max_time:.2f}s"
                )
            
            if loaded > 0:
                # Rough estimate of time saved (using average time for loaded tasks)
                avg_task_time = (sum(stats['task_times'].values()) / len(stats['task_times'])) if stats['task_times'] else 10
                estimated_time_saved = loaded * avg_task_time
                logger.info(
                    f"  Estimated time saved: ~{estimated_time_saved:.0f} seconds"
                )

# ==================================================================================== #

def run_statistical_analysis(
    config: Dict,
    metadata: Dict,
    tables: Dict,
    project_dir: Any,
    use_process_pool: bool = False,
):
    """Convenience function to run statistical analysis with loading options.
    
    Args:
        config:           Analysis configuration
        metadata:         Dictionary of metadata
        tables:           Dictionary of tables
        project_dir:      Project directory path
        use_process_pool:
    
    Returns:
        StatisticalAnalysis instance with results
    """
    analyzer = StatisticalAnalysis(
        config=config,
        metadata=metadata,
        tables=tables,
        project_dir=project_dir,
        use_process_pool=use_process_pool
    )
    
    # Validate table retrieval before running analysis
    analyzer.validate_table_retrieval()
    
    analyzer.run()
    return analyzer

# TODO: More evil functions
'''        
# After running differential abundance analysis
volcano_fig = volcano_plot(
    results_df=differential_results,
    output_dir=self.project_dir / 'stats' / 'visualizations'
)

# Create a comprehensive dashboard
dashboard_fig = create_statistical_summary_dashboard(
    statistical_results=self.results,
    output_dir=self.project_dir / 'stats' / 'visualizations'
)
    def run_comprehensive_analysis(self, **kwargs) -> Dict:
        """Run all available advanced analyses."""
        comprehensive_results = {}
        
        logger.info("Starting comprehensive statistical analysis...")
        
        # 1. Core microbiome analysis
        logger.info("Running core microbiome analysis...")
        try:
            core_results = self.run_core_microbiome_analysis(
                prevalence_threshold=kwargs.get('prevalence_threshold', 0.8),
                abundance_threshold=kwargs.get('abundance_threshold', 0.01)
            )
            comprehensive_results['core_microbiome'] = core_results
        except Exception as e:
            logger.error(f"Core microbiome analysis failed: {e}")
        
        # 2. Correlation analysis
        continuous_vars = kwargs.get('continuous_variables', 
                                   self.config.get('stats', {}).get('continuous_variables', ['ph', 'distance_from_facility_km']))
        if continuous_vars:
            logger.info("Running correlation analysis...")
            try:
                correlation_results = self.run_batch_correlation_analysis(continuous_vars)
                comprehensive_results['correlations'] = correlation_results
            except Exception as e:
                logger.error(f"Correlation analysis failed: {e}")
        
        # 3. Network analysis
        logger.info("Running network analysis...")
        try:
            network_methods = kwargs.get('network_methods', ['sparcc', 'spearman'])
            network_threshold = kwargs.get('network_threshold', 0.3)
            network_results = self.run_network_analysis_batch(
                methods=network_methods,
                threshold=network_threshold
            )
            comprehensive_results['networks'] = network_results
        except Exception as e:
            logger.error(f"Network analysis failed: {e}")
        
        # 4. Generate summary
        logger.info("Generating comprehensive summary...")
        try:
            summary_path = self.project_dir / 'comprehensive_analysis_summary.md'
            self.export_results_summary(summary_path)
            comprehensive_results['summary_path'] = str(summary_path)
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
        
        # Store all results
        self.advanced_results.update(comprehensive_results)
        
        logger.info("Comprehensive analysis completed!")
        return comprehensive_results
    
    def export_results_summary(self, output_path: Union[str, Path]) -> None:
        """Export a comprehensive summary of all results."""
        summary = self.get_summary_statistics()
        
        # Create summary report
        report_lines = [
            "# Statistical Analysis Summary Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total tests executed: {summary['total_tests_run']}",
            f"Group columns analyzed: {', '.join(summary['group_columns_analyzed'])}",
            "",
            "## Performance Metrics",
            f"Analysis time: {summary['performance_metrics'].get('analysis_time_seconds', 0):.2f} seconds",
            f"Cache hit ratio: {summary['performance_metrics'].get('cache_hit_ratio', 0):.2%}",
            f"Results loaded from files: {summary['performance_metrics'].get('load_statistics', {}).get('loaded_from_files', 0)}",
            f"Results calculated fresh: {summary['performance_metrics'].get('load_statistics', {}).get('calculated_fresh', 0)}",
            "",
            "## Significant Features by Test Type",
        ]
        
        for test_name, count in summary['significant_features_by_test'].items():
            test_display_name = TEST_CONFIG.get(test_name, {}).get('name', test_name)
            report_lines.append(f"- {test_display_name}: {count} significant features")
        
        report_lines.extend([
            "",
            "## Effect Size Summaries",
        ])
        
        for test_name, stats in summary['effect_sizes_summary'].items():
            test_display_name = TEST_CONFIG.get(test_name, {}).get('name', test_name)
            report_lines.extend([
                f"### {test_display_name}",
                f"- Mean effect size: {stats['mean']:.4f}",
                f"- Standard deviation: {stats['std']:.4f}",
                f"- Range: {stats['min']:.4f} to {stats['max']:.4f}",
                f"- Count: {stats['count']}",
                ""
            ])
        
        # Write summary report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Summary report exported to {output_path}")
    
    def get_summary_statistics(self) -> Dict:
        """Generate summary statistics with vectorized operations."""
        summary = {
            'total_tests_run': 0,
            'significant_features_by_test': {},
            'effect_sizes_summary': {},
            'group_columns_analyzed': list(self.results.keys()),
            'performance_metrics': {
                'cache_hit_ratio': len(self._data_cache._cache) / max(1, len(self._data_cache._access_order)),
                'load_statistics': self.load_statistics.copy()
            }
        }
        
        # Vectorized processing
        all_results = []
        
        for group_col, group_results in self.results.items():
            for table_type, levels in group_results.items():
                for level, tests in levels.items():
                    for test_name, result in tests.items():
                        if isinstance(result, pd.DataFrame) and not result.empty:
                            summary['total_tests_run'] += 1
                            
                            # Count significant features
                            test_key = test_name
                            if test_key not in summary['significant_features_by_test']:
                                summary['significant_features_by_test'][test_key] = 0
                            summary['significant_features_by_test'][test_key] += len(result)
                            
                            # Collect effect sizes for vectorized processing
                            effect_col = TEST_CONFIG.get(test_name, {}).get('effect_col')
                            if effect_col and effect_col in result.columns:
                                all_results.append({
                                    'test': test_name,
                                    'effects': result[effect_col].dropna().values
                                })
        
        # Vectorized effect size summary computation
        for result_data in all_results:
            test_name = result_data['test']
            effects = result_data['effects']
            
            if len(effects) > 0:
                if test_name not in summary['effect_sizes_summary']:
                    summary['effect_sizes_summary'][test_name] = {
                        'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0
                    }
                
                # Use numpy for fast computation
                summary['effect_sizes_summary'][test_name] = {
                    'mean': float(np.mean(effects)),
                    'std': float(np.std(effects)),
                    'min': float(np.min(effects)),
                    'max': float(np.max(effects)),
                    'count': len(effects)
                }
        
        return summary

    def get_analysis_recommendations(self) -> List[str]:
        """Provide analysis recommendations based on data characteristics."""
        recommendations = []
        
        # Analyze data characteristics
        total_samples = 0
        total_features = 0
        
        for table_type in self.tables:
            for level in self.tables[table_type]:
                table = self.tables[table_type][level]
                df = table_to_df(table)
                total_samples += len(df)
                total_features += len(df.columns)
        
        avg_samples = total_samples / (len(self.tables) * max(1, len(self.tables.get(list(self.tables.keys())[0], {}))))
        avg_features = total_features / (len(self.tables) * max(1, len(self.tables.get(list(self.tables.keys())[0], {}))))
        
        # Sample size recommendations
        if avg_samples < 20:
            recommendations.append(
                "Small sample size detected. Consider using non-parametric tests "
                "(Mann-Whitney U, Kruskal-Wallis) instead of parametric tests."
            )
        
        if avg_samples > 100:
            recommendations.append(
                "Large sample size detected. Both parametric and non-parametric tests "
                "should be reliable. Consider using enhanced statistical tests for "
                "automatic test selection."
            )
        
        # Feature recommendations
        if avg_features > 1000:
            recommendations.append(
                "High-dimensional data detected. Consider using differential abundance "
                "analysis with appropriate multiple testing correction."
            )
        
        # Group structure recommendations
        group_structures = []
        for group_column in self.group_columns:
            col_name = group_column['name']
            for table_type in self.metadata:
                for level in self.metadata[table_type]:
                    metadata = self.metadata[table_type][level]
                    if col_name in metadata.columns:
                        n_groups = metadata[col_name].nunique()
                        group_structures.append(n_groups)
                        break
        
        if any(n > 2 for n in group_structures):
            recommendations.append(
                "Multiple groups detected. Consider using ANOVA or Kruskal-Wallis tests "
                "for overall group differences, followed by post-hoc pairwise comparisons."
            )
        
        # Network analysis recommendations
        if avg_features > 50:
            recommendations.append(
                "Sufficient features for network analysis. Consider running microbial "
                "co-occurrence network analysis to identify feature interactions."
            )
        
        return recommendations
    
    def force_recalculate_tasks(self, patterns: List[str]) -> None:
        """Add patterns to force recalculation list."""
        self.force_recalculate.update(patterns)
        logger.info(f"Added force recalculation patterns: {patterns}")
    
    def clear_result_cache(self, group_column: str = None) -> None:
        """Clear cached results for a specific group or all groups."""
        if self.result_loader:
            if group_column:
                # Clear specific group cache
                cache_keys_to_remove = [
                    key for key in self.result_loader._load_cache.keys()
                    if f"/{group_column}/" in key
                ]
                for key in cache_keys_to_remove:
                    del self.result_loader._load_cache[key]
                logger.info(f"Cleared result cache for group: {group_column}")
            else:
                self.result_loader.clear_cache()
                logger.info("Cleared all result caches")
    
    def get_load_report(self) -> Dict:
        """Get detailed report of what was loaded vs calculated."""
        return {
            'summary': self.load_statistics.copy(),
            'settings': {
                'load_existing': self.load_existing,
                'max_file_age_hours': self.max_file_age_hours,
                'force_recalculate_patterns': list(self.force_recalculate)
            }
        }
    
    def invalidate_results(
        self, 
        group_column: str = None, 
        table_type: str = None, 
        level: str = None, 
        test: str = None
    ) -> int:
        """Invalidate (delete) specific result files to force recalculation.
        
        Returns the number of files deleted.
        """
        deleted_count = 0
        stats_dir = self.project_dir / 'stats'
        
        if not stats_dir.exists():
            return 0
        
        # Build search patterns
        if group_column and table_type and level and test:
            # Specific test file
            file_path = stats_dir / group_column / table_type / level / f'{test}.tsv'
            if file_path.exists():
                file_path.unlink()
                deleted_count += 1
                # Also delete correlation matrix for network analysis
                if test == 'network_analysis':
                    corr_path = stats_dir / group_column / table_type / level / f'{test}_correlation_matrix.tsv'
                    if corr_path.exists():
                        corr_path.unlink()
                        deleted_count += 1
        else:
            # Pattern-based deletion
            for group_dir in stats_dir.iterdir():
                if group_dir.is_dir() and (not group_column or group_dir.name == group_column):
                    for table_dir in group_dir.iterdir():
                        if table_dir.is_dir() and (not table_type or table_dir.name == table_type):
                            for level_dir in table_dir.iterdir():
                                if level_dir.is_dir() and (not level or level_dir.name == level):
                                    for result_file in level_dir.glob('*.tsv'):
                                        if not test or result_file.stem == test or result_file.stem.startswith(f'{test}_'):
                                            result_file.unlink()
                                            deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"Invalidated {deleted_count} result files")
            # Clear relevant cache entries
            self.clear_result_cache(group_column)
        
        return deleted_count
    
    
    
    def cleanup(self) -> None:
        """Clean up resources and caches."""
        self._data_cache.clear()
        if self.result_loader:
            self.result_loader.clear_cache()
        logger.info("Statistical analysis cleanup completed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

# ========================== UTILITY FUNCTIONS ========================== #

def batch_save_results(results: Dict, base_path: Path, format: str = 'tsv') -> None:
    """Efficiently batch save results."""
    save_tasks = []
    
    for group_col, group_results in results.items():
        for table_type, levels in group_results.items():
            for level, tests in levels.items():
                for test_name, result in tests.items():
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        output_dir = base_path / group_col / table_type / level
                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_path = output_dir / f'{test_name}.{format}'
                        save_tasks.append((result, output_path))
    
    # Execute saves in parallel if beneficial
    if len(save_tasks) > 10:  # Threshold for parallel saving
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(lambda r, p: r.to_csv(p, sep='\t', index=True), result, path)
                for result, path in save_tasks
            ]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Save failed: {e}")
    else:
        # Sequential save for small numbers
        for result, path in save_tasks:
            try:
                result.to_csv(path, sep='\t', index=True)
            except Exception as e:
                logger.error(f"Save failed for {path}: {e}")

def run_statistical_analysis_with_loading(
    config: Dict,
    tables: Dict,
    metadata: Dict,
    mode: str,
    group_columns: List,
    project_dir: Union[str, Path],
    load_existing: bool = True,
    max_file_age_hours: Optional[float] = 24,
    force_recalculate: List[str] = None,
    **kwargs
) -> StatisticalAnalysis:
    """Convenience function to run statistical analysis with loading options.
    
    Args:
        config: Analysis configuration
        tables: Dictionary of tables
        metadata: Dictionary of metadata
        mode: Analysis mode
        group_columns: List of group columns to analyze
        project_dir: Project directory path
        load_existing: Whether to load existing results (default: True)
        max_file_age_hours: Maximum age of files to load (None for no limit)
        force_recalculate: List of patterns to force recalculation
        **kwargs: Additional arguments passed to StatisticalAnalysis
    
    Returns:
        StatisticalAnalysis instance with results
    """
    return StatisticalAnalysis(
        config=config,
        tables=tables,
        metadata=metadata,
        mode=mode,
        group_columns=group_columns,
        project_dir=project_dir,
        load_existing=load_existing,
        max_file_age_hours=max_file_age_hours,
        force_recalculate=force_recalculate or [],
        **kwargs
    )
'''
