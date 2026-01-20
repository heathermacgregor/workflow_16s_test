# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import hashlib
import json
import logging
import multiprocessing as mp
import os
import time
import warnings
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
    merge_table_with_meta
)
from workflow_16s.utils.io import export_h5py
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ============================== OPTIMIZATION UTILITIES ============================== #

class TaskResult(NamedTuple):
    """Structured result for parallel tasks."""
    task_id: str
    table_type: str
    level: str
    test: str
    result: Optional[pd.DataFrame]
    error: Optional[str]
    loaded_from_file: bool = False
    processing_time: float = 0.0
    

class DataCache:
    """Lightweight caching for preprocessed data with memory optimization."""
    def __init__(self, max_size: int = 128):
        self._cache = {}
        self._access_order = []
        self._max_size = max_size
        self._memory_usage = 0
        self._max_memory_mb = 1024  # 1GB max memory
    
    def get(self, key: str) -> Optional[Tuple]:
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key][0]  # Return data only
        return None
    
    def put(self, key: str, value: Tuple) -> None:
        # Estimate memory usage
        item_size = self._estimate_size(value)
        
        if key in self._cache: # Update existing item
            old_size = self._cache[key][1]
            self._memory_usage -= old_size
            self._access_order.remove(key)
        elif len(self._cache) >= self._max_size or self._memory_usage + item_size > self._max_memory_mb * 1024 * 1024:
            # Remove least recently used items until we have space
            while (len(self._cache) >= self._max_size or 
                   self._memory_usage + item_size > self._max_memory_mb * 1024 * 1024) and self._access_order:
                lru_key = self._access_order.pop(0)
                lru_size = self._cache[lru_key][1]
                self._memory_usage -= lru_size
                del self._cache[lru_key]
        
        self._cache[key] = (value, item_size)
        self._memory_usage += item_size
        self._access_order.append(key)
    
    def _estimate_size(self, obj) -> int:
        """Estimate object size in bytes."""
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.memory_usage(deep=True).sum()
        elif isinstance(obj, tuple):
            return sum(self._estimate_size(item) for item in obj)
        else:
            return 1000  # Default estimate for other objects
    
    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()
        self._memory_usage = 0

# ============================= RESULT LOADING UTILITIES ============================= #

class ResultLoader:
    """Handles loading and validation of existing results with enhanced caching."""
    
    def __init__(self, base_path: Path, max_age_hours: Optional[float] = None):
        self.base_path = base_path
        self.max_age_hours = max_age_hours
        self._load_cache = {}
        self._metadata_cache = {}
    
    def should_load_result(self, result_path: Path, config_hash: str = None) -> bool:
        """Check if result should be loaded based on existence, age, and configuration."""
        if not result_path.exists():
            return False
        
        # Check file age if specified
        if self.max_age_hours is not None:
            file_mtime = datetime.fromtimestamp(result_path.stat().st_mtime)
            age_threshold = datetime.now() - timedelta(hours=self.max_age_hours)
            if file_mtime <= age_threshold:
                return False
        
        # Check configuration compatibility if hash provided
        if config_hash:
            config_check_path = result_path.parent / ".config_hash"
            if config_check_path.exists():
                try:
                    with open(config_check_path, 'r') as f:
                        saved_hash = f.read().strip()
                    if saved_hash != config_hash:
                        return False  # Configuration has changed
                except Exception as e:
                    pass
        
        return True
    
    def load_result(self, result_path: Path) -> Optional[pd.DataFrame]:
        """Load result from file with error handling and optimization."""
        cache_key = str(result_path)
        
        # Check cache first
        if cache_key in self._load_cache:
            return self._load_cache[cache_key]
        
        try:
            # Read with optimized parameters
            if result_path.suffix.lower() in ['.tsv', '.txt']:
                result = pd.read_csv(
                    result_path, sep='\t', index_col=0, 
                    low_memory=False, memory_map=True
                )
            elif result_path.suffix.lower() == '.csv':
                result = pd.read_csv(
                    result_path, index_col=0, 
                    low_memory=False, memory_map=True
                )
            else:
                logger.warning(f"Unsupported file format: {result_path}")
                return None
                
            # Basic validation
            if result.empty:
                logger.warning(f"Empty result file: {result_path}")
                return None
            # Optimize memory usage
            result = self._optimize_dataframe_memory(result)
            # Cache the result
            self._load_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.warning(f"Failed to load result from {result_path}: {e}")
            return None
    
    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif df[col].min() >= np.iinfo(np.int16).min and df[col].max() <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif df[col].min() >= 0 and df[col].max() <= 255:
                df[col] = df[col].astype(np.uint8)
        
        for col in df.select_dtypes(include=['float64']).columns:
            # Preserve precision for p-values
            if col not in ['p_value', 'q_value']:  
                df[col] = pd.to_numeric(df[col], downcast='float')

        # Convert object columns to category 
        for col in df.select_dtypes(include=['object']).columns: 
            # Cardinality threshold
            if len(df[col].unique()) / len(df[col]) < 0.5:  
                df[col] = df[col].astype('category')
        
        return df
    
    def get_existing_results(
        self, 
        group_column: str, 
        tasks: List[Tuple[str, str, str]],
        config_hash: str = None
    ) -> Dict[str, Dict]:
        """Get all existing results for a group column with configuration validation."""
        existing_results = {}
        
        for table_type, level, test in tasks:
            output_dir = self.base_path / group_column / table_type / level
            
            # Check for main result file
            result_path = output_dir / f'{test}.tsv'
            
            if self.should_load_result(result_path, config_hash):
                result = self.load_result(result_path)
                if result is not None:
                    if table_type not in existing_results: # Initialize nested structure
                        existing_results[table_type] = {}
                    if level not in existing_results[table_type]:
                        existing_results[table_type][level] = {}
                    
                    existing_results[table_type][level][test] = result
                    
                    if test == 'network_analysis': # Load additional files for network analysis
                        corr_path = output_dir / f'{test}_correlation_matrix.tsv'
                        if corr_path.exists() and self.should_load_result(corr_path, config_hash):
                            corr_result = self.load_result(corr_path)
                            if corr_result is not None:
                                existing_results[table_type][level][f'{test}_correlation_matrix'] = corr_result
        return existing_results
    
    def clear_cache(self):
        """Clear the loading cache."""
        self._load_cache.clear()
        self._metadata_cache.clear()

# ================================== CONFIGURATION =================================== #

logger = logging.getLogger("workflow_16s")

# Pre-compiled test configuration for faster access
TEST_CONFIG = {
    "fisher": {
        "key": "fisher", "func": fisher_exact_bonferroni,
        "name": "Fisher exact (Bonferroni)", "effect_col": "proportion_diff",
        "alt_effect_col": "odds_ratio", "parallel_safe": True,
        "requires_group_values": True
    },
    "ttest": {
        "key": "ttest", "func": ttest,
        "name": "Student t‑test", "effect_col": "mean_difference",
        "alt_effect_col": "cohens_d", "parallel_safe": True,
        "requires_group_values": True
    },
    "mwu_bonferroni": {
        "key": "mwub", "func": mwu_bonferroni,
        "name": "Mann–Whitney U (Bonferroni)", "effect_col": "effect_size_r",
        "alt_effect_col": "median_difference", "parallel_safe": True,
        "requires_group_values": True
    },
    "kruskal_bonferroni": {
        "key": "kwb", "func": kruskal_bonferroni,
        "name": "Kruskal–Wallis (Bonferroni)", "effect_col": "epsilon_squared",
        "alt_effect_col": None, "parallel_safe": True,
        "requires_group_values": True
    },
    "enhanced_stats": {
        "key": "enhanced", "func": enhanced_statistical_tests,
        "name": "Enhanced Statistical Tests", "effect_col": "effect_size",
        "alt_effect_col": None, "parallel_safe": False,
        "requires_group_values": False
    },
    "differential_abundance": {
        "key": "diffabund", "func": differential_abundance_analysis,
        "name": "Differential Abundance Analysis", "effect_col": "log2_fold_change",
        "alt_effect_col": "fold_change", "parallel_safe": False,
        "requires_group_values": True
    },
    "anova": {
        "key": "anova", "func": anova,
        "name": "One-way ANOVA", "effect_col": "eta_squared",
        "alt_effect_col": None, "parallel_safe": True,
        "requires_group_values": True
    },
    "spearman_correlation": {
        "key": "spearman", "func": spearman_correlation,
        "name": "Spearman Correlation", "effect_col": "rho",
        "alt_effect_col": None, "parallel_safe": True,
        "requires_group_values": False
    },
    "network_analysis": {
        "key": "network", "func": microbial_network_analysis,
        "name": "Network Analysis", "effect_col": "correlation",
        "alt_effect_col": "abs_correlation", "parallel_safe": False,
        "requires_group_values": False
    }
}

DEFAULT_TESTS = {
    "raw": ["ttest"],
    "filtered": ['mwu_bonferroni', 'kruskal_bonferroni'],
    "normalized": ['ttest', 'mwu_bonferroni', 'kruskal_bonferroni'],
    "clr_transformed": ['ttest', 'mwu_bonferroni', 'kruskal_bonferroni'],
    "presence_absence": ["fisher"]
}

# ==================================== FUNCTIONS ===================================== #

def _init_nested_dict(dictionary: Dict, keys: List[str]) -> None:
    """Initialize nested dictionary levels efficiently."""
    current = dictionary
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current.setdefault(keys[-1], {})


def get_enabled_tasks(
    config: Dict, 
    tables: Dict[str, Dict[str, Table]]
) -> List[Tuple[str, str, str]]:
    """Task enumeration with early filtering."""
    stats_config = config.get('stats', {})
    table_config = stats_config.get('tables', {})
    
    tasks = []
    known_tests = set(TEST_CONFIG.keys())
    
    # Pre-filter enabled table types
    enabled_table_types = [
        table_type for table_type, type_config in table_config.items()
        if type_config.get('enabled', False) and table_type in tables
    ]
    
    for table_type in enabled_table_types:
        type_config = table_config[table_type]
        available_levels = set(tables[table_type].keys())
        
        # Filter levels
        configured_levels = set(type_config.get('levels', available_levels))
        enabled_levels = available_levels & configured_levels
        
        # Filter tests
        configured_tests = set(type_config.get('tests', DEFAULT_TESTS.get(table_type, [])))
        enabled_tests = configured_tests & known_tests
        
        # Generate tasks 
        tasks.extend([
            (table_type, level, test)
            for level in enabled_levels
            for test in enabled_tests
        ])
    
    return tasks


def get_group_column_values(group_column: Dict, metadata: pd.DataFrame) -> List[Any]:
    """Optimized group column value extraction."""
    if 'values' in group_column and group_column['values']:
        return group_column['values']
    
    if group_column['type'] == 'bool':
        return [True, False]
    
    col_name = group_column['name']
    if col_name in metadata.columns:
        # Use more efficient unique value extraction
        return metadata[col_name].drop_duplicates().tolist()
    
    return []


def run_single_statistical_test(
    task_data: Tuple[str, str, str, Table, pd.DataFrame, str, List[Any], Path]
) -> TaskResult:
    """Optimized single test execution for parallel processing."""
    table_type, level, test, table, metadata, group_column, group_values, output_dir = task_data
    task_id = f"{table_type}_{level}_{test}"
    start_time = time.time()
    
    try:
        # Prepare data once
        table_aligned, metadata_aligned = update_table_and_metadata(table, metadata)
        test_func = TEST_CONFIG[test]["func"]
        # Handle different function signatures efficiently
        if test in {'enhanced_stats', 'differential_abundance'}:
            result = test_func(
                table=table_aligned,
                metadata=metadata_aligned,
                group_column=group_column
            )
        elif test == 'network_analysis':
            corr_matrix, edges_df = test_func(table=table_aligned)
            # Save correlation matrix
            corr_path = output_dir / f'{test}_correlation_matrix.tsv'
            corr_matrix.to_csv(corr_path, sep='\t')
            result = edges_df
        elif test == 'spearman_correlation':
            # Skip if column not found
            if group_column not in metadata_aligned.columns:
                return TaskResult(task_id, table_type, level, test, None, "Column not found", False, 0)
            result = test_func(
                table=table_aligned,
                metadata=metadata_aligned,
                continuous_column=group_column
            )
        else:
            # Check if test requires group values
            if TEST_CONFIG[test].get("requires_group_values", True):
                result = test_func(
                    table=table_aligned,
                    metadata=metadata_aligned,
                    group_column=group_column,
                    group_column_values=group_values
                )
            else:
                result = test_func(
                    table=table_aligned,
                    metadata=metadata_aligned,
                    group_column=group_column
                )
        
        # Save results
        if isinstance(result, pd.DataFrame) and not result.empty:
            output_path = output_dir / f'{test}.tsv'
            result.to_csv(output_path, sep='\t', index=True)
            # Save configuration hash to validate future loads
            config_hash_path = output_dir / ".config_hash"
            if not config_hash_path.exists():
                # Create a simple hash of test configuration
                config_str = f"{test}_{table_type}_{level}"
                config_hash = hashlib.md5(config_str.encode()).hexdigest()
                with open(config_hash_path, 'w') as f:
                    f.write(config_hash)
        
        processing_time = time.time() - start_time
        return TaskResult(task_id, table_type, level, test, result, None, False, processing_time)
        
    except Exception as e:
        error_msg = f"Test '{test}' failed for {table_type}/{level}: {str(e)}"
        logger.error(error_msg)
        processing_time = time.time() - start_time
        return TaskResult(task_id, table_type, level, test, None, error_msg, False, processing_time)


def _calculate_config_hash(config: Dict, group_column: str, table_type: str, level: str, test: str) -> str:
    """Calculate a hash for configuration to validate result compatibility."""
    config_data = {
        'group_column': group_column,
        'table_type': table_type,
        'level': level,
        'test': test,
        'test_config': TEST_CONFIG.get(test, {}),
        'stats_config': config.get('stats', {})
    }
    
    config_str = json.dumps(config_data, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()

# ===================================== CLASSES ====================================== #

class StatisticalAnalysis:
    """Highly optimized Statistical Analysis class with result loading."""
    def __init__(
        self,
        config: Dict,
        tables: Dict,
        metadata: Dict,
        mode: str,
        group_columns: List,
        project_dir: Union[str, Path],
        max_workers: Optional[int] = None,
        use_process_pool: bool = False,
        load_existing: bool = True,
        max_file_age_hours: Optional[float] = None,
        force_recalculate: List[str] = None
    ) -> None:
        self.config = config
        self.project_dir = Path(project_dir.final) if hasattr(project_dir, 'final') else Path(project_dir)
        self.mode = mode
        self.tables = tables
        self.metadata = metadata
        self.group_columns = group_columns
        
        # Result loading configuration
        self.load_existing = load_existing
        self.max_file_age_hours = max_file_age_hours
        self.force_recalculate = set(force_recalculate or [])
        
        # Optimization settings
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.use_process_pool = use_process_pool
        
        # Initialize caching and result loader
        self._data_cache = DataCache()
        self.result_loader = ResultLoader(
            self.project_dir / 'stats', 
            max_file_age_hours
        ) if load_existing else None
        
        # Add NFC facilities if enabled
        if (self.config.get("nfc_facilities", {}).get('enabled', False) and 
            'facility_match' in self.metadata["raw"]["genus"].columns):
            self.group_columns.append({
                'name': 'facility_match', 
                'type': 'bool', 
                'values': [True, False]
            })
        
        self.results: Dict = {}
        self.advanced_results: Dict = {}
        self.load_statistics = {
            'total_tasks': 0,
            'loaded_from_files': 0,
            'calculated_fresh': 0,
            'load_time_saved_seconds': 0,
            'task_times': {}
        }
        
        # Pre-validate configuration
        validation_issues = self.validate_configuration()
        if validation_issues['errors']:
            logger.error(f"Configuration errors: {validation_issues['errors']}")
            raise ValueError("Configuration validation failed")
        
        # Run analysis with optimization
        self._run_optimized_analysis()
    
    def _get_cached_data(self, table_type: str, level: str) -> Tuple[Table, pd.DataFrame]:
        """Get cached aligned table and metadata."""
        cache_key = f"{table_type}_{level}"
        cached = self._data_cache.get(cache_key)
        
        if cached is not None:
            return cached
        
        # Prepare and cache data
        table = self.tables[table_type][level]
        metadata = self.metadata[table_type][level]
        table_aligned, metadata_aligned = update_table_and_metadata(table, metadata)
        
        cached_data = (table_aligned, metadata_aligned)
        self._data_cache.put(cache_key, cached_data)
        
        return cached_data
    
    def _should_force_recalculate(self, group_column: str, table_type: str, level: str, test: str) -> bool:
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
    
    def _run_optimized_analysis(self) -> None:
        """Run analysis with parallel processing and result loading."""
        start_time = time.time()
        
        # Process each group column
        for group_column in self.group_columns:
            col_name = group_column['name']
            col_values = get_group_column_values(group_column, self.metadata["raw"]["genus"])
            
            logger.info(f"Processing group column: {col_name}")
            logger.info(f"Values: {col_values}")
            
            self.results[col_name] = self._run_parallel_for_group(col_name, col_values)
        
        # Calculate time savings
        end_time = time.time()
        analysis_time = end_time - start_time
        self.load_statistics['analysis_time_seconds'] = analysis_time
        
        # Log loading statistics
        self._log_load_statistics()
    
    def _run_parallel_for_group(self, group_column: str, group_values: List[Any]) -> Dict:
        """Run statistical analysis with parallel processing and result loading."""
        tasks = get_enabled_tasks(self.config, self.tables)
        if not tasks:
            return {}
        
        self.load_statistics['total_tasks'] += len(tasks)
        
        # Load existing results if enabled
        existing_results = {}
        remaining_tasks = []
        
        if self.load_existing and self.result_loader:
            # Calculate configuration hash for validation
            config_hashes = {}
            for table_type, level, test in tasks:
                config_hashes[(table_type, level, test)] = _calculate_config_hash(
                    self.config, group_column, table_type, level, test
                )
            
            existing_results = self.result_loader.get_existing_results(
                group_column, tasks, config_hash=config_hashes.get((table_type, level, test), None)
            )
            
            # Filter out tasks that have existing results and don't need to be recalculated
            for table_type, level, test in tasks:
                should_force = self._should_force_recalculate(group_column, table_type, level, test)
                
                if (not should_force and 
                    table_type in existing_results and 
                    level in existing_results[table_type] and 
                    test in existing_results[table_type][level]):
                    
                    self.load_statistics['loaded_from_files'] += 1
                    logger.info(f"Loaded existing result: {group_column}/{table_type}/{level}/{test}")
                else:
                    remaining_tasks.append((table_type, level, test))
                    if should_force:
                        logger.info(f"Force recalculating: {group_column}/{table_type}/{level}/{test}")
        else:
            remaining_tasks = tasks
        
        self.load_statistics['calculated_fresh'] += len(remaining_tasks)
        
        # Separate parallel-safe and sequential tasks from remaining tasks
        parallel_tasks = []
        sequential_tasks = []
        
        for table_type, level, test in remaining_tasks:
            if TEST_CONFIG[test].get('parallel_safe', True):
                parallel_tasks.append((table_type, level, test))
            else:
                sequential_tasks.append((table_type, level, test))
        
        group_stats = existing_results.copy()  # Start with loaded results
        
        # Process parallel tasks
        if parallel_tasks:
            calculated_results = self._process_parallel_tasks(
                parallel_tasks, group_column, group_values
            )
            group_stats = self._merge_results(group_stats, calculated_results)
        
        # Process sequential tasks
        if sequential_tasks:
            calculated_results = self._process_sequential_tasks(
                sequential_tasks, group_column, group_values
            )
            group_stats = self._merge_results(group_stats, calculated_results)
        
        return group_stats
    
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
    
    def _process_parallel_tasks(
        self, 
        tasks: List[Tuple[str, str, str]], 
        group_column: str, 
        group_values: List[Any]
    ) -> Dict:
        """Process tasks in parallel."""
        results = {}
        
        if not tasks:
            return results
        
        # Prepare task data
        task_data_list = []
        for table_type, level, test in tasks:
            output_dir = self.project_dir / 'stats' / group_column / table_type / level
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get cached data
            table_aligned, metadata_aligned = self._get_cached_data(table_type, level)
            
            task_data = (
                table_type, level, test, table_aligned, metadata_aligned,
                group_column, group_values, output_dir
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
                task_desc = f"Parallel analysis for '{group_column}' ({len(tasks)} tasks)"
                task_id = progress.add_task(_format_task_desc(task_desc), total=len(future_to_task))
                
                for future in as_completed(future_to_task):
                    task_result = future.result()
                    
                    # Store result and timing
                    _init_nested_dict(results, [task_result.table_type, task_result.level])
                    results[task_result.table_type][task_result.level][task_result.test] = task_result.result
                    
                    # Record processing time
                    task_key = f"{task_result.table_type}_{task_result.level}_{task_result.test}"
                    self.load_statistics['task_times'][task_key] = task_result.processing_time
                    
                    if task_result.error:
                        logger.warning(f"Task failed: {task_result.error}")
                    
                    progress.update(task_id, advance=1)
        
        return results
    
    def _process_sequential_tasks(
        self, 
        tasks: List[Tuple[str, str, str]], 
        group_column: str, 
        group_values: List[Any]
    ) -> Dict:
        """Process tasks sequentially."""
        results = {}
        
        if not tasks:
            return results
        
        with get_progress_bar() as progress:
            task_desc = f"Sequential analysis for '{group_column}' ({len(tasks)} tasks)"
            task_id = progress.add_task(_format_task_desc(task_desc), total=len(tasks))
            
            for table_type, level, test in tasks:
                output_dir = self.project_dir / 'stats' / group_column / table_type / level
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Get cached data
                table_aligned, metadata_aligned = self._get_cached_data(table_type, level)
                
                task_data = (
                    table_type, level, test, table_aligned, metadata_aligned,
                    group_column, group_values, output_dir
                )
                
                start_time = time.time()
                task_result = run_single_statistical_test(task_data)
                processing_time = time.time() - start_time
                
                # Store result and timing
                _init_nested_dict(results, [task_result.table_type, task_result.level])
                results[task_result.table_type][task_result.level][task_result.test] = task_result.result
                
                # Record processing time
                task_key = f"{task_result.table_type}_{task_result.level}_{task_result.test}"
                self.load_statistics['task_times'][task_key] = processing_time
                
                if task_result.error:
                    logger.warning(f"Task failed: {task_result.error}")
                
                progress.update(task_id, advance=1)
        
        return results
    
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
    
    def get_effect_size(self, test_name: str, row: pd.Series) -> Optional[float]:
        """Optimized effect size extraction."""
        test_config = TEST_CONFIG.get(test_name)
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

    def run_core_microbiome_analysis(
        self, 
        prevalence_threshold: float = 0.8, 
        abundance_threshold: float = 0.01
    ) -> Dict:
        """Run core microbiome analysis for all groups."""
        core_results = {}
        
        for group_column in self.group_columns:
            col = group_column['name']
            core_results[col] = {}
            
            with get_progress_bar() as progress:
                main_desc = f"Running core microbiome analysis for '{col}'"
                main_desc_fmt = _format_task_desc(main_desc)
                main_n = len(self.tables) * len(self.tables['raw'])
                main_task = progress.add_task(main_desc_fmt, total=main_n)
                
                for table_type in self.tables: 
                    if table_type == "clr_transformed":
                        core_results[col][table_type] = {}
                        logger.debug(
                            f"Skipping core microbiome analysis for table type '{table_type}'. "
                            f"Will error due to float division by zero."
                        )
                        continue
                        
                    for level in self.tables[table_type]:
                        level_desc = f"{table_type.replace('_', ' ').title()} ({level.title()})"
                        level_desc_fmt = _format_task_desc(level_desc)
                        progress.update(main_task, description=level_desc_fmt)
                        
                        # Use cached data
                        table_aligned, metadata_aligned = self._get_cached_data(table_type, level)
                        
                        try:
                            core_features = core_microbiome(
                                table=table_aligned,
                                metadata=metadata_aligned,
                                group_column=col,
                                prevalence_threshold=prevalence_threshold,
                                abundance_threshold=abundance_threshold
                            )
                            
                            _init_nested_dict(core_results, [col, table_type, level])
                            core_results[col][table_type][level] = core_features
                            
                            # Save results
                            output_dir = self.project_dir / 'core_microbiome' / col / table_type / level
                            output_dir.mkdir(parents=True, exist_ok=True)
                            
                            for group, core_df in core_features.items():
                                output_path = output_dir / f'core_features_{group}.tsv'
                                core_df.to_csv(output_path, sep='\t', index=False)
                                
                        except Exception as e:
                            logger.error(f"Core microbiome analysis failed for {col}/{table_type}/{level}: {e}")
                            
                        finally:
                            progress.update(main_task, advance=1)
                    
                progress.update(main_task, description=_format_task_desc(main_desc))
        
        self.advanced_results['core_microbiome'] = core_results
        return core_results
    
    def run_batch_correlation_analysis(self, continuous_variables: List[str]) -> Dict:
        """Run correlation analysis for multiple continuous variables."""
        correlation_results = {}
        
        for var in continuous_variables:
            correlation_results[var] = {}
            
            for table_type in self.tables:                        
                for level in self.tables[table_type]:
                    # Check if variable exists in metadata
                    metadata = self.metadata[table_type][level]
                    table = self.tables[table_type][level]
                    if var not in metadata.columns:
                        continue

                    # Filter out samples with missing values
                    metadata = metadata.dropna(subset=[var])
                    if len(metadata) < 5:  # Require min samples
                        logger.warning(f"Skipping {var}/{table_type}/{level}: only {len(metadata)} valid samples")
                        continue
                        
                    table_aligned, metadata_aligned = update_table_and_metadata(table, metadata)
                    
                    try:
                        result = spearman_correlation(
                            table=table_aligned,
                            metadata=metadata_aligned,
                            continuous_column=var
                        )
                        
                        _init_nested_dict(correlation_results, [var, table_type, level])
                        correlation_results[var][table_type][level] = result
                        
                        # Save results
                        output_dir = self.project_dir / 'correlations' / var / table_type / level
                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_path = output_dir / 'spearman_correlations.tsv'
                        result.to_csv(output_path, sep='\t', index=False)
                        
                    except Exception as e:
                        logger.error(f"Correlation analysis failed for {var}/{table_type}/{level}: {e}")
        
        self.advanced_results['correlations'] = correlation_results
        return correlation_results
    
    def run_network_analysis_batch(
        self, 
        methods: List[str] = ['sparcc', 'spearman'], 
        threshold: float = 0.3
    ) -> Dict:
        """Run network analysis for multiple correlation methods."""
        network_results = {}
        
        for method in methods:
            for table_type in self.tables:
                for level in self.tables[table_type]:
                    _init_nested_dict(network_results, [table_type, level, method])
                    if table_type == "clr_transformed_presence_absence":
                        network_results[table_type][level][method] = {}
                        logger.debug(
                            f"Skipping network analysis for table type '{table_type}' with '{method}'. "
                            f"Will error due to `abs_correlation`."
                        )
                        continue
                    # Use cached data
                    table_aligned, _ = self._get_cached_data(table_type, level)
                    
                    try:
                        corr_matrix, edges_df = microbial_network_analysis(
                            table=table_aligned,
                            method=method,
                            threshold=threshold
                        )
                        
                        network_results[table_type][level][method] = {
                            'correlation_matrix': corr_matrix,
                            'edges': edges_df
                        }
                        
                        # Save results
                        output_dir = self.project_dir / 'networks' / table_type / level / method 
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        corr_path = output_dir / 'correlation_matrix.tsv'
                        edges_path = output_dir / 'network_edges.tsv'
                        
                        corr_matrix.to_csv(corr_path, sep='\t')
                        edges_df.to_csv(edges_path, sep='\t', index=False)
                        
                        # Generate network statistics
                        network_stats = self._calculate_network_statistics(edges_df)
                        stats_path = output_dir / 'network_statistics.tsv'
                        pd.DataFrame([network_stats]).to_csv(stats_path, sep='\t', index=False)
                        
                    except Exception as e:
                        logger.error(f"Network analysis failed for {method}/{table_type}/{level}: {e}")
        
        self.advanced_results['networks'] = network_results
        return network_results
    
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
    
    def get_top_features_across_tests(self, n_features: int = 10) -> pd.DataFrame:
        """Get top features that appear consistently across multiple tests."""
        feature_counts = {}
        feature_effects = {}
        
        for group_col, group_results in self.results.items():
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
        
        return summary_df
    
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
    
    def validate_configuration(self) -> Dict[str, List[str]]:
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
