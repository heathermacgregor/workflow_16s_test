# ==================================================================================== #

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
logger = logging.getLogger("workflow_16s")
# ==================================================================================== #

# Pre-compiled test configuration for faster access
TestConfig = {
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

TestDefaults = {
    "raw": ["ttest"],
    "filtered": ['mwu_bonferroni', 'kruskal_bonferroni'],
    "normalized": ['ttest', 'mwu_bonferroni', 'kruskal_bonferroni'],
    "clr_transformed": ['ttest', 'mwu_bonferroni', 'kruskal_bonferroni'],
    "presence_absence": ["fisher"]
}

# ==================================================================================== #

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

# ==================================================================================== #

class LocalResultLoader:
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
      

# ==================================================================================== #

def run_single_statistical_test(
    task_data: Tuple[str, str, str, Table, pd.DataFrame, str, List[Any], Path]
) -> TaskResult:
    """Optimized single test execution for parallel processing."""
    table_type, level, test, table, metadata, name, values, output_dir = task_data
    task_id = f"{table_type}_{level}_{test}"
    start_time = time.time()
    
    try:
        # Prepare data once
        table, metadata = align_table_and_metadata(table, metadata)
        test_func = TestConfig[test]["func"]
        # Handle different function signatures efficiently
        if test in {'enhanced_stats', 'differential_abundance'}:
            result = test_func(
                table=table,
                metadata=metadata,
                group_column=name
            )
        elif test == 'network_analysis':
            corr_matrix, edges_df = test_func(table=table)
            # Save correlation matrix
            corr_path = output_dir / f'{test}_correlation_matrix.tsv'
            corr_matrix.to_csv(corr_path, sep='\t')
            result = edges_df
        elif test == 'spearman_correlation':
            # Skip if column not found
            if name not in metadata.columns:
                return TaskResult(task_id, table_type, level, test, None, "Column not found", False, 0)
            result = test_func(
                table=table,
                metadata=metadata,
                continuous_column=name
            )
        else:
            # Check if test requires group values
            if TestConfig[test].get("requires_group_values", True):
                result = test_func(
                    table=table,
                    metadata=metadata,
                    group_column=name,
                    group_column_values=values
                )
            else:
                result = test_func(
                    table=table,
                    metadata=metadata,
                    group_column=name
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


def get_enabled_tasks(
    config: Dict, 
    tables: Dict[str, Dict[str, Table]]
) -> List[Tuple[str, str, str]]:
    """Task enumeration with early filtering."""
    stats_config = config.get('stats', {})
    table_config = stats_config.get('tables', {})
    
    tasks = []
    known_tests = set(TestConfig.keys())
    
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
        configured_tests = set(type_config.get('tests', TestDefaults.get(table_type, [])))
        enabled_tests = configured_tests & known_tests
        
        # Generate tasks 
        tasks.extend([
            (table_type, level, test)
            for level in enabled_levels
            for test in enabled_tests
        ])
    
    return tasks


def calculate_config_hash(config: Dict, group_column: str, table_type: str, level: str, test: str) -> str:
    """Calculate a hash for configuration to validate result compatibility."""
    config_data = {
        'group_column': group_column,
        'table_type': table_type,
        'level': level,
        'test': test,
        'test_config': TestConfig.get(test, {}),
        'stats_config': config.get('stats', {})
    }
    
    config_str = json.dumps(config_data, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()
  

