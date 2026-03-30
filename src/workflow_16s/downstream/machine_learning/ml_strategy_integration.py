"""
ML Strategy Integration Layer

Bridges MLStrategyOrchestrator with the existing ML pipeline (run_selection_with_config).
Handles:
1. Strategy-to-config translation
2. Sequential strategy execution
3. Benchmarking and metrics collection
4. Per-strategy result organization
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict
import pandas as pd

from workflow_16s.utils.logger import get_logger
from workflow_16s.downstream.machine_learning.ml_strategy_config import (
    MLStrategyConfig, 
    STRATEGY_REGISTRY,
    get_strategy
)
from workflow_16s.downstream.machine_learning.workflows.feature_selection import (
    run_selection_with_config
)

logger = get_logger("strategy_integration")


class StrategyExecutor:
    """
    Executes ML strategies sequentially with full benchmarking and result tracking.
    Handles interaction between MLStrategyOrchestrator and the ML pipeline.
    """
    
    def __init__(self, output_dir: Path, ml_config: Any):
        """
        Parameters
        ----------
        output_dir : Path
            Base directory for strategy outputs
        ml_config : MLConfig
            Configuration object from config_schema
        """
        self.output_dir = Path(output_dir)
        self.ml_config = ml_config
        self.results = {}
        self.benchmarks = {}
        
    def execute_strategies(
        self,
        adata: Any,
        strategy_names: List[str],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Execute multiple strategies sequentially.
        
        Parameters
        ----------
        adata : AnnData
            Input expression matrix
        strategy_names : List[str]
            List of strategy names to execute
        verbose : bool
            Print progress information
            
        Returns
        -------
        Dict[str, Any]
            Results organized as:
            {
                'results': {strategy_name: pipeline_results},
                'benchmarks': {strategy_name: timing_metadata},
                'summary': overall_statistics
            }
        """
        logger.info("="*70)
        logger.info(f"🚀 STRATEGY EXECUTION START")
        logger.info(f"   Total strategies: {len(strategy_names)}")
        logger.info("="*70)
        
        execution_summary = {
            'total_strategies': len(strategy_names),
            'completed': 0,
            'failed': 0,
            'total_time_hours': 0,
            'strategies': []
        }
        
        overall_start = time.time()
        
        for idx, strategy_name in enumerate(strategy_names, 1):
            logger.info(f"\n[{idx}/{len(strategy_names)}] Executing strategy: {strategy_name}")
            logger.info("-" * 70)
            
            try:
                strategy_config = get_strategy(strategy_name)
                if strategy_config is None:
                    logger.error(f"   ❌ Strategy not found: {strategy_name}")
                    execution_summary['failed'] += 1
                    continue
                
                # Execute single strategy
                result, benchmark = self._execute_single_strategy(
                    adata=adata,
                    strategy_config=strategy_config,
                    strategy_idx=idx
                )
                
                self.results[strategy_name] = result
                self.benchmarks[strategy_name] = benchmark
                execution_summary['completed'] += 1
                execution_summary['strategies'].append({
                    'name': strategy_name,
                    'status': 'completed',
                    'duration_hours': benchmark.get('total_time_hours', 0),
                    'output_dir': str(result.get('output_dir', ''))
                })
                
                logger.info(f"   ✅ Complete in {benchmark['total_time_hours']:.2f} hours")
                
            except Exception as e:
                logger.error(f"   ❌ Strategy failed: {str(e)}", exc_info=True)
                execution_summary['failed'] += 1
                execution_summary['strategies'].append({
                    'name': strategy_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        overall_elapsed = time.time() - overall_start
        execution_summary['total_time_hours'] = overall_elapsed / 3600.0
        
        logger.info("\n" + "="*70)
        logger.info(f"✅ STRATEGY EXECUTION COMPLETE")
        logger.info(f"   Completed: {execution_summary['completed']}/{len(strategy_names)}")
        logger.info(f"   Failed: {execution_summary['failed']}")
        logger.info(f"   Total time: {execution_summary['total_time_hours']:.2f} hours")
        logger.info("="*70)
        
        return {
            'results': self.results,
            'benchmarks': self.benchmarks,
            'summary': execution_summary
        }
    
    def _execute_single_strategy(
        self,
        adata: Any,
        strategy_config: MLStrategyConfig,
        strategy_idx: int
    ) -> tuple:
        """
        Execute a single strategy with benchmarking.
        
        Returns
        -------
        tuple
            (pipeline_result, benchmark_metadata)
        """
        strategy_start = time.time()
        strategy_name = strategy_config.name
        
        # Create strategy-specific output directory
        strategy_output_dir = self.output_dir / f"{strategy_idx}_{strategy_name}"
        strategy_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"   Strategy: {strategy_config.name}")
        logger.info(f"   Description: {strategy_config.description}")
        logger.info(f"   Features: {strategy_config.feature_set.value} + {strategy_config.preprocessing.value}")
        logger.info(f"   CV Method: {strategy_config.cv_method.value}")
        logger.info(f"   Filter: {strategy_config.filter_policy.value}")
        logger.info(f"   Output: {strategy_output_dir}")
        
        # Translate strategy config to ML config parameters
        config_patch = self._translate_strategy_to_config(strategy_config)
        
        # Apply config patch (temporarily modify ml_config for this strategy)
        original_values = {}
        is_dict_config = isinstance(self.ml_config, dict)
        
        for key, value in config_patch.items():
            if is_dict_config:
                original_values[key] = self.ml_config.get(key)
                self.ml_config[key] = value
            else:
                if hasattr(self.ml_config, key):
                    original_values[key] = getattr(self.ml_config, key)
                setattr(self.ml_config, key, value)
        
        try:
            # Call the unified pipeline orchestrator
            logger.info(f"   Invoking run_selection_with_config...")
            
            pipeline_result = run_selection_with_config(
                adata=adata,
                output_base_dir=strategy_output_dir,
                ml_config=self.ml_config
            )
            
            strategy_elapsed = time.time() - strategy_start
            
            # Collect benchmark metadata
            benchmark = {
                'strategy_name': strategy_name,
                'feature_set': strategy_config.feature_set.value,
                'preprocessing': strategy_config.preprocessing.value,
                'cv_method': strategy_config.cv_method.value,
                'filter_policy': strategy_config.filter_policy.value,
                'start_time': time.time() - strategy_elapsed,
                'end_time': time.time(),
                'total_time_seconds': strategy_elapsed,
                'total_time_hours': strategy_elapsed / 3600.0,
                'estimated_time_hours': strategy_config.estimated_runtime_hours,
                'min_samples_required': strategy_config.min_samples_per_class,
                'hyperparameters': strategy_config.hyperparameters if strategy_config.hyperparameters else {},
                'output_dir': str(strategy_output_dir),
                'status': 'completed'
            }
            
            # Save strategy metadata
            self._save_strategy_metadata(strategy_output_dir, benchmark, pipeline_result)
            
            return pipeline_result, benchmark
            
        finally:
            # Restore original config values
            for key, value in original_values.items():
                if is_dict_config:
                    if value is not None:
                        self.ml_config[key] = value
                    else:
                        self.ml_config.pop(key, None)
                else:
                    setattr(self.ml_config, key, value)
    
    def _translate_strategy_to_config(self, strategy_config: MLStrategyConfig) -> Dict[str, Any]:
        """
        Translate MLStrategyConfig to modifications for ml_config.
        
        This modifies configuration to reflect strategy choices:
        - Batch settings based on feature_set
        - CV strategy based on cv_method
        - Preprocessing based on preprocessing method
        - Filtering based on filter_policy
        """
        config_patch = {}
        
        # Map feature sets to config settings
        if strategy_config.feature_set.value == 'taxonomy_only':
            config_patch['include_batch_features'] = False
            config_patch['include_metadata_features'] = False
        elif strategy_config.feature_set.value == 'taxonomy_batch':
            config_patch['include_batch_features'] = True
            config_patch['include_metadata_features'] = False
        elif strategy_config.feature_set.value == 'taxonomy_metadata':
            config_patch['include_batch_features'] = False
            config_patch['include_metadata_features'] = True
        elif strategy_config.feature_set.value == 'all':
            config_patch['include_batch_features'] = True
            config_patch['include_metadata_features'] = True
        
        # Preprocessing method
        if strategy_config.preprocessing.value == 'conqur':
            config_patch['apply_batch_correction'] = True
        else:  # identity
            config_patch['apply_batch_correction'] = False
        
        # CV strategy
        cv_strategy_map = {
            'kfold': 'kfold',
            'lopocv': 'lopocv',
            'spatial': 'spatial'
        }
        config_patch['cv_strategy'] = cv_strategy_map.get(strategy_config.cv_method.value, 'kfold')
        
        # Filter policy
        if strategy_config.filter_policy.value == 'multiclass':
            config_patch['eligibility_mode'] = 'filter'
        elif strategy_config.filter_policy.value == 'variance':
            config_patch['eligibility_mode'] = 'filter'
        else:  # none
            config_patch['eligibility_mode'] = 'raw'
        
        return config_patch
    
    def _save_strategy_metadata(
        self,
        strategy_output_dir: Path,
        benchmark: Dict[str, Any],
        pipeline_result: Dict[str, Any]
    ) -> None:
        """Save strategy execution metadata to JSON files."""
        metadata = {
            'benchmark': benchmark,
            'pipeline_result_summary': {
                'keys': list(pipeline_result.keys()) if pipeline_result else [],
                'num_results': len(pipeline_result) if pipeline_result else 0
            }
        }
        
        metadata_file = strategy_output_dir / 'strategy_metadata.json'
        with open(metadata_file, 'w') as f:
            # Custom JSON encoder for Path objects
            json.dump(metadata, f, indent=2, default=str)
        
        logger.debug(f"   📝 Metadata saved: {metadata_file}")
    
    def export_summary_report(self) -> Path:
        """
        Export comprehensive summary report of all strategy executions.
        
        Returns
        -------
        Path
            Path to summary CSV file
        """
        summary_data = []
        
        for strategy_name, benchmark in self.benchmarks.items():
            summary_data.append({
                'strategy': strategy_name,
                'feature_set': benchmark.get('feature_set', ''),
                'preprocessing': benchmark.get('preprocessing', ''),
                'cv_method': benchmark.get('cv_method', ''),
                'filter_policy': benchmark.get('filter_policy', ''),
                'actual_time_hours': benchmark.get('total_time_hours', 0),
                'estimated_time_hours': benchmark.get('estimated_time_hours', 0),
                'status': benchmark.get('status', 'unknown'),
                'output_dir': benchmark.get('output_dir', '')
            })
        
        if not summary_data:
            logger.warning("No benchmark data to export")
            return None
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / 'strategy_execution_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"✅ Summary report: {summary_file}")
        return summary_file


def run_strategies_from_config(
    adata: Any,
    ml_config: Any,
    output_dir: Path
) -> Dict[str, Any]:
    """
    High-level convenience function to execute all configured strategies.
    
    Used by workflow.py to run strategies when ml.strategies is set in config.
    
    Parameters
    ----------
    adata : AnnData
        Expression matrix
    ml_config : MLConfig
        Configuration object
    output_dir : Path
        Base output directory
        
    Returns
    -------
    Dict[str, Any]
        Results from StrategyExecutor.execute_strategies()
    """
    strategies_to_run = getattr(ml_config, 'strategies', [])
    
    if not strategies_to_run:
        logger.warning("No strategies configured (ml.strategies is empty)")
        return {}
    
    logger.info(f"🎯 Running {len(strategies_to_run)} configured strategies")
    
    executor = StrategyExecutor(
        output_dir=output_dir,
        ml_config=ml_config
    )
    
    results = executor.execute_strategies(
        adata=adata,
        strategy_names=strategies_to_run
    )
    
    executor.export_summary_report()
    
    return results
