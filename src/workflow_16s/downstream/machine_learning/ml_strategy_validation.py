"""
ML Strategy Validation & Testing

Diagnostic tools for validating strategy implementations:
1. Quick validation of strategy configurations
2. Smoke tests on small data subsets
3. Diagnostic reports
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from workflow_16s.utils.logger import get_logger
from workflow_16s.downstream.machine_learning.ml_strategy_config import (
    STRATEGY_REGISTRY,
    get_strategy,
    list_strategies,
    MLStrategyConfig
)
from workflow_16s.downstream.machine_learning.ml_strategy_integration import (
    StrategyExecutor
)

logger = get_logger("strategy_validation")


class StrategyValidator:
    """
    Validates strategy configurations and runs diagnostic tests.
    """
    
    @staticmethod
    def validate_configuration() -> Tuple[bool, List[str]]:
        """
        Validate all registered strategies have valid configurations.
        
        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list_of_errors)
        """
        logger.info("✅ Validating strategy registry...")
        errors = []
        
        if not STRATEGY_REGISTRY:
            errors.append("❌ Empty strategy registry")
            return False, errors
        
        logger.info(f"   Found {len(STRATEGY_REGISTRY)} registered strategies")
        
        for strategy_name, config in STRATEGY_REGISTRY.items():
            # Check required fields
            if not isinstance(config, MLStrategyConfig):
                errors.append(f"❌ {strategy_name}: Not an MLStrategyConfig instance")
                continue
            
            if not config.name:
                errors.append(f"❌ {strategy_name}: Missing name")
            
            if not config.description:
                errors.append(f"❌ {strategy_name}: Missing description")
            
            if config.feature_set is None:
                errors.append(f"❌ {strategy_name}: Missing feature_set")
            
            if config.preprocessing is None:
                errors.append(f"❌ {strategy_name}: Missing preprocessing")
            
            if config.cv_method is None:
                errors.append(f"❌ {strategy_name}: Missing cv_method")
            
            if config.filter_policy is None:
                errors.append(f"❌ {strategy_name}: Missing filter_policy")
            
            logger.debug(f"   ✓ {strategy_name}")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info(f"✅ All {len(STRATEGY_REGISTRY)} strategies valid")
        else:
            logger.error(f"❌ Found {len(errors)} configuration errors:")
            for error in errors:
                logger.error(f"   {error}")
        
        return is_valid, errors
    
    @staticmethod
    def diagnose_strategy(strategy_name: str) -> Dict[str, Any]:
        """
        Provide detailed diagnostic for a single strategy.
        
        Parameters
        ----------
        strategy_name : str
            Strategy to diagnose
            
        Returns
        -------
        Dict[str, Any]
            Diagnostic information
        """
        logger.info(f"📋 Diagnosing strategy: {strategy_name}")
        logger.info("-" * 70)
        
        config = get_strategy(strategy_name)
        if config is None:
            logger.error(f"❌ Strategy not found: {strategy_name}")
            return {'status': 'not_found'}
        
        diagnostic = {
            'name': config.name,
            'description': config.description,
            'feature_set': config.feature_set.value,
            'preprocessing': config.preprocessing.value,
            'cv_method': config.cv_method.value,
            'filter_policy': config.filter_policy.value,
            'estimated_runtime_hours': config.estimated_runtime_hours,
            'recommended_min_samples': getattr(config, 'recommended_min_samples', config.min_samples_per_class * 10),
            'min_studies': config.min_studies,
            'hyperparameters': str(config.hyperparameters) if config.hyperparameters else None,
            'status': 'valid'
        }
        
        logger.info(f"   Name: {diagnostic['name']}")
        logger.info(f"   Features: {diagnostic['feature_set']}")
        logger.info(f"   Preprocessing: {diagnostic['preprocessing']}")
        logger.info(f"   CV: {diagnostic['cv_method']}")
        logger.info(f"   Filter: {diagnostic['filter_policy']}")
        logger.info(f"   Est. Runtime: {diagnostic['estimated_runtime_hours']} hours")
        logger.info(f"   Min Samples: {diagnostic['recommended_min_samples']}")
        logger.info(f"   Min Studies: {diagnostic['min_studies']}")
        
        return diagnostic
    
    @staticmethod
    def print_all_strategies() -> None:
        """Print all available strategies in a formatted table."""
        logger.info("\n" + "=" * 90)
        logger.info("📚 AVAILABLE STRATEGIES")
        logger.info("=" * 90)
        
        strategies = list_strategies()
        logger.info(f"Total: {len(strategies)} strategies\n")
        
        # Group by feature set for readability
        by_feature = {}
        for name in strategies:
            config = get_strategy(name)
            feature = config.feature_set.value if config else 'unknown'
            if feature not in by_feature:
                by_feature[feature] = []
            by_feature[feature].append(name)
        
        for feature_set in sorted(by_feature.keys()):
            logger.info(f"🔹 {feature_set}:")
            for strategy_name in sorted(by_feature[feature_set]):
                config = get_strategy(strategy_name)
                logger.info(f"   • {strategy_name}")
                logger.info(f"     └─ {config.description}")


class StrategySmokeTester:
    """
    Run smoke tests on strategy implementations.
    Tests with small data subsets to catch configuration issues.
    """
    
    @staticmethod
    def run_smoke_test(
        adata: Any,
        strategy_name: str,
        ml_config: Any,
        output_dir: Path,
        sample_size: int = 100,
        feature_subsample: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run a quick smoke test of a strategy on a data subset.
        
        Parameters
        ----------
        adata : AnnData
            Full expression matrix
        strategy_name : str
            Strategy to test
        ml_config : MLConfig
            Configuration object
        output_dir : Path
            Output directory for test results
        sample_size : int
            Number of samples to use for test (default 100)
        feature_subsample : Optional[int]
            Number of features to subsample (default: use all)
            
        Returns
        -------
        Dict[str, Any]
            Test results with status, timing, errors
        """
        logger.info("=" * 70)
        logger.info(f"🧪 SMOKE TEST: {strategy_name}")
        logger.info("=" * 70)
        
        test_start = time.time()
        result = {
            'strategy': strategy_name,
            'status': 'pending',
            'start_time': test_start,
            'sample_size': min(sample_size, adata.n_obs),
            'n_features': adata.n_vars if feature_subsample is None else min(feature_subsample, adata.n_vars),
            'errors': []
        }
        
        # Validate strategy exists
        config = get_strategy(strategy_name)
        if config is None:
            result['status'] = 'fail'
            result['errors'].append(f"Strategy not found: {strategy_name}")
            logger.error(f"❌ Strategy not found: {strategy_name}")
            return result
        
        logger.info(f"✓ Strategy found")
        logger.info(f"✓ Creating test data subset ({result['sample_size']} samples × {result['n_features']} features)")
        
        # Create test data
        try:
            adata_test = adata[:sample_size].copy()
            
            if feature_subsample is not None and feature_subsample < adata.n_vars:
                # Random feature subsample
                feature_idx = np.random.choice(adata.n_vars, size=feature_subsample, replace=False)
                adata_test = adata_test[:, feature_idx]
            
            logger.info(f"✓ Test data prepared: {adata_test.n_obs} × {adata_test.n_vars}")
            
        except Exception as e:
            result['status'] = 'fail'
            result['errors'].append(f"Data preparation failed: {str(e)}")
            logger.error(f"❌ Data prep failed: {str(e)}")
            return result
        
        # Create test output directory
        test_output_dir = Path(output_dir) / f"smoke_test_{strategy_name}"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✓ Output directory: {test_output_dir}")
        logger.info(f"✓ Executing strategy...")
        
        # Execute strategy
        try:
            executor = StrategyExecutor(
                output_dir=test_output_dir,
                ml_config=ml_config
            )
            
            execution_results, benchmark = executor._execute_single_strategy(
                adata=adata_test,
                strategy_config=config,
                strategy_idx=1
            )
            
            result['status'] = 'pass'
            result['execution_time_seconds'] = benchmark.get('total_time_seconds', 0)
            result['execution_time_hours'] = benchmark.get('total_time_hours', 0)
            
            logger.info(f"✅ Smoke test PASSED")
            logger.info(f"   Execution time: {result['execution_time_hours']:.3f} hours ({result['execution_time_seconds']:.1f}s)")
            
        except Exception as e:
            result['status'] = 'fail'
            result['errors'].append(f"Strategy execution failed: {str(e)}")
            logger.error(f"❌ Strategy execution failed: {str(e)}", exc_info=True)
        
        result['end_time'] = time.time()
        result['total_seconds'] = result['end_time'] - result['start_time']
        
        return result
    
    @staticmethod
    def run_all_smoke_tests(
        adata: Any,
        ml_config: Any,
        output_dir: Path,
        sample_size: int = 100
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run smoke tests for all registered strategies.
        
        Parameters
        ----------
        adata : AnnData
            Expression matrix
        ml_config : MLConfig
            Configuration
        output_dir : Path
            Output directory
        sample_size : int
            Samples per test
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Per-strategy test results
        """
        strategies = list_strategies()
        logger.info(f"🧪 Running smoke tests for {len(strategies)} strategies")
        logger.info("=" * 70)
        
        all_results = {}
        passed = 0
        failed = 0
        
        for strategy_name in strategies:
            try:
                result = StrategySmokeTester.run_smoke_test(
                    adata=adata,
                    strategy_name=strategy_name,
                    ml_config=ml_config,
                    output_dir=output_dir,
                    sample_size=sample_size
                )
                
                all_results[strategy_name] = result
                
                if result['status'] == 'pass':
                    passed += 1
                else:
                    failed += 1
                
                logger.info(f"   {'✅ PASS' if result['status'] == 'pass' else '❌ FAIL'}: {strategy_name}")
                
            except Exception as e:
                logger.error(f"   ❌ CRASH: {strategy_name}: {str(e)}")
                all_results[strategy_name] = {
                    'status': 'crash',
                    'error': str(e)
                }
                failed += 1
        
        logger.info("\n" + "=" * 70)
        logger.info(f"📊 SMOKE TEST SUMMARY")
        logger.info(f"   Passed: {passed}/{len(strategies)}")
        logger.info(f"   Failed: {failed}/{len(strategies)}")
        logger.info("=" * 70)
        
        return all_results


# Convenience exports
__all__ = [
    'StrategyValidator',
    'StrategySmokeTester'
]
