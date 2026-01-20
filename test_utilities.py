#!/usr/bin/env python
"""
Test script for new performance utilities.

Run this to verify all improvements are working correctly.
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

def test_monitoring():
    """Test memory and timing monitoring."""
    print("\n=== Testing Monitoring ===")
    try:
        from workflow_16s.utils.monitoring import get_monitor, track_phase, log_memory_usage
        
        # Test basic memory logging
        log_memory_usage("Initial")
        
        # Test phase tracking
        monitor = get_monitor()
        with track_phase("Test Phase 1"):
            time.sleep(0.1)
            # Allocate some memory
            _ = [i for i in range(100000)]
        
        with track_phase("Test Phase 2"):
            time.sleep(0.2)
        
        # Generate summary
        summary = monitor.generate_summary()
        assert "Test Phase 1" in summary
        assert "Test Phase 2" in summary
        assert "Total Runtime" in summary
        
        print("✅ Monitoring: PASS")
        print(f"   - Phase tracking works")
        print(f"   - Memory logging works")
        print(f"   - Summary generation works")
        return True
        
    except Exception as e:
        print(f"❌ Monitoring: FAIL - {e}")
        return False


def test_validation():
    """Test results validation."""
    print("\n=== Testing Validation ===")
    try:
        from workflow_16s.utils.validation import ResultsValidator, validate_results
        
        validator = ResultsValidator()
        
        # Test valid DataFrame
        valid_df = pd.DataFrame({
            'taxon': ['genus1', 'genus2', 'genus3'],
            'p_value': [0.001, 0.05, 0.2],
            'q_value': [0.01, 0.1, 0.3]
        })
        assert validator.validate_dataframe(valid_df, "Test Results")
        assert validator.validate_statistical_results(valid_df, "Spearman")
        
        # Test invalid p-values (should catch)
        invalid_df = pd.DataFrame({
            'taxon': ['genus1', 'genus2'],
            'p_value': [-0.5, 1.5]  # Invalid!
        })
        validator2 = ResultsValidator()
        assert not validator2.validate_statistical_results(invalid_df, "Bad Test")
        assert len(validator2.errors) > 0
        
        # Test ML validation
        ml_results = {
            'target1': {
                'oob_score': 0.75,
                'feature_importance': pd.DataFrame({
                    'feature': ['f1', 'f2'],
                    'importance': [0.6, 0.4]
                })
            }
        }
        assert validator.validate_ml_results(ml_results['target1'])
        
        print("✅ Validation: PASS")
        print(f"   - DataFrame validation works")
        print(f"   - Statistical validation works")
        print(f"   - ML validation works")
        print(f"   - Error detection works")
        return True
        
    except Exception as e:
        print(f"❌ Validation: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_plotting_limits():
    """Test smart plotting limits."""
    print("\n=== Testing Plotting Limits ===")
    try:
        from workflow_16s.utils.plotting_limits import get_plot_limiter, limit_for_plotting
        
        # Test with small result set (no limiting)
        small_df = pd.DataFrame({
            'taxon': [f'genus{i}' for i in range(100)],
            'p_value': np.random.random(100)
        })
        limited_small, was_limited = limit_for_plotting(
            small_df,
            max_plots=1000,
            name="Small Test"
        )
        assert not was_limited
        assert len(limited_small) == 100
        
        # Test with large result set (should limit)
        large_df = pd.DataFrame({
            'taxon': [f'genus{i}' for i in range(5000)],
            'p_value': np.random.random(5000)
        })
        limited_large, was_limited = limit_for_plotting(
            large_df,
            max_plots=1000,
            name="Large Test"
        )
        assert was_limited
        assert len(limited_large) < len(large_df)
        assert len(limited_large) <= 1000
        
        # Verify top results preserved (sorted by p-value)
        sorted_large = large_df.sort_values('p_value')
        assert limited_large['p_value'].min() == sorted_large['p_value'].min()
        
        print("✅ Plotting Limits: PASS")
        print(f"   - Small datasets not limited: {len(limited_small)} plots")
        print(f"   - Large datasets limited: {len(large_df)} → {len(limited_large)} plots")
        print(f"   - Top results preserved correctly")
        return True
        
    except Exception as e:
        print(f"❌ Plotting Limits: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_auto_tuning():
    """Test auto-tuning configuration."""
    print("\n=== Testing Auto-Tuning ===")
    try:
        from workflow_16s.utils.auto_tune import get_auto_tuner, auto_tune_config
        
        # Test with small dataset
        config_small = {
            'variance_threshold': 1e-6,
            'top_n': 50,
            'min_samples_per_group': 3
        }
        tuned_small = auto_tune_config(
            n_samples=500,
            n_features=5000,
            config=config_small
        )
        # Should keep default for small dataset
        assert 'variance_threshold' in tuned_small
        
        # Test with large dataset
        config_large = {
            'variance_threshold': 1e-6,
            'top_n': 50,
            'min_samples_per_group': 3,
            'max_plots': 1000,
            'alpha': 0.05
        }
        tuned_large = auto_tune_config(
            n_samples=50000,
            n_features=500000,
            config=config_large
        )
        
        # Should adjust for large dataset
        assert tuned_large['variance_threshold'] > config_large['variance_threshold']
        assert tuned_large['top_n'] > config_large['top_n']
        assert tuned_large['min_samples_per_group'] > config_large['min_samples_per_group']
        assert tuned_large['max_plots'] < config_large['max_plots']
        
        # Check tuner tracked adjustments
        tuner = get_auto_tuner()
        assert len(tuner.adjustments) > 0
        
        print("✅ Auto-Tuning: PASS")
        print(f"   - Small dataset tuning works")
        print(f"   - Large dataset tuning works")
        print(f"   - {len(tuner.adjustments)} parameters adjusted for large dataset")
        return True
        
    except Exception as e:
        print(f"❌ Auto-Tuning: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_processing():
    """Test parallel file processing capability."""
    print("\n=== Testing Parallel Processing ===")
    try:
        from joblib import Parallel, delayed
        import os
        
        # Simple parallel task
        def square(x):
            return x ** 2
        
        # Test with different backends
        results_loky = Parallel(n_jobs=2, backend='loky')(
            delayed(square)(i) for i in range(10)
        )
        assert len(results_loky) == 10
        assert results_loky[5] == 25
        
        # Test CPU detection
        n_cpus = os.cpu_count() or 1
        assert n_cpus >= 1
        
        print("✅ Parallel Processing: PASS")
        print(f"   - joblib Parallel works")
        print(f"   - Detected {n_cpus} CPUs")
        print(f"   - loky backend works")
        return True
        
    except Exception as e:
        print(f"❌ Parallel Processing: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Testing Workflow 16S Performance Utilities")
    print("="*60)
    
    results = {
        'Monitoring': test_monitoring(),
        'Validation': test_validation(),
        'Plotting Limits': test_plotting_limits(),
        'Auto-Tuning': test_auto_tuning(),
        'Parallel Processing': test_parallel_processing(),
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:.<40} {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All utilities working correctly!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check installation.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
