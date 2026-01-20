#!/usr/bin/env python3
"""
Test script to verify new feature integrations:
1. Longitudinal analysis
2. Power analysis
3. Rarefaction curves

This script checks that imports work and config sections are accessible.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all new modules can be imported."""
    print("Testing imports...")
    
    try:
        from workflow_16s.downstream.longitudinal import (
            calculate_temporal_stability,
            trajectory_clustering,
            run_zibr,
            run_maaslin2_longitudinal
        )
        print("✅ Longitudinal analysis imports successful")
    except ImportError as e:
        print(f"❌ Longitudinal analysis import failed: {e}")
        return False
    
    try:
        from workflow_16s.downstream.power_analysis import (
            estimate_permanova_power,
            estimate_da_power
        )
        print("✅ Power analysis imports successful")
    except ImportError as e:
        print(f"❌ Power analysis import failed: {e}")
        return False
    
    try:
        from workflow_16s.downstream.diversity.alpha.rarefaction import (
            generate_rarefaction_curves,
            calculate_rarefaction_curve
        )
        print("✅ Rarefaction analysis imports successful")
    except ImportError as e:
        print(f"❌ Rarefaction analysis import failed: {e}")
        return False
    
    return True


def test_config():
    """Test that config sections are accessible."""
    print("\nTesting config sections...")
    
    try:
        from workflow_16s.config import get_config
        from pathlib import Path
        
        config_path = Path(__file__).parent / "config" / "config.yaml"
        config = get_config(config_path)
        
        # Check longitudinal config
        if 'longitudinal' in config:
            print(f"✅ Longitudinal config found: enabled={config['longitudinal'].get('enabled')}")
        else:
            print("❌ Longitudinal config section missing")
            return False
        
        # Check power_analysis config
        if 'power_analysis' in config:
            print(f"✅ Power analysis config found: enabled={config['power_analysis'].get('enabled')}")
        else:
            print("❌ Power analysis config section missing")
            return False
        
        # Check rarefaction config
        if 'rarefaction' in config:
            print(f"✅ Rarefaction config found: enabled={config['rarefaction'].get('enabled')}")
        else:
            print("❌ Rarefaction config section missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_workflow_integration():
    """Test that workflow steps can import the new functions."""
    print("\nTesting workflow integration...")
    
    try:
        from workflow_16s.downstream.steps.preprocessing import run_preprocessing_pipeline
        from workflow_16s.downstream.steps.analysis import run_analysis_suite
        print("✅ Workflow step imports successful")
        return True
    except ImportError as e:
        print(f"❌ Workflow step import failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing New Feature Integrations")
    print("=" * 60)
    
    results = []
    
    results.append(("Import Test", test_imports()))
    results.append(("Config Test", test_config()))
    results.append(("Workflow Integration Test", test_workflow_integration()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! New features are properly integrated.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
