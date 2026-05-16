"""
Phase 1 Batch 4 tests: Verify auto_tune module compatibility shim.

This test validates that the utils/auto_tune.py shim correctly re-exports
all public symbols from core/auto_tune.py without modification.
"""

import importlib
import pytest


def test_auto_tune_compat_shim():
    """Verify auto_tune shim exports match core module."""
    # Import both canonical and compat versions
    core_module = importlib.import_module("workflow_16s.core.auto_tune")
    compat_module = importlib.import_module("workflow_16s.utils.auto_tune")
    
    # Check that key exports exist and are identical
    exports = ["AutoTuner", "get_auto_tuner", "auto_tune_config"]
    
    for name in exports:
        assert hasattr(core_module, name), f"core.auto_tune missing {name}"
        assert hasattr(compat_module, name), f"utils.auto_tune missing {name}"
        
        # Verify identity (not copy)
        core_obj = getattr(core_module, name)
        compat_obj = getattr(compat_module, name)
        assert core_obj is compat_obj, (
            f"{name} is not the same object: "
            f"core={id(core_obj)}, compat={id(compat_obj)}"
        )
    
    # Verify deprecation warning is issued
    with pytest.warns(DeprecationWarning, match="workflow_16s.utils.auto_tune"):
        importlib.reload(compat_module)
