"""Tests for Phase 1 second batch: biom_utils, constants."""

import importlib


def test_biom_utils_compat_shim():
    """Verify biom_utils.py shim re-exports from core."""
    compat = importlib.import_module("workflow_16s.utils.biom_utils")
    core = importlib.import_module("workflow_16s.core.biom_utils")

    for name in ["import_feature_table", "import_merged_feature_table", 
                 "to_biom", "to_df", "export_h5py"]:
        assert hasattr(core, name), f"core missing {name}"
        assert hasattr(compat, name), f"shim missing {name}"
        assert getattr(compat, name) is getattr(core, name)


def test_constants_compat_shim():
    """Verify constants.py shim re-exports from core."""
    compat = importlib.import_module("workflow_16s.utils.constants")
    core = importlib.import_module("workflow_16s.core.constants")

    # Test a sample of constants
    for name in ["DEFAULT_CONFIG", "SAMPLE_ID_COLUMN", "DEFAULT_ALPHA_METRICS", 
                 "TAXONOMIC_LEVELS", "DEFAULT_METRIC"]:
        assert hasattr(core, name), f"core missing {name}"
        assert hasattr(compat, name), f"shim missing {name}"
        assert getattr(compat, name) is getattr(core, name)
