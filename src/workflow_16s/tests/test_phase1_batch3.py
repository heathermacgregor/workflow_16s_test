"""Tests for Phase 1 batch 3: taxonomy, validation."""

import importlib


def test_taxonomy_compat_shim():
    """Verify taxonomy.py shim re-exports from core."""
    compat = importlib.import_module("workflow_16s.utils.taxonomy")
    core = importlib.import_module("workflow_16s.core.taxonomy")

    for name in ["FaprotaxDB", "FaprotaxError", "DownloaderError", "ParserError"]:
        assert hasattr(core, name), f"core missing {name}"
        assert hasattr(compat, name), f"shim missing {name}"
        assert getattr(compat, name) is getattr(core, name)


def test_validation_compat_shim():
    """Verify validation.py shim re-exports from core."""
    compat = importlib.import_module("workflow_16s.utils.validation")
    core = importlib.import_module("workflow_16s.core.validation")

    for name in ["ResultsValidator", "get_validator", "validate_results"]:
        assert hasattr(core, name), f"core missing {name}"
        assert hasattr(compat, name), f"shim missing {name}"
        assert getattr(compat, name) is getattr(core, name)
