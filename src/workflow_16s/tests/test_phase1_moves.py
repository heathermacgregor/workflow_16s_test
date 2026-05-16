"""Tests for Phase 1 module moves: progress, dir_utils."""

import importlib


def test_progress_compat_shim():
    """Verify progress.py shim re-exports from core."""
    compat = importlib.import_module("workflow_16s.utils.progress")
    core = importlib.import_module("workflow_16s.core.progress")

    assert hasattr(core, "get_progress_bar")
    assert hasattr(compat, "get_progress_bar")
    assert compat.get_progress_bar is core.get_progress_bar

    assert hasattr(core, "MofNCompleteColumn")
    assert hasattr(compat, "MofNCompleteColumn")
    assert compat.MofNCompleteColumn is core.MofNCompleteColumn


def test_dir_utils_compat_shim():
    """Verify dir_utils.py shim re-exports from core."""
    compat = importlib.import_module("workflow_16s.utils.dir_utils")
    core = importlib.import_module("workflow_16s.core.dir_utils")

    for name in ["Project", "Analysis", "SubSet", "RawData", "QIIME"]:
        assert hasattr(core, name), f"core missing {name}"
        assert hasattr(compat, name), f"shim missing {name}"
        assert getattr(compat, name) is getattr(core, name)
