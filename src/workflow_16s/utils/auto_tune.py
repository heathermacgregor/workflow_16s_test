"""
Compatibility shim for auto_tune module.

.. deprecated::
    This module has been moved to workflow_16s.core.auto_tune.
    Please update your imports to use the new location.
"""

from __future__ import annotations
import warnings

warnings.warn(
    "workflow_16s.utils.auto_tune has been moved to workflow_16s.core.auto_tune. "
    "Please update your imports.",
    DeprecationWarning,
    stacklevel=2
)

from workflow_16s.core.auto_tune import *  # noqa: F401, F403

__all__ = [
    "AutoTuner",
    "get_auto_tuner",
    "auto_tune_config",
]

