# workflow_16s/utils/validation.py
"""Compatibility shim re-exporting validation utilities from core/validation.

Keep this shim for one release cycle to preserve deep imports. Consumers
should update imports to `workflow_16s.core.validation`.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "workflow_16s.utils.validation has moved to workflow_16s.core.validation; "
    "please update imports to use the new path.",
    DeprecationWarning,
)

from workflow_16s.core.validation import *  # noqa: F401,F403

try:
    from workflow_16s.core import validation as _core_validation

    __all__ = getattr(_core_validation, "__all__", [])
except Exception:
    __all__ = []
