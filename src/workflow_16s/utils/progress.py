"""Compatibility shim re-exporting progress utilities from core/progress.

Keep this shim for one release cycle to preserve deep imports. Consumers
should update imports to `workflow_16s.core.progress`.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "workflow_16s.utils.progress has moved to workflow_16s.core.progress; "
    "please update imports to use the new path.",
    DeprecationWarning,
)

from workflow_16s.core.progress import *  # noqa: F401,F403

try:
    from workflow_16s.core import progress as _core_progress

    __all__ = getattr(_core_progress, "__all__", [])
except Exception:
    __all__ = []
