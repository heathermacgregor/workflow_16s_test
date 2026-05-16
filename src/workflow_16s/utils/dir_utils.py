"""Compatibility shim re-exporting directory utilities from core/dir_utils.

Keep this shim for one release cycle to preserve deep imports. Consumers
should update imports to `workflow_16s.core.dir_utils`.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "workflow_16s.utils.dir_utils has moved to workflow_16s.core.dir_utils; "
    "please update imports to use the new path.",
    DeprecationWarning,
)

from workflow_16s.core.dir_utils import *  # noqa: F401,F403

try:
    from workflow_16s.core import dir_utils as _core_dir_utils

    __all__ = getattr(_core_dir_utils, "__all__", [])
except Exception:
    __all__ = []