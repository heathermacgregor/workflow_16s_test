"""Compatibility shim re-exporting constants from core/constants.

Keep this shim for one release cycle to preserve deep imports. Consumers
should update imports to `workflow_16s.core.constants`.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "workflow_16s.utils.constants has moved to workflow_16s.core.constants; "
    "please update imports to use the new path.",
    DeprecationWarning,
)

from workflow_16s.core.constants import *  # noqa: F401,F403

try:
    from workflow_16s.core import constants as _core_constants

    __all__ = getattr(_core_constants, "__all__", [])
except Exception:
    __all__ = []