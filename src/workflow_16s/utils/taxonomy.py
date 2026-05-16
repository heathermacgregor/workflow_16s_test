"""Compatibility shim re-exporting FAPROTAX taxonomy utilities from core/taxonomy.

Keep this shim for one release cycle to preserve deep imports. Consumers
should update imports to `workflow_16s.core.taxonomy`.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "workflow_16s.utils.taxonomy has moved to workflow_16s.core.taxonomy; "
    "please update imports to use the new path.",
    DeprecationWarning,
)

from workflow_16s.core.taxonomy import *  # noqa: F401,F403

try:
    from workflow_16s.core import taxonomy as _core_taxonomy

    __all__ = getattr(_core_taxonomy, "__all__", [])
except Exception:
    __all__ = []