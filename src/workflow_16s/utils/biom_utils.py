"""Compatibility shim re-exporting BIOM utilities from core/biom_utils.

Keep this shim for one release cycle to preserve deep imports. Consumers
should update imports to `workflow_16s.core.biom_utils`.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "workflow_16s.utils.biom_utils has moved to workflow_16s.core.biom_utils; "
    "please update imports to use the new path.",
    DeprecationWarning,
)

from workflow_16s.core.biom_utils import *  # noqa: F401,F403

try:
    from workflow_16s.core import biom_utils as _core_biom_utils

    __all__ = getattr(_core_biom_utils, "__all__", [])
except Exception:
    __all__ = []
        