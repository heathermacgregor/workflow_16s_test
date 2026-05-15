"""Compatibility shim re-exporting functionality from core/io/anndata.

Keep this shim for one release cycle to preserve deep imports. Consumers
should update imports to `workflow_16s.core.io.anndata`.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "workflow_16s.utils.io.anndata has moved to workflow_16s.core.io.anndata; "
    "please update imports to use the new path.",
    DeprecationWarning,
)

from workflow_16s.core.io.anndata import *  # noqa: F401,F403

try:
    from workflow_16s.core.io import anndata as _core_anndata

    __all__ = getattr(_core_anndata, "__all__", [])
except Exception:
    __all__ = []
