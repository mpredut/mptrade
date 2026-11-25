"""1:1 SSOT store owner for ``nav_history_state.json``."""
from __future__ import annotations

import os
from typing import Any, Optional, Sequence

from cachepaths import NAV_HISTORY_STATE_FILENAME, cache_relpath, resolve_statedb_filepath

from .base import StateStoreBase


class NavHistoryStateStore(StateStoreBase):
    """Exclusive 1:1 owner for reading and persisting ``nav_history_state.json``."""

    FILENAME = NAV_HISTORY_STATE_FILENAME
    SUBDOMAIN = "portfolio"

    @classmethod
    def resolve_workspace_path(cls, workspace_dir: str) -> str:
        """Resolve ``nav_history_state.json`` inside ``workspace_dir`` (``statedb/portfolio/`` with pre-cutover fallback)."""
        return str(resolve_statedb_filepath(os.path.join(workspace_dir, cache_relpath(cls.FILENAME))))

    @classmethod
    def read_raw_doc(cls, workspace_dir: str) -> dict[str, Any]:
        """Load the persisted NAV history JSON envelope with ``.previous`` crash recovery."""
        path = cls.resolve_workspace_path(workspace_dir)
        doc = cls.read_file_dict(path, default={}, recover_previous=True)
        return doc if isinstance(doc, dict) else {}

    @classmethod
    def save_points_doc(
        cls,
        path: str,
        points: Sequence[dict[str, Any]],
        *,
        updated_ts: float,
        indent: Optional[int] = 2,
    ) -> None:
        """Atomically persist the canonical versioned NAV history envelope."""
        cls.save_state_file(
            path,
            {
                "version": 1,
                "updated_ts": round(updated_ts, 3),
                "points": list(points),
            },
            indent=indent,
        )
