"""1:1 SSOT store owner for ``os_resources_watchdog_state.json``."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional

from cachepaths import (
    OS_RESOURCES_WATCHDOG_STATE_FILENAME,
    clean_scalar_str,
    state_path,
)
from state_io import read_json_dict, save_json_state

from .base import StateStoreBase


class OsResourcesWatchdogStateStore(StateStoreBase):
    """Exclusive 1:1 owner for ``orchestratorOS/livecheck/os_resources_watchdog.py`` runtime state."""

    FILENAME = OS_RESOURCES_WATCHDOG_STATE_FILENAME
    SUBDOMAIN = "system"
    STORAGE_ROOT = "statedb"

    @classmethod
    def default_state_path(cls, workspace_dir: Optional[str] = None) -> Path:
        """Return canonical ``statedb/system/os_resources_watchdog_state.json`` path."""
        if workspace_dir and (clean_ws := clean_scalar_str(workspace_dir)):
            return Path(clean_ws) / "statedb" / cls.SUBDOMAIN / cls.FILENAME
        return Path(state_path(cls.FILENAME))

    @classmethod
    def load_state(cls, path: str | os.PathLike) -> dict[str, Any]:
        """Load watchdog state from ``path`` with ``.previous`` crash recovery."""
        state = read_json_dict(Path(path), default={}, recover_previous=True)
        return state if isinstance(state, dict) else {}

    @classmethod
    def save_state(cls, path: str | os.PathLike, state: dict[str, Any]) -> bool:
        """Atomically persist watchdog state to ``path``."""
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            save_json_state(target, state)
            return True
        except Exception:
            return False

    @classmethod
    def inspect_workspace(
        cls,
        workspace_dir: str,
        *,
        now_ts: Optional[float] = None,
        read_json_fn: Optional[Callable[[str], Any]] = None,
    ) -> tuple[Optional[Path], Optional[dict[str, Any]], Optional[float]]:
        """Locate and read canonical ``statedb/system/os_resources_watchdog_state.json`` for ``workspace_dir``."""
        cand = cls.default_state_path(workspace_dir)
        if not cls.has_disk_file(str(cand)):
            return None, None, None
        age_h: Optional[float] = None
        if now_ts is not None and os.path.exists(cand):
            try:
                age_h = max(0.0, (now_ts - os.path.getmtime(cand)) / 3600.0)
            except OSError:
                age_h = None
        payload = (
            read_json_fn(str(cand))
            if callable(read_json_fn)
            else read_json_dict(cand, default=None, recover_previous=True)
        )
        return cand, (payload if isinstance(payload, dict) else None), age_h
