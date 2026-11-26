"""1:1 SSOT store owner for ``notification_delivery_state.json``."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from cachepaths import (
    NOTIFICATION_DELIVERY_STATE_FILENAME,
    clean_scalar_str,
    state_path,
)
from state_io import generation_file_lock, read_json_dict, save_json_state

from .base import StateStoreBase


class NotificationDeliveryStateStore(StateStoreBase):
    """Exclusive 1:1 owner for daily notification deduplication & rate-limit state."""

    FILENAME = NOTIFICATION_DELIVERY_STATE_FILENAME
    SUBDOMAIN = "system"
    STORAGE_ROOT = "statedb"

    @classmethod
    def default_state_path(cls, workspace_dir: Optional[str] = None) -> Path:
        """Return canonical ``statedb/system/notification_delivery_state.json`` path."""
        if workspace_dir and (clean_ws := clean_scalar_str(workspace_dir)):
            return Path(clean_ws) / "statedb" / cls.SUBDOMAIN / cls.FILENAME
        return Path(state_path(cls.FILENAME))

    @classmethod
    def resolve_state_path(
        cls,
        explicit_path: Optional[str] = None,
        *,
        workspace_dir: Optional[str] = None,
    ) -> Path:
        """Return explicit ``NOTIFICATION_STATE_FILE`` override when set, else canonical statedb path."""
        if explicit_path and (clean_p := clean_scalar_str(explicit_path)):
            return Path(clean_p).expanduser()
        return cls.default_state_path(workspace_dir=workspace_dir)

    @classmethod
    def state_lock(cls, path: Path):
        """Return cross-process generation file lock for ``path``."""
        return generation_file_lock(path, lock_path=path.with_suffix(path.suffix + ".lock"))

    @classmethod
    def read_state(cls, path: Path) -> dict[str, Any]:
        """Read notification delivery state under cross-process lock."""
        with cls.state_lock(path):
            state = read_json_dict(path, default={}, recover_previous=True, repair_files=True)
        return state if isinstance(state, dict) else {}

    @classmethod
    def write_state(cls, path: Path, state: dict[str, Any]) -> None:
        """Atomically persist notification delivery state under cross-process lock."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with cls.state_lock(path):
            save_json_state(
                path,
                state,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
