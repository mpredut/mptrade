"""1:1 SSOT store owner for ``smart_stop_loss_state.json``."""
from __future__ import annotations

from typing import Any, Callable, Optional

from cachepaths import SMART_STOP_LOSS_STATE_FILENAME, resolve_statedb_filepath, state_path
from state_io import read_json_dict_cached

from .base import StateStoreBase


class SmartStopLossStateStore(StateStoreBase):
    """Exclusive 1:1 owner for reading and persisting ``smart_stop_loss_state.json``."""

    FILENAME = SMART_STOP_LOSS_STATE_FILENAME
    SUBDOMAIN = "bots"

    @classmethod
    def default_path(cls) -> str:
        """Return default canonical path for ``smart_stop_loss_state.json``."""
        return state_path(cls.FILENAME)

    @classmethod
    def resolve_file_path(cls, state_file: str) -> str:
        """Resolve an explicit ``smart_stop_loss_state.json`` file path."""
        return str(resolve_statedb_filepath(state_file))

    @classmethod
    def read_cached(cls, state_file: str, memo: Optional[dict] = None) -> Any:
        """Read ``smart_stop_loss_state.json`` through the mtime-keyed cache."""
        return read_json_dict_cached(state_file, memo)

    @classmethod
    def read_for_update(cls, resolved_path: str) -> dict[str, Any]:
        """Read current on-disk state with ``.previous`` recovery and file repair before locked update."""
        raw = cls.read_file_dict(
            resolved_path,
            default={},
            recover_previous=True,
            repair_files=True,
        )
        return raw if isinstance(raw, dict) else {}

    @classmethod
    def save_state(cls, resolved_path: str, payload: dict[str, Any], *, indent: int = 2) -> None:
        """Atomically persist ``smart_stop_loss_state.json``."""
        cls.save_state_file(resolved_path, payload, indent=indent)

    @classmethod
    def read_from_cachedb(
        cls,
        cachedb_dir: Optional[str],
        *,
        read_json_fn: Optional[Callable[[str], Any]] = None,
    ) -> dict[str, Any]:
        """Read-only accessor for ``smart_stop_loss_state.json`` inside ``cachedb_dir``."""
        return cls.read_dict(cachedb_dir, read_json_fn=read_json_fn, recover_previous=True)
