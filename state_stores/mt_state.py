"""1:1 SSOT store owner for ``mt_state.json``."""
from __future__ import annotations

from typing import Any, Callable, Optional

from cachepaths import MT_STATE_FILENAME, state_path

from .base import StateStoreBase


class MtStateStore(StateStoreBase):
    """Exclusive 1:1 owner for reading and persisting ``mt_state.json``."""

    FILENAME = MT_STATE_FILENAME
    SUBDOMAIN = "bots"

    @classmethod
    def default_path(cls) -> str:
        """Return default canonical path for ``mt_state.json``."""
        return state_path(cls.FILENAME)

    @classmethod
    def read_state(cls, state_file: str) -> Optional[dict[str, Any]]:
        """Read ``mt_state.json`` from ``state_file`` with ``.previous`` crash recovery."""
        data = cls.read_file_dict(state_file, default=None, recover_previous=True)
        return data if isinstance(data, dict) else None

    @classmethod
    def save_state(cls, state_file: str, payload: dict[str, Any], *, indent: int = 2) -> None:
        """Atomically persist ``mt_state.json``."""
        cls.save_state_file(state_file, payload, indent=indent)

    @classmethod
    def read_from_cachedb(
        cls,
        cachedb_dir: Optional[str],
        *,
        read_json_fn: Optional[Callable[[str], Any]] = None,
    ) -> dict[str, Any]:
        """Read-only accessor for ``mt_state.json`` inside ``cachedb_dir``."""
        return cls.read_dict(cachedb_dir, read_json_fn=read_json_fn, recover_previous=True)
