"""1:1 SSOT store owner for ``rtrade_pairs_state.json``."""
from __future__ import annotations

from typing import Any, Optional

from cachepaths import RTRADE_PAIRS_STATE_FILENAME, resolve_cachedb_filepath
from state_io import atomic_write_json, read_json_file

from .base import StateStoreBase


class RtradePairsStateStore(StateStoreBase):
    """Exclusive 1:1 storage owner for ``rtrade_pairs_state.json``."""

    FILENAME = RTRADE_PAIRS_STATE_FILENAME
    SUBDOMAIN = "state"

    @classmethod
    def resolve_file_path(cls, path: str, *, migrate: bool = False) -> str:
        """Resolve an explicit ``rtrade_pairs_state.json`` file path."""
        return str(resolve_cachedb_filepath(path, migrate=migrate))

    @classmethod
    def resolve_lock_path(
        cls,
        path: Optional[str],
        lock_path: Optional[str],
        *,
        migrate: bool = False,
    ) -> Optional[str]:
        """Resolve lock path aligned with ``path`` across subdomain migrations."""
        if not lock_path:
            return None
        if path and lock_path == f"{path}.lock":
            return f"{cls.resolve_file_path(path, migrate=migrate)}.lock"
        return str(resolve_cachedb_filepath(lock_path, migrate=migrate))

    @classmethod
    def read_raw(cls, path: str) -> Any:
        """Read raw JSON payload from ``path`` with ``.previous`` recovery (propagates ``FileNotFoundError``)."""
        eff_path = cls.resolve_file_path(path, migrate=False)
        return read_json_file(eff_path, recover_previous=True)

    @classmethod
    def write_atomic(cls, path: str, data: Any) -> None:
        """Atomically persist validated rtrade pairs payload with dirty-check."""
        atomic_write_json(
            cls.resolve_file_path(path, migrate=True),
            data,
            sort_keys=True,
            separators=(",", ":"),
            skip_if_unchanged=True,
        )
