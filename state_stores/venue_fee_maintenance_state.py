"""1:1 SSOT store owner for ``venue_fee_maintenance_state.json``."""
from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Any, Callable, Optional

from botcore import FileLock
from cachepaths import (
    VENUE_FEE_MAINTENANCE_STATE_FILENAME,
    clean_scalar_str,
    resolve_statedb_filepath,
    state_path,
)

from .base import StateStoreBase


class VenueFeeMaintenanceStateStore(StateStoreBase):
    """Exclusive 1:1 owner for reading and locked-persisting ``venue_fee_maintenance_state.json``."""

    FILENAME = VENUE_FEE_MAINTENANCE_STATE_FILENAME
    SUBDOMAIN = "state"

    @classmethod
    def resolve_path(cls, cachedb_dir: Optional[str] = None, *, migrate: bool = False) -> str:
        """Resolve canonical ``statedb/state/venue_fee_maintenance_state.json`` (with pre-cutover fallback)."""
        if cachedb_dir and (clean_dir := clean_scalar_str(cachedb_dir)):
            resolved = str(
                resolve_statedb_filepath(os.path.join(clean_dir, cls.FILENAME), migrate=migrate)
            )
            if os.path.exists(resolved) or os.path.abspath(os.path.dirname(resolved)) != os.path.abspath(clean_dir):
                return resolved
            return os.path.join(clean_dir, cls.SUBDOMAIN, cls.FILENAME)
        return str(resolve_statedb_filepath(state_path(cls.FILENAME), migrate=migrate))

    @classmethod
    def read_state(
        cls,
        cachedb_dir: Optional[str],
        *,
        read_json_fn: Optional[Callable[[str], Any]] = None,
    ) -> dict[str, Any]:
        """Read ``venue_fee_maintenance_state.json`` with ``.previous`` crash recovery."""
        if not cachedb_dir or not clean_scalar_str(cachedb_dir):
            return {}
        if callable(read_json_fn):
            for cand in cls.candidate_paths(cachedb_dir):
                raw = read_json_fn(cand)
                if isinstance(raw, dict) and raw:
                    return raw
            return {}
        state_path = cls.resolve_path(cachedb_dir)
        payload = cls.read_file_dict(state_path, default={}, recover_previous=True)
        return payload if isinstance(payload, dict) else {}

    @classmethod
    @contextmanager
    def locked_read_for_update(
        cls,
        cachedb_dir: Optional[str] = None,
        *,
        dry_run: bool = False,
    ):
        """Acquire ``FileLock`` on the fee-maintenance state file and yield ``(state_path, prev_state)``."""
        state_path = cls.resolve_path(cachedb_dir, migrate=not dry_run)
        os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
        with FileLock(f"{state_path}.lock"):
            prev_state = cls.read_file_dict(
                state_path,
                default={},
                recover_previous=True,
                repair_files=not dry_run,
            )
            yield state_path, (prev_state if isinstance(prev_state, dict) else {})

    @classmethod
    def save_state(cls, state_path: str, payload: dict[str, Any], *, indent: int = 2) -> None:
        """Atomically persist updated fee-maintenance state."""
        cls.save_state_file(state_path, payload, indent=indent)
