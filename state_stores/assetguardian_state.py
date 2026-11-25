"""1:1 SSOT store owner for ``assetguardian_state.json``."""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

from botcore import FileLock
from cachepaths import ASSETGUARDIAN_STATE_FILENAME, resolve_statedb_filepath, state_path

from .base import StateStoreBase


class AssetguardianStateStore(StateStoreBase):
    """Exclusive 1:1 owner for reading and locked-persisting ``assetguardian_state.json``."""

    FILENAME = ASSETGUARDIAN_STATE_FILENAME
    SUBDOMAIN = "bots"

    @classmethod
    def default_path(cls) -> str:
        """Return default canonical path for ``assetguardian_state.json``."""
        return state_path(cls.FILENAME)

    @classmethod
    def resolve_paths(
        cls,
        path: Optional[str] = None,
        lock_path: Optional[str] = None,
        *,
        migrate: bool = False,
    ) -> tuple[str, str]:
        """Resolve ``(state_path, lock_path)`` keeping sibling lock files aligned across migrations."""
        raw_path = path or cls.default_path()
        target_path = str(resolve_statedb_filepath(raw_path, migrate=migrate))
        eff_lock = (
            f"{target_path}.lock"
            if (not lock_path or lock_path == f"{raw_path}.lock")
            else str(resolve_statedb_filepath(lock_path, migrate=migrate))
        )
        return target_path, eff_lock

    @classmethod
    def load_state(
        cls,
        path: Optional[str] = None,
        *,
        lock_path: Optional[str] = None,
        fail_closed: bool = True,
    ) -> dict[str, Any]:
        """Load ``assetguardian_state.json`` under ``FileLock`` with fail-closed corruption protection."""
        target_path, eff_lock = cls.resolve_paths(path, lock_path, migrate=True)
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with FileLock(eff_lock):
            data = cls.load_state_file(
                target_path,
                default_factory=dict,
                fail_closed=fail_closed,
                label="AssetGuardian",
                root_type=dict,
            )
            return data if isinstance(data, dict) else {}

    @classmethod
    def save_state(
        cls,
        value: Any,
        path: Optional[str] = None,
        *,
        lock_path: Optional[str] = None,
    ) -> None:
        """Persist ``assetguardian_state.json`` atomically under ``FileLock`` with dirty-check."""
        target_path, eff_lock = cls.resolve_paths(path, lock_path, migrate=True)
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with FileLock(eff_lock):
            cls.save_state_file(
                target_path,
                value,
                indent=None,
                sort_keys=True,
                separators=(",", ":"),
                skip_if_unchanged=True,
            )

    @classmethod
    def read_from_cachedb(
        cls,
        cachedb_dir: Optional[str],
        *,
        read_json_fn: Optional[Callable[[str], Any]] = None,
    ) -> dict[str, Any]:
        """Read-only accessor for ``assetguardian_state.json`` inside ``cachedb_dir``."""
        return cls.read_dict(cachedb_dir, read_json_fn=read_json_fn, recover_previous=True)
