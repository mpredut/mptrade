"""1:1 SSOT store owner for ``trailing_state.json``."""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

from botcore import _parse_bool_flag, clean_json_value
from cachepaths import TRAILING_STATE_FILENAME, resolve_venue_trailing_state_path

from .base import StateStoreBase


class TrailingStateStore(StateStoreBase):
    """Exclusive 1:1 owner for reading and persisting venue ``trailing_state.json`` files."""

    FILENAME = TRAILING_STATE_FILENAME
    SUBDOMAIN = "bots"

    @classmethod
    def resolve_venue_path(
        cls,
        venue: str,
        *,
        cache_dir: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        resolve_subdir: bool = False,
    ) -> str:
        """Resolve canonical venue path for ``trailing_state.json``."""
        return str(
            resolve_venue_trailing_state_path(
                venue,
                cache_dir=cache_dir,
                workspace_dir=workspace_dir,
                resolve_subdir=resolve_subdir,
            )
        )

    @classmethod
    def load_state(
        cls,
        state_file: str,
        *,
        fail_closed: bool = True,
        label: str = "trailing",
        read_only: bool = False,
    ) -> dict[str, Any]:
        """Load a venue ``trailing_state.json`` file with fail-closed corruption protection."""
        loaded = cls.load_state_file(
            state_file,
            default_factory=dict,
            fail_closed=fail_closed,
            label=label,
            root_type=dict,
            repair_files=not _parse_bool_flag(read_only, default=False),
        )
        return loaded if isinstance(loaded, dict) else {}

    @classmethod
    def save_state(cls, state_file: str, state: dict[str, Any], *, indent: int = 2) -> None:
        """Atomically persist a venue ``trailing_state.json`` file."""
        target_dir = os.path.dirname(state_file)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        cls.save_state_file(
            state_file,
            clean_json_value(state, stringify_keys=False),
            indent=indent,
        )

    @classmethod
    def read_venue_state(
        cls,
        venue: str,
        *,
        cache_dir: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        read_json_fn: Optional[Callable[[str], Any]] = None,
    ) -> dict[str, Any]:
        """Read-only accessor for a venue's ``trailing_state.json``."""
        target = cls.resolve_venue_path(venue, cache_dir=cache_dir, workspace_dir=workspace_dir)
        if callable(read_json_fn):
            raw = read_json_fn(target)
        else:
            raw = cls.read_file_dict(target, default={}, recover_previous=True)
        return raw if isinstance(raw, dict) else {}
