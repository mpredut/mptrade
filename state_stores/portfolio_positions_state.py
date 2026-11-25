"""1:1 SSOT store owner for Layer 5 ``portfolio_positions_state.json``."""
from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Optional

from cachepaths import PORTFOLIO_POSITIONS_STATE_FILENAME
from state_io import generation_file_lock, read_json_dict, save_json_state

from .base import StateStoreBase
from .venue_balances_state import sanitize_balance_state_payload


class PortfolioPositionsStateStore(StateStoreBase):
    """Exclusive 1:1 owner for reading and locked-updating ``portfolio_positions_state.json`` (``pos:*`` only)."""

    FILENAME = PORTFOLIO_POSITIONS_STATE_FILENAME
    SUBDOMAIN = "portfolio"

    @classmethod
    def load_dict(
        cls,
        cachedb_dir: Optional[str],
        *,
        use_cache: bool = False,
        read_json_fn: Optional[Callable[[str], Any]] = None,
        require_exists: bool = False,
    ) -> dict[str, Any]:
        """Read ``portfolio_positions_state.json`` strictly filtered to ``pos:*`` dict entries."""
        raw = cls.read_dict(
            cachedb_dir,
            use_cache=use_cache,
            read_json_fn=read_json_fn,
            require_exists=require_exists,
            recover_previous=True,
        )
        return {
            k: v for k, v in raw.items() if isinstance(k, str) and k.startswith("pos:") and isinstance(v, dict)
        }

    @classmethod
    def upsert_locked(
        cls,
        pos_path: str,
        updated_positions: Mapping[str, Any],
        *,
        fallback_state: Optional[Mapping[str, Any]] = None,
        read_disk: bool = True,
        save_fn: Optional[Callable[..., Any]] = None,
        indent: int = 2,
    ) -> bool:
        """Single canonical locked producer writer for ``portfolio_positions_state.json``."""
        if not pos_path or not isinstance(updated_positions, Mapping):
            return False
        pos_updates = {
            k: v
            for k, v in updated_positions.items()
            if isinstance(k, str) and k.startswith("pos:") and isinstance(v, Mapping)
        }
        if not pos_updates:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(pos_path)), exist_ok=True)
        with generation_file_lock(pos_path):
            disk_exists = cls.has_disk_file(pos_path)
            disk_raw = (
                read_json_dict(pos_path, default=None, recover_previous=True, repair_files=True)
                if (read_disk and disk_exists)
                else None
            )
            disk_dict = (
                {k: v for k, v in disk_raw.items() if isinstance(k, str) and k.startswith("pos:") and isinstance(v, dict)}
                if isinstance(disk_raw, dict)
                else None
            )
            src_pos: dict[str, Any] = (
                dict(disk_dict)
                if disk_dict is not None
                else {
                    k: v
                    for k, v in (fallback_state.items() if isinstance(fallback_state, Mapping) else ())
                    if isinstance(k, str) and k.startswith("pos:") and isinstance(v, Mapping)
                }
            )
            updated_upper = {k.upper() for k in pos_updates}
            base_pos = {
                k: v
                for k, v in src_pos.items()
                if k in pos_updates or k.upper() not in updated_upper
            }
            merged_raw = {**base_pos, **pos_updates}
            sanitized = sanitize_balance_state_payload(merged_raw, prev_state=src_pos)
            should_write = (
                (sanitized != disk_dict)
                if disk_dict is not None
                else (sanitized != src_pos or (bool(sanitized) and not os.path.exists(pos_path)))
            )
            if not should_write:
                return False
            writer = save_fn if callable(save_fn) else save_json_state
            writer(pos_path, sanitized, indent=indent)
            return True
