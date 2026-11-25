"""1:1 SSOT store owner for Layer 5 ``portfolio_positions_state.json``."""
from __future__ import annotations

import os
from typing import Any, Callable, Collection, Mapping, Optional

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
        drop_symbols: Optional[Collection[str]] = None,
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
        resolved_path = cls._resolve_disk_path(pos_path)
        os.makedirs(os.path.dirname(os.path.abspath(resolved_path)), exist_ok=True)
        with generation_file_lock(resolved_path):
            disk_exists = cls.has_disk_file(resolved_path)
            disk_raw = (
                read_json_dict(resolved_path, default=None, recover_previous=True, repair_files=True)
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
            drop_upper = {
                str(s).strip().upper()
                for s in (drop_symbols or ())
                if isinstance(s, str) and str(s).strip()
            }
            updated_upper = {k.upper() for k in pos_updates}
            base_pos: dict[str, Any] = {}
            for k, v in src_pos.items():
                if k not in pos_updates and k.upper() in updated_upper:
                    continue
                if drop_upper and k not in pos_updates:
                    sym_k = k.split(":", 2)[2].strip().upper() if k.count(":") >= 2 else ""
                    sym_v = str(v.get("symbol") or "").strip().upper() if isinstance(v, Mapping) else ""
                    if (sym_k and sym_k in drop_upper) or (sym_v and sym_v in drop_upper):
                        continue
                base_pos[k] = v
            merged_raw = {**base_pos, **pos_updates}
            sanitized = sanitize_balance_state_payload(merged_raw, prev_state=src_pos)
            should_write = (
                (sanitized != disk_dict)
                if disk_dict is not None
                else (sanitized != src_pos or (bool(sanitized) and not os.path.exists(resolved_path)))
            )
            if not should_write:
                return False
            if callable(save_fn):
                save_fn(pos_path, sanitized, indent=indent)
            else:
                save_json_state(resolved_path, sanitized, indent=indent)
            return True
