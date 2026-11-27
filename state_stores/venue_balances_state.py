"""1:1 SSOT store owner for ``venue_balances_state.json``."""
from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Optional

from cachepaths import VENUE_BALANCES_STATE_FILENAME, safe_nonneg_float
from core.time_market import normalize_epoch_sec
from state_io import generation_file_lock, read_json_dict, save_json_state

from .base import StateStoreBase

INTERNAL_BALANCE_SCRATCH_KEYS: frozenset[str] = frozenset(
    {
        "_balance_ts",
        "_base_balance_ts",
        "_quote_balance_ts",
        "_prev_locked_quote_usd",
        "_prev_locked_base_qty",
    }
)


def strip_internal_balance_scratch_keys(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of ``entry`` with all internal balance scratch keys removed."""
    return {k: v for k, v in entry.items() if k not in INTERNAL_BALANCE_SCRATCH_KEYS}


def _same_balance_qty(curr_val: Any, prev_val: Any, *, absent_as_zero: bool = False) -> bool:
    if absent_as_zero:
        c_num = safe_nonneg_float(curr_val, default=0.0) or 0.0
        p_num = safe_nonneg_float(prev_val, default=0.0) or 0.0
        return abs(c_num - p_num) <= 1e-12
    if curr_val is None and prev_val is None:
        return True
    if curr_val is None or prev_val is None:
        return False
    c_num = safe_nonneg_float(curr_val, default=None)
    p_num = safe_nonneg_float(prev_val, default=None)
    if c_num is not None and p_num is not None:
        return abs(c_num - p_num) <= 1e-12
    return curr_val == prev_val


def sanitize_balance_state_payload(
    payload: Mapping[str, Any],
    *,
    prev_state: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Strip internal scratch keys via SSOT and preserve balance timestamps on price-only ``pos:*`` updates."""
    prev_map = prev_state if isinstance(prev_state, Mapping) else {}
    cleaned: dict[str, Any] = {}
    for k, v in payload.items():
        if not isinstance(v, Mapping):
            cleaned[k] = v
            continue
        entry = strip_internal_balance_scratch_keys(v)
        if isinstance(k, str) and k.startswith("pos:"):
            for bal_ts_key in ("base_bal_ts", "quote_bal_ts"):
                if bal_ts_key in entry and normalize_epoch_sec(entry.get(bal_ts_key), default=None) is None:
                    entry.pop(bal_ts_key, None)
            prev_entry = prev_map.get(k)
            if isinstance(prev_entry, Mapping):
                prev_ts = normalize_epoch_sec(prev_entry.get("ts"), default=0.0) or 0.0
                new_ts = normalize_epoch_sec(entry.get("ts"), default=0.0) or 0.0
                if prev_ts > 0.0 and new_ts > prev_ts:
                    curr_locked_base = (
                        entry.get("locked_base_qty") if entry.get("locked_base_qty") is not None else entry.get("locked")
                    )
                    prev_locked_base = (
                        prev_entry.get("locked_base_qty")
                        if prev_entry.get("locked_base_qty") is not None
                        else prev_entry.get("locked")
                    )
                    if (
                        "base_bal_ts" not in entry
                        and _same_balance_qty(entry.get("free_base"), prev_entry.get("free_base"))
                        and _same_balance_qty(curr_locked_base, prev_locked_base, absent_as_zero=True)
                    ):
                        prev_bts = normalize_epoch_sec(prev_entry.get("base_bal_ts"), default=0.0) or prev_ts
                        if prev_bts > 0.0:
                            entry["base_bal_ts"] = prev_bts
                    curr_locked_quote = (
                        entry.get("locked_quote_usd")
                        if entry.get("locked_quote_usd") is not None
                        else entry.get("locked_quote")
                    )
                    prev_locked_quote = (
                        prev_entry.get("locked_quote_usd")
                        if prev_entry.get("locked_quote_usd") is not None
                        else prev_entry.get("locked_quote")
                    )
                    if (
                        "quote_bal_ts" not in entry
                        and _same_balance_qty(entry.get("free_quote"), prev_entry.get("free_quote"))
                        and _same_balance_qty(curr_locked_quote, prev_locked_quote, absent_as_zero=True)
                    ):
                        prev_qts = normalize_epoch_sec(prev_entry.get("quote_bal_ts"), default=0.0) or prev_ts
                        if prev_qts > 0.0:
                            entry["quote_bal_ts"] = prev_qts
        cleaned[k] = entry
    return cleaned


class VenueBalancesStateStore(StateStoreBase):
    """Exclusive 1:1 owner for reading and locked-updating ``venue_balances_state.json``."""

    FILENAME = VENUE_BALANCES_STATE_FILENAME
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
        """Read ``venue_balances_state.json`` across ``state/`` and legacy root layouts."""
        return cls.read_dict(
            cachedb_dir,
            use_cache=use_cache,
            read_json_fn=read_json_fn,
            require_exists=require_exists,
            recover_previous=True,
        )

    @classmethod
    def upsert_locked(
        cls,
        bcache_path: str,
        updated_bals: Mapping[str, Any],
        *,
        fallback_state: Optional[Mapping[str, Any]] = None,
        read_disk: bool = True,
        save_fn: Optional[Callable[..., Any]] = None,
        indent: int = 2,
    ) -> bool:
        """Single canonical locked producer writer for ``venue_balances_state.json``."""
        if not bcache_path or not isinstance(updated_bals, Mapping) or not updated_bals:
            return False
        resolved_path = cls._resolve_disk_path(bcache_path)
        os.makedirs(os.path.dirname(os.path.abspath(resolved_path)), exist_ok=True)
        with generation_file_lock(resolved_path):
            disk_exists = cls.has_disk_file(resolved_path)
            disk_bals = (
                read_json_dict(resolved_path, default=None, recover_previous=True, repair_files=True)
                if (read_disk and disk_exists)
                else None
            )
            disk_dict = disk_bals if isinstance(disk_bals, dict) else None
            src_bals: dict[str, Any] = (
                dict(disk_dict)
                if disk_dict is not None
                else (dict(fallback_state) if isinstance(fallback_state, Mapping) else {})
            )
            updated_upper_keys = {str(uk).upper() for uk in updated_bals}
            base_bals = {
                k: v
                for k, v in src_bals.items()
                if not (isinstance(k, str) and k.startswith(("open_orders:", "open_orders_verified_ts:")))
                and (k in updated_bals or str(k).upper() not in updated_upper_keys)
            }
            merged_raw = {**base_bals, **updated_bals}
            sanitized = sanitize_balance_state_payload(merged_raw, prev_state=src_bals)
            should_write = (
                (sanitized != disk_dict)
                if disk_dict is not None
                else (sanitized != src_bals or (bool(sanitized) and not os.path.exists(resolved_path)))
            )
            if not should_write:
                return False
            if callable(save_fn):
                save_fn(bcache_path, sanitized, indent=indent)
            else:
                save_json_state(resolved_path, sanitized, indent=indent)
            return True
