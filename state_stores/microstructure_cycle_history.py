"""1:1 SSOT store owner for ``microstructure_cycle_history.json`` (locked read-merge-write)."""
from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Optional

from botcore import normalize_epoch_sec, safe_pos_float, safe_pos_int
from cachepaths import MICROSTRUCTURE_CYCLE_HISTORY_FILENAME
from state_io import generation_file_lock, read_json_dict, read_json_file, save_json_state

from .base import StateStoreBase


def _snap_ts(item: Any) -> float:
    if not isinstance(item, Mapping):
        return 0.0
    return normalize_epoch_sec(item.get("ts"), default=0.0) or 0.0


def _merge_symbol_history_series(
    disk_rows: Any,
    new_rows: Any,
    *,
    max_age_sec: float = 14400.0,
    max_points: int = 90,
) -> list[dict[str, Any]]:
    by_ts: dict[float, dict[str, Any]] = {}
    for src in (disk_rows, new_rows):
        if not isinstance(src, list):
            continue
        for row in src:
            if not isinstance(row, Mapping):
                continue
            ts_v = _snap_ts(row)
            if ts_v <= 0.0:
                continue
            by_ts[round(ts_v, 3)] = dict(row)
    if not by_ts:
        return []
    ordered = [by_ts[k] for k in sorted(by_ts)]
    latest_ts = _snap_ts(ordered[-1])
    if max_age_sec > 0.0 and latest_ts > 0.0:
        cutoff = latest_ts - max_age_sec
        ordered = [r for r in ordered if _snap_ts(r) >= cutoff]
    if max_points > 0 and len(ordered) > max_points:
        ordered = ordered[-max_points:]
    return ordered


class MicrostructureCycleHistoryStore(StateStoreBase):
    """Exclusive 1:1 owner for reading and locked read-merge-write updating ``microstructure_cycle_history.json``."""

    FILENAME = MICROSTRUCTURE_CYCLE_HISTORY_FILENAME
    SUBDOMAIN = "telemetry"
    STORAGE_ROOT = "cachedb"

    @classmethod
    def load_path(
        cls,
        mhist_path: str,
        *,
        read_json_fn: Optional[Callable[..., Any]] = None,
    ) -> dict[str, Any]:
        """Read ``microstructure_cycle_history.json`` from an explicit resolved path."""
        if not mhist_path or not cls.has_disk_file(mhist_path):
            return {}
        if callable(read_json_fn):
            try:
                raw = read_json_fn(mhist_path, default={})
            except TypeError:
                raw = read_json_fn(mhist_path)
            return dict(raw) if isinstance(raw, dict) else {}
        raw_d = read_json_dict(mhist_path, default=None, recover_previous=True)
        if isinstance(raw_d, dict):
            return dict(raw_d)
        raw_fb = read_json_file(mhist_path, default={})
        return dict(raw_fb) if isinstance(raw_fb, dict) else {}

    @classmethod
    def upsert_history_locked(
        cls,
        mhist_path: str,
        updates: Mapping[str, Any],
        *,
        max_age_sec: float = 14400.0,
        max_points: int = 90,
        indent: int = 2,
    ) -> bool:
        """Atomically merge per-symbol cycle history updates under ``generation_file_lock`` (prevents lost updates)."""
        if not mhist_path or not isinstance(updates, Mapping):
            return False
        os.makedirs(os.path.dirname(os.path.abspath(mhist_path)), exist_ok=True)
        with generation_file_lock(mhist_path):
            disk_exists = cls.has_disk_file(mhist_path)
            disk_raw = (
                read_json_dict(mhist_path, default=None, recover_previous=True, repair_files=True)
                if disk_exists
                else None
            )
            disk_dict = dict(disk_raw) if isinstance(disk_raw, dict) else {}
            merged: dict[str, Any] = dict(disk_dict)
            for sym_k, rows in updates.items():
                if not isinstance(sym_k, str) or not sym_k.strip():
                    continue
                if isinstance(rows, list):
                    prev_rows_acc: list[Any] = []
                    stale_case_keys = [
                        dk for dk in list(merged)
                        if dk != sym_k and str(dk).upper() == sym_k.upper() and dk not in updates
                    ]
                    for sk in stale_case_keys:
                        sk_val = merged.pop(sk, None)
                        if isinstance(sk_val, list):
                            prev_rows_acc.extend(sk_val)
                    prev_own = merged.get(sym_k)
                    if isinstance(prev_own, list):
                        prev_rows_acc.extend(prev_own)
                    if not rows and prev_own is None and not prev_rows_acc:
                        merged[sym_k] = []
                    else:
                        merged[sym_k] = _merge_symbol_history_series(
                            prev_rows_acc,
                            rows,
                            max_age_sec=safe_pos_float(max_age_sec, default=14400.0) or 14400.0,
                            max_points=safe_pos_int(max_points, default=90) or 90,
                        )
                else:
                    merged[sym_k] = rows
            if disk_exists and merged == disk_dict:
                return True
            save_json_state(mhist_path, merged, indent=indent)
            return True
