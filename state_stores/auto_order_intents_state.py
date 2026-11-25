"""1:1 SSOT store owner for ``auto_order_intents_state.json``."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import os
import threading
from typing import Any, Callable, Optional

from botcore import FileLock, _parse_bool_flag, clean_json_value, clean_scalar_str
from cachepaths import (
    AUTO_ORDER_INTENTS_STATE_FILENAME,
    resolve_statedb_dir_from_cachedb,
    resolve_statedb_filepath,
)
from state_io import (
    StateReadError,
    generation_lock_path,
    load_json_state,
    read_json_recovering,
    save_json_state,
)

from .base import StateStoreBase

_INTENTS_LOCK_LOCAL = threading.local()
_LEGACY_INTENTS_FILENAMES: tuple[str, ...] = (
    "autonomous_order_intents_state.json",
    "autonomous_trade_intents_state.json",
    "autonomous_trade_intents.json",
)


class AutoOrderIntentsStateStore(StateStoreBase):
    """Exclusive 1:1 owner for durable read/write/lock access to ``auto_order_intents_state.json``."""

    FILENAME = AUTO_ORDER_INTENTS_STATE_FILENAME
    SUBDOMAIN = "orders"

    @classmethod
    def candidate_paths(cls, cachedb_dir: str) -> list[str]:
        """Return deduplicated candidate paths across ``statedb/orders/``, ``cachedb/orders/``, ``cachedb/state/``, and root."""
        statedb_dir = resolve_statedb_dir_from_cachedb(cachedb_dir)
        dirs = tuple(
            dict.fromkeys((
                os.path.join(statedb_dir, cls.SUBDOMAIN),
                os.path.join(cachedb_dir, cls.SUBDOMAIN),
                os.path.join(cachedb_dir, "state"),
                cachedb_dir,
            ))
        )
        names = (cls.FILENAME, *_LEGACY_INTENTS_FILENAMES)
        cands: list[str] = [cls.resolve_path(cachedb_dir)]
        for d in dirs:
            for nm in names:
                cands.append(os.path.join(d, nm))
        return list(dict.fromkeys(cands))

    @classmethod
    @contextmanager
    def lock(cls, intents_file: str):
        """Reentrant (per-thread) cross-process lock serializing read-modify-write of the durable intents file."""
        raw_p = clean_scalar_str(os.fspath(intents_file)) if isinstance(intents_file, (str, os.PathLike)) else None
        if raw_p is None:
            raise ValueError(f"Invalid durable intents_file: {intents_file!r}")
        key = generation_lock_path(str(resolve_statedb_filepath(raw_p, migrate=True)))
        depths = getattr(_INTENTS_LOCK_LOCAL, "depths", None)
        if depths is None:
            depths = _INTENTS_LOCK_LOCAL.depths = {}
        if depths.get(key):
            depths[key] += 1
            try:
                yield
            finally:
                depths[key] -= 1
            return
        with FileLock(key):
            depths[key] = 1
            try:
                yield
            finally:
                depths.pop(key, None)

    @classmethod
    def load_state(
        cls,
        intents_file: str,
        reconcile_fn: Optional[Callable[[list[dict[str, Any]]], Any]] = None,
        *,
        read_only: bool = False,
        fail_closed: Optional[bool] = None,
    ) -> list[dict[str, Any]]:
        """Load all order intents from ``intents_file`` with ``.previous`` crash recovery."""
        is_ro = _parse_bool_flag(read_only, False)
        eff_fail_closed = (not is_ro) if fail_closed is None else _parse_bool_flag(fail_closed, not is_ro)
        lock_cm = nullcontext() if is_ro else cls.lock(intents_file)
        with lock_cm:
            data = load_json_state(
                intents_file,
                default_factory=list,
                fail_closed=eff_fail_closed,
                label="auto_order_intents",
                root_type=list,
                repair_files=not is_ro,
            )
            items = [item for item in data if isinstance(item, dict)]
            if callable(reconcile_fn):
                reconcile_fn(items)
            return items

    @classmethod
    def load_from_cachedb(
        cls,
        cachedb_dir: Optional[str],
        *,
        read_json_fn: Optional[Callable[[str], Any]] = None,
    ) -> list[dict[str, Any]]:
        """Read-only loader accepting both canonical ``list`` and legacy ``{"intents": [...]}`` schemas."""
        if not cachedb_dir:
            return []
        resolved_candidates = cls.candidate_paths(cachedb_dir)
        raw_payload: Any = None
        if callable(read_json_fn):
            for cand_p in resolved_candidates:
                try:
                    raw_payload = read_json_fn(cand_p)
                except Exception:
                    raw_payload = None
                if isinstance(raw_payload, (list, dict)) and raw_payload:
                    break
        if raw_payload is None:
            for cand_p in resolved_candidates:
                if not (os.path.isfile(cand_p) or os.path.isfile(f"{cand_p}.previous")):
                    continue
                cand_payload = read_json_recovering(cand_p, default=None)
                if isinstance(cand_payload, (list, dict)):
                    raw_payload = cand_payload
                    break
        int_list = (
            raw_payload.get("intents")
            if isinstance(raw_payload, dict)
            else (raw_payload if isinstance(raw_payload, list) else [])
        )
        return [it for it in (int_list or ()) if isinstance(it, dict)]

    @classmethod
    def save_state(
        cls,
        intents_file: str,
        intents: list[dict[str, Any]],
        log: Optional[Any] = None,
    ) -> bool:
        """Transactionally persist all intents using ``.previous`` staging and atomic replace."""
        try:
            cleaned = clean_json_value(intents if isinstance(intents, list) else [])
            with cls.lock(intents_file):
                save_json_state(intents_file, cleaned, indent=2)
            return True
        except Exception as exc:
            if log:
                log.error("Failed saving intents to %s: %s", intents_file, exc)
            return False

    @classmethod
    def upsert_intent(
        cls,
        intents_file: str,
        intent: dict[str, Any],
        *,
        extract_id_fn: Callable[[dict[str, Any]], str],
        reconcile_fn: Optional[Callable[[list[dict[str, Any]]], Any]] = None,
        log: Optional[Any] = None,
    ) -> bool:
        """Atomically merge one intent (by ``intent_id``) into the freshly reloaded durable intents file."""
        if not isinstance(intent, dict):
            return False
        iid = extract_id_fn(intent)
        with cls.lock(intents_file):
            try:
                intents = cls.load_state(
                    intents_file, reconcile_fn=reconcile_fn, fail_closed=True
                )
            except StateReadError as exc:
                if log:
                    log.error("Refusing to overwrite corrupt intents queue %s: %s", intents_file, exc)
                return False
            idx = next(
                (k for k in range(len(intents) - 1, -1, -1) if iid and extract_id_fn(intents[k]) == iid),
                None,
            )
            if idx is None:
                intents.append(intent)
            else:
                intents[idx] = intent
            return cls.save_state(intents_file, intents, log=log)
