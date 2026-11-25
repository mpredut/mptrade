"""Layer 2.5 SSOT base contract for 1:1 typed state file stores.

Every critical ``mptrade`` state file owns exactly one module in ``state_stores/<file_stem>.py``
exposing exactly one ``<PascalCase>Store`` subclass of ``StateStoreBase``.
Modules in ``state_stores/`` depend strictly on Layer 1-2 primitives (``botcore``, ``cachepaths``,
``state_io``, ``core/*``) so every upper layer can read/write through the single owner without
import cycles.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable, Optional

from cachepaths import (
    cache_path,
    clean_scalar_str,
    resolve_cachedb_filepath,
    resolve_statedb_dir_from_cachedb,
    resolve_statedb_filepath,
    state_path,
)
from state_io import (
    generation_file_lock,
    load_json_state,
    read_json_dict,
    read_json_dict_cached,
    save_json_state,
)


@dataclass(frozen=True, slots=True)
class StateFileSpec:
    """Declarative SSOT specification binding one critical disk file 1:1 to its owner store module & class."""

    filename: str
    const_name: str
    owner_module: str
    store_class: str
    subdomain: str = "state"
    is_jsonl: bool = False
    storage_root: str = "statedb"


class StateStoreBase:
    """Level 2 reusable base for 1:1 typed durable state and multi-key cache file owners."""

    FILENAME: str = ""
    SUBDOMAIN: str = "state"
    IS_JSONL: bool = False
    STORAGE_ROOT: str = "statedb"

    @classmethod
    def resolve_path(cls, cachedb_dir: Optional[str] = None, *, migrate: bool = False) -> str:
        """Resolve the canonical on-disk path for this store's file (honoring ``STORAGE_ROOT``)."""
        if cls.STORAGE_ROOT == "cachedb":
            if cachedb_dir and (clean_dir := clean_scalar_str(cachedb_dir)):
                return str(resolve_cachedb_filepath(os.path.join(clean_dir, cls.FILENAME), migrate=migrate))
            return str(resolve_cachedb_filepath(cache_path(cls.FILENAME), migrate=migrate))
        if cachedb_dir and (clean_dir := clean_scalar_str(cachedb_dir)):
            return str(resolve_statedb_filepath(os.path.join(clean_dir, cls.FILENAME), migrate=migrate))
        return str(resolve_statedb_filepath(state_path(cls.FILENAME), migrate=migrate))

    @classmethod
    def candidate_paths(cls, cachedb_dir: str) -> list[str]:
        """Return deduplicated candidate disk paths honoring ``STORAGE_ROOT`` (never probing phantom ``statedb`` cache dirs)."""
        if cls.STORAGE_ROOT == "cachedb":
            cands = [
                cls.resolve_path(cachedb_dir),
                os.path.join(cachedb_dir, cls.SUBDOMAIN, cls.FILENAME),
            ]
            if cls.SUBDOMAIN == "account":
                cands.append(os.path.join(cachedb_dir, "orders", cls.FILENAME))
            elif cls.SUBDOMAIN == "llm":
                cands.append(os.path.join(cachedb_dir, "system", cls.FILENAME))
            cands.append(os.path.join(cachedb_dir, cls.FILENAME))
            return list(dict.fromkeys(cands))
        statedb_dir = resolve_statedb_dir_from_cachedb(cachedb_dir)
        cands = [
            cls.resolve_path(cachedb_dir),
            os.path.join(statedb_dir, cls.SUBDOMAIN, cls.FILENAME),
            os.path.join(cachedb_dir, cls.SUBDOMAIN, cls.FILENAME),
        ]
        if cls.SUBDOMAIN in ("portfolio", "bots"):
            cands.append(os.path.join(cachedb_dir, "state", cls.FILENAME))
        elif cls.SUBDOMAIN == "llm":
            cands.append(os.path.join(cachedb_dir, "system", cls.FILENAME))
        cands.append(os.path.join(cachedb_dir, cls.FILENAME))
        return list(dict.fromkeys(cands))

    @classmethod
    def _resolve_disk_path(cls, path: str) -> str:
        """Resolve an explicit path through canonical cachedb/statedb subdomain routing."""
        return str(resolve_cachedb_filepath(path)) if path else ""

    @classmethod
    def has_disk_file(cls, path: str) -> bool:
        """Return True when ``path`` or its crash-recovery ``.previous`` backup exists."""
        return bool(path and (os.path.exists(path) or os.path.exists(f"{path}.previous")))

    @classmethod
    def generation_lock(cls, path: str):
        """Return a cross-process generation lock context manager for ``path``."""
        return generation_file_lock(path)

    @classmethod
    def read_dict(
        cls,
        cachedb_dir: Optional[str],
        *,
        use_cache: bool = False,
        read_json_fn: Optional[Callable[[str], Any]] = None,
        require_exists: bool = False,
        recover_previous: bool = True,
    ) -> dict[str, Any]:
        """Read this store's JSON dict from ``cachedb_dir`` with ``.previous`` crash recovery."""
        if not cachedb_dir:
            return {}
        for cand in cls.candidate_paths(cachedb_dir):
            has_file = cls.has_disk_file(cand)
            if require_exists and not has_file:
                continue
            if callable(read_json_fn):
                raw = read_json_fn(cand)
            elif use_cache:
                if not has_file:
                    continue
                raw = read_json_dict_cached(cand)
                if isinstance(raw, dict):
                    return raw
                continue
            else:
                if not has_file:
                    continue
                raw = read_json_dict(cand, default=None, recover_previous=recover_previous)
                if isinstance(raw, dict):
                    return raw
                continue
            if isinstance(raw, dict) and raw:
                return raw
        return {}

    @classmethod
    def read_file_dict(
        cls,
        path: str,
        *,
        default: Any = None,
        recover_previous: bool = True,
        repair_files: bool = False,
    ) -> Any:
        """Read a JSON dict directly from an explicit file ``path``."""
        return read_json_dict(
            path,
            default=default,
            recover_previous=recover_previous,
            repair_files=repair_files,
        )

    @classmethod
    def load_state_file(
        cls,
        path: str,
        *,
        default_factory: Callable[[], Any] = dict,
        fail_closed: bool = True,
        label: str = "",
        root_type: type = dict,
        repair_files: bool = True,
    ) -> Any:
        """Load a durable JSON state file with fail-closed corruption protection."""
        return load_json_state(
            path,
            default_factory=default_factory,
            fail_closed=fail_closed,
            label=label or cls.__name__,
            root_type=root_type,
            repair_files=repair_files,
        )

    @classmethod
    def save_state_file(
        cls,
        path: str,
        value: Any,
        *,
        indent: Optional[int] = 2,
        sort_keys: bool = False,
        separators: Any = None,
        skip_if_unchanged: bool = False,
    ) -> None:
        """Atomically persist ``value`` to ``path`` with ``.previous`` staging."""
        target_dir = os.path.dirname(os.path.abspath(path))
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        kwargs: dict[str, Any] = {}
        if indent is not None:
            kwargs["indent"] = indent
        if sort_keys:
            kwargs["sort_keys"] = True
        if separators is not None:
            kwargs["separators"] = separators
        if skip_if_unchanged:
            kwargs["skip_if_unchanged"] = True
        save_json_state(path, value, **kwargs)
