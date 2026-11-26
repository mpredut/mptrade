"""1:1 SSOT store owner for ``price_watcher_state.json``."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from cachepaths import (
    EXACT_LEGACY_FILENAMES,
    PRICE_WATCHER_STATE_FILENAME,
    clean_scalar_str,
    get_cachedb_dir,
    get_statedb_dir,
)
from state_io import load_json_state, save_json_state

from .base import StateStoreBase


class PriceWatcherStateStore(StateStoreBase):
    """Exclusive 1:1 owner for ``market_monitor/pricewatcher.py`` notification cooldown & escalation state."""

    FILENAME = PRICE_WATCHER_STATE_FILENAME
    SUBDOMAIN = "prices"
    STORAGE_ROOT = "statedb"

    @classmethod
    def default_state_path(cls, workspace_dir: Optional[str] = None) -> Path:
        """Return canonical ``statedb/prices/price_watcher_state.json`` path."""
        if workspace_dir and (clean_ws := clean_scalar_str(workspace_dir)):
            return Path(clean_ws) / "statedb" / cls.SUBDOMAIN / cls.FILENAME
        return Path(get_statedb_dir()) / cls.SUBDOMAIN / cls.FILENAME

    @classmethod
    def _legacy_candidates(cls, target: Path) -> list[Path]:
        """Return ordered legacy ``cache_watcher_all_price.cooldowns.json`` paths for automatic cutover."""
        legacy_names = EXACT_LEGACY_FILENAMES.get(cls.FILENAME, ())
        if not legacy_names or target.name != cls.FILENAME:
            return []
        cands: list[Path] = []
        # 1. Sibling cachedb/prices/<legacy> relative to <ws>/statedb/prices/price_watcher_state.json
        in_statedb_tree = target.parent.name == cls.SUBDOMAIN and target.parent.parent.name == "statedb"
        if in_statedb_tree:
            ws_root = target.parent.parent.parent
            for leg_name in legacy_names:
                cands.append(ws_root / "cachedb" / cls.SUBDOMAIN / leg_name)
                cands.append(ws_root / "cachedb" / leg_name)
        # 2. Active process cachedb/prices/<legacy> ONLY when target belongs to the active statedb root
        # (prevents isolated tmp_path tests from ever migrating a live workspace's legacy file).
        active_sdir = Path(get_statedb_dir())
        target_abs = os.path.abspath(str(target))
        active_sdir_abs = os.path.abspath(str(active_sdir))
        same_active_root = (
            target_abs == active_sdir_abs
            or target_abs.startswith(active_sdir_abs + os.sep)
        )
        if same_active_root or not in_statedb_tree:
            cdir = Path(get_cachedb_dir())
            for leg_name in legacy_names:
                if same_active_root:
                    cands.append(cdir / cls.SUBDOMAIN / leg_name)
                cands.append(target.with_name(leg_name))
        else:
            for leg_name in legacy_names:
                cands.append(target.with_name(leg_name))
        out: list[Path] = []
        seen: set[str] = set()
        for cand in cands:
            key = os.path.abspath(str(cand))
            if key not in seen and key != os.path.abspath(str(target)):
                seen.add(key)
                out.append(cand)
        return out

    @classmethod
    def migrate_legacy_if_needed(cls, path: str | os.PathLike) -> bool:
        """Move legacy ``cachedb/prices/cache_watcher_all_price.cooldowns.json`` into ``statedb/prices/``."""
        target = Path(path)
        if cls.has_disk_file(str(target)):
            return False
        for legacy in cls._legacy_candidates(target):
            if not cls.has_disk_file(str(legacy)):
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                if legacy.exists():
                    os.replace(legacy, target)
                leg_prev = Path(f"{legacy}.previous")
                tgt_prev = Path(f"{target}.previous")
                if leg_prev.exists() and not tgt_prev.exists():
                    os.replace(leg_prev, tgt_prev)
                return True
            except OSError:
                return False
        return False

    @classmethod
    def load_state(cls, path: str | os.PathLike) -> dict[str, Any]:
        """Load persisted price-watcher cooldown state from ``path`` (migrating legacy file when present)."""
        target = Path(path)
        cls.migrate_legacy_if_needed(target)
        if not cls.has_disk_file(str(target)):
            return {}
        raw = load_json_state(
            str(target),
            default_factory=dict,
            fail_closed=False,
            label="PriceChecker cooldowns",
        )
        return raw if isinstance(raw, dict) else {}

    @classmethod
    def save_state(cls, path: str | os.PathLike, payload: dict[str, Any]) -> bool:
        """Atomically persist price-watcher cooldown state to ``path``."""
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            save_json_state(str(target), payload, indent=2)
            return True
        except Exception:
            return False
