"""1:1 SSOT store owner for ``notify_queue_state.json``."""
from __future__ import annotations

import os
from typing import Sequence

from cachepaths import (
    EXACT_LEGACY_FILENAMES,
    NOTIFY_QUEUE_STATE_FILENAME,
    get_statedb_dir,
    resolve_statedb_dir_from_cachedb,
    resolve_statedb_filepath,
)
from state_io import atomic_write_text, read_text_lines

from .base import StateStoreBase


class NotifyQueueStateStore(StateStoreBase):
    """Exclusive 1:1 owner for durable queued notifications in ``notify_queue_state.json``."""

    FILENAME = NOTIFY_QUEUE_STATE_FILENAME
    SUBDOMAIN = "system"
    IS_JSONL = True

    @classmethod
    def migrate_legacy_if_needed(cls, queue_path: str) -> bool:
        """Atomically migrate sibling legacy ``notify_queue.json`` (and sidecars) into ``notify_queue_state.json``."""
        if not queue_path or os.path.basename(queue_path) != cls.FILENAME:
            return False
        if cls.has_disk_file(queue_path):
            return False
        q_dir = os.path.dirname(queue_path)
        for leg_name in EXACT_LEGACY_FILENAMES.get(cls.FILENAME, ()):
            legacy = os.path.join(q_dir, leg_name)
            if not cls.has_disk_file(legacy):
                continue
            try:
                os.makedirs(q_dir, exist_ok=True)
                if os.path.exists(legacy):
                    os.replace(legacy, queue_path)
                for sfx in (".previous", ".generation.lock"):
                    leg_sc = f"{legacy}{sfx}"
                    tgt_sc = f"{queue_path}{sfx}"
                    if os.path.exists(leg_sc) and not os.path.exists(tgt_sc):
                        os.replace(leg_sc, tgt_sc)
                return True
            except OSError:
                return False
        return False

    @classmethod
    def default_queue_file(cls) -> str:
        """Return default canonical path ``statedb/system/notify_queue_state.json``, migrating legacy files when present."""
        target = os.path.join(get_statedb_dir(), cls.SUBDOMAIN, cls.FILENAME)
        cls.migrate_legacy_if_needed(target)
        return target

    @classmethod
    def candidate_paths(cls, cachedb_dir: str) -> list[str]:
        """Return deduplicated candidate paths across ``statedb/system/``, ``cachedb/system/``, and root."""
        statedb_dir = resolve_statedb_dir_from_cachedb(cachedb_dir)
        return list(
            dict.fromkeys((
                cls.resolve_path(cachedb_dir),
                os.path.join(statedb_dir, cls.SUBDOMAIN, cls.FILENAME),
                os.path.join(cachedb_dir, cls.SUBDOMAIN, cls.FILENAME),
                os.path.join(cachedb_dir, cls.FILENAME),
            ))
        )

    @classmethod
    def read_lines_nolock(cls, queue_path: str) -> list[str]:
        """Read raw lines (preserving line endings) from ``queue_path`` while caller holds lock."""
        cls.migrate_legacy_if_needed(queue_path)
        resolved = str(resolve_statedb_filepath(queue_path))
        if not os.path.exists(resolved):
            return []
        return list(read_text_lines(resolved, keepends=True, errors="replace"))

    @classmethod
    def write_lines_nolock(cls, queue_path: str, lines: Sequence[str]) -> None:
        """Atomically replace ``queue_path`` with newline-delimited JSON lines while caller holds lock."""
        cls.migrate_legacy_if_needed(queue_path)
        resolved = str(resolve_statedb_filepath(queue_path))
        content = "".join(f"{ln}\n" for ln in lines)
        atomic_write_text(resolved, content)
