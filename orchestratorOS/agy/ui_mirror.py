"""Mirror permanent topic threads from the isolated agy-bot store into the Web UI store.

Only content files, the summary row and annotations are written. The daemon's in-memory list, its proto
caches and the visible-dir layout are owned by the daemon and agy_store_fix; the mirrored state surfaces
on the next idle `manage_agy.sh ui-sync` restart.
"""
from __future__ import annotations

import fcntl
import logging
import os
import shutil
import sqlite3
from typing import Optional

from orchestratorOS.admin.agy_store_fix import LOCK, visible_conversations_dir
from orchestratorOS.agy.paths import resolve_bot_cli_base, resolve_ui_cli_base
from orchestratorOS.agy.summary_store import (
    conversation_db,
    get_thread_step_count,
    read_summary_row,
    summaries_db,
    upsert_ui_visible_summary,
)

logger = logging.getLogger("orchestratorOS.agy.ui_mirror")

UI_TRAJECTORY_SOURCE = 1   # trajectory_meta.source of conversations created in the UI (CLI print mode = 17)


def _checkpoint_wal(db_path: str) -> None:
    wal = db_path + "-wal"
    if os.path.exists(wal) and os.path.getsize(wal) > 0:
        conn = sqlite3.connect(db_path, timeout=15)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
        finally:
            conn.close()


def _remove_wal_sidecars(db_path: str) -> None:
    for ext in ("-wal", "-shm"):
        sidecar = db_path + ext
        if os.path.exists(sidecar):
            try:
                os.remove(sidecar)
            except OSError:
                pass


def sync_topic_thread_from_ui(
    cid: str,
    bot_cli_base: Optional[str] = None,
    ui_cli_base: Optional[str] = None,
) -> bool:
    """Pull newer user turns from the Web UI store back into the isolated bot store before a CLI turn."""
    dst_base = resolve_bot_cli_base(bot_cli_base)
    src_base = resolve_ui_cli_base(ui_cli_base)
    if not cid or os.path.abspath(src_base) == os.path.abspath(dst_base):
        return False
    ui_db = conversation_db(src_base, cid)
    if not os.path.exists(ui_db):
        vis_db = os.path.join(visible_conversations_dir(src_base), f"{cid}.db")
        if not os.path.exists(vis_db):
            return False
        ui_db = vis_db

    ui_row = read_summary_row(src_base, cid)
    bot_row = read_summary_row(dst_base, cid)
    if str((ui_row or {}).get("status") or "") == "CASCADE_RUN_STATUS_RUNNING":
        return False
    ui_steps = int((ui_row or {}).get("step_count") or 0)
    bot_steps = int((bot_row or {}).get("step_count") or 0)
    bot_db = conversation_db(dst_base, cid)
    bot_exists = os.path.exists(bot_db)
    ui_mtime = max((os.stat(p).st_mtime for p in (ui_db, ui_db + "-wal") if os.path.exists(p)), default=0.0)
    bot_mtime = max((os.stat(p).st_mtime for p in (bot_db, bot_db + "-wal") if os.path.exists(p)), default=0.0)

    ui_is_ahead = (
        ui_steps > bot_steps
        or (not bot_exists and ui_steps > 0)
        or (bot_exists and ui_steps >= bot_steps and ui_mtime > bot_mtime + 2.0)
    )
    if not ui_is_ahead:
        return False

    lock_fh = None
    try:
        try:
            lock_fh = open(LOCK, "w")
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
        except OSError:
            pass
        _checkpoint_wal(ui_db)
        if bot_exists:
            _checkpoint_wal(bot_db)

        os.makedirs(os.path.dirname(bot_db), exist_ok=True)
        shutil.copy2(ui_db, bot_db + ".tmp")
        _remove_wal_sidecars(bot_db)
        os.replace(bot_db + ".tmp", bot_db)
        mtime = os.stat(ui_db).st_mtime
        os.utime(bot_db, (mtime, mtime))

        ui_brain = os.path.join(src_base, "brain", cid)
        if os.path.isdir(ui_brain):
            shutil.copytree(ui_brain, os.path.join(dst_base, "brain", cid), dirs_exist_ok=True)

        bot_sum_db = summaries_db(dst_base)
        if ui_row is not None and os.path.exists(bot_sum_db):
            merged = dict(ui_row)
            if bot_row and bot_row.get("project_id"):
                merged["project_id"] = bot_row["project_id"]
            if not str(merged.get("title") or "").strip() and bot_row and bot_row.get("title"):
                merged["title"] = bot_row["title"]
            conn = sqlite3.connect(bot_sum_db)
            try:
                cols = [r[1] for r in conn.execute("PRAGMA table_info(conversation_summaries)")]
                common = [c for c in cols if c in merged]
                conn.execute(
                    f"INSERT OR REPLACE INTO conversation_summaries ({','.join(common)}) "
                    f"VALUES ({','.join('?' for _ in common)})",
                    [merged[c] for c in common],
                )
                conn.commit()
            finally:
                conn.close()
        return True
    except (OSError, sqlite3.Error) as e:
        logger.warning("Failed syncing topic thread %s from the UI store: %s", cid, e)
        return False
    finally:
        if lock_fh is not None:
            try:
                lock_fh.close()
            except OSError:
                pass


def mirror_topic_thread_to_ui(
    cid: str,
    title: str,
    bot_cli_base: Optional[str] = None,
    ui_cli_base: Optional[str] = None,
) -> bool:
    """Copy one topic thread into the UI store as a UI-visible conversation named ``title``."""
    src_base = resolve_bot_cli_base(bot_cli_base)
    dst_base = resolve_ui_cli_base(ui_cli_base)
    src_db = conversation_db(src_base, cid)
    if os.path.abspath(src_base) == os.path.abspath(dst_base) or not os.path.exists(src_db):
        return False
    row = read_summary_row(src_base, cid)
    if row is None or not os.path.exists(os.path.join(dst_base, "conversation_summaries.db")):
        return False
    _, ui_steps, ui_status = get_thread_step_count(cid, dst_base)
    bot_steps = int(row.get("step_count") or 0)
    if ui_status == "CASCADE_RUN_STATUS_RUNNING":
        return False
    if ui_steps > bot_steps:
        sync_topic_thread_from_ui(cid, bot_cli_base=src_base, ui_cli_base=dst_base)
        return False
    lock_fh = None
    try:
        try:
            lock_fh = open(LOCK, "w")
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
        except OSError:
            pass
        _checkpoint_wal(src_db)
        dst_db = conversation_db(dst_base, cid)
        os.makedirs(os.path.dirname(dst_db), exist_ok=True)
        if os.path.exists(dst_db):
            _checkpoint_wal(dst_db)
        shutil.copy2(src_db, dst_db + ".tmp")
        _remove_wal_sidecars(dst_db)
        os.replace(dst_db + ".tmp", dst_db)
        conn = sqlite3.connect(dst_db)
        try:
            conn.execute("UPDATE trajectory_meta SET source = ?", (UI_TRAJECTORY_SOURCE,))
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
        _remove_wal_sidecars(dst_db)
        mtime = os.stat(dst_db).st_mtime
        os.utime(src_db, (mtime, mtime))

        # A previous mirror hard-linked into the visible dir is now stale; drop it so agy_store_fix moves the
        # rewritten content there on the next idle restart instead of treating it as a diverged copy.
        stale = os.path.join(visible_conversations_dir(dst_base), f"{cid}.db")
        if os.path.exists(stale) and not os.path.samefile(stale, dst_db):
            os.remove(stale)

        src_brain = os.path.join(src_base, "brain", cid)
        if os.path.isdir(src_brain):
            shutil.copytree(src_brain, os.path.join(dst_base, "brain", cid), dirs_exist_ok=True)

        upsert_ui_visible_summary(dst_base, row, title)
        return True
    except (OSError, sqlite3.Error) as e:
        logger.warning("Failed mirroring topic thread %s to the UI store: %s", cid, e)
        return False
    finally:
        if lock_fh is not None:
            try:
                lock_fh.close()
            except OSError:
                pass
