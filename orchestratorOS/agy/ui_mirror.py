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
import sys
from typing import Optional

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from botcore import safe_nonneg_int
from orchestratorOS.agy.paths import (
    resolve_bot_cli_base,
    resolve_store_lock_path,
    resolve_ui_cli_base,
    visible_conversations_dir,
)
from orchestratorOS.agy.summary_store import (
    _clean_cid_set,
    checkpoint_db,
    conversation_db,
    daemon_loaded_titles,
    delete_threads,
    drop_wal_sidecars,
    get_thread_step_count,
    read_summary_row,
    summaries_db,
    upsert_summary_row_dict,
    upsert_ui_visible_summary,
)
from state_io import durable_replace_file

logger = logging.getLogger("orchestratorOS.agy.ui_mirror")

UI_TRAJECTORY_SOURCE = 1   # trajectory_meta.source of conversations created in the UI (CLI print mode = 17)


def sync_topic_thread_from_ui(
    cid: str,
    bot_cli_base: Optional[str] = None,
    ui_cli_base: Optional[str] = None,
) -> bool:
    """Pull newer user turns from the Web UI store back into the isolated bot store before a CLI turn."""
    dst_base = resolve_bot_cli_base(bot_cli_base)
    src_base = resolve_ui_cli_base(ui_cli_base)
    clean_cids = _clean_cid_set([cid])
    if not clean_cids or os.path.abspath(src_base) == os.path.abspath(dst_base):
        return False
    cid = next(iter(clean_cids))
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
    ui_steps = safe_nonneg_int((ui_row or {}).get("step_count"))
    bot_steps = safe_nonneg_int((bot_row or {}).get("step_count"))
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
            lock_fh = open(resolve_store_lock_path(), "w")
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
        except OSError:
            pass
        checkpoint_db(ui_db)
        if bot_exists:
            checkpoint_db(bot_db)

        os.makedirs(os.path.dirname(bot_db), exist_ok=True)
        shutil.copy2(ui_db, bot_db + ".tmp")
        drop_wal_sidecars(bot_db, force=True)
        durable_replace_file(bot_db + ".tmp", bot_db)
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
            upsert_summary_row_dict(dst_base, merged)
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


def _clone_trajectory_for_ghost_cid(src_db: str, ghost_db: str, src_cid: str, ghost_cid: str) -> bool:
    """Clone ``src_db`` into ``ghost_db``, rewriting equal-length ``src_cid`` -> ``ghost_cid`` across tables.

    When a permanent topic rotates while the Web UI daemon is still running (e.g. ``ui-sync`` deferred by
    active interactive sessions), the daemon's in-memory list still points ``clean_title`` at ``ghost_cid``.
    Materializing ``ghost_db`` with ``cascade_id = ghost_cid`` allows clicking the topic in the Web UI to
    open the live rotated turns immediately without waiting for a daemon restart; ``agy_store_fix.py pre``
    reaps ``ghost_db`` automatically on the next restart because ``ghost_cid`` has no summary row in SQLite.
    """
    src_b = src_cid.encode("utf-8")
    dst_b = ghost_cid.encode("utf-8")
    if not src_b or len(src_b) != len(dst_b) or src_cid == ghost_cid or not os.path.exists(src_db):
        return False
    tmp_db = ghost_db + ".tmp"
    os.makedirs(os.path.dirname(ghost_db), exist_ok=True)
    if os.path.exists(ghost_db):
        checkpoint_db(ghost_db)
    shutil.copy2(src_db, tmp_db)
    conn = sqlite3.connect(tmp_db)
    try:
        tables = [
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if isinstance(r[0], str) and r[0] and not r[0].startswith("sqlite_")
        ]
        for tbl in tables:
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info(`{tbl}`)").fetchall()]
            if not cols:
                continue
            col_sql = ",".join(f"`{c}`" for c in cols)
            rows = conn.execute(f"SELECT rowid, {col_sql} FROM `{tbl}`").fetchall()
            for row in rows:
                rid = row[0]
                vals = list(row[1:])
                changed = False
                for idx, val in enumerate(vals):
                    if isinstance(val, str) and src_cid in val:
                        vals[idx] = val.replace(src_cid, ghost_cid)
                        changed = True
                    elif isinstance(val, (bytes, bytearray, memoryview)):
                        raw_v = bytes(val)
                        if src_b in raw_v:
                            vals[idx] = raw_v.replace(src_b, dst_b)
                            changed = True
                if changed:
                    set_sql = ",".join(f"`{c}`=?" for c in cols)
                    conn.execute(f"UPDATE `{tbl}` SET {set_sql} WHERE rowid=?", vals + [rid])
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    drop_wal_sidecars(tmp_db, force=True)
    drop_wal_sidecars(ghost_db, force=True)
    durable_replace_file(tmp_db, ghost_db)
    return True


def _mirror_to_daemon_ghost_cids(
    dst_base: str,
    dst_db: str,
    src_brain: str,
    active_cid: str,
    clean_title: str,
) -> None:
    """Keep any retired daemon-loaded conversation IDs for ``clean_title`` readable with latest topic turns."""
    if not clean_title or not os.path.exists(dst_db):
        return
    loaded_titles = daemon_loaded_titles(dst_base)
    if not loaded_titles:
        return
    for ghost_cid, ghost_title in loaded_titles.items():
        if ghost_cid == active_cid or (ghost_title or "").strip() != clean_title:
            continue
        if read_summary_row(dst_base, ghost_cid) is not None:
            continue
        try:
            if _clone_trajectory_for_ghost_cid(dst_db, conversation_db(dst_base, ghost_cid), active_cid, ghost_cid):
                if src_brain and os.path.isdir(src_brain):
                    shutil.copytree(src_brain, os.path.join(dst_base, "brain", ghost_cid), dirs_exist_ok=True)
        except (OSError, sqlite3.Error) as exc:
            logger.debug("Failed updating ghost topic trajectory %s -> %s: %s", active_cid, ghost_cid, exc)


def mirror_topic_thread_to_ui(
    cid: str,
    title: str,
    bot_cli_base: Optional[str] = None,
    ui_cli_base: Optional[str] = None,
) -> bool:
    """Copy one topic thread into the UI store as a UI-visible conversation named ``title``."""
    clean_cids = _clean_cid_set([cid])
    if not clean_cids:
        return False
    cid = next(iter(clean_cids))
    src_base = resolve_bot_cli_base(bot_cli_base)
    dst_base = resolve_ui_cli_base(ui_cli_base)
    src_db = conversation_db(src_base, cid)
    if os.path.abspath(src_base) == os.path.abspath(dst_base) or not os.path.exists(src_db):
        return False
    row = read_summary_row(src_base, cid)
    if row is None or not os.path.exists(os.path.join(dst_base, "conversation_summaries.db")):
        return False
    _, ui_steps, ui_status = get_thread_step_count(cid, dst_base)
    bot_steps = safe_nonneg_int(row.get("step_count"))
    if ui_status == "CASCADE_RUN_STATUS_RUNNING":
        return False
    if ui_steps > bot_steps:
        sync_topic_thread_from_ui(cid, bot_cli_base=src_base, ui_cli_base=dst_base)
        return False
    lock_fh = None
    try:
        try:
            lock_fh = open(resolve_store_lock_path(), "w")
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
        except OSError:
            pass
        checkpoint_db(src_db)
        dst_db = conversation_db(dst_base, cid)
        os.makedirs(os.path.dirname(dst_db), exist_ok=True)
        if os.path.exists(dst_db):
            checkpoint_db(dst_db)
        shutil.copy2(src_db, dst_db + ".tmp")
        drop_wal_sidecars(dst_db, force=True)
        durable_replace_file(dst_db + ".tmp", dst_db)
        conn = sqlite3.connect(dst_db)
        try:
            conn.execute("UPDATE trajectory_meta SET source = ?", (UI_TRAJECTORY_SOURCE,))
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
        drop_wal_sidecars(dst_db, force=True)
        mtime = os.stat(dst_db).st_mtime
        os.utime(src_db, (mtime, mtime))

        # Drop any stale visible-dir copy so pending_ui_sync() detects the updated topic trajectory and
        # the next agy_store_fix pre/post cycle moves and re-links the fresh file cleanly (diverged=0).
        visible_db = os.path.join(visible_conversations_dir(dst_base), f"{cid}.db")
        if os.path.lexists(visible_db) and (not os.path.exists(visible_db) or not os.path.samefile(visible_db, dst_db)):
            os.remove(visible_db)

        src_brain = os.path.join(src_base, "brain", cid)
        if os.path.isdir(src_brain):
            shutil.copytree(src_brain, os.path.join(dst_base, "brain", cid), dirs_exist_ok=True)

        upsert_ui_visible_summary(dst_base, row, title)
        clean_title = (title or "").strip()
        if clean_title:
            try:
                sconn = sqlite3.connect(summaries_db(dst_base))
                try:
                    dup_cids = {
                        r[0].strip()
                        for r in sconn.execute(
                            "SELECT conversation_id FROM conversation_summaries WHERE title = ? AND conversation_id != ?",
                            (clean_title, cid),
                        )
                        if r and isinstance(r[0], str) and r[0].strip()
                    }
                finally:
                    sconn.close()
                if dup_cids:
                    delete_threads(dup_cids, dst_base, defer_active_ui_files=True)
            except sqlite3.Error:
                pass
            _mirror_to_daemon_ghost_cids(dst_base, dst_db, src_brain, cid, clean_title)
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
