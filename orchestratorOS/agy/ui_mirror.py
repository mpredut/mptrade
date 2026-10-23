"""Mirror permanent topic threads from the isolated agy-bot store into the Web UI store.

Only content files, the summary row and annotations are written. The daemon's in-memory list, its proto
caches and the visible-dir layout are owned by the daemon and agy_store_fix; the mirrored state surfaces
on the next idle `manage_agy.sh ui-sync` restart.
"""
from __future__ import annotations

import logging
import os
import shutil
import sqlite3
from typing import Optional

from orchestratorOS.admin.agy_store_fix import visible_conversations_dir
from orchestratorOS.agy.paths import resolve_bot_cli_base, resolve_ui_cli_base
from orchestratorOS.agy.summary_store import conversation_db, read_summary_row, upsert_ui_visible_summary

logger = logging.getLogger("orchestratorOS.agy.ui_mirror")

UI_TRAJECTORY_SOURCE = 1   # trajectory_meta.source of conversations created in the UI (CLI print mode = 17)


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
    try:
        dst_db = conversation_db(dst_base, cid)
        os.makedirs(os.path.dirname(dst_db), exist_ok=True)
        shutil.copy2(src_db, dst_db + ".tmp")
        os.replace(dst_db + ".tmp", dst_db)
        conn = sqlite3.connect(dst_db)
        try:
            conn.execute("UPDATE trajectory_meta SET source = ?", (UI_TRAJECTORY_SOURCE,))
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()

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
