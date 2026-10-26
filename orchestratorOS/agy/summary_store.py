"""Read/write operations on an Antigravity conversation store (conversations/, annotations/, summaries).

The store layout is owned by the agy daemon/CLI; these helpers only touch what our autonomous flows need:
thread step counts, permanent titles, deletion and orphan cleanup.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
import re
import shutil
import sqlite3
from typing import Iterable, Optional, Set, Tuple

from orchestratorOS.agy.protobuf import drop_entries, replace_summary_entry, summary_ui_visible
from state_io import atomic_write_bytes

logger = logging.getLogger("orchestratorOS.agy.summary_store")

SUMMARIES_DB = "conversation_summaries.db"
SUMMARIES_PB_FILES = ("agyhub_summaries_proto.pb", "jetbox_summaries_proto.pb")


def summaries_db(cli_base: str) -> str:
    return os.path.join(cli_base, SUMMARIES_DB)


def conversation_db(cli_base: str, cid: str) -> str:
    return os.path.join(cli_base, "conversations", f"{cid}.db")


def checkpoint_db(path: str) -> bool:
    """Flush SQLite WAL into the main database file. Returns False if it could not be done safely."""
    wal = path + "-wal"
    try:
        if not os.path.exists(wal) or os.path.getsize(wal) == 0:
            return True
        conn = sqlite3.connect(path, timeout=15)
        try:
            row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            return bool(row) and row[0] == 0
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.warning("checkpoint failed for %s: %s", path, e)
        return False


def drop_wal_sidecars(path: str, *, force: bool = False) -> None:
    """Remove -wal and -shm sidecar files next to ``path`` (refuses non-empty -wal unless ``force=True``)."""
    for ext in ("-wal", "-shm"):
        sidecar = path + ext
        if os.path.exists(sidecar):
            try:
                if not force and ext == "-wal" and os.path.getsize(sidecar) > 0:
                    logger.warning("refusing to drop non-empty %s", sidecar)
                    continue
                os.remove(sidecar)
            except OSError as e:
                logger.warning("cannot remove %s: %s", sidecar, e)


def parse_db_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse a summaries-db datetime string into an aware UTC datetime."""
    if not ts_str or not isinstance(ts_str, str):
        return None
    try:
        ts_part = ts_str.split("+")[0].rstrip("Zz").strip()
        if ts_part.startswith("0001-01-01"):
            return datetime(1, 1, 1, tzinfo=timezone.utc)
        if "." in ts_part:
            main_ts, frac = ts_part.split(".", 1)
            frac_digits = "".join(ch for ch in frac if ch.isdigit())
            ts_part = f"{main_ts}.{frac_digits[:6].ljust(6, '0')}" if frac_digits else main_ts
        return datetime.fromisoformat(f"{ts_part}+00:00")
    except (ValueError, TypeError, AttributeError):
        return None


def resolve_project_ids(names_or_ids: Iterable[str], cli_bases: Iterable[str]) -> Set[str]:
    """Expand project names/ids into every matching name and id found in the stores' config/projects."""
    resolved = {n for n in names_or_ids if n}
    for base in cli_bases:
        pdir = os.path.join(os.path.dirname(os.path.abspath(base)), "config", "projects")
        if not os.path.isdir(pdir):
            continue
        for fname in os.listdir(pdir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(pdir, fname), encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            pid, pname = data.get("id"), data.get("name")
            if resolved & {pid, pname}:
                resolved.update(x for x in (pid, pname) if x)
    return resolved


def get_thread_step_count(cid: str, cli_base: str) -> Tuple[bool, int, str]:
    """(content file exists, step_count, run status) of a conversation in ``cli_base``."""
    if not cid:
        return (False, 0, "")
    exists = os.path.exists(conversation_db(cli_base, cid))
    row = read_summary_row(cli_base, cid)
    return (exists, int(row.get("step_count") or 0), str(row.get("status") or "")) if row else (exists, 0, "")


def read_annotation_title(cli_base: str, cid: str) -> str:
    """Read ``title`` from annotations/<cid>.pbtxt, or '' if absent."""
    try:
        with open(os.path.join(cli_base, "annotations", f"{cid}.pbtxt"), encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except OSError:
        return ""
    m = re.search(r'title:\s*"((?:[^"\\]|\\.)*)"', txt)
    return m.group(1).replace('\\"', '"').replace("\\\\", "\\").strip() if m else ""


def write_annotation_title(cli_base: str, cid: str, title: str) -> None:
    """Set ``title`` in annotations/<cid>.pbtxt, preserving the other annotation lines."""
    annot_dir = os.path.join(cli_base, "annotations")
    os.makedirs(annot_dir, exist_ok=True)
    path = os.path.join(annot_dir, f"{cid}.pbtxt")
    preserved = []
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="ignore") as f:
            preserved = [ln.rstrip("\n") for ln in f if ln.strip() and not ln.lstrip().startswith("title:")]
    esc = title.replace("\\", "\\\\").replace('"', '\\"')
    content = ("\n".join([f'title:"{esc}"'] + preserved) + "\n").encode("utf-8")
    atomic_write_bytes(path, content)


def upsert_ui_visible_summary(cli_base: str, row: dict, title: str) -> None:
    """Insert/replace a summary row with the CLI source cleared and ``title`` applied (column + proto)."""
    row = dict(row, title=title, source="", status="CASCADE_RUN_STATUS_IDLE", not_fully_idle=0)
    if row.get("raw_summary"):
        row["raw_summary"] = summary_ui_visible(row["raw_summary"], title)
    conn = sqlite3.connect(summaries_db(cli_base))
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(conversation_summaries)")]
        common = [c for c in cols if c in row]
        conn.execute(
            f"INSERT OR REPLACE INTO conversation_summaries ({','.join(common)}) "
            f"VALUES ({','.join('?' for _ in common)})",
            [row[c] for c in common],
        )
        conn.commit()
    finally:
        conn.close()
    raw_sum = row.get("raw_summary")
    if raw_sum:
        for name in SUMMARIES_PB_FILES:
            pb_path = os.path.join(cli_base, name)
            if os.path.exists(pb_path):
                with open(pb_path, "rb") as f:
                    buf, updated = replace_summary_entry(f.read(), row["conversation_id"], raw_sum)
                if updated:
                    atomic_write_bytes(pb_path, buf)
    write_annotation_title(cli_base, row["conversation_id"], title)


def read_summary_row(cli_base: str, cid: str) -> Optional[dict]:
    db = summaries_db(cli_base)
    if not os.path.exists(db):
        return None
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM conversation_summaries WHERE conversation_id=?", (cid,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def _drop_summaries(cli_base: str, cids: Set[str]) -> None:
    if not cids:
        return
    db = summaries_db(cli_base)
    if os.path.exists(db):
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                f"DELETE FROM conversation_summaries WHERE conversation_id IN ({','.join('?' for _ in cids)})",
                list(cids),
            )
            conn.commit()
        finally:
            conn.close()
    for name in SUMMARIES_PB_FILES:
        path = os.path.join(cli_base, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                buf, removed = drop_entries(f.read(), cids)
            if removed:
                atomic_write_bytes(path, buf)


def delete_threads(cids: Set[str], cli_base: str, dry_run: bool = False) -> Tuple[int, int]:
    """Delete conversations (content, sidecars, annotations, presence, brain, summaries). Returns (count, bytes)."""
    if not cids:
        return (0, 0)
    freed = 0
    for cid in cids:
        db = conversation_db(cli_base, cid)
        files = [db, db + "-shm", db + "-wal",
                 os.path.join(cli_base, "annotations", f"{cid}.pbtxt"),
                 os.path.join(cli_base, "presence", f"{cid}.lock")]
        for path in files:
            if os.path.exists(path):
                freed += os.path.getsize(path)
                if not dry_run:
                    os.remove(path)
        brain = os.path.join(cli_base, "brain", cid)
        if os.path.isdir(brain):
            for root, _, names in os.walk(brain):
                freed += sum(os.path.getsize(os.path.join(root, n)) for n in names)
            if not dry_run:
                shutil.rmtree(brain)
    if not dry_run:
        _drop_summaries(cli_base, set(cids))
    return (len(cids), freed)


def clean_orphan_summaries(cli_base: str, dry_run: bool = False) -> int:
    """Remove summary rows (and proto entries / annotations) whose conversation file no longer exists."""
    db = summaries_db(cli_base)
    if not os.path.exists(db):
        return 0
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("SELECT conversation_id, title FROM conversation_summaries").fetchall()
    finally:
        conn.close()
    active = os.environ.get("CURRENT_CONVERSATION_ID")
    orphans = {cid for cid, _ in rows if cid != active and not os.path.exists(conversation_db(cli_base, cid))}
    for cid, title in rows:
        if cid in orphans:
            logger.info("Found orphan summary: %s (%s)", cid, title)
    if orphans and not dry_run:
        _drop_summaries(cli_base, orphans)
        for cid in orphans:
            annot = os.path.join(cli_base, "annotations", f"{cid}.pbtxt")
            if os.path.exists(annot):
                os.remove(annot)
    return len(orphans)
