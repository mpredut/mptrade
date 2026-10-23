"""Read/write operations on an Antigravity conversation store (conversations/, annotations/, summaries).

The store layout is owned by the agy daemon/CLI; these helpers only touch what our autonomous flows need:
thread step counts, permanent titles, deletion and orphan cleanup.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
import shutil
import sqlite3
from typing import Iterable, Optional, Set, Tuple

from orchestratorOS.agy.protobuf import drop_entries, summary_ui_visible

logger = logging.getLogger("orchestratorOS.agy.summary_store")

SUMMARIES_DB = "conversation_summaries.db"
SUMMARIES_PB_FILES = ("agyhub_summaries_proto.pb", "jetbox_summaries_proto.pb")


def summaries_db(cli_base: str) -> str:
    return os.path.join(cli_base, SUMMARIES_DB)


def conversation_db(cli_base: str, cid: str) -> str:
    return os.path.join(cli_base, "conversations", f"{cid}.db")


def parse_db_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse a summaries-db datetime string into an aware UTC datetime."""
    if not ts_str:
        return None
    try:
        ts_part = ts_str.split("+")[0].strip()
        if ts_part.startswith("0001-01-01"):
            return datetime(1, 1, 1, tzinfo=timezone.utc)
        if "." in ts_part:
            main_ts, frac = ts_part.split(".", 1)
            ts_part = f"{main_ts}.{frac[:6]}"
        return datetime.fromisoformat(f"{ts_part}+00:00")
    except ValueError:
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
            pid, pname = data.get("id"), data.get("name")
            if resolved & {pid, pname}:
                resolved.update(x for x in (pid, pname) if x)
    return resolved


def get_thread_step_count(cid: str, cli_base: str) -> Tuple[bool, int, str]:
    """(content file exists, step_count, run status) of a conversation in ``cli_base``."""
    if not cid:
        return (False, 0, "")
    exists = os.path.exists(conversation_db(cli_base, cid))
    db = summaries_db(cli_base)
    if not os.path.exists(db):
        return (exists, 0, "")
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT step_count, status FROM conversation_summaries WHERE conversation_id=?", (cid,)
        ).fetchone()
    finally:
        conn.close()
    return (exists, int(row[0] or 0), str(row[1] or "")) if row else (exists, 0, "")


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
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join([f'title:"{esc}"'] + preserved) + "\n")


def upsert_ui_visible_summary(cli_base: str, row: dict, title: str) -> None:
    """Insert/replace a summary row with the CLI source cleared and ``title`` applied (column + proto)."""
    row = dict(row, title=title, source="")
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
                with open(path + ".tmp", "wb") as f:
                    f.write(buf)
                os.replace(path + ".tmp", path)


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
