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
import sys
from typing import Any, Iterable, Optional, Set, Tuple

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from orchestratorOS.agy.protobuf import drop_entries, replace_summary_entry, summary_ui_visible
from botcore import _parse_bool_flag, safe_nonneg_int
from state_io import atomic_write_bytes, read_json_dict

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
    force_b = _as_bool_flag(force, False)
    for ext in ("-wal", "-shm"):
        sidecar = path + ext
        if os.path.exists(sidecar):
            try:
                if not force_b and ext == "-wal" and os.path.getsize(sidecar) > 0:
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
        raw = ts_str.strip()
        if raw.startswith("0001-01-01"):
            return datetime(1, 1, 1, tzinfo=timezone.utc)
        tz_suffix = "+00:00"
        if raw.endswith(("Z", "z")):
            raw = raw[:-1].strip()
        else:
            m_tz = re.search(r"([+-]\d{2}:\d{2})\s*$", raw)
            if m_tz:
                tz_suffix = m_tz.group(1)
                raw = raw[: m_tz.start()].strip()
            elif "+" in raw:
                raw = raw.split("+", 1)[0].strip()
        if "." in raw:
            main_ts, frac = raw.split(".", 1)
            frac_digits = "".join(ch for ch in frac if ch.isdigit())
            raw = f"{main_ts}.{frac_digits[:6].ljust(6, '0')}" if frac_digits else main_ts
        return datetime.fromisoformat(f"{raw}{tz_suffix}").astimezone(timezone.utc)
    except (ValueError, TypeError, AttributeError):
        return None


def resolve_project_ids(names_or_ids: Iterable[str], cli_bases: Iterable[str]) -> Set[str]:
    """Expand project names/ids into every matching name and id found in the stores' config/projects."""
    resolved = {
        n.strip()
        for n in (names_or_ids or ())
        if isinstance(n, str) and n.strip()
    }
    for base in (cli_bases or ()):
        if not isinstance(base, str) or not base.strip():
            continue
        pdir = os.path.join(os.path.dirname(os.path.abspath(base)), "config", "projects")
        if not os.path.isdir(pdir):
            continue
        for fname in os.listdir(pdir):
            if not fname.endswith(".json"):
                continue
            data = read_json_dict(os.path.join(pdir, fname), default=None)
            if not isinstance(data, dict):
                continue
            raw_id, raw_name = data.get("id"), data.get("name")
            pid = raw_id.strip() if isinstance(raw_id, str) else ""
            pname = raw_name.strip() if isinstance(raw_name, str) else ""
            if resolved & {x for x in (pid, pname) if x}:
                resolved.update(x for x in (pid, pname) if x)
    return resolved


_safe_step_count = safe_nonneg_int


def get_thread_step_count(cid: str, cli_base: str) -> Tuple[bool, int, str]:
    """(content file exists, step_count, run status) of a conversation in ``cli_base``."""
    if not cid or not isinstance(cid, str) or not cid.strip():
        return (False, 0, "")
    clean_cid = cid.strip()
    exists = os.path.exists(conversation_db(cli_base, clean_cid))
    if not exists and isinstance(cli_base, str) and os.path.basename(os.path.normpath(cli_base)) == "antigravity-cli":
        from orchestratorOS.agy.paths import visible_conversations_dir
        exists = os.path.exists(os.path.join(visible_conversations_dir(cli_base), f"{clean_cid}.db"))
    row = read_summary_row(cli_base, clean_cid)
    if not row:
        return (exists, 0, "")
    raw_status = row.get("status")
    status_str = raw_status.strip() if isinstance(raw_status, str) else ""
    return (exists, _safe_step_count(row.get("step_count")), status_str)


def read_annotation_title(cli_base: str, cid: str) -> str:
    """Read ``title`` from annotations/<cid>.pbtxt, or '' if absent."""
    from state_io import read_text_file

    txt = read_text_file(
        os.path.join(cli_base, "annotations", f"{cid}.pbtxt"),
        default="",
        errors="ignore",
    )
    m = re.search(r'title:\s*"((?:[^"\\]|\\.)*)"', txt)
    return m.group(1).replace('\\"', '"').replace("\\\\", "\\").strip() if m else ""


def write_annotation_title(cli_base: str, cid: str, title: str) -> None:
    """Set ``title`` in annotations/<cid>.pbtxt, preserving the other annotation lines."""
    from state_io import read_text_lines

    clean_cids = _clean_cid_set([cid])
    if not clean_cids:
        return
    clean_cid = next(iter(clean_cids))
    title_str = ("" if title is None or isinstance(title, bool) else str(title)).replace("\r", " ").replace("\n", " ").strip()
    annot_dir = os.path.join(cli_base, "annotations")
    os.makedirs(annot_dir, exist_ok=True)
    path = os.path.join(annot_dir, f"{clean_cid}.pbtxt")
    preserved = [
        ln
        for ln in read_text_lines(path, skip_empty=True, errors="ignore")
        if not ln.lstrip().startswith("title:")
    ]
    esc = title_str.replace("\\", "\\\\").replace('"', '\\"')
    content = ("\n".join([f'title:"{esc}"'] + preserved) + "\n").encode("utf-8")
    atomic_write_bytes(path, content)


def upsert_summary_row_dict(cli_base: str, row_dict: dict) -> None:
    """Insert or replace a conversation_summaries row matching the table's existing columns."""
    if not isinstance(row_dict, dict):
        return
    conn = sqlite3.connect(summaries_db(cli_base))
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(conversation_summaries)")]
        common = [c for c in cols if c in row_dict]
        if common:
            conn.execute(
                f"INSERT OR REPLACE INTO conversation_summaries ({','.join(common)}) "
                f"VALUES ({','.join('?' for _ in common)})",
                [row_dict[c] for c in common],
            )
            conn.commit()
    finally:
        conn.close()


def upsert_ui_visible_summary(cli_base: str, row: dict, title: str) -> None:
    """Insert/replace a summary row with the CLI source cleared and ``title`` applied (column + proto)."""
    if not isinstance(row, dict):
        return
    clean_cids = _clean_cid_set([row.get("conversation_id")])
    if not clean_cids:
        return
    clean_cid = next(iter(clean_cids))
    title_str = ("" if title is None or isinstance(title, bool) else str(title)).replace("\r", " ").replace("\n", " ").strip()
    row = dict(row, conversation_id=clean_cid, title=title_str, source="", status="CASCADE_RUN_STATUS_IDLE", not_fully_idle=0)
    if row.get("raw_summary"):
        row["raw_summary"] = summary_ui_visible(row["raw_summary"], title_str)
    upsert_summary_row_dict(cli_base, row)
    raw_sum = row.get("raw_summary")
    if raw_sum:
        from state_io import read_bytes_file

        for name in SUMMARIES_PB_FILES:
            pb_path = os.path.join(cli_base, name)
            if os.path.exists(pb_path):
                buf, updated = replace_summary_entry(read_bytes_file(pb_path, default=b""), clean_cid, raw_sum)
                if updated:
                    atomic_write_bytes(pb_path, buf)
    write_annotation_title(cli_base, clean_cid, title_str)


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


def _clean_cid_set(cids: Iterable[Any]) -> Set[str]:
    return {
        c.strip()
        for c in (cids or ())
        if isinstance(c, str)
        and c.strip()
        and c.strip() not in {".", ".."}
        and "/" not in c
        and "\\" not in c
    }


_as_bool_flag = _parse_bool_flag


def _drop_summaries(cli_base: str, cids: Set[str]) -> None:
    clean_cids = _clean_cid_set(cids)
    if not clean_cids:
        return
    db = summaries_db(cli_base)
    if os.path.exists(db):
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                f"DELETE FROM conversation_summaries WHERE conversation_id IN ({','.join('?' for _ in clean_cids)})",
                list(clean_cids),
            )
            conn.commit()
        finally:
            conn.close()
    for name in SUMMARIES_PB_FILES:
        path = os.path.join(cli_base, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                buf, removed = drop_entries(f.read(), clean_cids)
            if removed:
                atomic_write_bytes(path, buf)


def delete_threads(cids: Set[str], cli_base: str, dry_run: bool = False) -> Tuple[int, int]:
    """Delete conversations (content, sidecars, annotations, presence, brain, summaries). Returns (count, bytes)."""
    dry_run = _as_bool_flag(dry_run, False)
    clean_cids = _clean_cid_set(cids)
    if not clean_cids:
        return (0, 0)
    vis_dir = ""
    if isinstance(cli_base, str) and os.path.basename(os.path.normpath(cli_base)) == "antigravity-cli":
        from orchestratorOS.agy.paths import visible_conversations_dir
        vis_dir = visible_conversations_dir(cli_base)
    freed = 0
    seen_inodes: set[Tuple[int, int]] = set()
    for cid in clean_cids:
        db = conversation_db(cli_base, cid)
        files = [db, db + "-shm", db + "-wal",
                 os.path.join(cli_base, "annotations", f"{cid}.pbtxt"),
                 os.path.join(cli_base, "presence", f"{cid}.lock")]
        if vis_dir:
            vis_db = os.path.join(vis_dir, f"{cid}.db")
            files.extend([vis_db, vis_db + "-shm", vis_db + "-wal"])
        for path in files:
            if os.path.exists(path):
                try:
                    st = os.stat(path)
                    ino_key = (st.st_dev, st.st_ino)
                    if ino_key not in seen_inodes:
                        seen_inodes.add(ino_key)
                        freed += st.st_size
                except OSError:
                    pass
                if not dry_run:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        brain = os.path.join(cli_base, "brain", cid)
        if os.path.isdir(brain):
            for root, _, names in os.walk(brain):
                freed += sum(os.path.getsize(os.path.join(root, n)) for n in names)
            if not dry_run:
                shutil.rmtree(brain)
    if not dry_run:
        _drop_summaries(cli_base, clean_cids)
    return (len(clean_cids), freed)


def clean_orphan_summaries(cli_base: str, dry_run: bool = False) -> int:
    """Remove summary rows (and proto entries / annotations) whose conversation file no longer exists."""
    dry_run = _as_bool_flag(dry_run, False)
    db = summaries_db(cli_base)
    if not os.path.exists(db):
        return 0
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("SELECT conversation_id, title FROM conversation_summaries").fetchall()
    finally:
        conn.close()
    vis_dir = ""
    if isinstance(cli_base, str) and os.path.basename(os.path.normpath(cli_base)) == "antigravity-cli":
        from orchestratorOS.agy.paths import visible_conversations_dir
        vis_dir = visible_conversations_dir(cli_base)
    active = (os.environ.get("CURRENT_CONVERSATION_ID") or "").strip()
    orphans = {
        cid.strip()
        for cid, _ in rows
        if isinstance(cid, str)
        and cid.strip()
        and cid.strip() != active
        and not os.path.exists(conversation_db(cli_base, cid.strip()))
        and not (vis_dir and os.path.exists(os.path.join(vis_dir, f"{cid.strip()}.db")))
    }
    for cid, title in rows:
        if isinstance(cid, str) and cid.strip() in orphans:
            logger.info("Found orphan summary: %s (%s)", cid.strip(), title)
    if orphans and not dry_run:
        _drop_summaries(cli_base, orphans)
        for cid in orphans:
            annot = os.path.join(cli_base, "annotations", f"{cid}.pbtxt")
            if os.path.exists(annot):
                os.remove(annot)
    return len(orphans)
