"""Read/write operations on an Antigravity conversation store (conversations/, annotations/, summaries).

The store layout is owned by the agy daemon/CLI; these helpers only touch what our autonomous flows need:
thread step counts, permanent titles, deletion and orphan cleanup.
"""
from __future__ import annotations

from datetime import datetime, timezone
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

from orchestratorOS.agy.paths import visible_conversations_dir
from orchestratorOS.agy.protobuf import (
    drop_entries,
    entry_cid,
    list_entry_cids,
    parse_fields,
    replace_summary_entry,
    summary_annotation_title,
    summary_ui_visible,
)
from botcore import _parse_bool_flag, optional_env, safe_nonneg_int
from state_io import (
    atomic_write_bytes,
    read_bytes_file,
    read_json_dict,
    read_text_file,
    read_text_lines,
)

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
    force_b = _parse_bool_flag(force, False)
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


def _visible_dir_if_cli_store(cli_base: str) -> str:
    if isinstance(cli_base, str) and os.path.basename(os.path.normpath(cli_base)) == "antigravity-cli":
        return visible_conversations_dir(cli_base)
    return ""


def get_thread_step_count(cid: str, cli_base: str) -> Tuple[bool, int, str]:
    """(content file exists, step_count, run status) of a conversation in ``cli_base``."""
    if not cid or not isinstance(cid, str) or not cid.strip():
        return (False, 0, "")
    clean_cid = cid.strip()
    exists = os.path.exists(conversation_db(cli_base, clean_cid))
    if not exists:
        vis_dir = _visible_dir_if_cli_store(cli_base)
        if vis_dir:
            exists = os.path.exists(os.path.join(vis_dir, f"{clean_cid}.db"))
    row = read_summary_row(cli_base, clean_cid)
    if not row:
        return (exists, 0, "")
    raw_status = row.get("status")
    status_str = raw_status.strip() if isinstance(raw_status, str) else ""
    return (exists, safe_nonneg_int(row.get("step_count")), status_str)


def read_annotation_title(cli_base: str, cid: str) -> str:
    """Read ``title`` from annotations/<cid>.pbtxt, or '' if absent."""
    txt = read_text_file(
        os.path.join(cli_base, "annotations", f"{cid}.pbtxt"),
        default="",
        errors="ignore",
    )
    m = re.search(r'title:\s*"((?:[^"\\]|\\.)*)"', txt)
    return m.group(1).replace('\\"', '"').replace("\\\\", "\\").strip() if m else ""


def _clean_cid_and_title(cid: Any, title: Any) -> Optional[tuple[str, str]]:
    clean_cids = _clean_cid_set([cid])
    if not clean_cids:
        return None
    title_str = ("" if title is None or isinstance(title, bool) else str(title)).replace("\r", " ").replace("\n", " ").strip()
    return next(iter(clean_cids)), title_str


def write_annotation_title(cli_base: str, cid: str, title: str) -> None:
    """Set ``title`` in annotations/<cid>.pbtxt, preserving the other annotation lines."""
    if (cleaned := _clean_cid_and_title(cid, title)) is None:
        return
    clean_cid, title_str = cleaned
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


def _replace_pb_summary_entries(cli_base: str, cid: str, raw_summary: Any) -> None:
    if not raw_summary:
        return
    for name in SUMMARIES_PB_FILES:
        pb_path = os.path.join(cli_base, name)
        if os.path.exists(pb_path):
            buf, updated = replace_summary_entry(read_bytes_file(pb_path, default=b""), cid, raw_summary)
            if updated:
                atomic_write_bytes(pb_path, buf)


def upsert_ui_visible_summary(cli_base: str, row: dict, title: str) -> None:
    """Insert/replace a summary row with the CLI source cleared and ``title`` applied (column + proto)."""
    if not isinstance(row, dict) or (cleaned := _clean_cid_and_title(row.get("conversation_id"), title)) is None:
        return
    clean_cid, title_str = cleaned
    row = dict(row, conversation_id=clean_cid, title=title_str, source="", status="CASCADE_RUN_STATUS_IDLE", not_fully_idle=0)
    if row.get("raw_summary"):
        row["raw_summary"] = summary_ui_visible(row["raw_summary"], title_str)
    upsert_summary_row_dict(cli_base, row)
    _replace_pb_summary_entries(cli_base, clean_cid, row.get("raw_summary"))
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


def iter_sqlite_user_table_columns(conn) -> list[tuple[str, list[str], str]]:
    """Return ``[(table_name, cols, col_sql), ...]`` for all non-system SQLite tables."""
    out: list[tuple[str, list[str], str]] = []
    tables = [
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if isinstance(r[0], str) and r[0] and not r[0].startswith("sqlite_")
    ]
    for tbl in tables:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info(`{tbl}`)").fetchall()]
        if cols:
            out.append((tbl, cols, ",".join(f"`{c}`" for c in cols)))
    return out


def snapshot_conversation_state(cli_base: str, cid: str) -> Optional[dict]:
    """Capture a lightweight in-memory rollback checkpoint of ``<cid>.db`` and its summary row."""
    if not cli_base or not isinstance(cli_base, str) or not cid or not isinstance(cid, str):
        return None
    clean_cid = cid.strip()
    db_path = conversation_db(cli_base, clean_cid)
    if not os.path.exists(db_path):
        return None
    tables_state: dict[str, dict[str, Any]] = {}
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        try:
            for tbl, cols, _ in iter_sqlite_user_table_columns(conn):
                if "idx" in cols:
                    max_idx = conn.execute(f"SELECT MAX(idx) FROM `{tbl}`").fetchone()[0]
                    tail_row = (
                        conn.execute(f"SELECT * FROM `{tbl}` WHERE idx=?", (max_idx,)).fetchone()
                        if max_idx is not None
                        else None
                    )
                    tables_state[tbl] = {
                        "mode": "idx",
                        "cols": cols,
                        "max_idx": max_idx,
                        "tail_row": tuple(tail_row) if tail_row is not None else None,
                    }
                else:
                    rows = [tuple(r) for r in conn.execute(f"SELECT * FROM `{tbl}`").fetchall()]
                    tables_state[tbl] = {
                        "mode": "full",
                        "cols": cols,
                        "rows": rows,
                    }
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.debug("snapshot_conversation_state failed for %s: %s", db_path, exc)
        return None
    return {
        "cid": clean_cid,
        "tables": tables_state,
        "summary_row": read_summary_row(cli_base, clean_cid),
    }


def restore_conversation_state(cli_base: str, cid: str, snapshot: Optional[dict]) -> bool:
    """Roll back ``<cid>.db`` and its summary row to a checkpoint captured by ``snapshot_conversation_state``."""
    if not isinstance(snapshot, dict) or not cli_base or not isinstance(cli_base, str) or not cid or not isinstance(cid, str):
        return False
    if not os.path.exists(db_path := conversation_db(cli_base, (clean_cid := cid.strip()))):
        return False
    tables_state = snapshot.get("tables")
    if not isinstance(tables_state, dict) or not tables_state:
        return False
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        try:
            for tbl, tstate in tables_state.items():
                if not isinstance(tstate, dict):
                    continue
                cols = tstate.get("cols") or []
                if not cols:
                    continue
                col_sql = ",".join(f"`{c}`" for c in cols)
                val_sql = ",".join("?" for _ in cols)
                if tstate.get("mode") == "idx":
                    max_idx = tstate.get("max_idx")
                    if max_idx is None:
                        conn.execute(f"DELETE FROM `{tbl}`")
                    else:
                        conn.execute(f"DELETE FROM `{tbl}` WHERE idx > ?", (max_idx,))
                        tail_row = tstate.get("tail_row")
                        if tail_row is not None:
                            conn.execute(
                                f"INSERT OR REPLACE INTO `{tbl}` ({col_sql}) VALUES ({val_sql})",
                                tail_row,
                            )
                else:
                    conn.execute(f"DELETE FROM `{tbl}`")
                    for row in tstate.get("rows") or ():
                        conn.execute(
                            f"INSERT INTO `{tbl}` ({col_sql}) VALUES ({val_sql})",
                            row,
                        )
            conn.commit()
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("restore_conversation_state failed for %s: %s", db_path, exc)
        return False
    summary_row = snapshot.get("summary_row")
    if isinstance(summary_row, dict):
        try:
            upsert_summary_row_dict(cli_base, summary_row)
            _replace_pb_summary_entries(cli_base, clean_cid, summary_row.get("raw_summary"))
        except (sqlite3.Error, OSError):
            pass
    return True


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


def daemon_loaded_titles(cli_base: str) -> dict[str, str]:
    """Map conversation IDs held in memory by the Web UI daemon to their persisted annotation titles."""
    out: dict[str, str] = {}
    if not cli_base or not isinstance(cli_base, str):
        return out
    data = read_bytes_file(os.path.join(cli_base, SUMMARIES_PB_FILES[0]), default=None)
    if not data:
        return out
    try:
        fields = parse_fields(data)
    except ValueError:
        return out
    for tag, wt, val in fields:
        if tag != 1 or wt != 2:
            continue
        cid = entry_cid(val).strip()
        if not cid:
            continue
        title = ""
        try:
            sub = dict((t, v) for t, _, v in parse_fields(val))
            raw_sum = sub.get(2, b"")
            if isinstance(raw_sum, (bytes, bytearray)):
                title = summary_annotation_title(bytes(raw_sum)).strip()
        except ValueError:
            title = ""
        if not title:
            title = read_annotation_title(cli_base, cid)
        out[cid] = title
    return out


def _drop_summaries(
    cli_base: str,
    cids: Set[str],
    *,
    keep_proto_cids: Optional[Set[str]] = None,
) -> None:
    clean_cids = _clean_cid_set(cids)
    if not clean_cids:
        return
    db = summaries_db(cli_base)
    if os.path.exists(db):
        try:
            conn = sqlite3.connect(db, timeout=5)
            try:
                conn.execute(
                    f"DELETE FROM conversation_summaries WHERE conversation_id IN ({','.join('?' for _ in clean_cids)})",
                    list(clean_cids),
                )
                conn.commit()
            finally:
                conn.close()
        except sqlite3.Error:
            pass
    proto_drop = clean_cids - _clean_cid_set(keep_proto_cids or ())
    if not proto_drop:
        return
    for name in SUMMARIES_PB_FILES:
        path = os.path.join(cli_base, name)
        if os.path.exists(path):
            buf, removed = drop_entries(read_bytes_file(path, default=b""), proto_drop)
            if removed:
                atomic_write_bytes(path, buf)


def delete_threads(
    cids: Set[str],
    cli_base: str,
    dry_run: bool = False,
    *,
    defer_active_ui_files: bool = False,
) -> Tuple[int, int]:
    """Delete conversations (content, sidecars, annotations, presence, brain, summaries). Returns (count, bytes).

    When ``defer_active_ui_files=True``, conversations still held in memory by the running Web UI daemon
    (present in ``agyhub_summaries_proto.pb``) have their ``conversation_summaries.db`` row removed so
    ``pending_ui_sync()`` queues ``gone:<cid>``, while their on-disk ``.db`` and ``brain/`` files stay
    readable in the Web UI until ``agy_store_fix.py pre`` reaps them cleanly on the next daemon restart.
    """
    dry_run = _parse_bool_flag(dry_run, False)
    defer_ui = _parse_bool_flag(defer_active_ui_files, False)
    clean_cids = _clean_cid_set(cids)
    if not clean_cids:
        return (0, 0)
    loaded_cids: Set[str] = set()
    if defer_ui and cli_base and isinstance(cli_base, str):
        if pb_data := read_bytes_file(os.path.join(cli_base, SUMMARIES_PB_FILES[0]), default=None):
            try:
                loaded_cids = list_entry_cids(pb_data)
            except ValueError:
                loaded_cids = set()
    deferred_cids = (clean_cids & loaded_cids) if defer_ui else set()
    immediate_cids = clean_cids - deferred_cids
    vis_dir = _visible_dir_if_cli_store(cli_base)
    freed = 0
    seen_inodes: set[Tuple[int, int]] = set()
    for cid in immediate_cids:
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
                for n in names:
                    try:
                        freed += os.path.getsize(os.path.join(root, n))
                    except OSError:
                        pass
            if not dry_run:
                shutil.rmtree(brain, ignore_errors=True)
    if not dry_run:
        _drop_summaries(cli_base, clean_cids, keep_proto_cids=deferred_cids)
    return (len(clean_cids), freed)


def clean_orphan_summaries(cli_base: str, dry_run: bool = False) -> int:
    """Remove summary rows (and proto entries / annotations) whose conversation file no longer exists."""
    dry_run = _parse_bool_flag(dry_run, False)
    db = summaries_db(cli_base)
    if not os.path.exists(db):
        return 0
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("SELECT conversation_id, title FROM conversation_summaries").fetchall()
    finally:
        conn.close()
    vis_dir = _visible_dir_if_cli_store(cli_base)
    active = optional_env("CURRENT_CONVERSATION_ID", "")
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
