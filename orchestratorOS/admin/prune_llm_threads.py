#!/usr/bin/env python3
"""Prune automated autonomous LLM threads older than a given age in Antigravity CLI."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import io
import logging
import os
import shutil
import sqlite3
import sys
from typing import List, Optional, Set, Tuple

import json

logger = logging.getLogger("prune_llm_threads")

DEFAULT_CLI_BASE = os.path.expanduser("~/.gemini/antigravity-cli")
DEFAULT_PROJECTS = ["autonommptrade", "default-cli-project", "f5f9d01f-01e5-4fac-80f2-97364688afab"]
PROTECTED_PROJECT_NAMES = ["mptrade"]
DEFAULT_RETENTION_DAYS = 1.0  # 1 day default
DEFAULT_MAX_AUTONOMOUS_THREADS = 30  # Keep well below agy's 500-thread global eviction cap

AUTONOMOUS_PROMPT_PATTERNS = [
    "ntfy macro veto alert",
    "macroeconomic and geopolitical risk officer",
    "principal quantitative crypto risk officer",
    "principal crypto quantitative risk strategist",
    "macroeconomic geopolitical risk assessment",
    "macroeconomic risk assessment",
    "macro geopolitical risk assessment",
    "macrogeopolitical risk assessment",
    "macroeconomic impact",
    "macroeconomic war",
    "iran war",
    "middle east war",
    "geopolitical risk",
    "crypto risk assessment",
    "crypto risk evaluation",
    "crypto bot trade risk",
    "crypto trade risk",
    "analiză risc tranzacție",
    "analiza risc tranzactie",
    "evaluated by gemini",
    "automated reasoning",
    "ntfy-macro",
    "ntfy-guard",
    "high-stake",
    "high-value",
    "trade review",
    "trade evaluation",
    "risk evaluation",
    "risk assessment",
    "morning reconciliation",
    "diagnostic audit",
]


def resolve_project_ids(project_names_or_ids: List[str] | Set[str], cli_base: str = DEFAULT_CLI_BASE) -> Set[str]:
    """Resolve project names/IDs into full set of matching IDs and UUIDs from ~/.gemini/config/projects."""
    resolved = set(project_names_or_ids)
    projects_dir = os.path.join(os.path.dirname(cli_base), "config", "projects")
    if not os.path.isdir(projects_dir):
        alt_dir = os.path.expanduser("~/.gemini/config/projects")
        if os.path.isdir(alt_dir):
            projects_dir = alt_dir

    if os.path.isdir(projects_dir):
        for fname in os.listdir(projects_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(projects_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    pid = data.get("id")
                    pname = data.get("name")
                    if any(target in (pid, pname) for target in list(resolved)):
                        if pid:
                            resolved.add(pid)
                        if pname:
                            resolved.add(pname)
                except Exception:
                    pass
    return resolved


def read_varint(stream: io.BytesIO) -> Optional[int]:
    """Read a protobuf varint from stream."""
    res = 0
    shift = 0
    while True:
        b = stream.read(1)
        if not b:
            return None
        val = b[0]
        res |= (val & 0x7F) << shift
        if not (val & 0x80):
            break
        shift += 7
    return res


def encode_varint(val: int) -> bytes:
    """Encode an integer as a protobuf varint."""
    res = bytearray()
    while True:
        b = val & 0x7F
        val >>= 7
        if val:
            res.append(b | 0x80)
        else:
            res.append(b)
            break
    return bytes(res)


def _skip_proto_field(stream: io.BytesIO, wire_type: int) -> Optional[bytes]:
    """Read and return raw bytes of a protobuf field value for the given wire type."""
    if wire_type == 0:
        v = read_varint(stream)
        return encode_varint(v) if v is not None else None
    if wire_type == 1:
        b = stream.read(8)
        return b if len(b) == 8 else None
    if wire_type == 2:
        ln = read_varint(stream)
        if ln is None:
            return None
        b = stream.read(ln)
        return encode_varint(ln) + b
    if wire_type == 5:
        b = stream.read(4)
        return b if len(b) == 4 else None
    return None


def _patch_summary_annotations_title(summary_bytes: bytes, new_title: str) -> bytes:
    """Update field 15 (annotations.title, tag 122) inside a serialized TrajectorySummary."""
    tb = new_title.encode("utf-8")
    title_entry = b"\x0a" + encode_varint(len(tb)) + tb

    nsub = io.BytesIO(summary_bytes)
    new_nested = bytearray()
    found_annotations = False

    while nsub.tell() < len(summary_bytes):
        ntag = read_varint(nsub)
        if ntag is None:
            break
        nwire = ntag & 0x7
        if ntag == 122 and nwire == 2:  # field 15: annotations
            nlen = read_varint(nsub)
            if nlen is None:
                break
            nd = nsub.read(nlen)
            asub = io.BytesIO(nd)
            other_annot_fields = bytearray()
            while asub.tell() < len(nd):
                atag = read_varint(asub)
                if atag is None:
                    break
                awire = atag & 0x7
                val_bytes = _skip_proto_field(asub, awire)
                if val_bytes is None:
                    break
                if atag != 10:  # preserve non-title annotation fields (e.g. last_user_view_time)
                    other_annot_fields.extend(encode_varint(atag))
                    other_annot_fields.extend(val_bytes)
            annot_payload = title_entry + bytes(other_annot_fields)
            new_nested.extend(encode_varint(122))
            new_nested.extend(encode_varint(len(annot_payload)))
            new_nested.extend(annot_payload)
            found_annotations = True
        else:
            val_bytes = _skip_proto_field(nsub, nwire)
            if val_bytes is None:
                break
            new_nested.extend(encode_varint(ntag))
            new_nested.extend(val_bytes)

    if not found_annotations:
        new_nested.extend(encode_varint(122))
        new_nested.extend(encode_varint(len(title_entry)))
        new_nested.extend(title_entry)

    return bytes(new_nested)


def _encode_mtime_entry(cid: str, mtime_sec: int) -> bytes:
    """Encode a tag=18 (file_mod_times map entry) record for agyhub_summaries_proto.pb."""
    cid_b = cid.encode("utf-8")
    inner = b"\x0a" + encode_varint(len(cid_b)) + cid_b + b"\x10" + encode_varint(mtime_sec)
    return b"\x12" + encode_varint(len(inner)) + inner


def ensure_proto_synced(cli_base: str = DEFAULT_CLI_BASE) -> int:
    """Ensure all summary records (tag 10) from jetbox_summaries_proto.pb exist in agyhub_summaries_proto.pb."""
    jb_path = os.path.join(cli_base, "jetbox_summaries_proto.pb")
    agy_path = os.path.join(cli_base, "agyhub_summaries_proto.pb")
    if not os.path.exists(jb_path) or not os.path.exists(agy_path):
        return 0

    def get_summary_records(path: str) -> Tuple[dict[str, bytes], Set[str]]:
        recs: dict[str, bytes] = {}
        mtimes_present: Set[str] = set()
        try:
            with open(path, "rb") as f:
                raw = f.read()
            stream = io.BytesIO(raw)
            while stream.tell() < len(raw):
                tag = read_varint(stream)
                if tag is None:
                    break
                length = read_varint(stream)
                if length is None:
                    break
                data = stream.read(length)
                sub = io.BytesIO(data)
                stag = read_varint(sub)
                cid = None
                if stag == 10:
                    slen = read_varint(sub)
                    if slen is not None:
                        cid = sub.read(slen).decode("utf-8", errors="ignore")
                if cid:
                    if tag == 10:
                        recs[cid] = encode_varint(tag) + encode_varint(length) + data
                    elif tag == 18:
                        mtimes_present.add(cid)
        except Exception:
            pass
        return recs, mtimes_present

    jb_recs, _ = get_summary_records(jb_path)
    agy_recs, agy_mtimes = get_summary_records(agy_path)
    added = 0
    with open(agy_path, "ab") as f:
        for cid, rec in jb_recs.items():
            if cid not in agy_recs:
                f.write(rec)
                added += 1
            if cid not in agy_mtimes:
                db_file = os.path.join(cli_base, "conversations", f"{cid}.db")
                if os.path.exists(db_file):
                    try:
                        mtime_sec = int(os.stat(db_file).st_mtime)
                        f.write(_encode_mtime_entry(cid, mtime_sec))
                    except Exception:
                        pass
    return added


def sync_thread_title(cid: str, new_title: str, cli_base: str = DEFAULT_CLI_BASE) -> bool:
    """Update title for a conversation ID across annotations, SQLite, and protobuf summaries."""
    ensure_proto_synced(cli_base)
    updated = False

    # 1. Update annotations/<cid>.pbtxt
    try:
        annot_dir = os.path.join(cli_base, "annotations")
        if os.path.isdir(annot_dir):
            pbtxt_path = os.path.join(annot_dir, f"{cid}.pbtxt")
            preserved_lines: List[str] = []
            if os.path.exists(pbtxt_path):
                with open(pbtxt_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if not line.lstrip().startswith("title:"):
                            preserved_lines.append(line.rstrip("\n"))
            escaped_title = new_title.replace("\\", "\\\\").replace('"', '\\"')
            out_lines = [f'title:"{escaped_title}"'] + [ln for ln in preserved_lines if ln.strip()]
            with open(pbtxt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(out_lines) + "\n")
            updated = True
    except Exception as e:
        logger.debug("Failed updating annotations pbtxt for %s: %s", cid, e)

    # 2. Update conversation_summaries.db (both title column and raw_summary blob if present)
    try:
        db_path = os.path.join(cli_base, "conversation_summaries.db")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cols = {r[1] for r in cur.execute("PRAGMA table_info(conversation_summaries)").fetchall()}
            if "raw_summary" in cols:
                cur.execute(
                    "SELECT raw_summary FROM conversation_summaries WHERE conversation_id=?",
                    (cid,),
                )
                row = cur.fetchone()
                if row and row[0]:
                    patched_raw = _patch_summary_annotations_title(row[0], new_title)
                    cur.execute(
                        "UPDATE conversation_summaries SET title=?, raw_summary=? WHERE conversation_id=?",
                        (new_title, patched_raw, cid),
                    )
                else:
                    cur.execute(
                        "UPDATE conversation_summaries SET title=? WHERE conversation_id=?",
                        (new_title, cid),
                    )
            else:
                cur.execute(
                    "UPDATE conversation_summaries SET title=? WHERE conversation_id=?",
                    (new_title, cid),
                )
            conn.commit()
            conn.close()
            updated = True
    except Exception as e:
        logger.debug("Failed updating SQLite title: %s", e)

    # 3. Update agyhub_summaries_proto.pb and jetbox_summaries_proto.pb
    db_file = os.path.join(cli_base, "conversations", f"{cid}.db")
    mtime_sec = int(os.stat(db_file).st_mtime) if os.path.exists(db_file) else None

    for pb_name in ["agyhub_summaries_proto.pb", "jetbox_summaries_proto.pb"]:
        pb_path = os.path.join(cli_base, pb_name)
        if not os.path.exists(pb_path):
            continue
        try:
            with open(pb_path, "rb") as f:
                raw = f.read()
            stream = io.BytesIO(raw)
            out = bytearray()
            seen_mtime = False
            while stream.tell() < len(raw):
                tag = read_varint(stream)
                if tag is None:
                    break
                length = read_varint(stream)
                if length is None:
                    break
                data = stream.read(length)
                if tag == 10:
                    sub = io.BytesIO(data)
                    matched_cid = None
                    new_data = bytearray()
                    while sub.tell() < len(data):
                        stag = read_varint(sub)
                        if stag is None:
                            break
                        swire = stag & 0x7
                        fnum = stag >> 3
                        if swire == 2:
                            slen = read_varint(sub)
                            sdata = sub.read(slen)
                            if fnum == 1:
                                matched_cid = sdata.decode("utf-8", errors="ignore")
                                new_data.extend(encode_varint(stag))
                                new_data.extend(encode_varint(slen))
                                new_data.extend(sdata)
                            elif fnum == 2 and matched_cid == cid:
                                patched_nested = _patch_summary_annotations_title(sdata, new_title)
                                new_data.extend(encode_varint(stag))
                                new_data.extend(encode_varint(len(patched_nested)))
                                new_data.extend(patched_nested)
                                updated = True
                            else:
                                new_data.extend(encode_varint(stag))
                                new_data.extend(encode_varint(slen))
                                new_data.extend(sdata)
                        elif swire == 0:
                            val = read_varint(sub)
                            new_data.extend(encode_varint(stag))
                            new_data.extend(encode_varint(val))
                    out.extend(encode_varint(tag))
                    out.extend(encode_varint(len(new_data)))
                    out.extend(new_data)
                elif tag == 18 and mtime_sec is not None:
                    sub = io.BytesIO(data)
                    stag = read_varint(sub)
                    entry_cid = None
                    if stag == 10:
                        slen = read_varint(sub)
                        if slen is not None:
                            entry_cid = sub.read(slen).decode("utf-8", errors="ignore")
                    if entry_cid == cid:
                        out.extend(_encode_mtime_entry(cid, mtime_sec))
                        seen_mtime = True
                    else:
                        out.extend(encode_varint(tag))
                        out.extend(encode_varint(length))
                        out.extend(data)
                else:
                    out.extend(encode_varint(tag))
                    out.extend(encode_varint(length))
                    out.extend(data)
            if pb_name == "agyhub_summaries_proto.pb" and mtime_sec is not None and not seen_mtime:
                out.extend(_encode_mtime_entry(cid, mtime_sec))
            tmp_path = pb_path + ".tmp"
            with open(tmp_path, "wb") as f:
                f.write(out)
            os.replace(tmp_path, pb_path)
        except Exception as e:
            logger.debug("Failed updating proto title in %s: %s", pb_name, e)
    return updated


def filter_proto_file(file_path: str, cids_to_remove: Set[str]) -> int:
    """Filter out entries with matching conversation IDs from a protobuf summaries file."""
    if not os.path.exists(file_path) or not cids_to_remove:
        return 0

    with open(file_path, "rb") as f:
        raw = f.read()

    stream = io.BytesIO(raw)
    out = bytearray()
    removed_count = 0

    while True:
        tag = read_varint(stream)
        if tag is None:
            break
        wire_type = tag & 0x7
        if wire_type != 2:
            break
        length = read_varint(stream)
        if length is None:
            break
        data = stream.read(length)

        sub_stream = io.BytesIO(data)
        sub_tag = read_varint(sub_stream)
        matched_cid: Optional[str] = None
        if sub_tag == 10:  # tag 10 = field 1, string (conversation_id)
            cid_len = read_varint(sub_stream)
            if cid_len is not None:
                matched_cid = sub_stream.read(cid_len).decode("utf-8", errors="ignore")

        if matched_cid and matched_cid in cids_to_remove:
            removed_count += 1
            continue

        out.extend(encode_varint(tag))
        out.extend(encode_varint(length))
        out.extend(data)

    tmp_path = file_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(out)
    os.replace(tmp_path, file_path)
    return removed_count


def parse_db_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse sqlite datetime string into UTC datetime object."""
    if not ts_str:
        return None
    try:
        ts_part = ts_str.split("+")[0].strip()
        if ts_part.startswith("0001-01-01"):
            return datetime(1, 1, 1, tzinfo=timezone.utc)
        if "." in ts_part:
            main_ts, frac = ts_part.split(".", 1)
            frac = frac[:6]
            clean_ts = f"{main_ts}.{frac}+00:00"
        else:
            clean_ts = f"{ts_part}+00:00"
        return datetime.fromisoformat(clean_ts)
    except Exception:
        return None


def is_automated_thread(title: str, preview: str) -> bool:
    """Check if thread content matches automated LLM reasoning tasks."""
    combined = f"{title} {preview}".lower()
    return any(pattern in combined for pattern in AUTONOMOUS_PROMPT_PATTERNS)


def prune_old_cli_threads(
    cli_base: str = DEFAULT_CLI_BASE,
    retention_days: float = DEFAULT_RETENTION_DAYS,
    project_id: Optional[str] = None,
    all_threads_in_project: bool = False,
    exclude_cids: Optional[Set[str]] = None,
    dry_run: bool = False,
    max_threads: Optional[int] = DEFAULT_MAX_AUTONOMOUS_THREADS,
) -> Tuple[int, int]:
    """Find and delete automated LLM threads older than retention_days or exceeding max_threads.

    Returns:
        (pruned_count, total_bytes_freed)
    """
    db_path = os.path.join(cli_base, "conversation_summaries.db")
    if not os.path.exists(db_path):
        logger.info("conversation_summaries.db not found at %s. Nothing to prune.", db_path)
        return (0, 0)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=retention_days)
    protected = set(exclude_cids or [])

    active_env_cid = os.environ.get("CURRENT_CONVERSATION_ID")
    if active_env_cid:
        protected.add(active_env_cid)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    protected_projects = resolve_project_ids(PROTECTED_PROJECT_NAMES, cli_base)
    if project_id and (project_id in protected_projects or project_id in PROTECTED_PROJECT_NAMES):
        logger.warning("Project '%s' is a protected user workspace and cannot be pruned.", project_id)
        conn.close()
        return (0, 0)

    raw_targets = [project_id] if project_id else DEFAULT_PROJECTS
    target_projects = resolve_project_ids(raw_targets, cli_base) - protected_projects

    if not target_projects:
        logger.info("No non-protected target projects found to prune.")
        conn.close()
        return (0, 0)

    placeholders_proj = ",".join("?" for _ in target_projects)
    cols = {r[1] for r in cursor.execute("PRAGMA table_info(conversation_summaries)").fetchall()}
    has_step_count = "step_count" in cols
    select_cols = (
        "conversation_id, title, preview, last_modified_time, project_id, step_count"
        if has_step_count
        else "conversation_id, title, preview, last_modified_time, project_id"
    )

    query = f"SELECT {select_cols} FROM conversation_summaries WHERE project_id IN ({placeholders_proj})"
    cursor.execute(query, list(target_projects))
    rows = cursor.fetchall()

    candidate_cids: Set[str] = set()
    kept_automated: List[Tuple[datetime, str]] = []
    for row in rows:
        cid, title, preview, ts_str, proj = row[:5]
        step_count = row[5] if has_step_count and len(row) > 5 and row[5] is not None else 0
        if cid in protected or proj in protected_projects:
            continue
        matched_auto_pattern = is_automated_thread(title or "", preview or "")
        if not matched_auto_pattern and step_count > 6:
            # Multi-turn interactive conversation: never prune even if --all is passed
            continue
        is_auto = all_threads_in_project or matched_auto_pattern
        if not is_auto:
            continue
        dt = parse_db_timestamp(ts_str)
        if dt is None or dt < cutoff:
            candidate_cids.add(cid)
        else:
            kept_automated.append((dt, cid))

    # Enforce hard count cap so autonomous threads never approach agy's 500-conversation eviction limit
    if max_threads is not None and max_threads >= 0 and len(kept_automated) > max_threads:
        kept_automated.sort(key=lambda item: item[0], reverse=True)
        for _, overflow_cid in kept_automated[max_threads:]:
            candidate_cids.add(overflow_cid)

    if not candidate_cids:
        logger.info("No candidate threads older than %.1f days found in %s.", retention_days, target_projects)
        conn.close()
        return (0, 0)

    logger.info(
        "Found %d candidate automated threads to prune (cutoff: %s UTC, max_threads=%s).",
        len(candidate_cids),
        cutoff.strftime("%Y-%m-%d %H:%M:%S"),
        max_threads,
    )

    total_bytes_freed = 0

    for cid in candidate_cids:
        files_to_check = [
            os.path.join(cli_base, "conversations", f"{cid}.db"),
            os.path.join(cli_base, "conversations", f"{cid}.db-shm"),
            os.path.join(cli_base, "conversations", f"{cid}.db-wal"),
            os.path.join(cli_base, "annotations", f"{cid}.pbtxt"),
            os.path.join(cli_base, "presence", f"{cid}.lock"),
        ]
        for f in files_to_check:
            if os.path.exists(f):
                try:
                    size = os.path.getsize(f)
                    total_bytes_freed += size
                    if not dry_run:
                        os.remove(f)
                except Exception as e:
                    logger.debug("Could not remove file %s: %s", f, e)

        brain_dir = os.path.join(cli_base, "brain", cid)
        if os.path.isdir(brain_dir):
            for root, _, files in os.walk(brain_dir):
                for f in files:
                    try:
                        total_bytes_freed += os.path.getsize(os.path.join(root, f))
                    except Exception:
                        pass
            if not dry_run:
                try:
                    shutil.rmtree(brain_dir)
                except Exception as e:
                    logger.debug("Could not remove brain dir %s: %s", brain_dir, e)

    if dry_run:
        logger.info("[DRY RUN] Would delete %d threads, freeing ~%.2f MB.", len(candidate_cids), total_bytes_freed / (1024 * 1024))
        conn.close()
        return (len(candidate_cids), total_bytes_freed)

    placeholders = ",".join("?" for _ in candidate_cids)
    cursor.execute(f"DELETE FROM conversation_summaries WHERE conversation_id IN ({placeholders})", list(candidate_cids))
    conn.commit()
    conn.close()

    for pb_name in ["agyhub_summaries_proto.pb", "jetbox_summaries_proto.pb"]:
        pb_path = os.path.join(cli_base, pb_name)
        removed = filter_proto_file(pb_path, candidate_cids)
        logger.debug("Filtered %d entries from %s", removed, pb_name)

    logger.info(
        "Successfully pruned %d automated threads older than %.1f days. Freed %.2f MB.",
        len(candidate_cids),
        retention_days,
        total_bytes_freed / (1024 * 1024),
    )
    return (len(candidate_cids), total_bytes_freed)


def clean_orphan_summaries(cli_base: str = DEFAULT_CLI_BASE, dry_run: bool = False) -> int:
    """Find and remove records from conversation_summaries.db that have no physical .db on disk."""
    db_path = os.path.join(cli_base, "conversation_summaries.db")
    conv_dir = os.path.join(cli_base, "conversations")
    if not os.path.exists(db_path) or not os.path.isdir(conv_dir):
        return 0

    active_cid = os.environ.get("CURRENT_CONVERSATION_ID")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT conversation_id, title FROM conversation_summaries")
    rows = cur.fetchall()

    orphans: Set[str] = set()
    for cid, title in rows:
        if active_cid and cid == active_cid:
            continue
        db_file = os.path.join(conv_dir, f"{cid}.db")
        if not os.path.exists(db_file):
            orphans.add(cid)
            logger.info("Found orphan summary: %s (%s)", cid, title)

    if not orphans:
        logger.info("No orphan summaries found.")
        conn.close()
        return 0

    if dry_run:
        logger.info("[DRY RUN] Would remove %d orphan summaries.", len(orphans))
        conn.close()
        return len(orphans)

    placeholders = ",".join("?" for _ in orphans)
    cur.execute(f"DELETE FROM conversation_summaries WHERE conversation_id IN ({placeholders})", list(orphans))
    conn.commit()
    conn.close()

    for pb_name in ["agyhub_summaries_proto.pb", "jetbox_summaries_proto.pb"]:
        pb_path = os.path.join(cli_base, pb_name)
        if os.path.exists(pb_path):
            filter_proto_file(pb_path, orphans)

    for cid in orphans:
        pbtxt = os.path.join(cli_base, "annotations", f"{cid}.pbtxt")
        if os.path.exists(pbtxt):
            try:
                os.remove(pbtxt)
            except Exception:
                pass

    logger.info("Successfully cleaned %d orphan summaries from database.", len(orphans))
    return len(orphans)


def reconcile_summaries_store(cli_base: str = DEFAULT_CLI_BASE) -> int:
    """Reconcile conversation_summaries.db, annotations/*.pbtxt, and protobuf summary caches."""
    conv_dir = os.path.join(cli_base, "conversations")
    annot_dir = os.path.join(cli_base, "annotations")
    backup_dir = os.path.join(cli_base, "backup_migration_mptrade")
    db_path = os.path.join(cli_base, "conversation_summaries.db")
    if not os.path.exists(db_path) or not os.path.isdir(conv_dir):
        return 0

    os.makedirs(annot_dir, exist_ok=True)
    mptrade_uuid = "3ed4aa5d-0873-407a-a8e9-c21a932a565c"
    autonom_uuid = "f5f9d01f-01e5-4fac-80f2-97364688afab"

    # Collect known titles and rows across backups
    backup_dbs = [
        os.path.join(cli_base, "conversation_summaries.db.pre_clean"),
        os.path.join(backup_dir, "conversation_summaries.db"),
        os.path.join(cli_base, "conversation_summaries.db.bak"),
        db_path,
    ]
    known_titles: dict[str, str] = {}
    row_by_cid: dict[str, list] = {}

    for src in backup_dbs:
        if not os.path.exists(src):
            continue
        try:
            c = sqlite3.connect(src)
            for r in c.execute("SELECT * FROM conversation_summaries").fetchall():
                cid = r[0]
                t = (r[1] or "").strip()
                if t and cid not in known_titles:
                    known_titles[cid] = t
                if cid not in row_by_cid or (not row_by_cid[cid][1] and t):
                    row_by_cid[cid] = list(r)
            c.close()
        except Exception:
            pass

    # Also read titles from annotations/*.pbtxt
    for fname in os.listdir(annot_dir):
        if fname.endswith(".pbtxt"):
            cid = fname[:-6]
            if cid not in known_titles:
                try:
                    with open(os.path.join(annot_dir, fname), "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            line_s = line.strip()
                            if line_s.startswith('title:"') and line_s.endswith('"'):
                                t = line_s[7:-1].replace('\\"', '"').replace("\\\\", "\\").strip()
                                if t:
                                    known_titles[cid] = t
                                break
                except Exception:
                    pass

    # Restore any missing mptrade .db files from backup_migration_mptrade/conversations
    backup_conv_dir = os.path.join(backup_dir, "conversations")
    if os.path.isdir(backup_conv_dir):
        for fname in os.listdir(backup_conv_dir):
            if fname.endswith(".db"):
                cid = fname[:-3]
                dst = os.path.join(conv_dir, fname)
                if not os.path.exists(dst) and cid in row_by_cid and row_by_cid[cid][8] == mptrade_uuid:
                    try:
                        shutil.copy2(os.path.join(backup_conv_dir, fname), dst)
                    except Exception:
                        pass

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    jb_out = bytearray()
    agy_out = bytearray()
    reconciled = 0

    for fname in sorted(os.listdir(conv_dir)):
        if not fname.endswith(".db"):
            continue
        cid = fname[:-3]
        db_f = os.path.join(conv_dir, fname)
        row = row_by_cid.get(cid)
        if not row:
            continue

        title = known_titles.get(cid) or (row[1] or "").strip()
        preview = (row[2] or "").strip()
        if not title:
            title = preview.splitlines()[0][:70] if preview else "Conversation"
        row[1] = title

        # Determine if this .db is an automated CLI (-p) invocation (source == 17) vs user session (source == 1)
        db_source = 1
        try:
            dc = sqlite3.connect(db_f)
            srow = dc.execute("SELECT source FROM trajectory_meta LIMIT 1").fetchone()
            if srow and srow[0] is not None:
                db_source = int(srow[0])
            dc.close()
        except Exception:
            pass

        step_cnt = int(row[3] or 0)
        is_auto = (db_source == 17) or (step_cnt <= 6 and is_automated_thread(title, preview) and row[8] == autonom_uuid)
        proj = autonom_uuid if is_auto else mptrade_uuid
        row[8] = proj

        raw = row[19] or b""
        if proj == mptrade_uuid:
            raw = raw.replace(autonom_uuid.encode(), mptrade_uuid.encode())
        else:
            raw = raw.replace(mptrade_uuid.encode(), autonom_uuid.encode())
        raw = _patch_summary_annotations_title(raw, title)
        row[19] = raw

        placeholders = ",".join("?" for _ in range(len(row)))
        cur.execute(f"INSERT OR REPLACE INTO conversation_summaries VALUES ({placeholders})", row)

        # Write annotations/<cid>.pbtxt
        pbtxt_path = os.path.join(annot_dir, f"{cid}.pbtxt")
        preserved: List[str] = []
        if os.path.exists(pbtxt_path):
            try:
                with open(pbtxt_path, "r", encoding="utf-8", errors="ignore") as f:
                    for ln in f:
                        if not ln.lstrip().startswith("title:"):
                            preserved.append(ln.rstrip("\n"))
            except Exception:
                pass
        esc = title.replace("\\", "\\\\").replace('"', '\\"')
        with open(pbtxt_path, "w", encoding="utf-8") as f:
            f.write("\n".join([f'title:"{esc}"'] + [ln for ln in preserved if ln.strip()]) + "\n")

        cid_b = cid.encode("utf-8")
        entry = b"\x0a" + encode_varint(len(cid_b)) + cid_b + b"\x12" + encode_varint(len(raw)) + raw
        rec10 = b"\x0a" + encode_varint(len(entry)) + entry
        jb_out.extend(rec10)
        agy_out.extend(rec10)
        try:
            mtime_sec = int(os.stat(db_f).st_mtime)
            agy_out.extend(_encode_mtime_entry(cid, mtime_sec))
        except Exception:
            pass
        reconciled += 1

    conn.commit()
    try:
        cur.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        pass
    conn.close()

    for pb_name, payload in (("jetbox_summaries_proto.pb", jb_out), ("agyhub_summaries_proto.pb", agy_out)):
        pb_path = os.path.join(cli_base, pb_name)
        tmp_path = pb_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(payload)
        os.replace(tmp_path, pb_path)

    logger.info("Reconciled %d conversation summaries across SQLite, annotations, and protobuf stores.", reconciled)
    return reconciled


# Alias for canonical naming
prune_llm_threads = prune_old_cli_threads


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune old automated LLM threads in Antigravity CLI.")
    parser.add_argument(
        "--reconcile-summaries",
        action="store_true",
        help="Reconcile conversation_summaries.db, annotations, and protobuf caches with conversations/*.db.",
    )
    parser.add_argument(
        "--clean-orphans",
        action="store_true",
        help="Remove entries from conversation_summaries.db that have no physical .db on disk.",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=DEFAULT_RETENTION_DAYS,
        help=f"Retention age in days (default: {DEFAULT_RETENTION_DAYS}).",
    )
    parser.add_argument(
        "--cli-base",
        type=str,
        default=DEFAULT_CLI_BASE,
        help=f"Path to Antigravity CLI directory (default: {DEFAULT_CLI_BASE}).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project ID to prune (default: default-cli-project and autonommptrade).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Prune all threads older than retention in the project, not only automated ones.",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=DEFAULT_MAX_AUTONOMOUS_THREADS,
        help=f"Maximum number of automated threads to retain (default: {DEFAULT_MAX_AUTONOMOUS_THREADS}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate pruning without modifying files or database.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose DEBUG logging.",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.reconcile_summaries:
        reconcile_summaries_store(cli_base=args.cli_base)
        return

    if args.clean_orphans:
        clean_orphan_summaries(cli_base=args.cli_base, dry_run=args.dry_run)
        return

    prune_old_cli_threads(
        cli_base=args.cli_base,
        retention_days=args.days,
        project_id=args.project,
        all_threads_in_project=args.all,
        dry_run=args.dry_run,
        max_threads=args.max_threads,
    )


if __name__ == "__main__":
    main()
