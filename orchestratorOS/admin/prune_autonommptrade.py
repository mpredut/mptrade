#!/usr/bin/env python3
"""Prune automated autonomous LLM threads (autonommptrade) older than a given age in Antigravity CLI."""
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

logger = logging.getLogger("prune_autonommptrade")

DEFAULT_CLI_BASE = os.path.expanduser("~/.gemini/antigravity-cli")
DEFAULT_PROJECTS = ["default-cli-project", "autonommptrade"]
DEFAULT_RETENTION_DAYS = 1.0  # 1 day default

AUTONOMOUS_PROMPT_PATTERNS = [
    "macroeconomic and geopolitical risk officer",
    "principal quantitative crypto risk officer",
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
    "autonommptrade",
    "ntfy-macro",
]


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
) -> Tuple[int, int]:
    """Find and delete automated LLM threads older than retention_days.

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

    target_projects = [project_id] if project_id else DEFAULT_PROJECTS
    placeholders_proj = ",".join("?" for _ in target_projects)

    query = (
        "SELECT conversation_id, title, preview, last_modified_time, project_id "
        f"FROM conversation_summaries WHERE project_id IN ({placeholders_proj})"
    )
    cursor.execute(query, target_projects)
    rows = cursor.fetchall()

    candidate_cids: Set[str] = set()
    for cid, title, preview, ts_str, proj in rows:
        if cid in protected:
            continue
        dt = parse_db_timestamp(ts_str)
        if dt is None or dt < cutoff:
            # If in autonommptrade project, prune by age automatically; if default project, filter by automated pattern
            if proj == "autonommptrade" or all_threads_in_project or is_automated_thread(title, preview):
                candidate_cids.add(cid)

    if not candidate_cids:
        logger.info("No candidate threads older than %.1f days found in %s.", retention_days, target_projects)
        conn.close()
        return (0, 0)

    logger.info(
        "Found %d candidate automated threads older than %.1f days (cutoff: %s UTC).",
        len(candidate_cids),
        retention_days,
        cutoff.strftime("%Y-%m-%d %H:%M:%S"),
    )

    total_bytes_freed = 0

    for cid in candidate_cids:
        files_to_check = [
            os.path.join(cli_base, "conversations", f"{cid}.db"),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune old automated LLM threads in Antigravity CLI.")
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

    prune_old_cli_threads(
        cli_base=args.cli_base,
        retention_days=args.days,
        project_id=args.project,
        all_threads_in_project=args.all,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
