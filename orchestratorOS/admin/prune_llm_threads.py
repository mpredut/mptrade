#!/usr/bin/env python3
"""Prune automated LLM threads from the isolated agy-bot Antigravity store.

Topic threads (orchestratorOS.agy.topics) are permanent and always protected; everything else the
autonomous callers left behind (one-off answers, legacy threads) is pruned by age and by a hard count cap.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import logging
import math
import os
import sqlite3
import sys
from typing import List, Optional, Set, Tuple

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from botcore import _parse_bool_flag, load_dotenv, required_env, required_float_env, required_int_env  # noqa: E402

load_dotenv(os.path.join(_ROOT_DIR, "config.env"))

from orchestratorOS.agy.paths import resolve_bot_cli_base, resolve_ui_cli_base  # noqa: E402
from orchestratorOS.agy.summary_store import (  # noqa: E402
    clean_orphan_summaries,
    delete_threads,
    parse_db_timestamp,
    resolve_project_ids,
)
from orchestratorOS.agy.topics import TOPIC_SPECS, get_active_topic_cids, sync_all_topic_titles  # noqa: E402

logger = logging.getLogger("prune_llm_threads")

AGY_DEFAULT_PROJECT = "default-cli-project"   # project agy assigns to `-p` calls without --project
PROTECTED_PROJECT_NAMES = ["mptrade"]         # interactive user workspace: never pruned
INTERACTIVE_MIN_STEPS = 6                     # more steps than this without an automated marker = human session
DEFAULT_RETENTION_DAYS = required_float_env("CLI_THREAD_RETENTION_DAYS")
DEFAULT_MAX_AUTONOMOUS_THREADS = required_int_env("LLM_MAX_AUTONOMOUS_THREADS")

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
    "macroeconomic shock",
    "macro shock",
    "shock assessment",
    "shock shield",
    "iran war",
    "middle east war",
    "geopolitical risk",
    "geopolitical macro",
    "crypto risk assessment",
    "crypto risk evaluation",
    "crypto bot trade risk",
    "crypto trade risk",
    "crypto macro",
    "crypto quantitative macro",
    "macro risk",
    "re-entry risk",
    "re-entry evaluation",
    "post-stop re-entry",
    "reconciliation audit",
    "morning reconciliation",
    "diagnostic audit",
    "acknowledge confirmation request",
    "acknowledge instruction test",
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
    "autonomous trade decisions",
    "lead sre and quantitative trading systems architect",
    "auto_macro_news_shield",
    *[spec.title.lower() for spec in TOPIC_SPECS.values()],
    *[f"[{key}]" for key in TOPIC_SPECS],
]


def is_automated_thread(title: str, preview: str) -> bool:
    """Check if thread content matches automated LLM reasoning tasks."""
    t_str = title if isinstance(title, str) else ""
    p_str = preview if isinstance(preview, str) else ""
    combined = f"{t_str} {p_str}".lower()
    return any(pattern in combined for pattern in AUTONOMOUS_PROMPT_PATTERNS)


_as_bool = _parse_bool_flag


def prune_old_cli_threads(
    cli_base: Optional[str] = None,
    retention_days: float = DEFAULT_RETENTION_DAYS,
    project_id: Optional[str] = None,
    all_threads_in_project: bool = False,
    exclude_cids: Optional[Set[str]] = None,
    dry_run: bool = False,
    max_threads: Optional[int] = DEFAULT_MAX_AUTONOMOUS_THREADS,
) -> Tuple[int, int]:
    """Delete automated LLM threads older than retention_days or beyond max_threads. Returns (count, bytes)."""
    dry_run = _as_bool(dry_run, False)
    all_threads_in_project = _as_bool(all_threads_in_project, False)
    base = resolve_bot_cli_base(cli_base)
    if not dry_run and cli_base is None:
        try:
            sync_all_topic_titles(cli_base=base)
        except Exception as e:
            logger.debug("Topic title sync skipped: %s", e)
    db_path = os.path.join(base, "conversation_summaries.db")
    if not os.path.exists(db_path):
        logger.info("conversation_summaries.db not found at %s. Nothing to prune.", db_path)
        return (0, 0)

    project_bases = [base, resolve_ui_cli_base()]
    if isinstance(exclude_cids, str):
        excl_set = {c.strip() for c in exclude_cids.split(",") if c.strip()}
    elif isinstance(exclude_cids, (set, list, tuple, frozenset)):
        excl_set = {str(c).strip() for c in exclude_cids if isinstance(c, str) and c.strip()}
    else:
        excl_set = set()
    protected = excl_set | get_active_topic_cids()
    active_env_cid = (os.environ.get("CURRENT_CONVERSATION_ID") or "").strip()
    if active_env_cid:
        protected.add(active_env_cid)

    protected_projects = resolve_project_ids(PROTECTED_PROJECT_NAMES, project_bases)
    norm_project_id = (
        project_id.strip()
        if isinstance(project_id, str) and project_id.strip()
        else None
    )
    if norm_project_id:
        resolved_req = resolve_project_ids([norm_project_id], project_bases)
        if norm_project_id in protected_projects or bool(resolved_req & protected_projects):
            logger.warning("Project '%s' is a protected user workspace and cannot be pruned.", norm_project_id)
            return (0, 0)
    autonomous_projects = resolve_project_ids([required_env("AGY_PROJECT")], project_bases)
    targets = (resolve_project_ids([norm_project_id], project_bases) if norm_project_id
               else autonomous_projects | {AGY_DEFAULT_PROJECT}) - protected_projects
    if not targets:
        logger.info("No non-protected target projects found to prune.")
        return (0, 0)

    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT * FROM conversation_summaries WHERE project_id IN ({','.join('?' for _ in targets)})",
            list(targets),
        ).fetchall()
    finally:
        conn.close()

    if retention_days is None or isinstance(retention_days, bool):
        retention_days = DEFAULT_RETENTION_DAYS
    else:
        try:
            rd_val = float(retention_days)
            retention_days = rd_val if math.isfinite(rd_val) and rd_val >= 0.0 else DEFAULT_RETENTION_DAYS
        except (TypeError, ValueError, OverflowError):
            retention_days = DEFAULT_RETENTION_DAYS

    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    candidates: Set[str] = set()
    kept: List[Tuple[datetime, str]] = []
    for row in rows:
        cid, proj = row["conversation_id"], row["project_id"]
        if cid in protected:
            continue
        raw_steps = row["step_count"] if "step_count" in row.keys() else 0
        try:
            if raw_steps is not None and not isinstance(raw_steps, bool):
                sf = float(raw_steps)
                steps = int(sf) if math.isfinite(sf) and sf >= 0 else 0
            else:
                steps = 0
        except (TypeError, ValueError, OverflowError):
            steps = 0
        automated = is_automated_thread(row["title"] or "", row["preview"] or "")
        if not automated and steps > INTERACTIVE_MIN_STEPS:
            continue   # multi-turn interactive conversation: never pruned, even with --all
        short_autonomous = proj in autonomous_projects and steps <= INTERACTIVE_MIN_STEPS
        if not (all_threads_in_project or automated or short_autonomous):
            continue
        dt = parse_db_timestamp(row["last_modified_time"])
        if dt is None or dt < cutoff:
            candidates.add(cid)
        else:
            kept.append((dt, cid))

    # Hard count cap so autonomous threads never approach agy's conversation eviction limit
    max_threads_i: Optional[int] = None
    if max_threads is not None:
        if isinstance(max_threads, bool):
            max_threads_i = DEFAULT_MAX_AUTONOMOUS_THREADS
        else:
            try:
                mt_f = float(max_threads)
                max_threads_i = int(mt_f) if math.isfinite(mt_f) and int(mt_f) >= 0 else DEFAULT_MAX_AUTONOMOUS_THREADS
            except (TypeError, ValueError, OverflowError):
                max_threads_i = DEFAULT_MAX_AUTONOMOUS_THREADS
    if max_threads_i is not None and 0 <= max_threads_i < len(kept):
        kept.sort(reverse=True)
        candidates.update(cid for _, cid in kept[max_threads_i:])
    if not candidates:
        logger.info("No candidate threads older than %.1f days found in %s.", retention_days, sorted(targets))
        return (0, 0)

    count, freed = delete_threads(candidates, base, dry_run=dry_run)
    logger.info("%s %d automated threads (cutoff %s UTC, max_threads=%s), %.2f MB.",
                "[DRY RUN] Would prune" if dry_run else "Pruned", count,
                cutoff.strftime("%Y-%m-%d %H:%M:%S"), max_threads, freed / (1024 * 1024))
    return (count, freed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune automated LLM threads from the agy-bot Antigravity store.")
    parser.add_argument("--clean-orphans", action="store_true",
                        help="Remove summary rows whose conversation file no longer exists.")
    parser.add_argument("--days", type=float, default=DEFAULT_RETENTION_DAYS,
                        help=f"Retention age in days (default: {DEFAULT_RETENTION_DAYS}).")
    parser.add_argument("--cli-base", type=str, default=None,
                        help="Antigravity store to prune (default: AGY_BOT_HOME store).")
    parser.add_argument("--project", type=str, default=None,
                        help="Project name/ID to prune (default: AGY_PROJECT and agy's default project).")
    parser.add_argument("--all", action="store_true",
                        help="Prune all short threads older than retention in the project, not only automated ones.")
    parser.add_argument("--max-threads", type=int, default=DEFAULT_MAX_AUTONOMOUS_THREADS,
                        help=f"Maximum automated threads to retain (default: {DEFAULT_MAX_AUTONOMOUS_THREADS}).")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without modifying files or database.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose DEBUG logging.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    if args.clean_orphans:
        n = clean_orphan_summaries(resolve_bot_cli_base(args.cli_base), dry_run=args.dry_run)
        logger.info("%s %d orphan summaries.", "[DRY RUN] Would remove" if args.dry_run else "Removed", n)
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
