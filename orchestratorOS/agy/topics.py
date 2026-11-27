"""Permanent LLM topic threads: one Antigravity conversation per topic, compacted in place.

A topic owns a single conversation for its whole life (fixed id and name). Its context is bounded by
Antigravity's `/compact` once LLM_TOPIC_COMPACT_STEPS trajectory steps accumulated since the previous
compaction; the thread is never rotated or deleted.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
import re
import sys
from typing import Any, Dict, Iterable, NamedTuple, Optional, Set

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from botcore import (
    TOPIC_ALIASES,
    TOPIC_SPECS,
    _parse_bool_flag,
    canonical_topic_key,
    exact_pos_int,
    load_dotenv,
    optional_env,
    required_env,
    required_int_env,
    safe_int,
    safe_nonneg_int,
    topic_spec,
)
load_dotenv()
from orchestratorOS.agy.paths import (
    expand_config_path,
    resolve_bot_cli_base,
    resolve_ui_cli_base,
    visible_conversations_dir,
)
from orchestratorOS.agy.summary_store import (
    daemon_loaded_titles,
    delete_threads,
    get_thread_step_count,
    latest_valid_file_mtime,
    read_summary_row,
    upsert_ui_visible_summary,
    write_annotation_title,
)
from orchestratorOS.agy.ui_mirror import (
    mirror_topic_thread_to_ui,
    sync_topic_thread_from_ui,
)
from state_io import extract_json_object
from state_stores.llm_topic_threads import LlmTopicThreadsStore

logger = logging.getLogger("orchestratorOS.agy.topics")

HIGH_STAKE_MODEL = required_env("LLM_HIGH_STAKE_MODEL")
EXECUTOR_MODEL = required_env("LLM_DEFAULT_MODEL")
LOW_STAKE_MODEL = required_env("LLM_LOW_STAKE_MODEL")


def get_topic_registry_path(registry_path: Optional[str] = None) -> str:
    """Registry JSON path: explicit, LLM_TOPIC_REGISTRY_PATH (tests/ops override), or statedb/llm/llm_topic_threads.json."""
    return LlmTopicThreadsStore.resolve_registry_path(
        registry_path,
        env_override=optional_env("LLM_TOPIC_REGISTRY_PATH", ""),
        expand_fn=expand_config_path,
        fallback_root=_ROOT_DIR,
    )


def load_topic_registry(
    registry_path: Optional[str] = None, *, repair_files: bool = False
) -> Dict[str, dict]:
    data = LlmTopicThreadsStore.read_registry(
        get_topic_registry_path(registry_path), repair_files=repair_files
    )
    for old_key, new_key in TOPIC_ALIASES.items():
        if old_key in data:
            old_entry = data.pop(old_key)
            if new_key not in data and isinstance(old_entry, dict):
                old_entry["topic_key"] = new_key
                data[new_key] = old_entry
    return data


def save_topic_registry(registry: Dict[str, dict], registry_path: Optional[str] = None) -> None:
    LlmTopicThreadsStore.save_registry(get_topic_registry_path(registry_path), registry)


def get_active_topic_cids(registry_path: Optional[str] = None) -> Set[str]:
    """Conversation ids owned by topics (protected from pruning)."""
    return {
        e["conversation_id"].strip()
        for e in load_topic_registry(registry_path).values()
        if isinstance(e, dict) and isinstance(e.get("conversation_id"), str) and e["conversation_id"].strip()
    }


def get_topic_canonical_titles(registry_path: Optional[str] = None) -> Dict[str, str]:
    """Map each topic's conversation_id to its canonical TopicSpec title (or registry title)."""
    out: Dict[str, str] = {}
    for key, entry in load_topic_registry(registry_path).items():
        if not isinstance(entry, dict):
            continue
        raw_cid = entry.get("conversation_id")
        if not isinstance(raw_cid, str) or not raw_cid.strip():
            continue
        cid = raw_cid.strip()
        resolved_key = canonical_topic_key(key)
        spec_title = TOPIC_SPECS[resolved_key].title if resolved_key in TOPIC_SPECS else ""
        raw_title = entry.get("title")
        title = spec_title or (raw_title.strip() if isinstance(raw_title, str) else "")
        if title:
            out[cid] = title
    return out


def sync_all_topic_titles(
    cli_base: Optional[str] = None,
    registry_path: Optional[str] = None,
) -> int:
    """Ensure registry, bot store, and UI store carry the canonical TopicSpec title for every registered topic."""
    base = resolve_bot_cli_base(cli_base)
    ui_base = resolve_ui_cli_base()
    reg = load_topic_registry(registry_path)
    updated = 0
    updated_titles: Dict[str, str] = {}
    ui_loaded_titles = daemon_loaded_titles(ui_base) if ui_base and os.path.isdir(ui_base) else {}
    for key, entry in reg.items():
        if not isinstance(entry, dict) or key not in TOPIC_SPECS:
            continue
        canonical = TOPIC_SPECS[key].title
        if entry.get("title") != canonical:
            entry["title"] = canonical
            updated_titles[key] = canonical
        raw_cid = entry.get("conversation_id")
        cid = raw_cid.strip() if isinstance(raw_cid, str) else ""
        if not cid:
            continue
        bot_row = read_summary_row(base, cid)
        ui_row = read_summary_row(ui_base, cid) if ui_base and os.path.isdir(ui_base) else None
        ui_has_db = bool(
            ui_base
            and (
                os.path.exists(os.path.join(ui_base, "conversations", f"{cid}.db"))
                or os.path.exists(os.path.join(visible_conversations_dir(ui_base), f"{cid}.db"))
            )
        )
        missing_ghost_db = bool(
            ui_base
            and any(
                gcid != cid
                and (gtitle or "").strip() == canonical
                and read_summary_row(ui_base, gcid) is None
                and not os.path.exists(os.path.join(ui_base, "conversations", f"{gcid}.db"))
                for gcid, gtitle in ui_loaded_titles.items()
            )
        )
        bot_steps = safe_nonneg_int((bot_row or {}).get("step_count"), 0)
        ui_steps = safe_nonneg_int((ui_row or {}).get("step_count"), 0)
        if ui_row is not None and ui_has_db and ui_steps >= bot_steps and ui_steps > 0 and not missing_ghost_db:
            if bot_row is not None and str(bot_row.get("title") or "").strip() != canonical:
                try:
                    upsert_ui_visible_summary(base, bot_row, canonical)
                except OSError:
                    pass
            if str(ui_row.get("title") or "").strip() != canonical:
                try:
                    upsert_ui_visible_summary(ui_base, ui_row, canonical)
                    updated += 1
                except OSError:
                    pass
            continue
        try:
            bot_row = read_summary_row(base, cid)
            if bot_row is not None:
                upsert_ui_visible_summary(base, bot_row, canonical)
            else:
                write_annotation_title(base, cid, canonical)
        except OSError:
            pass
        mirror_topic_thread_to_ui(cid, canonical, bot_cli_base=base)
        updated += 1
    if updated_titles:
        reg_path = get_topic_registry_path(registry_path)
        with LlmTopicThreadsStore.generation_lock(reg_path):
            fresh_reg = load_topic_registry(registry_path, repair_files=True)
            dirty = False
            for key, canonical in updated_titles.items():
                cur_entry = fresh_reg.get(key)
                if isinstance(cur_entry, dict) and cur_entry.get("title") != canonical:
                    cur_entry["title"] = canonical
                    dirty = True
            if dirty:
                save_topic_registry(fresh_reg, registry_path)
    return updated


def format_topic_turn_header(topic_key: Optional[str], thread_title: Optional[str]) -> Optional[str]:
    """Normalize the per-turn header so every topic thread always starts with its canonical TopicSpec title."""
    raw = thread_title.strip() if isinstance(thread_title, str) else ""
    resolved = canonical_topic_key(topic_key) if topic_key else ""
    if not resolved or resolved not in TOPIC_SPECS:
        return raw or None
    canonical = TOPIC_SPECS[resolved].title
    if not raw or raw == canonical or raw.startswith(canonical):
        return raw or canonical
    cleaned = _NTFY_TOPIC_PREFIX_RE.sub("", raw)
    return f"{canonical} \u2014 {cleaned}" if cleaned else canonical



_NTFY_TOPIC_PREFIX_RE = re.compile(r"^ntfy-(?:guard|macro):\s*")


class TopicSession(NamedTuple):
    """How the next turn of a topic thread must be issued."""
    conversation_id: Optional[str]   # permanent thread to resume; None only before the thread exists
    history_prefix: str              # carried-over decisions, only when the thread must be (re)created
    compact: bool                    # compact the thread in-place with `/compact` after the turn completes


def _load_topic_entry(
    topic_key: str,
    registry_path: Optional[str] = None,
    *,
    repair_files: bool = False,
) -> tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
    resolved = canonical_topic_key(topic_key)
    reg = load_topic_registry(registry_path, repair_files=repair_files)
    entry = reg.get(resolved)
    return resolved, reg, entry if isinstance(entry, dict) else None


def format_topic_history_prefix(
    topic_key: str,
    registry_path: Optional[str] = None,
) -> str:
    """Format the rolling decision memory for a topic as a prompt prefix."""
    _, _, entry = _load_topic_entry(topic_key, registry_path)
    if entry is None:
        return ""
    history = [h for h in entry.get("rolling_history") or [] if isinstance(h, dict)]
    if not history:
        return ""
    lines = ["[TOPIC MEMORY \u2014 PRIOR DECISIONS IN THIS TOPIC]:"]
    lines += [f"- [{h.get('utc', '')}] {h.get('summary', '')}" for h in history]
    return "\n".join(lines) + "\n\n"


def _rotate_topic_conversation(
    base: str,
    entry: Dict[str, Any],
    old_cid: str,
    new_cid: Optional[str],
) -> None:
    clean_cid = old_cid.strip() if isinstance(old_cid, str) else ""
    if clean_cid:
        try:
            delete_threads({clean_cid}, base)
        except Exception:
            pass
        try:
            ui_base = resolve_ui_cli_base()
            if ui_base and os.path.abspath(ui_base) != os.path.abspath(base):
                delete_threads({clean_cid}, ui_base, defer_active_ui_files=True)
        except Exception:
            pass
    entry["conversation_id"] = new_cid
    entry["compacted_at_step"] = 0
    entry["needs_full_prompt"] = True
    entry["pending_rotation"] = False
    entry["consecutive_resume_timeouts"] = 0
    entry["rotation_count"] = safe_nonneg_int(entry.get("rotation_count"), 0) + 1


def record_topic_slow_rotation(
    topic_key: str,
    conversation_id: Optional[str] = None,
    *,
    cli_base: Optional[str] = None,
    registry_path: Optional[str] = None,
    rotate_immediately: bool = False,
) -> int:
    """Mark a topic thread as slow (schedule rotation on next turn or rotate now) and return consecutive slow count."""
    topic_spec(topic_key)
    resolved, base = canonical_topic_key(topic_key), resolve_bot_cli_base(cli_base)
    target_cid = conversation_id.strip() if isinstance(conversation_id, str) and conversation_id.strip() else None
    reg_path = get_topic_registry_path(registry_path)
    with LlmTopicThreadsStore.generation_lock(reg_path):
        _, reg, entry = _load_topic_entry(resolved, registry_path, repair_files=True)
        if entry is None:
            entry = {"topic_key": resolved, "turn_count": 0, "rolling_history": []}
        raw_cid = entry.get("conversation_id")
        cur_cid = raw_cid.strip() if isinstance(raw_cid, str) and raw_cid.strip() else None
        count = safe_nonneg_int(entry.get("consecutive_slow_rotations"), 0) + 1
        if _parse_bool_flag(rotate_immediately, False) and cur_cid and (target_cid is None or cur_cid == target_cid):
            _rotate_topic_conversation(base, entry, cur_cid, None)
        else:
            entry["pending_rotation"] = True
        entry["consecutive_slow_rotations"] = count
        entry["updated_ts"] = datetime.now(timezone.utc).timestamp()
        reg[resolved] = entry
        save_topic_registry(reg, registry_path)
    return count


def get_topic_slow_rotation_count(
    topic_key: str,
    *,
    registry_path: Optional[str] = None,
) -> int:
    """Return current ``consecutive_slow_rotations`` count for ``topic_key``."""
    resolved = canonical_topic_key(topic_key)
    _, _, entry = _load_topic_entry(resolved, registry_path)
    if entry is None:
        return 0
    return safe_nonneg_int(entry.get("consecutive_slow_rotations"), 0)


def reset_topic_slow_rotations(
    topic_keys: Optional[Iterable[str]] = None,
    *,
    registry_path: Optional[str] = None,
) -> int:
    """Reset ``consecutive_slow_rotations`` to 0 for given topics (or all topics when ``topic_keys`` is None)."""
    reg_path = get_topic_registry_path(registry_path)
    target_set = (
        {canonical_topic_key(k) for k in topic_keys if isinstance(k, str) and k.strip()}
        if topic_keys is not None
        else None
    )
    reset_count = 0
    with LlmTopicThreadsStore.generation_lock(reg_path):
        reg = load_topic_registry(registry_path, repair_files=True)
        changed = False
        for k, entry in reg.items():
            if not isinstance(entry, dict):
                continue
            if target_set is not None and k not in target_set:
                continue
            if safe_nonneg_int(entry.get("consecutive_slow_rotations"), 0) > 0:
                entry["consecutive_slow_rotations"] = 0
                changed = True
                reset_count += 1
        if changed:
            save_topic_registry(reg, registry_path)
    return reset_count


def record_topic_resume_timeout(
    topic_key: str,
    conversation_id: Optional[str] = None,
    *,
    cli_base: Optional[str] = None,
    registry_path: Optional[str] = None,
    max_consecutive_timeouts: int = 2,
) -> bool:
    """Increment consecutive resume-timeout counter and rotate the topic thread once the threshold is reached.

    Returns True when the topic conversation was rotated to a fresh thread (preserving rolling memory).
    """
    topic_spec(topic_key)
    resolved, base = canonical_topic_key(topic_key), resolve_bot_cli_base(cli_base)
    max_to = exact_pos_int(max_consecutive_timeouts, default=2) or 2
    target_cid = conversation_id.strip() if isinstance(conversation_id, str) and conversation_id.strip() else None
    reg_path = get_topic_registry_path(registry_path)
    with LlmTopicThreadsStore.generation_lock(reg_path):
        _, reg, entry = _load_topic_entry(resolved, registry_path, repair_files=True)
        if entry is None:
            return False
        raw_cid = entry.get("conversation_id")
        cur_cid = raw_cid.strip() if isinstance(raw_cid, str) and raw_cid.strip() else None
        if not cur_cid or (target_cid is not None and cur_cid != target_cid):
            return False
        count = safe_nonneg_int(entry.get("consecutive_resume_timeouts"), 0) + 1
        rotated = False
        if count >= max_to:
            logger.warning(
                "Topic '%s' conversation %s hit %d consecutive resume timeouts (limit=%d); rotating to a fresh thread.",
                resolved,
                cur_cid,
                count,
                max_to,
            )
            _rotate_topic_conversation(base, entry, cur_cid, None)
            rotated = True
        else:
            entry["consecutive_resume_timeouts"] = count
        entry["updated_ts"] = datetime.now(timezone.utc).timestamp()
        reg[resolved] = entry
        save_topic_registry(reg, registry_path)
    return rotated


def prepare_topic_session(
    topic_key: str,
    cli_base: Optional[str] = None,
    compact_steps: Optional[int] = None,
    registry_path: Optional[str] = None,
    max_steps: Optional[int] = None,
) -> TopicSession:
    """Resolve the permanent conversation of a topic and whether this turn must compact or rotate it."""
    topic_spec(topic_key)
    resolved, base = canonical_topic_key(topic_key), resolve_bot_cli_base(cli_base)
    if compact_steps is None:
        compact_steps = required_int_env("LLM_TOPIC_COMPACT_STEPS")
    compact_steps_i = exact_pos_int(compact_steps, default=None)
    if compact_steps_i is None:
        raise ValueError(f"LLM_TOPIC_COMPACT_STEPS must be > 0, got {compact_steps}")
    raw_max = max_steps if max_steps is not None else optional_env("LLM_TOPIC_MAX_STEPS", "")
    max_steps_i = safe_nonneg_int(raw_max, 0)

    _, reg, entry = _load_topic_entry(resolved, registry_path)
    if entry is None:
        return TopicSession(None, "", False)

    raw_cid = entry.get("conversation_id")
    cid = raw_cid.strip() if isinstance(raw_cid, str) else ""
    if cid:
        pending_rot = _parse_bool_flag(entry.get("pending_rotation"), False)
        sync_topic_thread_from_ui(cid, bot_cli_base=base)
        exists, step_count, _ = get_thread_step_count(cid, base)
        if exists and not pending_rot and (max_steps_i <= 0 or step_count < max_steps_i):
            compacted_at = safe_nonneg_int(entry.get("compacted_at_step"), 0)
            if compacted_at > step_count:
                compacted_at = 0
            hist_prefix = (
                format_topic_history_prefix(resolved, registry_path)
                if _parse_bool_flag(entry.get("needs_history_sync"), False)
                else ""
            )
            return TopicSession(cid, hist_prefix, (step_count - compacted_at) >= compact_steps_i)
        reg_path = get_topic_registry_path(registry_path)
        with LlmTopicThreadsStore.generation_lock(reg_path):
            _, reg, fresh_entry = _load_topic_entry(
                resolved, registry_path, repair_files=True
            )
            eff_entry = fresh_entry if fresh_entry is not None else entry
            if exists:
                if pending_rot:
                    logger.info(
                        "Topic '%s' conversation %s had pending rotation from prior slow turn; rotating to a fresh thread with rolling memory.",
                        resolved,
                        cid,
                    )
                else:
                    logger.info(
                        "Topic '%s' conversation %s reached %d steps (max %d); rotating to a fresh thread with rolling memory.",
                        resolved,
                        cid,
                        step_count,
                        max_steps_i,
                    )
                _rotate_topic_conversation(base, eff_entry, cid, None)
            else:
                # Runtime resilience: the permanent thread vanished from the store; recreate it once with memory.
                logger.warning("Topic '%s' conversation %s no longer exists; recreating the topic thread.", resolved, cid)
                eff_entry["conversation_id"] = None
                eff_entry["compacted_at_step"] = 0
                eff_entry["needs_full_prompt"] = True
                eff_entry["pending_rotation"] = False
            reg[resolved] = eff_entry
            save_topic_registry(reg, registry_path)

    return TopicSession(None, format_topic_history_prefix(resolved, registry_path), False)


def is_topic_resume_warm(
    topic_key: str,
    *,
    cli_base: Optional[str] = None,
    registry_path: Optional[str] = None,
    max_steps: Optional[int] = None,
    conversation_id: Optional[str] = None,
) -> bool:
    """Return True when the topic's active uncompacted thread already holds the full policy prompt.

    Returns False on fresh, rotated, recreated, or just-compacted threads so the next turn always
    re-establishes the complete Static-First policy and rules before compact resume turns begin.
    """
    try:
        topic_spec(topic_key)
    except Exception:
        return False
    resolved = canonical_topic_key(topic_key)
    try:
        base = resolve_bot_cli_base(cli_base)
    except Exception:
        return False
    _, _, entry = _load_topic_entry(resolved, registry_path)
    if entry is None:
        return False
    raw_cid = entry.get("conversation_id")
    cid = raw_cid.strip() if isinstance(raw_cid, str) else ""
    if not cid:
        return False
    if conversation_id is not None and cid != str(conversation_id).strip():
        return False
    if _parse_bool_flag(entry.get("needs_full_prompt"), False) or _parse_bool_flag(entry.get("pending_rotation"), False):
        return False
    if safe_nonneg_int(entry.get("turn_count"), 0) <= 0:
        return False
    exists, step_count, _ = get_thread_step_count(cid, base)
    if not exists or step_count <= 0:
        return False
    raw_max = max_steps if max_steps is not None else optional_env("LLM_TOPIC_MAX_STEPS", "")
    max_steps_i = safe_nonneg_int(raw_max, 0)
    if max_steps_i > 0 and step_count >= max_steps_i:
        return False
    compacted_at = safe_nonneg_int(entry.get("compacted_at_step"), 0)
    return step_count > compacted_at


def get_topic_thread_state(
    topic_key: str,
    *,
    cli_base: Optional[str] = None,
    registry_path: Optional[str] = None,
    max_steps: Optional[int] = None,
) -> tuple[str, int, bool]:
    """Read-only topic thread state returning ``(conversation_id, compacted_at_step, is_warm)``."""
    try:
        topic_spec(topic_key)
    except Exception:
        return "", 0, False
    resolved = canonical_topic_key(topic_key)
    _, _, entry = _load_topic_entry(resolved, registry_path)
    if entry is None:
        return "", 0, False
    raw_cid = entry.get("conversation_id")
    cid = raw_cid.strip() if isinstance(raw_cid, str) else ""
    compacted_at = safe_nonneg_int(entry.get("compacted_at_step"), 0)
    is_warm = (
        is_topic_resume_warm(
            resolved,
            cli_base=cli_base,
            registry_path=registry_path,
            max_steps=max_steps,
            conversation_id=cid,
        )
        if cid
        else False
    )
    return cid, compacted_at, is_warm


def get_topic_turn_count(
    topic_key: str,
    *,
    registry_path: Optional[str] = None,
) -> int:
    """Read-only cumulative turn count recorded for ``topic_key`` in the topic registry."""
    try:
        topic_spec(topic_key)
    except Exception:
        return 0
    resolved = canonical_topic_key(topic_key)
    _, _, entry = _load_topic_entry(resolved, registry_path)
    return safe_nonneg_int(entry.get("turn_count"), 0) if isinstance(entry, dict) else 0


def has_pending_ui_turn(
    topic_key: str,
    *,
    cli_base: Optional[str] = None,
    ui_cli_base: Optional[str] = None,
    registry_path: Optional[str] = None,
) -> bool:
    """Read-only check whether the Web UI store has a newer user turn on this topic's thread."""
    try:
        topic_spec(topic_key)
    except Exception:
        return False
    resolved = canonical_topic_key(topic_key)
    _, _, entry = _load_topic_entry(resolved, registry_path)
    if entry is None:
        return False
    raw_cid = entry.get("conversation_id")
    cid = raw_cid.strip() if isinstance(raw_cid, str) else ""
    if not cid:
        return False
    recorded_steps = safe_nonneg_int(entry.get("step_count"), 0)
    try:
        bot_base = resolve_bot_cli_base(cli_base)
    except Exception:
        return False
    try:
        ui_base = resolve_ui_cli_base(ui_cli_base)
    except Exception:
        ui_base = ""
    _, bot_steps, _ = get_thread_step_count(cid, bot_base)
    if recorded_steps > 0 and bot_steps > recorded_steps:
        return True
    if not ui_base or os.path.abspath(ui_base) == os.path.abspath(bot_base):
        return False
    ui_exists, ui_steps, ui_status = get_thread_step_count(cid, ui_base)
    if ui_status == "CASCADE_RUN_STATUS_RUNNING":
        return True
    if ui_steps > max(bot_steps, recorded_steps):
        return True
    if ui_exists and ui_steps >= bot_steps > 0:
        ui_db = os.path.join(ui_base, "conversations", f"{cid}.db")
        if not os.path.exists(ui_db):
            vis_db = os.path.join(visible_conversations_dir(ui_base), f"{cid}.db")
            ui_db = vis_db if os.path.exists(vis_db) else ""
        bot_db = os.path.join(bot_base, "conversations", f"{cid}.db")
        if ui_db and os.path.exists(bot_db):
            ui_mtime = latest_valid_file_mtime((ui_db, ui_db + "-wal"))
            bot_mtime = latest_valid_file_mtime((bot_db, bot_db + "-wal"))
            if ui_mtime > 0.0 and bot_mtime > 0.0 and ui_mtime > bot_mtime + 2.0:
                return True
    return False


_SHORT_SUMMARY_KEYS = (
    "decision",
    "threat_level",
    "market_bias",
    "risk_level",
    "recommended_action",
    "recommended_brake",
    "suggested_scale",
    "confidence",
)
_PROSE_SUMMARY_KEYS = ("summary", "reason")


def condense_decision(response_text: Any, max_len: int = 360) -> str:
    """Single-line summary of an LLM JSON decision (falls back to the whitespace-collapsed text)."""
    if not isinstance(response_text, str):
        return ""
    condensed = " ".join(response_text.split())
    parsed = extract_json_object(response_text)
    if isinstance(parsed, dict):
        parts = [f"{k}={parsed[k]}" for k in _SHORT_SUMMARY_KEYS if parsed.get(k) not in (None, "")]
        mtds = parsed.get("market_trading_decisions")
        mtd = parsed.get("market_trading_decision")
        raw_trades = mtds if isinstance(mtds, list) and mtds else ([mtd] if isinstance(mtd, dict) else [])
        t_items = [
            (
                f"{d.get('decision', 'HOLD')}(LIMIT@{d.get('limit_price')}):{d.get('symbol', '')}@{d.get('confidence', 0)}"
                if str(d.get("order_type") or "").upper() == "LIMIT" and d.get("limit_price") is not None
                else f"{d.get('decision', 'HOLD')}:{d.get('symbol', '')}@{d.get('confidence', 0)}"
            )
            for d in raw_trades
            if isinstance(d, dict)
        ]
        if t_items:
            parts.append(f"trade={','.join(t_items)}")
        parts.extend(f"{k}={parsed[k]}" for k in _PROSE_SUMMARY_KEYS if parsed.get(k) not in (None, ""))
        if parts:
            condensed = "; ".join(parts)
    max_len = max(4, safe_int(max_len, 360))
    return condensed if len(condensed) <= max_len else condensed[: max_len - 3] + "..."


def record_topic_turn(
    topic_key: str,
    conversation_id: Optional[str],
    response_text: str,
    compacted: bool = False,
    cli_base: Optional[str] = None,
    max_history_entries: Optional[int] = None,
    registry_path: Optional[str] = None,
    replace_conversation: bool = False,
    schedule_rotation: bool = False,
    count_slow_rotation: bool = False,
    mark_needs_full_prompt: bool = False,
) -> None:
    """Update the topic registry after a completed turn and mirror the thread into the Web UI store.

    The topic's conversation id is adopted once and replaced only when rotated or explicitly replaced
    via ``replace_conversation=True``; ``compacted`` marks a turn that compacted the thread in-place
    with `/compact` so the next compaction is scheduled from the current step count.
    """
    spec = topic_spec(topic_key)
    resolved, base = canonical_topic_key(topic_key), resolve_bot_cli_base(cli_base)
    compacted = _parse_bool_flag(compacted, default=False)
    replace_conv = _parse_bool_flag(replace_conversation, default=False)
    max_history_entries = safe_nonneg_int(max_history_entries, default=None)
    if max_history_entries is None:
        max_history_entries = required_int_env("LLM_TOPIC_HISTORY_MAX_ENTRIES")

    condensed = condense_decision(response_text)
    reg_path = get_topic_registry_path(registry_path)
    with LlmTopicThreadsStore.generation_lock(reg_path):
        _, reg, entry = _load_topic_entry(resolved, registry_path, repair_files=True)
        if entry is None:
            entry = {"topic_key": resolved, "turn_count": 0, "rolling_history": []}

        clean_cid = conversation_id.strip() if isinstance(conversation_id, str) and conversation_id.strip() else None
        raw_existing = entry.get("conversation_id")
        existing_cid = raw_existing.strip() if isinstance(raw_existing, str) and raw_existing.strip() else None
        if clean_cid and existing_cid and clean_cid != existing_cid and (
            replace_conv or (not get_thread_step_count(existing_cid, base)[0] and get_thread_step_count(clean_cid, base)[0])
        ):
            _rotate_topic_conversation(base, entry, existing_cid, clean_cid)
            existing_cid = clean_cid
        elif not existing_cid and clean_cid:
            entry["conversation_id"] = clean_cid
            entry["compacted_at_step"] = 0
            existing_cid = clean_cid
        active_cid = existing_cid
        if active_cid:
            _, step_count, _ = get_thread_step_count(active_cid, base)
            entry["step_count"] = step_count
            if active_cid == clean_cid:
                entry["needs_history_sync"] = False
                if compacted:
                    entry["compacted_at_step"] = step_count
                    entry["compaction_count"] = safe_nonneg_int(entry.get("compaction_count"), 0) + 1
                    entry["needs_full_prompt"] = True
                else:
                    entry["needs_full_prompt"] = False
            elif _parse_bool_flag(mark_needs_full_prompt, False):
                entry["needs_full_prompt"] = True
                entry["needs_history_sync"] = True
        entry["topic_key"] = resolved
        entry["consecutive_resume_timeouts"] = 0
        if _parse_bool_flag(schedule_rotation, False):
            entry["pending_rotation"] = True
            entry["consecutive_slow_rotations"] = safe_nonneg_int(entry.get("consecutive_slow_rotations"), 0) + 1
        elif _parse_bool_flag(count_slow_rotation, False):
            entry["pending_rotation"] = False
            entry["consecutive_slow_rotations"] = safe_nonneg_int(entry.get("consecutive_slow_rotations"), 0) + 1
        else:
            entry["pending_rotation"] = False
            entry["consecutive_slow_rotations"] = 0
        entry["turn_count"] = safe_nonneg_int(entry.get("turn_count"), 0) + 1
        entry["updated_ts"] = datetime.now(timezone.utc).timestamp()
        entry["title"] = spec.title

        history = [h for h in entry.get("rolling_history") or [] if isinstance(h, dict)]
        if condensed:
            history.append({"utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "summary": condensed})
        if max_history_entries > 0:
            if len(history) > max_history_entries:
                action_quota = max(1, max_history_entries // 3)
                action_idxs = [
                    idx for idx, item in enumerate(history)
                    if any(
                        tok in str(item.get("summary") or "")
                        for tok in ("BUY(", "BUY:", "SELL(", "SELL:", "decision=BUY", "decision=SELL")
                    )
                ]
                pinned = set(action_idxs[-action_quota:])
                recent_quota = max(0, max_history_entries - len(pinned))
                for idx in range(len(history) - 1, -1, -1):
                    if len(pinned) >= max_history_entries or recent_quota <= 0:
                        break
                    if idx not in pinned:
                        pinned.add(idx)
                        recent_quota -= 1
                history = [history[idx] for idx in sorted(pinned)]
            entry["rolling_history"] = history
        else:
            entry["rolling_history"] = []
        reg[resolved] = entry
        save_topic_registry(reg, registry_path)
    if active_cid and (clean_cid is not None or not _parse_bool_flag(mark_needs_full_prompt, False)):
        try:
            bot_row = read_summary_row(base, active_cid)
            if bot_row is not None:
                upsert_ui_visible_summary(base, bot_row, spec.title)
            else:
                write_annotation_title(base, active_cid, spec.title)
        except OSError:
            pass
        mirror_topic_thread_to_ui(active_cid, spec.title, bot_cli_base=base)
