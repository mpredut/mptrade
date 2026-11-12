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
from typing import Any, Dict, NamedTuple, Optional, Set

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
    required_int_env,
    safe_int,
    safe_nonneg_int,
    topic_spec,
)
from cachepaths import cache_path, resolve_cachedb_filepath

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
    read_summary_row,
    upsert_ui_visible_summary,
    write_annotation_title,
)
from orchestratorOS.agy.ui_mirror import (
    mirror_topic_thread_to_ui,
    sync_topic_thread_from_ui,
)
from state_io import extract_json_object, read_json_dict, save_json_state

logger = logging.getLogger("orchestratorOS.agy.topics")

HIGH_STAKE_MODEL = optional_env("LLM_HIGH_STAKE_MODEL", "gemini-4-argon-high")
EXECUTOR_MODEL = optional_env("LLM_DEFAULT_MODEL", "gemini-4-argon-medium")
LOW_STAKE_MODEL = optional_env("LLM_LOW_STAKE_MODEL", "gpt-oss-120b-medium")


def get_topic_registry_path(registry_path: Optional[str] = None) -> str:
    """Registry JSON path: explicit, LLM_TOPIC_REGISTRY_PATH (tests/ops override), or cachedb/llm_topic_threads.json."""
    raw = registry_path or optional_env("LLM_TOPIC_REGISTRY_PATH", "")
    if raw:
        return str(resolve_cachedb_filepath(expand_config_path(raw)))
    try:
        return cache_path("llm_topic_threads.json")
    except Exception:
        return os.path.join(_ROOT_DIR, "cachedb", "llm_topic_threads.json")


def load_topic_registry(registry_path: Optional[str] = None) -> Dict[str, dict]:
    data = dict(read_json_dict(get_topic_registry_path(registry_path), default={}, recover_previous=True))
    for old_key, new_key in TOPIC_ALIASES.items():
        if old_key in data:
            old_entry = data.pop(old_key)
            if new_key not in data and isinstance(old_entry, dict):
                old_entry["topic_key"] = new_key
                data[new_key] = old_entry
    return data


def save_topic_registry(registry: Dict[str, dict], registry_path: Optional[str] = None) -> None:
    path = get_topic_registry_path(registry_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_json_state(path, registry, indent=2)


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


def _apply_and_mirror_topic_title(base: str, cid: str, title: str) -> None:
    try:
        bot_row = read_summary_row(base, cid)
        if bot_row is not None:
            upsert_ui_visible_summary(base, bot_row, title)
        else:
            write_annotation_title(base, cid, title)
    except OSError:
        pass
    mirror_topic_thread_to_ui(cid, title, bot_cli_base=base)


def sync_all_topic_titles(
    cli_base: Optional[str] = None,
    registry_path: Optional[str] = None,
) -> int:
    """Ensure registry, bot store, and UI store carry the canonical TopicSpec title for every registered topic."""
    base = resolve_bot_cli_base(cli_base)
    ui_base = resolve_ui_cli_base()
    reg = load_topic_registry(registry_path)
    updated = 0
    changed_reg = False
    ui_loaded_titles = daemon_loaded_titles(ui_base) if ui_base and os.path.isdir(ui_base) else {}
    for key, entry in reg.items():
        if not isinstance(entry, dict) or key not in TOPIC_SPECS:
            continue
        canonical = TOPIC_SPECS[key].title
        if entry.get("title") != canonical:
            entry["title"] = canonical
            changed_reg = True
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
        _apply_and_mirror_topic_title(base, cid, canonical)
        updated += 1
    if changed_reg:
        save_topic_registry(reg, registry_path)
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
    cleaned = re.sub(r"^ntfy-(?:guard|macro):\s*", "", raw)
    return f"{canonical} — {cleaned}" if cleaned else canonical



class TopicSession(NamedTuple):
    """How the next turn of a topic thread must be issued."""
    conversation_id: Optional[str]   # permanent thread to resume; None only before the thread exists
    history_prefix: str              # carried-over decisions, only when the thread must be (re)created
    compact: bool                    # compact the thread in-place with `/compact` after the turn completes


def _load_topic_entry(
    topic_key: str,
    registry_path: Optional[str] = None,
) -> tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
    resolved = canonical_topic_key(topic_key)
    reg = load_topic_registry(registry_path)
    entry = reg.get(resolved)
    return resolved, reg, entry if isinstance(entry, dict) else None


def _resolve_topic_context(
    topic_key: str,
    cli_base: Optional[str] = None,
) -> tuple[Dict[str, str], str, str]:
    spec = topic_spec(topic_key)
    return spec, canonical_topic_key(topic_key), resolve_bot_cli_base(cli_base)


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
    lines = ["[TOPIC MEMORY — PRIOR DECISIONS IN THIS TOPIC]:"]
    lines += [f"- [{h.get('utc', '')}] {h.get('summary', '')}" for h in history]
    return "\n".join(lines) + "\n\n"


def _delete_topic_thread_everywhere(base: str, cid: str) -> None:
    """Remove a retired topic conversation from both the isolated bot store and the Web UI store."""
    if not cid or not isinstance(cid, str):
        return
    clean_cid = cid.strip()
    if not clean_cid:
        return
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


def _rotate_topic_conversation(
    base: str,
    entry: Dict[str, Any],
    old_cid: str,
    new_cid: Optional[str],
) -> None:
    _delete_topic_thread_everywhere(base, old_cid)
    entry["conversation_id"] = new_cid
    entry["compacted_at_step"] = 0
    entry["rotation_count"] = safe_nonneg_int(entry.get("rotation_count"), 0) + 1


def prepare_topic_session(
    topic_key: str,
    cli_base: Optional[str] = None,
    compact_steps: Optional[int] = None,
    registry_path: Optional[str] = None,
    max_steps: Optional[int] = None,
) -> TopicSession:
    """Resolve the permanent conversation of a topic and whether this turn must compact or rotate it."""
    _, resolved, base = _resolve_topic_context(topic_key, cli_base)
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
        sync_topic_thread_from_ui(cid, bot_cli_base=base)
        exists, step_count, _ = get_thread_step_count(cid, base)
        if exists:
            if max_steps_i > 0 and step_count >= max_steps_i:
                logger.info(
                    "Topic '%s' conversation %s reached %d steps (max %d); rotating to a fresh thread with rolling memory.",
                    resolved,
                    cid,
                    step_count,
                    max_steps_i,
                )
                _rotate_topic_conversation(base, entry, cid, None)
                reg[resolved] = entry
                save_topic_registry(reg, registry_path)
                return TopicSession(None, format_topic_history_prefix(resolved, registry_path), False)
            compacted_at = safe_nonneg_int(entry.get("compacted_at_step"), 0)
            if compacted_at > step_count:
                compacted_at = 0
            since = step_count - compacted_at
            return TopicSession(cid, "", since >= compact_steps_i)
        # Runtime resilience: the permanent thread vanished from the store; recreate it once with memory.
        logger.warning("Topic '%s' conversation %s no longer exists; recreating the topic thread.", resolved, cid)
        entry["conversation_id"] = None
        entry["compacted_at_step"] = 0
        reg[resolved] = entry
        save_topic_registry(reg, registry_path)

    return TopicSession(None, format_topic_history_prefix(resolved, registry_path), False)


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


def _fmt_trade_decision_item(d: Dict[str, Any]) -> str:
    if str(d.get("order_type") or "").upper() == "LIMIT" and d.get("limit_price") is not None:
        return f"{d.get('decision', 'HOLD')}(LIMIT@{d.get('limit_price')}):{d.get('symbol', '')}@{d.get('confidence', 0)}"
    return f"{d.get('decision', 'HOLD')}:{d.get('symbol', '')}@{d.get('confidence', 0)}"


def condense_decision(response_text: Any, max_len: int = 360) -> str:
    """Single-line summary of an LLM JSON decision (falls back to the whitespace-collapsed text)."""
    if not isinstance(response_text, str):
        return ""
    condensed = " ".join(response_text.split())
    parsed = extract_json_object(response_text)
    if isinstance(parsed, dict):
        parts = [f"{k}={parsed[k]}" for k in _SHORT_SUMMARY_KEYS if parsed.get(k) not in (None, "")]
        mtds = parsed.get("market_trading_decisions")
        if isinstance(mtds, list) and mtds:
            t_items = [_fmt_trade_decision_item(d) for d in mtds if isinstance(d, dict)]
            if t_items:
                parts.append(f"trade={','.join(t_items)}")
        else:
            mtd = parsed.get("market_trading_decision")
            if isinstance(mtd, dict):
                parts.append(f"trade={_fmt_trade_decision_item(mtd)}")
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
) -> None:
    """Update the topic registry after a completed turn and mirror the thread into the Web UI store.

    The topic's conversation id is adopted once and replaced only when rotated or explicitly replaced
    via ``replace_conversation=True``; ``compacted`` marks a turn that compacted the thread in-place
    with `/compact` so the next compaction is scheduled from the current step count.
    """
    spec, resolved, base = _resolve_topic_context(topic_key, cli_base)
    compacted = _parse_bool_flag(compacted, default=False)
    replace_conv = _parse_bool_flag(replace_conversation, default=False)
    max_history_entries = safe_nonneg_int(max_history_entries, default=None)
    if max_history_entries is None:
        max_history_entries = required_int_env("LLM_TOPIC_HISTORY_MAX_ENTRIES")

    _, reg, entry = _load_topic_entry(resolved, registry_path)
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
        if compacted and active_cid == clean_cid:
            entry["compacted_at_step"] = step_count
            entry["compaction_count"] = safe_nonneg_int(entry.get("compaction_count"), 0) + 1
    entry["topic_key"] = resolved
    entry["turn_count"] = safe_nonneg_int(entry.get("turn_count"), 0) + 1
    entry["updated_ts"] = datetime.now(timezone.utc).timestamp()
    entry["title"] = spec.title

    history = [h for h in entry.get("rolling_history") or [] if isinstance(h, dict)]
    condensed = condense_decision(response_text)
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
    if active_cid:
        _apply_and_mirror_topic_title(base, active_cid, spec.title)
