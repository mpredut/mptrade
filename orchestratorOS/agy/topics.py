"""Permanent LLM topic threads: one Antigravity conversation per topic, compacted in place.

A topic owns a single conversation for its whole life (fixed id and name). Its context is bounded by
Antigravity's `/compact` once LLM_TOPIC_COMPACT_STEPS trajectory steps accumulated since the previous
compaction; the thread is never rotated or deleted.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
import re
from typing import Dict, NamedTuple, Optional, Set

from botcore import required_int_env
from orchestratorOS.agy.paths import expand_config_path, resolve_bot_cli_base
from orchestratorOS.agy.summary_store import get_thread_step_count, write_annotation_title

logger = logging.getLogger("orchestratorOS.agy.topics")

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass(frozen=True)
class TopicSpec:
    """Permanent identity and capabilities of an LLM topic thread."""
    title: str                    # thread name shown in the Web UI (never changes)
    can_trade: bool = False       # may submit trade intents to the AutonomousIntentGateway
    can_self_heal: bool = False   # may apply verified code-remediation patches


TOPIC_SPECS: Dict[str, TopicSpec] = {
    "auto_technical_repair": TopicSpec(
        "🛠 [auto_technical_repair] System Audit & Code Self-Heal (03:35)", can_self_heal=True
    ),
    "auto_trade_executor": TopicSpec(
        "🎯 [auto_trade_executor] Autonomous Trade Decisions (TAO & BTC 5m)", can_trade=True
    ),
    "macro_geopolitical": TopicSpec("🌍 [auto_macro_news_shield] Geopolitical Crisis & Breaking News Brake"),
    "macro_sentiment": TopicSpec("🧭 [auto_macro_sentiment] Fear & Greed, Breadth & Market Bias (3h)"),
    "guard_high_stake": TopicSpec("🛡 [auto_guard_large_order] Pre-Trade Veto for Large Orders (>=700 EUR)"),
    "guard_post_stop": TopicSpec("🛡 [auto_guard_post_stop] Pre-Trade Veto for Post-Stop Re-Buy Bounce"),
}


def topic_spec(topic_key: str) -> TopicSpec:
    """Spec of a registered topic; unknown topics fail fast."""
    if topic_key not in TOPIC_SPECS:
        raise KeyError(f"Unknown LLM topic '{topic_key}'; register it in TOPIC_SPECS")
    return TOPIC_SPECS[topic_key]


def get_topic_registry_path(registry_path: Optional[str] = None) -> str:
    """Registry JSON path: explicit, LLM_TOPIC_REGISTRY_PATH (tests/ops override), or cachedb/llm_topic_threads.json."""
    raw = registry_path or os.environ.get("LLM_TOPIC_REGISTRY_PATH", "").strip()
    return expand_config_path(raw) if raw else os.path.join(_ROOT_DIR, "cachedb", "llm_topic_threads.json")


def load_topic_registry(registry_path: Optional[str] = None) -> Dict[str, dict]:
    path = get_topic_registry_path(registry_path)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("Unreadable topic registry %s: %s", path, e)
        return {}
    return data if isinstance(data, dict) else {}


def save_topic_registry(registry: Dict[str, dict], registry_path: Optional[str] = None) -> None:
    from state_io import atomic_write_json

    path = get_topic_registry_path(registry_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    atomic_write_json(path, registry, indent=2)


def get_active_topic_cids(registry_path: Optional[str] = None) -> Set[str]:
    """Conversation ids owned by topics (protected from pruning)."""
    return {
        str(e["conversation_id"])
        for e in load_topic_registry(registry_path).values()
        if isinstance(e, dict) and e.get("conversation_id")
    }


class TopicSession(NamedTuple):
    """How the next turn of a topic thread must be issued."""
    conversation_id: Optional[str]   # permanent thread to resume; None only before the thread exists
    history_prefix: str              # carried-over decisions, only when the thread must be (re)created
    compact: bool                    # issue the turn as `/compact <prompt>` (in-place context compaction)


def prepare_topic_session(
    topic_key: str,
    cli_base: Optional[str] = None,
    compact_steps: Optional[int] = None,
    registry_path: Optional[str] = None,
) -> TopicSession:
    """Resolve the permanent conversation of a topic and whether this turn must compact it."""
    topic_spec(topic_key)
    base = resolve_bot_cli_base(cli_base)
    if compact_steps is None:
        compact_steps = required_int_env("LLM_TOPIC_COMPACT_STEPS")
    if compact_steps <= 0:
        raise ValueError(f"LLM_TOPIC_COMPACT_STEPS must be > 0, got {compact_steps}")

    reg = load_topic_registry(registry_path)
    entry = reg.get(topic_key)
    if not isinstance(entry, dict):
        return TopicSession(None, "", False)

    cid = str(entry.get("conversation_id") or "").strip()
    if cid:
        from orchestratorOS.agy.ui_mirror import sync_topic_thread_from_ui

        sync_topic_thread_from_ui(cid, bot_cli_base=base)
        exists, step_count, _ = get_thread_step_count(cid, base)
        if exists:
            since = step_count - int(entry.get("compacted_at_step") or 0)
            return TopicSession(cid, "", since >= compact_steps)
        # Runtime resilience: the permanent thread vanished from the store; recreate it once with memory.
        logger.warning("Topic '%s' conversation %s no longer exists; recreating the topic thread.", topic_key, cid)
        entry["conversation_id"] = None
        entry["compacted_at_step"] = 0
        reg[topic_key] = entry
        save_topic_registry(reg, registry_path)

    history = [h for h in entry.get("rolling_history") or [] if isinstance(h, dict)]
    if not history:
        return TopicSession(None, "", False)
    lines = ["[TOPIC MEMORY — PRIOR DECISIONS IN THIS TOPIC]:"]
    lines += [f"- [{h.get('utc', '')}] {h.get('summary', '')}" for h in history]
    return TopicSession(None, "\n".join(lines) + "\n\n", False)


_SUMMARY_KEYS = ("decision", "threat_level", "market_bias", "risk_level", "recommended_action",
                 "recommended_brake", "suggested_scale", "confidence", "summary", "reason")


def condense_decision(response_text: str, max_len: int = 220) -> str:
    """Single-line summary of an LLM JSON decision (falls back to the whitespace-collapsed text)."""
    raw = (response_text or "").strip()
    fences = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    for block in reversed(fences):
        cand = block.strip()
        if cand.startswith("{") and cand.endswith("}"):
            raw = cand
            break
    else:
        if fences:
            raw = fences[-1].strip()
    condensed = " ".join(raw.split())
    try:
        parsed = json.loads(condensed)
    except ValueError:
        start, end = condensed.find("{"), condensed.rfind("}")
        try:
            parsed = json.loads(condensed[start : end + 1]) if 0 <= start < end else None
        except ValueError:
            parsed = None
    if isinstance(parsed, dict):
        parts = [f"{k}={parsed[k]}" for k in _SUMMARY_KEYS if parsed.get(k) not in (None, "")]
        mtd = parsed.get("market_trading_decision")
        if isinstance(mtd, dict):
            parts.append(f"trade={mtd.get('decision', 'HOLD')}:{mtd.get('symbol', '')}@{mtd.get('confidence', 0)}")
        if parts:
            condensed = "; ".join(parts)
    return condensed if len(condensed) <= max_len else condensed[: max_len - 3] + "..."


def record_topic_turn(
    topic_key: str,
    conversation_id: Optional[str],
    response_text: str,
    compacted: bool = False,
    cli_base: Optional[str] = None,
    max_history_entries: Optional[int] = None,
    registry_path: Optional[str] = None,
) -> None:
    """Update the topic registry after a completed turn and mirror the thread into the Web UI store.

    The topic's conversation id is adopted once and then never replaced; ``compacted`` marks a turn that
    was issued as `/compact <prompt>` so the next compaction is scheduled from the current step count.
    """
    from orchestratorOS.agy.ui_mirror import mirror_topic_thread_to_ui

    spec = topic_spec(topic_key)
    base = resolve_bot_cli_base(cli_base)
    if max_history_entries is None:
        max_history_entries = required_int_env("LLM_TOPIC_HISTORY_MAX_ENTRIES")

    reg = load_topic_registry(registry_path)
    entry = reg.get(topic_key)
    if not isinstance(entry, dict):
        entry = {"topic_key": topic_key, "turn_count": 0, "rolling_history": []}

    if not entry.get("conversation_id") and conversation_id:
        entry["conversation_id"] = conversation_id
        entry["compacted_at_step"] = 0
    active_cid = entry.get("conversation_id")
    if active_cid:
        _, step_count, _ = get_thread_step_count(active_cid, base)
        entry["step_count"] = step_count
        if compacted and active_cid == conversation_id:
            entry["compacted_at_step"] = step_count
            entry["compaction_count"] = int(entry.get("compaction_count") or 0) + 1
    entry["topic_key"] = topic_key
    entry["turn_count"] = int(entry.get("turn_count") or 0) + 1
    entry["updated_ts"] = datetime.now(timezone.utc).timestamp()
    entry["title"] = spec.title

    history = [h for h in entry.get("rolling_history") or [] if isinstance(h, dict)]
    condensed = condense_decision(response_text)
    if condensed:
        history.append({"utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "summary": condensed})
    entry["rolling_history"] = history[-max_history_entries:] if max_history_entries > 0 else history
    reg[topic_key] = entry
    save_topic_registry(reg, registry_path)
    if active_cid:
        try:
            write_annotation_title(base, active_cid, spec.title)
        except OSError:
            pass
        mirror_topic_thread_to_ui(active_cid, spec.title, bot_cli_base=base)
