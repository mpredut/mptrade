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
import math
import os
import re
from typing import Any, Dict, NamedTuple, Optional, Set

from botcore import _parse_bool_flag, required_int_env
from orchestratorOS.agy.paths import expand_config_path, resolve_bot_cli_base
from orchestratorOS.agy.summary_store import (
    get_thread_step_count,
    read_summary_row,
    upsert_ui_visible_summary,
    write_annotation_title,
)
from orchestratorOS.agy.ui_mirror import (
    mirror_topic_thread_to_ui,
    sync_topic_thread_from_ui,
)

logger = logging.getLogger("orchestratorOS.agy.topics")

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass(frozen=True)
class TopicSpec:
    """Permanent identity and capabilities of an LLM topic thread."""
    title: str                    # thread name shown in the Web UI (never changes)
    can_trade: bool = False       # may submit trade intents to the AutonomousIntentGateway
    can_self_heal: bool = False   # may apply verified code-remediation patches
    asset_scope: str = "all"      # "crypto" | "equities" | "all"

    @property
    def bracket_tag(self) -> str:
        """Canonical '[auto_...]' tag from the thread title."""
        m = re.search(r"(\[[^\]]+\])", self.title)
        return m.group(1) if m else self.title

    @property
    def ntfy_prefix(self) -> str:
        """Canonical '<icon> [auto_...]' prefix from the beginning of the thread title."""
        m = re.match(r"^([^\[]*\[[^\]]+\])", self.title)
        return m.group(1).strip() if m else self.title


TOPIC_SPECS: Dict[str, TopicSpec] = {
    "auto_technical_repair": TopicSpec(
        "🛠 [auto_technical_repair] System Audit & Code Self-Heal (03:35)", can_self_heal=True
    ),
    "auto_cryptotrade_executor": TopicSpec(
        "🎯 [auto_cryptotrade_executor] Autonomous Crypto Trade Decisions (Binance/Kraken/HL 10m)",
        can_trade=True,
        asset_scope="crypto",
    ),
    "auto_sharetrade_executor": TopicSpec(
        "📈 [auto_sharetrade_executor] Autonomous Share Trade Decisions (Trading212 RTH 10m)",
        can_trade=True,
        asset_scope="equities",
    ),
    "macro_geopolitical": TopicSpec("🌍 [auto_macro_geopolitical] Geopolitical Crisis & Breaking News Brake"),
    "macro_sentiment": TopicSpec("🧭 [auto_macro_sentiment] Fear & Greed, Breadth & Market Bias (3h)"),
    "guard_high_stake": TopicSpec("🛡 [auto_guard_large_order] Pre-Trade Veto for Large Orders (>=700 EUR)"),
    "guard_post_stop": TopicSpec("🛡 [auto_guard_post_stop] Pre-Trade Veto for Post-Stop Re-Buy Bounce"),
}

TOPIC_ALIASES: Dict[str, str] = {
    "auto_trade_executor": "auto_cryptotrade_executor",
}


def canonical_topic_key(topic_key: Any) -> str:
    """Resolve legacy topic aliases (e.g. auto_trade_executor -> auto_cryptotrade_executor)."""
    if not isinstance(topic_key, str):
        return ""
    cleaned = topic_key.strip()
    return TOPIC_ALIASES.get(cleaned, cleaned)


def topic_spec(topic_key: str) -> TopicSpec:
    """Spec of a registered topic; unknown topics fail fast."""
    resolved = canonical_topic_key(topic_key)
    if resolved not in TOPIC_SPECS:
        raise KeyError(f"Unknown LLM topic '{topic_key}'; register it in TOPIC_SPECS")
    return TOPIC_SPECS[resolved]


def get_topic_registry_path(registry_path: Optional[str] = None) -> str:
    """Registry JSON path: explicit, LLM_TOPIC_REGISTRY_PATH (tests/ops override), or cachedb/llm_topic_threads.json."""
    raw = registry_path or os.environ.get("LLM_TOPIC_REGISTRY_PATH", "").strip()
    if raw:
        return expand_config_path(raw)
    try:
        from cachepaths import cache_path
        return cache_path("llm_topic_threads.json")
    except Exception:
        return os.path.join(_ROOT_DIR, "cachedb", "llm_topic_threads.json")


def load_topic_registry(registry_path: Optional[str] = None) -> Dict[str, dict]:
    from state_io import previous_generation_path, read_recovering_previous

    path = get_topic_registry_path(registry_path)
    if not os.path.exists(path) and not os.path.exists(previous_generation_path(path)):
        return {}
    try:
        data = read_recovering_previous(path, json.loads, default=None)
    except (OSError, ValueError) as e:
        logger.warning("Unreadable topic registry %s: %s", path, e)
        return {}
    if not isinstance(data, dict):
        return {}
    for old_key, new_key in TOPIC_ALIASES.items():
        if old_key in data:
            old_entry = data.pop(old_key)
            if new_key not in data and isinstance(old_entry, dict):
                old_entry["topic_key"] = new_key
                data[new_key] = old_entry
    return data


def save_topic_registry(registry: Dict[str, dict], registry_path: Optional[str] = None) -> None:
    from state_io import save_json_state

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
        spec_title = TOPIC_SPECS[key].title if key in TOPIC_SPECS else ""
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
    from orchestratorOS.agy.ui_mirror import mirror_topic_thread_to_ui

    base = resolve_bot_cli_base(cli_base)
    reg = load_topic_registry(registry_path)
    updated = 0
    changed_reg = False
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


def format_topic_history_prefix(
    topic_key: str,
    registry_path: Optional[str] = None,
) -> str:
    """Format the rolling decision memory for a topic as a prompt prefix."""
    resolved = canonical_topic_key(topic_key)
    reg = load_topic_registry(registry_path)
    entry = reg.get(resolved)
    if not isinstance(entry, dict):
        return ""
    history = [h for h in entry.get("rolling_history") or [] if isinstance(h, dict)]
    if not history:
        return ""
    lines = ["[TOPIC MEMORY — PRIOR DECISIONS IN THIS TOPIC]:"]
    lines += [f"- [{h.get('utc', '')}] {h.get('summary', '')}" for h in history]
    return "\n".join(lines) + "\n\n"


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None or isinstance(val, bool):
        return default
    try:
        fval = float(val)
        if not math.isfinite(fval):
            return default
        return int(round(fval))
    except (TypeError, ValueError, OverflowError):
        return default


def prepare_topic_session(
    topic_key: str,
    cli_base: Optional[str] = None,
    compact_steps: Optional[int] = None,
    registry_path: Optional[str] = None,
) -> TopicSession:
    """Resolve the permanent conversation of a topic and whether this turn must compact it."""
    topic_spec(topic_key)
    resolved = canonical_topic_key(topic_key)
    base = resolve_bot_cli_base(cli_base)
    if compact_steps is None:
        compact_steps = required_int_env("LLM_TOPIC_COMPACT_STEPS")
    if isinstance(compact_steps, bool):
        raise ValueError(f"LLM_TOPIC_COMPACT_STEPS must be > 0, got {compact_steps}")
    try:
        compact_steps_f = float(compact_steps)
        if not math.isfinite(compact_steps_f) or not compact_steps_f.is_integer():
            raise ValueError(f"LLM_TOPIC_COMPACT_STEPS must be > 0, got {compact_steps}")
        compact_steps_i = int(compact_steps_f)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"LLM_TOPIC_COMPACT_STEPS must be > 0, got {compact_steps}") from exc
    if compact_steps_i <= 0:
        raise ValueError(f"LLM_TOPIC_COMPACT_STEPS must be > 0, got {compact_steps}")

    reg = load_topic_registry(registry_path)
    entry = reg.get(resolved)
    if not isinstance(entry, dict):
        return TopicSession(None, "", False)

    raw_cid = entry.get("conversation_id")
    cid = raw_cid.strip() if isinstance(raw_cid, str) else ""
    if cid:
        from orchestratorOS.agy.ui_mirror import sync_topic_thread_from_ui

        sync_topic_thread_from_ui(cid, bot_cli_base=base)
        exists, step_count, _ = get_thread_step_count(cid, base)
        if exists:
            compacted_at = _safe_int(entry.get("compacted_at_step"), 0)
            if compacted_at < 0 or compacted_at > step_count:
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


def extract_json_object(text: Any) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from pure JSON or a hybrid Markdown + trailing JSON response."""
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    if "```" in cleaned:
        fences = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        for block in reversed(fences):
            cand = block.strip()
            if not cand:
                continue
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return obj
            except ValueError:
                pass
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    pos = len(cleaned)
    while True:
        idx = cleaned.rfind("{", 0, pos)
        if idx < 0:
            return None
        try:
            obj, _ = decoder.raw_decode(cleaned[idx:])
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
        pos = idx


def condense_decision(response_text: Any, max_len: int = 220) -> str:
    """Single-line summary of an LLM JSON decision (falls back to the whitespace-collapsed text)."""
    if not isinstance(response_text, str):
        return ""
    condensed = " ".join(response_text.split())
    parsed = extract_json_object(response_text)
    if isinstance(parsed, dict):
        parts = [f"{k}={parsed[k]}" for k in _SHORT_SUMMARY_KEYS if parsed.get(k) not in (None, "")]
        mtds = parsed.get("market_trading_decisions")
        if isinstance(mtds, list) and mtds:
            t_items = [
                f"{d.get('decision', 'HOLD')}:{d.get('symbol', '')}@{d.get('confidence', 0)}"
                for d in mtds
                if isinstance(d, dict)
            ]
            if t_items:
                parts.append(f"trade={','.join(t_items)}")
        else:
            mtd = parsed.get("market_trading_decision")
            if isinstance(mtd, dict):
                parts.append(f"trade={mtd.get('decision', 'HOLD')}:{mtd.get('symbol', '')}@{mtd.get('confidence', 0)}")
        parts.extend(f"{k}={parsed[k]}" for k in _PROSE_SUMMARY_KEYS if parsed.get(k) not in (None, ""))
        if parts:
            condensed = "; ".join(parts)
    max_len = max(4, _safe_int(max_len, 220))
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
    compacted the thread in-place with `/compact` so the next compaction is scheduled from the current step count.
    """
    from orchestratorOS.agy.ui_mirror import mirror_topic_thread_to_ui

    spec = topic_spec(topic_key)
    resolved = canonical_topic_key(topic_key)
    base = resolve_bot_cli_base(cli_base)
    compacted = _parse_bool_flag(compacted, default=False)
    if max_history_entries is None or isinstance(max_history_entries, bool):
        max_history_entries = required_int_env("LLM_TOPIC_HISTORY_MAX_ENTRIES")
    else:
        parsed_max = _safe_int(max_history_entries, -1)
        max_history_entries = (
            parsed_max if parsed_max >= 0 else required_int_env("LLM_TOPIC_HISTORY_MAX_ENTRIES")
        )

    reg = load_topic_registry(registry_path)
    entry = reg.get(resolved)
    if not isinstance(entry, dict):
        entry = {"topic_key": resolved, "turn_count": 0, "rolling_history": []}

    clean_cid = conversation_id.strip() if isinstance(conversation_id, str) and conversation_id.strip() else None
    raw_existing = entry.get("conversation_id")
    existing_cid = raw_existing.strip() if isinstance(raw_existing, str) and raw_existing.strip() else None
    if not existing_cid and clean_cid:
        entry["conversation_id"] = clean_cid
        entry["compacted_at_step"] = 0
        existing_cid = clean_cid
    active_cid = existing_cid
    if active_cid:
        _, step_count, _ = get_thread_step_count(active_cid, base)
        entry["step_count"] = step_count
        if compacted and active_cid == clean_cid:
            entry["compacted_at_step"] = step_count
            entry["compaction_count"] = _safe_int(entry.get("compaction_count"), 0) + 1
    entry["topic_key"] = resolved
    entry["turn_count"] = _safe_int(entry.get("turn_count"), 0) + 1
    entry["updated_ts"] = datetime.now(timezone.utc).timestamp()
    entry["title"] = spec.title

    history = [h for h in entry.get("rolling_history") or [] if isinstance(h, dict)]
    condensed = condense_decision(response_text)
    if condensed:
        history.append({"utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "summary": condensed})
    entry["rolling_history"] = history[-max_history_entries:] if max_history_entries > 0 else []
    reg[resolved] = entry
    save_topic_registry(reg, registry_path)
    if active_cid:
        try:
            bot_row = read_summary_row(base, active_cid)
            if bot_row is not None:
                upsert_ui_visible_summary(base, bot_row, spec.title)
            else:
                write_annotation_title(base, active_cid, spec.title)
        except OSError:
            pass
        mirror_topic_thread_to_ui(active_cid, spec.title, bot_cli_base=base)
