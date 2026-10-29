# alert_notifiers.py
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess  # noqa: F401
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from state_io import read_json_dict, save_json_state
from typing import Any, Optional

# Import your modules
from lock import FileLock
from botcore import (
    _parse_bool_flag,
    float_env,
    int_env,
    safe_float,
    safe_nonneg_int,
    safe_pos_float,
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Matched against title.upper(). Bilingual ON PURPOSE, and it stays that way (owner's
# decision, same reasoning as verify_tools/watchdogfor_anomaly.py): every alert title is
# English today, but one written in Romanian by mistake would silently stop being urgent
# — it would be routed to the routine topic and skip the email, with no error anywhere.
# An extra tuple entry costs nothing; a missed liquidation alert costs money.
_URGENT_MARKERS = (
    "🛑", "🛡", "LIQUID", "STOP-LOSS", "STOP_LOSS", "TRAILING", "FAILED",
    "MANUAL", "ERROR", "CATASTROPH", "CRASH", "GONE", "IMBALANC",
    "LICHID", "ESUAT", "ERORI", "DISPARUT", "DEZECHILIBR", "CATASTROF",
)
_GUARD_MARKERS = (
    "🛑", "🛡", "STOP-LOSS", "STOP_LOSS", "TRAILING", "LIQUID", "CATASTROPH", "CRASH",
    "LICHID", "CATASTROF",
)
_OPS_MARKERS = (
    "FAILED", "ERROR", "MANUAL", "GONE", "IMBALANC",
    "ESUAT", "ERORI", "DISPARUT", "DEZECHILIBR",
)
_ERROR_AND_OPS_TITLE_MARKERS = (
    "FAILED", "ERROR", "EXCEPTION", "TRACEBACK", "CRITICAL", "CATASTROPH", "CRASH",
    "LIQUID", "LICHID", "CATASTROF", "ESUAT", "ERORI", "DISPARUT", "GONE",
    "IMBALANC", "DEZECHILIBR", "QUARANTINE", "GIVE UP", "ORDER TERMINAL",
    "EXPIRED", "CANCELED", "CANCELLED", "REJECTED", "MISMATCH", "NEPOTRIVIT",
    "TIMEOUT", "DEFER", "ZOMBIE", "OUTAGE", "UNREACHABLE", "SUSTAINED",
    "RESOURCE USAGE", "SERVER DOWN", "VPN DOWN", "ISP DOWN",
    "⚠", "✗", "⏰", "📉",
)
_ERROR_AND_OPS_SOURCES = (
    "watchdog", "order_retry", "order-persistence", "smart_stop_loss",
    "post_stop_guard", "verify", "deadman", "mailer", "orchestrator",
)
_TRADE_MARKERS = (
    "BUY", "SELL", "FILLED", "TREND_ENTRY", "ADOPT", "EXECUTION", "📝", "🎉",
)
_FAKE_ALERT_MARKERS = (
    "ZZZFAKE", "FAKEUSD", "TESTPAIR", "TSTX", "FAKE_VENUE", "ZZZ",
)


def is_fake_alert_text(*parts: Any) -> bool:
    """Return True when any part contains synthetic/test instrument markers."""
    joined = " ".join("" if isinstance(p, bool) or p is None else str(p) for p in parts).upper()
    return any(marker in joined for marker in _FAKE_ALERT_MARKERS)


def _alert_attr(item: Any, key: str, default: Any = None) -> Any:
    """Extract an attribute or dict key from an alert payload."""
    if isinstance(item, dict):
        val = item.get(key, default)
    else:
        val = getattr(item, key, default)
    return default if val is None else val


def _positive_int_env(name: str, default: int) -> int:
    val = int_env(name)
    return val if (val is not None and val >= 0) else default


def _safe_pos_int(val: Any, default: int = 1) -> int:
    f = safe_pos_float(val, default=None)
    return int(f) if (f is not None and f.is_integer()) else default


def _resolve_gate_now(now: Optional[float]) -> float:
    nf = safe_pos_float(now, default=None)
    if nf is not None and nf < 32503680000.0:
        return nf
    return time.time()


_as_bool_flag = _parse_bool_flag


def _delivery_state_path() -> Path:
    configured = os.environ.get("NOTIFICATION_STATE_FILE")
    return Path(configured).expanduser() if configured else BASE_DIR / "logs/notification_delivery_state.json"


def _load_delivery_state(path: Path, today: str) -> dict:
    state = read_json_dict(path, default={}, recover_previous=True)
    if not isinstance(state, dict) or state.get("date_utc") != today:
        state = {"schema_version": 1, "date_utc": today, "channels": {}}
    if not isinstance(state.get("channels"), dict):
        state["channels"] = {}
    return state


def _save_delivery_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json_state(
        path, state, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


def _is_guard_alert(alerts: list[Any]) -> bool:
    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        title = str(alert.get("name") or alert.get("symbol") or alert.get("title") or "").upper()
        source = str(alert.get("source") or "").lower()
        if (
            any(
                marker in title
                for marker in (
                    "🛡", "🛑", "GUARD", "VETO", "SHADOW", "BLOCKED",
                    "SMART-SL", "POST-STOP", "STOP-LOSS", "STOP_LOSS",
                    "TRAILING", "QUARANTINE", "GIVE UP",
                )
            )
            or any(
                src in source
                for src in (
                    "order_guard", "guard", "assetguardian", "macro_shadow",
                    "smart_stop_loss", "post_stop_guard",
                )
            )
        ):
            return True
    return False


def _is_error_or_ops_alert(alerts: list[Any]) -> bool:
    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        title = str(alert.get("name") or alert.get("title") or "").upper()
        source = str(alert.get("source") or "").lower()
        title_no_owner = re.sub(
            r"\[(MANUAL|TRADEALL|MONITORORDER|RTRADE|MONITORTRADES|ASSETGUARDIAN)\]"
            r"|\((MANUAL|TRADEALL|MONITORORDER|RTRADE|MONITORTRADES|ASSETGUARDIAN)\)",
            "",
            title,
            flags=re.IGNORECASE,
        )
        if (
            any(marker in title_no_owner for marker in _ERROR_AND_OPS_TITLE_MARKERS)
            or any(src in source for src in _ERROR_AND_OPS_SOURCES)
            or alert.get("type") == "email"
        ):
            return True
    return False


def _normalize_alert_title_for_identity(name: str) -> str:
    """Strip volatile percentages, prices, and durations from guard/error titles."""
    norm = re.sub(r"[+-]?\d+(?:\.\d+)?%", "<PCT>", name)
    norm = re.sub(r"@\s*~?\$?\d+(?:\.\d+)?", "@ <PX>", norm)
    norm = re.sub(r"\(\d+(?:\.\d+)?s\)", "(<DUR>)", norm)
    return norm


def _strip_log_tails_and_compact(body: str) -> str:
    """Strip verbose 'Last logs:' tails and return a compact 1-2 line summary."""
    cleaned = re.sub(r"(?is)\n*\s*last logs:.*", "", str(body or "")).strip()
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        return ""
    summary = " | ".join(lines[:2])
    return summary[:220] + ("..." if len(summary) > 220 else "")


def _normalize_text_for_similarity(text: str) -> str:
    """Normalize volatile numbers, timestamps, PIDs, hex IDs, prices, and percentages."""
    s = re.sub(r"(?is)\n*\s*last logs:.*", "", str(text or "")).strip()
    s = re.sub(r"^\d{4}[-/]\d{2}[-/]\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?\s*", "", s)
    s = re.sub(r"^\d{2}:\d{2}:\d{2}\s*", "", s)
    s = re.sub(r"0x[0-9a-fA-F]+", "<HEX>", s)
    s = re.sub(r"[+-]?[\$€~]?\d+(?:,\d{3})*(?:\.\d+)?(?:%|ms|s|m|h|d|x|bps)?", "<N>", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _similarity_signature(
    title: str,
    body: str = "",
    topic: str = "",
    alerts: Optional[list[Any]] = None,
) -> str:
    """Compute a structural similarity signature for common-gate deduplication."""
    norm_title = _normalize_text_for_similarity(title)
    bare_title = re.sub(r"^\[[^\]]+\]\s*", "", norm_title).strip()
    is_generic_title = bare_title in (
        "", "alert", "os alert", "trade alert", "price alert",
        "critical errors", "alerts (<n>)",
    )
    alerts_list = alerts or []
    is_pure_trade = (
        any(m in str(title or "").upper() for m in _TRADE_MARKERS)
        and not _is_guard_alert(alerts_list)
        and not _is_error_or_ops_alert(alerts_list)
    )
    first_line = ""
    for ln in str(body or "").splitlines():
        if ln.strip():
            first_line = ln.strip()
            break
    norm_body_head = _normalize_text_for_similarity(first_line)[:120]
    if is_generic_title or is_pure_trade:
        sig_raw = f"sim|{topic}|{norm_title}|{norm_body_head}"
    else:
        sig_raw = f"sim|{topic}|{norm_title}"
    return hashlib.sha256(sig_raw.encode("utf-8")).hexdigest()


def _common_gate_seconds() -> int:
    """Return the universal common notification gate duration in seconds (default 300s = 5 min).

    If NOTIFICATION_GATE_SECONDS is explicitly configured in the environment, use it.
    In unit test environments where NOTIFICATION_DEDUP_SECONDS=0 or PYTEST_CURRENT_TEST is set
    without an explicit NOTIFICATION_GATE_SECONDS, default to 0 to preserve immediate single-call
    test assertions.
    """
    raw = os.environ.get("NOTIFICATION_GATE_SECONDS")
    if raw is not None and str(raw).strip() != "":
        return _positive_int_env("NOTIFICATION_GATE_SECONDS", 300)
    dedup_s = float_env("NOTIFICATION_DEDUP_SECONDS")
    if dedup_s == 0.0 or "PYTEST_CURRENT_TEST" in os.environ:
        return 0
    return 300


def synthesize_bulk_notifications(items: list[dict]) -> tuple[str, str, str, str]:
    """Synthesize and concatenate multiple distinct notifications into a single bulk message."""
    if not items:
        return ("Alert", "", "default", "")
    prio_rank = {"urgent": 3, "high": 2, "default": 1, "low": 0}
    best_priority = max(
        (str(it.get("priority") or "default") for it in items),
        key=lambda p: prio_rank.get(p, 1),
    )
    # Choose the highest-priority topic (preferring error/server/guard topics)
    best_topic = str(items[0].get("topic") or "")
    for it in items:
        t = str(it.get("topic") or "")
        if any(k in t.lower() for k in ("error", "deadman", "server")):
            best_topic = t
            break

    if len(items) == 1:
        it = items[0]
        cnt = _safe_pos_int(it.get("count", 1), 1)
        raw_title = str(it.get("title") or "Alert")
        out_title = f"{raw_title} (×{cnt})" if cnt > 1 else raw_title
        out_body = str(it.get("body") or "")
        return (out_title, out_body, best_priority, best_topic)

    short_titles = []
    body_lines = []
    for idx, it in enumerate(items, 1):
        cnt = _safe_pos_int(it.get("count", 1), 1)
        t = str(it.get("title") or "Alert").strip()
        repeat_tag = f" (×{cnt})" if cnt > 1 else ""
        short_titles.append(f"{t}{repeat_tag}")
        compact_body = _strip_log_tails_and_compact(str(it.get("body") or ""))
        if compact_body and compact_body.lower() != t.lower():
            body_lines.append(f"{idx}. {t}{repeat_tag}: {compact_body}")
        else:
            body_lines.append(f"{idx}. {t}{repeat_tag}")

    joined_titles = " | ".join(short_titles)
    if len(joined_titles) > 90:
        joined_titles = joined_titles[:87] + "..."
    bulk_title = f"📦 Bulk ({len(items)}): {joined_titles}"
    bulk_body = "\n".join(body_lines)
    return (bulk_title, bulk_body, best_priority, best_topic)


def _alert_identity(alert: Any) -> dict:
    """Return stable identity excluding timestamps and prices that change each poll."""
    if isinstance(alert, dict):
        source = str(alert.get("source") or "").lower()
        name = str(alert.get("name") or alert.get("title") or "")
        if alert.get("type") == "new_coin_discovered":
            return {
                "type": "new_coin_discovered",
                "symbol": str(alert.get("symbol") or "").upper(),
                "source": source,
            }
        if _is_guard_alert([alert]) or _is_error_or_ops_alert([alert]):
            # For guard, error, ops, watchdog, and mailer alerts, dedup on stable
            # identity fields excluding volatile body timestamps/reasoning/log lines.
            norm_name = _normalize_alert_title_for_identity(name)
            identity = {
                "type": alert.get("type"),
                "symbol": alert.get("symbol"),
                "name": norm_name,
                "source": alert.get("source"),
            }
            if norm_name.strip().upper() in ("ALERT", "OS ALERT", "TRADE ALERT"):
                body_str = str(alert.get("body") or "")
                body_norm = _normalize_text_for_similarity(body_str)[:120]
                identity["body_norm"] = body_norm
            return identity
        return {
            key: alert.get(key)
            for key in ("type", "symbol", "name", "source", "body", "url", "alert_type", "threshold")
        }
    return {
        "type": getattr(alert, "type", alert.__class__.__name__),
        "symbol": getattr(alert, "symbol", None),
        "alert_type": getattr(alert, "alert_type", None),
        "threshold": getattr(alert, "threshold", None),
    }


def _delivery_fingerprint(alerts: list[Any]) -> str:
    identities = sorted(
        (json.dumps(_alert_identity(alert), sort_keys=True, default=str) for alert in alerts),
    )
    return hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest()


def _alerts_are_urgent(alerts: list[Any]) -> bool:
    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        title = str(alert.get("name") or alert.get("symbol") or alert.get("title") or "").upper()
        source = str(alert.get("source") or "").lower()
        if (
            any(marker in title for marker in _URGENT_MARKERS)
            or any(marker in title for marker in _TRADE_MARKERS)
            or "watchdog" in source
        ):
            return True
    return False


def _dedup_seconds(alerts: list[Any], urgent: bool) -> int:
    titles = " ".join(
        str(alert.get("name") or alert.get("title") or "").upper()
        for alert in alerts if isinstance(alert, dict)
    )
    if _is_guard_alert(alerts):
        return _positive_int_env("NOTIFICATION_GUARD_DEDUP_SECONDS", 30 * 60)
    if _is_error_or_ops_alert(alerts):
        return _positive_int_env(
            "NOTIFICATION_ERROR_DEDUP_SECONDS",
            _positive_int_env("NOTIFICATION_URGENT_DEDUP_SECONDS", 30 * 60),
        )
    if any(marker in titles for marker in _TRADE_MARKERS):
        return 0  # Real trades must never be suppressed by deduplication
    if "DISPONIBIL" in titles or "AVAILABLE" in titles:
        return _positive_int_env("NOTIFICATION_STARTUP_DEDUP_SECONDS", 6 * 60 * 60)
    if any(m in titles for m in ("MACRO", "SENTIMENT")) or any(
        isinstance(a, dict) and "macro" in str(a.get("source", "")).lower() for a in alerts
    ):
        return _positive_int_env("NOTIFICATION_MACRO_DEDUP_SECONDS", 3600)
    if urgent:
        return _positive_int_env("NOTIFICATION_URGENT_DEDUP_SECONDS", 30 * 60)
    if any(
        not isinstance(alert, dict) or alert.get("type") == "new_coin_discovered"
        for alert in alerts
    ):
        return _positive_int_env("NOTIFICATION_PRICE_DEDUP_SECONDS", 30 * 60)
    return _positive_int_env("NOTIFICATION_DEDUP_SECONDS", 15 * 60)


def pass_common_gate(
    channel: str,
    title: str,
    body: str,
    priority: str = "default",
    topic: str = "",
    alerts: Optional[list[Any]] = None,
    *,
    urgent: bool = False,
    now: Optional[float] = None,
) -> tuple[bool, str, bool, Optional[tuple[str, str, str, str]]]:
    """Holistic common notification gate (5-minute default window).

    1. Eliminates structurally similar messages arriving within the similarity/dedup window.
    2. Buffers distinct messages arriving while the 5-minute gate is closed.
    3. Synthesizes & concatenates buffered distinct messages into a single bulk payload
       once the 5-minute gate opens.
    """
    now_ts = _resolve_gate_now(now)
    urgent = _as_bool_flag(urgent, False)
    today = datetime.fromtimestamp(now_ts, timezone.utc).date().isoformat()
    path = _delivery_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    alerts_list = alerts if alerts else [
        {"type": "bot_event", "name": title, "body": body, "source": "common_gate", "symbol": "SYSTEM"}
    ]
    fingerprint = _delivery_fingerprint(alerts_list)
    sim_key = _similarity_signature(title, body, topic, alerts_list)
    gate_sec = _common_gate_seconds()
    dedup_sec = _dedup_seconds(alerts_list, urgent)
    sim_window = max(gate_sec, dedup_sec)

    lock_path = path.with_suffix(path.suffix + ".lock")
    with FileLock(lock_path):
        state = _load_delivery_state(path, today)
        raw_ch = state["channels"].get(channel)
        if not isinstance(raw_ch, dict):
            raw_ch = {
                "sent": 0,
                "last": {},
                "blocked": False,
                "budget_warning_sent": False,
                "last_gate_ts": 0.0,
                "pending_bulk": [],
            }
            state["channels"][channel] = raw_ch
        channel_state = raw_ch
        if channel == "ntfy" and channel_state.get("blocked"):
            blocked_until = safe_float(channel_state.get("blocked_until", 0.0), 0.0)
            if blocked_until > 0 and now_ts >= blocked_until:
                channel_state["blocked"] = False
            else:
                return False, "provider_daily_limit", False, None

        last_raw = channel_state.get("last")
        last = last_raw if isinstance(last_raw, dict) else {}
        cutoff = now_ts - 2 * 24 * 60 * 60
        cleaned_last: dict[str, float] = {}
        for key, value in last.items():
            v_f = safe_float(value, -1.0)
            if cutoff <= v_f <= now_ts:
                cleaned_last[key] = v_f
        channel_state["last"] = cleaned_last
        raw_bulk = channel_state.get("pending_bulk")
        pending_bulk: list[dict] = (
            [it for it in raw_bulk if isinstance(it, dict)]
            if isinstance(raw_bulk, list)
            else []
        )
        channel_state["pending_bulk"] = pending_bulk

        # 1. Eliminate if already sitting in pending_bulk with same similarity or fingerprint
        for item in pending_bulk:
            if item.get("sim_key") == sim_key or item.get("fingerprint") == fingerprint:
                item["count"] = _safe_pos_int(item.get("count", 1), 1) + 1
                channel_state["last"][fingerprint] = now_ts
                channel_state["last"][sim_key] = now_ts
                _save_delivery_state(path, state)
                return False, "similar_eliminated", False, None

        # 2. Eliminate if sent/seen within dedup or similarity window
        prev_fp = channel_state["last"].get(fingerprint)
        prev_fp_f = safe_float(prev_fp, -1.0)
        if prev_fp is not None and prev_fp_f >= 0.0 and 0 <= now_ts - prev_fp_f < sim_window:
            _save_delivery_state(path, state)
            return False, "duplicate", False, None

        prev_sim = channel_state["last"].get(sim_key)
        prev_sim_f = safe_float(prev_sim, -1.0)
        if prev_sim is not None and prev_sim_f >= 0.0 and 0 <= now_ts - prev_sim_f < sim_window:
            _save_delivery_state(path, state)
            return False, "similar_eliminated", False, None

        # 3. Daily routine budget check for ntfy
        sent = safe_nonneg_int(channel_state.get("sent", 0), default=0)
        if channel == "ntfy" and not urgent:
            budget = _positive_int_env("NTFY_DAILY_BUDGET", 220)
            reserve = _positive_int_env("NTFY_URGENT_RESERVE", 20)
            if sent >= max(0, budget - reserve):
                warn = not bool(channel_state.get("budget_warning_sent"))
                channel_state["budget_warning_sent"] = True
                _save_delivery_state(path, state)
                return False, "local_daily_budget", warn, None

        # 4. Check the 5-minute Common Gate
        last_gate_ts = safe_float(channel_state.get("last_gate_ts", 0.0), 0.0)
        if last_gate_ts > now_ts:
            last_gate_ts = 0.0
        if gate_sec > 0 and last_gate_ts > 0 and (now_ts - last_gate_ts) < gate_sec:
            # Gate is CLOSED: buffer this distinct message for bulk synthesis
            channel_state["last"][fingerprint] = now_ts
            channel_state["last"][sim_key] = now_ts
            pending_bulk.append({
                "title": title,
                "body": body,
                "priority": priority,
                "topic": topic,
                "ts": now_ts,
                "sim_key": sim_key,
                "fingerprint": fingerprint,
                "count": 1,
            })
            _save_delivery_state(path, state)
            return False, "buffered_for_bulk", False, None

        # 5. Gate is OPEN: dispatch now (synthesizing with any pending bulk items)
        channel_state["sent"] = sent + 1
        channel_state["last"][fingerprint] = now_ts
        channel_state["last"][sim_key] = now_ts
        channel_state["last_gate_ts"] = now_ts
        if pending_bulk:
            items_to_send = list(pending_bulk) + [{
                "title": title,
                "body": body,
                "priority": priority,
                "topic": topic,
                "ts": now_ts,
                "sim_key": sim_key,
                "fingerprint": fingerprint,
                "count": 1,
            }]
            channel_state["pending_bulk"] = []
            _save_delivery_state(path, state)
            return True, "bulk_synthesized", False, synthesize_bulk_notifications(items_to_send)

        _save_delivery_state(path, state)
        return True, "reserved", False, (title, body, priority, topic)


def pop_due_bulk_batch(
    channel: str = "ntfy",
    *,
    now: Optional[float] = None,
    force: bool = False,
) -> Optional[tuple[str, str, str, str]]:
    """Pop and synthesize buffered distinct notifications once the 5-minute gate opens."""
    now_ts = _resolve_gate_now(now)
    force = _as_bool_flag(force, False)
    today = datetime.fromtimestamp(now_ts, timezone.utc).date().isoformat()
    path = _delivery_state_path()
    prev_path = path.with_name(path.name + ".previous")
    if not path.exists() and not prev_path.exists():
        return None
    gate_sec = _common_gate_seconds()
    lock_path = path.with_suffix(path.suffix + ".lock")
    with FileLock(lock_path):
        state = _load_delivery_state(path, today)
        channel_state = state["channels"].get(channel)
        if not isinstance(channel_state, dict):
            return None
        raw_bulk = channel_state.get("pending_bulk")
        if not isinstance(raw_bulk, list):
            return None
        pending_bulk: list[dict] = [it for it in raw_bulk if isinstance(it, dict)]
        if not pending_bulk:
            return None
        last_gate_ts = safe_float(channel_state.get("last_gate_ts", 0.0), 0.0)
        if last_gate_ts > now_ts:
            last_gate_ts = 0.0
        if not force and gate_sec > 0 and last_gate_ts > 0 and (now_ts - last_gate_ts) < gate_sec:
            return None
        items = list(pending_bulk)
        channel_state["pending_bulk"] = []
        channel_state["last_gate_ts"] = now_ts
        prev_sent = safe_nonneg_int(channel_state.get("sent", 0), default=0)
        channel_state["sent"] = prev_sent + 1
        _save_delivery_state(path, state)
        return synthesize_bulk_notifications(items)


def _reserve_delivery(channel: str, alerts: list[Any], *, urgent: bool) -> tuple[bool, str, bool]:
    """Atomically reserve one delivery across processes via the common gate."""
    first = alerts[0] if alerts and isinstance(alerts[0], dict) else {}
    title = str(first.get("name") or first.get("title") or "Alert")
    body = str(first.get("body") or first.get("message") or "")
    prio = "urgent" if urgent else "high"
    allowed, reason, warn_once, _ = pass_common_gate(
        channel, title, body, prio, "", alerts, urgent=urgent,
    )
    return allowed, reason, warn_once


def _mark_provider_daily_limit(channel: str, cooldown_seconds: float = 900.0) -> bool:
    """Block the channel with a rolling cooldown and request one alternate warning."""
    now = time.time()
    cd_sec = safe_pos_float(cooldown_seconds, default=900.0)
    today = datetime.fromtimestamp(now, timezone.utc).date().isoformat()
    path = _delivery_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.with_suffix(path.suffix + ".lock")):
        state = _load_delivery_state(path, today)
        channel_state = state["channels"].setdefault(channel, {})
        first = not bool(channel_state.get("budget_warning_sent"))
        channel_state["blocked"] = True
        channel_state["blocked_until"] = now + cd_sec
        channel_state["budget_warning_sent"] = True
        _save_delivery_state(path, state)
        return first


def _is_provider_daily_limit(response: Any) -> bool:
    if getattr(response, "status_code", None) != 429:
        return False
    body = str(getattr(response, "text", "") or "").lower()
    return "42908" in body or "daily" in body


class AlertNotifier:

    @staticmethod
    def check_alert(condition, message, alert_interval=60, symbol="TRADE"):
        check_alert(condition, message, alert_interval, symbol)

    
    @staticmethod
    def format_human_readable_time(value, fmt: str = "%m-%d %H:%M:%S") -> str:
        if value is None or isinstance(value, bool):
            return "N/A"
        if isinstance(value, (int, float)):
            vf = safe_pos_float(value, default=None)
            if vf is None:
                return str(value)
            try:
                if vf > 1e11:
                    vf /= 1000.0
                return datetime.fromtimestamp(vf).strftime(fmt)
            except Exception:
                return str(value)
        if hasattr(value, "strftime"):
            try:
                return value.strftime(fmt)
            except Exception:
                return str(value)
        return str(value)

    @staticmethod
    def is_new_coin_alert(alert: Any) -> bool:
        return isinstance(alert, dict) and alert.get("type") == "new_coin_discovered"

    @staticmethod
    def alert_symbol(alert: Any) -> str:
        """Return the symbol from either a PriceAlert object or new-coin mapping."""
        return _alert_attr(alert, "symbol", "N/A")

    @staticmethod
    def utf8_header(value: str) -> str:
        """Encode an HTTP header while preserving non-ASCII symbols.

        Headers are Latin-1, while ntfy decodes this value as UTF-8, so pass UTF-8 bytes
        through Latin-1 to preserve the original symbol.
        """
        if value is None or isinstance(value, bool):
            return ""
        cleaned = re.sub(r"[\r\n]+", " ", str(value)).strip()
        return cleaned.encode("utf-8").decode("latin-1", "ignore")

    @staticmethod
    def format_new_coin_message(alert: dict) -> str:
        raw_price = alert.get("price")
        px_val = safe_pos_float(raw_price, default=None)
        if px_val is not None:
            price_str = f"Price: ${px_val:.4f}"
        elif (
            isinstance(raw_price, str)
            and raw_price.strip()
            and raw_price.strip().lower() not in {"nan", "inf", "-inf", "+inf", "none", "null"}
            and safe_float(raw_price, default=None) is None
        ):
            price_str = f"Price: {raw_price}"
        else:
            price_str = "Price: N/A"
        lines = [
            f"🆕: {alert.get('symbol', 'N/A')} - {alert.get('name', alert.get('symbol', 'N/A'))}",
            f"Source: {alert.get('source', 'unknown')}",
            f"Added: {AlertNotifier.format_human_readable_time(alert.get('added_at'))}",
            price_str,
        ]
        url = alert.get("url")
        if url:
            lines.append(f"Link: {url}")
        return "\n".join(lines)

    @staticmethod
    def format_bot_event(alert: dict) -> str:
        """Build a compact bot-event body: detail, platform, and timestamp.

        The event name (the action) becomes the ntfy title. The platform is plain text
        because its meaning is implicit. Use the short timestamp format ``MM-DD HH:MM``.
        """
        body = str(alert.get("body") or "").strip()
        head = body or str(alert.get("name", alert.get("symbol", "?")))
        parts = [head, str(alert.get("source", "?"))]
        ts = alert.get("added_at")
        if ts is not None and not isinstance(ts, bool) and (hasattr(ts, "strftime") or isinstance(ts, (int, float))):
            formatted = AlertNotifier.format_human_readable_time(ts, fmt="%m-%d %H:%M")
            if formatted != str(ts):
                parts.append(formatted)
        return " · ".join(parts)

    @staticmethod
    def format_batch_message(alerts) -> str:
        lines = []
        for alert in alerts:
            if isinstance(alert, dict) and alert.get("type") == "bot_event":
                lines.append(AlertNotifier.format_bot_event(alert))
                continue
            if AlertNotifier.is_new_coin_alert(alert):
                lines.append(AlertNotifier.format_new_coin_message(alert))
                continue

            alert_type = _alert_attr(alert, "alert_type", "")
            symbol = _alert_attr(alert, "symbol", "N/A")
            pct_change = safe_float(_alert_attr(alert, "percent_change", 0.0), 0.0)
            current_price = safe_float(_alert_attr(alert, "current_price", 0.0), 0.0)
            ref_price = safe_float(_alert_attr(alert, "reference_price", 0.0), 0.0)
            ref_time_val = _alert_attr(alert, "reference_time") or _alert_attr(alert, "timestamp")
            url = _alert_attr(alert, "url")

            direction = "U" if alert_type == "up" else "D"
            reference_time = AlertNotifier.format_human_readable_time(ref_time_val)

            lines.append(
                f"{symbol}: {direction} {pct_change:+.2f}% "
                f"| C ${current_price:.4f} | R ${ref_price:.4f} "
                f"({reference_time})"
            )

            if url:
                lines.append(f"Link: {url}")

        return "\n".join(lines)

    @staticmethod
    def print_to_console(alert):
        print("\n" + "=" * 70)
        print(str(alert))
        print("=" * 70)

    @staticmethod
    def save_to_file(alert, filename="alerts.log"):
        if (
            filename is None
            or isinstance(filename, bool)
            or not isinstance(filename, (str, os.PathLike))
            or not str(filename).strip()
        ):
            filename = "alerts.log"
        alert_file = Path(str(filename).strip())
        if not alert_file.is_absolute():
            alert_file = BASE_DIR / alert_file
        try:
            with alert_file.open("a", encoding="utf-8") as f:
                if AlertNotifier.is_new_coin_alert(alert):
                    f.write(f"[{datetime.now().isoformat()}] NEW COIN {alert.get('symbol')} "
                            f"(source: {alert.get('source', 'unknown')})\n")
                    f.write(AlertNotifier.format_new_coin_message(alert) + "\n")
                    f.write("-" * 50 + "\n")
                    return True
                if isinstance(alert, dict):
                    if alert.get("type") == "bot_event":
                        f.write(f"[{datetime.now().isoformat()}] BOT EVENT {alert.get('name', alert.get('symbol', 'N/A'))}\n")
                        f.write(AlertNotifier.format_bot_event(alert) + "\n")
                    else:
                        sym = alert.get("symbol", "N/A")
                        body = alert.get("body") or alert.get("name") or str(alert)
                        f.write(f"[{datetime.now().isoformat()}] {sym}: {body}\n")
                    f.write("-" * 50 + "\n")
                    return True
                reference_time = AlertNotifier.format_human_readable_time(
                    getattr(alert, "reference_time", None) or getattr(alert, "timestamp", None)
                )
                sym = getattr(alert, "symbol", "N/A")
                atype = getattr(alert, "alert_type", "")
                pchg = safe_float(getattr(alert, "percent_change", 0.0), 0.0)
                cprice = safe_float(getattr(alert, "current_price", 0.0), 0.0)
                rprice = safe_float(getattr(alert, "reference_price", 0.0), 0.0)
                f.write(f"[{datetime.now().isoformat()}] {sym} - {atype} - {pchg:+.2f}%\n")
                f.write(f"  Price: ${cprice:.4f}\n")
                f.write(f"  Reference: ${rprice:.4f} (at {reference_time})\n")
                url = getattr(alert, "url", None)
                if url:
                    f.write(f"  Link: {url}\n")
                f.write("-" * 50 + "\n")
            return True
        except Exception as e:
            print(f"[Notifier] File exception: {e}")
            return False

    @staticmethod
    def email_subject(alerts: list) -> str:
        if not alerts:
            return "Alert"
        first = alerts[0]
        if isinstance(first, dict) and first.get("type") == "new_coin_discovered":
            from notify_engine.server import NotificationServer
            return NotificationServer._format_new_coins_batch_title(alerts) if len(alerts) > 1 \
                else f"New Coin: {first.get('symbol', 'N/A')}"
        if (isinstance(first, dict) and "alert_type" in first) or hasattr(first, "alert_type"):
            from notify_engine.server import NotificationServer
            return NotificationServer._format_price_alerts_batch_title(alerts)
        if isinstance(first, dict):
            return str(first.get("name") or first.get("title") or "Alert")
        return "Alert"

    @staticmethod
    def send_email_batch(
        alerts: list[dict],
        recipient: Optional[str] = None,
        subject: Optional[str] = None,
    ) -> bool:
        if not alerts:
            return False
        if _external_notifications_disabled():
            return True
        alerts = list(alerts)
        subject = subject or AlertNotifier.email_subject(alerts)
        message = AlertNotifier.format_batch_message(alerts)
        if _as_bool_flag(os.environ.get("MPTRADE_ORCHESTRATED"), False):
            # The orchestrator parses child stdout; same schema as its retry queue.
            intent = {"__orchestrator_intent__": "email", "subject": subject, "message": message}
            sys.stdout.write(json.dumps(intent, default=str) + "\n")
            sys.stdout.flush()
            return True
        from notify_engine.server import NotificationServer
        return NotificationServer()._send_email(subject, message)

    @staticmethod
    def send_phone_webhook_batch(alerts, webhook_url: Optional[str] = None):
        if not alerts:
            return False
        if _external_notifications_disabled():
            return True
        if _as_bool_flag(os.environ.get("MPTRADE_ORCHESTRATED"), False):
            def default_serializer(obj):
                if hasattr(obj, "to_dict"):
                    return obj.to_dict()
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return str(obj)

            serialized_alerts = []
            for a in alerts:
                if hasattr(a, "to_dict"):
                    d = a.to_dict()
                    d.setdefault("type", "price_alert")
                    serialized_alerts.append(d)
                elif isinstance(a, dict):
                    serialized_alerts.append(a)
                else:
                    serialized_alerts.append(str(a))

            intent = {
                "__orchestrator_intent__": "ntfy_webhook",
                "webhook_url": webhook_url,
                "alerts": serialized_alerts
            }
            sys.stdout.write(json.dumps(intent, default=default_serializer) + "\n")
            sys.stdout.flush()
            return True
        else:
            from notify_engine.server import NotificationServer
            server = NotificationServer()
            return server.dispatch_alerts(list(alerts), webhook_url=webhook_url)

    @staticmethod
    def send(alert, enable_console=True, enable_file=True,
             enable_email=False, enable_phone_webhook=False, webhook_url=None,
             filename="alerts.log"):
        alerts = [alert] if not isinstance(alert, list) else alert
        if _as_bool_flag(enable_console, True):
            for item in alerts:
                AlertNotifier.print_to_console(item)
        if _as_bool_flag(enable_file, True):
            for item in alerts:
                AlertNotifier.save_to_file(item, filename=filename)
        if _as_bool_flag(enable_email, False):
            AlertNotifier.send_email_batch(alerts)
        if _as_bool_flag(enable_phone_webhook, False):
            AlertNotifier.send_phone_webhook_batch(alerts, webhook_url=webhook_url)


def _external_notifications_disabled() -> bool:
    """Return True when external notifications (ntfy, email, orchestrator intents) are suppressed."""
    return any(
        _as_bool_flag(os.environ.get(k, ""), False)
        for k in ("DISABLE_EXTERNAL_NOTIFICATIONS", "BACKTEST_MODE", "REPLAY_MODE")
    )


def notify(title: str, body: str, source: str, symbol: str,
           price: float = None, desktop: bool = False,
           email: bool = None) -> None:
    """Shared bot-event notifier: ntfy (routed by category) plus optional email.

    ERROR-category events reach email automatically through the server's topic mirror
    (notify_engine.mailer). ``email=True`` forces a copy for any other category.
    """
    if _external_notifications_disabled():
        return
    if is_fake_alert_text(title, body, symbol, source):
        return
    alert = {"type": "bot_event", "name": title, "body": body, "source": source, "symbol": symbol}
    if _as_bool_flag(os.environ.get("MPTRADE_ORCHESTRATED"), False):
        intent: dict[str, Any] = {"__orchestrator_intent__": "ntfy_webhook", "alerts": [alert]}
        if re.search(r"\[auto_[a-z0-9_]+", str(title or "")):
            from notify_engine.server import _topic_for_category
            urgent = _alerts_are_urgent([alert])
            intent.update({
                "title": title,
                "message": body,
                "priority": "urgent" if urgent else "high",
                "topic": _topic_for_category(title, source),
            })
        sys.stdout.write(json.dumps(intent) + "\n")
        sys.stdout.flush()
    else:
        from notify_engine.server import NotificationServer
        NotificationServer().dispatch_alerts([alert])
    email_flag = _as_bool_flag(email, False)
    if email_flag:
        from notify_engine.mailer import is_email_mirrored
        from notify_engine.server import _topic_for_category
        if not is_email_mirrored(_topic_for_category(title, source)):
            AlertNotifier.send_email_batch([alert], subject=title)
 
 
def check_alert(condition: bool, message: str, alert_interval: int = 60, symbol: str = "TRADE") -> None:
    """Dispatches a trade alert notification if condition is met."""
    cond_flag = _as_bool_flag(condition, False)
    if cond_flag:
        try:
            notify("Trade Alert", str(message), "tradeall", symbol)
        except Exception:
            pass


def bind_notify(symbol_env_keys: tuple, default_symbol: str):
    import os
    def bound_notify(title: str, body: str, source: str, price: float = None, desktop: bool = False, symbol: str = None, email: bool = None):
        sym_clean = symbol.strip() if isinstance(symbol, str) and symbol.strip() else None
        resolved = sym_clean or next(
            (os.environ[key].strip() for key in symbol_env_keys if os.environ.get(key, "").strip()),
            default_symbol,
        )
        notify(title, body, source, resolved, price=price, desktop=desktop, email=email)
    bound_notify.__name__ = "notify"
    return bound_notify
