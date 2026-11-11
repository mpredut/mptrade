# alert_notifiers.py
from __future__ import annotations

import hashlib
import os
import re
import sys
sys.modules.setdefault("alertnotifiers", sys.modules[__name__])
import time
from datetime import datetime, timezone
from pathlib import Path
from state_io import encode_json_payload, read_json_dict, save_json_state
from typing import Any, Optional

# Import your modules
from lock import FileLock
from botcore import (
    INVALID_SCALAR_STRINGS,
    _is_unmodified_root_key,
    _parse_bool_flag,
    exact_pos_int,
    float_env,
    format_notify_price,
    format_notify_qty,
    int_env,
    is_cache_fresh,
    is_live_network_backfill_enabled,
    normalize_epoch_sec,
    obj_attr,
    parse_dotenv,
    safe_float,
    safe_int,
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
_GUARD_ALERT_TITLE_MARKERS = (
    "🛡", "🛑", "GUARD", "VETO", "SHADOW", "BLOCKED", "REFUSED",
    "SMART-SL", "POST-STOP", "STOP-LOSS", "STOP_LOSS",
    "TRAILING", "QUARANTINE", "GIVE UP",
)
_GUARD_ALERT_SOURCES = (
    "order_guard", "guard", "assetguardian", "macro_shadow",
    "smart_stop_loss", "post_stop_guard",
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
    "BUY", "SELL", "FILLED", "TREND_ENTRY", "ADOPT", "EXECUTION", "OPEN ", "CLOSE ", "📝", "🎉",
)
_FAKE_ALERT_MARKERS = (
    "ZZZFAKE", "FAKEUSD", "TESTPAIR", "TSTX", "FAKE_VENUE", "ZZZ",
)
_KNOWN_OWNERS = (
    "MANUAL", "TRADEALL", "MONITORORDER", "RTRADE", "MONITORTRADES", "ORDER_RETRY",
    "ASSETGUARDIAN", "AI_EXECUTOR", "SPOT_DCA", "TRAILING_STOP", "ORDER_MANAGER", "HL_BOT",
)
_OWNER_PAT = "|".join(_KNOWN_OWNERS)
_VENUE_PAT = "BINANCE|KRAKEN|HYPERLIQUID|HL|T212|TRADING212"
_OWNER_TAG_RE = re.compile(
    rf"\[(?:(?:{_VENUE_PAT}):)?({_OWNER_PAT})\]"
    rf"|\[({_VENUE_PAT})\]"
    rf"|\((?:(?:{_VENUE_PAT}):)?({_OWNER_PAT})\)",
    flags=re.IGNORECASE,
)
_GATE_SUPPRESSED_REASONS = (
    "duplicate", "similar_eliminated", "queue_full_dropped", "topic_queue_full_dropped",
)
_GATE_ACCEPTED_REASONS = ("buffered_for_bulk", *_GATE_SUPPRESSED_REASONS)


def sanitize_single_line(val: Any, default: str = "") -> str:
    """Coerce ``val`` to a single-line stripped string (rejecting booleans/None) or return ``default``."""
    if isinstance(val, bool) or val is None:
        return default
    cleaned = re.sub(r"[\r\n]+", " ", str(val)).strip()
    return cleaned if cleaned else default


def strip_owner_tags(text: Any) -> str:
    """Remove bot/manual owner bracket or paren tags from an alert title."""
    s = "" if isinstance(text, bool) or text is None else str(text)
    return _OWNER_TAG_RE.sub("", s)


def format_owner_tags(
    owner: Any,
    *,
    ignore: tuple[str, ...] = ("unspecified", "unknown", "order_retry"),
) -> tuple[str, str]:
    """Return ``(owner_tag, owner_body)`` as ``(' [owner]', ' (owner)')`` when valid, else ``('', '')``."""
    if owner is None or isinstance(owner, bool):
        return "", ""
    s = str(owner).strip()
    if not s or s.lower() in ignore or s.lower().startswith("alt:"):
        return "", ""
    return f" [{s}]", f" ({s})"


def format_trade_venue_tag(
    venue: Any = "binance",
    owner: Any = None,
    *,
    ignore: tuple[str, ...] = ("unspecified", "unknown", "order_retry"),
) -> str:
    """Return ``'[venue:owner]'`` when owner is valid, else ``'[venue]'``."""
    v = "" if isinstance(venue, bool) or venue is None else str(venue).strip().lower()
    _, owner_body = format_owner_tags(owner, ignore=ignore)
    o = owner_body.strip(" ()").lower() if owner_body else ""
    if v and o:
        return f"[{v}:{o}]"
    if v:
        return f"[{v}]"
    return f"[{o}]" if o else ""


def format_trade_notification(
    status: str,
    side: Any,
    symbol: Any,
    price_str: Any,
    qty: Any,
    val: Any,
    quote: Any,
    order_id: Any,
    *,
    venue: Any = "binance",
    owner: Any = None,
    fills_count: int = 1,
    cum_qty: Optional[float] = None,
    cum_val: Optional[float] = None,
    cum_fills: int = 0,
) -> tuple[str, str]:
    """Build canonical ``(title, body)`` for TRADES notifications:
    Title: ``BUY/SELL <SYMBOL>@<PRICE> [venue:module]``
    Body:  ``FILLED/PLACED ~<VAL> <QUOTE> (qty: <QTY>, order: <ID>)``
    """
    side_s = str(side or "").strip().upper() or "ORDER"
    sym_s = str(symbol or "").strip().upper()
    vtag = format_trade_venue_tag(venue, owner)
    title = f"{side_s} {sym_s}@{price_str} {vtag}".strip()
    st_s = str(status or "FILLED").strip().upper()
    qf = safe_pos_float(qty, default=0.0) or 0.0
    vf = safe_pos_float(val, default=0.0) or 0.0
    q_ccy = str(quote or "USDC").strip().upper() or "USDC"
    fills_part = f", {fills_count} fills" if fills_count > 1 else ""
    cum_sfx = ""
    if cum_fills > fills_count and cum_qty is not None and cum_val is not None:
        cum_sfx = f" [total: ~{cum_val:.2f} {q_ccy}, qty: {cum_qty:.4f}, {cum_fills} fills]"
    body = f"{st_s} ~{vf:.2f} {q_ccy} (qty: {qf:.4f}, order: {order_id}{fills_part}){cum_sfx}"
    return title, body


def make_bot_event_alert(
    title: Any,
    body: Any,
    source: Any = "system",
    symbol: Any = "SYSTEM",
    **extra: Any,
) -> dict[str, Any]:
    """Construct a canonical bot_event alert dictionary."""
    return {
        "type": "bot_event",
        "name": title,
        "body": body,
        "source": source,
        "symbol": symbol,
        **extra,
    }


def is_fake_alert_text(*parts: Any) -> bool:
    """Return True when any part contains synthetic/test instrument markers."""
    joined = " ".join("" if isinstance(p, bool) or p is None else str(p) for p in parts).upper()
    return any(marker in joined for marker in _FAKE_ALERT_MARKERS)


def format_ntfy_url(topic: Any) -> str:
    """Format an ntfy topic name or full HTTP(S) URL into a canonical ntfy endpoint URL."""
    if not isinstance(topic, str) or isinstance(topic, bool):
        return ""
    clean = topic.split("#", 1)[0].strip()
    if not clean:
        return ""
    if clean.startswith(("http://", "https://")):
        rest = clean.split("://", 1)[1].rstrip("/")
        return clean if "/" in rest and rest.split("/", 1)[1].strip() else ""
    clean_topic = clean.lstrip("/")
    return f"https://ntfy.sh/{clean_topic}" if clean_topic else ""


def _positive_int_env(name: str, default: int) -> int:
    val = int_env(name)
    return val if (val is not None and val >= 0) else default


def _resolve_gate_now(now: Optional[float]) -> float:
    nf = safe_pos_float(now, default=None)
    if nf is not None and nf < 32503680000.0:
        return nf
    return time.time()


def _delivery_state_path() -> Path:
    configured = os.environ.get("NOTIFICATION_STATE_FILE")
    return Path(configured).expanduser() if configured else BASE_DIR / "logs/notification_delivery_state.json"


def _load_delivery_state(path: Path, today: str) -> dict:
    state = read_json_dict(path, default={}, recover_previous=True)
    if not isinstance(state, dict) or not isinstance(state.get("channels"), dict):
        return {"schema_version": 1, "date_utc": today, "channels": {}}
    if state.get("date_utc") != today:
        state["schema_version"] = 1
        state["date_utc"] = today
        for ch in state["channels"].values():
            if isinstance(ch, dict):
                ch["sent"] = 0
                ch["budget_warning_sent"] = False
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
        title = strip_owner_tags(str(alert.get("name") or alert.get("symbol") or alert.get("title") or "").upper())
        source = str(alert.get("source") or "").lower()
        if any(m in title for m in _GUARD_ALERT_TITLE_MARKERS) or any(s in source for s in _GUARD_ALERT_SOURCES):
            return True
    return False


def _is_error_or_ops_alert(alerts: list[Any]) -> bool:
    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        title_no_owner = strip_owner_tags(str(alert.get("name") or alert.get("title") or "").upper())
        source = str(alert.get("source") or "").lower()
        if (
            any(m in title_no_owner for m in _ERROR_AND_OPS_TITLE_MARKERS)
            or any(s in source for s in _ERROR_AND_OPS_SOURCES)
            or alert.get("type") == "email"
        ):
            return True
    return False


_REDUNDANT_BULK_LINE_RE = re.compile(
    r"^(?:Mode:\s*\S+\s*·\s*Status:\s*Simulated/Observed"
    r"|Action:\s*[^·]+·\s*Bias:\s*[^(]+\(([^)]+)\)"
    r"|Brake:\s*[^·]+·\s*Threat:\s*[^(]+\(([^)]+)\)"
    r"|Risk:\s*(.+))$",
    flags=re.IGNORECASE,
)
_LAST_LOGS_RE = re.compile(r"(?is)\n*\s*last logs:.*")
_AUTO_TOPIC_TAG_RE = re.compile(r"\[auto_[a-z0-9_]+")
_BRACKET_GROUP_RE = re.compile(r"\[([^\[\]]+)\]")
_BODY_VENUE_RE = re.compile(r"\bVenue:\s*([A-Za-z0-9_]+)", re.IGNORECASE)
_LEADING_BRACKET_RE = re.compile(r"^\[[^\]]+\]\s*")
_SIMILARITY_SUBS = (
    (re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?\s*"), ""),
    (re.compile(r"^\d{2}:\d{2}:\d{2}\s*"), ""),
    (re.compile(r"0x[0-9a-fA-F]+"), "<HEX>"),
    (re.compile(r"[+-]?[\$€~]?\d+(?:,\d{3})*(?:\.\d+)?(?:%|ms|s|m|h|d|x|bps)?"), "<N>"),
    (re.compile(r"\s+"), " "),
)
_IDENTITY_NAME_SUBS = (
    (re.compile(r"[+-]?\d+(?:\.\d+)?%"), "<PCT>"),
    (re.compile(r"@\s*~?\$?\d+(?:\.\d+)?"), "@ <PX>"),
    (re.compile(r"\(\d+(?:\.\d+)?s\)"), "(<DUR>)"),
)


def _normalize_text_for_similarity(text: str) -> str:
    """Normalize volatile numbers, timestamps, PIDs, hex IDs, prices, and percentages."""
    s = _LAST_LOGS_RE.sub("", str(text or "")).strip()
    for pat, repl in _SIMILARITY_SUBS:
        s = pat.sub(repl, s)
    return s.strip().lower()


_PRIO_RANK = {"urgent": 3, "high": 2, "default": 1, "low": 0}
_MAX_FINGERPRINT_HISTORY = 200


def _canonical_topic_key(topic: Any) -> str:
    """Normalize a topic name or URL into a canonical per-topic queue key."""
    if isinstance(topic, bool) or topic is None:
        return "_default_"
    s = str(topic).strip().rstrip("/")
    if "/" in s:
        s = s.rsplit("/", 1)[-1].strip()
    return s or "_default_"


def _config_notification_int(key: str, default: int) -> int:
    """Resolve a notification interval from env/config.env, defaulting to 0 when unconfigured in non-live mode."""
    raw = os.environ.get(key)
    has_explicit_env = raw is not None and str(raw).strip() != "" and not _is_unmodified_root_key(key)
    if has_explicit_env:
        return _positive_int_env(key, default)
    dedup_s = float_env("NOTIFICATION_DEDUP_SECONDS")
    if dedup_s == 0.0 or not is_live_network_backfill_enabled():
        return 0
    if raw is not None and str(raw).strip() != "":
        return _positive_int_env(key, default)
    cfg = parse_dotenv(str(BASE_DIR / "config.env"))
    cfg_val = safe_int(cfg.get(key), default=None)
    return cfg_val if (cfg_val is not None and cfg_val >= 0) else default


def _topic_bulk_seconds() -> int:
    """Return the per-topic batching/bulk interval in seconds (default 300s = 5 min)."""
    raw_topic = os.environ.get("NOTIFICATION_TOPIC_BULK_SECONDS")
    if (
        raw_topic is not None
        and str(raw_topic).strip() != ""
        and not _is_unmodified_root_key("NOTIFICATION_TOPIC_BULK_SECONDS")
    ):
        return _positive_int_env("NOTIFICATION_TOPIC_BULK_SECONDS", 300)
    raw_gate = os.environ.get("NOTIFICATION_GATE_SECONDS")
    if (
        raw_gate is not None
        and str(raw_gate).strip() != ""
        and not _is_unmodified_root_key("NOTIFICATION_GATE_SECONDS")
    ):
        return min(300, _positive_int_env("NOTIFICATION_GATE_SECONDS", 300))
    return _config_notification_int("NOTIFICATION_TOPIC_BULK_SECONDS", 300)


def _queue_max_items() -> int:
    """Maximum total notifications buffered across all topics and retry queues."""
    return max(1, _positive_int_env("NOTIFICATION_QUEUE_MAX_ITEMS", 50))


def _normalize_channel_state(state: dict, channel: str, now_ts: float) -> dict:
    """Normalize and bound per-channel delivery state so state files remain compact."""
    raw_ch = state["channels"].get(channel)
    if not isinstance(raw_ch, dict):
        raw_ch = {}
        state["channels"][channel] = raw_ch
    cutoff = now_ts - 2 * 24 * 60 * 60
    last_raw = raw_ch.get("last")
    valid_last = [
        (str(k), vf)
        for k, v in (last_raw.items() if isinstance(last_raw, dict) else ())
        if cutoff <= (vf := safe_float(v, -1.0)) <= now_ts
    ]
    if len(valid_last) > _MAX_FINGERPRINT_HISTORY:
        valid_last = sorted(valid_last, key=lambda kv: kv[1])[-_MAX_FINGERPRINT_HISTORY:]
    raw_ch["last"] = dict(valid_last)

    topic_raw = raw_ch.get("topic_last_ts")
    raw_ch["topic_last_ts"] = {
        str(k): vf
        for k, v in (topic_raw.items() if isinstance(topic_raw, dict) else ())
        if cutoff <= (vf := safe_float(v, -1.0)) <= now_ts
    }
    raw_bulk = raw_ch.get("pending_bulk")
    raw_ch["pending_bulk"] = (
        [
            it
            for it in raw_bulk
            if isinstance(it, dict) and cutoff <= safe_float(it.get("ts"), now_ts) <= now_ts
        ]
        if isinstance(raw_bulk, list)
        else []
    )
    lg = safe_float(raw_ch.get("last_gate_ts", 0.0), 0.0)
    raw_ch["last_gate_ts"] = 0.0 if (lg < 0.0 or lg > now_ts) else lg
    return raw_ch


def _is_ntfy_channel_blocked(channel: str, channel_state: dict, now_ts: float) -> bool:
    if channel != "ntfy" or not channel_state.get("blocked"):
        return False
    blocked_until = safe_float(channel_state.get("blocked_until", 0.0), 0.0)
    if blocked_until > 0 and now_ts >= blocked_until:
        channel_state["blocked"] = False
        return False
    return True


_KNOWN_NOTIFY_VENUES = frozenset({"binance", "kraken", "hyperliquid", "t212", "bybit", "coinbase", "okx"})
_BULK_TITLE_STOPWORDS = frozenset({
    "a", "an", "and", "are", "at", "by", "for", "from", "in", "is", "of", "on", "or", "the", "to", "with",
})
_TOPIC_CATEGORY_SPECS = (
    ("deadman", "DEADMAN", 3),
    ("error", "ERROR", 3),
    ("guard", "GUARD", 2),
    ("trade", "TRADES", 2),
    ("server", "SERVER", 1),
    ("macro", "MACRO", 1),
    ("price", "PRICE", 0),
)


def _topic_category_label(topic: str) -> str:
    tl = str(topic or "").lower()
    return next((label for key, label, _ in _TOPIC_CATEGORY_SPECS if key in tl), "")


def _topic_tier(topic: str) -> int:
    tl = str(topic or "").lower()
    return next((tier for key, _, tier in _TOPIC_CATEGORY_SPECS if key in tl), 0)


def synthesize_bulk_notifications(items: list[dict]) -> tuple[str, str, str, str]:
    """Synthesize and concatenate multiple distinct notifications into a single bulk message."""
    if not items:
        return ("Alert", "", "default", "")
    best_priority = max(
        (str(it.get("priority") or "default") for it in items),
        key=lambda p: _PRIO_RANK.get(p, 1),
    )
    best_topic = str(items[0].get("topic") or "")
    best_tier = _topic_tier(best_topic)
    for it in items[1:]:
        t = str(it.get("topic") or "")
        tier = _topic_tier(t)
        if tier > best_tier:
            best_topic = t
            best_tier = tier

    if len(items) == 1:
        it = items[0]
        cnt = exact_pos_int(it.get("count", 1), 1)
        raw_title = str(it.get("title") or "Alert")
        out_title = f"{raw_title} (×{cnt})" if cnt > 1 else raw_title
        out_body = str(it.get("body") or "")
        return (out_title, out_body, best_priority, best_topic)

    body_lines = []
    for idx, it in enumerate(items, 1):
        cnt = exact_pos_int(it.get("count", 1), 1)
        t = str(it.get("title") or "Alert").strip()
        repeat_tag = f" (×{cnt})" if cnt > 1 else ""
        cleaned = _LAST_LOGS_RE.sub("", str(it.get("body") or "")).strip()
        raw_lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        if not raw_lines:
            compact_body = ""
        else:
            lines: list[str] = []
            prefix_tag = ""
            for l_idx, ln in enumerate(raw_lines):
                if l_idx == 0 and (m := _REDUNDANT_BULK_LINE_RE.match(ln)):
                    inner = (m.group(1) or m.group(2) or "").strip()
                    risk_only = (m.group(3) or "").strip()
                    if inner:
                        prefix_tag = f"[{inner}]"
                    elif risk_only:
                        prefix_tag = f"[Risk: {risk_only}]"
                    continue
                lines.append(ln)
            if not lines:
                compact_body = prefix_tag or raw_lines[0]
            else:
                compact_body = " | ".join(lines[:4])
                if prefix_tag:
                    compact_body = f"{prefix_tag} {compact_body}"
                if len(compact_body) > 900:
                    cut = compact_body[:900].rsplit(" ", 1)[0]
                    compact_body = (cut if len(cut) >= 450 else compact_body[:900]) + "..."
        if compact_body and compact_body.lower() != t.lower():
            body_lines.append(f"{idx}. {t}{repeat_tag}: {compact_body}")
        else:
            body_lines.append(f"{idx}. {t}{repeat_tag}")

    item_venues: list[str] = []
    item_tags: list[list[str]] = []
    item_tokens: list[list[str]] = []
    categories: list[str] = []

    for it in items:
        raw_t = str(it.get("title") or "Alert").strip()
        cat = _topic_category_label(str(it.get("topic") or ""))
        if cat and cat not in categories:
            categories.append(cat)

        venue = ""
        tags: list[str] = []
        for match in _BRACKET_GROUP_RE.finditer(raw_t):
            inner = match.group(1).strip()
            if not inner:
                continue
            if ":" in inner:
                v_part, m_part = inner.split(":", 1)
                if v_part.strip().lower() in _KNOWN_NOTIFY_VENUES:
                    venue = v_part.strip().lower()
                    if m_part.strip():
                        tags.append(m_part.strip())
                    continue
            if inner.lower() in _KNOWN_NOTIFY_VENUES:
                venue = inner.lower()
            else:
                tags.append(inner)
        if not venue:
            body_venue_m = _BODY_VENUE_RE.search(str(it.get("body") or ""))
            if body_venue_m and body_venue_m.group(1).lower() in _KNOWN_NOTIFY_VENUES:
                venue = body_venue_m.group(1).lower()
        item_venues.append(venue)
        item_tags.append(tags)

        stripped = _BRACKET_GROUP_RE.sub(" ", raw_t)
        stripped = re.sub(r"@\s*\$?[0-9][0-9.,]*", " ", stripped)
        tokens = [
            tok for tok in re.findall(r"[A-Za-z][A-Za-z0-9_]*", stripped)
            if tok.lower() not in _BULK_TITLE_STOPWORDS and len(tok) > 1
        ]
        item_tokens.append(tokens)

    non_empty_venues = [v for v in item_venues if v]
    common_venue = non_empty_venues[0] if non_empty_venues and len(set(non_empty_venues)) == 1 else ""

    common_tag = ""
    if item_tags and all(item_tags):
        first_tags = item_tags[0]
        shared = [
            tg for tg in first_tags
            if all(any(tg.lower() == other.lower() for other in tg_list) for tg_list in item_tags[1:])
        ]
        if shared:
            common_tag = shared[0]
        elif all(
            any(
                other.lower() in ("ai_executor", "auto_cryptotrade_executor", "auto_sharetrade_executor", "auto_trade_executor")
                for other in tg_list
            )
            for tg_list in item_tags
        ):
            common_tag = "ai_executor"
        else:
            auto_prefixes = []
            for tg_list in item_tags:
                auto_match = next((tg for tg in tg_list if tg.lower().startswith("auto_")), "")
                parts = auto_match.lower().split("_")
                auto_prefixes.append("_".join(parts[:2]) if len(parts) >= 2 else "")
            if auto_prefixes and all(p and p == auto_prefixes[0] for p in auto_prefixes):
                common_tag = auto_prefixes[0]

    if common_venue and common_tag:
        bracket_str = f"[{common_venue}:{common_tag}]"
    elif common_venue:
        bracket_str = f"[{common_venue}]"
    elif common_tag:
        bracket_str = f"[{common_tag}]"
    else:
        bracket_str = ""

    common_words: list[str] = []
    if item_tokens and all(item_tokens):
        seen_lower: set[str] = set()
        for tok in item_tokens[0]:
            tl = tok.lower()
            if tl in seen_lower:
                continue
            if all(any(tl == other.lower() for other in tok_list) for tok_list in item_tokens[1:]):
                common_words.append(tok)
                seen_lower.add(tl)

    if common_words:
        core_label = " ".join(common_words[:4])
    elif categories:
        core_label = " / ".join(categories)
    else:
        core_label = _topic_category_label(best_topic) or "Alerts"

    raw_summary = f"{core_label} {bracket_str}" if (bracket_str and core_label) else (bracket_str or core_label)
    summary_subject = (raw_summary or "").strip() or _topic_category_label(best_topic) or "Alerts"
    bulk_title = f"📦 Bulk ({len(items)}): {summary_subject}"
    bulk_body = "\n\n".join(body_lines)
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
            norm_name = name
            for pat, repl in _IDENTITY_NAME_SUBS:
                norm_name = pat.sub(repl, norm_name)
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
        (encode_json_payload(_alert_identity(alert), sort_keys=True, default=str) for alert in alerts),
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
    if any(m in titles for m in ("AUTO_MACRO_", "MACRO_SENTIMENT", "MACRO_GEOPOLITICAL")):
        return _config_notification_int("NOTIFICATION_MACRO_DEDUP_SECONDS", 3600)
    if _is_guard_alert(alerts):
        return _positive_int_env("NOTIFICATION_GUARD_DEDUP_SECONDS", 30 * 60)
    if _is_error_or_ops_alert(alerts):
        return _positive_int_env(
            "NOTIFICATION_ERROR_DEDUP_SECONDS",
            _positive_int_env("NOTIFICATION_URGENT_DEDUP_SECONDS", 30 * 60),
        )
    if any(marker in titles for marker in _TRADE_MARKERS):
        return 0  # Real trades must never be suppressed by deduplication
    if any(m in titles for m in ("DISPONIBIL", "AVAILABLE", "LISTED")):
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
    """Bounded Producer-Consumer notification gate:

    1. Deduplicates / merges structurally similar notifications.
    2. Batches same-topic messages arriving within NOTIFICATION_TOPIC_BULK_SECONDS (default 5m),
       strictly bounded by NOTIFICATION_TOPIC_BULK_MAX_ITEMS and NOTIFICATION_QUEUE_MAX_ITEMS.
    3. Routes every outgoing single or bulk message through the central single-gate dispatcher
       (NOTIFICATION_GATE_SECONDS, default 10m).
    """
    now_ts = _resolve_gate_now(now)
    urgent = _parse_bool_flag(urgent, False)
    today = datetime.fromtimestamp(now_ts, timezone.utc).date().isoformat()
    path = _delivery_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    alerts_list = alerts if alerts else [make_bot_event_alert(title, body, "common_gate")]
    t_key = _canonical_topic_key(topic)
    fingerprint = _delivery_fingerprint(alerts_list)
    norm_title = _normalize_text_for_similarity(title)
    bare_title = _LEADING_BRACKET_RE.sub("", norm_title).strip()
    is_generic_title = bare_title in (
        "", "alert", "os alert", "trade alert", "price alert",
        "critical errors", "alerts (<n>)",
    )
    is_pure_trade = (
        any(m in strip_owner_tags(str(title or "")).upper() for m in _TRADE_MARKERS)
        and not _is_guard_alert(alerts_list)
        and not _is_error_or_ops_alert(alerts_list)
    )
    first_line = next((ln.strip() for ln in str(body or "").splitlines() if ln.strip()), "")
    norm_body_head = _normalize_text_for_similarity(first_line)[:120]
    if is_pure_trade:
        raw_t = re.sub(r"\s+", " ", str(title or "")).strip().lower()
        raw_b = re.sub(r"\s+", " ", first_line).strip().lower()[:120]
        sig_raw = f"sim|{topic}|{raw_t}|{raw_b}"
    elif is_generic_title:
        sig_raw = f"sim|{topic}|{norm_title}|{norm_body_head}"
    else:
        sig_raw = f"sim|{topic}|{norm_title}"
    sim_key = hashlib.sha256(sig_raw.encode("utf-8")).hexdigest()
    gate_sec = _config_notification_int("NOTIFICATION_GATE_SECONDS", 180)
    topic_bulk_sec = _topic_bulk_seconds()
    sim_window = max(gate_sec, topic_bulk_sec, _dedup_seconds(alerts_list, urgent))

    with FileLock(path.with_suffix(path.suffix + ".lock")):
        state = _load_delivery_state(path, today)
        channel_state = _normalize_channel_state(state, channel, now_ts)
        if _is_ntfy_channel_blocked(channel, channel_state, now_ts):
            return False, "provider_daily_limit", False, None

        last_map: dict[str, float] = channel_state["last"]
        topic_last: dict[str, float] = channel_state["topic_last_ts"]
        pending_bulk: list[dict] = channel_state["pending_bulk"]

        # 1. Coalesce if already queued in pending_bulk for the same topic
        for item in pending_bulk:
            if _canonical_topic_key(item.get("topic")) == t_key and (
                item.get("sim_key") == sim_key or item.get("fingerprint") == fingerprint
            ):
                item["count"] = exact_pos_int(item.get("count", 1), 1) + 1
                last_map[fingerprint] = now_ts
                last_map[sim_key] = now_ts
                _save_delivery_state(path, state)
                return False, "similar_eliminated", False, None

        # 2. Suppress if seen within deduplication / similarity window
        for k, reason in ((fingerprint, "duplicate"), (sim_key, "similar_eliminated")):
            prev_f = safe_float(last_map.get(k), -1.0)
            if prev_f >= 0.0 and 0 <= now_ts - prev_f < sim_window:
                _save_delivery_state(path, state)
                return False, reason, False, None

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

        # 4. Producer-Consumer Gate Evaluation (5m per-topic batching + 10m central single-gate)
        new_item = {
            "title": str(title or "Alert")[:240],
            "body": str(body or "")[:1200],
            "priority": priority,
            "topic": topic,
            "ts": now_ts,
            "sim_key": sim_key,
            "fingerprint": fingerprint,
            "count": 1,
        }
        last_gate_ts = channel_state["last_gate_ts"]
        last_topic_ts = safe_float(topic_last.get(t_key, 0.0), 0.0)
        new_rank = _PRIO_RANK.get(str(priority or "default"), 1)
        central_gate_closed = is_cache_fresh(last_gate_ts, now_ts, gate_sec)
        topic_window_closed = is_cache_fresh(last_topic_ts, now_ts, topic_bulk_sec)
        has_due_other_topic = any(
            (other_tk := _canonical_topic_key(it.get("topic"))) != t_key
            and _PRIO_RANK.get(str(it.get("priority") or "default"), 1) >= new_rank
            and not is_cache_fresh(topic_last.get(other_tk, 0.0), now_ts, topic_bulk_sec)
            for it in pending_bulk
        )

        if central_gate_closed or topic_window_closed or has_due_other_topic:
            topic_max = max(1, _positive_int_env("NOTIFICATION_TOPIC_BULK_MAX_ITEMS", 10))
            queue_max = _queue_max_items()

            def _evict_lowest_if_full(indices, limit: int, drop_reason: str) -> Optional[str]:
                if len(indices) < limit:
                    return None
                victim_idx = min(indices, key=lambda i: _PRIO_RANK.get(str(pending_bulk[i].get("priority") or "default"), 1))
                if new_rank <= _PRIO_RANK.get(str(pending_bulk[victim_idx].get("priority") or "default"), 1):
                    return drop_reason
                evicted = pending_bulk.pop(victim_idx)
                if isinstance(evicted, dict):
                    last_map.pop(str(evicted.get("fingerprint") or ""), None)
                    last_map.pop(str(evicted.get("sim_key") or ""), None)
                return None

            topic_indices = [i for i, it in enumerate(pending_bulk) if _canonical_topic_key(it.get("topic")) == t_key]
            q_reason = (
                _evict_lowest_if_full(topic_indices, topic_max, "topic_queue_full_dropped")
                or _evict_lowest_if_full(range(len(pending_bulk)), queue_max, "queue_full_dropped")
            )
            if q_reason is None:
                pending_bulk.append(new_item)
                q_reason = "buffered_for_bulk"
                last_map[fingerprint] = now_ts
                last_map[sim_key] = now_ts
            _save_delivery_state(path, state)
            return False, q_reason, False, None

        # 5. Both Per-Topic Window and Central Single Gate are OPEN: consume & dispatch this topic now
        same_topic_items = [it for it in pending_bulk if _canonical_topic_key(it.get("topic")) == t_key]
        channel_state["pending_bulk"] = [it for it in pending_bulk if _canonical_topic_key(it.get("topic")) != t_key]
        channel_state["sent"] = sent + 1
        last_map[fingerprint] = now_ts
        last_map[sim_key] = now_ts
        channel_state["last_gate_ts"] = now_ts
        topic_last[t_key] = now_ts
        _save_delivery_state(path, state)
        if same_topic_items:
            return True, "bulk_synthesized", False, synthesize_bulk_notifications(same_topic_items + [new_item])
        return True, "reserved", False, (title, body, priority, topic)


def pop_due_bulk_batch(
    channel: str = "ntfy",
    *,
    now: Optional[float] = None,
    force: bool = False,
) -> Optional[tuple[str, str, str, str]]:
    """Consumer: pop and synthesize the next due per-topic batch once the central single-gate opens."""
    now_ts = _resolve_gate_now(now)
    force = _parse_bool_flag(force, False)
    today = datetime.fromtimestamp(now_ts, timezone.utc).date().isoformat()
    path = _delivery_state_path()
    if not path.exists() and not path.with_name(path.name + ".previous").exists():
        return None
    gate_sec = _config_notification_int("NOTIFICATION_GATE_SECONDS", 180)
    topic_bulk_sec = _topic_bulk_seconds()
    with FileLock(path.with_suffix(path.suffix + ".lock")):
        state = _load_delivery_state(path, today)
        if channel not in state["channels"]:
            return None
        channel_state = _normalize_channel_state(state, channel, now_ts)
        if _is_ntfy_channel_blocked(channel, channel_state, now_ts):
            return None
        pending_bulk: list[dict] = channel_state["pending_bulk"]
        if not pending_bulk:
            return None
        last_gate_ts = channel_state["last_gate_ts"]
        if is_cache_fresh(last_gate_ts, now_ts, gate_sec, force_refresh=force):
            return None

        topic_last: dict[str, float] = channel_state["topic_last_ts"]
        due_topics: list[str] = []
        for it in pending_bulk:
            tk = _canonical_topic_key(it.get("topic"))
            if tk not in due_topics and not is_cache_fresh(topic_last.get(tk, 0.0), now_ts, topic_bulk_sec, force_refresh=force):
                due_topics.append(tk)
        if not due_topics:
            return None

        # Dispatch highest-priority due topic first, breaking ties by earliest arrival order
        chosen_topic = max(
            due_topics,
            key=lambda tk: max(
                _PRIO_RANK.get(str(it.get("priority") or "default"), 1)
                for it in pending_bulk
                if _canonical_topic_key(it.get("topic")) == tk
            ),
        )
        items = [it for it in pending_bulk if _canonical_topic_key(it.get("topic")) == chosen_topic]
        channel_state["pending_bulk"] = [it for it in pending_bulk if _canonical_topic_key(it.get("topic")) != chosen_topic]
        channel_state["last_gate_ts"] = now_ts
        topic_last[chosen_topic] = now_ts
        channel_state["sent"] = safe_nonneg_int(channel_state.get("sent", 0), default=0) + 1
        _save_delivery_state(path, state)
        return synthesize_bulk_notifications(items)


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


def _emit_orchestrator_intent(payload: dict[str, Any], *, default: Any = None) -> bool:
    sys.stdout.write(encode_json_payload(payload, default=default) + "\n")
    sys.stdout.flush()
    return True


class AlertNotifier:

    @staticmethod
    def format_human_readable_time(value, fmt: str = "%m-%d %H:%M:%S") -> str:
        if value is None or isinstance(value, bool):
            return "N/A"
        if isinstance(value, (int, float)):
            vf = normalize_epoch_sec(value, default=None)
            if vf is None:
                return str(value)
            try:
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
    def utf8_header(value: Any, default: str = "") -> str:
        """Encode an HTTP header while preserving non-ASCII symbols.

        Headers are Latin-1, while ntfy decodes this value as UTF-8, so pass UTF-8 bytes
        through Latin-1 to preserve the original symbol.
        """
        cleaned = sanitize_single_line(value, default)
        return cleaned.encode("utf-8").decode("latin-1", "ignore") if cleaned else ""

    @staticmethod
    def _extract_price_alert_fields(alert: Any) -> tuple[str, str, float, float, float, str, Any]:
        ref_time_val = obj_attr(alert, "reference_time") or obj_attr(alert, "timestamp")
        return (
            obj_attr(alert, "symbol", "N/A"),
            obj_attr(alert, "alert_type", ""),
            safe_float(obj_attr(alert, "percent_change", 0.0), 0.0),
            safe_float(obj_attr(alert, "current_price", 0.0), 0.0),
            safe_float(obj_attr(alert, "reference_price", 0.0), 0.0),
            AlertNotifier.format_human_readable_time(ref_time_val),
            obj_attr(alert, "url"),
        )

    @staticmethod
    def format_new_coin_message(alert: dict) -> str:
        raw_price = alert.get("price")
        px_val = safe_pos_float(raw_price, default=None)
        if px_val is not None:
            price_str = f"Price: ${px_val:.4f}"
        elif (
            isinstance(raw_price, str)
            and raw_price.strip()
            and raw_price.strip().lower() not in INVALID_SCALAR_STRINGS
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

            symbol, alert_type, pct_change, current_price, ref_price, reference_time, url = (
                AlertNotifier._extract_price_alert_fields(alert)
            )
            direction = "U" if alert_type == "up" else "D"

            lines.append(
                f"{symbol}: {direction} {pct_change:+.2f}% "
                f"| C ${current_price:.4f} | R ${ref_price:.4f} "
                f"({reference_time})"
            )

            if url:
                lines.append(f"Link: {url}")

        return "\n".join(lines)

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
            now_iso = datetime.now().isoformat()
            with alert_file.open("a", encoding="utf-8") as f:
                if AlertNotifier.is_new_coin_alert(alert):
                    f.write(
                        f"[{now_iso}] NEW COIN {alert.get('symbol')} "
                        f"(source: {alert.get('source', 'unknown')})\n"
                        f"{AlertNotifier.format_new_coin_message(alert)}\n"
                    )
                elif isinstance(alert, dict):
                    if alert.get("type") == "bot_event":
                        f.write(
                            f"[{now_iso}] BOT EVENT {alert.get('name', alert.get('symbol', 'N/A'))}\n"
                            f"{AlertNotifier.format_bot_event(alert)}\n"
                        )
                    else:
                        body = alert.get("body") or alert.get("name") or str(alert)
                        f.write(f"[{now_iso}] {alert.get('symbol', 'N/A')}: {body}\n")
                else:
                    sym, atype, pchg, cprice, rprice, reference_time, url = (
                        AlertNotifier._extract_price_alert_fields(alert)
                    )
                    f.write(
                        f"[{now_iso}] {sym} - {atype} - {pchg:+.2f}%\n"
                        f"  Price: ${cprice:.4f}\n"
                        f"  Reference: ${rprice:.4f} (at {reference_time})\n"
                    )
                    if url:
                        f.write(f"  Link: {url}\n")
                f.write("-" * 50 + "\n")
            return True
        except Exception as e:
            print(f"[Notifier] File exception: {e}")
            return False

    @staticmethod
    def _format_price_alerts_batch_title(alerts: list) -> str:
        count = len(alerts)
        items = []
        symbols = []
        for a in alerts:
            sym = obj_attr(a, "symbol", "N/A")
            atype = obj_attr(a, "alert_type", "")
            dir_str = "▲" if atype == "up" else ("▼" if atype == "down" else "")
            if sym and sym != "N/A":
                if sym not in symbols:
                    symbols.append(sym)
                items.append(f"{sym} {dir_str}".strip())

        items_str = ", ".join(items)
        syms_str = ", ".join(symbols)

        if len(symbols) == 1:
            return f"{symbols[0]} ({count} alerts)"
        if items_str and len(items_str) <= 50:
            return f"Price Alerts ({count}): {items_str}"
        if syms_str and len(syms_str) <= 50:
            return f"Price Alerts ({count}): {syms_str}"
        return f"Price Alerts ({count} coins)"

    @staticmethod
    def _format_new_coins_batch_title(alerts: list) -> str:
        count = len(alerts)
        symbols = []
        for a in alerts:
            sym = obj_attr(a, "symbol", "N/A")
            if sym and sym != "N/A" and sym not in symbols:
                symbols.append(sym)
        syms_str = ", ".join(symbols)
        if syms_str and len(syms_str) <= 50:
            return f"New Coins ({count}): {syms_str}"
        return f"New Coins Discovered ({count})"

    @staticmethod
    def email_subject(alerts: list) -> str:
        if not alerts:
            return "Alert"
        first = alerts[0]
        if isinstance(first, dict) and first.get("type") == "new_coin_discovered":
            return (
                AlertNotifier._format_new_coins_batch_title(alerts)
                if len(alerts) > 1
                else f"New Coin: {first.get('symbol', 'N/A')}"
            )
        if (isinstance(first, dict) and "alert_type" in first) or hasattr(first, "alert_type"):
            return AlertNotifier._format_price_alerts_batch_title(alerts)
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
        if _parse_bool_flag(os.environ.get("MPTRADE_ORCHESTRATED"), False):
            # The orchestrator parses child stdout; same schema as its retry queue.
            return _emit_orchestrator_intent(
                {"__orchestrator_intent__": "email", "subject": subject, "message": message},
                default=str,
            )
        from notify_engine.server import NotificationServer
        return NotificationServer()._send_email(subject, message)

    @staticmethod
    def send_phone_webhook_batch(alerts, webhook_url: Optional[str] = None):
        if not alerts:
            return False
        if _external_notifications_disabled():
            return True
        if _parse_bool_flag(os.environ.get("MPTRADE_ORCHESTRATED"), False):
            return _emit_orchestrator_intent(
                {
                    "__orchestrator_intent__": "ntfy_webhook",
                    "webhook_url": webhook_url,
                    "alerts": [
                        {**{"type": "price_alert"}, **a.to_dict()}
                        if hasattr(a, "to_dict")
                        else (a if isinstance(a, dict) else str(a))
                        for a in alerts
                    ],
                },
                default=lambda obj: (
                    obj.to_dict()
                    if hasattr(obj, "to_dict")
                    else (obj.isoformat() if isinstance(obj, datetime) else str(obj))
                ),
            )
        from notify_engine.server import NotificationServer
        return NotificationServer().dispatch_alerts(list(alerts), webhook_url=webhook_url)

    @staticmethod
    def send(alert, enable_console=True, enable_file=True,
             enable_email=False, enable_phone_webhook=False, webhook_url=None,
             filename="alerts.log"):
        alerts = [alert] if not isinstance(alert, list) else alert
        if _parse_bool_flag(enable_console, True):
            for item in alerts:
                print(f"\n{'=' * 70}\n{item}\n{'=' * 70}")
        if _parse_bool_flag(enable_file, True):
            for item in alerts:
                AlertNotifier.save_to_file(item, filename=filename)
        if _parse_bool_flag(enable_email, False):
            AlertNotifier.send_email_batch(alerts)
        if _parse_bool_flag(enable_phone_webhook, False):
            AlertNotifier.send_phone_webhook_batch(alerts, webhook_url=webhook_url)


def _external_notifications_disabled() -> bool:
    """Return True when external notifications (ntfy, email, orchestrator intents) are suppressed."""
    return any(
        _parse_bool_flag(os.environ.get(k, ""), False)
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
    alert = make_bot_event_alert(title, body, source, symbol)
    if _parse_bool_flag(os.environ.get("MPTRADE_ORCHESTRATED"), False):
        intent: dict[str, Any] = {"__orchestrator_intent__": "ntfy_webhook", "alerts": [alert]}
        if _AUTO_TOPIC_TAG_RE.search(str(title or "")):
            from notify_engine.server import _topic_for_category
            urgent = _alerts_are_urgent([alert])
            intent.update({
                "title": title,
                "message": body,
                "priority": "urgent" if urgent else "high",
                "topic": _topic_for_category(title, source),
            })
        _emit_orchestrator_intent(intent)
    else:
        from notify_engine.server import NotificationServer
        NotificationServer().dispatch_alerts([alert])
    if _parse_bool_flag(email, False):
        from notify_engine.mailer import is_email_mirrored
        from notify_engine.server import _topic_for_category
        if not is_email_mirrored(_topic_for_category(title, source)):
            AlertNotifier.send_email_batch([alert], subject=title)


def bind_notify(symbol_env_keys: tuple, default_symbol: str):
    def bound_notify(title: str, body: str, source: str, price: float = None, desktop: bool = False, symbol: str = None, email: bool = None):
        sym_clean = symbol.strip() if isinstance(symbol, str) and symbol.strip() else None
        resolved = sym_clean or next(
            (os.environ[key].strip() for key in symbol_env_keys if os.environ.get(key, "").strip()),
            default_symbol,
        )
        notify(title, body, source, resolved, price=price, desktop=desktop, email=email)
    bound_notify.__name__ = "notify"
    return bound_notify


def dispatch_order_event_notification(
    symbol: Any,
    side: Any,
    order_id: Any,
    price: Any,
    quantity: Any,
    *,
    default_verb: str,
    default_icon: str,
    venue: Optional[str] = None,
    owner: Optional[str] = None,
    source: Optional[str] = None,
    order_type: Optional[str] = None,
    allow_ai_executor: bool = True,
    trade_tag: Optional[str] = None,
    notional_usd: Optional[float] = None,
    thesis: Optional[str] = None,
    title: Optional[str] = None,
    body: Optional[str] = None,
    notify_fn: Optional[Any] = None,
    include_price: bool = True,
) -> bool:
    """Dispatch a standardized order placement or execution notification."""
    sym_s = str(symbol or "TRADE").strip() or "TRADE"
    owner_norm = str(owner or "").strip().lower()
    if owner_norm == "ai_executor" and not _parse_bool_flag(allow_ai_executor, True):
        return False
    px_f = safe_pos_float(price, default=0.0) or 0.0
    qty_f = safe_pos_float(quantity, default=0.0) or 0.0
    notional_f = safe_pos_float(notional_usd, default=None)
    if notional_f is None:
        notional_f = px_f * qty_f
    side_s = str(side or "").strip().upper() or "ORDER"
    ord_type_s = str(order_type or "MARKET").strip().upper()
    venue_s = str(venue or source or "venue").strip()
    qty_s = format_notify_qty(qty_f)
    px_s = format_notify_price(px_f, comma=True)
    tag_s = str(trade_tag).strip() if trade_tag and not isinstance(trade_tag, bool) else ""
    if tag_s:
        tag_prefix = f"{tag_s} "
        owner_tag, owner_body = "", ""
    else:
        tag_prefix = ""
        owner_tag, owner_body = format_owner_tags(owner_norm)
    thesis_sfx = f"\nThesis: {thesis}" if thesis is not None and not isinstance(thesis, bool) else ""
    eff_title = title or f"{default_icon} {tag_prefix}{default_verb} {side_s} {sym_s} ${notional_f:.0f}{owner_tag}"
    eff_body = body or (
        f"Venue: {venue_s} · Qty: {qty_s} @ ${px_s} ({ord_type_s} · ${notional_f:,.0f})\n"
        f"Order ID: {order_id}{owner_body}{thesis_sfx}"
    )
    if callable(notify_fn):
        if include_price:
            notify_fn(title=eff_title, body=eff_body, symbol=sym_s, price=px_f)
        else:
            notify_fn(title=eff_title, body=eff_body, symbol=sym_s)
        return True
    notify(
        title=eff_title,
        body=eff_body,
        source=str(source or venue or owner or "llm_trader"),
        symbol=sym_s,
        price=px_f if include_price else None,
    )
    return True
