#!/usr/bin/env python3
import asyncio
import contextlib
import os
import json
import logging
import re
import threading
import time
import requests

_GATED_DISPATCH_TLS = threading.local()
from collections import defaultdict
from typing import Dict, Any, List

from botcore import (
    check_cooldown_gate,
    load_env_stack,
    obj_attr,
    required_env,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
)
from cachepaths import cache_path, normalize_venue_name
from state_io import atomic_write_text, read_json_dict, read_text_lines
from notify_engine.alertnotifiers import (
    AlertNotifier,
    _AUTO_TOPIC_TAG_RE,
    _GATE_ACCEPTED_REASONS,
    _GATE_SUPPRESSED_REASONS,
    _TRADE_MARKERS,
    _alerts_are_urgent,
    _canonical_topic_key,
    _external_notifications_disabled,
    _is_provider_daily_limit,
    _mark_provider_daily_limit,
    _queue_max_items,
    format_ntfy_url,
    is_fake_alert_text,
    make_bot_event_alert,
    pass_common_gate,
    pop_due_bulk_batch,
    sanitize_single_line,
    strip_owner_tags,
)
from notify_engine.mailer import (
    _config as _email_config,
    is_email_mirrored,
    last_error_is_auth_failure,
    send_email,
)

_MAX_QUEUE_RETRIES = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] Orchestrator: %(message)s")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_GUARD_MARKERS = (
    "🛑", "🛡", "STOP-LOSS", "STOP_LOSS", "SMART-SL", "POST-STOP", "TRAILING", "LIQUID", "CATASTROPH", "CRASH",
    "LICHID", "CATASTROF", "IMBALANCE", "DEZECHILIBR", "QUARANTINE", "SL ", " SL",
    "REFUSED", "BLOCKED", "SKIPPED",
)
_PRIMARY_ERROR_MARKERS = (
    "ERROR", "FAILED", "EXCEPTION", "CRITICAL", "GIVE UP", "CORRUPTION", "REJECTED", "✗", "UNRECOGNIZED",
)
_BILINGUAL_ERROR_MARKERS = ("ESUAT", "ERORI", "DISPARUT", "GONE")
_OPS_MARKERS = (*_PRIMARY_ERROR_MARKERS, *_BILINGUAL_ERROR_MARKERS, "MANUAL")
_EXPLICIT_ERROR_MARKERS = (*_PRIMARY_ERROR_MARKERS, *_BILINGUAL_ERROR_MARKERS, "CRASH")
_SERVER_MARKERS = (
    "CONFIG RELOAD", "CONFIG CHANGED", "CONFIG PROPOSAL", "PILOT BACKTEST", "BACKTEST:",
    "SERVER STATUS", "SERVER DOWN", "DEADMAN",
)
_PRICE_ALERT_SOURCES = ("coinmarketcap", "coingecko", "dexscreener", "pricechecker", "notifier")
_PRICE_WATCHER_BOTS = ("price_watcher", "price_watcher.py", "price_alert")
_INTERNAL_LOG_MARKERS = (
    "Failed to send ntfy alert",
    "Email delivery failed",
    "[Notifier] File exception",
    "Error flushing notification queue",
    "ntfy daily limit reached",
)
_PROVIDER_LABELS = {
    "kraken": "Kraken",
    "hyperliquid": "Hyperliquid",
    "t212": "T212",
    "binance": "Binance",
}
_STRIP_BOT_PREFIX_RE = re.compile(
    r"^\[(?:price_watcher|macro_analyzer)(?:\.py)?\]\s*",
    flags=re.IGNORECASE,
)
_TRAILING_VENUE_TAG_RE = re.compile(
    r"\[(?:binance|kraken|hyperliquid|hl|t212)(?::[a-z0-9_]+)?\]\s*$",
    flags=re.IGNORECASE,
)

def _clean_str(val: Any, default: str = "", *, strip: bool = False) -> str:
    if isinstance(val, bool) or val is None:
        return default
    s = str(val).strip() if strip else str(val)
    return s if s else default


def _prefix_delayed(val: Any, default: str = "Alert") -> str:
    s = _clean_str(val, default, strip=True)
    return s if s.startswith("[DELAYED]") else f"[DELAYED] {s}"


def _ensure_env_loaded() -> None:
    load_env_stack()


def resolve_topic(category: str) -> str:
    cat = _clean_str(category, strip=True).upper()
    if cat in ("SERVER", "SERVER_STATE", "SERVER_STATUS", "DEADMAN"):
        cat = "SERVER"
    key = f"NTFY_TOPIC_{cat}"
    if not os.environ.get(key, "").strip():
        _ensure_env_loaded()
    return required_env(key)

def _category_for_title_and_source(title: str, source: str) -> str:
    t = _clean_str(title).upper()
    t_no_owner = strip_owner_tags(t)
    s = _clean_str(source).lower()
    if "AUTO_TECHNICAL_REPAIR" in t or s == "ai_supervisor":
        if any(m in t_no_owner for m in _EXPLICIT_ERROR_MARKERS):
            return "ERROR"
        return "SERVER"
    if "MACRO" in t or "macro" in s or "geopolitical" in s:
        return "MACRO"
    if any(m in t_no_owner for m in _PRIMARY_ERROR_MARKERS) or "watchdog" in s:
        return "ERROR"
    if any(m in t_no_owner for m in _GUARD_MARKERS) or any(m in s for m in ("guard", "trail", "assetguardian", "stop")):
        return "GUARD"
    if any(m in t_no_owner for m in _SERVER_MARKERS) or "deadman" in s or "scheduled_pilot" in s:
        return "SERVER"

    # Real trades (order launched, order executed, fills) route to TRADES unless explicitly failed/errored
    has_trade_marker = any(m in t_no_owner for m in _TRADE_MARKERS) or s in ("tradeall", "rtrade", "monitororder")
    has_explicit_error = any(m in t_no_owner for m in _EXPLICIT_ERROR_MARKERS) or "watchdog" in s
    if has_trade_marker and not has_explicit_error:
        return "TRADES"

    if any(m in t_no_owner for m in _OPS_MARKERS) or "watchdog" in s:
        return "ERROR"

    if (
        "alert" in s or "price" in s or "prag" in t.lower() or "threshold" in t.lower()
        or "listed on" in t.lower() or "coin" in t.lower()
        or any(src in s for src in _PRICE_ALERT_SOURCES)
        or any(m in t for m in ("▲", "▼", "📈", "📉", "RISE", "DROP"))
    ):
        return "PRICE"
    return "TRADES"

def _topic_for_category(title: str, source: str) -> str:
    cat = _category_for_title_and_source(title, source)
    return resolve_topic(cat)

def _resolve_provider_label(bot_name: str = "", source: str = "") -> str:
    bot_str = _clean_str(bot_name)
    src_str = _clean_str(source)
    for cand in (normalize_venue_name(src_str), normalize_venue_name(bot_str)):
        if cand in _PROVIDER_LABELS:
            return _PROVIDER_LABELS[cand]
    s, b = src_str.lower(), bot_str.lower()
    combined = f"{s} {b}"
    if "kraken" in combined or "xstock" in b:
        return "Kraken"
    if "hyperliquid" in combined or "hl" in s or b.startswith("hl"):
        return "Hyperliquid"
    if "t212" in combined or "trading212" in s:
        return "T212"
    if "binance" in combined or "monitororder" in combined or b in ("rtrade", "tradeall", "monitortrades", "order_retry"):
        return "Binance"
    if "-" in bot_str:
        return bot_str.split("-")[0]
    return bot_str or (src_str.capitalize() if src_str else "")

class NotificationServer:
    def __init__(self):
        _ensure_env_loaded()
        self.ntfy_token = self._load_ntfy_token()
        self.rules = self._load_rules()
        self.cooldowns: Dict[str, float] = {}
        self.traceback_buffers: Dict[str, List[str]] = defaultdict(list)
        
        # Persistent Queue File
        self.queue_file = os.environ.get("NOTIFICATION_QUEUE_FILE") or cache_path("notify_queue.json")
        os.makedirs(os.path.dirname(self.queue_file), exist_ok=True)

    def _load_ntfy_token(self) -> str:
        tok = os.environ.get("NTFY_TOKEN", "").strip()
        if not tok:
            _ensure_env_loaded()
            tok = os.environ.get("NTFY_TOKEN", "").strip()
        return tok

    def _load_rules(self) -> List[Dict[str, Any]]:
        rules_path = os.path.join(os.path.dirname(__file__), "rules.json")
        try:
            data = read_json_dict(rules_path)
            for rule in data.get("rules", []):
                rule["regex_obj"] = re.compile(rule["match_regex"])
            return data.get("rules", [])
        except Exception as e:
            logging.error(f"Failed to load rules.json: {e}")
            return []

    def _resolve_topic(self, category: str) -> str:
        return resolve_topic(category)

    def _default_error_topic(self) -> str:
        try:
            return self._resolve_topic("ERROR")
        except Exception:
            return "ntfy-error-1978"
        
    def _enqueue_payload(self, intent_type: str, data: dict):
        try:
            payload = dict(data) if isinstance(data, dict) else {}
            payload["__orchestrator_intent__"] = intent_type
            max_q = _queue_max_items()
            existing = (
                [ln.strip() for ln in read_text_lines(self.queue_file, keepends=True, errors="replace") if ln.strip()]
                if os.path.exists(self.queue_file)
                else []
            )
            if len(existing) >= max_q:
                logging.warning(f"Retry queue full ({len(existing)} >= {max_q}); dropping oldest notification.")
                existing = existing[-(max_q - 1):] if max_q > 1 else []
            existing.append(json.dumps(payload))
            atomic_write_text(self.queue_file, "".join(ln + "\n" for ln in existing))
        except Exception as e:
            logging.error(f"Failed to enqueue notification: {e}")

    @staticmethod
    def _ntfy_gate_accepted() -> bool:
        return getattr(_GATED_DISPATCH_TLS, "last_gate_reason", "") in _GATE_ACCEPTED_REASONS

    def _apply_ntfy_gate(
        self,
        title: str,
        body: str,
        priority: str,
        topic: str,
        alerts: list | None = None,
        *,
        urgent: bool = False,
        skip_email: bool = False,
    ) -> tuple[str, str, str, str] | None:
        _GATED_DISPATCH_TLS.last_gate_reason = ""
        allowed, reason, _, out_payload = pass_common_gate(
            "ntfy", title, body, priority, topic, alerts, urgent=urgent,
        )
        _GATED_DISPATCH_TLS.last_gate_reason = reason
        if allowed:
            return out_payload or (title, body, priority, topic)
        logging.info(f"ntfy delivery gated by common policy ({reason}): {title}")
        if reason in ("buffered_for_bulk", "topic_queue_full_dropped", "queue_full_dropped"):
            flushed = self.flush_pending_bulk()
            if flushed and reason in ("topic_queue_full_dropped", "queue_full_dropped"):
                allowed_retry, reason_retry, _, out_retry = pass_common_gate(
                    "ntfy", title, body, priority, topic, alerts, urgent=urgent,
                )
                _GATED_DISPATCH_TLS.last_gate_reason = reason_retry
                if allowed_retry:
                    return out_retry or (title, body, priority, topic)
                logging.info(f"ntfy delivery gated after due-batch flush ({reason_retry}): {title}")
        if (
            getattr(_GATED_DISPATCH_TLS, "last_gate_reason", reason) not in _GATE_SUPPRESSED_REASONS
            and not skip_email
            and is_email_mirrored(topic)
        ):
            self._send_email(title, body)
        return None

    @staticmethod
    @contextlib.contextmanager
    def _gated_dispatch_scope(*, orig_email: tuple[str, str] | None = None, skip_email: bool | None = None):
        prev_active = getattr(_GATED_DISPATCH_TLS, "active", False)
        prev_orig = getattr(_GATED_DISPATCH_TLS, "orig_email", None)
        prev_skip = getattr(_GATED_DISPATCH_TLS, "skip_email", False)
        _GATED_DISPATCH_TLS.active = True
        if orig_email is not None:
            _GATED_DISPATCH_TLS.orig_email = orig_email
        if skip_email is not None:
            _GATED_DISPATCH_TLS.skip_email = skip_email
        try:
            yield
        finally:
            _GATED_DISPATCH_TLS.active = prev_active
            _GATED_DISPATCH_TLS.orig_email = prev_orig
            _GATED_DISPATCH_TLS.skip_email = prev_skip

    def _gate_and_send_ntfy(
        self,
        title: str,
        body: str,
        priority: str,
        topic: str,
        alerts: list | None = None,
        *,
        urgent: bool = False,
    ) -> bool:
        gated = self._apply_ntfy_gate(title, body, priority, topic, alerts, urgent=urgent)
        if gated is None:
            return self._ntfy_gate_accepted()
        out_title, out_body, out_prio, out_topic = gated
        with self._gated_dispatch_scope(orig_email=(title, body)):
            return self._send_ntfy(out_title, out_body, out_prio, out_topic)

    def flush_pending_bulk(self, *, now: float | None = None, force: bool = False) -> bool:
        """Consumer: flush and synthesize the next due per-topic batch once the central gate opens."""
        flushed = False
        batch = pop_due_bulk_batch("ntfy", now=now, force=force)
        if batch:
            bulk_title, bulk_body, bulk_prio, bulk_topic = batch
            if not bulk_topic:
                bulk_topic = self._default_error_topic()
            with self._gated_dispatch_scope(skip_email=True):
                if self._send_ntfy(bulk_title, bulk_body, bulk_prio, bulk_topic):
                    flushed = True
        email_batch = pop_due_bulk_batch("email", now=now, force=force)
        if email_batch:
            email_subj, email_body, _, _ = email_batch
            if self._send_email(email_subj, email_body, is_retry=False, dedup=False):
                flushed = True
        return flushed

    def _send_ntfy(self, title: str, message: str, priority: str, topic: str, is_retry: bool = False, skip_email: bool = False) -> bool:
        topic_str = _clean_str(topic, strip=True)
        if not topic_str:
            return True
        title = _clean_str(title, "Alert", strip=True)
        message = _clean_str(message)
        raw_title, raw_message = title, message
        priority = _clean_str(priority, "default", strip=True)
        if _external_notifications_disabled():
            logging.debug(f"External notifications disabled: would send to {topic_str}: {title}")
            return True

        if is_fake_alert_text(title, message, topic_str):
            logging.info(f"Skipping test/fake alert in _send_ntfy: {title}")
            return True

        if (
            not is_retry
            and not getattr(_GATED_DISPATCH_TLS, "active", False)
            and not getattr(self, "_in_gated_dispatch", False)
        ):
            gated = self._apply_ntfy_gate(
                title, message, priority, topic_str, urgent=(priority == "urgent"), skip_email=skip_email
            )
            if gated is None:
                return self._ntfy_gate_accepted()
            title, message, priority, topic_str = gated

        # ERROR / DEADMAN topics are mirrored to email (policy: notify_engine.mailer).
        # Sent before the push so an ntfy outage or quota never swallows the email.
        if (
            not is_retry
            and not skip_email
            and not getattr(_GATED_DISPATCH_TLS, "skip_email", False)
            and is_email_mirrored(topic_str)
        ):
            em_title, em_body = getattr(_GATED_DISPATCH_TLS, "orig_email", None) or (raw_title, raw_message)
            self._send_email(em_title, em_body)
        
        url = format_ntfy_url(topic_str)
        if not url:
            return True
        headers = {
            "Title": AlertNotifier.utf8_header(title, "Alert") or "Alert",
            "Priority": sanitize_single_line(priority, "default"),
        }
        if self.ntfy_token:
            headers["Authorization"] = f"Bearer {self.ntfy_token}"
            
        try:
            resp = requests.post(url, data=message.encode("utf-8"), headers=headers, timeout=5)
            if getattr(resp, "status_code", None) == 429:
                if _is_provider_daily_limit(resp):
                    text = getattr(resp, "text", "") or ""
                    logging.warning(f"ntfy daily limit reached: {text}")
                    _mark_provider_daily_limit("ntfy")
                    return False
                retry_after = getattr(resp, "headers", {}).get("Retry-After")
                if retry_after:
                    ra_f = safe_nonneg_float(retry_after, default=None)
                    time.sleep(min(ra_f, 30.0) if ra_f is not None else 1.0)
                    resp = requests.post(url, data=message.encode("utf-8"), headers=headers, timeout=5)
            if hasattr(resp, "raise_for_status"):
                resp.raise_for_status()
            elif getattr(resp, "status_code", 200) >= 400:
                raise requests.HTTPError(f"HTTP {getattr(resp, 'status_code', 0)}")
            logging.info(f"Dispatched ntfy alert to [{topic_str}] (priority={priority}): {title}")
            return True
        except Exception as e:
            if not is_retry:
                logging.error(f"Failed to send ntfy alert (queueing): {e}")
                self._enqueue_payload("ntfy_webhook", {
                    "title": title,
                    "message": message,
                    "priority": priority,
                    "topic": topic_str
                })
            return False
            
    def _send_email(self, subject: str, message: str, is_retry: bool = False, *, dedup: bool = True) -> bool:
        # Retries and synthesized bulk batches bypass dedup: the first attempt already reserved this fingerprint.
        ok = send_email(subject, message, dedup=(dedup and not is_retry))
        if not ok and not is_retry and _email_config() is not None and not last_error_is_auth_failure():
            self._enqueue_payload("email", {"subject": subject, "message": message})
        return ok

    _format_price_alerts_batch_title = staticmethod(AlertNotifier._format_price_alerts_batch_title)
    _format_new_coins_batch_title = staticmethod(AlertNotifier._format_new_coins_batch_title)

    def dispatch_alerts(self, alerts: list, webhook_url: str = None, bot_name: str = "") -> bool:
        if not isinstance(alerts, (list, tuple)) or not alerts:
            return False
        bot_name = _clean_str(bot_name)

        first = alerts[0]
        # Resolve title, body, and source intelligently based on alert payload type
        if isinstance(first, str):
            title = "Alert" if len(alerts) == 1 else f"Alerts ({len(alerts)})"
            body = "\n".join(str(a) for a in alerts)
            source = "price_alert" if "price" in body.lower() else "system"
        else:
            alert_type = first.get("type") if isinstance(first, dict) else None
            has_alert_type = ("alert_type" in first) if isinstance(first, dict) else hasattr(first, "alert_type")
            if alert_type == "price_alert" or has_alert_type:
                if len(alerts) == 1:
                    sym = obj_attr(first, "symbol", "N/A")
                    atype = obj_attr(first, "alert_type", "alert")
                    pchg = safe_float(obj_attr(first, "percent_change", 0.0), 0.0)
                    dir_str = "▲" if atype == "up" else "▼"
                    title = f"{sym} {dir_str} {pchg:+.2f}%"
                else:
                    title = self._format_price_alerts_batch_title(alerts)
                body = AlertNotifier.format_batch_message(alerts)
                source = obj_attr(first, "source") or "price_alert"
            elif alert_type == "new_coin_discovered":
                if len(alerts) == 1:
                    sym = obj_attr(first, "symbol", "N/A")
                    title = f"New Coin: {sym}"
                else:
                    title = self._format_new_coins_batch_title(alerts)
                body = AlertNotifier.format_batch_message(alerts)
                source = obj_attr(first, "source") or "price_alert"
            else:
                base_name = obj_attr(first, "name") or obj_attr(first, "title") or "Alert"
                if len(alerts) == 1:
                    title = base_name
                    body = obj_attr(first, "body") or (
                        "" if alert_type == "bot_event" else (obj_attr(first, "message") or obj_attr(first, "symbol") or "")
                    )
                else:
                    title = f"{base_name} ({len(alerts)})"
                    body = AlertNotifier.format_batch_message(alerts)
                source = obj_attr(first, "source") or ""

        title = _clean_str(title, "Alert")
        body = _clean_str(body)
        source = _clean_str(source)

        topic = _canonical_topic_key(webhook_url) if isinstance(webhook_url, str) and webhook_url.strip() else None
        if not topic:
            topic = _topic_for_category(title, source)

        cat = _category_for_title_and_source(title, source)
        price_topic = self._resolve_topic("PRICE")
        trades_topic = self._resolve_topic("TRADES")

        # Strip any existing [price_watcher] or [macro_analyzer] prefix
        cleaned_title = _STRIP_BOT_PREFIX_RE.sub("", title).strip() or "Alert"

        has_auto_topic_tag = bool(_AUTO_TOPIC_TAG_RE.search(cleaned_title))
        is_macro_bot = bot_name.lower() in ("macro_analyzer", "macro_analyzer.py")
        is_price = (
            topic == price_topic
            or cat == "PRICE"
            or bot_name.lower() in _PRICE_WATCHER_BOTS
            or source.lower() in ("price_alert", *_PRICE_ALERT_SOURCES)
            or any(m in cleaned_title for m in ("▲", "▼"))
        )

        if is_price or is_macro_bot or has_auto_topic_tag:
            full_title = cleaned_title
        elif topic == trades_topic or cat == "TRADES":
            # For trade notifications, preserve trailing [venue:module] / [venue] tag or prepend [Provider]
            provider = _resolve_provider_label(bot_name, source)
            if bot_name and cleaned_title.startswith(f"[{bot_name}]"):
                cleaned_title = cleaned_title[len(f"[{bot_name}]"):].strip()
            if _TRAILING_VENUE_TAG_RE.search(cleaned_title):
                full_title = cleaned_title
            elif provider and not cleaned_title.startswith(f"[{provider}]"):
                full_title = f"[{provider}] {cleaned_title}"
            else:
                full_title = cleaned_title
        else:
            if bot_name and not cleaned_title.startswith(f"[{bot_name}]"):
                full_title = f"[{bot_name}] {cleaned_title}"
            else:
                full_title = cleaned_title

        # Block synthetic/fake test instruments from ever leaking to external topics
        all_syms = " ".join(str(obj_attr(a, "symbol", "")) for a in alerts)
        if is_fake_alert_text(full_title, body, all_syms):
            logging.info(f"Skipping test/fake alert: {full_title}")
            return True

        urgent = _alerts_are_urgent(alerts)
        priority = "urgent" if urgent else "high"
        return self._gate_and_send_ntfy(
            full_title,
            body,
            priority,
            topic,
            alerts=alerts,
            urgent=urgent,
        )

    @staticmethod
    def _collapse_traceback(lines: list[str], *, threshold: int = 12) -> str:
        if len(lines) > threshold:
            return lines[0] + "\n  ...\n" + "\n".join(lines[-10:])
        return "\n".join(lines)

    def process_line(self, line: str, bot_name: str):
        if not isinstance(line, str) or isinstance(line, bool):
            return
        bot_name = _clean_str(bot_name)
        clean_stripped = line.strip()

        # Multi-line Python traceback buffering
        if self.traceback_buffers[bot_name]:
            is_log_prefix = (
                clean_stripped.startswith(("[", "{"))
                or (len(clean_stripped) >= 10 and clean_stripped[:4].isdigit() and clean_stripped[4] in ("-", "/", ":"))
            )
            if is_log_prefix:
                # Flush incomplete traceback buffer before processing new log line
                prev_tb = "\n".join(self.traceback_buffers[bot_name])
                self.traceback_buffers[bot_name] = []
                self.process_line(prev_tb, bot_name)
            else:
                is_indented = line.startswith(" ") or line.startswith("\t")
                is_chain_header = (
                    clean_stripped.startswith("During handling of the above exception")
                    or clean_stripped.startswith("The above exception was the direct cause")
                    or clean_stripped.startswith("Traceback (most recent call last):")
                )
                if clean_stripped == "":
                    return
                elif is_indented or is_chain_header:
                    self.traceback_buffers[bot_name].append(clean_stripped)
                    if len(self.traceback_buffers[bot_name]) >= 40:
                        line = self._collapse_traceback(self.traceback_buffers[bot_name])
                        self.traceback_buffers[bot_name] = []
                    else:
                        return
                else:
                    # Terminal exception line reached!
                    self.traceback_buffers[bot_name].append(clean_stripped)
                    line = self._collapse_traceback(self.traceback_buffers[bot_name])
                    self.traceback_buffers[bot_name] = []
        elif "\n" not in clean_stripped and "Traceback (most recent call last):" in clean_stripped:
            self.traceback_buffers[bot_name] = [clean_stripped]
            return

        # 1. Check for Explicit AlertNotifier Intent (JSON)
        if "__orchestrator_intent__" in line:
            try:
                intent_pos = line.find("__orchestrator_intent__")
                brace_idx = line.rfind("{", 0, intent_pos)
                payload = None
                if brace_idx != -1:
                    try:
                        payload, _ = json.JSONDecoder().raw_decode(line, brace_idx)
                    except json.JSONDecodeError:
                        payload = None
                if payload is None:
                    start_idx = line.find("{")
                    end_idx = line.rfind("}")
                    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                        payload = json.loads(line[start_idx:end_idx + 1])
                    else:
                        payload = json.loads(line)
                if not isinstance(payload, dict):
                    payload = None
                intent = payload.get("__orchestrator_intent__") if payload else None
                if intent == "ntfy_webhook":
                    # Direct retry or pre-formatted [auto_...] topic payload
                    if "title" in payload:
                        t_val = _clean_str(payload.get("title"), "Alert", strip=True)
                        m_val = _clean_str(payload.get("message"))
                        p_val = _clean_str(payload.get("priority"), "default", strip=True)
                        top_val = _clean_str(payload.get("topic"), strip=True) or self._resolve_topic("ERROR")
                        self._gate_and_send_ntfy(
                            t_val,
                            m_val,
                            p_val,
                            top_val,
                            alerts=payload.get("alerts"),
                            urgent=(p_val == "urgent"),
                        )
                        return

                    # Standard alert formatting
                    alerts = payload.get("alerts", [])
                    webhook_url = payload.get("webhook_url")
                    self.dispatch_alerts(alerts, webhook_url=webhook_url, bot_name=bot_name)

                elif intent == "email":
                    # Same schema as the retry queue; rendered by AlertNotifier.send_email_batch.
                    s_val = _clean_str(payload.get("subject"), "Alert", strip=True)
                    em_val = _clean_str(payload.get("message"))
                    if not em_val and isinstance(payload.get("alerts"), list) and payload["alerts"]:
                        em_val = AlertNotifier.format_batch_message(payload["alerts"])
                    self._send_email(s_val, em_val)
                return
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to decode orchestrator intent JSON from {bot_name}: {e} (raw line: {line.strip()})")
                return
            except Exception as e:
                logging.error(f"Error processing orchestrator intent from {bot_name}: {e} (raw line: {line.strip()})", exc_info=True)
                return

        # 2. Check for the [NTFY] prefix shortcut
        if "[NTFY]" in line:
            if is_fake_alert_text(line):
                return
            parts = line.split("[NTFY]", 1)
            if len(parts) > 1:
                msg = parts[1].strip()
                if bot_name.lower() in ("price_watcher", "price_watcher.py"):
                    title = "Price Alert"
                else:
                    title = f"[{bot_name}] Alert" if bot_name else "Alert"
                ntfy_alert = make_bot_event_alert(title, msg, bot_name or "orchestrator")
                self._gate_and_send_ntfy(
                    title,
                    msg,
                    "high",
                    self._resolve_topic("ERROR"),
                    alerts=[ntfy_alert],
                    urgent=(title != "Price Alert"),
                )
                return

        # 3. Check Intelligent Rules (Regex)
        if is_fake_alert_text(line) or any(m in line for m in _INTERNAL_LOG_MARKERS):
            return
        for rule in self.rules:
            if rule["regex_obj"].search(line):
                rule_key = f"{bot_name}_{rule['name']}"
                now = time.time()
                cooldown_sec = safe_nonneg_float(rule.get("cooldown_minutes", 0), 0.0) * 60.0
                if check_cooldown_gate(self.cooldowns, rule_key, now, cooldown_sec, max_entries=200):
                    topic_cat = rule.get("topic_category", "ERROR").upper()
                    topic = self._resolve_topic(topic_cat)
                    if topic_cat == "PRICE" or bot_name.lower() in _PRICE_WATCHER_BOTS:
                        full_title = rule["name"]
                    elif topic_cat == "TRADES":
                        provider = _resolve_provider_label(bot_name)
                        full_title = f"[{provider}] {rule['name']}" if provider else rule["name"]
                    else:
                        full_title = f"[{bot_name}] {rule['name']}" if bot_name else rule["name"]

                    rule_alert = make_bot_event_alert(full_title, line.strip(), bot_name or "orchestrator")
                    self._gate_and_send_ntfy(
                        full_title,
                        line.strip(),
                        rule.get("priority", "default"),
                        topic,
                        alerts=[rule_alert],
                        urgent=(rule.get("priority") == "urgent" or topic_cat == "TRADES"),
                    )
                break

    async def flush_queue_loop(self):
        """Background task to flush synthesized bulk batches and resend delayed notifications."""
        while True:
            await asyncio.sleep(60)
            try:
                await asyncio.to_thread(self.flush_pending_bulk)
            except Exception as e:
                logging.error(f"Error flushing pending bulk notifications: {e}")

            if not os.path.exists(self.queue_file):
                continue
            
            try:
                lines = read_text_lines(self.queue_file, keepends=True, errors="replace")
                if not lines:
                    continue
                
                read_count = len(lines)
                remaining = []
                success_count = 0
                queue_dirty = False
                failed_intents: set[str] = set()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        queue_dirty = True
                        continue
                    
                    try:
                        payload = json.loads(line)
                    except Exception:
                        queue_dirty = True
                        continue
                    if not isinstance(payload, dict):
                        queue_dirty = True
                        continue
                    
                    intent = payload.get("__orchestrator_intent__")
                    if intent not in failed_intents:
                        success = False
                        if intent == "ntfy_webhook":
                            if "title" in payload:
                                title = _prefix_delayed(payload.get("title"))
                                msg = _clean_str(payload.get("message"))
                                prio = _clean_str(payload.get("priority"), "default", strip=True)
                                top = _clean_str(payload.get("topic"), strip=True) or self._default_error_topic()
                                success = await asyncio.to_thread(
                                    self._send_ntfy, title, msg, prio, top, True,
                                )
                            elif "alerts" in payload:
                                success = await asyncio.to_thread(
                                    self.dispatch_alerts,
                                    payload.get("alerts") or [],
                                    payload.get("webhook_url"),
                                    "",
                                )
                            else:
                                success = True
                        elif intent == "email":
                            subj = _prefix_delayed(payload.get("subject"))
                            emsg = _clean_str(payload.get("message"))
                            success = await asyncio.to_thread(
                                self._send_email, subj, emsg, True,
                            )
                        else:
                            success = True  # Unknown intent, drop it

                        if success:
                            success_count += 1
                            queue_dirty = True
                            continue
                        failed_intents.add(intent)
                        queue_dirty = True
                        if intent == "email" and last_error_is_auth_failure():
                            logging.warning(
                                f"Dropping queued email due to permanent SMTP authentication failure: {payload.get('subject', '')!r}"
                            )
                            continue
                        retries = safe_nonneg_int(payload.get("retry_count", 0), default=0) + 1
                        if retries >= _MAX_QUEUE_RETRIES:
                            logging.warning(
                                f"Dropping queued {intent} notification after {retries} failed retries"
                            )
                        else:
                            payload["retry_count"] = retries
                            remaining.append(json.dumps(payload))
                    elif intent == "email" and last_error_is_auth_failure():
                        queue_dirty = True
                        logging.warning(
                            f"Dropping queued email due to permanent SMTP authentication failure: {payload.get('subject', '')!r}"
                        )
                    else:
                        remaining.append(line)
                        
                if queue_dirty or success_count > 0 or len(remaining) != len(lines):
                    # Preserve any notifications appended concurrently while network I/O ran
                    newly_appended = []
                    if os.path.exists(self.queue_file):
                        cur_lines = read_text_lines(self.queue_file, keepends=True, errors="replace")
                        if len(cur_lines) > read_count:
                            newly_appended = [
                                l.strip() for l in cur_lines[read_count:] if l.strip()
                            ]
                    combined = (remaining + newly_appended)[-_queue_max_items():]
                    content = "".join(r + "\n" for r in combined)
                    atomic_write_text(self.queue_file, content)
                    if success_count > 0:
                        logging.info(f"Flushed {success_count} delayed notifications from queue.")

            except Exception as e:
                logging.error(f"Error flushing notification queue: {e}")
